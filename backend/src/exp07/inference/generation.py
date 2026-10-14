"""Autoregressive text generation using Selective Layer Residency and DynamicCache.

Implements the exact execution pattern from experiment.ipynb Cells 129 and 131:
- Sequential 28-layer execution with strictly 1 layer resident on CPU
- Tied input/output embeddings (@ model.model.embed_tokens.weight.transpose(0, 1))
- DynamicCache sequence propagation
- Real layer load/exec/release latency accounting
- Job cancellation support
- ChatML template formatting, EOS early-stopping, and streaming token callback
"""

from __future__ import annotations

import time
from typing import Callable, Generator, List, Optional
import torch
from transformers.cache_utils import DynamicCache

from exp07.config import ModelMetadata
from exp07.inference.layer_residency import (
    get_process_rss_gb,
    is_finite_tensor,
    load_layer_weights,
    release_layer_weights,
)
from exp07.schemas import GenerationJobResponse, LayerTimingRecord


def _sample_next_token(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
) -> int:
    """Greedy or nucleus sampling with repetition penalty."""
    logits = logits.clone().squeeze().float()

    if repetition_penalty > 1.0 and len(generated_ids) > 0:
        for tid in set(generated_ids):
            if logits[tid] < 0:
                logits[tid] = logits[tid] * repetition_penalty
            else:
                logits[tid] = logits[tid] / repetition_penalty

    if not do_sample or temperature <= 0.01:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / max(temperature, 1e-4)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def run_slr_generation(
    model: torch.nn.Module,
    tokenizer,
    safetensors_file: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
    stop_sequences: Optional[List[str]] = None,
    job_id: str = "job_default",
    cancel_check: Optional[Callable[[], bool]] = None,
    telemetry_callback: Optional[Callable[[int, int, float, float, float], None]] = None,
    token_callback: Optional[Callable[[int, str, float], None]] = None,
    metadata: ModelMetadata = ModelMetadata(),
    dtype: torch.dtype = torch.bfloat16,
) -> GenerationJobResponse:
    """Execute real SLR autoregressive text generation with strict 1-layer residency."""
    start_total_time = time.perf_counter()

    # 1. Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_tokens = int(input_ids.shape[1])
    prefix_len = prompt_tokens

    # 2. Fresh DynamicCache per generation
    cache = DynamicCache()

    layer_timings: List[LayerTimingRecord] = []
    generated_token_ids: List[int] = []

    peak_active_params = 0
    peak_active_layers = 0
    peak_rss = get_process_rss_gb()
    stop_reason = "token_limit"

    eos_ids = {151645, 151643}  # Qwen2.5 <|im_end|> and <|endoftext|>
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    # -------------------------------------------------------------------------
    # A. Prefix forward (Prefill)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids).to(dtype)

    position_ids = torch.arange(
        prefix_len,
        dtype=torch.long,
    ).unsqueeze(0)

    with torch.no_grad():
        cos, sin = model.model.rotary_emb(
            hidden_states,
            position_ids,
        )

    for layer_idx in range(metadata.num_hidden_layers):
        if cancel_check and cancel_check():
            del cache
            return GenerationJobResponse(
                job_id=job_id,
                status="cancelled",
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                generated_text="",
                generated_token_ids=[],
                generated_tokens=0,
                stop_reason="cancelled",
                error="Cancelled during prefix pass",
            )

        # Load layer
        t_load = time.perf_counter()
        loaded_count, active_params = load_layer_weights(
            model.model.layers[layer_idx],
            layer_idx,
            safetensors_file,
            dtype=dtype,
            expected_layer_params=metadata.expected_layer_params,
        )
        load_sec = time.perf_counter() - t_load

        peak_active_params = max(peak_active_params, active_params)
        peak_active_layers = max(peak_active_layers, 1)
        current_rss = get_process_rss_gb()
        peak_rss = max(peak_rss, current_rss)

        # Execute layer
        t_exec = time.perf_counter()
        hidden_states = model.model.layers[layer_idx](
            hidden_states,
            position_embeddings=(cos, sin),
            past_key_values=cache,
            cache_position=position_ids[0],
            position_ids=position_ids,
            use_cache=True,
        )
        exec_sec = time.perf_counter() - t_exec

        # Release layer back to META
        t_rel = time.perf_counter()
        release_layer_weights(
            model.model.layers[layer_idx],
            expected_layer_params=metadata.expected_layer_params,
        )
        rel_sec = time.perf_counter() - t_rel

        if telemetry_callback:
            telemetry_callback(layer_idx, active_params, load_sec, exec_sec, rel_sec)

        layer_timings.append(
            LayerTimingRecord(
                layer_idx=layer_idx,
                load_sec=load_sec,
                exec_sec=exec_sec,
                release_sec=rel_sec,
                active_params=active_params,
                active_layers=1,
                rss_gb=current_rss,
                kv_cache_len=prefix_len,
            )
        )

    # Prefix logits via final norm + tied embeddings
    with torch.no_grad():
        hidden_states = model.model.norm(hidden_states)
        prefix_logits = (
            hidden_states[:, -1, :]
            @ model.model.embed_tokens.weight.transpose(0, 1)
        )

    next_token = _sample_next_token(
        prefix_logits,
        generated_token_ids,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )
    generated_token_ids.append(next_token)
    ttft_sec = time.perf_counter() - start_total_time

    token_str = tokenizer.decode([next_token], skip_special_tokens=False)
    if token_callback:
        token_callback(next_token, token_str, ttft_sec)

    # Check immediate EOS
    if next_token in eos_ids:
        stop_reason = "eos"
    else:
        # ---------------------------------------------------------------------
        # B. Autoregressive decode loop
        # ---------------------------------------------------------------------
        for gen_idx in range(max_new_tokens - 1):
            if cancel_check and cancel_check():
                stop_reason = "cancelled"
                break

            current_pos = prefix_len + gen_idx
            next_ids = torch.tensor([[next_token]], dtype=torch.long)

            with torch.no_grad():
                hidden = model.model.embed_tokens(next_ids).to(dtype)

            decode_pos_ids = torch.tensor([[current_pos]], dtype=torch.long)

            with torch.no_grad():
                decode_cos, decode_sin = model.model.rotary_emb(
                    hidden,
                    decode_pos_ids,
                )

            for layer_idx in range(metadata.num_hidden_layers):
                if cancel_check and cancel_check():
                    stop_reason = "cancelled"
                    break

                t0 = time.perf_counter()
                _, active_params = load_layer_weights(
                    model.model.layers[layer_idx],
                    layer_idx,
                    safetensors_file,
                    dtype=dtype,
                    expected_layer_params=metadata.expected_layer_params,
                )
                load_sec = time.perf_counter() - t0

                t0 = time.perf_counter()
                hidden = model.model.layers[layer_idx](
                    hidden,
                    position_embeddings=(decode_cos, decode_sin),
                    past_key_values=cache,
                    cache_position=torch.tensor([current_pos], dtype=torch.long),
                    position_ids=decode_pos_ids,
                    use_cache=True,
                )
                exec_sec = time.perf_counter() - t0

                t0 = time.perf_counter()
                release_layer_weights(
                    model.model.layers[layer_idx],
                    expected_layer_params=metadata.expected_layer_params,
                )
                rel_sec = time.perf_counter() - t0

                if telemetry_callback:
                    telemetry_callback(layer_idx, active_params, load_sec, exec_sec, rel_sec)

            if cancel_check and cancel_check():
                stop_reason = "cancelled"
                break

            with torch.no_grad():
                normed = model.model.norm(hidden)
                logits = (
                    normed[:, -1, :]
                    @ model.model.embed_tokens.weight.transpose(0, 1)
                )

            next_token = _sample_next_token(
                logits,
                generated_token_ids,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )

            # Check EOS
            if next_token in eos_ids:
                stop_reason = "eos"
                break

            generated_token_ids.append(next_token)
            tok_time = time.perf_counter() - start_total_time
            token_str = tokenizer.decode([next_token], skip_special_tokens=False)

            if token_callback:
                token_callback(next_token, token_str, tok_time)

            # Check stop sequences
            if stop_sequences:
                accumulated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
                if any(seq in accumulated_text for seq in stop_sequences):
                    stop_reason = "stop_sequence"
                    break

    total_generation_sec = time.perf_counter() - start_total_time
    decode_sec = max(total_generation_sec - ttft_sec, 0.0001)
    decode_tokens = max(len(generated_token_ids) - 1, 0)

    decode_throughput = (decode_tokens / decode_sec) if decode_tokens > 0 else None
    inter_token_latency = (decode_sec / decode_tokens) if decode_tokens > 0 else None

    tokens_per_second = (
        len(generated_token_ids) / total_generation_sec
        if total_generation_sec > 0 and len(generated_token_ids) > 0
        else None
    )

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    # Clean up DynamicCache to return 100% of model to META
    del cache

    return GenerationJobResponse(
        job_id=job_id,
        status="completed" if stop_reason != "cancelled" else "cancelled",
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        generated_text=generated_text,
        generated_token_ids=generated_token_ids,
        generated_tokens=len(generated_token_ids),
        stop_reason=stop_reason,
        time_to_first_token_sec=ttft_sec,
        total_generation_sec=total_generation_sec,
        tokens_per_second=tokens_per_second,
        decode_throughput=decode_throughput,
        inter_token_latency_sec=inter_token_latency,
        peak_active_params=peak_active_params,
        peak_active_layers=peak_active_layers,
        peak_process_rss_gb=peak_rss,
        engine_type="SelectiveLayerResidencyEngine",
        is_simulated=False,
        layer_timings=layer_timings,
    )
