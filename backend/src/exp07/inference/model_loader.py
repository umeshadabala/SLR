"""Model loading and initialization on META device for Selective Layer Residency."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2ForCausalLM,
)

from exp07.config import ModelMetadata
from exp07.inference.layer_residency import set_nested_parameter


def build_meta_model(
    config_dir_or_file: str,
    model_metadata: ModelMetadata = ModelMetadata(),
) -> Tuple[Qwen2ForCausalLM, AutoConfig]:
    """Instantiate Qwen2ForCausalLM entirely on the META device."""
    config = AutoConfig.from_pretrained(
        config_dir_or_file,
        local_files_only=True,
    )

    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    meta_params = sum(
        p.numel() for p in model.parameters() if p.device.type == "meta"
    )

    assert total_params == model_metadata.expected_total_params, (
        f"Expected {model_metadata.expected_total_params} params, got {total_params}"
    )
    assert meta_params == total_params, (
        f"Expected all {total_params} on meta device, got {meta_params}"
    )

    return model, config


def materialize_non_layer_weights(
    model: Qwen2ForCausalLM,
    config: AutoConfig,
    safetensors_file: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Load embeddings, final norm, and rotary embedding buffers onto CPU."""
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        # 1. Embeddings
        embed = f.get_tensor("model.embed_tokens.weight").to(dtype)
        set_nested_parameter(model.model.embed_tokens, "weight", embed)

        # 2. Final RMSNorm
        norm = f.get_tensor("model.norm.weight").to(dtype)
        set_nested_parameter(model.model.norm, "weight", norm)

    # 3. Rotary Embedding inv_freq reconstruction (matching experiment.ipynb Cell 129)
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = float(
        rope_parameters.get(
            "rope_theta",
            getattr(config, "rope_theta", 1_000_000.0),
        )
    )
    head_dim = config.hidden_size // config.num_attention_heads
    rotary_dim = head_dim

    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32)
            / rotary_dim
        )
    )

    model.model.rotary_emb.inv_freq = inv_freq
    if hasattr(model.model.rotary_emb, "original_inv_freq"):
        model.model.rotary_emb.original_inv_freq = inv_freq.clone()


def load_tokenizer_if_available(source_dir: str) -> Optional[AutoTokenizer]:
    """Load local fast tokenizer if available."""
    try:
        return AutoTokenizer.from_pretrained(
            source_dir,
            local_files_only=True,
            use_fast=True,
        )
    except Exception:
        return None
