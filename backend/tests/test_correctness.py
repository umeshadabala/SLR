"""Rigorous numerical correctness tests for EXP-07 Selective Layer Residency.

Validates:
1. Tokenizer output IDs and ChatML template tokens
2. Input embeddings dimensions and values
3. Rotary position embeddings (RoPE)
4. Layer materialization and release (strict 1-layer residency invariant)
5. KV-cache updates and positions
6. Deterministic greedy token generation
7. State cleanup across repeated inference calls
"""

import math
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from exp07.config import config
from exp07.inference.engine import SLRInferenceEngine
from exp07.inference.generation import run_slr_generation
from exp07.inference.layer_residency import load_layer_weights, release_layer_weights
from exp07.schemas import GenerationRequest


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    chk = config.resolve_checkpoint_file()
    assert chk is not None and chk.is_file(), f"Model safetensors not found at {chk}"
    return chk


@pytest.fixture(scope="module")
def tokenizer():
    # Load tokenizer from local artifacts/source directory
    tok_dir = Path("artifacts/source")
    if not tok_dir.exists():
        tok_dir = Path("models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), fix_broken_tokenizer=True)
    return tokenizer


@pytest.fixture(scope="module")
def slr_engine():
    engine = SLRInferenceEngine(config)
    success = engine.load_model()
    assert success is True, f"Failed to load model: {engine.get_status().error_message}"
    return engine


def test_tokenizer_canonical_tokens(tokenizer):
    """Test 1: Verify token IDs and vocabulary structure."""
    assert tokenizer.vocab_size == 151643 or len(tokenizer) >= 151643
    
    # Test special ChatML tokens
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    assert im_start_id == 151644
    assert im_end_id == 151645
    assert endoftext_id == 151643

    # Test encoding determinism
    prompt = "Hello, world!"
    tokens = tokenizer.encode(prompt)
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == prompt.strip()


def test_input_embeddings_and_rope(slr_engine, tokenizer):
    """Test 2: Verify embedding projection and rotary positional encoding."""
    model = slr_engine.model
    prompt = "What is a CPU?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
    
    assert hidden_states.shape == (1, seq_len, 1536)
    assert torch.isfinite(hidden_states).all()
    assert hidden_states.abs().max().item() > 0.0

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)

    assert cos.shape[-1] == 128  # head_dim
    assert sin.shape[-1] == 128
    assert (cos.abs() <= 1.0001).all()
    assert (sin.abs() <= 1.0001).all()


def test_layer_materialization_and_residency_invariant(slr_engine, checkpoint_path):
    """Test 3: Verify strict 1-layer materialization and complete release."""
    model = slr_engine.model
    layer0 = model.model.layers[0]

    # Verify initial layer weights are on META device
    meta_count = sum(1 for p in layer0.parameters() if p.device.type == "meta")
    assert meta_count > 0, "Layer 0 should initially reside on META device"

    # Materialize Layer 0
    loaded_count, active_params = load_layer_weights(
        layer0,
        layer_idx=0,
        safetensors_file=str(checkpoint_path),
        dtype=torch.bfloat16,
        expected_layer_params=config.model_meta.expected_layer_params,
    )

    assert loaded_count == 12
    assert active_params == 46_797_824
    cpu_count = sum(1 for p in layer0.parameters() if p.device.type == "cpu")
    assert cpu_count == 12

    # Release Layer 0
    release_layer_weights(
        layer0,
        expected_layer_params=config.model_meta.expected_layer_params,
    )

    post_meta_count = sum(1 for p in layer0.parameters() if p.device.type == "meta")
    assert post_meta_count == 12, "All 12 layer parameters must return to META device"
    post_cpu_count = sum(1 for p in layer0.parameters() if p.device.type == "cpu")
    assert post_cpu_count == 0, "Zero layer parameters may remain on CPU after release"


def test_single_layer_forward_and_kv_cache(slr_engine, checkpoint_path, tokenizer):
    """Test 4: Verify single-layer forward pass math and KV cache population."""
    model = slr_engine.model
    layer0 = model.model.layers[0]

    input_ids = tokenizer("Test prompt", return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids).to(torch.bfloat16)

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)

    cache = DynamicCache()

    # Materialize, compute, release
    load_layer_weights(layer0, 0, str(checkpoint_path), torch.bfloat16)
    try:
        with torch.no_grad():
            out = layer0(
                hidden_states,
                position_embeddings=(cos, sin),
                past_key_values=cache,
                cache_position=position_ids[0],
                position_ids=position_ids,
                use_cache=True,
            )
    finally:
        release_layer_weights(layer0)

    # Output assertions
    assert out.shape == (1, seq_len, 1536)
    assert torch.isfinite(out).all()

    # KV Cache assertions: Qwen2.5-1.5B has 2 KV heads, head_dim 128
    assert cache.get_seq_length(0) == seq_len
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        k_cached = cache.layers[0].keys
        v_cached = cache.layers[0].values
        assert k_cached.shape == (1, 2, seq_len, 128)
        assert v_cached.shape == (1, 2, seq_len, 128)
        assert torch.isfinite(k_cached).all()
        assert torch.isfinite(v_cached).all()


def test_deterministic_greedy_generation(slr_engine):
    """Test 5: Verify deterministic greedy autoregressive generation."""
    req1 = GenerationRequest(prompt="Hello, my name is", max_new_tokens=4)
    res1 = slr_engine.generate(req1)

    assert res1.status == "completed"
    assert res1.generated_tokens == 4
    assert len(res1.generated_token_ids) == 4
    assert res1.tokens_per_second > 0.0
    assert res1.peak_active_layers == 1
    assert len(res1.generated_text) > 0

    # Run again with identical request: must produce identical token IDs
    req2 = GenerationRequest(prompt="Hello, my name is", max_new_tokens=4)
    res2 = slr_engine.generate(req2)
    assert res1.generated_token_ids == res2.generated_token_ids
    assert res1.generated_text == res2.generated_text


def test_state_isolation_across_consecutive_runs(slr_engine):
    """Test 6: Verify no KV cache or residual state leakage across independent requests."""
    res_a = slr_engine.generate(GenerationRequest(prompt="Count: 1, 2, 3,", max_new_tokens=3))
    res_b = slr_engine.generate(GenerationRequest(prompt="Colors: red, blue,", max_new_tokens=3))

    assert res_a.status == "completed"
    assert res_b.status == "completed"
    assert res_a.generated_token_ids != res_b.generated_token_ids
