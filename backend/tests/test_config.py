"""Unit tests for EXP-07 configuration and architecture constants."""

from pathlib import Path
from exp07.config import AppConfig, ModelMetadata


def test_model_metadata_constants():
    meta = ModelMetadata()
    assert meta.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert meta.num_hidden_layers == 28
    assert meta.hidden_size == 1536
    assert meta.intermediate_size == 8960
    assert meta.num_attention_heads == 12
    assert meta.num_key_value_heads == 2
    assert meta.vocab_size == 151936
    assert meta.tie_word_embeddings is True

    # Exact parameter math
    assert meta.expected_total_params == 1_543_714_304
    assert meta.expected_transformer_params == 1_310_339_072
    assert meta.expected_layer_params == 46_797_824

    # 28 layers * layer params = total transformer params
    assert meta.num_hidden_layers * meta.expected_layer_params == meta.expected_transformer_params

    # BF16 storage per layer
    assert abs(meta.layer_bf16_mb - 89.26) < 0.01


def test_checkpoint_resolution_missing():
    config = AppConfig(checkpoint_file="/nonexistent/path/model.safetensors")
    assert config.resolve_checkpoint_file() is None
