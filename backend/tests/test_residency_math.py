"""Unit tests for Selective Layer Residency parameter operations."""

import pytest
import torch
from torch import nn

from exp07.inference.layer_residency import (
    count_device_params,
    is_finite_tensor,
    set_nested_meta_parameter,
    set_nested_parameter,
)


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.k_proj = nn.Linear(32, 32, bias=False)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = nn.Linear(32, 64, bias=False)


def test_set_nested_parameter_hierarchical():
    layer = DummyLayer()
    # Initially all on CPU
    assert count_device_params(layer, "cpu") == 32 * 32 * 2 + 32 * 64

    # Replace self_attn.q_proj.weight
    new_weights = torch.ones(32, 32)
    set_nested_parameter(layer, "self_attn.q_proj.weight", new_weights)
    assert torch.equal(layer.self_attn.q_proj.weight, new_weights)


def test_set_nested_meta_parameter():
    layer = DummyLayer()
    # Convert self_attn.q_proj.weight to meta device
    set_nested_meta_parameter(
        layer,
        "self_attn.q_proj.weight",
        torch.Size([32, 32]),
        torch.float32,
    )
    assert layer.self_attn.q_proj.weight.device.type == "meta"
    assert layer.self_attn.k_proj.weight.device.type == "cpu"

    # CPU params should have decreased by 32*32
    assert count_device_params(layer, "meta") == 32 * 32


def test_tensor_finite_check():
    clean = torch.tensor([1.0, 2.0, 3.0])
    assert is_finite_tensor(clean) is True

    has_nan = torch.tensor([1.0, float("nan"), 3.0])
    assert is_finite_tensor(has_nan) is False

    has_inf = torch.tensor([1.0, float("inf"), 3.0])
    assert is_finite_tensor(has_inf) is False
