"""Core Selective Layer Residency (SLR) algorithms from experiment.ipynb.

Directly adapted from validated notebook Cells 129, 130, and 131:
- Resolves hierarchical submodule parameters: parent = layer.get_submodule(...)
- Materializes 1 layer at a time from Safetensors to CPU
- Enforces 0 CPU parameters on release, restoring Parameter(device='meta')
- Maintains strictly 1 active layer invariant
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import psutil
import torch
from torch import nn
from safetensors import safe_open


def get_process_rss_gb() -> float:
    """Current process Resident Set Size in Gigabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def get_process_rss_mb() -> float:
    """Current process Resident Set Size in Megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def is_finite_tensor(x: torch.Tensor) -> bool:
    """Check whether all elements in a tensor are finite."""
    return bool(torch.isfinite(x.float()).all().item())


def count_device_params(module: nn.Module, device_type: str) -> int:
    """Count total elements of parameters residing on the specified device."""
    return sum(
        p.numel()
        for p in module.parameters()
        if p.device.type == device_type
    )


def set_nested_parameter(
    module: nn.Module,
    parameter_name: str,
    tensor: torch.Tensor,
) -> None:
    """Replace a nested Parameter with an active CPU tensor.

    Example:
        parameter_name = 'self_attn.q_proj.weight'
    Resolves:
        parent = module.get_submodule('self_attn.q_proj')
        parent.weight = nn.Parameter(tensor)
    """
    parts = parameter_name.split(".")
    if len(parts) == 1:
        parent = module
        leaf_name = parts[0]
    else:
        parent_path = ".".join(parts[:-1])
        leaf_name = parts[-1]
        parent = module.get_submodule(parent_path)

    old = getattr(parent, leaf_name)
    if not isinstance(old, nn.Parameter):
        raise TypeError(
            f"{parameter_name}: expected nn.Parameter, got {type(old)}"
        )

    new_param = nn.Parameter(
        tensor,
        requires_grad=False,
    )
    setattr(parent, leaf_name, new_param)


def set_nested_meta_parameter(
    module: nn.Module,
    parameter_name: str,
    shape: torch.Size,
    dtype: torch.dtype,
) -> None:
    """Replace a parameter with a META tensor while preserving its shape and structure."""
    parts = parameter_name.split(".")
    if len(parts) == 1:
        parent = module
        leaf_name = parts[0]
    else:
        parent_path = ".".join(parts[:-1])
        leaf_name = parts[-1]
        parent = module.get_submodule(parent_path)

    old = getattr(parent, leaf_name)
    meta_tensor = torch.empty(
        shape,
        dtype=dtype,
        device="meta",
    )

    new_param = nn.Parameter(
        meta_tensor,
        requires_grad=False,
    )
    setattr(parent, leaf_name, new_param)


def load_layer_weights(
    layer: nn.Module,
    layer_idx: int,
    safetensors_file: str,
    dtype: torch.dtype = torch.bfloat16,
    expected_layer_params: int = 46_797_824,
) -> Tuple[int, int]:
    """Load all 12 parameters of layer_idx from safetensors into CPU memory."""
    checkpoint_prefix = f"model.layers.{layer_idx}."
    loaded_count = 0
    loaded_params = 0

    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for parameter_name, _ in layer.named_parameters():
            full_key = checkpoint_prefix + parameter_name
            if full_key not in keys:
                raise KeyError(f"Missing checkpoint tensor: {full_key}")

            tensor = f.get_tensor(full_key)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)

            set_nested_parameter(layer, parameter_name, tensor)
            loaded_count += 1
            loaded_params += tensor.numel()

    cpu_params = sum(
        p.numel() for p in layer.parameters() if p.device.type == "cpu"
    )
    meta_params = sum(
        p.numel() for p in layer.parameters() if p.device.type == "meta"
    )

    if expected_layer_params > 0:
        assert loaded_count == 12, f"Expected 12 tensors, loaded {loaded_count}"
        assert loaded_params == expected_layer_params, (
            f"Expected {expected_layer_params} params, loaded {loaded_params}"
        )
        assert cpu_params == expected_layer_params
        assert meta_params == 0

    return loaded_count, loaded_params


def release_layer_weights(
    layer: nn.Module,
    expected_layer_params: int = 46_797_824,
) -> None:
    """Return all parameters of a layer back to META tensors."""
    parameter_snapshot = [
        (name, param.shape, param.dtype, param.requires_grad)
        for name, param in layer.named_parameters()
    ]

    for name, shape, dtype, _ in parameter_snapshot:
        set_nested_meta_parameter(layer, name, shape, dtype)

    cpu_params = sum(
        p.numel() for p in layer.parameters() if p.device.type == "cpu"
    )
    meta_params = sum(
        p.numel() for p in layer.parameters() if p.device.type == "meta"
    )

    if expected_layer_params > 0:
        assert cpu_params == 0, f"Expected 0 CPU params on release, found {cpu_params}"
        assert meta_params == expected_layer_params, (
            f"Expected {expected_layer_params} META params, found {meta_params}"
        )
