# EXP-07 System Architecture

## Overview
**EXP-07** implements a modular, research-oriented architecture for demonstrating and benchmarking **Selective Layer Residency (SLR)** using `Qwen/Qwen2.5-1.5B-Instruct` (source revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`).

```
┌────────────────────────────────────────────────────────┐
│               Flutter / Dart Frontend                 │
│  - Dashboard  - Inference  - Residency Telemetry       │
│  - Benchmarks - Experiments - Settings                 │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP JSON / SSE (127.0.0.1:8000)
┌───────────────────────────▼────────────────────────────┐
│              FastAPI Python Backend (Darwin)           │
│  - Model Lifecycle   - DynamicCache Execution          │
│  - Telemetry Stream  - Benchmark & Results DB          │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│             Selective Layer Residency Engine           │
│  - Model initialized on PyTorch META device            │
│  - Persistent Safetensors Handle                       │
│  - 1 Active Layer on CPU (46.8M BF16 params = 89.26MB) │
│  - Release back to META device after forward step      │
│  - Tied Embeddings & Final RMSNorm Projection          │
└────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

1. **Decoupled Frontend and Inference Service**:
   - The Flutter UI communicates with the engine exclusively via typed HTTP contracts.
   - The UI never fakes token generation, latency, or device parameters.
   - If the backend is offline or the model is not loaded, explicit truthful states (`offline`, `not_loaded`, `checkpoint_missing`) are displayed.

2. **Strict Invariant: Single Layer Materialization**:
   - All 28 transformer layers of Qwen2.5-1.5B are initialized as `torch.device("meta")` parameters.
   - Only non-layer parameters (token embeddings, final RMSNorm, and rotary embedding `inv_freq`) remain permanently resident in memory.
   - During prefix evaluation and autoregressive decoding, layer weights are loaded hierarchically from Safetensors (`set_nested_parameter`), executed with `DynamicCache`, and immediately released back to `meta` tensors (`set_nested_meta_parameter`).
   - A layer is never reported as released until its PyTorch parameter objects are confirmed to be on the `meta` device with 0 CPU resident parameters.

3. **Autoregressive State Management**:
   - A fresh `DynamicCache` instance is created per generation request.
   - After generation concludes or is cancelled, `DynamicCache` is cleared, returning the model to 100% META transformer layers.
