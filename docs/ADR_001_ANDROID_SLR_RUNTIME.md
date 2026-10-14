# ADR 001: Android On-Device Selective Layer Residency (SLR) Runtime Architecture

**Status**: Accepted  
**Date**: October 2026  
**Context**: EXP-07 Research Milestone — Selective Layer Residency on Physical Android  
**Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` (Revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)

---

## 1. Context and Problem Statement

The goal of EXP-07 is to execute genuine inference with the 1.54B-parameter Qwen2.5-1.5B model on a physical Android mobile device under strict memory safety constraints, proving the Selective Layer Residency (SLR) invariant:
> **Strict Invariant**: At most one transformer decoder layer (out of 28) is active as a materialized compute layer in memory at any point during prefill or autoregressive decoding.

On commodity mobile devices (typically 4GB–8GB RAM, with 2GB–4GB reserved for system, baseband, GPU, and display services), loading a full 1.5B parameter model in BF16 (~3.08 GB) leads to immediate Low Memory Killer (LMK) eviction. Furthermore, the inference runtime must guarantee an independent safety fuse, track genuine operating system process RSS (`VmRSS`), and maintain exact numerical accuracy.

---

## 2. Evaluated Options and Rejection Rationale

### Option A: Embedded Python with PyTorch Wheels (Kivy / Buildozer / Chaquopy)
- **Investigation**: Prior experiments (EXP-06 Kivy attempt) bundled Python 3.11 with ARM64 PyTorch wheels into `selectivellm-0.1.0-arm64-v8a-debug.apk`.
- **Findings**:
  1. PyTorch C++ runtime and Python interpreter baseline consumption exceeded 1.2 GB RSS before loading any model weights.
  2. Severe dynamic linker (`dlopen`) issues occurred across diverse Android libc/NDK versions.
  3. Startup latency exceeded 12 seconds, triggering Application Not Responding (ANR) dialogs.
- **Decision**: **REJECTED**. Unviable for stable mobile execution.

### Option B: Monolithic ExecuTorch Standard LLM Runner
- **Investigation**: PyTorch's ExecuTorch edge runtime provides an LLM runner for mobile devices.
- **Findings**:
  1. The standard ExecuTorch LLM runner assumes a single monolithic `.pte` model graph loaded or mapped as a single entity.
  2. It does not provide fine-grained, layer-by-layer dynamic lifecycle hooks (load layer $\rightarrow$ compute $\rightarrow$ unmap/free $\rightarrow$ verify 0 parameters $\rightarrow$ load next layer).
  3. Memory planning in ExecuTorch pre-allocates tensor arenas for all operations concurrently unless an entirely custom multi-model graph partitioner is built.
- **Decision**: **REJECTED as a monolithic runner**.

### Option C: Native Android Selective Streamer with Memory Safety Fuse (Selected)
- **Architecture**:
  1. **Direct Safetensors Streaming Engine**: Safetensors has an 8-byte header length, a JSON metadata dictionary with byte offsets, and sequential binary tensor buffers. Using Android `FileChannel` / random-access reads, the runtime accesses layer $i$'s 12 tensors (46.8M parameters = 89.26 MB) on demand from app storage, executes forward computation, and immediately releases the memory buffer.
  2. **Dedicated MemorySafetyController**: Runs on an independent background thread (sampling at 250ms), querying `/proc/self/status` (`VmRSS`, `VmHWM`) and `ActivityManager.MemoryInfo`. It enforces an atomic cancellation flag that prevents subsequent layer scheduling when memory crosses configurable warning (e.g. 70%) or hard abort thresholds (default 75% of physical RAM).
  3. **Dual Execution Environment**:
     - On Android: Flutter connects via Android Platform Channels (`MethodChannel` / `EventChannel`) to the native engine and safety controller.
     - On macOS/Workstation: Flutter connects via HTTP to the Python reference backend (`backend/src/exp07`), preserving 100% research continuity.
- **Decision**: **ACCEPTED**.

---

## 3. Detailed Component Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Flutter UI Presentation Layer                   │
│   Dashboard  │  Inference  │  Memory Safety Panel  │  Safe Benchmark   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Platform Channels / HTTP
┌───────────────────────────────────▼────────────────────────────────────┐
│                  Platform-Agnostic SLR Repository                      │
│        (Auto-detects Android Native vs. Workstation Python REST)       │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │ Android Platform                │ Workstation
┌──────────────────▼──────────────────┐   ┌──────────▼───────────────────┐
│     Android Native SLR Engine       │   │   Python FastAPI Reference   │
│  - Safetensors Direct Channel       │   │  - PyTorch Meta Device       │
│  - 28-Layer Sequential Streamer     │   │  - DynamicCache Autoreg      │
│  - Tied Embeddings & RMSNorm        │   │  - 127.0.0.1:8000            │
└──────────────────┬──────────────────┘   └──────────────────────────────┘
                   │
┌──────────────────▼─────────────────────────────────────────────────────┐
│           Independent MemorySafetyController & Diagnostics             │
│  - Samples /proc/self/status VmRSS every 250ms                         │
│  - Monitors ActivityManager.MemoryInfo system pressure                 │
│  - FSM: NORMAL -> WARNING -> ABORT_REQUESTED -> CLEANUP -> ABORTED    │
│  - Flushes crash-resilient diagnostic journal to disk before exit      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Consequences and Invariants

1. **Strict 1-Layer Invariant**: At any time, active materialized transformer parameters never exceed 46,797,824 (89.26 MB BF16).
2. **Honest Metric Reporting**: Process RSS from `/proc/self/status` is reported separately from theoretical parameter memory.
3. **Graceful Degradation & Survival**: If memory pressure breaches thresholds, the fuse stops token generation, cleans up the KV-cache, persists the benchmark report to disk, and requires explicit user acknowledgment before another run can start.
