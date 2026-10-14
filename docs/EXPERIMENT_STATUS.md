# SLR Research Status & Experiment Records

## Target Model Specification
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Revision**: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Parameters**: 1,543,714,304 (1.54B)
- **Transformer Layers**: 28 layers
- **Parameters per Layer**: 46,797,824 (46.8M)
- **BF16 Size per Layer**: 89.26 MB

## Summary of Completed Historical Experiments

| Experiment | Title | Key Findings | Artifacts |
|---|---|---|---|
| **EXP-01** | Model Inventory & Baseline | Safetensors tensor inventory validated; 1.54B parameter count confirmed. | `results/historical/EXP-01` |
| **EXP-02** | Single Layer Residency Invariant | Verified loading 1 layer onto CPU and resetting to META. | `results/historical/EXP-02A`, `EXP-02B` |
| **EXP-03** | 28-Layer Sequential Streaming | Full sequential execution across 28 layers with tied output projection. | `results/historical/EXP-03` |
| **EXP-04** | Persistent SafeOpen & Prefetch | Persistent handles reduced load overhead by ~18%. Double-buffer prefetch tested. | `results/historical/EXP-04` |
| **EXP-05** | Quantization Tradeoffs | Per-tensor INT8 rejected due to severe drift. Mixed MLP-INT8 achieved ~45MB layer storage with controlled drift. | `results/historical/EXP-05` |
| **EXP-05.5** | Multi-token KV-Cache | Preliminary multi-token KV-cache generation verified. | `results/historical/EXP-05.5` |
| **EXP-06.5** | Autoregressive Decode Equivalence | Corrected hierarchical parameter resolution (`get_submodule`); 30-prompt stress test passed with DynamicCache. | `experiment.ipynb` (Cells 128–132) |
| **EXP-07 (Workstation)** | Flutter + Python Rebuild | Complete architecture rebuild with Flutter UI and decoupled Python service on Apple Silicon Mac. | Current Repository |
| **EXP-07 (Android)** | On-Device SLR & Memory Safety Fuse | Native Android Safetensors streaming engine with independent `MemorySafetyController`, `/proc/self/status` RSS telemetry, and 18-test safe benchmark suite. | `../apks/exp07-slr-debug.apk` |

## Current EXP-07 Status (Android Milestone)
- **Native Android Engine**: Operational in `app/android/app/src/main/kotlin/com/example/exp07_slr/`.
  - `DeviceDiagnostics.kt`: Reads `/proc/self/status` VmRSS/VmHWM and `ActivityManager.MemoryInfo`.
  - `MemorySafetyController.kt`: Independent 250ms polling thread enforcing configurable 75% RAM limit, atomic cancellation flags, controlled cleanup sequencing, and crash-resilient journal logging.
  - `SafetensorsParser.kt`: Zero-copy random-access tensor parser with O(1) BF16-to-FP32 casting.
  - `AndroidSlrEngine.kt`: Real layer-by-layer forward execution (RoPE, RMSNorm, Grouped-Query Attention with KV Cache, SwiGLU MLP).
  - `AndroidBenchmarkSuite.kt`: 18 automated tests with truthful status tracking.
- **Compiled Validation APK**:
  - `app/build/app/outputs/flutter-apk/app-debug.apk` (144 MB)
  - Mirrored to `../apks/exp07-slr-debug.apk`
- **Flutter UI**: Live Memory Safety Panel with active residency counters, RSS meters, warning banners, single layer verification, and one-tap safe benchmark runner.
