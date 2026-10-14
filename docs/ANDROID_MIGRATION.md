# Android Migration Plan for Selective Layer Residency

## Objective
Migrate the validated Mac SLR inference engine to run **on-device on Android** with genuine layer-by-layer residency, DynamicCache propagation, and measured hardware constraints.

## Background: Previous Kivy Attempt
In prior experiments, an attempt was made to package a Python runtime into an Android APK using Kivy and Buildozer (`selectivellm-0.1.0-arm64-v8a-debug.apk` in `../apks`). While the application ran on the development workstation, the APK crashed on target Android devices due to native shared library loader failures, PyTorch wheel ABI incompatibilities, and excessive memory pressure during startup.

## Migration Principles
1. **Frontend Isolation**:
   - The Flutter UI is decoupled from the inference backend.
   - On Android, Flutter will serve as the presentation and orchestration layer, communicating with an on-device native service.
2. **Native C++ / NDK Engine or Embedded Executor**:
   - Rather than embedding a heavy full Python/PyTorch runtime on Android, investigate:
     - **Option A (Recommended)**: A lightweight native C++ / ggml / ONNX Runtime execution controller that reads Safetensors via `mmap` or standard I/O and evaluates one layer at a time.
     - **Option B**: Embedded Python via Chaquopy with stripped down PyTorch Mobile C++ libraries.
3. **Step-by-Step Feasibility Gates**:
   - [ ] **Gate 1**: Verify Android SDK and NDK toolchains on dev machine (SDK 36.0.0 detected).
   - [ ] **Gate 2**: Validate native filesystem read performance for a 2.8 GB file on internal Android storage (UFS 2.2 / 3.1).
   - [ ] **Gate 3**: Load exactly one Qwen transformer layer (46.8M parameters = ~89.26 MB) into native memory.
   - [ ] **Gate 4**: Execute a single layer matrix multiplication and verify numerical equivalence against Mac reference output.
   - [ ] **Gate 5**: Release layer and verify Android process RSS drops or stabilizes.
   - [ ] **Gate 6**: Loop across all 28 layers with KV-cache updates.
   - [ ] **Gate 7**: Measure genuine on-device thermal throttling, battery draw, and latency.
