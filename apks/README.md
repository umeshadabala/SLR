# Pre-built Android APK Packages

This directory contains pre-built Android installation packages for Selective Layer Residency (SLR).

## Available APKs

1. **`exp07-slr-neon-latest.apk.zip`** (68 MB)
   - **Recommended Build**: Latest optimized build featuring native C++ ARM NEON SIMD kernels, direct BF16 memory-mapped GEMV/GEMM, vectorized multi-token prefill, and strict 1/28 layer residency.
   - **Installation**:
     ```bash
     unzip exp07-slr-neon-latest.apk.zip
     adb install -r app/build/app/outputs/apk/debug/app-debug.apk
     ```

2. **`selectivellm-0.1.0-arm64-v8a-debug.apk`** (21 MB)
   - Standalone arm64-v8a targeted package.
   - **Installation**:
     ```bash
     adb install -r selectivellm-0.1.0-arm64-v8a-debug.apk
     ```

3. **`exp07-slr-debug.apk.zip`** (66 MB)
   - Baseline EXP-07 debug build.
   - **Installation**:
     ```bash
     unzip exp07-slr-debug.apk.zip
     adb install -r exp07-slr-debug.apk
     ```

> *Note: Due to GitHub's 100 MB single-file limit, multi-ABI debug packages (>150 MB) are compressed with standard zip.*
