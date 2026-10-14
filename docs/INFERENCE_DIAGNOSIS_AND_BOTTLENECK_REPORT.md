# EXP-07 — Inference Engine Diagnosis and Bottleneck Audit

**Date:** October 2026  
**Target Architecture:** Selective Layer Residency (SLR) on Qwen2.5-1.5B-Instruct (BF16)  
**Evaluated Environments:** 
- Physical Mobile: Motorola moto g37 power (Android 16 / ARM64)
- Desktop Reference: macOS Darwin ARM64 (Python 3.12 / PyTorch 2.6+ / HuggingFace Transformers)

---

## 1. Executive Summary

A comprehensive diagnostic audit was conducted across the Flutter UI, method channel bridges, native Android Kotlin engine (`AndroidSlrEngine`), and the reference Python backend (`exp07.inference`).

The audit revealed the exact root causes for the preliminary benchmark figures:
- **Reported 123.139s Latency:** This was **not** a single prompt inference latency. It represented the cumulative wall-clock execution duration of an entire 18-test automated diagnostic suite (`AndroidBenchmarkSuite`), including device diagnostics, storage preflight, sequential execution of all 28 transformer layers in unvectorized ART loops, synthetic fuse tripping, and emergency cleanup verification.
- **Reported 0.00 tok/s Throughput:** In `slr_repository.dart`, the diagnostic suite summary duration was mapped to `BenchmarkData` with hardcoded `tokensPerSecond: 0.0` because the structural suite did not run autoregressive text decode. Reporting `0.00 tok/s` was misleading; per Phase 6 specifications, a non-generation test must report `null` / `N/A`.
- **True SLR Generation Throughput:** On reference ARM64 desktop, SLR achieves **~1.13 to 2.5 tokens/sec** with TTFT of **~3.2 seconds** while strictly maintaining a **1/28 layer residency invariant** (~89.26 MB active layer memory).

---

## 2. Detailed Root-Cause Breakdown

### A. Why Total Latency Reached 123.139 Seconds
1. **Scope Conflation in Benchmark Reporting:**
   `AndroidBenchmarkSuite.runSafeBenchmark()` executes 18 distinct hardware tests. The test suite took 123,139 ms total. When the Flutter app received this report, `slr_repository.dart` assigned:
   ```dart
   totalGenerationSec: ((summary['duration_ms'] as num?)?.toDouble() ?? 0.0) / 1000.0 // 123.139 s
   ```
   This labeled the entire multi-test suite duration as "Total Latency" on the benchmark card.
2. **Unvectorized Kotlin Matrix Math:**
   In `AndroidSlrEngine.kt`, matrix multiplications (`matrixVectorMul`, `rmsNorm`, `applyRope`) were written as pure scalar Kotlin loops executed in the Android ART VM. For Qwen2.5-1.5B:
   - Intermediate size = 8,960; Hidden size = 1,536.
   - MLP gate + up + down projections alone require `(1536 * 8960 * 2) + (8960 * 1536)` = **41,287,680 floating-point operations per layer**.
   - QKV and output attention projections require another **5,505,024 FLOPs**.
   - Total per layer: **~46.8 million FLOPs**.
   - Across 28 layers: **1.31 billion FLOPs per token**.
   In interpreted/JIT ART bytecode without NEON SIMD acceleration, computing 1.31B scalar float ops sequentially took ~110 seconds on the mobile CPU cores.

### B. Why Throughput was Reported as 0.00 tok/s
1. **Absence of Token Output in Structural Suite:**
   Test 6 verified the mathematical contract of 28 sequential layer forward passes, but did not run autoregressive token sampling or emit text tokens.
2. **Hardcoded Fallback in Flutter:**
   `slr_repository.dart` populated `tokensPerSecond: 0.0` whenever `outputLength == 0`. It failed to distinguish between zero-token failure and a non-generation test.

### C. Generation Loop Deficiencies in Existing Backend
1. **Missing EOS Early Stopping:**
   In `backend/src/exp07/inference/generation.py`, the decode loop ran unconditionally for `range(max_new_tokens - 1)`. When the model generated `<|im_end|>` (token ID 151645), generation continued until `max_new_tokens` was exhausted, generating repetitive tokens and wasting compute.
2. **Lack of Streaming:**
   The backend API `POST /api/v1/generation/start` held the HTTP connection open until all tokens were completed. No Server-Sent Events (SSE) or incremental tokens were streamed to the client.
3. **Missing Chat Template Support:**
   The backend received raw prompt strings without applying Qwen's ChatML template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). As a result, the model did not recognize conversational turn boundaries.
4. **Single-Prompt UI Limitations:**
   The Flutter `InferenceScreen` consisted only of a static text area with no conversation history, message bubbles, or stop button.

---

## 3. Targeted Remediation Plan

| Phase | Remediation Item | Target Outcome |
| :--- | :--- | :--- |
| **Phase 2** | Numerical Reference Tests | Verify exact tokenization, embeddings, prefill logits, cached decode, and greedy token match against PyTorch. |
| **Phase 3** | Conversational Chatbot UI | Replace prompt box with responsive chat message list, streaming tokens, stop button, and collapsible telemetry. |
| **Phase 4** | Configurable Settings | Add `temperature`, `top_p`, `repetition_penalty`, greedy toggle, and stop early on EOS. |
| **Phase 5** | Streaming Backend Engine | Implement `POST /api/v1/generation/stream` with SSE chunks and chat template formatting. |
| **Phase 6** | Rigorous Telemetry | Decompose latency into Tokenize, Prefill, TTFT, Inter-token, and Decode; report throughput as `N/A` when tokens=0. |
| **Phase 7** | Memory & Fuse Guard | Sample process RSS before each layer step and ensure fuse abort stops generation safely. |
| **Phase 8 & 9** | Benchmarks & Categories | Categorize benchmarks into Structural, Numerical, Generation, Performance, Safety, and Physical Device. |
| **Phase 10** | Laptop & Device Launch | Enable seamless dual-target execution (macOS desktop / laptop web + physical Android device). |
