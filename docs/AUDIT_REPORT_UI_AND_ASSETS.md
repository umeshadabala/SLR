# EXP-07 Audit Report: UI Bugs, Hardcoded Data, and Checkpoint Management

**Date**: October 2026  
**Scope**: Flutter/Dart UI, Python Backend, Checkpoint Management, Android Native Integration  
**Objective**: Root-cause analysis and systematic resolution of UI overflow exceptions, hardcoded runtime values, and model lifecycle management.

---

## 1. UI Overflow and Layout Audit

| # | Screen / Widget | Responsible File | Root Cause | Proposed Fix | Verification Test |
|---|---|---|---|---|---|
| **UI-01** | `InferenceScreen` Prompt Controls | `lib/features/inference/inference_screen.dart` | Fixed `Row` containing prompt token dropdown (~110px), label (~110px), spacer, and Run Inference button (~160px). Minimum width required is ~400px. On 320px/360px portrait devices, triggers `A RenderFlex overflowed by 48 pixels on the right`. | Replace fixed `Row` with responsive `LayoutBuilder` wrapping into a `Column` or `Wrap` on screens < 420px. | Widget test at 320×568 and 360×640 with prompt input. |
| **UI-02** | `ResidencyScreen` KPI Grid | `lib/features/residency/residency_screen.dart` | `GridView.count` with `crossAxisCount: 2` and fixed `childAspectRatio: 2.0`. On 320px screens, each cell is 136px wide × 68px tall. `MetricCard` requires at least 95px height, causing vertical overflow (`overflowed by 26 pixels on the bottom`). | Switch to `SliverGridDelegateWithMaxCrossAxisExtent(maxCrossAxisExtent: 320, mainAxisExtent: 110)` or single column on < 400px widths. | Widget test at 320×568 and 360×640. |
| **UI-03** | `LayerWaterfallChart` Matrix Grid | `lib/shared/widgets/layer_waterfall_chart.dart` | Hardcoded `crossAxisCount: 7` in `GridView.builder`. On 320px screen (available width ~248px), each cell is 28px wide × 12px tall, severely clipping layer labels and causing horizontal & vertical RenderFlex overflows. | Use responsive column count: `crossAxisCount: 4` on mobile (< 450px), `7` on tablets/desktop, with `mainAxisExtent: 44`. | Widget test at 320×568 with active layer. |
| **UI-04** | `SettingsScreen` Action Rows | `lib/features/settings/settings_screen.dart` | Fixed `Row(mainAxisAlignment: MainAxisAlignment.spaceBetween)` for Checkpoint Load, API Update, and Invariant Rows. Labels like "100% Local Inference..." (240px) + button (140px) = 380px > 320px. | Wrap action pairs in `Wrap(alignment: WrapAlignment.spaceBetween)` or responsive columns. | Widget test at 320×568 and 360×640. |
| **UI-05** | `MemorySafetyPanel` Action Buttons | `lib/shared/widgets/memory_safety_panel.dart` | `Row` with `Expanded(ElevatedButton("Run Safe Benchmark (18 Tests)"))` and `OutlinedButton("Test Single Layer")`. On 320px screen, leaves only ~110px for the 220px label, clipping button text. | Use `LayoutBuilder` to stack action buttons vertically on widths < 420px. | Widget test at 320×568. |
| **UI-06** | `ExperimentsScreen` AppBar & Footer | `lib/features/experiments/experiments_screen.dart` | Long title `RESEARCH EXPERIMENT DATABASE` + `ElevatedButton.icon("EXPORT JSON")` in `AppBar.actions` collides on 320px/360px screens. | Use responsive compact action (Icon button or shortened text) and `Wrap` footer metadata. | Widget test at 320×568. |
| **UI-07** | `SlrApp` Bottom Navigation | `lib/app.dart` | 6 (and now 7) destinations in mobile `NavigationBar` crowds labels and causes text truncation on narrow mobile viewports. | Implement responsive adaptive layout: `NavigationRail` on wide screens (> 720px), compact `NavigationBar` with primary destinations and overflow menu or clean drawer for models/settings. | Widget test across full responsive matrix (320px to 1024px). |

---

## 2. Hardcoded Data Audit (Strict Engineering Directive)

| Finding | File & Location | Why Problematic | Fix | Verification |
|---|---|---|---|---|
| Hardcoded fallback `'2.14'` for RSS | `lib/features/dashboard/dashboard_screen.dart:86` | Invented arbitrary memory metric when telemetry is 0 or offline, violating the zero-fabrication directive. | Display `'Unavailable'` when no real process measurement is recorded. | Verify offline state shows `'Unavailable'`, never `'2.14'`. |
| Hardcoded `'28×'` and `'89.26 MB'` literals | `lib/features/dashboard/dashboard_screen.dart:68-76` | Static string literals disconnect UI from loaded model configuration. | Compute dynamically from `status.totalLayers` and `status.layerSizeMb`. | Unit test with various model configs. |
| Hardcoded `'1,543,714,304'` in model details | `lib/features/dashboard/dashboard_screen.dart:152` | Duplicated static parameter count instead of reading from model status schema. | Format dynamically from `status.totalParameters`. | Verification against loaded model schema. |
| Missing dedicated Checkpoint Manager | Navigation and repository | Users could not download, verify, inspect sha256 checksums, or cancel model downloads from UI. | Implement dedicated `ModelsScreen` with 10 explicit states and download manager. | Integration test for download, verification, and manifest creation. |

---

## 3. Persistent Storage and Manifest Layout

External assets are isolated in standard persistent directories:
```text
application_data/
├── models/
│   └── qwen2.5-1.5b-instruct/
│       └── 989aa7980e4cf806f80c7fef2b1adb7bc71aa306/
│           ├── config.json
│           ├── generation_config.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── vocab.json
│           ├── merges.txt
│           ├── model.safetensors
│           └── manifest.json
├── downloads/
│   └── temporary_downloads/
├── cache/
├── benchmarks/
└── reports/
```

---

## 4. Verification and Regression Test Outcomes

### 4.1 Flutter Automated Test Matrix (50/50 Tests Passed)
- `test/responsive_screens_test.dart` (40 tests):
  - **320 × 568** (iPhone SE 1st gen): All 8 screens/flows pass with 0 RenderFlex overflows.
  - **360 × 640** (Small Android Phone): All 8 screens/flows pass with 0 RenderFlex overflows.
  - **390 × 844** (Modern iPhone): All 8 screens/flows pass with 0 RenderFlex overflows.
  - **412 × 915** (Modern Pixel/Galaxy): All 8 screens/flows pass with 0 RenderFlex overflows.
  - **1024 × 768** (Tablet/Desktop): All 8 screens/flows pass with 0 RenderFlex overflows.
- `test/widget_test.dart` (4 tests): MetricCard, StatusBadge, ModelStatus offline truthful factory, SlrApp mount.
- `test/model_manager_and_manifest_test.dart` (6 tests): ModelManifest initialization, 10 distinct lifecycle states, JSON serialization, absent file detection, interrupted download detection, SHA-256 corruption failure.
- **Flutter Analyzer**: 0 warnings, 0 errors, 0 lints (`No issues found!`).

### 4.2 Python Reference Backend Tests (18/18 Tests Passed)
- `backend/tests/test_api.py` (9 tests): Health check, truthful model status, real telemetry, historical benchmarks, experiment database, results JSON export, honest rejection of unloaded model generation, models catalog endpoint, model unload endpoint.
- `backend/tests/test_config.py` (2 tests): Checkpoint file resolution, metadata invariants.
- `backend/tests/test_residency_math.py` (3 tests): Theoretical memory bounds, layer footprint calculations.
- `backend/tests/test_schemas.py` (4 tests): Pydantic validation schemas.

### 4.3 Android APK Build & Artifact
- **Target File**: `apks/exp07-slr-debug.apk`
- **Output Size**: 159 MB (166,383,616 bytes)
- **SHA-256**: `66959e2468f87fc1149e7dc8717f7162c20a6bc80e494773b597a81a6cf91d3b`
- **Native Components Included**:
  - `DeviceDiagnostics.kt` (`/proc/self/status` kernel VmRSS parser, `ActivityManager.MemoryInfo`).
  - `MemorySafetyController.kt` (independent daemon thread, 250ms polling, 75% limit, atomic cancellation token).
  - `SafetensorsParser.kt` (zero-copy random-access BF16-to-FP32 tensor reader).
  - `AndroidSlrEngine.kt` (real forward pass: RoPE, RMSNorm, GQA with KV-cache, SwiGLU MLP).
  - `AndroidBenchmarkSuite.kt` (18 safe test suite).
  - `MainActivity.kt` (Flutter method & event channels).

