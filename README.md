# EXP-07 — Selective Layer Residency (SLR) System

A research application and evaluation platform for **Selective Layer Residency (SLR)** inference with low-memory language models.

Target Model: **`Qwen/Qwen2.5-1.5B-Instruct`** (Revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`).

---

## Architecture

- **Frontend**: Flutter 3.47 / Dart 3.13 (`app/`)
  - Dashboard: Live model state, connection status, memory indicators
  - Inference: Prompt input, execution controls, real output, latency breakdown
  - Residency: Live 28-layer transformer matrix, layer load/exec/release breakdown
  - Benchmarks: Historical validated results from EXP-01 through EXP-06.5
  - Experiments: Complete experiment series index with full JSON export
  - Settings: Checkpoint path resolution, local endpoint configuration
- **Backend**: Python 3.12 + FastAPI + PyTorch + Transformers (`backend/`)
  - Strict 1-layer residency invariant via PyTorch `meta` device
  - DynamicCache autoregressive generation directly adapted from validated research (`experiment.ipynb` Cells 128–132)
  - Isolated virtual environment (`backend/.venv`) preserving system packages
  - REST API bound to loopback `127.0.0.1:8000`

---

## Quick Start (Mac Development)

### 1. Run Automated Test Suite
```bash
./scripts/test_all.sh
```
Runs both PyTest (16 tests) and Flutter analyzer & widget tests (4 tests).

### 2. Start the Backend Service
```bash
./scripts/start_backend.sh
```
Listens on `http://127.0.0.1:8000`. Health check: `http://127.0.0.1:8000/health`.

### 3. Launch Development Environment (Backend + Flutter)
```bash
./scripts/run_dev.sh
```
Starts the Python backend and launches the Flutter application in Chrome.

---

## Checkpoint Setup
To execute local inference with the full model:
1. Download or link `Qwen2.5-1.5B-Instruct` checkpoint (`model.safetensors`, `config.json`, `tokenizer.json`).
2. Place files into:
   ```
   checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors
   ```
   or configure the `SLR_CHECKPOINT_DIR` environment variable:
   ```bash
   export SLR_CHECKPOINT_DIR="/path/to/checkpoint"
   ```
3. If no checkpoint is present, the application truthfully displays a `checkpoint_missing` status with instructions. Per research rules, it **never** fabricates or simulates model outputs.

---

## Directory Layout

```
.
├── README.md
├── .gitignore
├── experiment.ipynb            # Original research notebook (preserved intact)
├── app/                        # Flutter frontend application
│   ├── lib/
│   │   ├── main.dart
│   │   ├── app.dart
│   │   ├── core/               # Theme tokens and API configuration
│   │   ├── data/               # Models, API client, and SlrRepository
│   │   ├── features/           # Dashboard, Inference, Residency, Benchmarks, etc.
│   │   └── shared/             # MetricCard, StatusBadge, WaterfallChart
│   └── test/                   # Widget tests
├── backend/                    # Python inference service
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── src/exp07/              # Engine, layer residency, schemas, routes
│   └── tests/                  # Unit and integration tests
├── configs/
│   └── model.example.json      # Model architecture configuration
├── docs/                       # Technical architecture & migration plans
│   ├── ARCHITECTURE.md
│   ├── MEMORY_ACCOUNTING.md
│   ├── ANDROID_MIGRATION.md
│   └── EXPERIMENT_STATUS.md
├── results/                    # Validated research results & exports
│   └── historical/             # Unpacked historical experiment archives
└── scripts/
    ├── start_backend.sh        # Backend launcher
    ├── run_dev.sh              # Full stack dev launcher
    └── test_all.sh             # Unified test runner
```
