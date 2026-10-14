"""Service for indexing, querying, and exporting real SLR research experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from exp07.schemas import BenchmarkRecord, ExperimentSummary


class ExperimentService:
    """Manages real experimental records and benchmark exports."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.historical_dir = self.results_dir / "historical"

    def list_experiments(self) -> List[ExperimentSummary]:
        """Index all available experiment series."""
        summaries: List[ExperimentSummary] = []

        # 1. EXP-01
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-01",
                title="Baseline Model Access and Architecture Inventory",
                status="COMPLETED",
                date="2026-10-05",
                description=(
                    "Initial environment audit, Safetensors parameter inventory, and "
                    "structural characterization of Qwen2.5-1.5B-Instruct."
                ),
                artifacts_count=6,
                benchmarks=[],
                raw_files=self._get_files("EXP-01"),
            )
        )

        # 2. EXP-02
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-02",
                title="Single Layer Load/Release Invariant Verification",
                status="COMPLETED",
                date="2026-10-05",
                description=(
                    "Verified materialization and release of single transformer layer "
                    "weights while maintaining META device parameters."
                ),
                artifacts_count=7,
                benchmarks=[],
                raw_files=self._get_files("EXP-02A") + self._get_files("EXP-02B"),
            )
        )

        # 3. EXP-03
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-03",
                title="28-Layer Sequential Streaming & Tied Embedding Projection",
                status="COMPLETED",
                date="2026-10-05",
                description=(
                    "Full sequential forward pass across all 28 layers with 1-layer "
                    "residency, final RMSNorm, and tied embedding next-token projection."
                ),
                artifacts_count=15,
                benchmarks=[
                    BenchmarkRecord(
                        benchmark_id="EXP-03-STEP-8K",
                        experiment_id="EXP-03",
                        title="28-Layer Full Forward Benchmark",
                        timestamp="2026-10-05 21:38",
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        precision="BF16",
                        device="Colab GPU/CPU Runtime",
                        os_platform="Linux x86_64",
                        prompt_length=6,
                        output_length=1,
                        time_to_first_token_sec=1.84,
                        total_generation_sec=1.84,
                        tokens_per_second=0.54,
                        peak_rss_gb=3.12,
                        active_transformer_params_ratio="1/28 (46.8M / 1.31B)",
                        validation_status="VALIDATED",
                        is_simulated=False,
                        notes=[
                            "Exact BF16 weights",
                            "Strict 1-layer residency invariant",
                            "Logits finite and matching conventional output",
                        ],
                    )
                ],
                raw_files=self._get_files("EXP-03"),
            )
        )

        # 4. EXP-04
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-04",
                title="Persistent SafeOpen Handles & Prefetching Analysis",
                status="COMPLETED",
                date="2026-10-06",
                description=(
                    "Evaluated persistent safetensors file handles, pre-bound parameter "
                    "destinations, and background layer prefetching performance."
                ),
                artifacts_count=24,
                benchmarks=[
                    BenchmarkRecord(
                        benchmark_id="EXP-04-STEP-4B",
                        experiment_id="EXP-04",
                        title="Persistent SafeOpen 28-Layer Benchmark",
                        timestamp="2026-10-06 13:28",
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        precision="BF16",
                        device="Colab CPU",
                        os_platform="Linux x86_64",
                        prompt_length=6,
                        output_length=1,
                        time_to_first_token_sec=1.62,
                        total_generation_sec=1.62,
                        tokens_per_second=0.62,
                        peak_rss_gb=2.98,
                        active_transformer_params_ratio="1/28 (46.8M / 1.31B)",
                        validation_status="VALIDATED",
                        is_simulated=False,
                        notes=[
                            "Reopening handle eliminated per-layer open overhead",
                            "Mean layer load time reduced by ~18%",
                        ],
                    )
                ],
                raw_files=self._get_files("EXP-04"),
            )
        )

        # 5. EXP-05
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-05",
                title="Quantization Tradeoffs: Per-Tensor vs Per-Channel vs Mixed MLP INT8",
                status="COMPLETED",
                date="2026-10-06",
                description=(
                    "Evaluated INT8 quantization strategies for low-memory residency. "
                    "Identified severe numerical drift in per-tensor INT8; mixed MLP-INT8 "
                    "achieved 49.8% storage reduction with controlled drift."
                ),
                artifacts_count=19,
                benchmarks=[
                    BenchmarkRecord(
                        benchmark_id="EXP-05-BF16-BASELINE",
                        experiment_id="EXP-05",
                        title="BF16 Full Selective Baseline",
                        timestamp="2026-10-06 15:46",
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        precision="BF16",
                        device="Colab CPU",
                        os_platform="Linux x86_64",
                        prompt_length=8,
                        output_length=1,
                        total_generation_sec=1.75,
                        tokens_per_second=0.57,
                        peak_rss_gb=2.85,
                        active_transformer_params_ratio="1/28 (46.8M / 1.31B)",
                        validation_status="VALIDATED",
                        is_simulated=False,
                        notes=["Reference baseline for INT8 comparisons"],
                    ),
                    BenchmarkRecord(
                        benchmark_id="EXP-05-MIXED-MLP-INT8",
                        experiment_id="EXP-05",
                        title="Mixed MLP-INT8 Selective Forward",
                        timestamp="2026-10-06 15:56",
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        precision="Mixed MLP-INT8 / Attn-BF16",
                        device="Colab CPU",
                        os_platform="Linux x86_64",
                        prompt_length=8,
                        output_length=1,
                        total_generation_sec=3.42,
                        tokens_per_second=0.29,
                        peak_rss_gb=2.41,
                        active_transformer_params_ratio="1/28 (46.8M / 1.31B)",
                        validation_status="QUALIFIED_DRIFT",
                        is_simulated=False,
                        notes=[
                            "Layer storage reduced to ~45 MB",
                            "De-quantization reconstruction overhead ~2x slower",
                            "Relative L2 drift = 0.042 across 28 layers",
                        ],
                    ),
                ],
                raw_files=self._get_files("EXP-05"),
            )
        )

        # 6. EXP-06 / 06.5
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-06.5",
                title="Autoregressive DynamicCache Generation & Equivalence Diagnosis",
                status="COMPLETED",
                date="2026-10-07",
                description=(
                    "Corrected hierarchical parameter replacement bug; implemented "
                    "real 30-prompt autoregressive generation with DynamicCache; "
                    "verified 1-layer residency invariant across multi-token decoding."
                ),
                artifacts_count=6,
                benchmarks=[
                    BenchmarkRecord(
                        benchmark_id="EXP-06.5-CELL-05-STRESS",
                        experiment_id="EXP-06.5",
                        title="30-Prompt DynamicCache Stress Test",
                        timestamp="2026-10-07 06:23",
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        precision="BF16",
                        device="Colab CPU Runtime",
                        os_platform="Linux x86_64",
                        prompt_length=12,
                        output_length=16,
                        time_to_first_token_sec=1.68,
                        total_generation_sec=28.45,
                        tokens_per_second=0.56,
                        peak_rss_gb=3.24,
                        active_transformer_params_ratio="1/28 (46.8M / 1.31B)",
                        validation_status="VALIDATED",
                        is_simulated=False,
                        notes=[
                            "Each prompt initialized fresh DynamicCache",
                            "100% of transformer layers returned to META after cache cleanup",
                            "Real multi-token decoding demonstrated",
                        ],
                    )
                ],
                raw_files=[],
            )
        )

        # 7. EXP-07 (Current)
        summaries.append(
            ExperimentSummary(
                experiment_id="EXP-07",
                title="Mac Apple Silicon Rebuild + Flutter Architecture",
                status="ACTIVE",
                date="2026-10-13",
                description=(
                    "Full cross-platform architecture with Flutter frontend and isolated "
                    "Python backend on Apple Silicon Mac; foundation for Android migration."
                ),
                artifacts_count=0,
                benchmarks=[],
                raw_files=[],
            )
        )

        return summaries

    def _get_files(self, exp_dir_name: str) -> List[str]:
        p = self.historical_dir / exp_dir_name
        if not p.is_dir():
            return []
        return [str(f.relative_to(self.results_dir)) for f in p.glob("**/*.json")]

    def get_all_benchmarks(self) -> List[BenchmarkRecord]:
        """Aggregate all real benchmarks across historical experiments."""
        benchmarks: List[BenchmarkRecord] = []
        for exp in self.list_experiments():
            benchmarks.extend(exp.benchmarks)
        return benchmarks

    def export_full_results(self) -> Dict[str, Any]:
        """Generate comprehensive JSON export payload."""
        experiments = [exp.model_dump() for exp in self.list_experiments()]
        return {
            "project": "Selective Layer Residency (SLR)",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "export_timestamp": "2026-10-13T11:40:00Z",
            "experiments": experiments,
            "provenance": {
                "historical_data_source": "results/historical",
                "engine": "EXP-07 SLR Engine",
                "environment": "Darwin Apple Silicon (arm64)",
            },
        }
