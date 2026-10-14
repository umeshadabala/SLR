"""FastAPI router implementing the EXP-07 SLR Engine HTTP contract."""

from __future__ import annotations

import platform
import threading
import time
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
import torch
import transformers

from exp07.config import config
from exp07.inference.engine import SLRInferenceEngine
from exp07.schemas import (
    BenchmarkRecord,
    ExperimentSummary,
    GenerationJobResponse,
    GenerationRequest,
    HealthResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelStatusResponse,
    TelemetryResponse,
    UnloadModelResponse,
)
from exp07.services.experiment_service import ExperimentService

router = APIRouter()

# Engine singleton and experiment service
engine = SLRInferenceEngine(config)
experiment_service = ExperimentService(config.results_dir)


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service health check returning real runtime and model status."""
    status = engine.get_status()
    reason = None
    if not status.is_loaded:
        reason = status.error_message or status.checkpoint_guidance or "Model is not loaded."
    return HealthResponse(
        status="ok",
        version="0.7.0",
        runtime_identity="exp07-slr-fastapi",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        inference_supported=True,
        model_loaded=status.is_loaded,
        model_state=status.state,
        model_id=status.model_id,
        unavailable_reason=reason,
    )


@router.get("/api/v1/model/status", response_model=ModelStatusResponse)
def get_model_status() -> ModelStatusResponse:
    """Truthful model availability and residency status."""
    return engine.get_status()


@router.post("/api/v1/model/load", response_model=LoadModelResponse)
def load_model(req: LoadModelRequest) -> LoadModelResponse:
    """Attempt to load the checkpoint onto the META device."""
    success = engine.load_model(
        checkpoint_dir=req.checkpoint_dir,
        checkpoint_file=req.checkpoint_file,
        dtype=req.dtype,
    )
    status = engine.get_status()
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load checkpoint: {status.error_message}",
        )

    return LoadModelResponse(
        success=True,
        state=status.state,
        message="Model successfully initialized on META device.",
        checkpoint_path=status.checkpoint_path,
        total_parameters=status.total_parameters,
        initial_meta_parameters=status.total_parameters,
        initial_cpu_parameters=0,
    )


@router.post("/api/v1/model/unload", response_model=UnloadModelResponse)
def unload_model() -> UnloadModelResponse:
    """Unload active model from memory."""
    engine.unload_model()
    return UnloadModelResponse(
        success=True,
        message="Model unloaded and resources reclaimed.",
    )


@router.get("/api/v1/models")
def list_models() -> List[Dict[str, Any]]:
    """Return model catalog with availability and verification status."""
    status = engine.get_status()
    meta = config.model_meta
    return [
        {
            "model_id": meta.model_id,
            "revision": meta.exact_revision,
            "expected_files": [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            ],
            "total_size_bytes": meta.total_bf16_bytes,
            "num_layers": meta.num_hidden_layers,
            "total_parameters": meta.expected_total_params,
            "is_loaded": status.is_loaded,
            "state": status.state,
            "checkpoint_path": status.checkpoint_path,
        }
    ]


@router.post("/api/v1/generation/start", response_model=GenerationJobResponse)
@router.post("/api/v1/generate/start", response_model=GenerationJobResponse)
def start_generation(req: GenerationRequest) -> GenerationJobResponse:
    """Trigger autoregressive generation with selective residency."""
    t_recv = time.perf_counter()
    status = engine.get_status()
    if not status.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model is not loaded (current state: '{status.state}'). "
                f"{status.checkpoint_guidance or status.error_message or ''}"
            ),
        )

    return engine.generate(req, t_server_received=t_recv)


@router.post("/api/v1/generation/stream")
@router.post("/api/v1/generate/stream")
def stream_generation(req: GenerationRequest):
    """Server-Sent Events (SSE) streaming autoregressive generation."""
    t_recv = time.perf_counter()
    status = engine.get_status()
    if not status.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model is not loaded (current state: '{status.state}'). "
                f"{status.checkpoint_guidance or status.error_message or ''}"
            ),
        )

    return StreamingResponse(
        engine.generate_stream(req, t_server_received=t_recv),
        media_type="text/event-stream",
    )


@router.get("/api/v1/results/runs")
def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve historical per-request measurement run records."""
    return engine.experiment_service.list_run_reports(limit=limit)


@router.get("/api/v1/results/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """Retrieve detailed measurement report for a single run."""
    report = engine.experiment_service.get_run_report(run_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Run report '{run_id}' not found.")
    return report


@router.get("/api/v1/generation/{job_id}", response_model=GenerationJobResponse)
def get_generation_job(job_id: str) -> GenerationJobResponse:
    """Query progress or result of a generation job."""
    if job_id not in engine.active_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return engine.active_jobs[job_id]


@router.post("/api/v1/generation/{job_id}/cancel")
def cancel_generation_job(job_id: str) -> Dict[str, Any]:
    """Request cancellation of an in-flight generation job."""
    success = engine.cancel(job_id)
    return {"job_id": job_id, "cancelled": success}


@router.get("/api/v1/telemetry", response_model=TelemetryResponse)
def get_telemetry() -> TelemetryResponse:
    """Real-time layer residency and process memory telemetry."""
    return engine.get_telemetry()


@router.get("/api/v1/benchmarks", response_model=List[BenchmarkRecord])
def list_benchmarks() -> List[BenchmarkRecord]:
    """Retrieve all validated SLR benchmark records."""
    return experiment_service.get_all_benchmarks()


@router.get("/api/v1/experiments", response_model=List[ExperimentSummary])
def list_experiments() -> List[ExperimentSummary]:
    """List historical and active research experiment series."""
    return experiment_service.list_experiments()


@router.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentSummary)
def get_experiment(experiment_id: str) -> ExperimentSummary:
    """Retrieve specific experiment series details."""
    for exp in experiment_service.list_experiments():
        if exp.experiment_id.lower() == experiment_id.lower():
            return exp
    raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")


@router.get("/api/v1/results/export")
def export_results() -> Dict[str, Any]:
    """Export complete research results and benchmark database as JSON."""
    return experiment_service.export_full_results()
