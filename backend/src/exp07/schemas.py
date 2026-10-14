"""Pydantic data schemas for EXP-07 SLR Engine REST API."""

from __future__ import annotations

import time
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.7.0"
    runtime_identity: str = "exp07-slr-fastapi"
    timestamp: float = Field(default_factory=time.time)
    python_version: str
    torch_version: str
    transformers_version: str
    inference_supported: bool = True
    model_loaded: bool = False
    model_state: str = "not_loaded"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    unavailable_reason: Optional[str] = None


class ModelStatusResponse(BaseModel):
    is_loaded: bool
    state: str = Field(
        ...,
        description=(
            "Current model state: 'ready', 'not_loaded', 'checkpoint_missing', "
            "'loading', 'executing', or 'error'"
        ),
    )
    model_id: str
    target_revision: str
    checkpoint_path: Optional[str] = None
    engine_type: str = "SelectiveLayerResidencyEngine"
    is_simulated: bool = False
    dtype: str = "bfloat16"
    total_parameters: int
    active_parameters: int
    active_layers: int
    total_layers: int
    layer_size_mb: float
    error_message: Optional[str] = None
    checkpoint_guidance: Optional[str] = None


class LoadModelRequest(BaseModel):
    checkpoint_dir: Optional[str] = None
    checkpoint_file: Optional[str] = None
    dtype: str = "bfloat16"


class LoadModelResponse(BaseModel):
    success: bool
    state: str
    message: str
    checkpoint_path: Optional[str] = None
    total_parameters: int
    initial_meta_parameters: int
    initial_cpu_parameters: int


class UnloadModelResponse(BaseModel):
    success: bool = True
    message: str = "Model unloaded successfully."


class ChatMessage(BaseModel):
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str = Field(..., min_length=1)


class GenerationRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    do_sample: bool = False
    stop_sequences: List[str] = Field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    job_id: Optional[str] = None


class LayerTimingRecord(BaseModel):
    layer_idx: int
    load_sec: float
    exec_sec: float
    release_sec: float
    active_params: int
    active_layers: int
    rss_gb: float
    kv_cache_len: int


class GenerationJobResponse(BaseModel):
    job_id: str
    status: str = Field(
        ...,
        description="'queued', 'running', 'completed', 'cancelled', or 'failed'",
    )
    prompt: str = ""
    prompt_tokens: int = 0
    generated_text: str = ""
    generated_token_ids: List[int] = Field(default_factory=list)
    generated_tokens: int = 0
    stop_reason: str = Field(
        default="eos",
        description="'eos', 'stop_sequence', 'token_limit', 'cancelled', or 'error'",
    )
    time_to_first_token_sec: Optional[float] = None
    total_generation_sec: Optional[float] = None
    tokens_per_second: Optional[float] = None
    decode_throughput: Optional[float] = None
    inter_token_latency_sec: Optional[float] = None
    peak_active_params: int = 0
    peak_active_layers: int = 0
    peak_process_rss_gb: float = 0.0
    engine_type: str = "SelectiveLayerResidencyEngine"
    is_simulated: bool = False
    error: Optional[str] = None
    layer_timings: List[LayerTimingRecord] = Field(default_factory=list)


class TelemetryResponse(BaseModel):
    is_engine_active: bool
    current_job_id: Optional[str] = None
    current_layer_idx: Optional[int] = None
    total_layers: int = 28
    active_transformer_layers: int = 0
    active_transformer_params: int = 0
    process_rss_gb: float = 0.0
    process_rss_mb: float = 0.0
    kv_cache_sequence_length: int = 0
    storage_bytes_read: Optional[int] = None
    last_layer_load_sec: Optional[float] = None
    last_layer_exec_sec: Optional[float] = None
    last_layer_release_sec: Optional[float] = None
    engine_mode: str = "selective_residency_bf16"
    is_simulated: bool = False
    timestamp: float = Field(default_factory=time.time)


class BenchmarkRecord(BaseModel):
    benchmark_id: str
    experiment_id: str
    title: str
    timestamp: str
    model: str
    precision: str
    device: str
    os_platform: str
    prompt_length: int
    output_length: int
    time_to_first_token_sec: Optional[float] = None
    total_generation_sec: float
    tokens_per_second: float
    peak_rss_gb: float
    active_transformer_params_ratio: str = "1/28 (46.8M / 1.31B)"
    validation_status: str
    is_simulated: bool = False
    notes: List[str] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    experiment_id: str
    title: str
    status: str
    date: str
    description: str
    artifacts_count: int
    benchmarks: List[BenchmarkRecord] = Field(default_factory=list)
    raw_files: List[str] = Field(default_factory=list)
