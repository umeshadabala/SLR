"""Unit tests for Pydantic schemas and serialization."""

from exp07.schemas import (
    BenchmarkRecord,
    GenerationJobResponse,
    HealthResponse,
    ModelStatusResponse,
    TelemetryResponse,
)


def test_health_response():
    resp = HealthResponse(
        python_version="3.12.13",
        torch_version="2.14.1",
        transformers_version="5.19.0",
    )
    assert resp.status == "ok"
    assert resp.version == "0.7.0"
    assert resp.runtime_identity == "exp07-slr-fastapi"
    assert resp.inference_supported is True
    assert resp.model_loaded is False
    assert resp.model_state == "not_loaded"
    data = resp.model_dump()
    assert "timestamp" in data


def test_model_status_response_checkpoint_missing():
    resp = ModelStatusResponse(
        is_loaded=False,
        state="checkpoint_missing",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        target_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        total_parameters=1_543_714_304,
        active_parameters=0,
        active_layers=0,
        total_layers=28,
        layer_size_mb=89.26,
        error_message="model.safetensors not found",
        checkpoint_guidance="Place model weights in checkpoints/",
    )
    assert resp.is_loaded is False
    assert resp.state == "checkpoint_missing"
    assert resp.is_simulated is False


def test_telemetry_response():
    tel = TelemetryResponse(
        is_engine_active=True,
        current_job_id="job123",
        current_layer_idx=5,
        total_layers=28,
        active_transformer_layers=1,
        active_transformer_params=46_797_824,
        process_rss_gb=1.45,
        process_rss_mb=1484.8,
        last_layer_load_sec=0.035,
        last_layer_exec_sec=0.012,
        last_layer_release_sec=0.005,
    )
    assert tel.active_transformer_layers == 1
    assert tel.active_transformer_params == 46_797_824
    assert tel.is_simulated is False


def test_benchmark_record():
    bm = BenchmarkRecord(
        benchmark_id="BM-01",
        experiment_id="EXP-05",
        title="Test Benchmark",
        timestamp="2026-10-06 12:00",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        precision="BF16",
        device="Darwin Apple Silicon",
        os_platform="macOS",
        prompt_length=10,
        output_length=5,
        total_generation_sec=2.5,
        tokens_per_second=2.0,
        peak_rss_gb=2.1,
        validation_status="VALIDATED",
    )
    assert bm.tokens_per_second == 2.0
    assert bm.is_simulated is False
