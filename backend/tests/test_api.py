"""Integration tests for EXP-07 SLR Engine FastAPI routes."""

from fastapi.testclient import TestClient
from exp07.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "torch_version" in data
    assert "transformers_version" in data
    assert data["runtime_identity"] == "exp07-slr-fastapi"
    assert "inference_supported" in data
    assert "model_loaded" in data
    assert "model_state" in data


def test_model_status_endpoint():
    res = client.get("/api/v1/model/status")
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert data["total_layers"] == 28
    assert data["total_parameters"] == 1_543_714_304
    assert data["is_simulated"] is False
    assert data["state"] in ["ready", "not_loaded", "checkpoint_missing"]


def test_telemetry_endpoint():
    res = client.get("/api/v1/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["total_layers"] == 28
    assert "process_rss_mb" in data
    assert data["is_simulated"] is False


def test_benchmarks_endpoint():
    res = client.get("/api/v1/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0  # Should include historical validated benchmarks
    assert data[0]["is_simulated"] is False


def test_experiments_endpoint():
    res = client.get("/api/v1/experiments")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    exp_ids = [e["experiment_id"] for e in data]
    assert "EXP-01" in exp_ids
    assert "EXP-05" in exp_ids
    assert "EXP-07" in exp_ids


def test_results_export_endpoint():
    res = client.get("/api/v1/results/export")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "Selective Layer Residency (SLR)"
    assert data["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "experiments" in data


def test_generation_without_loaded_model_fails_honestly():
    # Per rule: UI and API must NEVER fake inference
    res = client.post(
        "/api/v1/generation/start",
        json={"prompt": "What is SLR?", "max_new_tokens": 4},
    )
    # If model is not loaded, should reject with 400
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "not loaded" in detail.lower()


def test_models_catalog_endpoint():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert data[0]["num_layers"] == 28


def test_unload_model_endpoint():
    res = client.post("/api/v1/model/unload")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    # Confirm model status after unload
    status_res = client.get("/api/v1/model/status")
    assert status_res.status_code == 200
    assert status_res.json()["is_loaded"] is False


def test_stream_endpoint_unloaded_rejects():
    res = client.post(
        "/api/v1/generation/stream",
        json={"prompt": "Hello", "max_new_tokens": 4},
    )
    assert res.status_code == 400
    assert "not loaded" in res.json()["detail"].lower()

