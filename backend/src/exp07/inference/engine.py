"""Production implementation of SLR Inference Engine for EXP-07."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from exp07.config import AppConfig, ModelMetadata
from exp07.inference.generation import run_slr_generation
from exp07.inference.interface import InferenceEngine
from exp07.inference.layer_residency import (
    count_device_params,
    get_process_rss_gb,
    get_process_rss_mb,
)
from exp07.inference.model_loader import (
    build_meta_model,
    load_tokenizer_if_available,
    materialize_non_layer_weights,
)
from exp07.schemas import (
    GenerationJobResponse,
    GenerationRequest,
    ModelStatusResponse,
    TelemetryResponse,
)

logger = logging.getLogger(__name__)


class SLRInferenceEngine(InferenceEngine):
    """Thread-safe Selective Layer Residency inference engine."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.metadata = config.model_meta
        self._lock = threading.Lock()

        # Model objects
        self.model: Optional[torch.nn.Module] = None
        self.model_config = None
        self.tokenizer = None
        self.loaded_checkpoint_path: Optional[str] = None
        self.is_loaded: bool = False
        self.last_error: Optional[str] = None

        # Job tracking
        self.active_jobs: Dict[str, GenerationJobResponse] = {}
        self.cancelled_jobs: set[str] = set()

        # Telemetry tracking
        self.is_active: bool = False
        self.current_job_id: Optional[str] = None
        self.current_layer_idx: Optional[int] = None
        self.active_layer_params: int = 0
        self.last_load_sec: Optional[float] = None
        self.last_exec_sec: Optional[float] = None
        self.last_release_sec: Optional[float] = None

        # Auto-discover checkpoint on startup
        self._auto_discover_checkpoint()

    def _auto_discover_checkpoint(self) -> None:
        """Probe configured candidates for existing checkpoints."""
        resolved = self.config.resolve_checkpoint_file()
        if resolved:
            self.loaded_checkpoint_path = str(resolved)
            logger.info(f"Discovered existing checkpoint at: {resolved}")
        else:
            logger.info("No local Qwen2.5-1.5B-Instruct checkpoint discovered on startup.")

    def get_status(self) -> ModelStatusResponse:
        """Truthful report of engine and model availability."""
        with self._lock:
            resolved = self.loaded_checkpoint_path or self.config.resolve_checkpoint_file()
            resolved_str = str(resolved) if resolved else None

            if self.is_loaded and self.model is not None:
                state = "executing" if self.is_active else "ready"
                error_msg = None
            elif resolved_str and Path(resolved_str).is_file():
                state = "not_loaded"
                error_msg = self.last_error
            else:
                state = "checkpoint_missing"
                error_msg = self.last_error or "Checkpoint 'model.safetensors' not found."

            guidance = None
            if state == "checkpoint_missing":
                guidance = (
                    "Place Qwen2.5-1.5B-Instruct safetensors checkpoint into "
                    "'checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors' or configure "
                    "SLR_CHECKPOINT_DIR environment variable."
                )

            active_params = 0
            active_layers = 0
            if self.is_loaded and self.model is not None:
                # Count active layer parameters on CPU
                for layer in self.model.model.layers:
                    cpu_p = sum(
                        p.numel() for p in layer.parameters() if p.device.type == "cpu"
                    )
                    if cpu_p > 0:
                        active_layers += 1
                        active_params += cpu_p

            return ModelStatusResponse(
                is_loaded=self.is_loaded,
                state=state,
                model_id=self.metadata.model_id,
                target_revision=self.metadata.exact_revision,
                checkpoint_path=resolved_str,
                engine_type="SelectiveLayerResidencyEngine",
                is_simulated=False,
                dtype="bfloat16",
                total_parameters=self.metadata.expected_total_params,
                active_parameters=active_params,
                active_layers=active_layers,
                total_layers=self.metadata.num_hidden_layers,
                layer_size_mb=self.metadata.layer_bf16_mb,
                error_message=error_msg,
                checkpoint_guidance=guidance,
            )

    def load_model(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> bool:
        """Load checkpoint onto META device and materialize non-layer parameters."""
        with self._lock:
            target_file: Optional[Path] = None
            if checkpoint_file:
                target_file = Path(checkpoint_file)
            elif checkpoint_dir:
                target_file = Path(checkpoint_dir) / "model.safetensors"
            else:
                target_file = self.config.resolve_checkpoint_file()

            if not target_file or not target_file.is_file():
                self.last_error = f"Checkpoint file not found: {target_file}"
                return False

            source_dir = str(target_file.parent)
            target_dtype = (
                torch.bfloat16 if dtype == "bfloat16" else torch.float16
            )

            try:
                # 1. Build META model
                model, model_config = build_meta_model(
                    source_dir,
                    model_metadata=self.metadata,
                )

                # 2. Materialize embeddings, norm, and RoPE
                materialize_non_layer_weights(
                    model,
                    model_config,
                    str(target_file),
                    dtype=target_dtype,
                )

                # 3. Load tokenizer
                self.tokenizer = load_tokenizer_if_available(source_dir)
                self.model = model
                self.model_config = model_config
                self.loaded_checkpoint_path = str(target_file)
                self.is_loaded = True
                self.last_error = None
                logger.info(f"Successfully loaded model from {target_file}")
                return True
            except Exception as e:
                self.last_error = f"Failed to load model: {str(e)}"
                logger.exception("Error during load_model")
                self.is_loaded = False
                return False

    def unload_model(self) -> None:
        """Unload model and free RAM."""
        with self._lock:
            self.model = None
            self.model_config = None
            self.tokenizer = None
            self.is_loaded = False
            self.loaded_checkpoint_path = None
            import gc
            gc.collect()

    def _prepare_prompt(self, request: GenerationRequest) -> str:
        if request.messages and len(request.messages) > 0:
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    return self.tokenizer.apply_chat_template(
                        [m.model_dump() for m in request.messages],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            except Exception:
                pass
            formatted = ""
            for m in request.messages:
                formatted += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
            return formatted
        return request.prompt or ""

    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[GenerationJobResponse], None]] = None,
    ) -> GenerationJobResponse:
        """Execute real generation or report accurate unavailable state."""
        job_id = request.job_id or str(uuid.uuid4())[:8]
        effective_prompt = self._prepare_prompt(request)

        with self._lock:
            if not self.is_loaded or self.model is None or not self.loaded_checkpoint_path:
                status_resp = self.get_status()
                job_err = GenerationJobResponse(
                    job_id=job_id,
                    status="failed",
                    prompt=effective_prompt,
                    error=(
                        f"Cannot run inference: engine status is '{status_resp.state}'. "
                        f"{status_resp.error_message or ''}"
                    ),
                    is_simulated=False,
                )
                self.active_jobs[job_id] = job_err
                return job_err

            if self.tokenizer is None:
                job_err = GenerationJobResponse(
                    job_id=job_id,
                    status="failed",
                    prompt=effective_prompt,
                    error="Tokenizer is not available for this checkpoint.",
                    is_simulated=False,
                )
                self.active_jobs[job_id] = job_err
                return job_err

            self.is_active = True
            self.current_job_id = job_id

        def cancel_check() -> bool:
            return job_id in self.cancelled_jobs

        def telemetry_cb(l_idx, a_params, t_load, t_exec, t_rel):
            self.current_layer_idx = l_idx
            self.active_layer_params = a_params
            self.last_load_sec = t_load
            self.last_exec_sec = t_exec
            self.last_release_sec = t_rel

        try:
            result = run_slr_generation(
                model=self.model,
                tokenizer=self.tokenizer,
                safetensors_file=self.loaded_checkpoint_path,
                prompt=effective_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                stop_sequences=request.stop_sequences,
                job_id=job_id,
                cancel_check=cancel_check,
                telemetry_callback=telemetry_cb,
                metadata=self.metadata,
            )
            self.active_jobs[job_id] = result
            return result
        except Exception as e:
            logger.exception("Generation error")
            err_res = GenerationJobResponse(
                job_id=job_id,
                status="failed",
                prompt=effective_prompt,
                error=f"Runtime error during layer execution: {str(e)}",
                is_simulated=False,
            )
            self.active_jobs[job_id] = err_res
            return err_res
        finally:
            with self._lock:
                self.is_active = False
                self.current_job_id = None
                self.current_layer_idx = None
                self.active_layer_params = 0

    def generate_stream(
        self,
        request: GenerationRequest,
    ):
        """Yield Server-Sent Events (SSE) data chunks as tokens generate."""
        import json
        import queue
        import threading

        job_id = request.job_id or str(uuid.uuid4())[:8]
        effective_prompt = self._prepare_prompt(request)
        token_q = queue.Queue()

        def on_token(token_id: int, token_str: str, elapsed_sec: float):
            token_q.put({"type": "token", "token_id": token_id, "text": token_str, "elapsed_sec": elapsed_sec})

        def run_worker():
            try:
                res = run_slr_generation(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    safetensors_file=self.loaded_checkpoint_path,
                    prompt=effective_prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    stop_sequences=request.stop_sequences,
                    job_id=job_id,
                    cancel_check=lambda: job_id in self.cancelled_jobs,
                    token_callback=on_token,
                    metadata=self.metadata,
                )
                token_q.put({"type": "done", "response": res.model_dump()})
            except Exception as ex:
                token_q.put({"type": "error", "error": str(ex)})

        worker_thread = threading.Thread(target=run_worker, daemon=True)
        worker_thread.start()

        while True:
            try:
                item = token_q.get(timeout=60.0)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ["done", "error"]:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    def cancel(self, job_id: str) -> bool:
        """Flag job for cancellation."""
        self.cancelled_jobs.add(job_id)
        if job_id in self.active_jobs:
            self.active_jobs[job_id].status = "cancelled"
        return True

    def get_telemetry(self) -> TelemetryResponse:
        """Provide instantaneous residency and process RSS telemetry."""
        return TelemetryResponse(
            is_engine_active=self.is_active,
            current_job_id=self.current_job_id,
            current_layer_idx=self.current_layer_idx,
            total_layers=self.metadata.num_hidden_layers,
            active_transformer_layers=1 if self.current_layer_idx is not None else 0,
            active_transformer_params=self.active_layer_params,
            process_rss_gb=get_process_rss_gb(),
            process_rss_mb=get_process_rss_mb(),
            kv_cache_sequence_length=0,
            last_layer_load_sec=self.last_load_sec,
            last_layer_exec_sec=self.last_exec_sec,
            last_layer_release_sec=self.last_release_sec,
            engine_mode="selective_residency_bf16",
            is_simulated=False,
            timestamp=time.time(),
        )

    def close(self) -> None:
        self.unload_model()
