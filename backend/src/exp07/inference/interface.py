"""Abstract inference engine interface for Selective Layer Residency."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from exp07.schemas import (
    GenerationJobResponse,
    GenerationRequest,
    ModelStatusResponse,
    TelemetryResponse,
)


class InferenceEngine(ABC):
    """Abstract interface decoupling API and inference implementation."""

    @abstractmethod
    def get_status(self) -> ModelStatusResponse:
        """Return truthful engine and model status."""
        pass

    @abstractmethod
    def load_model(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> bool:
        """Load model onto META device and materialize non-layer weights."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model and free resources."""
        pass

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[GenerationJobResponse], None]] = None,
    ) -> GenerationJobResponse:
        """Execute autoregressive generation with selective layer residency."""
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel an in-flight generation request."""
        pass

    @abstractmethod
    def get_telemetry(self) -> TelemetryResponse:
        """Get live telemetry of current residency, layer status, and memory."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean shutdown of engine resources."""
        pass
