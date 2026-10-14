"""Configuration and model metadata for EXP-07 Selective Layer Residency (SLR)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelMetadata:
    """Canonical characteristics of Qwen/Qwen2.5-1.5B-Instruct."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    exact_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    num_hidden_layers: int = 28
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    rope_theta: float = 1_000_000.0

    # Theoretical parameter counts
    expected_total_params: int = 1_543_714_304
    expected_transformer_params: int = 1_310_339_072
    expected_layer_params: int = 46_797_824

    # Theoretical storage sizes in BF16 (bytes)
    total_bf16_bytes: int = 3_087_428_608  # ~2.875 GB
    transformer_bf16_bytes: int = 2_620_678_144  # ~2.4407 GB
    layer_bf16_bytes: int = 93_595_648  # 46,797,824 * 2 bytes = ~89.26 MB
    layer_bf16_mb: float = 89.260032


@dataclass
class AppConfig:
    """Runtime configuration for EXP-07 SLR Engine."""

    # Server settings
    host: str = os.getenv("SLR_HOST", "127.0.0.1")
    port: int = int(os.getenv("SLR_PORT", "8000"))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://localhost:8080",
            "http://localhost:*",
        ]
    )

    # Model and checkpoint resolution
    # Order of checks:
    # 1. SLR_CHECKPOINT_DIR env var
    # 2. Local relative checkpoints/Qwen2.5-1.5B-Instruct
    # 3. /content/EXP-05/checkpoints/source (Colab compatibility if mounted)
    checkpoint_dir: Optional[str] = os.getenv("SLR_CHECKPOINT_DIR", None)
    checkpoint_file: Optional[str] = os.getenv("SLR_CHECKPOINT_FILE", None)

    # Storage paths
    results_dir: str = os.getenv("SLR_RESULTS_DIR", "results")
    configs_dir: str = os.getenv("SLR_CONFIGS_DIR", "configs")

    # Architecture metadata
    model_meta: ModelMetadata = field(default_factory=ModelMetadata)

    def resolve_checkpoint_file(self) -> Optional[Path]:
        """Locate the actual model.safetensors file if present."""
        if self.checkpoint_file:
            p = Path(self.checkpoint_file)
            return p.resolve() if p.is_file() else None

        candidates: list[Path] = []

        if self.checkpoint_dir:
            candidates.append(Path(self.checkpoint_dir) / "model.safetensors")

        # Standard project candidate paths
        candidates.extend([
            Path("artifacts/source/model.safetensors"),
            Path("checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors"),
            Path("checkpoints/source/model.safetensors"),
            Path("models/model.safetensors"),
            Path("../checkpoints/source/model.safetensors"),
            Path("/content/EXP-05/checkpoints/source/model.safetensors"),
        ])

        for c in candidates:
            if c.is_file():
                return c.resolve()

        return None


# Global default configuration instance
config = AppConfig()
