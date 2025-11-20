"""Boundary detection head scaffold (synthetic baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoundaryConfig:
    output_dir: Path
    embed_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 1e-3
    device: str = "cpu"


def train_boundary_model(config: BoundaryConfig) -> dict:
    """Placeholder for a boundary detection training loop."""
    raise NotImplementedError("Boundary training not implemented yet.")
