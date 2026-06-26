from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PatchList = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters for the MD-TLN model."""

    input_length: int = 6
    forecast_horizon: int = 1
    interval_minutes: int = 5

    history_channels: int = 2
    spatial_channels: int = 1
    external_channels: int = 0
    output_channels: int = 2

    grid_height: int = 32
    grid_width: int = 32

    encoding_dim: int = 256
    patch_sizes: PatchList = field(
        default_factory=lambda: (
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (8, 8),
            (8, 8),
            (8, 8),
            (8, 8),
        )
    )
    attention_layers: int = 1
    cross_attention_layers: int = 1
    attention_heads: int = 1
    dropout: float = 0.2
    sce_reduction: int = 16
    use_skip: bool = True
    use_channel_reduction: bool = True
    norm_type: Literal["layer", "batch"] = "layer"
    output_activation: Literal["relu", "tanh", "identity"] = "relu"

    def validate(self) -> None:
        if self.input_length <= 0:
            raise ValueError("input_length must be positive.")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if self.forecast_horizon > self.input_length:
            raise ValueError("forecast_horizon cannot exceed input_length.")
        if self.encoding_dim % len(self.patch_sizes) != 0:
            raise ValueError("encoding_dim must be divisible by patch scale count.")
        for patch_h, patch_w in self.patch_sizes:
            if self.grid_height % patch_h != 0 or self.grid_width % patch_w != 0:
                raise ValueError(
                    f"Patch size {(patch_h, patch_w)} must divide the grid "
                    f"size {(self.grid_height, self.grid_width)}."
                )


@dataclass(frozen=True)
class DataConfig:
    """Data-window parameters from the manuscript problem formulation."""

    input_length: int = 6
    forecast_horizon: int = 1
    interval_minutes: int = 5
    grid_height: int = 32
    grid_width: int = 32


@dataclass(frozen=True)
class TrainingConfig:
    """Training parameters aligned with the manuscript training details."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    gradient_clip_norm: float | None = None
    checkpoint_path: str | None = "best_md_tln.pt"

