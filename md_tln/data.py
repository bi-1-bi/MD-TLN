from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass(frozen=True)
class StationGridMapping:
    station_id: str
    row: int
    col: int


class MinMaxScaler:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.minimum: float | None = None
        self.maximum: float | None = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        self.minimum = float(np.min(values))
        self.maximum = float(np.max(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (values - self.minimum) / (self.maximum - self.minimum + self.epsilon)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return values * (self.maximum - self.minimum + self.epsilon) + self.minimum


def make_supervised_sequences(
    values: np.ndarray,
    config: DataConfig = DataConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time-indexed tensor to sliding input-target windows."""

    if values.ndim < 2:
        raise ValueError("values must have shape [time, ...].")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_window = config.input_length + config.forecast_horizon
    for start in range(values.shape[0] - total_window + 1):
        split = start + config.input_length
        end = split + config.forecast_horizon
        inputs.append(values[start:split])
        targets.append(values[split:end])

    if not inputs:
        raise ValueError("Not enough timesteps to build a supervised dataset.")

    return np.stack(inputs), np.stack(targets)


def station_series_to_grid(
    station_values: np.ndarray,
    mappings: Sequence[StationGridMapping],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Map [time, station, channel] data to [time, channel, height, width]."""

    if station_values.ndim != 3:
        raise ValueError("station_values must have shape [time, station, channel].")

    time_steps, _, channels = station_values.shape
    grid = np.zeros((time_steps, channels, height, width), dtype=station_values.dtype)
    for station_idx, mapping in enumerate(mappings):
        if not (0 <= mapping.row < height and 0 <= mapping.col < width):
            raise ValueError(f"Station {mapping.station_id} is outside the grid.")
        grid[:, :, mapping.row, mapping.col] += station_values[:, station_idx, :]
    return grid


class PassengerFlowDataset(Dataset):
    """Torch dataset for MD-TLN inputs."""

    def __init__(
        self,
        history: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        spatial: np.ndarray | torch.Tensor | None = None,
        external: np.ndarray | torch.Tensor | None = None,
        time_label: np.ndarray | torch.Tensor | None = None,
        day_type_label: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        self.history = torch.as_tensor(history, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.spatial = None if spatial is None else torch.as_tensor(spatial, dtype=torch.float32)
        self.external = None if external is None else torch.as_tensor(external, dtype=torch.float32)
        self.time_label = None if time_label is None else torch.as_tensor(time_label, dtype=torch.long)
        self.day_type_label = (
            None if day_type_label is None else torch.as_tensor(day_type_label, dtype=torch.long)
        )

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "history": self.history[index],
            "target": self.target[index],
        }
        if self.spatial is not None:
            item["spatial"] = self.spatial[index]
        if self.external is not None:
            item["external"] = self.external[index]
        if self.time_label is not None:
            item["time_label"] = self.time_label[index]
        if self.day_type_label is not None:
            item["day_type_label"] = self.day_type_label[index]
        return item

