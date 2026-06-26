from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatialWeightConfig:
    """Optional fixed weights for spatial input encoding."""

    flow_weight: float | None = None
    period_weight: float | None = None
    duration_weight: float | None = None
    transfer_weight: float | None = None
    epsilon: float = 1e-12


def minmax_normalize(series: pd.Series, epsilon: float = 1e-12) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    span = maximum - minimum
    if span < epsilon:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
    return (series - minimum) / (span + epsilon)


def entropy_weights(
    frame: pd.DataFrame,
    columns: Sequence[str],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute entropy weights for normalized criterion columns."""

    values = frame.loc[:, columns].astype(float).to_numpy()
    values = np.clip(values, 0.0, None)
    column_sums = values.sum(axis=0, keepdims=True) + epsilon
    probabilities = values / column_sums
    n_samples = max(values.shape[0], 2)
    entropy = -(probabilities * np.log(probabilities + epsilon)).sum(axis=0)
    entropy = entropy / np.log(n_samples)
    divergence = 1.0 - entropy
    if float(divergence.sum()) < epsilon:
        weights = np.ones_like(divergence) / len(divergence)
    else:
        weights = divergence / divergence.sum()
    return {column: float(weight) for column, weight in zip(columns, weights)}


def combine_spatial_criteria(
    frame: pd.DataFrame,
    *,
    flow_col: str,
    period_col: str,
    duration_col: str,
    transfer_col: str,
    config: SpatialWeightConfig = SpatialWeightConfig(),
) -> pd.Series:
    """Apply the manuscript spatial input formula to OD or station pairs."""

    normalized = pd.DataFrame(index=frame.index)
    normalized["flow"] = minmax_normalize(frame[flow_col], config.epsilon)
    normalized["period"] = minmax_normalize(frame[period_col], config.epsilon)
    normalized["duration"] = minmax_normalize(frame[duration_col], config.epsilon)
    normalized["transfer"] = minmax_normalize(frame[transfer_col], config.epsilon)

    fixed = {
        "flow": config.flow_weight,
        "period": config.period_weight,
        "duration": config.duration_weight,
        "transfer": config.transfer_weight,
    }
    if all(value is not None for value in fixed.values()):
        weights = {name: float(value) for name, value in fixed.items()}
    else:
        weights = entropy_weights(
            normalized,
            ["flow", "period", "duration", "transfer"],
            config.epsilon,
        )

    return (
        weights["flow"] * normalized["flow"]
        + weights["period"] * normalized["period"]
        + weights["duration"] * normalized["duration"]
        + weights["transfer"] * (1.0 - normalized["transfer"])
    )


def spatial_matrix_from_edges(
    edge_frame: pd.DataFrame,
    station_to_index: Mapping[str, int],
    value_col: str,
    origin_col: str = "origin",
    destination_col: str = "destination",
) -> np.ndarray:
    """Create a square spatial matrix from weighted station-pair records."""

    size = len(station_to_index)
    matrix = np.zeros((size, size), dtype=np.float32)
    for row in edge_frame.itertuples(index=False):
        origin = getattr(row, origin_col)
        destination = getattr(row, destination_col)
        if origin not in station_to_index or destination not in station_to_index:
            continue
        matrix[station_to_index[origin], station_to_index[destination]] = float(
            getattr(row, value_col)
        )
    return matrix

