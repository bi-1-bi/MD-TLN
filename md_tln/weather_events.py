from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalFeatureConfig:
    """Weather and calendar feature parameters."""

    weather_categories: tuple[str, ...] = (
        "clear",
        "cloudy",
        "light_rain",
        "moderate_rain",
        "heavy_rain",
    )
    holiday_dates: tuple[str, ...] = (
        "2024-10-01",
        "2024-10-02",
        "2024-10-03",
        "2024-10-04",
        "2024-10-05",
        "2024-10-06",
        "2024-10-07",
    )


@dataclass(frozen=True)
class EventDecayConfig:
    """Large-scale event parameters from the Method section."""

    distance_decay: float = 0.1
    distance_unit: str = "km"


@dataclass(frozen=True)
class LargeScaleEvent:
    """A single event with time interval, intensity, and location."""

    start_time: datetime | str
    end_time: datetime | str
    intensity: float
    location: tuple[float, float]
    name: str = "event"

    def active_mask(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        start = pd.Timestamp(self.start_time)
        end = pd.Timestamp(self.end_time)
        mask = (timestamps >= start) & (timestamps <= end)
        return np.asarray(mask, dtype=np.float32)


def encode_weather(
    weather: Sequence[str],
    categories: Sequence[str],
) -> pd.DataFrame:
    """One-hot encode weather categories."""

    category_index = {name: idx for idx, name in enumerate(categories)}
    matrix = np.zeros((len(weather), len(categories)), dtype=np.float32)
    for row, label in enumerate(weather):
        if label not in category_index:
            continue
        matrix[row, category_index[label]] = 1.0
    return pd.DataFrame(matrix, columns=[f"weather_{name}" for name in categories])


def encode_holiday(
    timestamps: Iterable[pd.Timestamp | datetime | str],
    holiday_dates: Iterable[str],
) -> np.ndarray:
    """Return a binary holiday indicator aligned to timestamps."""

    date_set = {pd.Timestamp(day).date() for day in holiday_dates}
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    return np.array([1.0 if ts.date() in date_set else 0.0 for ts in index], dtype=np.float32)


def haversine_km(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Compute great-circle distance between lon-lat coordinates."""

    lon1, lat1 = point_a
    lon2, lat2 = point_b
    radius_km = 6371.0088
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


def compute_event_intensity(
    timestamps: Iterable[pd.Timestamp | datetime | str],
    station_locations: Mapping[str, tuple[float, float]],
    events: Sequence[LargeScaleEvent],
    config: EventDecayConfig = EventDecayConfig(),
) -> pd.DataFrame:
    """Compute station-level event intensity with temporal activation and decay."""

    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    station_ids = list(station_locations.keys())
    intensity = np.zeros((len(index), len(station_ids)), dtype=np.float32)

    for event in events:
        active = event.active_mask(index)[:, None]
        distances = np.array(
            [
                haversine_km(station_locations[station_id], event.location)
                for station_id in station_ids
            ],
            dtype=np.float32,
        )
        decay = np.exp(-config.distance_decay * distances)[None, :]
        intensity += active * float(event.intensity) * decay

    return pd.DataFrame(
        intensity,
        index=index,
        columns=[f"event_intensity_{station_id}" for station_id in station_ids],
    )


def build_external_feature_frame(
    timestamps: Iterable[pd.Timestamp | datetime | str],
    weather: Sequence[str],
    station_locations: Mapping[str, tuple[float, float]],
    events: Sequence[LargeScaleEvent] | None = None,
    feature_config: ExternalFeatureConfig = ExternalFeatureConfig(),
    event_config: EventDecayConfig = EventDecayConfig(),
) -> pd.DataFrame:
    """Build a feature frame for weather, holiday, and large-scale events."""

    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if len(weather) != len(index):
        raise ValueError("weather and timestamps must have the same length.")

    weather_frame = encode_weather(weather, feature_config.weather_categories)
    weather_frame.index = index

    holiday = encode_holiday(index, feature_config.holiday_dates)
    holiday_frame = pd.DataFrame({"holiday": holiday}, index=index)

    if events:
        event_frame = compute_event_intensity(
            index,
            station_locations,
            events,
            event_config,
        )
    else:
        event_frame = pd.DataFrame(index=index)

    return pd.concat([weather_frame, holiday_frame, event_frame], axis=1)
