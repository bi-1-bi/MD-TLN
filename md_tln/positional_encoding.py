from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_encoding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create standard sinusoidal positional encodings."""

    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


class SpatioTemporalPositionalEncoding(nn.Module):
    """Fixed temporal, height, and width sinusoidal encoding."""

    def __init__(self, channels: int, temporal_channels: int = 32) -> None:
        super().__init__()
        if channels <= temporal_channels:
            raise ValueError("channels must be larger than temporal_channels.")
        spatial_channels = channels - temporal_channels
        if spatial_channels % 2 != 0:
            raise ValueError("channels - temporal_channels must be even.")
        self.channels = channels
        self.temporal_channels = temporal_channels
        self.height_channels = spatial_channels // 2
        self.width_channels = spatial_channels // 2

    def forward(
        self,
        batch: int,
        steps: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        temporal = sinusoidal_encoding(
            steps,
            self.temporal_channels,
            device=device,
            dtype=dtype,
        )
        height_pe = sinusoidal_encoding(
            height,
            self.height_channels,
            device=device,
            dtype=dtype,
        )
        width_pe = sinusoidal_encoding(
            width,
            self.width_channels,
            device=device,
            dtype=dtype,
        )

        temporal = temporal.view(1, steps, self.temporal_channels, 1, 1)
        temporal = temporal.expand(batch, steps, -1, height, width)

        height_pe = height_pe.transpose(0, 1).view(1, 1, self.height_channels, height, 1)
        height_pe = height_pe.expand(batch, steps, -1, height, width)

        width_pe = width_pe.transpose(0, 1).view(1, 1, self.width_channels, 1, width)
        width_pe = width_pe.expand(batch, steps, -1, height, width)

        return torch.cat([temporal, height_pe, width_pe], dim=2)

