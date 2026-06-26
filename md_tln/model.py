from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .patching import patchify_2d, unpatchify_2d
from .positional_encoding import SpatioTemporalPositionalEncoding


@dataclass
class MDTLNOutput:
    prediction: torch.Tensor
    time_aux: torch.Tensor | None = None
    day_type_aux: torch.Tensor | None = None


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        residual: bool = True,
        batch_norm: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return self.activation(self.norm(self.conv(x)) + residual)


class SqueezeChannelExcitation(nn.Module):
    """Squeeze-and-Channel Excitation attention from the Method section.

    For an input feature map X, the module computes
    X * sigmoid(W2(ReLU(W1(GAP(X))))), where GAP is global average pooling
    over the spatial dimensions and W1/W2 are the two fully connected layers
    with reduction ratio r.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.excitation(weights).view(batch, channels, 1, 1)
        return x * weights


SCEAttention = SqueezeChannelExcitation


class FeatureEncoder(nn.Module):
    """Stacked 2D convolutional encoder for one input branch."""

    def __init__(self, input_channels: int, encoding_dim: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(input_channels, encoding_dim // 4, residual=False)
        self.conv2 = ConvBlock(encoding_dim // 4, encoding_dim // 4)
        self.conv3 = ConvBlock(encoding_dim // 4, encoding_dim // 2)
        self.conv4 = ConvBlock(encoding_dim // 2, encoding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c4, c3, c2, c1


class SpatioConditionalFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = ConvBlock(
            channels * 2,
            channels,
            kernel_size=3,
            residual=False,
            activation="leaky_relu",
        )
        self.spatial_project = ConvBlock(
            channels,
            channels,
            kernel_size=1,
            residual=False,
            activation="identity",
        )

    def forward(
        self,
        history: torch.Tensor,
        external: torch.Tensor,
        spatial: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, channels, height, width = history.shape
        hx = torch.cat([history, external], dim=2).reshape(
            batch * steps,
            channels * 2,
            height,
            width,
        )
        sx = spatial.reshape(batch * steps, channels, height, width)
        fused = self.fuse(hx) + self.spatial_project(sx)
        fused = fused.reshape(batch, steps, channels, height, width)
        return fused + position


class PatchTransformerScale(nn.Module):
    def __init__(
        self,
        channels: int,
        sub_channels: int,
        patch_size: tuple[int, int],
        *,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.sub_channels = sub_channels
        patch_dim = sub_channels * patch_size[0] * patch_size[1]
        self.query = nn.Conv2d(channels, sub_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, sub_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, sub_channels, kernel_size=1)
        self.attention = nn.MultiheadAttention(
            patch_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def _project(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = x.shape
        x = x.reshape(batch * steps, x.shape[2], height, width)
        x = layer(x)
        return x.reshape(batch, steps, self.sub_channels, height, width)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        q = self._project(self.query, query)
        k = self._project(self.key, context)
        v = self._project(self.value, context)

        q_patch = patchify_2d(q, self.patch_size)
        k_patch = patchify_2d(k, self.patch_size)
        v_patch = patchify_2d(v, self.patch_size)

        attended, _ = self.attention(q_patch, k_patch, v_patch)
        output_shape = (
            context.shape[0],
            context.shape[1],
            self.sub_channels,
            context.shape[3],
            context.shape[4],
        )
        return unpatchify_2d(attended, output_shape, self.patch_size)


class MultiScalePatchTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        sub_channels = config.encoding_dim // len(config.patch_sizes)
        self.scales = nn.ModuleList(
            [
                PatchTransformerScale(
                    config.encoding_dim,
                    sub_channels,
                    patch_size,
                    heads=config.attention_heads,
                    dropout=config.dropout,
                )
                for patch_size in config.patch_sizes
            ]
        )
        self.ffn = ConvBlock(config.encoding_dim, config.encoding_dim)
        self.dropout = nn.Dropout(config.dropout)
        if config.norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(config.encoding_dim)
            self.norm2 = nn.BatchNorm2d(config.encoding_dim)
        else:
            self.norm1 = nn.LayerNorm(
                [config.encoding_dim, config.grid_height, config.grid_width]
            )
            self.norm2 = nn.LayerNorm(
                [config.encoding_dim, config.grid_height, config.grid_width]
            )

    def _apply_framewise(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        x = layer(x)
        return x.reshape(batch, steps, channels, height, width)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat(
            [scale(context, query) for scale in self.scales],
            dim=2,
        )
        attention_out = query + self.dropout(multi_scale)
        attention_out = self._apply_framewise(self.norm1, attention_out)

        ffn_out = self._apply_framewise(self.ffn, attention_out)
        output = attention_out + self.dropout(ffn_out)
        return self._apply_framewise(self.norm2, output)


class Decoder(nn.Module):
    def __init__(self, output_channels: int, encoding_dim: int, use_skip: bool) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.conv5 = ConvBlock(encoding_dim, encoding_dim // 2)
        self.skip5 = ConvBlock(encoding_dim, encoding_dim // 2, kernel_size=1, residual=False)
        self.conv6 = ConvBlock(encoding_dim // 2, encoding_dim // 4)
        self.skip6 = ConvBlock(encoding_dim // 2, encoding_dim // 4, kernel_size=1, residual=False)
        self.conv7 = ConvBlock(encoding_dim // 4, encoding_dim // 4)
        self.skip7 = ConvBlock(encoding_dim // 2, encoding_dim // 4, kernel_size=1, residual=False)
        self.conv8 = ConvBlock(
            encoding_dim // 4,
            output_channels,
            activation="identity",
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        c4, c3, c2, c1 = features
        x = self.conv5(c4)
        if self.use_skip:
            x = self.skip5(torch.cat([x, c3], dim=1))

        x = self.conv6(x)
        if self.use_skip:
            x = self.skip6(torch.cat([x, c2], dim=1))

        x = self.conv7(x)
        if self.use_skip:
            x = self.skip7(torch.cat([x, c1], dim=1))

        return self.conv8(x)


class MDTLN(nn.Module):
    """Multi-Dimensional Transformer-LSTM Network implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.history_encoder = FeatureEncoder(config.history_channels, config.encoding_dim)
        self.spatial_encoder = FeatureEncoder(config.spatial_channels, config.encoding_dim)
        self.external_encoder = FeatureEncoder(
            max(1, config.external_channels),
            config.encoding_dim,
        )

        self.position = SpatioTemporalPositionalEncoding(config.encoding_dim)
        self.fusion = SpatioConditionalFusion(config.encoding_dim)
        self.sce_attention = SqueezeChannelExcitation(
            config.encoding_dim,
            config.sce_reduction,
        )
        self.self_attention = nn.ModuleList(
            [MultiScalePatchTransformer(config) for _ in range(config.attention_layers)]
        )
        self.cross_attention = nn.ModuleList(
            [MultiScalePatchTransformer(config) for _ in range(config.cross_attention_layers)]
        )
        self.decoder = Decoder(config.output_channels, config.encoding_dim, config.use_skip)

        flat_dim = config.input_length * config.encoding_dim * config.grid_height * config.grid_width
        self.time_aux = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 24),
        )
        self.day_type_aux = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 4),
        )

    def _zeros_like_branch(self, history: torch.Tensor, channels: int) -> torch.Tensor:
        return history.new_zeros(
            history.shape[0],
            history.shape[1],
            channels,
            history.shape[3],
            history.shape[4],
        )

    def _encode_branch(
        self,
        encoder: FeatureEncoder,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        encoded = encoder(x)
        return tuple(
            item.reshape(batch, steps, item.shape[1], height, width) for item in encoded
        )

    def _flatten_frames(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        return x.reshape(batch * steps, channels, height, width)

    def _apply_framewise(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        x = layer(x)
        return x.reshape(batch, steps, channels, height, width)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.output_activation == "relu":
            return torch.relu(x)
        if self.config.output_activation == "tanh":
            return torch.tanh(x)
        return x

    def forward(
        self,
        history: torch.Tensor,
        spatial: torch.Tensor | None = None,
        external: torch.Tensor | None = None,
    ) -> MDTLNOutput:
        if history.ndim != 5:
            raise ValueError("history must have shape [B, T, C, H, W].")

        batch, steps, _, height, width = history.shape
        if steps != self.config.input_length:
            raise ValueError(f"Expected {self.config.input_length} steps, got {steps}.")
        if (height, width) != (self.config.grid_height, self.config.grid_width):
            raise ValueError(
                f"Expected grid {(self.config.grid_height, self.config.grid_width)}, "
                f"got {(height, width)}."
            )

        if spatial is None:
            spatial = self._zeros_like_branch(history, self.config.spatial_channels)
        if external is None:
            external = self._zeros_like_branch(history, max(1, self.config.external_channels))

        hist_c4, hist_c3, hist_c2, hist_c1 = self._encode_branch(
            self.history_encoder,
            history,
        )
        spatial_c4, _, _, _ = self._encode_branch(self.spatial_encoder, spatial)
        external_c4, _, _, _ = self._encode_branch(self.external_encoder, external)

        position = self.position(
            batch,
            steps,
            height,
            width,
            device=history.device,
            dtype=history.dtype,
        )
        fused = self.fusion(hist_c4, external_c4, spatial_c4, position)
        fused = self._apply_framewise(self.sce_attention, fused)

        hidden = fused
        for layer in self.self_attention:
            hidden = layer(hidden, hidden)
        for layer in self.cross_attention:
            hidden = layer(fused, hidden)

        flat_hidden = hidden.reshape(batch, -1)
        time_aux = self.time_aux(flat_hidden)
        day_type_aux = self.day_type_aux(flat_hidden)

        decoded = self.decoder(
            (
                self._flatten_frames(hidden),
                self._flatten_frames(hist_c3),
                self._flatten_frames(hist_c2),
                self._flatten_frames(hist_c1),
            )
        )
        decoded = decoded.reshape(
            batch,
            steps,
            self.config.output_channels,
            height,
            width,
        )
        decoded = self._apply_output_activation(decoded)
        prediction = decoded[:, -self.config.forecast_horizon :]
        return MDTLNOutput(prediction, time_aux, day_type_aux)
