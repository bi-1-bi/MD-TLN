from __future__ import annotations

import torch


def patchify_2d(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Split a [B, T, C, H, W] tensor into non-overlapping patch vectors."""

    if x.ndim != 5:
        raise ValueError(f"Expected [B, T, C, H, W], got shape {tuple(x.shape)}.")

    batch, steps, channels, height, width = x.shape
    patch_h, patch_w = patch_size

    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Patch size {patch_size} must divide spatial size {(height, width)}."
        )

    grid_h = height // patch_h
    grid_w = width // patch_w

    x = x.reshape(batch * steps, channels, height, width)
    x = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.reshape(batch, steps, grid_h, grid_w, channels * patch_h * patch_w)
    return x.reshape(batch, steps * grid_h * grid_w, -1)


def unpatchify_2d(
    patches: torch.Tensor,
    output_shape: tuple[int, int, int, int, int],
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """Restore patch vectors to a [B, T, C, H, W] tensor."""

    batch, steps, channels, height, width = output_shape
    patch_h, patch_w = patch_size
    grid_h = height // patch_h
    grid_w = width // patch_w
    expected_tokens = steps * grid_h * grid_w
    expected_dim = channels * patch_h * patch_w

    if patches.shape != (batch, expected_tokens, expected_dim):
        raise ValueError(
            "Patch tensor shape mismatch: expected "
            f"{(batch, expected_tokens, expected_dim)}, got {tuple(patches.shape)}."
        )

    x = patches.reshape(batch, steps, grid_h, grid_w, channels, patch_h, patch_w)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(batch, steps, channels, height, width)

