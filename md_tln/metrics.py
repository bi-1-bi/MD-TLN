from __future__ import annotations

import math

import torch


@torch.no_grad()
def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().float()
    target = target.detach().float()
    error = prediction - target
    mse = torch.mean(error.square()).item()
    mae = torch.mean(error.abs()).item()
    rmse = math.sqrt(mse)
    denom = torch.sum((target - torch.mean(target)).square()).item()
    if denom <= 1e-12:
        r2 = float("nan")
    else:
        r2 = 1.0 - torch.sum(error.square()).item() / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

