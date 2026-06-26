from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .losses import MDTLNLoss
from .model import MDTLN


def _to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if value is None else value.to(device)


def _unpack_batch(batch: Any, device: torch.device) -> dict[str, torch.Tensor | None]:
    if isinstance(batch, dict):
        return {
            "history": _to_device(batch["history"], device),
            "target": _to_device(batch["target"], device),
            "spatial": _to_device(batch.get("spatial"), device),
            "external": _to_device(batch.get("external"), device),
            "time_label": _to_device(batch.get("time_label"), device),
            "day_type_label": _to_device(batch.get("day_type_label"), device),
        }

    if len(batch) < 2:
        raise ValueError("Tuple batches must contain at least history and target.")

    return {
        "history": _to_device(batch[0], device),
        "target": _to_device(batch[1], device),
        "spatial": _to_device(batch[2], device) if len(batch) > 2 else None,
        "external": _to_device(batch[3], device) if len(batch) > 3 else None,
        "time_label": _to_device(batch[4], device) if len(batch) > 4 else None,
        "day_type_label": _to_device(batch[5], device) if len(batch) > 5 else None,
    }


def train_one_epoch(
    model: MDTLN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MDTLNLoss,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = None,
) -> float:
    model.train()
    total = 0.0
    for raw_batch in loader:
        batch = _unpack_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["history"], batch["spatial"], batch["external"])
        loss = criterion(
            output,
            batch["target"],
            time_label=batch["time_label"],
            day_type_label=batch["day_type_label"],
        )
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: MDTLN,
    loader: DataLoader,
    criterion: MDTLNLoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for raw_batch in loader:
        batch = _unpack_batch(raw_batch, device)
        output = model(batch["history"], batch["spatial"], batch["external"])
        loss = criterion(
            output,
            batch["target"],
            time_label=batch["time_label"],
            day_type_label=batch["day_type_label"],
        )
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def fit(
    model: MDTLN,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    config: TrainingConfig = TrainingConfig(),
    criterion: MDTLNLoss | None = None,
    device: str | torch.device | None = None,
) -> list[dict[str, float]]:
    """Train MD-TLN and optionally monitor validation loss."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    criterion = criterion or MDTLNLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        row = {"epoch": float(epoch), "train_loss": train_loss}

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                stale_epochs = 0
                if config.checkpoint_path:
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "training_config": asdict(config),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }
                    Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, config.checkpoint_path)
            else:
                stale_epochs += 1

            if stale_epochs >= config.early_stopping_patience:
                history.append(row)
                break

        history.append(row)

    return history

