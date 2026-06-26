from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import MDTLNOutput


@dataclass(frozen=True)
class LossConfig:
    """Loss parameters for primary and auxiliary supervision."""

    primary_weight: float = 0.7
    use_time_auxiliary: bool = True
    use_day_type_auxiliary: bool = True


class MDTLNLoss(nn.Module):
    def __init__(self, config: LossConfig = LossConfig()) -> None:
        super().__init__()
        if not 0.0 <= config.primary_weight <= 1.0:
            raise ValueError("primary_weight must be in [0, 1].")
        self.config = config
        self.primary = nn.MSELoss()
        self.auxiliary = nn.CrossEntropyLoss()

    def forward(
        self,
        output: MDTLNOutput,
        target: torch.Tensor,
        *,
        time_label: torch.Tensor | None = None,
        day_type_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        primary_loss = self.primary(output.prediction, target)
        auxiliary_losses: list[torch.Tensor] = []

        if (
            self.config.use_time_auxiliary
            and output.time_aux is not None
            and time_label is not None
        ):
            auxiliary_losses.append(self.auxiliary(output.time_aux, time_label))

        if (
            self.config.use_day_type_auxiliary
            and output.day_type_aux is not None
            and day_type_label is not None
        ):
            auxiliary_losses.append(self.auxiliary(output.day_type_aux, day_type_label))

        if not auxiliary_losses:
            return primary_loss

        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return (
            self.config.primary_weight * primary_loss
            + (1.0 - self.config.primary_weight) * auxiliary_loss
        )

