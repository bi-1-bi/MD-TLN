from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from md_tln import DataConfig, MDTLN, ModelConfig, TrainingConfig
from md_tln.data import PassengerFlowDataset, make_supervised_sequences
from md_tln.train import fit


def main() -> None:
    data_config = DataConfig(input_length=6, forecast_horizon=1, grid_height=8, grid_width=8)
    model_config = ModelConfig(
        input_length=data_config.input_length,
        forecast_horizon=data_config.forecast_horizon,
        history_channels=2,
        spatial_channels=1,
        external_channels=1,
        output_channels=2,
        grid_height=8,
        grid_width=8,
        encoding_dim=64,
        patch_sizes=((2, 2), (2, 2), (4, 4), (4, 4)),
        sce_reduction=8,
        dropout=0.1,
    )
    values = torch.rand(40, 2, 8, 8).numpy()
    spatial = torch.ones(40, 1, 8, 8).numpy()
    external = torch.zeros(40, 1, 8, 8).numpy()

    x, y = make_supervised_sequences(values, data_config)
    sx, _ = make_supervised_sequences(spatial, data_config)
    ex, _ = make_supervised_sequences(external, data_config)
    loader = DataLoader(PassengerFlowDataset(x, y, sx, ex), batch_size=4, shuffle=True)

    model = MDTLN(model_config)
    result = fit(
        model,
        loader,
        config=TrainingConfig(epochs=1, batch_size=4, checkpoint_path=None),
        device="cpu",
    )
    print(result)


if __name__ == "__main__":
    main()

