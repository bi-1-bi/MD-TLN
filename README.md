# MD-TLN

<p align="center">
  <strong>A clean PyTorch implementation of MD-TLN for metro passenger flow prediction.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">
  <img src="docs/assets/md-tln-framework.png" alt="MD-TLN framework" width="100%">
</p>

This repository contains a cleaned implementation of the MD-TLN method for
metro passenger flow prediction. It is organized from the original project code
and aligned with the Method section of the manuscript.

The repository intentionally excludes experiment-only analysis code, including
station functional labeling, POI clustering, spatiotemporal correlation figures,
heat-map plotting, residual plotting, and result-comparison scripts.

## Highlights

- Multi-source input handling for historical flow, spatial topology, and
  external disturbance features.
- Weather, holiday, and large-scale event feature encoding.
- Entropy-weighted spatial relationship construction.
- Parallel convolutional encoders with spatio-conditional feature fusion.
- Squeeze-and-Channel Excitation attention.
- Multi-scale patch-wise Transformer.
- Decoder with optional auxiliary supervision.
- Training loop with Adam, MSE-based objective, validation monitoring, and
  early stopping.

## Architecture

### Multi-Scale Patch-Wise Transformer

<p align="center">
  <img src="docs/assets/multi-scale-patch-transformer.png" alt="Multi-scale patch-wise Transformer" width="100%">
</p>

## Paper-Aligned Defaults

- Historical input length: 6 steps.
- Time interval: 5 minutes.
- Forecast horizon: 1 step.
- Optimizer: Adam.
- Initial learning rate: 1e-4.
- Batch size: 32.
- Training epochs: 100.
- Primary loss: MSE.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```text
md_tln_open_source/
  docs/
    assets/
      md-tln-framework.png
      multi-scale-patch-transformer.png
  md_tln/
    config.py              # Shared dataclass configuration
    data.py                # Sliding windows and tensor datasets
    losses.py              # Primary and auxiliary losses
    model.py               # MD-TLN architecture
    patching.py            # Patchify and unpatchify utilities
    positional_encoding.py # Sinusoidal temporal/spatial encoding
    spatial_weights.py     # Entropy-weighted spatial input encoding
    train.py               # Training and validation loops
    weather_events.py      # Weather, holiday, and large-scale event features
  scripts/
    smoke_test.py          # Minimal import/forward/backward check
    run_txt_experiment.py  # Small/full runs on cleaned TXT AFC data
  requirements.txt
  NOTICE
  LICENSE
```

## Input Shapes

The model expects tensors in the following format:

```text
history:  [batch, input_length, history_channels, height, width]
spatial:  [batch, input_length, spatial_channels, height, width] or None
external: [batch, input_length, external_channels, height, width] or None
target:   [batch, forecast_horizon, output_channels, height, width]
```

If `spatial` or `external` is omitted, the model substitutes zero context for
that branch. This is useful for ablation studies while keeping the same model
interface.

## Quick Start

```python
from md_tln import MDTLN, ModelConfig

config = ModelConfig(
    input_length=6,
    forecast_horizon=1,
    history_channels=2,
    spatial_channels=1,
    external_channels=8,
    output_channels=2,
    grid_height=32,
    grid_width=32,
)

model = MDTLN(config)
```

Run a basic smoke test:

```bash
python scripts/smoke_test.py
```

Run cleaned TXT data:

```bash
python scripts/run_txt_experiment.py --data-files path/to/cleaned.txt --sample-rows 100000
```

## Experiment Notes

Lightweight local verification results are recorded in
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md). These runs are sanity/full-data
execution checks with CPU-friendly settings, not a full 100-epoch GPU
reproduction of the manuscript experiments.

## Notes For Release

- Keep data files, trained weights, generated figures, and virtual environments
  out of the repository.
- Add a dataset README if public data or processed tensors are released later.
- The implementation keeps the model code separated from experiment analysis so
  reviewers can inspect the proposed method directly.
