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
  <img src="fig/md-tln-framework.png" alt="MD-TLN framework" width="100%">
</p>

This repository contains an implementation of the MD-TLN method for
metro passenger flow prediction. It is organized from the original project code
and aligned with the Method section of the manuscript.

## Highlights

- Multi-source input handling for historical flow, spatial topology, and
  external disturbance features.
- Weather, holiday, and large-scale event feature encoding.
- Entropy-weighted spatial relationship construction.
- Parallel convolutional encoders with spatio-conditional feature fusion.
- Squeeze-and-Channel Excitation attention.
- Multi-scale patch-wise Transformer.
- Decoder with optional auxiliary supervision.
- Training and validation utilities for model development.

## Get Started

### 1. Clone The Repository

```bash
git clone https://github.com/Lilin-Chen/MD-TLN.git
cd MD-TLN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python scripts/verify_installation.py
```

If the script finishes without errors, the environment and model pipeline are
ready.

### 4. Run With Data

```bash
python scripts/run_txt_experiment.py --data-files path/to/data.txt
```

The script reads prepared AFC data files, builds station-grid tensors, trains
the model, and writes outputs under the `runs/` directory.

## Data Preparation

This repository does not include raw AFC data or processed tensors. Users should
prepare data files locally and pass them to the experiment script through
`--data-files`.

For multiple files:

```bash
python scripts/run_txt_experiment.py --data-files file_1.txt file_2.txt file_3.txt
```

Generated outputs are ignored by Git and should not be committed to the
repository.

## Outputs

Running the experiment script creates a result file under `runs/`. The file
records metadata, training history, and evaluation metrics for the current run.

The `runs/` directory is excluded from version control to keep the repository
lightweight.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
