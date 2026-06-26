# Experiment Results

These runs verify the cleaned MD-TLN implementation on cleaned TXT AFC data.
They were executed locally with PyTorch CPU, so the settings are intentionally
lightweight: one training epoch, `encoding_dim=64`, and a 32 x 32 station grid.
They are sanity/full-data execution results, not a 100-epoch GPU reproduction of
the manuscript experiments.

## SCE-Attention Check

The model now applies Squeeze-and-Channel Excitation exactly as described in the
Method section:

1. Global average pooling over spatial dimensions.
2. Fully connected reduction by ratio r.
3. ReLU activation.
4. Fully connected restoration to C channels.
5. Sigmoid channel weights.
6. Channel-wise reweighting of the fused feature map.

The SCE module is applied after spatio-conditional feature fusion and before the
multi-scale patch-wise Transformer.

## Small TXT Run

- Data: first 200,000 rows from one cleaned TXT file.
- Raw rows read: 200,000.
- Time range parsed: 2024-10-01 05:30:00 to 2024-10-02 00:55:00.
- Time steps: 234.
- Stations used: 263.
- Train samples: 182.
- Validation samples: 46.
- Epochs: 1.
- Batch size: 8.
- Train loss: 0.2318367712.
- Validation loss: 0.0123653497.
- Original-scale MSE: 525.1746826172.
- Original-scale MAE: 13.2986154556.
- Original-scale RMSE: 22.9166900450.
- Original-scale R2: -83.3082360259.
- Elapsed time: 16.134 seconds.

## Full TXT Run

- Data: five cleaned TXT files covering October 2024.
- Raw rows read: 144,727,156.
- Time range parsed: 2024-10-01 00:00:00 to 2024-10-31 23:55:00.
- Time steps: 8,928.
- Stations used: 263.
- Total in/out flow count: 289,118,208.
- Train samples: 7,137.
- Validation samples: 1,785.
- Epochs: 1.
- Batch size: 16.
- Train loss: 0.0368152852.
- Validation loss: 0.0003446890.
- Original-scale MSE: 4,328.0483398438.
- Original-scale MAE: 16.3771038055.
- Original-scale RMSE: 65.7879042062.
- Original-scale R2: -0.1274477869.
- Elapsed time: 1,086.506 seconds.

The full run reads and aggregates the TXT data in chunks, filters timestamps to
2024-10-01 through 2024-10-31, builds five-minute station-grid tensors, and then
trains/evaluates the MD-TLN model once.

