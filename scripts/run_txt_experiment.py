from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from md_tln import DataConfig, MDTLN, ModelConfig, TrainingConfig
from md_tln.data import PassengerFlowDataset, make_supervised_sequences
from md_tln.metrics import regression_metrics
from md_tln.train import fit


TIME_COLUMNS = ("进闸时间", "出闸时间")
STATION_COLUMNS = ("进站车站", "出站车站")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MD-TLN on cleaned TXT AFC data.")
    parser.add_argument("--data-files", nargs="+", required=True, help="Cleaned TXT files.")
    parser.add_argument("--output-dir", default="runs/txt_experiment")
    parser.add_argument("--sample-rows", type=int, default=None, help="Read only N rows for a small run.")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoding-dim", type=int, default=64)
    parser.add_argument("--grid-height", type=int, default=32)
    parser.add_argument("--grid-width", type=int, default=32)
    parser.add_argument("--max-stations", type=int, default=263)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-name", default=None)
    parser.add_argument("--start-time", default="2024-10-01 00:00:00")
    parser.add_argument("--end-time", default="2024-10-31 23:59:59")
    return parser.parse_args()


def normalize_time_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(
        r"^(\d{4}/\d{2}/\d{2}) (\d{2})/(\d{2})/(\d{2})$",
        r"\1 \2:\3:\4",
        regex=True,
    )
    return text


def read_txt_chunks(path: Path, chunksize: int, sample_rows: int | None):
    remaining = sample_rows
    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        encoding="utf-8-sig",
        usecols=list(TIME_COLUMNS + STATION_COLUMNS),
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    ):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        yield chunk
        if remaining is not None and remaining <= 0:
            break


def aggregate_txt_files(
    paths: list[Path],
    chunksize: int,
    sample_rows: int | None,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> tuple[dict[tuple[pd.Timestamp, str], int], dict[tuple[pd.Timestamp, str], int], Counter, int]:
    inbound: dict[tuple[pd.Timestamp, str], int] = defaultdict(int)
    outbound: dict[tuple[pd.Timestamp, str], int] = defaultdict(int)
    station_counter: Counter = Counter()
    total_rows = 0

    for path in paths:
        rows_for_file = sample_rows
        for chunk in read_txt_chunks(path, chunksize, rows_for_file):
            total_rows += len(chunk)
            in_time = pd.to_datetime(normalize_time_text(chunk["进闸时间"]), errors="coerce")
            out_time = pd.to_datetime(normalize_time_text(chunk["出闸时间"]), errors="coerce")
            in_station = chunk["进站车站"].astype("string").str.strip()
            out_station = chunk["出站车站"].astype("string").str.strip()

            in_frame = pd.DataFrame({"time": in_time.dt.floor("5min"), "station": in_station}).dropna()
            out_frame = pd.DataFrame({"time": out_time.dt.floor("5min"), "station": out_station}).dropna()
            in_frame = in_frame[
                (in_frame["time"] >= start_time) & (in_frame["time"] <= end_time)
            ]
            out_frame = out_frame[
                (out_frame["time"] >= start_time) & (out_frame["time"] <= end_time)
            ]

            station_counter.update(in_frame["station"].tolist())
            station_counter.update(out_frame["station"].tolist())

            for (ts, station), count in in_frame.groupby(["time", "station"]).size().items():
                inbound[(ts, station)] += int(count)
            for (ts, station), count in out_frame.groupby(["time", "station"]).size().items():
                outbound[(ts, station)] += int(count)

            if sample_rows is not None:
                rows_for_file = max(0, rows_for_file - len(chunk))
                if rows_for_file <= 0:
                    break

    return inbound, outbound, station_counter, total_rows


def build_tensors(
    inbound: dict[tuple[pd.Timestamp, str], int],
    outbound: dict[tuple[pd.Timestamp, str], int],
    station_counter: Counter,
    *,
    height: int,
    width: int,
    max_stations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    stations = [station for station, _ in station_counter.most_common(max_stations)]
    station_to_cell = {station: idx for idx, station in enumerate(stations)}

    all_times = [key[0] for key in inbound.keys()] + [key[0] for key in outbound.keys()]
    if not all_times:
        raise ValueError("No valid timestamps were parsed from the TXT data.")
    timeline = pd.date_range(min(all_times), max(all_times), freq="5min")
    time_to_idx = {ts: idx for idx, ts in enumerate(timeline)}

    values = np.zeros((len(timeline), 2, height, width), dtype=np.float32)
    station_mask = np.zeros((len(timeline), 1, height, width), dtype=np.float32)
    external = np.zeros((len(timeline), 1, height, width), dtype=np.float32)

    for station, cell in station_to_cell.items():
        row, col = divmod(cell, width)
        if row >= height:
            continue
        station_mask[:, 0, row, col] = 1.0

    for (ts, station), count in inbound.items():
        cell = station_to_cell.get(station)
        if cell is None:
            continue
        row, col = divmod(cell, width)
        if row < height:
            values[time_to_idx[ts], 0, row, col] += count

    for (ts, station), count in outbound.items():
        cell = station_to_cell.get(station)
        if cell is None:
            continue
        row, col = divmod(cell, width)
        if row < height:
            values[time_to_idx[ts], 1, row, col] += count

    holiday_dates = {pd.Timestamp(f"2024-10-{day:02d}").date() for day in range(1, 8)}
    for idx, ts in enumerate(timeline):
        if ts.date() in holiday_dates:
            external[idx, 0] = station_mask[idx, 0]

    metadata = {
        "timesteps": len(timeline),
        "start_time": str(timeline[0]),
        "end_time": str(timeline[-1]),
        "stations_used": len(station_to_cell),
        "total_flow": int(values.sum()),
    }
    return values, station_mask, external, metadata


def normalize_by_train(values: np.ndarray, split_idx: int) -> tuple[np.ndarray, float, float]:
    train_values = values[:split_idx]
    minimum = float(train_values.min())
    maximum = float(train_values.max())
    scaled = (values - minimum) / (maximum - minimum + 1e-8)
    return scaled.astype(np.float32), minimum, maximum


@torch.no_grad()
def evaluate_original_scale(
    model: MDTLN,
    loader: DataLoader,
    y_min: float,
    y_max: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        history = batch["history"].to(device)
        spatial = batch["spatial"].to(device)
        external = batch["external"].to(device)
        target = batch["target"].to(device)
        output = model(history, spatial, external).prediction
        predictions.append(output.cpu() * (y_max - y_min + 1e-8) + y_min)
        targets.append(target.cpu() * (y_max - y_min + 1e-8) + y_min)
    return regression_metrics(torch.cat(predictions), torch.cat(targets))


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = [Path(item) for item in args.data_files]
    inbound, outbound, station_counter, total_rows = aggregate_txt_files(
        data_files,
        args.chunksize,
        args.sample_rows,
        pd.Timestamp(args.start_time),
        pd.Timestamp(args.end_time),
    )
    values, spatial, external, metadata = build_tensors(
        inbound,
        outbound,
        station_counter,
        height=args.grid_height,
        width=args.grid_width,
        max_stations=args.max_stations,
    )

    split_time = max(1, int(values.shape[0] * 0.8))
    values_scaled, y_min, y_max = normalize_by_train(values, split_time)

    data_config = DataConfig(input_length=6, forecast_horizon=1, grid_height=args.grid_height, grid_width=args.grid_width)
    x, y = make_supervised_sequences(values_scaled, data_config)
    sx, _ = make_supervised_sequences(spatial, data_config)
    ex, _ = make_supervised_sequences(external, data_config)

    split_samples = max(1, int(x.shape[0] * 0.8))
    train_ds = PassengerFlowDataset(x[:split_samples], y[:split_samples], sx[:split_samples], ex[:split_samples])
    val_ds = PassengerFlowDataset(x[split_samples:], y[split_samples:], sx[split_samples:], ex[split_samples:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model_config = ModelConfig(
        input_length=6,
        forecast_horizon=1,
        history_channels=2,
        spatial_channels=1,
        external_channels=1,
        output_channels=2,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        encoding_dim=args.encoding_dim,
        patch_sizes=((4, 4), (4, 4), (8, 8), (8, 8)),
        attention_layers=1,
        cross_attention_layers=1,
        sce_reduction=8,
        dropout=0.1,
    )
    model = MDTLN(model_config)
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_path=None,
    )
    device = torch.device(args.device)
    history = fit(
        model,
        train_loader,
        val_loader,
        config=training_config,
        device=device,
    )
    metrics = evaluate_original_scale(model, val_loader, y_min, y_max, device)

    result = {
        "data_files": [str(path) for path in data_files],
        "sample_rows_per_file": args.sample_rows,
        "raw_rows_read": total_rows,
        "metadata": metadata,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoding_dim": args.encoding_dim,
        "grid_height": args.grid_height,
        "grid_width": args.grid_width,
        "start_time_filter": args.start_time,
        "end_time_filter": args.end_time,
        "history": history,
        "metrics_original_scale": metrics,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
