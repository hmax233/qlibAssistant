#!/usr/bin/env python3
"""Render auditable E0--E6 epoch curves from immutable training histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = (
    "open1_close2",
    "close1_open2",
    "open1_open2",
    "close1_close2",
)
HORIZON_LABELS = {
    "open1_close2": "T+1 open → T+2 close",
    "close1_open2": "T+1 close → T+2 open",
    "open1_open2": "T+1 open → T+2 open",
    "close1_close2": "T+1 close → T+2 close",
}


def parse_metric_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("metric input must be EXPERIMENT=CSV")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("metric input must be EXPERIMENT=CSV")
    return name.strip(), Path(raw_path).expanduser()


def load_history(name: str, path: Path, expected_epochs: int) -> pd.DataFrame:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing epoch history for {name}: {path}")
    frame = pd.read_csv(path)
    required = {"epoch", "learning_rate", "epoch_seconds"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} epoch history missing columns: {sorted(missing)}")
    if len(frame) != expected_epochs:
        raise ValueError(
            f"{name} must contain exactly {expected_epochs} epochs, got {len(frame)}"
        )
    epoch = pd.to_numeric(frame["epoch"], errors="raise").astype(int)
    expected = np.arange(1, expected_epochs + 1)
    if not np.array_equal(epoch.to_numpy(), expected):
        raise ValueError(f"{name} epochs must be contiguous 1..{expected_epochs}")
    numeric_columns = [
        column
        for column in frame.columns
        if column == "learning_rate"
        or column == "epoch_seconds"
        or column in {"train_nll", "train_nll_scaled", "nll_scaled", "nll_scaled_3leg"}
        or column.endswith("_rank_ic")
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{name} epoch history contains non-finite required metrics")
    frame[numeric_columns] = numeric
    frame.insert(0, "experiment", name)
    return frame


def valid_nll_column(frame: pd.DataFrame) -> str:
    for candidate in ("nll_scaled", "nll_scaled_3leg"):
        if candidate in frame:
            return candidate
    raise ValueError(f"{frame.experiment.iloc[0]} has no validation NLL column")


def train_nll_column(frame: pd.DataFrame) -> str:
    for candidate in ("train_nll_scaled", "train_nll"):
        if candidate in frame:
            return candidate
    raise ValueError(f"{frame.experiment.iloc[0]} has no training NLL column")


def summarize(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name, frame in histories.items():
        valid_column = valid_nll_column(frame)
        train_column = train_nll_column(frame)
        best_nll_row = frame.loc[frame[valid_column].idxmin()]
        row: dict[str, float | int | str] = {
            "experiment": name,
            "epochs": len(frame),
            "total_seconds": float(frame["epoch_seconds"].sum()),
            "mean_epoch_seconds": float(frame["epoch_seconds"].mean()),
            "final_learning_rate": float(frame["learning_rate"].iloc[-1]),
            "final_train_nll": float(frame[train_column].iloc[-1]),
            "best_valid_nll": float(best_nll_row[valid_column]),
            "best_valid_nll_epoch": int(best_nll_row["epoch"]),
        }
        for horizon in HORIZONS:
            column = f"{horizon}_rank_ic"
            if column not in frame:
                row[f"{horizon}_best_rank_ic"] = np.nan
                row[f"{horizon}_best_rank_ic_epoch"] = np.nan
                continue
            best = frame.loc[frame[column].idxmax()]
            row[f"{horizon}_best_rank_ic"] = float(best[column])
            row[f"{horizon}_best_rank_ic_epoch"] = int(best["epoch"])
        rows.append(row)
    return pd.DataFrame(rows)


def render(histories: dict[str, pd.DataFrame], output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize(histories)
    summary.to_csv(output / "training_curve_summary.csv", index=False)

    figure, axes = plt.subplots(4, 2, figsize=(18, 18), constrained_layout=True)
    for name, frame in histories.items():
        epoch = frame["epoch"]
        axes[0, 0].plot(epoch, frame[train_nll_column(frame)], label=name)
        axes[0, 1].plot(epoch, frame[valid_nll_column(frame)], label=name)
        axes[1, 0].plot(epoch, frame["learning_rate"], label=name)
        axes[1, 1].plot(epoch, frame["epoch_seconds"], label=name)
        for axis, horizon in zip(axes[2:].flat, HORIZONS, strict=True):
            column = f"{horizon}_rank_ic"
            if column in frame:
                axis.plot(epoch, frame[column], label=name)

    axes[0, 0].set_title("Training NLL")
    axes[0, 1].set_title("Validation NLL")
    axes[1, 0].set_title("Learning-rate schedule")
    axes[1, 1].set_title("Epoch wall time")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("learning rate (log scale)")
    axes[1, 1].set_ylabel("seconds")
    for axis, horizon in zip(axes[2:].flat, HORIZONS, strict=True):
        axis.set_title(f"Validation Rank IC — {HORIZON_LABELS[horizon]}")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4)
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Alpha360 E0–E6 training curves",
        "",
        "All curves come from the persisted per-epoch CSV files. Validation metrics are descriptive; held-out Test is not used here.",
        "",
        "| Experiment | Epochs | Total min | Mean sec/epoch | Best valid NLL (epoch) | Final LR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.epochs} | {row.total_seconds / 60:.1f} | "
            f"{row.mean_epoch_seconds:.1f} | {row.best_valid_nll:.6f} "
            f"({int(row.best_valid_nll_epoch)}) | {row.final_learning_rate:.2e} |"
        )
    lines.extend(["", "![Training curves](training_curves.png)", ""])
    (output / "training_curves.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="append", type=parse_metric_spec, required=True)
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_epochs < 1:
        raise ValueError("expected epochs must be positive")
    names = [name for name, _ in args.metrics]
    if len(names) != len(set(names)):
        raise ValueError("duplicate experiment names")
    histories = {
        name: load_history(name, path, args.expected_epochs)
        for name, path in args.metrics
    }
    render(histories, args.output.expanduser().resolve())
    print(f"output={args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
