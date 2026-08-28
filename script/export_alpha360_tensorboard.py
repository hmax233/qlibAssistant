#!/usr/bin/env python3
"""Export completed Alpha360 E0-E6 CSV evidence to TensorBoard event files."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


DEFAULT_SOURCE = Path(
    ".qlibAssistant/analysis/alpha360_e0_e6_260828"
)


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_nll(frame: pd.DataFrame) -> pd.Series | None:
    for column in ("nll_scaled_3leg", "nll_scaled"):
        if column in frame:
            return frame[column]
    columns = [column for column in frame if column.endswith("_nll_log_return")]
    return frame[columns].mean(axis=1) if columns else None


def export_epoch_run(csv_path: Path, output_root: Path) -> None:
    frame = pd.read_csv(csv_path)
    if frame.empty or "epoch" not in frame:
        raise ValueError(f"Invalid epoch metrics: {csv_path}")

    writer = SummaryWriter(str(output_root / csv_path.stem))
    train_column = "train_nll_scaled" if "train_nll_scaled" in frame else "train_nll"
    valid_nll = _valid_nll(frame)

    for row_index, row in frame.iterrows():
        step = int(row["epoch"])
        common = {
            "loss/train_nll": row.get(train_column),
            "optimization/learning_rate": row.get("learning_rate"),
            "timing/epoch_wall_time_seconds": row.get("epoch_seconds"),
        }
        if valid_nll is not None:
            common["loss/validation_nll"] = valid_nll.iloc[row_index]
        for tag, value in common.items():
            finite = _finite(value)
            if finite is not None:
                writer.add_scalar(tag, finite, step)

        for column, value in row.items():
            if column in {
                "epoch", train_column, "learning_rate", "epoch_seconds",
                "best_valid_nll", "days", "nll_scaled", "nll_scaled_3leg",
            }:
                continue
            finite = _finite(value)
            if finite is None:
                continue
            if "_" in column:
                horizon, metric = column.rsplit("_", 1)
                # Keep complete metric names such as rank_ic, coverage50 and nll_log_return.
                for suffix in (
                    "rank_icir", "rank_ic", "nll_log_return", "coverage50",
                    "coverage80", "coverage95", "brier",
                ):
                    marker = f"_{suffix}"
                    if column.endswith(marker):
                        horizon = column[: -len(marker)]
                        metric = suffix
                        break
                if metric != "mae":
                    writer.add_scalar(f"validation/{horizon}/{metric}", finite, step)
            else:
                writer.add_scalar(f"epoch_metrics/{column}", finite, step)

        best = _finite(row.get("best_valid_nll"))
        if best is not None:
            writer.add_scalar("loss/best_validation_nll", best, step)

    writer.add_text(
        "run/source",
        f"Imported from `{csv_path.resolve()}`; values are preserved from the completed run.",
        0,
    )
    writer.flush()
    writer.close()


def export_frozen_ensemble(source: Path, output_root: Path) -> None:
    summary_path = source / "report_fallback" / "concise_summary.csv"
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    writer = SummaryWriter(str(output_root / "E7_frozen_selection_ensemble"))
    for step, row in frame.reset_index(drop=True).iterrows():
        horizon = str(row["horizon"])
        for column in (
            "selection_rank_ic", "selection_rank_icir", "test_rank_ic", "test_rank_icir",
            "test_top1_cumulative", "test_top1_win_rate",
            "mainboard_selection_rule_test_net_cumulative",
            "mainboard_selection_rule_test_win_rate",
            "mainboard_selection_rule_test_max_drawdown",
            "mainboard_CSI300_cumulative", "mainboard_CSI1000_cumulative",
        ):
            finite = _finite(row.get(column))
            if finite is not None:
                writer.add_scalar(f"{horizon}/{column}", finite, step + 1)
        writer.add_text(
            f"{horizon}/selected_components",
            str(row.get("selected_components", "")),
            step + 1,
        )
    writer.flush()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or source / "tensorboard").resolve()
    epoch_root = source / "epoch_metrics"
    csv_paths = sorted(epoch_root.glob("E*.csv"))
    expected = {f"E{i}" for i in range(7)}
    actual = {path.stem.split("_", 1)[0] for path in csv_paths}
    if actual != expected:
        raise RuntimeError(f"Expected E0-E6, found {sorted(actual)}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        export_epoch_run(csv_path, output)
    export_frozen_ensemble(source, output)
    print(f"TensorBoard logdir: {output}")
    print(f"Runs exported: {len(csv_paths)} training runs + frozen ensemble")


if __name__ == "__main__":
    main()
