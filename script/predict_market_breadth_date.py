#!/usr/bin/env python3
"""Predict next-day all-A-share market breadth with saved recorder models."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / ".qlibAssistant/cache/market_breadth_daily.csv"
DEFAULT_SOURCE_REPORT = (
    ROOT / ".qlibAssistant/analysis/market_breadth_fold3_260727_v2_015952"
)
MLRUNS = ROOT / ".qlibAssistant/mlruns"
TARGET_COLUMNS = {
    "next_up_ratio",
    "next_mainboard_up_ratio",
    "next_ret_mean",
    "next_market_up",
}


def condition_name(value: float) -> str:
    if value < 0.35:
        return "极弱"
    if value < 0.45:
        return "偏弱"
    if value <= 0.55:
        return "中性"
    if value <= 0.65:
        return "偏强"
    return "极强"


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Signal date (YYYY-MM-DD); default: latest row")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--output-root", default=str(ROOT / ".qlibAssistant/analysis"))
    parser.add_argument(
        "--ensemble-sizes",
        default="2,4,6,8,20",
        help="Comma-separated top-N model counts ranked on selection-validation MAE",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    source_report = Path(args.source_report).expanduser().resolve()
    metrics_path = source_report / "recorder_metrics.csv"
    data = pd.read_csv(dataset_path, parse_dates=["datetime"]).set_index("datetime")
    data = data.sort_index()
    if args.date:
        signal_date = pd.Timestamp(args.date)
        eligible = data.loc[data.index <= signal_date]
        if eligible.empty or eligible.index[-1] != signal_date:
            raise SystemExit(f"dataset has no exact row for {args.date}")
        latest = eligible.iloc[[-1]]
    else:
        latest = data.iloc[[-1]]
        signal_date = latest.index[-1]

    recorder_metrics = pd.read_csv(
        metrics_path,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    rows = []
    for row in recorder_metrics.itertuples(index=False):
        artifact_dir = (
            MLRUNS / str(row.experiment_id) / str(row.recorder_id) / "artifacts"
        )
        model = load_pickle(artifact_dir / "params.pkl")
        feature_columns = load_pickle(artifact_dir / "feature_columns.pkl")
        missing = sorted(set(feature_columns) - set(latest.columns))
        if missing:
            raise SystemExit(
                f"{row.model}_{int(row.train_months)}m missing features: {missing[:5]}"
            )
        prediction = float(
            np.clip(model.predict(latest.loc[:, feature_columns])[0], 0.0, 1.0)
        )
        rows.append(
            {
                "config": f"{row.model}_{int(row.train_months)}m",
                "model": row.model,
                "train_months": int(row.train_months),
                "experiment_id": str(row.experiment_id),
                "recorder_id": str(row.recorder_id),
                "selection_valid_MAE": float(row.selection_valid_MAE),
                "selection_valid_AUC": float(row.selection_valid_direction_AUC),
                "test_MAE": float(row.test_MAE),
                "test_AUC": float(row.test_direction_AUC),
                "predicted_up_ratio": prediction,
                "condition": condition_name(prediction),
            }
        )

    models = pd.DataFrame(rows).sort_values(
        ["selection_valid_MAE", "selection_valid_AUC"],
        ascending=[True, False],
    )
    models.insert(0, "selection_rank", np.arange(1, len(models) + 1))

    requested_sizes = [
        int(value) for value in args.ensemble_sizes.split(",") if value.strip()
    ]
    ensemble_rows = []
    for size in requested_sizes:
        selected = models.head(min(size, len(models))).copy()
        raw_weights = 1.0 / selected["selection_valid_MAE"].clip(lower=1e-6)
        weights = raw_weights / raw_weights.sum()
        prediction = float(np.average(selected["predicted_up_ratio"], weights=weights))
        ensemble_rows.append(
            {
                "ensemble_size": len(selected),
                "predicted_up_ratio": prediction,
                "predicted_market_up": prediction > 0.5,
                "condition": condition_name(prediction),
                "selected_configs": ",".join(selected["config"]),
                "weighted_mean_individual_test_MAE": float(
                    np.average(selected["test_MAE"], weights=weights)
                ),
                "weighted_mean_individual_test_AUC": float(
                    np.average(selected["test_AUC"], weights=weights)
                ),
            }
        )
    ensembles = pd.DataFrame(ensemble_rows).drop_duplicates("ensemble_size")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / f"market_breadth_prediction_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    models.to_csv(output_dir / "model_predictions.csv", index=False)
    ensembles.to_csv(output_dir / "ensemble_predictions.csv", index=False)
    config = {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "target_date": "next trading day",
        "target": (
            "fraction of eligible all-A-shares with positive adjusted "
            "close-to-close return on the next trading day"
        ),
        "dataset": str(dataset_path),
        "source_report": str(source_report),
        "current_up_ratio": (
            float(latest.iloc[0]["up_ratio"])
            if "up_ratio" in latest.columns
            else None
        ),
        "current_mainboard_up_ratio": (
            float(latest.iloc[0]["mainboard_up_ratio"])
            if "mainboard_up_ratio" in latest.columns
            else None
        ),
    }
    (output_dir / "prediction_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"signal_date={config['signal_date']}")
    print(
        "current_up_ratio="
        f"{config['current_up_ratio']:.4f} "
        "current_mainboard_up_ratio="
        f"{config['current_mainboard_up_ratio']:.4f}"
    )
    print(ensembles.to_string(index=False))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
