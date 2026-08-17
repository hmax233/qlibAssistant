#!/usr/bin/env python3
"""Learn T+1 upper-limit risk and calibrate a return ensemble without test tuning.

The return ensemble is left unchanged.  A separate classifier is fitted on
train, early-stopped on valid, and its score penalty is selected only on
selection-validation using the same strict execution simulator used for the
final test report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qlibAssistant/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from qlib.constant import REG_CN
from qlib.data import D
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(ROOT / "script"))
from evaluate_hard_risk_filters import RULES, risk_execution_features, simulate, summarize  # noqa: E402
from report_mainboard_matrix import benchmark_returns  # noqa: E402
from run_multisource_fixed_v2 import (  # noqa: E402
    Split,
    daily_zscore,
    join_split,
    load_new_factors,
    make_dataset,
    qlib_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="csi1000_mainboard")
    parser.add_argument("--intraday-root", type=Path, required=True)
    parser.add_argument("--daily-source", type=Path, required=True)
    parser.add_argument("--selected-features", type=Path, required=True)
    parser.add_argument("--selection-prediction", type=Path, required=True)
    parser.add_argument("--test-prediction", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--penalties", default="0,0.25,0.5,0.75,1,1.5,2,3,4")
    parser.add_argument("--topk-objective", type=int, default=1)
    parser.add_argument("--provider-uri", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    parser.add_argument("--run-tag", required=True)
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()


def load_upper_limit_label(index: pd.MultiIndex) -> pd.Series:
    start = str(index.get_level_values("datetime").min().date())
    end = str(index.get_level_values("datetime").max().date())
    instruments = sorted(index.get_level_values("instrument").unique())
    data = D.features(
        instruments,
        ["Ref($change,-1)", "Ref($close,-1)"],
        start_time=start,
        end_time=end,
        freq="day",
    )
    data.columns = ["next_change", "next_close"]
    if set(data.index.names) == {"datetime", "instrument"}:
        data = data.reorder_levels(["datetime", "instrument"]).sort_index()
    # Mainboard-only universe: a rounded +10% close is conservatively treated
    # as unavailable. Missing T+1 quotes are also unavailable for training.
    unavailable = data["next_close"].isna() | data["next_change"].ge(0.095)
    return unavailable.reindex(index).fillna(True).astype("int8").rename("unavailable")


def classifier_metrics(label: pd.Series, probability: np.ndarray) -> dict:
    values = label.to_numpy(dtype=int)
    if len(np.unique(values)) < 2:
        return {"roc_auc": np.nan, "average_precision": np.nan, "positive_rate": float(values.mean())}
    return {
        "roc_auc": float(roc_auc_score(values, probability)),
        "average_precision": float(average_precision_score(values, probability)),
        "positive_rate": float(values.mean()),
    }


def adjusted_frame(base: pd.DataFrame, probability: pd.Series, penalty: float) -> pd.DataFrame:
    common = base.index.intersection(probability.index)
    base_score = daily_zscore(base.loc[common, "score"])
    risk_rank = probability.loc[common].groupby(level="datetime").rank(pct=True)
    return pd.DataFrame({
        "score": base_score - penalty * risk_rank,
        "label": base.loc[common, "label"],
        "unavailable_probability": probability.loc[common],
        "unavailable_risk_rank": risk_rank,
    }).dropna(subset=["score", "label"])


def main() -> None:
    args = parse_args()
    output = ROOT / ".qlibAssistant/experiments" / args.run_tag
    output.mkdir(parents=True, exist_ok=False)
    split = Split()
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    selected = set(json.loads(args.selected_features.read_text(encoding="utf-8"))["selected_features"])
    new = load_new_factors(args.intraday_root, args.daily_source, selected)
    earliest = (pd.Timestamp(split.train_end) - pd.DateOffset(months=60)).strftime("%Y-%m-%d")
    dataset = make_dataset(args, earliest, split.test_end, split)
    intervals = {
        "train": (earliest, split.train_end),
        "valid": (split.valid_start, split.valid_end),
        "selection": (split.selection_start, split.selection_end),
        "test": (split.test_start, split.test_end),
    }
    prepared = {}
    for name, interval in intervals.items():
        alpha, raw_label = qlib_slice(dataset, interval)
        features, _ = join_split(alpha, raw_label, new)
        unavailable = load_upper_limit_label(features.index)
        prepared[name] = (features, unavailable)
        print(f"prepared {name} rows={len(features)} unavailable={unavailable.mean():.3%}", flush=True)

    x_train, y_train = prepared["train"]
    x_valid, y_valid = prepared["valid"]
    names = list(x_train.columns)
    positive = max(int(y_train.sum()), 1)
    negative = len(y_train) - positive
    # A capped class weight makes the rare-event classifier sensitive without
    # turning its output into an unusably extreme pseudo-probability.
    scale_pos_weight = min(negative / positive, 20.0)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=names)
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=names)
    model = xgb.train(
        {
            "objective": "binary:logistic", "eval_metric": ["auc", "aucpr"],
            "eta": 0.035, "max_depth": 7, "min_child_weight": 30,
            "subsample": 0.82, "colsample_bytree": 0.75,
            "lambda": 10.0, "alpha": 0.2, "scale_pos_weight": scale_pos_weight,
            "nthread": args.threads, "seed": 20260818,
        },
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=50,
    )
    model.save_model(output / "unavailable_classifier.json")
    iteration = (0, model.best_iteration + 1)
    diagnostics = {}
    probabilities = {}
    for name in ("valid", "selection", "test"):
        features, label = prepared[name]
        probability = model.predict(
            xgb.DMatrix(features, feature_names=names), iteration_range=iteration
        )
        probabilities[name] = pd.Series(probability, index=features.index, name="unavailable_probability")
        diagnostics[name] = classifier_metrics(label, probability)
    (output / "classifier_metrics.json").write_text(
        json.dumps({"best_iteration": int(model.best_iteration), "scale_pos_weight": scale_pos_weight,
                    **diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selection_base = load_prediction(args.selection_prediction)
    test_base = load_prediction(args.test_prediction)
    selection_execution = risk_execution_features(split.selection_start, split.selection_end)
    selection_benchmarks = benchmark_returns(split.selection_start, split.selection_end)
    penalties = [float(value) for value in args.penalties.split(",") if value]
    rows = []
    selection_frames = {}
    for penalty in penalties:
        frame = adjusted_frame(selection_base, probabilities["selection"], penalty)
        selection_frames[penalty] = frame
        for topk in (1, 3, 5, 10):
            daily = simulate(frame, selection_execution, topk, RULES["baseline"], fallback=False)
            rows.append({"penalty": penalty, "topk": topk, **summarize(daily, selection_benchmarks)})
    calibration = pd.DataFrame(rows)
    calibration.to_csv(output / "selection_penalty_grid.csv", index=False)
    objective = calibration[calibration["topk"].eq(args.topk_objective)].sort_values(
        ["net_cumulative", "net_max_drawdown"], ascending=False
    )
    selected_penalty = float(objective.iloc[0]["penalty"])
    selection_final = selection_frames[selected_penalty]
    test_final = adjusted_frame(test_base, probabilities["test"], selected_penalty)
    selection_final.reset_index().to_parquet(output / "execution_aware_selection.parquet", index=False)
    test_final.reset_index().to_parquet(output / "execution_aware_test.parquet", index=False)
    (output / "selection.json").write_text(json.dumps({
        "selected_penalty": selected_penalty,
        "selection_objective": f"strict baseline Top{args.topk_objective} net cumulative",
        "selection_uses_test": False,
        "base_selection_prediction": str(args.selection_prediction.resolve()),
        "base_test_prediction": str(args.test_prediction.resolve()),
        "test_prediction": str((output / "execution_aware_test.parquet").resolve()),
        "warning": "penalty selected on one six-month selection window; test remains the only final report",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(8, 5))
    for topk, group in calibration.groupby("topk", sort=False):
        axis.plot(group["penalty"], group["net_cumulative"], marker="o", label=f"Top{topk}")
    axis.axvline(selected_penalty, color="black", linestyle="--", linewidth=1,
                 label=f"selected={selected_penalty:g}")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Unavailable-risk rank penalty")
    axis.set_ylabel("Selection strict net cumulative")
    axis.set_title("Execution-aware penalty selected without test")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "selection_penalty_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"selected_penalty={selected_penalty:g} output={output}")


if __name__ == "__main__":
    main()
