#!/usr/bin/env python3
"""Evaluate saved cross-market boosters on an existing fixed Fold window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from qlib.constant import REG_CN

from common import MODELS, REPORTS, ROOT, atomic_json
from train_global_then_a import (
    add_market_features,
    daily_metrics,
    load_period,
    matrix,
    predict,
)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "script") not in sys.path:
    sys.path.insert(0, str(ROOT / "script"))

from evaluate_hard_risk_filters import (  # noqa: E402
    RULES,
    risk_execution_features,
    simulate,
    summarize,
)
from report_mainboard_matrix import benchmark_returns  # noqa: E402


DEFAULT_COMPARISON = (
    ROOT
    / ".qlibAssistant"
    / "analysis"
    / "source_hard_filter_20260728_225020"
)
DEFAULT_MAINBOARD_COMPARISON = (
    ROOT
    / ".qlibAssistant"
    / "analysis"
    / "hard_risk_filter_20260728_223233"
)


def comparison_dates(folder: Path, fold: str) -> pd.DatetimeIndex:
    path = folder / "source_hard_filter_daily.csv"
    daily = pd.read_csv(path, parse_dates=["datetime"])
    selected = daily[
        daily["source"].eq("XGBoost240")
        & daily["fold"].eq(fold)
        & daily["topk"].eq(1)
        & daily["rule"].eq("baseline")
        & daily["fallback"].eq(False)
    ]
    return pd.DatetimeIndex(sorted(selected["datetime"].unique()))


def load_boosters(run_tag: str) -> dict[str, xgb.Booster]:
    model_dir = MODELS / run_tag
    filenames = {
        "global": "global_pretrain.ubj",
        "finetuned": "global_then_a_finetuned.ubj",
        "a_only": "a_only_control.ubj",
    }
    boosters = {}
    for name, filename in filenames.items():
        booster = xgb.Booster()
        booster.load_model(model_dir / filename)
        boosters[name] = booster
    return boosters


def as_legacy_frame(
    frame: pd.DataFrame, score: np.ndarray
) -> pd.DataFrame:
    result = frame[["date", "symbol", "label_abs"]].copy()
    result["score"] = score
    result = result.rename(
        columns={
            "date": "datetime",
            "symbol": "instrument",
            "label_abs": "label",
        }
    )
    return result.set_index(["datetime", "instrument"]).sort_index()


def previous_source_rows(folder: Path, fold: str) -> pd.DataFrame:
    detail = pd.read_csv(folder / "source_hard_filter_by_fold.csv")
    return detail[
        detail["fold"].eq(fold)
        & detail["topk"].eq(1)
        & detail["rule"].isin(["baseline", "event_guard"])
        & detail["fallback"].eq(False)
    ].copy()


def mainboard_rows(
    folder: Path,
    fold: str,
    dates: pd.DatetimeIndex,
    benchmarks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Re-summarize Mainboard20 on the exact shared comparison dates."""

    daily = pd.read_csv(folder / "hard_filter_daily.csv", parse_dates=["datetime"])
    selected = daily[
        daily["fold"].eq(fold)
        & daily["topk"].eq(1)
        & daily["rule"].isin(["baseline", "event_guard"])
        & daily["fallback"].eq(False)
        & daily["datetime"].isin(dates)
    ].copy()
    rows = []
    for rule_name, group in selected.groupby("rule", sort=False):
        group = group.set_index("datetime").sort_index()
        rows.append(
            {
                "source": "Mainboard20",
                "fold": fold,
                "topk": 1,
                "rule": rule_name,
                "fallback": False,
                **summarize(group, benchmarks),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--fold", default="fold3")
    parser.add_argument("--start", default="2026-02-17")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--mainboard-comparison-dir",
        type=Path,
        default=DEFAULT_MAINBOARD_COMPARISON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output_dir or (
        ROOT
        / ".qlibAssistant"
        / "analysis"
        / f"cross_market_{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)

    dates = comparison_dates(args.comparison_dir, args.fold)
    frame = load_period(
        "A",
        args.start,
        args.end,
        0,
        0,
        20260731,
        120,
        0.20,
        training_filter=False,
    )
    frame = frame[frame["date"].isin(dates)].copy()
    frame = add_market_features(frame)
    features = json.loads(
        (REPORTS / args.run_tag / "feature_names.json").read_text(encoding="utf-8")
    )
    dmatrix = matrix(frame, features, label=False)
    scores = {
        name: predict(booster, dmatrix)
        for name, booster in load_boosters(args.run_tag).items()
    }

    benchmark_file = REPORTS / args.run_tag / "benchmark_daily_returns.csv"
    exact_benchmarks = pd.read_csv(
        benchmark_file, index_col="date", parse_dates=True
    ).reindex(dates)
    exact_rows = []
    for name, score in scores.items():
        metrics = daily_metrics(frame, score, exact_benchmarks)
        exact_rows.append({"source": name, **metrics})
    pd.DataFrame(exact_rows).to_csv(
        output / "exact_close_metrics_0047pct.csv", index=False
    )

    qlib.init(
        provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"),
        region=REG_CN,
    )
    execution = risk_execution_features(args.start, args.end)
    legacy_benchmarks = benchmark_returns(args.start, args.end)
    legacy_rows = []
    legacy_daily = []
    for name, score in scores.items():
        model_frame = as_legacy_frame(frame, score)
        for rule_name in ("baseline", "event_guard"):
            daily = simulate(
                model_frame,
                execution,
                topk=1,
                rule=RULES[rule_name],
                fallback=False,
            )
            metrics = summarize(daily, legacy_benchmarks)
            legacy_rows.append(
                {
                    "source": name,
                    "fold": args.fold,
                    "topk": 1,
                    "rule": rule_name,
                    "fallback": False,
                    **metrics,
                }
            )
            legacy_daily.append(
                daily.reset_index().assign(source=name, rule=rule_name)
            )
    pd.DataFrame(legacy_rows).to_csv(
        output / "comparable_stateful_015pct.csv", index=False
    )
    pd.concat(legacy_daily, ignore_index=True).to_csv(
        output / "comparable_stateful_daily.csv", index=False
    )

    previous = previous_source_rows(args.comparison_dir, args.fold)
    mainboard = mainboard_rows(
        args.mainboard_comparison_dir,
        args.fold,
        dates,
        legacy_benchmarks,
    )
    previous = pd.concat([previous, mainboard], ignore_index=True, sort=False)
    previous.to_csv(output / "previous_multisource_fold3.csv", index=False)
    combined = pd.concat(
        [
            pd.DataFrame(legacy_rows).assign(group="cross_market"),
            previous.assign(group="previous_multisource"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(output / "fold3_comparison.csv", index=False)
    metadata = {
        "run_tag": args.run_tag,
        "fold": args.fold,
        "nominal_start": args.start,
        "nominal_end": args.end,
        "common_signal_days": len(dates),
        "actual_start": dates.min().strftime("%Y-%m-%d"),
        "actual_end": dates.max().strftime("%Y-%m-%d"),
        "model_universe": "A-share main board; historical ST excluded",
        "exact_close_report": (
            "Exact daily limit prices, no rank fallback, 0.0235% each side, "
            "fresh one-day positions."
        ),
        "comparable_report": (
            "Same legacy stateful simulator as previous sources: no fallback, "
            "0.15% turnover cost, approximate Qlib limit execution; both "
            "baseline and exploratory event_guard."
        ),
        "event_guard_warning": (
            "Uses complete T+1 daily low/close while assuming T+1 close "
            "execution; exploratory and same-bar idealized."
        ),
        "output": str(output),
    }
    atomic_json(output / "metadata.json", metadata)
    print(pd.DataFrame(legacy_rows).to_string(index=False))
    print("\nPrevious multisource:\n")
    print(previous.to_string(index=False))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
