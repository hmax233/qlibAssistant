#!/usr/bin/env python3
"""Apply the same hard risk filters to XGB240 and Fixed Ensemble sources."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN

from evaluate_hard_risk_filters import (
    RULES,
    risk_execution_features,
    simulate,
    summarize,
)
from evaluate_mainboard_ensemble_sizes import daily_zscore
from report_mainboard_matrix import (
    BASELINE_XGB240,
    MLRUNS,
    ROOT,
    as_series,
    benchmark_returns,
    load_pickle,
    mainboard_mask,
)


DEFAULT_WINDOW_DETAIL = (
    ROOT
    / ".qlibAssistant/analysis/window_length_comparison_20260720"
    / "window_fold_detail.csv"
)
FIXED_COMPONENTS = (
    ("XGBoost", 60),
    ("XGBoost", 120),
    ("LightGBM", 84),
    ("CatBoost", 120),
)


def load_prediction(
    experiment_id: str,
    recorder_id: str,
) -> tuple[pd.Series, pd.Series]:
    artifacts = MLRUNS / str(experiment_id) / str(recorder_id) / "artifacts"
    prediction = as_series(load_pickle(artifacts / "pred.pkl"), "score")
    label = as_series(load_pickle(artifacts / "label.pkl"), "label")
    prediction = prediction.loc[mainboard_mask(prediction.index)]
    label = label.loc[mainboard_mask(label.index)]
    return prediction, label


def build_xgb240_sources() -> list[dict]:
    sources = []
    for fold, (experiment_id, recorder_id) in BASELINE_XGB240.items():
        prediction, label = load_prediction(experiment_id, recorder_id)
        sources.append(
            {
                "source": "XGBoost240",
                "fold": fold,
                "score": prediction,
                "label": label,
                "components": [
                    {
                        "model": "XGBoost",
                        "train_months": 240,
                        "experiment_id": experiment_id,
                        "recorder_id": recorder_id,
                    }
                ],
            }
        )
    return sources


def fixed_component_rows(detail: pd.DataFrame, fold: str) -> pd.DataFrame:
    selected = []
    for model, months in FIXED_COMPONENTS:
        match = detail[
            detail["fold"].eq(fold)
            & detail["model_name"].eq(model)
            & detail["train_months"].eq(months)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one {fold} {model}_{months}m recorder, got {len(match)}"
            )
        selected.append(match.iloc[0])
    return pd.DataFrame(selected)


def build_fixed_sources(detail: pd.DataFrame) -> list[dict]:
    sources = []
    for fold in ("fold1", "fold2", "fold3"):
        selected = fixed_component_rows(detail, fold)
        component_scores = {}
        labels = []
        metadata = []
        for _, row in selected.iterrows():
            prediction, label = load_prediction(
                row["experiment_id"], row["recorder_id"]
            )
            key = f"{row['model_name']}_{int(row['train_months'])}m"
            component_scores[key] = daily_zscore(prediction)
            labels.append(label.rename(key))
            metadata.append(
                {
                    "model": row["model_name"],
                    "train_months": int(row["train_months"]),
                    "experiment_id": str(row["experiment_id"]),
                    "recorder_id": str(row["recorder_id"]),
                    "valid_rank_icir": float(row["valid_Rank ICIR"]),
                }
            )

        xgb_score = pd.concat(
            [
                component_scores["XGBoost_60m"],
                component_scores["XGBoost_120m"],
            ],
            axis=1,
            join="inner",
        ).mean(axis=1)
        family_scores = {
            "XGBoost": xgb_score,
            "LightGBM": component_scores["LightGBM_84m"],
            "CatBoost": component_scores["CatBoost_120m"],
        }
        family_weights = {
            "XGBoost": selected.loc[
                selected["model_name"].eq("XGBoost"), "valid_Rank ICIR"
            ].mean(),
            "LightGBM": selected.loc[
                selected["model_name"].eq("LightGBM"), "valid_Rank ICIR"
            ].iloc[0],
            "CatBoost": selected.loc[
                selected["model_name"].eq("CatBoost"), "valid_Rank ICIR"
            ].iloc[0],
        }
        total_weight = sum(family_weights.values())
        score = sum(
            family_scores[family] * family_weights[family] / total_weight
            for family in family_scores
        ).rename("score")
        label = pd.concat(labels, axis=1, join="inner").mean(
            axis=1, skipna=False
        ).rename("label")
        sources.append(
            {
                "source": "FixedArchitecture",
                "fold": fold,
                "score": score,
                "label": label,
                "components": metadata,
            }
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-detail", default=str(DEFAULT_WINDOW_DETAIL))
    parser.add_argument("--topks", default="1,3")
    parser.add_argument(
        "--output-dir",
        default=(
            ".qlibAssistant/analysis/source_hard_filter_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    args = parser.parse_args()

    qlib.init(
        provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"),
        region=REG_CN,
    )
    detail = pd.read_csv(
        args.window_detail,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    sources = build_xgb240_sources() + build_fixed_sources(detail)
    starts = [
        item["score"].index.get_level_values("datetime").min() for item in sources
    ]
    ends = [
        item["score"].index.get_level_values("datetime").max() for item in sources
    ]
    execution = risk_execution_features(str(min(starts).date()), str(max(ends).date()))
    benchmarks = benchmark_returns(str(min(starts).date()), str(max(ends).date()))

    topks = [int(value) for value in args.topks.split(",") if value]
    rows = []
    daily_rows = []
    source_metadata = []
    for item in sources:
        frame = pd.concat(
            [item["score"].rename("score"), item["label"].rename("label")],
            axis=1,
        ).dropna()
        source_metadata.append(
            {
                "source": item["source"],
                "fold": item["fold"],
                "test_start": frame.index.get_level_values("datetime")
                .min()
                .strftime("%Y-%m-%d"),
                "test_end": frame.index.get_level_values("datetime")
                .max()
                .strftime("%Y-%m-%d"),
                "components": item["components"],
                "is_exact_current_fixed": (
                    item["source"] == "FixedArchitecture"
                    and item["fold"] == "fold3"
                ),
            }
        )
        for topk in topks:
            for rule_name, rule in RULES.items():
                for fallback in (False, True):
                    daily = simulate(frame, execution, topk, rule, fallback)
                    rows.append(
                        {
                            "source": item["source"],
                            "fold": item["fold"],
                            "topk": topk,
                            "rule": rule_name,
                            "fallback": fallback,
                            **summarize(daily, benchmarks),
                        }
                    )
                    daily_rows.append(
                        daily.reset_index().assign(
                            source=item["source"],
                            fold=item["fold"],
                            topk=topk,
                            rule=rule_name,
                            fallback=fallback,
                        )
                    )

    by_fold = pd.DataFrame(rows)
    numeric = [
        column
        for column in by_fold.select_dtypes(include=np.number).columns
        if column != "topk"
    ]
    average = (
        by_fold.groupby(["source", "topk", "rule", "fallback"])[numeric]
        .mean()
        .reset_index()
    )
    worst = (
        by_fold.groupby(["source", "topk", "rule", "fallback"])[numeric]
        .min()
        .reset_index()
    )

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    by_fold.to_csv(output / "source_hard_filter_by_fold.csv", index=False)
    average.to_csv(output / "source_hard_filter_three_fold_average.csv", index=False)
    worst.to_csv(output / "source_hard_filter_worst_fold.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(
        output / "source_hard_filter_daily.csv", index=False
    )
    (output / "source_components.json").write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "rules": RULES,
        "topks": topks,
        "xgb240": "The project's exact three XGB240 fold recorders.",
        "fixed": (
            "Fold3 is the exact frozen four-recorder Fixed Ensemble. Fold1/2 "
            "use the same model/window architecture with their corresponding "
            "fold recorders and family-level validation Rank ICIR weighting."
        ),
        "board_filter": "STAR Market and ChiNext removed before ranking.",
        "timing": (
            "Signal T, filter and trade at T+1 close, hold to T+2 close."
        ),
    }
    (output / "report_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    display = average[
        [
            "source",
            "topk",
            "rule",
            "fallback",
            "net_cumulative",
            "net_max_drawdown",
            "net_sharpe_rf0",
            "average_cash_ratio",
            "CSI1000_cumulative",
            "net_excess_vs_CSI1000",
            "CSI300_cumulative",
            "net_excess_vs_CSI300",
        ]
    ]
    display = display[
        display["rule"].isin(["baseline", "event_guard", "event_and_weak_guard"])
        & ~display["fallback"]
    ].sort_values(["source", "topk", "rule"])
    print(display.to_string(index=False))
    print(f"\noutput_dir={output}")


if __name__ == "__main__":
    main()
