#!/usr/bin/env python3
"""Audit whether saved Fold3 predictions align with the intended return date."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_source_hard_filters import (
    DEFAULT_WINDOW_DETAIL,
    build_fixed_sources,
    build_xgb240_sources,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".qlibAssistant" / "analysis"
RAW = Path("/Users/hmax/investment_data/cross_market_daily/raw/a")


def targets(label: pd.Series) -> dict[str, pd.Series]:
    label = label.sort_index()
    later = label.groupby(level="instrument", sort=False).shift(-1)
    later_two = label.groupby(level="instrument", sort=False).shift(-2)
    return {
        "intended_t1_to_t2": label,
        "later_t2_to_t3": later,
        "two_day_t1_to_t3": (1 + label) * (1 + later) - 1,
        "later_t3_to_t4": later_two,
        "three_day_t1_to_t4": (1 + label) * (1 + later) * (1 + later_two) - 1,
    }


def evaluate(score: pd.Series, target: pd.Series) -> dict:
    frame = pd.concat(
        [score.rename("score"), target.rename("target")], axis=1
    ).dropna()
    daily_rank_ic = frame.groupby(level="datetime").apply(
        lambda group: group["score"].corr(group["target"], method="spearman"),
        include_groups=False,
    ).dropna()
    daily_ic = frame.groupby(level="datetime").apply(
        lambda group: group["score"].corr(group["target"]),
        include_groups=False,
    ).dropna()
    top = (
        frame.reset_index()
        .sort_values(["datetime", "score"], ascending=[True, False])
        .groupby("datetime", as_index=False)
        .head(1)
        .set_index("datetime")["target"]
        .sort_index()
    )
    return {
        "days": int(top.index.nunique()),
        "mean_IC": float(daily_ic.mean()),
        "ICIR": float(daily_ic.mean() / daily_ic.std()),
        "mean_RankIC": float(daily_rank_ic.mean()),
        "RankICIR": float(daily_rank_ic.mean() / daily_rank_ic.std()),
        "Top1_win_rate": float((top > 0).mean()),
        "Top1_gross_cumulative": float((1 + top).prod() - 1),
        "Top1_average_return": float(top.mean()),
    }


def verify_raw_label(label: pd.Series, sample_size: int = 200) -> dict:
    recent = label.dropna()
    recent = recent[
        recent.index.get_level_values("datetime") >= pd.Timestamp("2026-06-01")
    ]
    if len(recent) > sample_size:
        recent = recent.sample(sample_size, random_state=20260731)
    differences = []
    for (date, symbol), stored in recent.items():
        path = RAW / f"{symbol}.parquet"
        if not path.exists():
            continue
        prices = (
            pd.read_parquet(path, columns=["datetime", "close"])
            .drop_duplicates("datetime")
            .set_index("datetime")
            .sort_index()
        )
        if date not in prices.index:
            continue
        position = prices.index.get_loc(date)
        if not isinstance(position, (int, np.integer)) or position + 2 >= len(prices):
            continue
        expected = (
            float(prices.iloc[position + 2]["close"])
            / float(prices.iloc[position + 1]["close"])
            - 1
        )
        differences.append(abs(float(stored) - expected))
    return {
        "raw_alignment_samples": len(differences),
        "raw_alignment_max_abs_error": (
            max(differences) if differences else np.nan
        ),
    }


def main() -> None:
    detail = pd.read_csv(
        DEFAULT_WINDOW_DETAIL,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    sources = build_xgb240_sources() + build_fixed_sources(detail)
    sources = [item for item in sources if item["fold"] == "fold3"]
    common_dates = None
    for item in sources:
        dates = set(item["score"].index.get_level_values("datetime"))
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = pd.DatetimeIndex(sorted(common_dates))

    rows = []
    for item in sources:
        score = item["score"][
            item["score"].index.get_level_values("datetime").isin(common_dates)
        ]
        label = item["label"][
            item["label"].index.get_level_values("datetime").isin(common_dates)
        ]
        raw_check = verify_raw_label(label)
        for target_name, target in targets(label).items():
            rows.append(
                {
                    "source": item["source"],
                    "fold": item["fold"],
                    "target": target_name,
                    **raw_check,
                    **evaluate(score, target),
                }
            )
    result = pd.DataFrame(rows)
    output = ANALYSIS / f"label_alignment_audit_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True)
    result.to_csv(output / "alignment_metrics.csv", index=False)
    metadata = {
        "intended_t1_to_t2": "Ref($close,-2)/Ref($close,-1)-1",
        "later_t2_to_t3": "The same stored label shifted one signal day later",
        "two_day_t1_to_t3": "(1+intended)*(1+later)-1",
        "later_t3_to_t4": "The stored label shifted two signal days later",
        "three_day_t1_to_t4": "(1+intended)*(1+later)*(1+later_two)-1",
        "scope": "Common Fold3 dates for XGBoost240 and FixedArchitecture",
        "warning": "Gross ranking diagnostic; no costs or execution filters.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
