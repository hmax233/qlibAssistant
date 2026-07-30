#!/usr/bin/env python3
"""Predict an A-mainboard ranking with a saved cross-market transfer run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common import FACTORS, MODELS, REPORTS, ROOT, atomic_json
from train_global_then_a import (
    add_market_features,
    load_limit_prices,
    matrix,
    predict,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def latest_a_date() -> str:
    calendar = Path.home() / ".qlib/qlib_data/cn_data/calendars/day.txt"
    return calendar.read_text(encoding="utf-8").splitlines()[-1].strip()


def load_date(date: str) -> pd.DataFrame:
    target = pd.Timestamp(date)
    pieces = []
    files = sorted((FACTORS / "a").glob("*.parquet"))
    for idx, path in enumerate(files, 1):
        try:
            part = pd.read_parquet(path, filters=[("date", "==", target)])
        except Exception:
            part = pd.read_parquet(path)
            part = part[part["date"].eq(target)]
        if not part.empty:
            pieces.append(part)
        if idx % 500 == 0:
            print(f"load latest factors {idx}/{len(files)} rows={len(pieces)}", flush=True)
    if not pieces:
        raise RuntimeError(f"no A-mainboard factors for {date}")
    frame = pd.concat(pieces, ignore_index=True)
    limits = load_limit_prices(target, target).rename(
        columns={"up_return": "signal_up_return"}
    )
    if not limits.empty:
        frame = frame.merge(
            limits[["date", "symbol", "signal_up_return"]],
            on=["date", "symbol"],
            how="left",
        )
        frame["is_st_signal"] = frame["signal_up_return"].between(0.03, 0.08)
        excluded = int(frame["is_st_signal"].fillna(False).sum())
        frame = frame[~frame["is_st_signal"].fillna(False)].copy()
        print(f"excluded current ST universe rows: {excluded}", flush=True)
    else:
        frame["signal_up_return"] = np.nan
        frame["is_st_signal"] = False
        print("warning: exact daily-limit cache unavailable; ST filter skipped", flush=True)
    return add_market_features(frame)


def attach_names(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        from roll.utils import get_normalized_stock_list

        names = get_normalized_stock_list()
        if names is not None and not names.empty:
            names = names.rename(columns={"code": "symbol"})
            return frame.merge(names[["symbol", "name"]], on="symbol", how="left")
    except Exception as exc:
        print(f"name lookup skipped: {exc}", flush=True)
    frame["name"] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    date = args.date or latest_a_date()
    report_dir = REPORTS / args.run_tag
    model_dir = MODELS / args.run_tag
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    features = json.loads((report_dir / "feature_names.json").read_text(encoding="utf-8"))
    frame = load_date(date)
    dmatrix = matrix(frame, features, label=False)

    boosters = {}
    for name, filename in {
        "global": "global_pretrain.ubj",
        "finetuned": "global_then_a_finetuned.ubj",
        "a_only": "a_only_control.ubj",
    }.items():
        booster = xgb.Booster()
        booster.load_model(model_dir / filename)
        boosters[name] = booster
    base_scores = {name: predict(model, dmatrix) for name, model in boosters.items()}
    chosen = summary["chosen_on_selection"]
    if chosen.startswith("blend_finetuned_"):
        weight = float(chosen.rsplit("_", 1)[-1])
        score = weight * base_scores["finetuned"] + (1 - weight) * base_scores["global"]
    else:
        score = base_scores[chosen]

    ranking = frame[["date", "symbol", "close"]].copy()
    ranking["score"] = score
    for name, values in base_scores.items():
        ranking[f"{name}_score"] = values
        ranking[f"{name}_rank"] = pd.Series(values, index=ranking.index).rank(
            method="first", ascending=False
        ).astype(int)
    ranking["rank"] = ranking["score"].rank(method="first", ascending=False).astype(int)
    ranking["rank_pct"] = ranking["score"].rank(pct=True, ascending=True)
    ranking["所属板块"] = np.where(
        ranking["symbol"].str.startswith("SH"), "沪市主板", "深市主板"
    )
    ranking = attach_names(ranking)
    ranking = ranking[
        [
            "rank",
            "symbol",
            "name",
            "所属板块",
            "date",
            "score",
            "rank_pct",
            "global_score",
            "global_rank",
            "finetuned_score",
            "finetuned_rank",
            "a_only_score",
            "a_only_rank",
            "close",
        ]
    ].sort_values("rank")
    output = (
        ROOT
        / ".qlibAssistant"
        / "analysis"
        / f"global_to_a_prediction_{pd.Timestamp(date):%Y%m%d}_{datetime.now():%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)
    ranking.head(args.topk).to_csv(output / "ranking.csv", index=False)
    model_union = ranking[
        (ranking["global_rank"] <= args.topk)
        | (ranking["finetuned_rank"] <= args.topk)
        | (ranking["a_only_rank"] <= args.topk)
    ].copy()
    model_union.sort_values(
        ["finetuned_rank", "a_only_rank", "global_rank"]
    ).to_csv(output / "model_rankings_union.csv", index=False)
    metadata = {
        "run_tag": args.run_tag,
        "signal_date": date,
        "chosen_on_selection": chosen,
        "candidate_rows": len(ranking),
        "output_rows": min(args.topk, len(ranking)),
        "model_union_rows": len(model_union),
        "score_semantics": "cross-sectional relative score, not absolute expected return",
        "expected_execution": "signal T close; buy T+1 close; sell T+2 close",
        "required_execution_guard": (
            "At T+1 near close, leave the slot in cash if the stock is locked "
            "at its exact up-limit; do not replace it with the next rank."
        ),
        "research_warning": (
            "The exact-limit one-year blind test did not pass the deployment "
            "threshold; this ranking is for research and paper trading only."
        ),
        "blind_test_metrics": summary["test_metrics"].get(chosen, {}),
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
    }
    atomic_json(output / "metadata.json", metadata)
    print(ranking.head(min(20, args.topk)).to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
