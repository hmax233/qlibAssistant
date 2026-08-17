#!/usr/bin/env python3
"""Export XGB240 and Fixed Ensemble scores on a reference stock-day universe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "script"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

from evaluate_source_hard_filters import (  # noqa: E402
    DEFAULT_WINDOW_DETAIL,
    build_fixed_sources,
    build_xgb240_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fold", default="fold3", choices=("fold1", "fold2", "fold3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = pd.read_parquet(args.reference)
    reference["datetime"] = pd.to_datetime(reference["datetime"])
    reference = reference.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()
    detail = pd.read_csv(DEFAULT_WINDOW_DETAIL)
    sources = build_xgb240_sources("mainboard") + build_fixed_sources(detail, "mainboard")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    aligned_indexes = []
    for source in sources:
        if source["fold"] != args.fold:
            continue
        frame = pd.concat([source["score"].rename("score"), reference["label"]], axis=1, join="inner").dropna()
        aligned_indexes.append(frame.index)
        name = "xgb240" if source["source"] == "XGBoost240" else "fixed_ensemble"
        output = args.output_dir / f"{name}_{args.fold}_aligned.parquet"
        frame.reset_index().to_parquet(output, index=False)
        dates = frame.index.get_level_values("datetime")
        rows.append({
            "source": name,
            "fold": args.fold,
            "start": dates.min(),
            "end": dates.max(),
            "days": dates.nunique(),
            "stock_days": len(frame),
            "output": str(output),
        })
    common_index = reference.dropna().index
    for index in aligned_indexes:
        common_index = common_index.intersection(index)
    reference_aligned = reference.loc[common_index].dropna().sort_index()
    reference_output = args.output_dir / "reference_aligned.parquet"
    reference_aligned.reset_index().to_parquet(reference_output, index=False)
    rows.append({
        "source": "reference",
        "fold": args.fold,
        "start": reference_aligned.index.get_level_values("datetime").min(),
        "end": reference_aligned.index.get_level_values("datetime").max(),
        "days": reference_aligned.index.get_level_values("datetime").nunique(),
        "stock_days": len(reference_aligned),
        "output": str(reference_output),
    })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "alignment_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
