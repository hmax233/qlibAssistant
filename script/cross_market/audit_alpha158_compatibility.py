#!/usr/bin/env python3
"""Compare locally generated A factors with Qlib's official Alpha158 expressions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data import D

from common import FACTORS, REPORTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SH600000")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-28")
    parser.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "alpha158_compatibility_sample.csv",
    )
    args = parser.parse_args()
    qlib.init(provider_uri=str(Path(args.provider).expanduser()), region=REG_CN)
    alpha = Alpha158.__new__(Alpha158)
    expressions, names = alpha.get_feature_config()
    official = D.features(
        [args.symbol],
        expressions,
        start_time=args.start,
        end_time=args.end,
        freq="day",
    )
    official.columns = names
    official = official.reset_index().rename(columns={"datetime": "date"})
    ours = pd.read_parquet(FACTORS / "a" / f"{args.symbol}.parquet")
    merged = ours.merge(official, on="date", suffixes=("_ours", "_official"))
    rows = []
    for name in names:
        left, right = merged[f"{name}_ours"], merged[f"{name}_official"]
        valid = left.notna() & right.notna()
        rows.append(
            {
                "factor": name,
                "rows": int(valid.sum()),
                "correlation": (
                    float(left[valid].corr(right[valid]))
                    if valid.sum() > 2
                    else np.nan
                ),
                "median_absolute_error": (
                    float((left[valid] - right[valid]).abs().median())
                    if valid.any()
                    else np.nan
                ),
            }
        )
    report = pd.DataFrame(rows).sort_values("correlation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(
        f"rows={len(merged)} factors={len(report)} "
        f"corr>=0.999={(report.correlation >= 0.999).sum()} "
        f"corr>=0.99={(report.correlation >= 0.99).sum()}"
    )
    print(report.head(20).to_string(index=False))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
