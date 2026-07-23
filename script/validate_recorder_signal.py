#!/usr/bin/env python3
"""Validate a saved recorder's historical signal once its future close is available."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--recorder-id", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument(
        "--label-expression",
        default="Ref($close,-2)/Ref($close,-1)-1",
        help="Default is the model's T+1-close to T+2-close target.",
    )
    parser.add_argument("--inspect-instruments", nargs="*")
    args = parser.parse_args()
    path = ROOT / ".qlibAssistant" / "mlruns" / args.experiment_id / args.recorder_id / "artifacts/pred.pkl"
    with path.open("rb") as stream:
        prediction = pickle.load(stream)
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.iloc[:, 0]
    score = prediction.xs(pd.Timestamp(args.signal_date), level="datetime").rename("score")
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN, kernels=1)
    label = D.features(
        D.instruments("csi1000"),
        [args.label_expression],
        start_time=args.signal_date,
        end_time=args.signal_date,
    ).iloc[:, 0].droplevel("datetime").rename("realized_return")
    result = pd.concat([score, label], axis=1).dropna().sort_values("score", ascending=False)
    print(f"stocks={len(result)} rank_ic={result.score.corr(result.realized_return, method='spearman'):.6f}")
    for topk in (1, 3, 10):
        selected = result.head(topk)
        print(
            f"top{topk}: mean_return={selected.realized_return.mean():.6%} "
            f"win_rate={(selected.realized_return > 0).mean():.2%}"
        )
    print(f"universe_mean={result.realized_return.mean():.6%}")
    print(result.head(10).to_string())
    if args.inspect_instruments:
        ranks = result.score.rank(ascending=False, method="min").rename("rank")
        print("INSPECT")
        print(result.join(ranks).reindex(args.inspect_instruments).to_string())


if __name__ == "__main__":
    main()
