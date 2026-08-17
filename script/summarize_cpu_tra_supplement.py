#!/usr/bin/env python3
"""Summarize CPU TRA seed runs and export fold-level seed ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_intraday_factor_experiment import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_dir.glob("fold*_seed*_{selection,test}.parquet"))
    if not paths:
        paths = sorted(args.run_dir.glob("fold*_seed*_*.parquet"))
    groups: dict[tuple[str, str], list[tuple[int, pd.DataFrame]]] = {}
    individual_rows = []
    for path in paths:
        stem = path.stem
        fold = stem.split("_", 1)[0].removeprefix("fold")
        seed = int(stem.split("_seed", 1)[1].split("_", 1)[0])
        split = stem.rsplit("_", 1)[1]
        frame = load_prediction(path)
        groups.setdefault((fold, split), []).append((seed, frame))
        individual_rows.append({"fold": fold, "seed": seed, "split": split, **metrics(frame, args.cost_rate)})

    ensemble_rows, correlation_rows = [], []
    for (fold, split), runs in sorted(groups.items()):
        score_columns = []
        labels = []
        for seed, frame in sorted(runs):
            score_columns.append(frame["score"].rename(f"seed{seed}"))
            labels.append(frame["label"].rename(f"seed{seed}"))
        scores = pd.concat(score_columns, axis=1, join="inner")
        label_frame = pd.concat(labels, axis=1, join="inner")
        max_label_delta = float((label_frame.max(axis=1) - label_frame.min(axis=1)).abs().max())
        ensemble = pd.DataFrame({"score": scores.mean(axis=1), "label": label_frame.iloc[:, 0]}).dropna()
        output_path = args.run_dir / f"fold{fold}_ensemble_{split}.parquet"
        ensemble.reset_index().to_parquet(output_path, index=False)
        ensemble_rows.append({
            "fold": fold,
            "split": split,
            "seeds": len(runs),
            "max_label_delta": max_label_delta,
            **metrics(ensemble, args.cost_rate),
        })
        corr = scores.corr(method="spearman")
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        correlation_rows.append({
            "fold": fold,
            "split": split,
            "seed_pairs": int(len(upper)),
            "mean_seed_spearman": float(upper.mean()) if len(upper) else np.nan,
            "min_seed_spearman": float(upper.min()) if len(upper) else np.nan,
            "max_seed_spearman": float(upper.max()) if len(upper) else np.nan,
        })

    individual = pd.DataFrame(individual_rows).sort_values(["fold", "split", "seed"])
    ensemble = pd.DataFrame(ensemble_rows).sort_values(["fold", "split"])
    correlations = pd.DataFrame(correlation_rows).sort_values(["fold", "split"])
    individual.to_csv(args.run_dir / "individual_metrics.csv", index=False)
    ensemble.to_csv(args.run_dir / "ensemble_metrics.csv", index=False)
    correlations.to_csv(args.run_dir / "seed_correlations.csv", index=False)
    (args.run_dir / "summary_method.json").write_text(json.dumps({
        "ensemble": "arithmetic mean of seed scores on the common stock-day intersection",
        "metrics": "daily cross-sectional IC/Rank IC and simple one-day top-k diagnostics",
        "cost_rate": args.cost_rate,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Seed ensemble metrics")
    print(ensemble[["fold", "split", "seeds", "Rank_IC", "Rank_ICIR", "Top1_net_cumulative", "Top3_net_cumulative"]].to_string(index=False))
    print("\nSeed score correlations")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
