#!/usr/bin/env python3
"""Exploratory, predeclared cross-model voting and persistence rules."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_variants import max_drawdown


ROOT = Path(__file__).resolve().parents[1]


def load_series(path: Path, name: str) -> pd.Series:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value.rename(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append", required=True, metavar="EXP_ID,REC_ID,NAME")
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    args = parser.parse_args()
    if len(args.component) < 2:
        raise SystemExit("至少需要两个component")

    scores, labels, names = [], [], []
    for spec in args.component:
        exp_id, rec_id, name = spec.split(",", 2)
        artifacts = ROOT / ".qlibAssistant" / "mlruns" / exp_id / rec_id / "artifacts"
        scores.append(load_series(artifacts / "pred.pkl", name))
        labels.append(load_series(artifacts / "label.pkl", "label"))
        names.append(name)
    frame = pd.concat(scores + [labels[0]], axis=1, join="inner").dropna()
    for name in names:
        frame[f"rank_{name}"] = frame[name].groupby(level="datetime").rank(ascending=False, method="min")
        frame[f"z_{name}"] = frame[name].groupby(level="datetime").transform(
            lambda x: (x - x.mean()) / x.std()
        )
    frame["ensemble_z"] = frame[[f"z_{name}" for name in names]].mean(axis=1)
    rank_cols = [f"rank_{name}" for name in names]

    rows = []
    for universe in ("all", "ex_star"):
        previous_consensus20: set[str] = set()
        for date, raw_group in frame.groupby(level="datetime", sort=True):
            if universe == "ex_star":
                instruments_raw = raw_group.index.get_level_values("instrument").astype(str)
                group = raw_group.loc[~instruments_raw.str.startswith(("SH688", "SH689"))]
            else:
                group = raw_group
            instruments = group.index.get_level_values("instrument")
            candidates = {
                "daily_ensemble_top1": pd.Series(True, index=group.index),
                "unanimous_top10": (group[rank_cols] <= 10).all(axis=1),
                "unanimous_top20": (group[rank_cols] <= 20).all(axis=1),
                "tight_vote_top10": ((group[rank_cols].mean(axis=1) <= 10) & ((group[rank_cols].max(axis=1) - group[rank_cols].min(axis=1)) <= 5)),
            }
            current20 = set(instruments[candidates["unanimous_top20"].to_numpy()])
            candidates["unanimous_top20_persist_2d"] = candidates["unanimous_top20"] & instruments.isin(previous_consensus20)
            for rule, mask in candidates.items():
                eligible = group.loc[mask]
                chosen = eligible.sort_values("ensemble_z", ascending=False).head(1)
                rows.append({
                    "datetime": date,
                    "universe_variant": universe,
                    "rule": rule,
                    "instrument": chosen.index.get_level_values("instrument")[0] if len(chosen) else "CASH",
                    "gross_return": float(chosen["label"].iloc[0]) if len(chosen) else 0.0,
                })
            previous_consensus20 = current20
    choices = pd.DataFrame(rows)

    summaries, details = [], []
    for universe, universe_choices in choices.groupby("universe_variant"):
        for rule, group in universe_choices.groupby("rule"):
            daily = group.sort_values("datetime").copy()
            previous = "CASH"
            turnover = []
            for instrument in daily["instrument"]:
                turnover.append(float(instrument != previous))
                previous = instrument
            daily["turnover"] = turnover
            daily["net_return"] = daily["gross_return"] - daily["turnover"] * args.cost_rate
            daily["net_equity"] = (1 + daily["net_return"]).cumprod()
            active = daily["instrument"] != "CASH"
            summaries.append({
                "universe_variant": universe,
                "rule": rule,
                "days": len(daily),
                "active_ratio": active.mean(),
                "active_days": int(active.sum()),
                "trade_win_rate": (daily.loc[active, "gross_return"] > 0).mean() if active.any() else np.nan,
                "gross_cumulative": (1 + daily["gross_return"]).prod() - 1,
                "net_cumulative": daily["net_equity"].iloc[-1] - 1,
                "average_turnover": daily["turnover"].mean(),
                "net_max_drawdown": max_drawdown(daily["net_equity"]),
            })
            details.append(daily.assign(universe_variant=universe))
    output = ROOT / ".qlibAssistant" / "analysis" / f"consensus_strategy_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output / "daily.csv", index=False)
    frame.reset_index().to_csv(output / "model_scores.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"report_dir={output}")


if __name__ == "__main__":
    main()
