#!/usr/bin/env python3
"""Reusable board/cost variants for prediction frames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_star_market(index: pd.Index) -> np.ndarray:
    instruments = index.get_level_values("instrument").astype(str)
    return instruments.str.startswith(("SH688", "SH689"))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def evaluate_board_variants(
    frame: pd.DataFrame,
    cost_rate: float = 0.0015,
    topks: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, daily_outputs = [], []
    for universe_name, current in (
        ("all", frame),
        ("ex_star", frame.loc[~is_star_market(frame.index)]),
    ):
        previous = {topk: None for topk in topks}
        rows = []
        for date, group in current.groupby(level="datetime", sort=True):
            ranked = group.dropna(subset=["score", "label"]).sort_values("score", ascending=False)
            row = {"datetime": date, "universe_variant": universe_name, "stocks": len(ranked)}
            for topk in topks:
                selected = ranked.head(topk)
                members = set(selected.index.get_level_values("instrument"))
                turnover = 1.0 if previous[topk] is None else 1 - len(members & previous[topk]) / topk
                gross = float(selected["label"].mean())
                row[f"Top{topk}_gross_return"] = gross
                row[f"Top{topk}_turnover"] = turnover
                row[f"Top{topk}_net_return"] = gross - turnover * cost_rate
                previous[topk] = members
            rows.append(row)
        daily = pd.DataFrame(rows).set_index("datetime")
        for topk in topks:
            daily[f"Top{topk}_gross_equity"] = (1 + daily[f"Top{topk}_gross_return"]).cumprod()
            daily[f"Top{topk}_net_equity"] = (1 + daily[f"Top{topk}_net_return"]).cumprod()
            net = daily[f"Top{topk}_net_return"]
            sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
            best_n = min(5, len(net))
            summaries.append({
                "universe_variant": universe_name,
                "topk": topk,
                "days": len(daily),
                "gross_cumulative": daily[f"Top{topk}_gross_equity"].iloc[-1] - 1,
                "net_cumulative": daily[f"Top{topk}_net_equity"].iloc[-1] - 1,
                "average_turnover": daily[f"Top{topk}_turnover"].mean(),
                "win_rate": (daily[f"Top{topk}_gross_return"] > 0).mean(),
                "net_max_drawdown": max_drawdown(daily[f"Top{topk}_net_equity"]),
                "net_annualized_sharpe_rf0": sharpe,
                "best_5_days_net_sum": net.nlargest(best_n).sum(),
                "all_days_net_sum": net.sum(),
                "best_5_days_share_of_positive_sum": net.nlargest(best_n).sum() / net.clip(lower=0).sum()
                if net.clip(lower=0).sum() > 0 else np.nan,
            })
        daily_outputs.append(daily.reset_index())
    return pd.DataFrame(summaries), pd.concat(daily_outputs, ignore_index=True)

