#!/usr/bin/env python3
"""Nested walk-forward search for low-turnover, selective Top1/Top3 strategies."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_stateful_trading import PROJECT_ROOT, ensemble_frame, index_returns


def prepare_days(frame: pd.DataFrame):
    return [
        (date, group.sort_values("score", ascending=False))
        for date, group in frame.groupby(level="datetime", sort=True)
    ]


def daily_features(days, slots: int) -> pd.DataFrame:
    rows = []
    for date, ranked in days:
        rows.append({
            "datetime": date,
            "strength": ranked.head(slots).score.mean(),
            "breadth": (ranked.score > 0).mean(),
            "gap": ranked.iloc[0].score - ranked.iloc[min(slots - 1, len(ranked) - 1)].score,
        })
    return pd.DataFrame(rows).set_index("datetime")


def thresholds(validation_days, slots: int) -> dict[str, dict[float, float]]:
    features = daily_features(validation_days, slots)
    quantiles = (0.0, 0.5, 0.6, 0.7, 0.8)
    return {
        column: {q: float(features[column].quantile(q)) for q in quantiles}
        for column in ("strength", "breadth", "gap")
    }


def candidates():
    for slots, entry_q, breadth_q, exit_rank, gap_q, min_hold in itertools.product(
        (1, 3), (0.0, 0.5, 0.6, 0.7, 0.8), (0.0, 0.5, 0.7),
        (3, 5, 10, 20), (0.0, 0.5, 0.7), (1, 3, 5),
    ):
        if slots == 3 and exit_rank < 3:
            continue
        yield {
            "slots": slots, "entry_q": entry_q, "breadth_q": breadth_q,
            "exit_rank": exit_rank, "gap_q": gap_q, "min_hold": min_hold,
        }


def simulate(days, params, cuts, cost_rate, benchmarks):
    slots = params["slots"]
    holdings: list[str] = []
    ages: dict[str, int] = {}
    rows = []
    for date, ranked in days:
        instruments = list(ranked.index.get_level_values("instrument"))
        scores = dict(zip(instruments, ranked.score))
        ranks = {instrument: rank + 1 for rank, instrument in enumerate(instruments)}
        strength = ranked.head(slots).score.mean()
        breadth = (ranked.score > 0).mean()
        can_enter = (
            strength >= cuts["strength"][params["entry_q"]]
            and breadth >= cuts["breadth"][params["breadth_q"]]
        )
        current = [item for item in holdings if item in ranks]
        kept = [
            item for item in current
            if ages.get(item, 0) < params["min_hold"] or ranks[item] <= params["exit_rank"]
        ][:slots]
        target = list(kept)
        if can_enter:
            target += [item for item in instruments if item not in target][: slots - len(target)]
        if len(target) == slots and params["gap_q"] > 0:
            challenger = next((item for item in instruments if item not in target), None)
            if challenger is not None:
                weakest = min(target, key=scores.get)
                if (
                    ages.get(weakest, 0) >= params["min_hold"]
                    and scores[challenger] - scores[weakest] >= cuts["gap"][params["gap_q"]]
                ):
                    target[target.index(weakest)] = challenger
        labels = ranked.label.droplevel("datetime")
        gross = labels.reindex(target).mean() * (len(target) / slots) if target else 0.0
        turnover = max(len(set(holdings) - set(target)), len(set(target) - set(holdings))) / slots
        net = gross - turnover * cost_rate
        rows.append({"datetime": date, "net_return": net, "turnover": turnover, "holdings": len(target)})
        ages = {item: ages.get(item, 0) + 1 if item in holdings else 1 for item in target}
        holdings = target
    daily = pd.DataFrame(rows).set_index("datetime")
    equity = (1 + daily.net_return).cumprod()
    cumulative = equity.iloc[-1] - 1
    drawdown = (equity / equity.cummax() - 1).min()
    result = {
        **params,
        "net_cumulative": cumulative,
        "max_drawdown": drawdown,
        "average_turnover": daily.turnover.mean(),
        "active_ratio": (daily.holdings > 0).mean(),
        "average_holdings": daily.holdings.mean(),
    }
    for name, returns in benchmarks.items():
        aligned = returns.reindex(daily.index).dropna()
        if aligned.empty:
            continue
        last = aligned.index.max()
        strategy = (1 + daily.loc[:last, "net_return"]).prod() - 1
        benchmark = (1 + aligned).prod() - 1
        result[f"{name}_cumulative"] = benchmark
        result[f"net_excess_vs_{name}"] = (1 + strategy) / (1 + benchmark) - 1
        result[f"return_diff_vs_{name}"] = strategy - benchmark
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail-csv",
        default=".qlibAssistant/analysis/window_length_comparison_formal_20260720/window_fold_detail.csv",
    )
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument("--top-validation", type=int, default=20)
    args = parser.parse_args()
    detail = pd.read_csv(PROJECT_ROOT / args.detail_csv)
    benchmarks = index_returns()
    validation_rows, test_rows = [], []
    for fold in ("fold1", "fold2", "fold3"):
        validation = ensemble_frame(detail, fold, "validation")
        test = ensemble_frame(detail, fold, "test")
        validation_days = prepare_days(validation)
        test_days = prepare_days(test)
        cuts_by_slots = {slots: thresholds(validation_days, slots) for slots in (1, 3)}
        fold_results = []
        for params in candidates():
            cuts = cuts_by_slots[params["slots"]]
            result = simulate(validation_days, params, cuts, args.cost_rate, benchmarks)
            # The benchmark-relative result is primary; drawdown and turnover break close ties.
            result["selection_score"] = (
                result.get("net_excess_vs_CSI1000", result["net_cumulative"])
                + 0.25 * result["max_drawdown"]
                - 0.02 * result["average_turnover"]
            )
            result["fold"] = fold
            fold_results.append(result)
        ranked = pd.DataFrame(fold_results).sort_values("selection_score", ascending=False)
        ranked["validation_rank"] = np.arange(1, len(ranked) + 1)
        validation_rows.append(ranked)
        for _, selected in ranked.head(args.top_validation).iterrows():
            params = {
                key: selected[key]
                for key in ("slots", "entry_q", "breadth_q", "exit_rank", "gap_q", "min_hold")
            }
            params["slots"] = int(params["slots"])
            params["exit_rank"] = int(params["exit_rank"])
            params["min_hold"] = int(params["min_hold"])
            cuts = cuts_by_slots[params["slots"]]
            result = simulate(test_days, params, cuts, args.cost_rate, benchmarks)
            result.update({
                "fold": fold,
                "validation_rank": int(selected["validation_rank"]),
                "validation_selection_score": selected["selection_score"],
            })
            test_rows.append(result)
    validation_output = pd.concat(validation_rows, ignore_index=True)
    test_output = pd.DataFrame(test_rows)
    selected_test = test_output[test_output.validation_rank == 1].copy()
    output = PROJECT_ROOT / ".qlibAssistant" / "analysis" / f"strategy_optimization_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    validation_output.to_csv(output / "validation_grid.csv", index=False)
    test_output.to_csv(output / "top_validation_candidates_test.csv", index=False)
    selected_test.to_csv(output / "nested_selected_test.csv", index=False)
    aggregate = selected_test.select_dtypes(include=[np.number]).mean().to_frame("three_fold_mean")
    aggregate["worst_fold"] = selected_test.select_dtypes(include=[np.number]).min()
    aggregate.to_csv(output / "nested_selected_aggregate.csv")
    print(selected_test.to_string(index=False))
    print(f"report_dir={output}")


if __name__ == "__main__":
    main()
