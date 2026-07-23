#!/usr/bin/env python3
"""在固定三Fold上评估选择性买入、缓冲卖出和相对优势替换策略。"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / ".qlibAssistant" / "mlruns"
INDEX_CACHE = PROJECT_ROOT / ".qlibAssistant" / "cache" / "tushare_index_daily.csv"
CONFIGS = [
    ("XGBoost", 60, "XGBoost"),
    ("XGBoost", 120, "XGBoost"),
    ("LightGBM", 84, "LightGBM"),
    ("CatBoost", 120, "CatBoost"),
    ("Linear", 84, "Linear"),
]


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def as_series(value, name):
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    value = value.copy()
    value.name = name
    return value


def daily_zscore(series):
    return series.groupby(level="datetime", group_keys=False).apply(
        lambda values: (values - values.mean()) / values.std()
    )


def load_component(row, split, config_name):
    artifacts = MLRUNS / str(int(row.experiment_id)) / str(row.recorder_id) / "artifacts"
    folder = artifacts / "valid_sig_analysis" if split == "validation" else artifacts
    prediction = as_series(load_pickle(folder / "pred.pkl"), config_name)
    label = as_series(load_pickle(folder / "label.pkl"), "label")
    return pd.concat([prediction, label], axis=1)


def ensemble_frame(detail, fold, split):
    components, metadata = [], []
    for model, months, family in CONFIGS:
        row = detail[(detail.fold == fold) & (detail.model_name == model) & (detail.train_months == months)].iloc[0]
        name = f"{model}_{months}"
        component = load_component(row, split, name)
        component[f"z_{name}"] = daily_zscore(component[name])
        components.append(component[[f"z_{name}"]])
        metadata.append((name, family, float(row["valid_Rank ICIR"])))
    first = detail[(detail.fold == fold) & (detail.model_name == CONFIGS[0][0]) & (detail.train_months == CONFIGS[0][1])].iloc[0]
    label_frame = load_component(first, split, "base")[["label"]]
    frame = pd.concat(components + [label_frame], axis=1).dropna()
    family_scores, family_weights = {}, {}
    for family in sorted({item[1] for item in metadata}):
        names = [item[0] for item in metadata if item[1] == family]
        family_scores[family] = frame[[f"z_{name}" for name in names]].mean(axis=1)
        family_weights[family] = np.mean([item[2] for item in metadata if item[1] == family])
    total = sum(family_weights.values())
    frame["score"] = sum(
        family_scores[family] * family_weights[family] / total for family in family_scores
    )
    return frame[["score", "label"]]


def validation_thresholds(frame, slots):
    rows = []
    for _, group in frame.groupby(level="datetime", sort=True):
        ranked = group.sort_values("score", ascending=False)
        rows.append(
            {
                "strength": ranked.head(slots)["score"].mean(),
                "replacement_gap": ranked.iloc[0]["score"] - ranked.iloc[slots - 1]["score"],
            }
        )
    daily = pd.DataFrame(rows)
    return {
        "strength_q50": daily.strength.quantile(0.5),
        "strength_q70": daily.strength.quantile(0.7),
        "gap_q50": daily.replacement_gap.quantile(0.5),
        "gap_q70": daily.replacement_gap.quantile(0.7),
    }


def target_holdings(ranked, holdings, rule, thresholds, slots):
    instruments = list(ranked.index.get_level_values("instrument"))
    score = dict(zip(instruments, ranked.score))
    rank = {instrument: idx + 1 for idx, instrument in enumerate(instruments)}
    strength = ranked.head(slots).score.mean()
    current = [instrument for instrument in holdings if instrument in score]

    if rule == "daily_top3":
        return instruments[:slots]
    if rule.startswith("buffer_top"):
        exit_rank = int(rule.replace("buffer_top", ""))
        kept = [instrument for instrument in current if rank[instrument] <= exit_rank]
        return kept + [instrument for instrument in instruments if instrument not in kept][: slots - len(kept)]
    if rule == "strength_q70_cash":
        return instruments[:slots] if strength >= thresholds["strength_q70"] else []
    if rule.startswith("margin_"):
        margin = thresholds["gap_q50" if rule == "margin_q50" else "gap_q70"]
        if not current:
            return instruments[:slots]
        result = current[:slots]
        while len(result) < slots:
            result.append(next(item for item in instruments if item not in result))
        for challenger in instruments:
            if challenger in result:
                continue
            weakest = min(result, key=lambda item: score[item])
            if score[challenger] - score[weakest] >= margin:
                result[result.index(weakest)] = challenger
            break
        return result
    if rule == "combined_q70_buffer10":
        result = [instrument for instrument in current if rank[instrument] <= 10]
        if strength >= thresholds["strength_q70"]:
            result += [item for item in instruments if item not in result][: slots - len(result)]
        result = result[:slots]
        if len(result) == slots:
            for challenger in instruments:
                if challenger in result:
                    continue
                weakest = min(result, key=lambda item: score[item])
                if score[challenger] - score[weakest] >= thresholds["gap_q70"]:
                    result[result.index(weakest)] = challenger
                break
        return result
    raise ValueError(rule)


def index_returns():
    cache = pd.read_csv(INDEX_CACHE, parse_dates=["datetime"])
    result = {}
    for name, group in cache.groupby("index"):
        close = group.drop_duplicates("datetime").set_index("datetime")["close"].sort_index()
        result[name] = close.shift(-2) / close.shift(-1) - 1
    return result


def simulate(frame, rule, thresholds, slots, cost_rate, benchmarks):
    holdings, rows = [], []
    for date, group in frame.groupby(level="datetime", sort=True):
        ranked = group.sort_values("score", ascending=False)
        target = target_holdings(ranked, holdings, rule, thresholds, slots)
        labels = ranked["label"].droplevel("datetime")
        invested = len(target) / slots
        gross = labels.reindex(target).mean() * invested if target else 0.0
        sells = len(set(holdings) - set(target))
        buys = len(set(target) - set(holdings))
        turnover = max(sells, buys) / slots
        net = gross - turnover * cost_rate
        rows.append({
            "datetime": date, "rule": rule, "holdings": ",".join(target),
            "holding_count": len(target), "gross_return": gross, "turnover": turnover,
            "cost": turnover * cost_rate, "net_return": net,
        })
        holdings = target
    daily = pd.DataFrame(rows).set_index("datetime")
    daily["gross_equity"] = (1 + daily.gross_return).cumprod()
    daily["net_equity"] = (1 + daily.net_return).cumprod()
    drawdown = daily.net_equity / daily.net_equity.cummax() - 1
    active = daily.holding_count > 0
    summary = {
        "rule": rule,
        "days": len(daily),
        "active_ratio": active.mean(),
        "average_holdings": daily.holding_count.mean(),
        "average_turnover": daily.turnover.mean(),
        "gross_cumulative": daily.gross_equity.iloc[-1] - 1,
        "net_cumulative": daily.net_equity.iloc[-1] - 1,
        "net_max_drawdown": drawdown.min(),
        "active_win_rate": (daily.loc[active, "net_return"] > 0).mean() if active.any() else np.nan,
    }
    for name, returns in benchmarks.items():
        aligned = returns.reindex(daily.index).dropna()
        if aligned.empty:
            continue
        last_date = aligned.index.max()
        benchmark_cumulative = (1 + aligned).prod() - 1
        strategy_cumulative = daily.loc[:last_date, "net_equity"].iloc[-1] - 1
        summary[f"{name}_last_date"] = last_date
        summary[f"{name}_cumulative"] = benchmark_cumulative
        summary[f"net_cumulative_at_{name}_end"] = strategy_cumulative
        summary[f"net_return_diff_vs_{name}"] = strategy_cumulative - benchmark_cumulative
        summary[f"net_excess_vs_{name}"] = (
            (1 + strategy_cumulative) / (1 + benchmark_cumulative) - 1
        )
    return summary, daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", default=".qlibAssistant/analysis/window_length_comparison_with_cost_20260720/window_fold_detail.csv")
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument("--slots", type=int, default=3)
    args = parser.parse_args()
    detail = pd.read_csv(PROJECT_ROOT / args.detail_csv)
    benchmarks = index_returns()
    rules = [
        "daily_top3", "buffer_top5", "buffer_top10", "buffer_top20",
        "strength_q70_cash", "margin_q50", "margin_q70", "combined_q70_buffer10",
    ]
    summaries, daily_outputs, threshold_rows = [], [], []
    for fold in ("fold1", "fold2", "fold3"):
        validation = ensemble_frame(detail, fold, "validation")
        test = ensemble_frame(detail, fold, "test")
        thresholds = validation_thresholds(validation, args.slots)
        threshold_rows.append({"fold": fold, **thresholds})
        for rule in rules:
            summary, daily = simulate(test, rule, thresholds, args.slots, args.cost_rate, benchmarks)
            summaries.append({"fold": fold, **summary})
            daily_outputs.append(daily.reset_index().assign(fold=fold))
    summary = pd.DataFrame(summaries)
    aggregate = summary.groupby("rule").agg(
        mean_net_cumulative=("net_cumulative", "mean"),
        worst_net_cumulative=("net_cumulative", "min"),
        mean_max_drawdown=("net_max_drawdown", "mean"),
        worst_max_drawdown=("net_max_drawdown", "min"),
        mean_turnover=("average_turnover", "mean"),
        mean_active_ratio=("active_ratio", "mean"),
        mean_holdings=("average_holdings", "mean"),
        mean_win_rate=("active_win_rate", "mean"),
        mean_csi1000=("CSI1000_cumulative", "mean"),
        mean_net_diff_vs_csi1000=("net_return_diff_vs_CSI1000", "mean"),
        mean_net_excess_vs_csi1000=("net_excess_vs_CSI1000", "mean"),
        worst_net_excess_vs_csi1000=("net_excess_vs_CSI1000", "min"),
        mean_csi300=("CSI300_cumulative", "mean"),
        mean_net_diff_vs_csi300=("net_return_diff_vs_CSI300", "mean"),
        mean_net_excess_vs_csi300=("net_excess_vs_CSI300", "mean"),
        worst_net_excess_vs_csi300=("net_excess_vs_CSI300", "min"),
    ).reset_index().sort_values("mean_net_cumulative", ascending=False)
    output = PROJECT_ROOT / ".qlibAssistant" / "analysis" / f"stateful_trading_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    summary.to_csv(output / "fold_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    pd.concat(daily_outputs, ignore_index=True).to_csv(output / "daily_positions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "validation_thresholds.csv", index=False)
    print(aggregate.to_string(index=False))
    print("\n逐Fold：")
    print(summary.pivot(index="rule", columns="fold", values="net_cumulative").to_string())
    print(f"报告目录: {output}")


if __name__ == "__main__":
    main()
