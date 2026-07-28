#!/usr/bin/env python3
"""Backtest close-time hard risk filters on the mainboard Top20 ensemble."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from evaluate_mainboard_ensemble_sizes import build_ensemble
from report_mainboard_matrix import (
    COST_RATE,
    ROOT,
    _can_trade,
    _quote_row,
    benchmark_returns,
    max_drawdown,
)


RULES = {
    "baseline": {},
    "no_limit_close": {
        "min_trade_change": -0.095,
    },
    "recent_limit_guard": {
        "max_limit_down_10d": 1,
    },
    "drawdown_guard": {
        "min_drawdown_20d": -0.30,
    },
    "event_guard": {
        "max_limit_down_10d": 1,
        "min_drawdown_20d": -0.30,
        "min_low_vs_prev_close": -0.095,
    },
    "event_and_weak_guard": {
        "max_limit_down_10d": 1,
        "min_drawdown_20d": -0.30,
        "min_low_vs_prev_close": -0.095,
        "min_trade_change": -0.05,
        "min_close_open_return": -0.03,
    },
}


def risk_execution_features(start: str, end: str) -> pd.DataFrame:
    """Features known by the T+1 close where the strategy places its order."""

    expressions = [
        "Ref($change,-1)",
        "Ref($close,-1)",
        "Ref($close,-2)/Ref($close,-1)-1",
        "Ref(Sum($change<=-0.095,10),-1)",
        "Ref($close,-1)/Ref(Max($close,20),-1)-1",
        "Ref($close,-1)/Ref($open,-1)-1",
        "Ref($low,-1)/$close-1",
    ]
    data = D.features(
        D.instruments("all"),
        expressions,
        start_time=start,
        end_time=end,
        freq="day",
    )
    data.columns = [
        "trade_change",
        "trade_close",
        "holding_return",
        "limit_down_count_10d",
        "drawdown_20d",
        "close_open_return",
        "low_vs_prev_close",
    ]
    if set(data.index.names) == {"datetime", "instrument"}:
        data = data.reorder_levels(["datetime", "instrument"]).sort_index()
    return data


def passes_rule(row: pd.Series | None, rule: dict) -> bool:
    if row is None or pd.isna(row.get("trade_close")):
        return False
    checks = (
        ("min_trade_change", "trade_change", np.greater),
        ("max_limit_down_10d", "limit_down_count_10d", np.less_equal),
        ("min_drawdown_20d", "drawdown_20d", np.greater),
        ("min_low_vs_prev_close", "low_vs_prev_close", np.greater),
        ("min_close_open_return", "close_open_return", np.greater),
    )
    for parameter, column, comparator in checks:
        if parameter not in rule:
            continue
        value = row.get(column)
        if pd.isna(value) or not comparator(float(value), float(rule[parameter])):
            return False
    return True


def simulate(
    frame: pd.DataFrame,
    execution: pd.DataFrame,
    topk: int,
    rule: dict,
    fallback: bool,
) -> pd.DataFrame:
    holdings: list[str] = []
    rows = []
    for date, group in frame.groupby(level="datetime", sort=True):
        ranked_codes = list(
            group.sort_values("score", ascending=False)
            .index.get_level_values("instrument")
        )
        try:
            quote = execution.xs(date, level="datetime")
        except KeyError:
            quote = pd.DataFrame(columns=execution.columns)
            quote.index.name = "instrument"

        risk_blocked = 0
        limit_buy_blocked = 0
        if fallback:
            target = []
            for instrument in ranked_codes:
                row = _quote_row(quote, instrument)
                if not passes_rule(row, rule):
                    risk_blocked += 1
                    continue
                if instrument not in holdings and not _can_trade(
                    quote, instrument, "buy"
                ):
                    limit_buy_blocked += 1
                    continue
                target.append(instrument)
                if len(target) == topk:
                    break
        else:
            target = []
            for instrument in ranked_codes[:topk]:
                row = _quote_row(quote, instrument)
                if passes_rule(row, rule):
                    target.append(instrument)
                else:
                    risk_blocked += 1
        target_set = set(target)

        sold = 0
        bought = 0
        blocked_sell = 0
        retained = []
        for instrument in holdings:
            if instrument in target_set:
                retained.append(instrument)
            elif _can_trade(quote, instrument, "sell"):
                sold += 1
            else:
                retained.append(instrument)
                blocked_sell += 1
        holdings = retained

        for instrument in target:
            if instrument in holdings or len(holdings) >= topk:
                continue
            if _can_trade(quote, instrument, "buy"):
                holdings.append(instrument)
                bought += 1
            else:
                limit_buy_blocked += 1

        realized = []
        for instrument in holdings:
            row = _quote_row(quote, instrument)
            value = np.nan if row is None else row.get("holding_return")
            realized.append(0.0 if pd.isna(value) else float(value))
        gross = sum(realized) / topk
        turnover = (sold + bought) / (2 * topk)
        rows.append(
            {
                "datetime": date,
                "gross": gross,
                "turnover": turnover,
                "net": gross - turnover * COST_RATE,
                "holding_count": len(holdings),
                "cash_ratio": (topk - len(holdings)) / topk,
                "risk_blocked": risk_blocked,
                "limit_buy_blocked": limit_buy_blocked,
                "limit_sell_blocked": blocked_sell,
            }
        )
    return pd.DataFrame(rows).set_index("datetime")


def summarize(
    daily: pd.DataFrame,
    benchmarks: dict[str, pd.Series],
) -> dict:
    net_equity = (1 + daily["net"]).cumprod()
    result = {
        "test_days": len(daily),
        "win_rate": float((daily["gross"] > 0).mean()),
        "gross_cumulative": float((1 + daily["gross"]).prod() - 1),
        "net_cumulative": float(net_equity.iloc[-1] - 1),
        "net_max_drawdown": max_drawdown(net_equity),
        "net_sharpe_rf0": (
            float(daily["net"].mean() / daily["net"].std() * np.sqrt(252))
            if daily["net"].std() > 0
            else np.nan
        ),
        "average_turnover": float(daily["turnover"].mean()),
        "average_cash_ratio": float(daily["cash_ratio"].mean()),
        "risk_blocked_orders": int(daily["risk_blocked"].sum()),
        "risk_blocked_days": int((daily["risk_blocked"] > 0).sum()),
        "limit_buy_blocked_orders": int(daily["limit_buy_blocked"].sum()),
        "limit_sell_blocked_orders": int(daily["limit_sell_blocked"].sum()),
    }
    for name, values in benchmarks.items():
        aligned = values.reindex(daily.index).dropna()
        strategy = daily.loc[aligned.index, "net"]
        benchmark_equity = (1 + aligned).cumprod()
        strategy_equity = (1 + strategy).cumprod()
        result[f"{name}_cumulative"] = float(benchmark_equity.iloc[-1] - 1)
        result[f"net_excess_vs_{name}"] = float(
            strategy_equity.iloc[-1] / benchmark_equity.iloc[-1] - 1
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail-csv",
        default=(
            ".qlibAssistant/analysis/mainboard_matrix_report_latest/"
            "mainboard_60_recorder_detail.csv"
        ),
    )
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--topks", default="1,3")
    parser.add_argument(
        "--output-dir",
        default=(
            ".qlibAssistant/analysis/hard_risk_filter_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    args = parser.parse_args()

    qlib.init(
        provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"),
        region=REG_CN,
    )
    detail = pd.read_csv(
        ROOT / args.detail_csv,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    start = detail["test_start"].min()
    end = detail["test_end"].max()
    execution = risk_execution_features(start, end)
    benchmarks = benchmark_returns(start, end)

    rows = []
    daily_outputs = []
    topks = [int(value) for value in args.topks.split(",") if value]
    for fold, fold_detail in detail.groupby("fold", sort=True):
        ranked = fold_detail.sort_values(
            ["valid_Rank_ICIR", "valid_Rank_IC", "model", "train_months"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        selected = ranked.head(args.ensemble_size)
        score, label, _ = build_ensemble(selected, "validation_rank_icir")
        frame = pd.concat([score.rename("score"), label.rename("label")], axis=1)
        frame = frame.dropna()
        for topk in topks:
            for rule_name, rule in RULES.items():
                for fallback in (False, True):
                    daily = simulate(frame, execution, topk, rule, fallback)
                    summary = summarize(daily, benchmarks)
                    rows.append(
                        {
                            "fold": fold,
                            "ensemble_size": args.ensemble_size,
                            "topk": topk,
                            "rule": rule_name,
                            "fallback": fallback,
                            **summary,
                        }
                    )
                    daily_outputs.append(
                        daily.reset_index().assign(
                            fold=fold,
                            topk=topk,
                            rule=rule_name,
                            fallback=fallback,
                        )
                    )

    by_fold = pd.DataFrame(rows)
    numeric = [
        column
        for column in by_fold.select_dtypes(include=np.number).columns
        if column not in {"ensemble_size", "topk"}
    ]
    average = (
        by_fold.groupby(["ensemble_size", "topk", "rule", "fallback"])[numeric]
        .mean()
        .reset_index()
    )
    worst = (
        by_fold.groupby(["ensemble_size", "topk", "rule", "fallback"])[numeric]
        .min()
        .reset_index()
    )

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    by_fold.to_csv(output / "hard_filter_by_fold.csv", index=False)
    average.to_csv(output / "hard_filter_three_fold_average.csv", index=False)
    worst.to_csv(output / "hard_filter_worst_fold.csv", index=False)
    pd.concat(daily_outputs, ignore_index=True).to_csv(
        output / "hard_filter_daily.csv", index=False
    )
    config = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "ensemble_size": args.ensemble_size,
        "topks": topks,
        "cost_rate": COST_RATE,
        "rules": RULES,
        "timing": (
            "Signal at T; all filter fields are observed by the T+1 close; "
            "trade at that close and hold until T+2 close."
        ),
        "fallback_false": "Invalid requested slots remain cash.",
        "fallback_true": "Invalid candidates are replaced by lower-ranked valid names.",
    }
    (output / "report_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    display = average[
        [
            "topk",
            "rule",
            "fallback",
            "net_cumulative",
            "net_max_drawdown",
            "net_sharpe_rf0",
            "average_cash_ratio",
            "risk_blocked_days",
            "CSI1000_cumulative",
            "net_excess_vs_CSI1000",
            "CSI300_cumulative",
            "net_excess_vs_CSI300",
        ]
    ].sort_values(["topk", "fallback", "net_cumulative"], ascending=[True, True, False])
    print(display.to_string(index=False))
    print(f"\noutput_dir={output}")


if __name__ == "__main__":
    main()
