#!/usr/bin/env python3
"""Test market-breadth timing gates on XGB240 and FixedArchitecture.

The stock signal is timestamped T, traded at the T+1 close, and earns the
T+1-close to T+2-close return.  Two market-information timings are reported:

``signal_date``
    Use the breadth forecast made after the T close.  This is operationally
    available, but it forecasts breadth on the buy day (T+1), not the whole
    holding return.

``execution_close``
    Use the breadth forecast made with T+1 close data.  It forecasts T+2 and
    is better aligned with the holding return, but trading at that same close
    is an idealized assumption unless a near-close/intraday pipeline exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN

from evaluate_hard_risk_filters import (
    RULES,
    passes_rule,
    risk_execution_features,
    summarize,
)
from evaluate_source_hard_filters import (
    DEFAULT_WINDOW_DETAIL,
    build_fixed_sources,
    build_xgb240_sources,
)
from report_mainboard_matrix import (
    COST_RATE,
    ROOT,
    _can_trade,
    _quote_row,
    benchmark_returns,
)


BREADTH_REPORTS = {
    "fold1": (
        ROOT
        / ".qlibAssistant/analysis/market_breadth_fold1_260727_night_025918"
        / "ensemble_test_predictions.csv"
    ),
    "fold2": (
        ROOT
        / ".qlibAssistant/analysis/market_breadth_fold2_260727_night_025940"
        / "ensemble_test_predictions.csv"
    ),
    "fold3": (
        ROOT
        / ".qlibAssistant/analysis/market_breadth_fold3_260727_v2_015952"
        / "ensemble_test_predictions.csv"
    ),
}


def load_breadth_series(
    path: Path,
    ensemble_size: int,
    timing: str,
) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    values = (
        frame.loc[frame["ensemble_size"].eq(ensemble_size)]
        .drop_duplicates("datetime")
        .set_index("datetime")["predicted_up_ratio"]
        .sort_index()
        .astype(float)
    )
    if timing == "signal_date":
        return values
    if timing == "execution_close":
        # A stock signal dated T uses the market forecast produced one trading
        # day later, at T+1 close.  Keep the stock-signal date as the index.
        return values.shift(-1)
    raise ValueError(f"Unknown market timing: {timing}")


def simulate_with_market_gate(
    frame: pd.DataFrame,
    execution: pd.DataFrame,
    topk: int,
    rule: dict,
    market_forecast: pd.Series | None,
    threshold: float | None,
) -> pd.DataFrame:
    """Run stateful execution with optional risk and market gates.

    Rejected slots remain cash.  The simulator never substitutes a lower
    ranked stock.  Existing positions that cannot be sold at their lower
    price limit remain in the portfolio.
    """

    holdings: list[str] = []
    rows: list[dict] = []
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

        market_value = np.nan
        market_blocked = 0
        market_missing = 0
        market_allowed = True
        if threshold is not None:
            if market_forecast is None:
                market_allowed = False
                market_missing = 1
            else:
                market_value = market_forecast.get(pd.Timestamp(date), np.nan)
                if pd.isna(market_value):
                    market_allowed = False
                    market_missing = 1
                elif float(market_value) < threshold:
                    market_allowed = False
                    market_blocked = 1

        risk_blocked = 0
        target: list[str] = []
        if market_allowed:
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
        limit_buy_blocked = 0
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
                "market_forecast": market_value,
                "market_blocked": market_blocked,
                "market_missing": market_missing,
                "limit_buy_blocked": limit_buy_blocked,
                "limit_sell_blocked": blocked_sell,
            }
        )
    return pd.DataFrame(rows).set_index("datetime")


def add_market_summary(result: dict, daily: pd.DataFrame) -> dict:
    invested = daily["holding_count"] > 0
    result.update(
        {
            "invested_days": int(invested.sum()),
            "invested_day_ratio": float(invested.mean()),
            "invested_day_win_rate": (
                float((daily.loc[invested, "gross"] > 0).mean())
                if invested.any()
                else np.nan
            ),
            "market_blocked_days": int(daily["market_blocked"].sum()),
            "market_missing_days": int(daily["market_missing"].sum()),
            "market_allowed_ratio": float(
                1
                - (daily["market_blocked"] + daily["market_missing"])
                .clip(upper=1)
                .mean()
            ),
            "mean_market_forecast": float(daily["market_forecast"].mean()),
        }
    )
    return result


def aggregate(
    by_fold: pd.DataFrame,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [
        column
        for column in by_fold.select_dtypes(include=np.number).columns
        if column not in {"topk", "market_ensemble_size", "market_threshold"}
    ]
    average = by_fold.groupby(group_columns, dropna=False)[numeric].mean().reset_index()
    worst = by_fold.groupby(group_columns, dropna=False)[numeric].min().reset_index()
    return average, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-detail", default=str(DEFAULT_WINDOW_DETAIL))
    parser.add_argument("--topks", default="1,3")
    parser.add_argument("--market-sizes", default="2,4,6,8,20")
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55")
    parser.add_argument(
        "--output-dir",
        default=(
            ".qlibAssistant/analysis/market_timing_filter_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    args = parser.parse_args()

    qlib.init(
        provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"),
        region=REG_CN,
    )
    detail = pd.read_csv(
        args.window_detail,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    sources = build_xgb240_sources() + build_fixed_sources(detail)
    starts = [
        item["score"].index.get_level_values("datetime").min() for item in sources
    ]
    ends = [
        item["score"].index.get_level_values("datetime").max() for item in sources
    ]
    start = str(min(starts).date())
    end = str(max(ends).date())
    execution = risk_execution_features(start, end)
    benchmarks = benchmark_returns(start, end)

    topks = [int(value) for value in args.topks.split(",") if value]
    market_sizes = [int(value) for value in args.market_sizes.split(",") if value]
    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    timings = ("signal_date", "execution_close")
    risk_modes = ("baseline", "event_guard")

    rows: list[dict] = []
    daily_rows: list[pd.DataFrame] = []
    for item in sources:
        frame = pd.concat(
            [item["score"].rename("score"), item["label"].rename("label")],
            axis=1,
        ).dropna()
        fold = item["fold"]

        # Common baselines are written once per source/fold/topk/risk mode.
        for topk in topks:
            for risk_mode in risk_modes:
                daily = simulate_with_market_gate(
                    frame,
                    execution,
                    topk,
                    RULES[risk_mode],
                    market_forecast=None,
                    threshold=None,
                )
                summary = add_market_summary(
                    summarize(daily, benchmarks),
                    daily,
                )
                rows.append(
                    {
                        "source": item["source"],
                        "fold": fold,
                        "topk": topk,
                        "risk_rule": risk_mode,
                        "market_timing": "none",
                        "market_ensemble_size": 0,
                        "market_threshold": np.nan,
                        **summary,
                    }
                )
                daily_rows.append(
                    daily.reset_index().assign(
                        source=item["source"],
                        fold=fold,
                        topk=topk,
                        risk_rule=risk_mode,
                        market_timing="none",
                        market_ensemble_size=0,
                        market_threshold=np.nan,
                    )
                )

        for market_size in market_sizes:
            for timing in timings:
                market_forecast = load_breadth_series(
                    BREADTH_REPORTS[fold],
                    market_size,
                    timing,
                )
                for threshold in thresholds:
                    for topk in topks:
                        for risk_mode in risk_modes:
                            daily = simulate_with_market_gate(
                                frame,
                                execution,
                                topk,
                                RULES[risk_mode],
                                market_forecast,
                                threshold,
                            )
                            summary = add_market_summary(
                                summarize(daily, benchmarks),
                                daily,
                            )
                            rows.append(
                                {
                                    "source": item["source"],
                                    "fold": fold,
                                    "topk": topk,
                                    "risk_rule": risk_mode,
                                    "market_timing": timing,
                                    "market_ensemble_size": market_size,
                                    "market_threshold": threshold,
                                    **summary,
                                }
                            )
                            daily_rows.append(
                                daily.reset_index().assign(
                                    source=item["source"],
                                    fold=fold,
                                    topk=topk,
                                    risk_rule=risk_mode,
                                    market_timing=timing,
                                    market_ensemble_size=market_size,
                                    market_threshold=threshold,
                                )
                            )

    by_fold = pd.DataFrame(rows)
    groups = [
        "source",
        "topk",
        "risk_rule",
        "market_timing",
        "market_ensemble_size",
        "market_threshold",
    ]
    average, worst = aggregate(by_fold, groups)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    by_fold.to_csv(output / "market_timing_by_fold.csv", index=False)
    average.to_csv(output / "market_timing_three_fold_average.csv", index=False)
    worst.to_csv(output / "market_timing_worst_fold.csv", index=False)
    primary_mask = (
        average["topk"].eq(1)
        & average["risk_rule"].eq("event_guard")
        & (
            average["market_timing"].eq("none")
            | (
                average["market_timing"].eq("signal_date")
                & average["market_ensemble_size"].eq(2)
            )
        )
    )
    average.loc[primary_mask].sort_values(
        ["source", "market_timing", "market_threshold"]
    ).to_csv(output / "primary_top1_comparison.csv", index=False)
    sensitivity_mask = (
        average["topk"].eq(1)
        & average["risk_rule"].eq("event_guard")
        & average["market_timing"].eq("signal_date")
    )
    average.loc[sensitivity_mask].sort_values(
        ["source", "market_threshold", "market_ensemble_size"]
    ).to_csv(output / "market_size_sensitivity.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(
        output / "market_timing_daily.csv",
        index=False,
    )
    config = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "stock_sources": ["XGBoost240", "FixedArchitecture"],
        "topks": topks,
        "risk_rules": {name: RULES[name] for name in risk_modes},
        "market_ensemble_sizes": market_sizes,
        "market_thresholds": thresholds,
        "breadth_reports": {
            fold: str(path.relative_to(ROOT))
            for fold, path in BREADTH_REPORTS.items()
        },
        "board_filter": "STAR Market and ChiNext removed before ranking.",
        "fallback": False,
        "cost_rate": COST_RATE,
        "stock_timing": "Signal T, trade T+1 close, hold to T+2 close.",
        "market_timing": {
            "signal_date": (
                "Use the forecast produced after T close for T+1 market breadth; "
                "operationally available to the existing pipeline."
            ),
            "execution_close": (
                "Use the forecast produced with T+1 close data for T+2 breadth; "
                "aligned with the holding period but same-close execution is "
                "idealized and requires a near-close/intraday pipeline."
            ),
        },
        "research_status": (
            "Thresholds and model sizes are swept on existing test folds. "
            "Results are exploratory and are not untouched confirmatory tests."
        ),
    }
    (output / "report_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    primary = average[
        average["topk"].eq(1)
        & average["market_ensemble_size"].isin([0, 2])
        & (
            average["market_timing"].eq("none")
            | average["market_threshold"].isin(thresholds)
        )
    ][
        [
            "source",
            "risk_rule",
            "market_timing",
            "market_ensemble_size",
            "market_threshold",
            "net_cumulative",
            "net_max_drawdown",
            "net_sharpe_rf0",
            "average_cash_ratio",
            "market_allowed_ratio",
            "CSI1000_cumulative",
            "net_excess_vs_CSI1000",
            "CSI300_cumulative",
            "net_excess_vs_CSI300",
        ]
    ].sort_values(
        [
            "source",
            "risk_rule",
            "market_timing",
            "market_threshold",
        ]
    )
    print(primary.to_string(index=False))
    print(f"\noutput_dir={output}")


if __name__ == "__main__":
    main()
