#!/usr/bin/env python3
"""Analyze whether recent saved Top1 picks recovered after the one-day label."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".qlibAssistant" / "analysis"
RAW = Path("/Users/hmax/investment_data/cross_market_daily/raw/a")
ONE_SIDE_COST = 0.000235
HORIZONS = (1, 2, 3, 5)


def latest_review() -> Path:
    folders = sorted(ANALYSIS.glob("recent_live_review_*"))
    if not folders:
        raise FileNotFoundError("No recent_live_review_* output exists")
    return folders[-1]


def price_frame(symbol: str) -> pd.DataFrame:
    frame = pd.read_parquet(RAW / f"{symbol}.parquet")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.drop_duplicates("datetime").set_index("datetime").sort_index()


def net_round_trip(gross: float) -> float:
    return gross - 2 * ONE_SIDE_COST


def staged_average_return(
    entry: float,
    closes: list[float],
    horizon: int,
    trigger: float = -0.01,
) -> tuple[float, bool]:
    """50% initial buy; add the reserved 50% after a <=-1% first close."""

    first_return = closes[0] / entry - 1
    add = first_return <= trigger
    invested = 0.5
    shares = 0.5 / entry
    buy_cost = 0.5 * ONE_SIDE_COST
    if add:
        shares += 0.5 / closes[0]
        invested += 0.5
        buy_cost += 0.5 * ONE_SIDE_COST
    final_value = shares * closes[horizon - 1] + (1 - invested)
    sell_cost = shares * closes[horizon - 1] * ONE_SIDE_COST
    return final_value - 1 - buy_cost - sell_cost, add


def main() -> None:
    review = latest_review()
    detail = pd.read_csv(
        review / "prediction_realization_detail.csv",
        parse_dates=["signal_date", "buy_date", "sell_date"],
    )
    detail = detail[detail["buy_date"].notna()].copy()
    rows = []
    for item in detail.itertuples(index=False):
        prices = price_frame(item.instrument)
        if item.buy_date not in prices.index:
            continue
        entry_position = prices.index.get_loc(item.buy_date)
        entry = float(prices.loc[item.buy_date, "close"])
        future = prices.iloc[entry_position + 1 :]
        row = {
            "source": item.source,
            "signal_date": item.signal_date,
            "buy_date": item.buy_date,
            "instrument": item.instrument,
            "name": item.name,
            "entry_close": entry,
            "available_future_days": len(future),
        }
        if len(future):
            row["max_close_return_available"] = (
                float(future["close"].max()) / entry - 1
            )
            row["min_close_return_available"] = (
                float(future["close"].min()) / entry - 1
            )
            row["max_intraday_favorable_available"] = (
                float(future["high"].max()) / entry - 1
            )
            row["max_intraday_adverse_available"] = (
                float(future["low"].min()) / entry - 1
            )
        closes = future["close"].astype(float).tolist()
        for horizon in HORIZONS:
            if len(closes) < horizon:
                row[f"hold_{horizon}d_net"] = np.nan
                row[f"staged_add_{horizon}d_net"] = np.nan
                row[f"staged_add_{horizon}d_triggered"] = np.nan
                continue
            row[f"hold_{horizon}d_net"] = net_round_trip(
                closes[horizon - 1] / entry - 1
            )
            staged, triggered = staged_average_return(entry, closes, horizon)
            row[f"staged_add_{horizon}d_net"] = staged
            row[f"staged_add_{horizon}d_triggered"] = triggered
        rows.append(row)
    paths = pd.DataFrame(rows)

    summaries = []
    for source, group in paths.groupby("source"):
        row = {"source": source}
        for horizon in HORIZONS:
            values = group[f"hold_{horizon}d_net"].dropna()
            staged = group[f"staged_add_{horizon}d_net"].dropna()
            row[f"hold_{horizon}d_n"] = len(values)
            row[f"hold_{horizon}d_win_rate"] = (
                float((values > 0).mean()) if len(values) else np.nan
            )
            row[f"hold_{horizon}d_cumulative"] = (
                float((1 + values).prod() - 1) if len(values) else np.nan
            )
            row[f"staged_add_{horizon}d_cumulative"] = (
                float((1 + staged).prod() - 1) if len(staged) else np.nan
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries)

    output = ANALYSIS / f"recent_holding_paths_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True)
    paths.to_csv(output / "holding_path_detail.csv", index=False)
    summary.to_csv(output / "holding_horizon_summary.csv", index=False)

    focus = paths[
        paths["instrument"].isin(["SZ001309", "SH603063", "SH603306", "SZ002149"])
    ].copy()
    display = focus[
        [
            "source",
            "signal_date",
            "instrument",
            "name",
            "available_future_days",
            "hold_1d_net",
            "hold_2d_net",
            "hold_3d_net",
            "hold_5d_net",
            "max_close_return_available",
            "min_close_return_available",
        ]
    ].copy()
    for column in display.columns[5:]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    metadata = {
        "source_review": str(review),
        "entry": "T+1 close",
        "holding_horizon": (
            "Hold Nd means exit at the Nth trading close after the entry close."
        ),
        "staged_add": (
            "Start with 50%; if first close is <=-1%, add the reserved 50%; "
            "exit at the selected horizon."
        ),
        "warning": (
            "This is a small post-hoc diagnostic, not evidence that averaging "
            "down is a validated strategy."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# 近期Top1延长持有与一次补仓观察

{display.to_markdown(index=False)}

持有Nd表示从推荐买入收盘开始，在之后第N个交易日收盘卖出。补仓方案仅用于
诊断：初始投入50%，若第一个持有日收盘跌幅不小于1%，再投入剩余50%。
样本极少且规则是在看到近期走势后提出，不能作为已验证交易策略。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(paths.to_string(index=False))
    print("\nSummary\n", summary.to_string(index=False))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
