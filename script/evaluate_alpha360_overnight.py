#!/usr/bin/env python3
"""Backtest Alpha360 T+1 close -> T+2 open rankings on a 100k account.

This is deliberately separate from model selection.  It evaluates the frozen
Test predictions, excludes STAR/ChiNext, checks exact daily price limits using
raw prices, and never changes the checkpoint after seeing Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".qlibAssistant/remote_runs/alpha360_cross_stock_fold3_120m_260827/completed/run"
DEFAULT_RAW = Path("/Users/hmax/investment_data/cross_market_daily/raw/a")
DEFAULT_BASIC = ROOT / ".qlibAssistant/supplemental/csi1000_mainboard_full/tushare_daily/daily_basic"
DEFAULT_LIMITS = Path("/Users/hmax/investment_data/supplemental/stk_limit/stk_limit_all.parquet")
DEFAULT_INDEX = ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"
DEFAULT_DAILY = ROOT / ".qlibAssistant/cache/tushare_daily_ohlc.parquet"
SCORE = "close1_open2_expected_return"
ACTUAL = "close1_open2_actual_return"


def mainboard(code: str) -> bool:
    return str(code).startswith(("SH60", "SZ00"))


def drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def commission(value: float, rate: float, minimum: float) -> float:
    return max(minimum, abs(value) * rate) if value else 0.0


def load_prices(codes: set[str], raw_root: Path, basic_root: Path,
                daily_cache: Path | None = None) -> pd.DataFrame:
    rows = []
    direct = pd.DataFrame()
    if daily_cache is not None and daily_cache.exists():
        direct = pd.read_parquet(daily_cache, columns=["trade_date", "instrument", "open", "close"])
        direct["trade_date"] = pd.to_datetime(direct["trade_date"])
        direct = direct.loc[direct["instrument"].isin(codes)].rename(
            columns={"trade_date": "datetime", "open": "raw_open", "close": "raw_close"}
        )
        direct = direct[["datetime", "instrument", "raw_open", "raw_close"]]
    for code in sorted(codes):
        raw_path, basic_path = raw_root / f"{code}.parquet", basic_root / f"{code}.parquet"
        if not raw_path.exists() or not basic_path.exists():
            continue
        adjusted = pd.read_parquet(raw_path, columns=["datetime", "open", "close"])
        adjusted["datetime"] = pd.to_datetime(adjusted["datetime"])
        adjusted = adjusted.drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()
        basic = pd.read_parquet(basic_path, columns=["trade_date", "close"])
        basic["trade_date"] = pd.to_datetime(basic["trade_date"])
        raw_close = basic.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"].astype(float)
        frame = adjusted.join(raw_close.rename("raw_close"), how="inner")
        frame.index.name = "datetime"
        scale = frame["raw_close"] / frame["close"].replace(0, np.nan)
        frame["raw_open"] = frame["open"] * scale
        frame["instrument"] = code
        rows.append(frame[["instrument", "raw_open", "raw_close"]].reset_index())
    if not rows and direct.empty:
        raise RuntimeError("No raw price files for selected candidates")
    reconstructed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined = pd.concat([reconstructed, direct], ignore_index=True)
    # The direct Tushare daily endpoint is authoritative where both sources exist.
    combined = combined.drop_duplicates(["datetime", "instrument"], keep="last")
    return combined.set_index(["datetime", "instrument"]).sort_index()


def row_or_none(frame: pd.DataFrame, date: pd.Timestamp, code: str):
    try:
        row = frame.loc[(date, code)]
    except KeyError:
        return None
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def raw_trade(prices: pd.DataFrame, limits: pd.DataFrame, buy_date: pd.Timestamp,
              sell_date: pd.Timestamp, code: str) -> dict:
    buy_row, sell_row = row_or_none(prices, buy_date, code), row_or_none(prices, sell_date, code)
    if buy_row is None or sell_row is None:
        return {"buyable": False, "sellable": False, "buy": np.nan, "sell": np.nan,
                "raw_return": np.nan, "reason": "missing_raw_quote"}
    buy, sell = float(buy_row["raw_close"]), float(sell_row["raw_open"])
    try:
        buy_limit = limits.loc[(buy_date, code)]
        sell_limit = limits.loc[(sell_date, code)]
        if isinstance(buy_limit, pd.DataFrame):
            buy_limit = buy_limit.iloc[-1]
        if isinstance(sell_limit, pd.DataFrame):
            sell_limit = sell_limit.iloc[-1]
        buyable = buy < float(buy_limit["up_limit"]) - 0.005
        sellable = sell > float(sell_limit["down_limit"]) + 0.005
        reason = "ok" if buyable and sellable else ("buy_at_up_limit" if not buyable else "sell_at_down_limit_open")
    except KeyError:
        buyable = sellable = False
        reason = "missing_exact_limit"
    return {"buyable": bool(buyable), "sellable": bool(sellable), "buy": buy, "sell": sell,
            "raw_return": sell / buy - 1.0 if buy > 0 else np.nan, "reason": reason}


def benchmark_returns(index: pd.DataFrame, signal_dates: list[pd.Timestamp], calendar: pd.DatetimeIndex):
    position = {date: i for i, date in enumerate(calendar)}
    result = {}
    for name in ("CSI1000", "CSI300"):
        frame = index.loc[index["index"].eq(name)].drop_duplicates("datetime").set_index("datetime").sort_index()
        values = {}
        for signal in signal_dates:
            i = position[signal]
            buy_date, sell_date = calendar[i + 1], calendar[i + 2]
            if buy_date in frame.index and sell_date in frame.index:
                values[signal] = float(frame.loc[sell_date, "open"] / frame.loc[buy_date, "close"] - 1.0)
        result[name] = pd.Series(values, dtype=float)
    return result


def simulate(predictions: pd.DataFrame, prices: pd.DataFrame, limits: pd.DataFrame,
             calendar: pd.DatetimeIndex, topk: int, fallback: bool, slippage_bps: float,
             capital: float, rate: float, minimum: float) -> tuple[pd.DataFrame, dict]:
    positions = {date: i for i, date in enumerate(calendar)}
    equity = float(capital)
    rows = []
    for signal, group in predictions.groupby("datetime", sort=True):
        i = positions[signal]
        buy_date, sell_date = calendar[i + 1], calendar[i + 2]
        ranked = group.sort_values(SCORE, ascending=False)
        candidates = ranked.head(max(topk * 5, 50)).itertuples(index=False)
        chosen, blocked_buy = [], 0
        reasons = {"buy_at_up_limit": 0, "missing_raw_quote": 0, "missing_exact_limit": 0}
        for candidate in candidates:
            trade = raw_trade(prices, limits, buy_date, sell_date, candidate.instrument)
            if trade["buyable"]:
                chosen.append((candidate, trade))
            else:
                blocked_buy += 1
                reasons[trade["reason"]] = reasons.get(trade["reason"], 0) + 1
                if not fallback:
                    chosen.append((None, trade))
            if len(chosen) == topk:
                break
        while len(chosen) < topk:
            chosen.append((None, {"reason": "candidate_exhausted"}))

        start_equity = equity
        slot_budget = start_equity / topk
        fees = pnl = gross_pnl = 0.0
        invested = blocked_sell = raw_label_differences = lot_too_expensive = 0
        names = []
        for candidate, trade in chosen:
            if candidate is None:
                continue
            if not trade["sellable"]:
                blocked_sell += 1
                continue
            buy = trade["buy"] * (1.0 + slippage_bps / 10000.0)
            sell = trade["sell"] * (1.0 - slippage_bps / 10000.0)
            shares = int((slot_budget / buy) // 100) * 100
            while shares > 0 and shares * buy + commission(shares * buy, rate, minimum) > slot_budget:
                shares -= 100
            if shares <= 0:
                lot_too_expensive += 1
                continue
            buy_value, sell_value = shares * buy, shares * sell
            charge = commission(buy_value, rate, minimum) + commission(sell_value, rate, minimum)
            pnl += sell_value - buy_value - charge
            gross_pnl += sell_value - buy_value
            fees += charge
            invested += 1
            names.append(candidate.instrument)
            if np.isfinite(candidate.close1_open2_actual_return) and abs(
                trade["raw_return"] - candidate.close1_open2_actual_return
            ) > 0.005:
                raw_label_differences += 1
        equity += pnl
        rows.append({"datetime": signal, "buy_date": buy_date, "sell_date": sell_date,
                     "equity": equity, "net": equity / start_equity - 1.0,
                     "gross": gross_pnl / start_equity, "pnl": pnl, "gross_pnl": gross_pnl,
                     "fees": fees, "invested_slots": invested, "blocked_buy_candidates": blocked_buy,
                     "blocked_buy_at_up_limit": reasons["buy_at_up_limit"],
                     "blocked_buy_missing_quote": reasons["missing_raw_quote"],
                     "blocked_buy_missing_limit": reasons["missing_exact_limit"],
                     "blocked_sell_slots": blocked_sell, "lot_too_expensive_slots": lot_too_expensive,
                     "cash_ratio_at_buy": 1.0 - invested / topk,
                     "holdings": ";".join(names), "raw_label_difference_gt_50bps": raw_label_differences})
    daily = pd.DataFrame(rows).set_index("datetime")
    positive = daily["net"].clip(lower=0).sum()
    summary = {"topk": topk, "fallback": fallback, "slippage_bps_each_side": slippage_bps,
               "days": len(daily), "net_cumulative": float(daily["equity"].iloc[-1] / capital - 1.0),
               "executable_gross_cumulative": float((1.0 + daily["gross"]).prod() - 1.0),
               "net_win_rate": float((daily["net"] > 0).mean()), "net_mean_daily": float(daily["net"].mean()),
               "net_sharpe_rf0": float(daily["net"].mean() / daily["net"].std() * np.sqrt(252)),
               "net_max_drawdown": drawdown(daily["equity"] / capital), "total_fees": float(daily["fees"].sum()),
               "average_cash_ratio": float(daily["cash_ratio_at_buy"].mean()),
               "blocked_buy_candidates": int(daily["blocked_buy_candidates"].sum()),
               "blocked_buy_at_up_limit": int(daily["blocked_buy_at_up_limit"].sum()),
               "blocked_buy_missing_quote": int(daily["blocked_buy_missing_quote"].sum()),
               "blocked_buy_missing_limit": int(daily["blocked_buy_missing_limit"].sum()),
               "blocked_sell_slots": int(daily["blocked_sell_slots"].sum()),
               "lot_too_expensive_slots": int(daily["lot_too_expensive_slots"].sum()),
               "raw_label_difference_gt_50bps": int(daily["raw_label_difference_gt_50bps"].sum()),
               "best5_positive_share": float(daily["net"].nlargest(5).sum() / positive) if positive > 0 else np.nan,
               "strict_open_exit_fully_executable": bool(daily["blocked_sell_slots"].sum() == 0)}
    return daily, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--daily-basic-root", type=Path, default=DEFAULT_BASIC)
    parser.add_argument("--exact-limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--daily-ohlc-cache", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--commission-rate", type=float, default=0.000235)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.run / "test_predictions.csv", parse_dates=["datetime"])
    predictions = predictions.loc[predictions["instrument"].map(mainboard)].dropna(subset=[SCORE, ACTUAL])
    ranked = predictions.sort_values(["datetime", SCORE], ascending=[True, False])
    candidate_codes = set(ranked.groupby("datetime").head(max(max(args.topks) * 5, 50))["instrument"])
    prices = load_prices(candidate_codes, args.raw_root, args.daily_basic_root, args.daily_ohlc_cache)
    limits = pd.read_parquet(args.exact_limits, columns=["date", "symbol", "up_limit", "down_limit"])
    limits["date"] = pd.to_datetime(limits["date"])
    limits = limits.drop_duplicates(["date", "symbol"], keep="last").set_index(["date", "symbol"]).sort_index()
    index = pd.read_csv(args.index_cache, parse_dates=["datetime"])
    calendar = pd.DatetimeIndex(sorted(index.loc[index["index"].eq("CSI1000"), "datetime"].unique()))
    benchmarks = benchmark_returns(index, list(ranked["datetime"].drop_duplicates().sort_values()), calendar)
    output = args.output or ROOT / ".qlibAssistant/analysis" / f"alpha360_overnight_backtest_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for topk in args.topks:
        ideal = ranked.groupby("datetime").head(topk).groupby("datetime")[ACTUAL].mean()
        for fallback in (False, True):
            for slippage in args.slippage_bps:
                daily, summary = simulate(ranked, prices, limits, calendar, topk, fallback, slippage,
                                          args.capital, args.commission_rate, args.minimum_commission)
                summary.update({"ideal_gross_cumulative": float((1 + ideal).prod() - 1.0),
                                "ideal_gross_win_rate": float((ideal > 0).mean())})
                for name, benchmark in benchmarks.items():
                    aligned = benchmark.reindex(daily.index).dropna()
                    strategy = daily.loc[aligned.index, "net"]
                    summary[f"{name}_common_days"] = len(aligned)
                    summary[f"{name}_overnight_cumulative"] = float((1 + aligned).prod() - 1.0)
                    summary[f"net_cumulative_on_{name}_days"] = float((1 + strategy).prod() - 1.0)
                    summary[f"net_excess_vs_{name}"] = float((1 + strategy).prod() / (1 + aligned).prod() - 1.0)
                rows.append(summary)
                daily.reset_index().to_csv(output / f"top{topk}_{slippage:g}bps_fallback-{int(fallback)}_daily.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "method.json").write_text(json.dumps({
        "signal": SCORE, "holding": "T+1 close to T+2 open", "pool": "CSI1000 mainboard only",
        "capital": args.capital, "lot_size": 100, "commission_rate_each_side": args.commission_rate,
        "minimum_commission": args.minimum_commission, "stamp_tax": "omitted per user preference",
        "slippage_bps_each_side": args.slippage_bps, "exact_limits": str(args.exact_limits),
        "buy_rule": "T+1 raw close must be at least half a tick below exact up limit",
        "sell_rule": "T+2 raw open must be at least half a tick above exact down limit",
        "fallback_false": "blocked requested slots remain cash", "fallback_true": "scan lower ranks, max 50",
        "raw_price_reconstruction": "Tushare raw close plus adjusted OHLC same-day scale; cash dividends excluded",
        "direct_daily_cache": str(args.daily_ohlc_cache),
        "benchmark": "Tushare CSI1000/CSI300 close-to-next-open gross return; no index transaction costs",
        "test_is_not_pristine": "this historical Fold3 Test has previously been viewed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
