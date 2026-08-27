#!/usr/bin/env python3
"""Strict four-horizon Alpha360 account backtest and uncertainty-rule selection.

Trading rules are selected on Selection-valid only.  The frozen Test split is
then evaluated without changing the chosen rule.  Raw Tushare OHLC, exact
daily limits, 100-share lots, minimum commissions, overlapping holding periods
and delayed exits from limit-down are handled explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from evaluate_alpha360_overnight import commission, mainboard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".qlibAssistant/remote_runs/alpha360_cross_stock_fold3_120m_260827/completed/run"
DEFAULT_DAILY = ROOT / ".qlibAssistant/cache/tushare_daily_ohlc.parquet"
DEFAULT_LIMITS = Path("/Users/hmax/investment_data/supplemental/stk_limit/stk_limit_all.parquet")
DEFAULT_INDEX = ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"

HORIZONS = {
    "open1_close2": {"entry": "open", "exit": "close", "sleeves": 2},
    "close1_open2": {"entry": "close", "exit": "open", "sleeves": 1},
    "open1_open2": {"entry": "open", "exit": "open", "sleeves": 1},
    "close1_close2": {"entry": "close", "exit": "close", "sleeves": 1},
}

RULES = (
    "mean_all", "probability_all", "risk_adjusted_all", "lcb05_all", "lcb10_all",
    "mean_p50", "mean_p55", "mean_p60", "risk_adjusted_p50",
    "lcb05_positive", "lcb10_positive", "mean_sigma_q30_positive",
    "mean_sigma_q50_positive", "mean_sigma_q70_positive",
    "gate_mean_p50", "gate_mean_p52", "gate_mean_p53", "gate_mean_p55", "gate_mean_p60",
    "gate_mean_sigma_q30", "gate_mean_sigma_q50", "gate_mean_sigma_q70", "gate_lcb05_positive",
    "gate_lcb10_positive",
)


@dataclass
class Position:
    instrument: str
    shares: int
    cash: float
    entry_value: float
    buy_fee: float
    entry_date: pd.Timestamp
    scheduled_exit: pd.Timestamp
    exit_phase: str
    last_price: float


@dataclass
class Slot:
    cash: float
    position: Position | None = None


def drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0


def load_inputs(args):
    prices = pd.read_parquet(args.daily_ohlc_cache)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.drop_duplicates(["trade_date", "instrument"], keep="last")
    prices = prices.set_index(["trade_date", "instrument"]).sort_index()
    limits = pd.read_parquet(args.exact_limits, columns=["date", "symbol", "up_limit", "down_limit"])
    limits["date"] = pd.to_datetime(limits["date"])
    limits = limits.drop_duplicates(["date", "symbol"], keep="last")
    limits = limits.set_index(["date", "symbol"]).sort_index()
    index = pd.read_csv(args.index_cache, parse_dates=["datetime"])
    calendar = pd.DatetimeIndex(sorted(index.loc[index["index"].eq("CSI1000"), "datetime"].unique()))
    return prices, limits, index, calendar


def benchmark_cumulative(index: pd.DataFrame, signal_dates, calendar: pd.DatetimeIndex,
                         horizon: str, benchmark: str) -> float:
    specification = HORIZONS[horizon]
    frame = index.loc[index["index"].eq(benchmark)].drop_duplicates("datetime").set_index("datetime")
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    sleeve_values = np.ones(specification["sleeves"], dtype=float)
    for number, signal in enumerate(sorted(pd.Timestamp(date) for date in signal_dates)):
        i = calendar_pos[signal]
        entry_date, exit_date = calendar[i + 1], calendar[i + 2]
        if entry_date not in frame.index or exit_date not in frame.index:
            continue
        realized = float(
            frame.loc[exit_date, specification["exit"]]
            / frame.loc[entry_date, specification["entry"]] - 1.0
        )
        sleeve_values[number % specification["sleeves"]] *= 1.0 + realized
    return float(sleeve_values.mean() - 1.0)


def attach_benchmarks(results: pd.DataFrame, predictions: pd.DataFrame,
                      index: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    results = results.copy()
    dates = predictions["datetime"].drop_duplicates()
    for benchmark in ("CSI1000", "CSI300"):
        values = {
            horizon: benchmark_cumulative(index, dates, calendar, horizon, benchmark)
            for horizon in HORIZONS
        }
        results[f"{benchmark}_gross_cumulative"] = results["horizon"].map(values)
        results[f"net_excess_vs_{benchmark}"] = (
            (1.0 + results["net_cumulative"])
            / (1.0 + results[f"{benchmark}_gross_cumulative"]) - 1.0
        )
    return results


def label_alignment(split: str, predictions: pd.DataFrame, prices: pd.DataFrame,
                    calendar: pd.DatetimeIndex) -> pd.DataFrame:
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    rows = []
    for horizon, specification in HORIZONS.items():
        top = (
            predictions.sort_values(["datetime", f"{horizon}_expected_return"], ascending=[True, False])
            .groupby("datetime").head(1)
        )
        raw, labels = [], []
        for candidate in top.itertuples(index=False):
            i = calendar_pos[candidate.datetime]
            entry = quote(prices, calendar[i + 1], candidate.instrument, specification["entry"])
            exit_price = quote(prices, calendar[i + 2], candidate.instrument, specification["exit"])
            if entry is None or exit_price is None:
                continue
            raw.append(exit_price / entry - 1.0)
            labels.append(getattr(candidate, f"{horizon}_actual_return"))
        raw_array, label_array = np.asarray(raw), np.asarray(labels)
        finite = np.isfinite(raw_array) & np.isfinite(label_array)
        compared_raw, compared_label = raw_array[finite], label_array[finite]
        rows.append({
            "split": split, "horizon": horizon, "rows": len(raw_array),
            "raw_cumulative": float(np.prod(1.0 + raw_array) - 1.0),
            "raw_up_rate": float((raw_array > 0).mean()),
            "compared_rows": len(compared_raw),
            "label_cumulative": float(np.prod(1.0 + compared_label) - 1.0),
            "raw_cumulative_on_compared_rows": float(np.prod(1.0 + compared_raw) - 1.0),
            "correlation": float(np.corrcoef(compared_raw, compared_label)[0, 1]),
            "max_absolute_difference": float(np.max(np.abs(compared_raw - compared_label))),
            "differences_over_50bps": int((np.abs(compared_raw - compared_label) > 0.005).sum()),
        })
    return pd.DataFrame(rows)


def prepare_rule(group: pd.DataFrame, horizon: str, rule: str) -> pd.DataFrame:
    mu = group[f"{horizon}_expected_return"].astype(float)
    sigma = group[f"{horizon}_return_std"].astype(float).clip(lower=1e-8)
    probability = group[f"{horizon}_probability_positive"].astype(float)
    eligible = pd.Series(True, index=group.index)
    if rule.startswith("probability"):
        score = probability
    elif rule.startswith("risk_adjusted"):
        score = mu / sigma
    elif rule.startswith("lcb05") or rule == "gate_lcb05_positive":
        score = mu - 0.5 * sigma
    elif rule.startswith("lcb10") or rule == "gate_lcb10_positive":
        score = mu - sigma
    else:
        score = mu
    if rule.endswith("_p50") or rule == "mean_p50":
        eligible &= probability >= 0.50
    elif rule == "mean_p55":
        eligible &= probability >= 0.55
    elif rule == "mean_p60":
        eligible &= probability >= 0.60
    if rule == "lcb05_positive":
        eligible &= (mu - 0.5 * sigma) > 0
    elif rule == "lcb10_positive":
        eligible &= (mu - sigma) > 0
    elif rule == "mean_sigma_q30_positive":
        eligible &= (sigma <= sigma.quantile(0.30)) & (mu > 0)
    elif rule == "mean_sigma_q50_positive":
        eligible &= (sigma <= sigma.quantile(0.50)) & (mu > 0)
    elif rule == "mean_sigma_q70_positive":
        eligible &= (sigma <= sigma.quantile(0.70)) & (mu > 0)
    result = group.loc[eligible].copy()
    result["strategy_score"] = score.loc[eligible]
    result = result.sort_values("strategy_score", ascending=False)
    if result.empty:
        return result
    top = result.iloc[0]
    gate = True
    if rule == "gate_mean_p50":
        gate = top[f"{horizon}_probability_positive"] >= 0.50
    elif rule == "gate_mean_p52":
        gate = top[f"{horizon}_probability_positive"] >= 0.52
    elif rule == "gate_mean_p53":
        gate = top[f"{horizon}_probability_positive"] >= 0.53
    elif rule == "gate_mean_p55":
        gate = top[f"{horizon}_probability_positive"] >= 0.55
    elif rule == "gate_mean_p60":
        gate = top[f"{horizon}_probability_positive"] >= 0.60
    elif rule == "gate_mean_sigma_q30":
        gate = (top[f"{horizon}_return_std"] <= sigma.quantile(0.30)) and (top[f"{horizon}_expected_return"] > 0)
    elif rule == "gate_mean_sigma_q50":
        gate = (top[f"{horizon}_return_std"] <= sigma.quantile(0.50)) and (top[f"{horizon}_expected_return"] > 0)
    elif rule == "gate_mean_sigma_q70":
        gate = (top[f"{horizon}_return_std"] <= sigma.quantile(0.70)) and (top[f"{horizon}_expected_return"] > 0)
    elif rule == "gate_lcb05_positive":
        gate = top["strategy_score"] > 0
    elif rule == "gate_lcb10_positive":
        gate = top["strategy_score"] > 0
    return result if gate else result.iloc[0:0]


def quote(frame: pd.DataFrame, date: pd.Timestamp, instrument: str, field: str) -> float | None:
    try:
        value = frame.loc[(date, instrument), field]
    except KeyError:
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value) if np.isfinite(value) else None


def limit(frame: pd.DataFrame, date: pd.Timestamp, instrument: str, field: str) -> float | None:
    try:
        value = frame.loc[(date, instrument), field]
    except KeyError:
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value) if np.isfinite(value) else None


def mark_slot(slot: Slot, date: pd.Timestamp, phase: str, prices: pd.DataFrame) -> float:
    if slot.position is None:
        return slot.cash
    position = slot.position
    marked = quote(prices, date, position.instrument, phase)
    if marked is not None:
        position.last_price = marked
    return position.cash + position.shares * position.last_price


def attempt_exit(
    slot: Slot,
    date: pd.Timestamp,
    phase: str,
    prices: pd.DataFrame,
    limits: pd.DataFrame,
    calendar_pos: dict[pd.Timestamp, int],
    slippage_bps: float,
    rate: float,
    minimum: float,
    trade_returns: list[float],
    counters: dict[str, int],
) -> bool:
    """Try one exit using only information available at this event."""

    position = slot.position
    if position is None:
        return False
    if phase != position.exit_phase or date < position.scheduled_exit:
        return False
    raw_price = quote(prices, date, position.instrument, phase)
    down = limit(limits, date, position.instrument, "down_limit")
    if raw_price is None or down is None:
        counters["blocked_sell_missing_attempts"] += 1
        return False
    position.last_price = raw_price
    if raw_price <= down + 0.005:
        counters["blocked_sell_down_limit_attempts"] += 1
        return False
    sell_price = raw_price * (1.0 - slippage_bps / 10000.0)
    sell_fee = commission(position.shares * sell_price, rate, minimum)
    proceeds = position.shares * sell_price - sell_fee
    final_cash = position.cash + proceeds
    starting_cash = position.cash + position.entry_value + position.buy_fee
    trade_returns.append(final_cash / starting_cash - 1.0)
    delayed = calendar_pos[date] - calendar_pos[position.scheduled_exit]
    counters["delayed_exit_trades"] += int(delayed > 0)
    counters["completed_exit_trades"] += 1
    slot.cash = final_cash
    slot.position = None
    return True


def simulate(predictions: pd.DataFrame, prices: pd.DataFrame, limits: pd.DataFrame,
             calendar: pd.DatetimeIndex, horizon: str, rule: str, topk: int,
             fallback: bool, slippage_bps: float, capital: float, rate: float,
             minimum: float) -> tuple[pd.DataFrame, dict]:
    specification = HORIZONS[horizon]
    sleeves = specification["sleeves"]
    entry_phase, exit_phase = specification["entry"], specification["exit"]
    slot_groups = [[Slot(capital / (sleeves * topk)) for _ in range(topk)] for _ in range(sleeves)]
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    rows, trade_returns = [], []
    counters = {
        "blocked_buy_up_limit": 0, "blocked_buy_missing": 0, "too_expensive": 0,
        "delayed_exit_trades": 0, "skipped_busy_slots": 0, "filtered_cash_slots": 0,
        "blocked_sell_missing_attempts": 0, "blocked_sell_down_limit_attempts": 0,
        "completed_exit_trades": 0, "unresolved_exit": 0,
    }
    grouped = list(predictions.groupby("datetime", sort=True))
    last_processed_event: int | None = None

    def advance_to(event_number: int) -> None:
        """Advance chronologically; never inspect an event before it occurs."""

        nonlocal last_processed_event
        if last_processed_event is None:
            last_processed_event = event_number - 1
        for number in range(last_processed_event + 1, event_number + 1):
            event_date = calendar[number // 2]
            event_phase = "open" if number % 2 == 0 else "close"
            for slot_group in slot_groups:
                for slot in slot_group:
                    attempt_exit(
                        slot, event_date, event_phase, prices, limits, calendar_pos,
                        slippage_bps, rate, minimum, trade_returns, counters,
                    )
        last_processed_event = event_number

    for signal_number, (signal, group) in enumerate(grouped):
        signal = pd.Timestamp(signal)
        i = calendar_pos[signal]
        entry_date, scheduled_exit = calendar[i + 1], calendar[i + 2]
        entry_event_number = 2 * calendar_pos[entry_date] + (0 if entry_phase == "open" else 1)
        advance_to(entry_event_number)
        assigned = slot_groups[signal_number % sleeves]
        ranked = prepare_rule(group, horizon, rule)
        candidates = list(ranked.head(max(50, topk * 5)).itertuples(index=False))
        used: set[str] = set()
        cursor = 0
        entries_this_signal = 0
        for rank_slot, slot in enumerate(assigned):
            if slot.position is not None:
                counters["skipped_busy_slots"] += 1
                continue
            attempts = candidates[cursor:] if fallback else candidates[rank_slot:rank_slot + 1]
            bought = False
            for candidate in attempts:
                cursor += 1
                if candidate.instrument in used:
                    continue
                raw_buy = quote(prices, entry_date, candidate.instrument, entry_phase)
                upper = limit(limits, entry_date, candidate.instrument, "up_limit")
                if raw_buy is None or upper is None:
                    counters["blocked_buy_missing"] += 1
                    if not fallback:
                        break
                    continue
                if raw_buy >= upper - 0.005:
                    counters["blocked_buy_up_limit"] += 1
                    if not fallback:
                        break
                    continue
                buy_price = raw_buy * (1.0 + slippage_bps / 10000.0)
                shares = int((slot.cash / buy_price) // 100) * 100
                while shares > 0 and shares * buy_price + commission(shares * buy_price, rate, minimum) > slot.cash:
                    shares -= 100
                if shares <= 0:
                    counters["too_expensive"] += 1
                    if not fallback:
                        break
                    continue
                buy_value = shares * buy_price
                buy_fee = commission(buy_value, rate, minimum)
                slot.position = Position(
                    instrument=candidate.instrument, shares=shares,
                    cash=slot.cash - buy_value - buy_fee, entry_value=buy_value,
                    buy_fee=buy_fee, entry_date=entry_date, scheduled_exit=scheduled_exit,
                    exit_phase=exit_phase, last_price=raw_buy,
                )
                used.add(candidate.instrument)
                bought = True
                entries_this_signal += 1
                break
            if not bought and not attempts:
                counters["filtered_cash_slots"] += 1
        total_equity = sum(mark_slot(slot, entry_date, entry_phase, prices)
                           for slot_group in slot_groups for slot in slot_group)
        rows.append({
            "datetime": signal, "entry_date": entry_date, "scheduled_exit": scheduled_exit,
            "equity_mark": total_equity,
            "entries": entries_this_signal,
            "active_positions": sum(slot.position is not None for sg in slot_groups for slot in sg),
        })

    # After the final signal, continue the same chronological state machine
    # through the available calendar. Positions still blocked at the data end
    # remain open and are marked, never retrospectively rejected at entry.
    if last_processed_event is not None:
        advance_to(2 * len(calendar) - 1)
    counters["unresolved_exit"] = sum(
        slot.position is not None for slot_group in slot_groups for slot in slot_group
    )
    final_date = calendar[-1]
    final_equity = sum(
        mark_slot(slot, final_date, "close", prices)
        for slot_group in slot_groups for slot in slot_group
    )
    daily = pd.DataFrame(rows).set_index("datetime")
    if len(daily):
        daily.iloc[-1, daily.columns.get_loc("equity_mark")] = final_equity
    active_signals = int((daily["entries"] > 0).sum())
    summary = {
        "horizon": horizon, "rule": rule, "topk": topk, "fallback": fallback,
        "slippage_bps_each_side": slippage_bps, "signal_days": len(daily),
        "active_signal_days": active_signals, "completed_trades": len(trade_returns),
        "net_cumulative": final_equity / capital - 1.0,
        "trade_win_rate": float((np.asarray(trade_returns) > 0).mean()) if trade_returns else np.nan,
        "mean_trade_return": float(np.mean(trade_returns)) if trade_returns else np.nan,
        "max_drawdown_marked": drawdown(daily["equity_mark"]),
        "final_equity": final_equity, **counters,
    }
    return daily, summary


def baseline_matrix(split, predictions, prices, limits, calendar, args, output):
    rows = []
    for horizon in HORIZONS:
        for topk in args.topks:
            for slippage in args.slippage_bps:
                daily, summary = simulate(
                    predictions, prices, limits, calendar, horizon, "mean_all", topk,
                    args.fallback, slippage, args.capital, args.commission_rate,
                    args.minimum_commission,
                )
                summary["split"] = split
                rows.append(summary)
                daily.reset_index().to_csv(
                    output / f"{split}_{horizon}_top{topk}_{slippage:g}bps_daily.csv", index=False
                )
    return pd.DataFrame(rows)


def uncertainty_grid(split, predictions, prices, limits, calendar, args):
    rows = []
    for horizon in HORIZONS:
        for rule in RULES:
            for slippage in args.slippage_bps:
                _, summary = simulate(
                    predictions, prices, limits, calendar, horizon, rule, 1, True,
                    slippage, args.capital, args.commission_rate, args.minimum_commission,
                )
                summary["split"] = split
                rows.append(summary)
    return pd.DataFrame(rows)


def select_rules(selection: pd.DataFrame, selection_slippage: float, minimum_active_days: int):
    eligible = selection.loc[
        selection["slippage_bps_each_side"].eq(selection_slippage)
        & selection["active_signal_days"].ge(minimum_active_days)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No uncertainty rule passed the minimum active-day constraint")
    chosen = (
        eligible.sort_values(
            ["horizon", "net_cumulative", "max_drawdown_marked"],
            ascending=[True, False, False],
        ).groupby("horizon", as_index=False).head(1)
    )
    return chosen[["horizon", "rule", "net_cumulative", "active_signal_days",
                   "trade_win_rate", "max_drawdown_marked"]].rename(columns={
                       "net_cumulative": "selection_net_cumulative",
                       "active_signal_days": "selection_active_signal_days",
                       "trade_win_rate": "selection_trade_win_rate",
                       "max_drawdown_marked": "selection_max_drawdown",
                   })


def evaluate_chosen(chosen, predictions, prices, limits, calendar, args, output, label):
    rows = []
    for selected in chosen.itertuples(index=False):
        for slippage in args.slippage_bps:
            daily, summary = simulate(
                predictions, prices, limits, calendar, selected.horizon, selected.rule,
                1, True, slippage, args.capital, args.commission_rate,
                args.minimum_commission,
            )
            summary["split"] = "test"
            rows.append(summary)
            daily.reset_index().to_csv(
                output / f"test_{label}_{selected.horizon}_{selected.rule}_{slippage:g}bps_daily.csv",
                index=False,
            )
    return pd.DataFrame(rows)


def select_robust_rules(valid: pd.DataFrame, selection: pd.DataFrame,
                        selection_slippage: float, minimum_active_days: int):
    columns = ["horizon", "rule", "net_cumulative", "active_signal_days",
               "trade_win_rate", "max_drawdown_marked"]
    left = valid.loc[valid["slippage_bps_each_side"].eq(selection_slippage), columns].rename(columns={
        "net_cumulative": "valid_net", "active_signal_days": "valid_active_days",
        "trade_win_rate": "valid_win_rate", "max_drawdown_marked": "valid_max_drawdown",
    })
    right = selection.loc[selection["slippage_bps_each_side"].eq(selection_slippage), columns].rename(columns={
        "net_cumulative": "selection_net", "active_signal_days": "selection_active_days",
        "trade_win_rate": "selection_win_rate", "max_drawdown_marked": "selection_max_drawdown",
    })
    merged = left.merge(right, on=["horizon", "rule"])
    merged = merged.loc[
        merged["valid_active_days"].ge(minimum_active_days)
        & merged["selection_active_days"].ge(minimum_active_days)
    ].copy()
    merged["worst_pretest_net"] = merged[["valid_net", "selection_net"]].min(axis=1)
    merged["mean_pretest_net"] = merged[["valid_net", "selection_net"]].mean(axis=1)
    return (
        merged.sort_values(
            ["horizon", "worst_pretest_net", "mean_pretest_net"],
            ascending=[True, False, False],
        ).groupby("horizon", as_index=False).head(1)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--daily-ohlc-cache", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--exact-limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--commission-rate", type=float, default=0.000235)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", nargs="+", type=float, default=[0.0, 5.0])
    parser.add_argument("--selection-slippage-bps", type=float, default=5.0)
    parser.add_argument("--minimum-active-days", type=int, default=30)
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--diagnostic-test-grid", action="store_true",
        help="Explicitly opt into a full Test rule grid; never use it for selection.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    prices, limits, index, calendar = load_inputs(args)
    output = args.output or ROOT / ".qlibAssistant/analysis" / f"alpha360_four_horizon_uncertainty_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    frames = {}
    for split in ("valid", "selection_valid", "test"):
        frame = pd.read_csv(args.run / f"{split}_predictions.csv", parse_dates=["datetime"])
        frame = frame.loc[frame["instrument"].map(mainboard)].copy()
        frames[split] = frame

    baseline = baseline_matrix("test", frames["test"], prices, limits, calendar, args, output)
    baseline = attach_benchmarks(baseline, frames["test"], index, calendar)
    valid_grid = uncertainty_grid("valid", frames["valid"], prices, limits, calendar, args)
    valid_grid = attach_benchmarks(valid_grid, frames["valid"], index, calendar)
    selection_grid = uncertainty_grid("selection_valid", frames["selection_valid"], prices, limits, calendar, args)
    selection_grid = attach_benchmarks(selection_grid, frames["selection_valid"], index, calendar)
    chosen = select_rules(selection_grid, args.selection_slippage_bps, args.minimum_active_days)
    chosen_test = evaluate_chosen(chosen, frames["test"], prices, limits, calendar, args, output, "selection")
    chosen_test = attach_benchmarks(chosen_test, frames["test"], index, calendar)
    robust_chosen = select_robust_rules(
        valid_grid, selection_grid, args.selection_slippage_bps, args.minimum_active_days
    )
    robust_test = evaluate_chosen(
        robust_chosen, frames["test"], prices, limits, calendar, args, output, "robust"
    )
    robust_test = attach_benchmarks(robust_test, frames["test"], index, calendar)
    test_grid = None
    if args.diagnostic_test_grid:
        test_grid = uncertainty_grid("test", frames["test"], prices, limits, calendar, args)
        test_grid = attach_benchmarks(test_grid, frames["test"], index, calendar)
    alignment = pd.concat([
        label_alignment(split, frame, prices, calendar) for split, frame in frames.items()
    ], ignore_index=True)
    baseline_top1 = baseline.loc[
        baseline["topk"].eq(1) & baseline["slippage_bps_each_side"].eq(args.selection_slippage_bps)
    ][["horizon", "net_cumulative", "trade_win_rate", "max_drawdown_marked"]].rename(columns={
        "net_cumulative": "baseline_test_net", "trade_win_rate": "baseline_test_win_rate",
        "max_drawdown_marked": "baseline_test_max_drawdown",
    })
    selected_slippage = chosen_test.loc[
        chosen_test["slippage_bps_each_side"].eq(args.selection_slippage_bps)
    ][["horizon", "rule", "net_cumulative", "active_signal_days", "trade_win_rate",
       "max_drawdown_marked", "CSI1000_gross_cumulative", "CSI300_gross_cumulative"]].rename(columns={
        "net_cumulative": "selected_test_net", "active_signal_days": "selected_test_active_days",
        "trade_win_rate": "selected_test_win_rate", "max_drawdown_marked": "selected_test_max_drawdown",
    })
    comparison = chosen.merge(baseline_top1, on="horizon").merge(
        selected_slippage, on=["horizon", "rule"]
    )
    comparison["test_net_change_vs_baseline"] = comparison["selected_test_net"] - comparison["baseline_test_net"]
    robust_test_selected = robust_test.loc[
        robust_test["slippage_bps_each_side"].eq(args.selection_slippage_bps)
    ][["horizon", "rule", "net_cumulative", "active_signal_days", "trade_win_rate",
       "max_drawdown_marked"]].rename(columns={
        "net_cumulative": "test_net", "active_signal_days": "test_active_days",
        "trade_win_rate": "test_win_rate", "max_drawdown_marked": "test_max_drawdown",
    })
    robust_comparison = robust_chosen.merge(baseline_top1, on="horizon").merge(
        robust_test_selected, on=["horizon", "rule"]
    )
    robust_comparison["test_net_change_vs_baseline"] = (
        robust_comparison["test_net"] - robust_comparison["baseline_test_net"]
    )
    baseline.to_csv(output / "test_baseline_four_horizons.csv", index=False)
    valid_grid.to_csv(output / "valid_uncertainty_grid.csv", index=False)
    selection_grid.to_csv(output / "selection_valid_uncertainty_grid.csv", index=False)
    chosen.to_csv(output / "selection_valid_chosen_rules.csv", index=False)
    chosen_test.to_csv(output / "test_selected_uncertainty_rules.csv", index=False)
    if test_grid is not None:
        test_grid.to_csv(output / "test_uncertainty_grid_diagnostic_only.csv", index=False)
    alignment.to_csv(output / "label_alignment.csv", index=False)
    comparison.to_csv(output / "uncertainty_selection_test_comparison.csv", index=False)
    robust_chosen.to_csv(output / "valid_selection_robust_chosen_rules.csv", index=False)
    robust_test.to_csv(output / "test_robust_uncertainty_rules.csv", index=False)
    robust_comparison.to_csv(output / "robust_selection_test_comparison.csv", index=False)
    (output / "method.json").write_text(json.dumps({
        "selection_policy": "choose highest 5bps net cumulative on Selection-valid, at least 30 active days; Test not used",
        "robust_policy": "maximize the worse 5bps net cumulative across Valid and Selection-valid, at least 30 active days in each; Test not used",
        "rules": RULES, "horizons": HORIZONS, "mainboard_only": True,
        "capital": args.capital, "commission_rate_each_side": args.commission_rate,
        "minimum_commission": args.minimum_commission, "stamp_tax": "omitted per user preference",
        "fallback": args.fallback, "lot_size": 100,
        "overlap": "open1_close2 uses two independent 50% temporal sleeves",
        "limit_down_exit": "chronological state machine retries later same-phase events; no future exit scan at entry",
        "diagnostic_test_grid_generated": args.diagnostic_test_grid,
        "test_warning": "A full Test rule grid is disabled by default and is never used for rule selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINE TEST")
    print(baseline.to_string(index=False))
    print("\nCHOSEN ON SELECTION-VALID")
    print(chosen.to_string(index=False))
    print("\nCHOSEN RULES ON TEST")
    print(chosen_test.to_string(index=False))
    print("\nROBUST VALID+SELECTION CHOSEN")
    print(robust_chosen.to_string(index=False))
    print("\nROBUST RULES ON TEST")
    print(robust_test.to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
