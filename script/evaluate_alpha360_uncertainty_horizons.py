#!/usr/bin/env python3
"""Strict four-horizon Alpha360 account backtest and uncertainty-rule selection.

Trading rules are selected on Selection-valid only.  The frozen Test split is
then evaluated without changing the chosen rule.  Raw Tushare OHLC, exact
daily limits, 100-share lots, minimum commissions, overlapping holding periods
and delayed exits from limit-down are handled explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from evaluate_alpha360_overnight import commission


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
    entry_value: float
    buy_fee: float
    entry_date: pd.Timestamp
    scheduled_exit: pd.Timestamp
    scheduled_exit_event: int
    exit_phase: str
    last_price: float


@dataclass
class Slot:
    cash: float
    position: Position | None = None


def drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0


def annualized_sharpe(equity: pd.Series, initial_equity: float) -> float:
    """Return the close-to-close, zero-risk-rate annualized Sharpe ratio."""

    if equity.empty:
        return float("nan")
    previous = equity.shift(1)
    previous.iloc[0] = initial_equity
    returns = equity / previous - 1.0
    deviation = float(returns.std(ddof=1))
    if not np.isfinite(deviation) or deviation <= 0:
        return float("nan")
    return float(returns.mean() / deviation * np.sqrt(252.0))


def excludes_star_and_chinext(code: str) -> bool:
    """Keep every board except STAR (SH68) and ChiNext (SZ30)."""

    value = str(code).upper()
    return not value.startswith(("SH68", "SZ30"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


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


def execution_input_audit(path: Path, rows: int, start, end) -> dict:
    """Freeze the local execution-data identity used by rule selection/backtest."""

    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size": resolved.stat().st_size,
        "rows": int(rows),
        "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(end).strftime("%Y-%m-%d"),
    }


def benchmark_equity(index: pd.DataFrame, signal_dates, calendar: pd.DatetimeIndex,
                     horizon: str, benchmark: str) -> pd.Series:
    """Simulate the benchmark with the same entry/exit phases and temporal sleeves."""

    specification = HORIZONS[horizon]
    frame = (
        index.loc[index["index"].eq(benchmark)]
        .drop_duplicates("datetime", keep="last")
        .set_index("datetime")
        .sort_index()
    )
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    signals = sorted(pd.Timestamp(date) for date in signal_dates)
    sleeves = [{"cash": 1.0 / specification["sleeves"], "units": 0.0}]
    sleeves *= specification["sleeves"]
    # Avoid aliasing the mutable dictionaries created by list multiplication.
    sleeves = [dict(value) for value in sleeves]
    entries: dict[int, list[int]] = {}
    exits: dict[int, list[int]] = {}
    maximum_event = -1
    for number, signal in enumerate(signals):
        if signal not in calendar_pos or calendar_pos[signal] + 2 >= len(calendar):
            raise ValueError(f"Benchmark signal has no complete T+2 horizon: {signal}")
        i = calendar_pos[signal]
        entry_date, exit_date = calendar[i + 1], calendar[i + 2]
        if entry_date not in frame.index or exit_date not in frame.index:
            continue
        sleeve = number % specification["sleeves"]
        entry_event = 2 * (i + 1) + (0 if specification["entry"] == "open" else 1)
        exit_event = 2 * (i + 2) + (0 if specification["exit"] == "open" else 1)
        entries.setdefault(entry_event, []).append(sleeve)
        exits.setdefault(exit_event, []).append(sleeve)
        maximum_event = max(maximum_event, exit_event)
    if maximum_event < 0:
        return pd.Series(dtype=float, name=benchmark)

    first_event = min(entries)
    records: list[dict] = []
    for event in range(first_event, maximum_event + 1):
        date = calendar[event // 2]
        phase = "open" if event % 2 == 0 else "close"
        if date not in frame.index:
            continue
        price = float(frame.loc[date, phase])
        if not np.isfinite(price) or price <= 0:
            continue
        for sleeve_number in exits.get(event, []):
            sleeve = sleeves[sleeve_number]
            if sleeve["units"]:
                sleeve["cash"] = sleeve["units"] * price
                sleeve["units"] = 0.0
        for sleeve_number in entries.get(event, []):
            sleeve = sleeves[sleeve_number]
            if sleeve["units"]:
                raise RuntimeError("Benchmark temporal sleeve was reused before its exit")
            sleeve["units"] = sleeve["cash"] / price
            sleeve["cash"] = 0.0
        equity = sum(
            sleeve["cash"] + sleeve["units"] * price
            for sleeve in sleeves
        )
        records.append({"datetime": date, "event": event, "equity": equity})
    event_frame = pd.DataFrame(records)
    if event_frame.empty:
        return pd.Series(dtype=float, name=benchmark)
    daily = (
        event_frame.sort_values("event").groupby("datetime", sort=True)["equity"].last()
    )
    daily.name = benchmark
    return daily


def benchmark_statistics(index: pd.DataFrame, signal_dates, calendar: pd.DatetimeIndex,
                         horizon: str, benchmark: str) -> dict[str, float]:
    equity = benchmark_equity(index, signal_dates, calendar, horizon, benchmark)
    with_initial = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    return {
        "gross_cumulative": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": drawdown(with_initial),
        "sharpe_rf0": annualized_sharpe(equity, 1.0),
    }


def benchmark_cumulative(index: pd.DataFrame, signal_dates, calendar: pd.DatetimeIndex,
                         horizon: str, benchmark: str) -> float:
    """Compatibility wrapper used by existing reports and tests."""

    return benchmark_statistics(index, signal_dates, calendar, horizon, benchmark)[
        "gross_cumulative"
    ]


def attach_benchmarks(results: pd.DataFrame, predictions: pd.DataFrame,
                      index: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    results = results.copy()
    dates = predictions["datetime"].drop_duplicates()
    for benchmark in ("CSI1000", "CSI300"):
        values = {
            horizon: benchmark_statistics(index, dates, calendar, horizon, benchmark)
            for horizon in HORIZONS
        }
        results[f"{benchmark}_gross_cumulative"] = results["horizon"].map(
            {horizon: metrics["gross_cumulative"] for horizon, metrics in values.items()}
        )
        results[f"{benchmark}_max_drawdown"] = results["horizon"].map(
            {horizon: metrics["max_drawdown"] for horizon, metrics in values.items()}
        )
        results[f"{benchmark}_sharpe_rf0"] = results["horizon"].map(
            {horizon: metrics["sharpe_rf0"] for horizon, metrics in values.items()}
        )
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
        correlation = (
            float(np.corrcoef(compared_raw, compared_label)[0, 1])
            if len(compared_raw) >= 2
            and np.std(compared_raw) > 0 and np.std(compared_label) > 0
            else float("nan")
        )
        rows.append({
            "split": split, "horizon": horizon, "rows": len(raw_array),
            "raw_cumulative": float(np.prod(1.0 + raw_array) - 1.0),
            "raw_up_rate": float((raw_array > 0).mean()) if len(raw_array) else float("nan"),
            "compared_rows": len(compared_raw),
            "label_cumulative": float(np.prod(1.0 + compared_label) - 1.0),
            "raw_cumulative_on_compared_rows": float(np.prod(1.0 + compared_raw) - 1.0),
            "correlation": correlation,
            "max_absolute_difference": (
                float(np.max(np.abs(compared_raw - compared_label)))
                if len(compared_raw) else float("nan")
            ),
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


def attempt_exit(
    slot: Slot,
    event_number: int,
    date: pd.Timestamp,
    phase: str,
    prices: pd.DataFrame,
    limits: pd.DataFrame,
    slippage_bps: float,
    rate: float,
    minimum: float,
    trade_returns: list[float],
    sell_notionals: list[float],
    fees: list[float],
    counters: dict[str, int],
) -> bool:
    """Try one exit using only information available at this event."""

    position = slot.position
    if position is None:
        return False
    if event_number < position.scheduled_exit_event:
        return False
    raw_price = quote(prices, date, position.instrument, phase)
    raw_volume = quote(prices, date, position.instrument, "vol") if "vol" in prices else None
    down = limit(limits, date, position.instrument, "down_limit")
    if raw_price is None or raw_price <= 0 or down is None or down <= 0:
        counters["blocked_sell_missing_attempts"] += 1
        return False
    if raw_volume is not None and raw_volume <= 0:
        counters["blocked_sell_suspended_attempts"] += 1
        return False
    position.last_price = raw_price
    if raw_price <= down + 0.005:
        counters["blocked_sell_down_limit_attempts"] += 1
        return False
    # Slippage cannot create a synthetic execution below the exchange limit.
    sell_price = max(raw_price * (1.0 - slippage_bps / 10000.0), down)
    sell_notional = position.shares * sell_price
    sell_fee = commission(sell_notional, rate, minimum)
    proceeds = sell_notional - sell_fee
    # Slot.cash is the single source of truth for uninvested cash.  Keeping a
    # second copy on Position makes marked equity count the residual twice.
    final_cash = slot.cash + proceeds
    starting_cash = slot.cash + position.entry_value + position.buy_fee
    trade_returns.append(final_cash / starting_cash - 1.0)
    counters["delayed_exit_trades"] += int(event_number > position.scheduled_exit_event)
    counters["completed_exit_trades"] += 1
    sell_notionals.append(sell_notional)
    fees.append(sell_fee)
    slot.cash = final_cash
    slot.position = None
    return True


def simulate(predictions: pd.DataFrame, prices: pd.DataFrame, limits: pd.DataFrame,
             calendar: pd.DatetimeIndex, horizon: str, rule: str, topk: int,
             fallback: bool, slippage_bps: float, capital: float, rate: float,
             minimum: float) -> tuple[pd.DataFrame, dict]:
    if topk <= 0 or capital <= 0 or rate < 0 or minimum < 0 or slippage_bps < 0:
        raise ValueError("Invalid account or execution parameter")
    specification = HORIZONS[horizon]
    sleeves = specification["sleeves"]
    entry_phase, exit_phase = specification["entry"], specification["exit"]
    slot_groups = [[Slot(capital / (sleeves * topk)) for _ in range(topk)] for _ in range(sleeves)]
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    trade_returns: list[float] = []
    buy_notionals: list[float] = []
    sell_notionals: list[float] = []
    fees: list[float] = []
    event_records: dict[int, dict] = {}
    signal_records: list[dict] = []
    counters = {
        "blocked_buy_up_limit": 0, "blocked_buy_missing": 0, "too_expensive": 0,
        "delayed_exit_trades": 0, "skipped_busy_slots": 0, "filtered_cash_slots": 0,
        "blocked_sell_missing_attempts": 0, "blocked_sell_down_limit_attempts": 0,
        "blocked_buy_suspended": 0, "blocked_sell_suspended_attempts": 0,
        "completed_exit_trades": 0, "unresolved_exit": 0,
        "fallback_replacements": 0,
    }
    grouped = list(predictions.groupby("datetime", sort=True))
    last_processed_event: int | None = None

    def snapshot(event_number: int, entries: int = 0) -> dict:
        date = calendar[event_number // 2]
        phase = "open" if event_number % 2 == 0 else "close"
        equity = 0.0
        gross_position_value = 0.0
        by_instrument: dict[str, float] = {}
        active_positions = 0
        for slot_group in slot_groups:
            for slot in slot_group:
                equity += slot.cash
                if slot.position is None:
                    continue
                active_positions += 1
                position = slot.position
                marked = quote(prices, date, position.instrument, phase)
                if marked is not None and marked > 0:
                    position.last_price = marked
                value = position.shares * position.last_price
                equity += value
                gross_position_value += value
                by_instrument[position.instrument] = by_instrument.get(position.instrument, 0.0) + value
        concentration = max(by_instrument.values(), default=0.0) / equity if equity > 0 else 0.0
        return {
            "datetime": date,
            "event": event_number,
            "phase": phase,
            "equity_mark": equity,
            "gross_exposure": gross_position_value / equity if equity > 0 else 0.0,
            "max_name_concentration": concentration,
            "active_positions": active_positions,
            "entries": entries,
        }

    def record_event(event_number: int, entries: int = 0) -> None:
        previous_entries = event_records.get(event_number, {}).get("entries", 0)
        event_records[event_number] = snapshot(event_number, previous_entries + entries)

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
                        slot, number, event_date, event_phase, prices, limits,
                        slippage_bps, rate, minimum, trade_returns, sell_notionals,
                        fees, counters,
                    )
            record_event(number)
        last_processed_event = event_number

    for signal_number, (signal, group) in enumerate(grouped):
        signal = pd.Timestamp(signal)
        if signal not in calendar_pos or calendar_pos[signal] + 2 >= len(calendar):
            raise ValueError(f"Signal has no complete T+2 execution horizon: {signal}")
        i = calendar_pos[signal]
        entry_date, scheduled_exit = calendar[i + 1], calendar[i + 2]
        entry_event_number = 2 * calendar_pos[entry_date] + (0 if entry_phase == "open" else 1)
        scheduled_exit_event = (
            2 * calendar_pos[scheduled_exit] + (0 if exit_phase == "open" else 1)
        )
        advance_to(entry_event_number)
        assigned = slot_groups[signal_number % sleeves]
        ranked = prepare_rule(group, horizon, rule)
        # True fallback traverses the complete ranking.  Capping this list at 50
        # silently changed "buy the next executable stock" into a cash rule.
        ranked_candidates = ranked if fallback else ranked.head(topk)
        candidates = list(ranked_candidates.itertuples(index=False))
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
                candidate_rank = cursor
                if fallback:
                    cursor += 1
                if candidate.instrument in used:
                    continue
                raw_buy = quote(prices, entry_date, candidate.instrument, entry_phase)
                raw_volume = (
                    quote(prices, entry_date, candidate.instrument, "vol")
                    if "vol" in prices else None
                )
                upper = limit(limits, entry_date, candidate.instrument, "up_limit")
                if raw_buy is None or raw_buy <= 0 or upper is None or upper <= 0:
                    counters["blocked_buy_missing"] += 1
                    if not fallback:
                        break
                    continue
                if raw_volume is not None and raw_volume <= 0:
                    counters["blocked_buy_suspended"] += 1
                    if not fallback:
                        break
                    continue
                if raw_buy >= upper - 0.005:
                    counters["blocked_buy_up_limit"] += 1
                    if not fallback:
                        break
                    continue
                # Preserve the exact exchange limit after applying adverse slippage.
                buy_price = min(raw_buy * (1.0 + slippage_bps / 10000.0), upper)
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
                slot.cash -= buy_value + buy_fee
                slot.position = Position(
                    instrument=candidate.instrument, shares=shares,
                    entry_value=buy_value,
                    buy_fee=buy_fee, entry_date=entry_date, scheduled_exit=scheduled_exit,
                    scheduled_exit_event=scheduled_exit_event,
                    exit_phase=exit_phase, last_price=raw_buy,
                )
                buy_notionals.append(buy_value)
                fees.append(buy_fee)
                if fallback and candidate_rank >= topk:
                    counters["fallback_replacements"] += 1
                used.add(candidate.instrument)
                bought = True
                entries_this_signal += 1
                break
            if not bought:
                counters["filtered_cash_slots"] += 1
        record_event(entry_event_number, entries_this_signal)
        signal_records.append({
            "datetime": signal, "entry_date": entry_date, "scheduled_exit": scheduled_exit,
            "entries": entries_this_signal,
        })

    # Continue only until the account is flat.  A blocked exit is retried at
    # each later observable phase (open, then close), never scanned in advance.
    if last_processed_event is not None:
        next_event = last_processed_event + 1
        while (
            any(slot.position is not None for group in slot_groups for slot in group)
            and next_event < 2 * len(calendar)
        ):
            advance_to(next_event)
            next_event += 1
    counters["unresolved_exit"] = sum(
        slot.position is not None for slot_group in slot_groups for slot in slot_group
    )
    event_frame = pd.DataFrame(event_records.values()).sort_values("event")
    if event_frame.empty:
        daily = pd.DataFrame(columns=[
            "phase", "equity_mark", "gross_exposure", "max_name_concentration",
            "active_positions", "entries",
        ])
        daily.index.name = "datetime"
        final_equity = capital
    else:
        last = event_frame.groupby("datetime", sort=True).tail(1).set_index("datetime")
        entries_by_day = event_frame.groupby("datetime", sort=True)["entries"].sum()
        last["entries"] = entries_by_day
        daily = last[[
            "phase", "equity_mark", "gross_exposure", "max_name_concentration",
            "active_positions", "entries",
        ]]
        final_equity = float(daily["equity_mark"].iloc[-1])
    active_signals = int(sum(record["entries"] > 0 for record in signal_records))
    equity_with_initial = pd.concat(
        [pd.Series([capital], dtype=float), daily["equity_mark"].reset_index(drop=True)],
        ignore_index=True,
    )
    average_equity = float(daily["equity_mark"].mean()) if len(daily) else capital
    total_traded_notional = float(sum(buy_notionals) + sum(sell_notionals))
    average_daily_turnover = (
        total_traded_notional / (2.0 * average_equity * len(daily))
        if len(daily) and average_equity > 0 else 0.0
    )
    summary = {
        "horizon": horizon, "rule": rule, "topk": topk, "fallback": fallback,
        "slippage_bps_each_side": slippage_bps, "signal_days": len(grouped),
        "equity_curve_days": len(daily),
        "active_signal_days": active_signals, "completed_trades": len(trade_returns),
        "net_cumulative": final_equity / capital - 1.0,
        "trade_win_rate": float((np.asarray(trade_returns) > 0).mean()) if trade_returns else np.nan,
        "mean_trade_return": float(np.mean(trade_returns)) if trade_returns else np.nan,
        "max_drawdown_marked": drawdown(equity_with_initial),
        "net_sharpe_rf0": annualized_sharpe(daily["equity_mark"], capital),
        "average_daily_turnover": average_daily_turnover,
        "annualized_turnover": average_daily_turnover * 252.0,
        "average_gross_exposure": float(daily["gross_exposure"].mean()) if len(daily) else 0.0,
        "max_gross_exposure": float(daily["gross_exposure"].max()) if len(daily) else 0.0,
        "average_max_name_concentration": (
            float(daily["max_name_concentration"].mean()) if len(daily) else 0.0
        ),
        "max_name_concentration": (
            float(daily["max_name_concentration"].max()) if len(daily) else 0.0
        ),
        "buy_notional": float(sum(buy_notionals)),
        "sell_notional": float(sum(sell_notionals)),
        "total_commission": float(sum(fees)),
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
                    predictions, prices, limits, calendar, horizon, rule, 1, args.fallback,
                    slippage, args.capital, args.commission_rate, args.minimum_commission,
                )
                summary["split"] = split
                rows.append(summary)
    return pd.DataFrame(rows)


def require_flat_account(frame: pd.DataFrame, label: str) -> None:
    """Fail closed when the available execution history cannot close a position."""

    if "unresolved_exit" not in frame:
        raise ValueError(f"{label} is missing unresolved_exit")
    unresolved = pd.to_numeric(frame["unresolved_exit"], errors="coerce")
    if unresolved.isna().any() or (unresolved != 0).any():
        count = int(unresolved.fillna(1).ne(0).sum())
        raise RuntimeError(f"{label} contains {count} account rows with unresolved exits")


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
                   "trade_win_rate", "max_drawdown_marked", "net_sharpe_rf0",
                   "average_daily_turnover", "max_name_concentration"]].rename(columns={
                       "net_cumulative": "selection_net_cumulative",
                       "active_signal_days": "selection_active_signal_days",
                       "trade_win_rate": "selection_trade_win_rate",
                       "max_drawdown_marked": "selection_max_drawdown",
                       "net_sharpe_rf0": "selection_sharpe_rf0",
                       "average_daily_turnover": "selection_average_daily_turnover",
                       "max_name_concentration": "selection_max_name_concentration",
                   })


def evaluate_chosen(chosen, predictions, prices, limits, calendar, args, output, label):
    rows = []
    for selected in chosen.itertuples(index=False):
        for slippage in args.slippage_bps:
            daily, summary = simulate(
                predictions, prices, limits, calendar, selected.horizon, selected.rule,
                1, args.fallback, slippage, args.capital, args.commission_rate,
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


def validate_prediction_frame(frame: pd.DataFrame, calendar: pd.DatetimeIndex,
                              split: str) -> pd.DataFrame:
    required = {"datetime", "instrument"}
    for horizon in HORIZONS:
        required.update({
            f"{horizon}_expected_return",
            f"{horizon}_return_std",
            f"{horizon}_probability_positive",
            f"{horizon}_actual_return",
        })
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{split} predictions are missing columns: {missing}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    if result.duplicated(["datetime", "instrument"]).any():
        raise ValueError(f"{split} predictions contain duplicate date/instrument keys")
    calendar_pos = {date: number for number, date in enumerate(calendar)}
    invalid_dates = [
        date for date in result["datetime"].drop_duplicates()
        if date not in calendar_pos or calendar_pos[date] + 2 >= len(calendar)
    ]
    if invalid_dates:
        raise ValueError(f"{split} has dates without a complete T+2 horizon: {invalid_dates[:3]}")
    for horizon in HORIZONS:
        numeric = [
            f"{horizon}_expected_return", f"{horizon}_return_std",
            f"{horizon}_probability_positive", f"{horizon}_actual_return",
        ]
        values = result[numeric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{split}/{horizon} contains non-finite prediction or label values")
        if (result[f"{horizon}_return_std"] < 0).any():
            raise ValueError(f"{split}/{horizon} contains a negative standard deviation")
        probability = result[f"{horizon}_probability_positive"]
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{split}/{horizon} probability is outside [0, 1]")
    return result.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def filter_board(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "all":
        return frame.copy()
    return frame.loc[frame["instrument"].map(excludes_star_and_chinext)].copy()


def validate_ensemble_selection_freeze(path: Path | None, selection_path: Path) -> dict | None:
    """Authenticate the model-ensemble pre-Test freeze without opening Test."""

    if path is None:
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("selection_split") != "selection_valid":
        raise ValueError("Ensemble manifest was not selected on selection_valid")
    if body.get("test_files_read") is not False:
        raise ValueError("Ensemble manifest is not a pre-Test freeze")
    expected = body.get("selection_valid_ensemble_predictions_sha256")
    actual = sha256(selection_path)
    if expected != actual:
        raise RuntimeError("Selection predictions differ from the frozen ensemble manifest")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "selection_predictions_sha256": actual,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--valid-predictions", type=Path)
    parser.add_argument("--selection-predictions", type=Path)
    parser.add_argument("--test-predictions", type=Path)
    parser.add_argument(
        "--ensemble-selection-manifest", type=Path,
        help="Optional authenticated E0-E6 pre-Test model-selection freeze.",
    )
    parser.add_argument("--daily-ohlc-cache", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--exact-limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--commission-rate", type=float, default=0.000235)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", nargs="+", type=float, default=[0.0, 5.0])
    parser.add_argument("--selection-slippage-bps", type=float, default=5.0)
    parser.add_argument("--minimum-active-days", type=int, default=30)
    parser.add_argument("--board-variant", choices=["mainboard", "all"], default="mainboard")
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
    selection_path = args.selection_predictions or args.run / "selection_valid_predictions.csv"
    test_path = args.test_predictions or args.run / "test_predictions.csv"
    valid_path = args.valid_predictions
    if valid_path is None and (args.run / "valid_predictions.csv").is_file():
        valid_path = args.run / "valid_predictions.csv"

    # Authenticate and freeze every model/rule choice before opening held-out Test.
    ensemble_freeze = validate_ensemble_selection_freeze(
        args.ensemble_selection_manifest, selection_path
    )
    selection_frame = validate_prediction_frame(
        pd.read_csv(selection_path, parse_dates=["datetime"]), calendar, "selection_valid"
    )
    selection_frame = filter_board(selection_frame, args.board_variant)
    selection_grid = uncertainty_grid(
        "selection_valid", selection_frame, prices, limits, calendar, args
    )
    require_flat_account(selection_grid, "Selection-valid uncertainty grid")
    selection_grid = attach_benchmarks(selection_grid, selection_frame, index, calendar)
    chosen = select_rules(selection_grid, args.selection_slippage_bps, args.minimum_active_days)

    valid_frame = valid_grid = robust_chosen = None
    if valid_path is not None:
        valid_frame = validate_prediction_frame(
            pd.read_csv(valid_path, parse_dates=["datetime"]), calendar, "valid"
        )
        if valid_frame["datetime"].max() >= selection_frame["datetime"].min():
            raise ValueError("Valid and Selection-valid are not strictly chronological")
        valid_frame = filter_board(valid_frame, args.board_variant)
        valid_grid = uncertainty_grid("valid", valid_frame, prices, limits, calendar, args)
        require_flat_account(valid_grid, "Valid uncertainty grid")
        valid_grid = attach_benchmarks(valid_grid, valid_frame, index, calendar)
        robust_chosen = select_robust_rules(
            valid_grid, selection_grid, args.selection_slippage_bps, args.minimum_active_days
        )

    selection_grid.to_csv(output / "selection_valid_uncertainty_grid.csv", index=False)
    chosen.to_csv(output / "selection_valid_chosen_rules.csv", index=False)
    if valid_grid is not None:
        valid_grid.to_csv(output / "valid_uncertainty_grid.csv", index=False)
        robust_chosen.to_csv(output / "valid_selection_robust_chosen_rules.csv", index=False)
    rule_manifest = {
        "schema_version": 2,
        "selection_split": "selection_valid",
        "selection_predictions": str(selection_path.resolve()),
        "selection_predictions_sha256": sha256(selection_path),
        "valid_predictions": str(valid_path.resolve()) if valid_path is not None else None,
        "valid_predictions_sha256": sha256(valid_path) if valid_path is not None else None,
        "ensemble_selection_freeze": ensemble_freeze,
        "selection_slippage_bps": args.selection_slippage_bps,
        "minimum_active_days": args.minimum_active_days,
        "board_variant": args.board_variant,
        "fallback": args.fallback,
        "execution_data_inputs": {
            "daily_ohlc": execution_input_audit(
                args.daily_ohlc_cache, len(prices),
                prices.index.get_level_values(0).min(), prices.index.get_level_values(0).max(),
            ),
            "exact_limits": execution_input_audit(
                args.exact_limits, len(limits),
                limits.index.get_level_values(0).min(), limits.index.get_level_values(0).max(),
            ),
            "index_cache": execution_input_audit(
                args.index_cache, len(index), index["datetime"].min(), index["datetime"].max(),
            ),
        },
        "chosen": chosen.to_dict(orient="records"),
        "robust_chosen": (
            robust_chosen.to_dict(orient="records") if robust_chosen is not None else None
        ),
        "test_opened": False,
    }
    pretest_manifest = output / "chosen_rule_manifest_pre_test.json"
    pretest_manifest.write_text(
        json.dumps(rule_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Only the frozen chosen rules are now evaluated on Test.
    test_frame = validate_prediction_frame(
        pd.read_csv(test_path, parse_dates=["datetime"]), calendar, "test"
    )
    if selection_frame["datetime"].max() >= test_frame["datetime"].min():
        raise ValueError("Selection-valid and Test are not strictly chronological")
    test_frame = filter_board(test_frame, args.board_variant)
    baseline = baseline_matrix("test", test_frame, prices, limits, calendar, args, output)
    require_flat_account(baseline, "Test baseline matrix")
    baseline = attach_benchmarks(baseline, test_frame, index, calendar)
    chosen_test = evaluate_chosen(
        chosen, test_frame, prices, limits, calendar, args, output, "selection"
    )
    require_flat_account(chosen_test, "Test selected-rule matrix")
    chosen_test = attach_benchmarks(chosen_test, test_frame, index, calendar)

    robust_test = robust_comparison = None
    if robust_chosen is not None:
        robust_test = evaluate_chosen(
            robust_chosen, test_frame, prices, limits, calendar, args, output, "robust"
        )
        require_flat_account(robust_test, "Test robust-rule matrix")
        robust_test = attach_benchmarks(robust_test, test_frame, index, calendar)
    test_grid = None
    if args.diagnostic_test_grid:
        test_grid = uncertainty_grid("test", test_frame, prices, limits, calendar, args)
        require_flat_account(test_grid, "Diagnostic Test uncertainty grid")
        test_grid = attach_benchmarks(test_grid, test_frame, index, calendar)
    alignment_frames = {"selection_valid": selection_frame, "test": test_frame}
    if valid_path is not None:
        alignment_frames["valid"] = valid_frame
    alignment = pd.concat([
        label_alignment(split, frame, prices, calendar)
        for split, frame in alignment_frames.items()
    ], ignore_index=True)
    baseline_top1 = baseline.loc[
        baseline["topk"].eq(1) & baseline["slippage_bps_each_side"].eq(args.selection_slippage_bps)
    ][["horizon", "net_cumulative", "trade_win_rate", "max_drawdown_marked",
       "net_sharpe_rf0", "average_daily_turnover", "max_name_concentration"]].rename(columns={
        "net_cumulative": "baseline_test_net", "trade_win_rate": "baseline_test_win_rate",
        "max_drawdown_marked": "baseline_test_max_drawdown",
        "net_sharpe_rf0": "baseline_test_sharpe_rf0",
        "average_daily_turnover": "baseline_test_average_daily_turnover",
        "max_name_concentration": "baseline_test_max_name_concentration",
    })
    selected_slippage = chosen_test.loc[
        chosen_test["slippage_bps_each_side"].eq(args.selection_slippage_bps)
    ][["horizon", "rule", "net_cumulative", "active_signal_days", "trade_win_rate",
       "max_drawdown_marked", "net_sharpe_rf0", "average_daily_turnover",
       "max_name_concentration", "CSI1000_gross_cumulative", "CSI300_gross_cumulative",
       "CSI1000_max_drawdown", "CSI300_max_drawdown", "CSI1000_sharpe_rf0",
       "CSI300_sharpe_rf0"]].rename(columns={
        "net_cumulative": "selected_test_net", "active_signal_days": "selected_test_active_days",
        "trade_win_rate": "selected_test_win_rate", "max_drawdown_marked": "selected_test_max_drawdown",
        "net_sharpe_rf0": "selected_test_sharpe_rf0",
        "average_daily_turnover": "selected_test_average_daily_turnover",
        "max_name_concentration": "selected_test_max_name_concentration",
    })
    comparison = chosen.merge(baseline_top1, on="horizon").merge(
        selected_slippage, on=["horizon", "rule"]
    )
    comparison["test_net_change_vs_baseline"] = comparison["selected_test_net"] - comparison["baseline_test_net"]
    if robust_test is not None:
        robust_test_selected = robust_test.loc[
            robust_test["slippage_bps_each_side"].eq(args.selection_slippage_bps)
        ][["horizon", "rule", "net_cumulative", "active_signal_days", "trade_win_rate",
           "max_drawdown_marked", "net_sharpe_rf0", "average_daily_turnover",
           "max_name_concentration"]].rename(columns={
            "net_cumulative": "test_net", "active_signal_days": "test_active_days",
            "trade_win_rate": "test_win_rate", "max_drawdown_marked": "test_max_drawdown",
            "net_sharpe_rf0": "test_sharpe_rf0",
            "average_daily_turnover": "test_average_daily_turnover",
            "max_name_concentration": "test_max_name_concentration",
        })
        robust_comparison = robust_chosen.merge(baseline_top1, on="horizon").merge(
            robust_test_selected, on=["horizon", "rule"]
        )
        robust_comparison["test_net_change_vs_baseline"] = (
            robust_comparison["test_net"] - robust_comparison["baseline_test_net"]
        )
    baseline.to_csv(output / "test_baseline_four_horizons.csv", index=False)
    chosen_test.to_csv(output / "test_selected_uncertainty_rules.csv", index=False)
    if test_grid is not None:
        test_grid.to_csv(output / "test_uncertainty_grid_diagnostic_only.csv", index=False)
    alignment.to_csv(output / "label_alignment.csv", index=False)
    comparison.to_csv(output / "uncertainty_selection_test_comparison.csv", index=False)
    if valid_grid is not None:
        robust_test.to_csv(output / "test_robust_uncertainty_rules.csv", index=False)
        robust_comparison.to_csv(output / "robust_selection_test_comparison.csv", index=False)
    rule_manifest.update({
        "test_opened": True,
        "pretest_rule_manifest": str(pretest_manifest.resolve()),
        "pretest_rule_manifest_sha256": sha256(pretest_manifest),
        "test_predictions": str(test_path.resolve()),
        "test_predictions_sha256": sha256(test_path),
    })
    (output / "evaluated_rule_manifest.json").write_text(
        json.dumps(rule_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "method.json").write_text(json.dumps({
        "selection_policy": (
            f"choose highest {args.selection_slippage_bps:g}bps net cumulative on "
            f"Selection-valid, at least {args.minimum_active_days} active days; Test not used"
        ),
        "robust_policy": (
            f"maximize the worse {args.selection_slippage_bps:g}bps net cumulative across "
            f"Valid and Selection-valid, at least {args.minimum_active_days} active days in each; "
            "frozen before Test is opened"
        ),
        "rules": RULES, "horizons": HORIZONS,
        "board_variant": args.board_variant,
        "board_variant_meaning": (
            "exclude SH68 STAR and SZ30 ChiNext only"
            if args.board_variant == "mainboard" else "include all prediction instruments"
        ),
        "capital": args.capital, "commission_rate_each_side": args.commission_rate,
        "minimum_commission": args.minimum_commission, "stamp_tax": "omitted per user preference",
        "fallback": args.fallback, "lot_size": 100,
        "overlap": "open1_close2 uses two independent 50% temporal sleeves",
        "limit_down_exit": (
            "chronological state machine retries at every later observable open/close event; "
            "no future exit scan at entry"
        ),
        "performance_path": (
            "daily marked account equity; Sharpe uses daily equity returns at rf=0; "
            "turnover is two-sided notional divided by 2*mean equity*days; concentration "
            "aggregates the same instrument across temporal sleeves"
        ),
        "diagnostic_test_grid_generated": args.diagnostic_test_grid,
        "test_warning": "A full Test rule grid is disabled by default and is never used for rule selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINE TEST")
    print(baseline.to_string(index=False))
    print("\nCHOSEN ON SELECTION-VALID")
    print(chosen.to_string(index=False))
    print("\nCHOSEN RULES ON TEST")
    print(chosen_test.to_string(index=False))
    if robust_chosen is not None:
        print("\nROBUST VALID+SELECTION CHOSEN")
        print(robust_chosen.to_string(index=False))
        print("\nROBUST RULES ON TEST")
        print(robust_test.to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
