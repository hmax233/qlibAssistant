#!/usr/bin/env python3
"""Aggregate Tushare minute bars into leakage-safe daily intraday factors.

All base factors for date T use only bars timestamped on T.  Rolling factors
are backward-looking within each instrument.  The resulting features are
intended for signals generated after the T close and must not be used for a
same-day close execution backtest without an explicit near-close cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / ".qlibAssistant/supplemental/minute_bars/15min"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/supplemental/intraday_daily_factors_15m"
EPS = 1e-12


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < EPS:
        return np.nan
    return float(numerator / denominator)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite Pearson correlation only when both vectors vary."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    left = left[valid]
    right = right[valid]
    if np.ptp(left) <= EPS or np.ptp(right) <= EPS:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def normalized_entropy(values: np.ndarray) -> float:
    values = np.nan_to_num(values.astype(float), nan=0.0)
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return np.nan
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def trend_metrics(close: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(close) & (close > 0)
    if valid.sum() < 3:
        return np.nan, np.nan
    y = np.log(close[valid])
    x = np.arange(len(close), dtype=float)[valid]
    x_centered = x - x.mean()
    denominator = np.square(x_centered).sum()
    if denominator <= 0:
        return np.nan, np.nan
    slope = float((x_centered * (y - y.mean())).sum() / denominator)
    fitted = y.mean() + slope * x_centered
    total = np.square(y - y.mean()).sum()
    r2 = 1.0 - safe_divide(np.square(y - fitted).sum(), total) if total > 0 else 0.0
    return slope * max(len(close) - 1, 1), float(r2)


def path_drawdown(close: np.ndarray, opening: float) -> tuple[float, float]:
    path = np.concatenate([[opening], close.astype(float)])
    path = path[np.isfinite(path) & (path > 0)]
    if len(path) < 2:
        return np.nan, np.nan
    running_max = np.maximum.accumulate(path)
    running_min = np.minimum.accumulate(path)
    drawdown = np.min(path / running_max - 1.0)
    runup = np.max(path / running_min - 1.0)
    return float(drawdown), float(runup)


def aggregate_day(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("trade_time")
    open_values = group["open"].to_numpy(float)
    close = group["close"].to_numpy(float)
    high = group["high"].to_numpy(float)
    low = group["low"].to_numpy(float)
    volume = np.nan_to_num(group["vol"].to_numpy(float), nan=0.0)
    amount = np.nan_to_num(group["amount"].to_numpy(float), nan=0.0)
    opening = float(open_values[0])
    closing = float(close[-1])
    daily_high = float(np.nanmax(high))
    daily_low = float(np.nanmin(low))
    previous = np.concatenate([[opening], close[:-1]])
    log_returns = np.log(np.maximum(close, EPS) / np.maximum(previous, EPS))
    simple_returns = close / np.maximum(previous, EPS) - 1.0
    ranges = high / np.maximum(low, EPS) - 1.0
    total_volume = float(volume.sum())
    total_amount = float(amount.sum())
    vwap = safe_divide(total_amount, total_volume)

    timestamps = pd.to_datetime(group["trade_time"])
    morning_mask = timestamps.dt.time <= pd.Timestamp("11:30").time()
    afternoon_mask = timestamps.dt.time >= pd.Timestamp("13:00").time()
    morning = group.loc[morning_mask]
    afternoon = group.loc[afternoon_mask]
    first_n = min(5, len(group))
    last_n = min(4, len(group))
    first_hour_return = safe_divide(float(close[first_n - 1]), opening) - 1.0
    last_hour_open = float(open_values[-last_n])
    last_hour_return = safe_divide(closing, last_hour_open) - 1.0
    morning_return = (
        safe_divide(float(morning["close"].iloc[-1]), opening) - 1.0
        if len(morning)
        else np.nan
    )
    afternoon_return = (
        safe_divide(float(afternoon["close"].iloc[-1]), float(afternoon["open"].iloc[0])) - 1.0
        if len(afternoon)
        else np.nan
    )
    lunch_gap = (
        safe_divide(float(afternoon["open"].iloc[0]), float(morning["close"].iloc[-1])) - 1.0
        if len(morning) and len(afternoon)
        else np.nan
    )
    signed_volume = float((np.sign(simple_returns) * volume).sum())
    signed_amount = float((np.sign(simple_returns) * amount).sum())
    price_volume_corr = safe_correlation(simple_returns, np.log1p(volume))
    range_volume_corr = safe_correlation(ranges, np.log1p(volume))
    return_autocorr_1 = safe_correlation(simple_returns[1:], simple_returns[:-1])
    volume_autocorr_1 = safe_correlation(np.log1p(volume[1:]), np.log1p(volume[:-1]))
    return_leads_volume_corr = safe_correlation(simple_returns[:-1], np.log1p(volume[1:]))
    trend_slope, trend_r2 = trend_metrics(close)
    max_drawdown, max_runup = path_drawdown(close, opening)
    realized_variance = float(np.square(log_returns).sum())
    upside_variance = float(np.square(np.maximum(log_returns, 0.0)).sum())
    downside_variance = float(np.square(np.minimum(log_returns, 0.0)).sum())
    bipower_variation = (
        float(np.pi / 2.0 * (np.abs(log_returns[1:]) * np.abs(log_returns[:-1])).sum())
        if len(log_returns) > 1 else np.nan
    )
    jump_variation_ratio = safe_divide(
        max(realized_variance - bipower_variation, 0.0), realized_variance
    )
    absolute_return_sum = float(np.abs(simple_returns).sum())
    path_efficiency = safe_divide(abs(closing / opening - 1.0), absolute_return_sum)
    running_amount = np.cumsum(amount)
    running_volume = np.cumsum(volume)
    running_vwap = np.divide(
        running_amount,
        running_volume,
        out=np.full_like(running_amount, np.nan, dtype=float),
        where=running_volume > EPS,
    )
    vwap_side = np.sign(close - running_vwap)
    valid_vwap_side = vwap_side[np.isfinite(vwap_side) & (vwap_side != 0)]
    vwap_cross_count = (
        int(np.count_nonzero(valid_vwap_side[1:] != valid_vwap_side[:-1]))
        if len(valid_vwap_side) > 1 else 0
    )
    time_axis = np.linspace(-0.5, 0.5, len(group), dtype=float)
    volume_profile_slope = safe_divide(
        float((time_axis * (volume - volume.mean())).sum()),
        float(np.square(time_axis).sum() * max(volume.mean(), EPS)),
    )
    top_abs_return_share = safe_divide(
        float(np.sort(np.abs(simple_returns))[-min(2, len(simple_returns)):].sum()),
        absolute_return_sum,
    )
    opening_bars = min(2, len(group))
    closing_bars = min(2, len(group))
    return pd.Series(
        {
            "bar_count": len(group),
            "session_open": opening,
            "session_close": closing,
            "intraday_return": safe_divide(closing, opening) - 1.0,
            "realized_volatility": float(np.sqrt(np.square(log_returns).sum())),
            "upside_realized_volatility": float(np.sqrt(upside_variance)),
            "downside_realized_volatility": float(
                np.sqrt(downside_variance)
            ),
            "downside_variance_share": safe_divide(downside_variance, realized_variance),
            "bipower_variation": bipower_variation,
            "jump_variation_ratio": jump_variation_ratio,
            "realized_skew": float(pd.Series(log_returns).skew()),
            "realized_kurtosis": float(pd.Series(log_returns).kurt()),
            "intraday_max_drawdown": max_drawdown,
            "intraday_max_runup": max_runup,
            "intraday_range": safe_divide(daily_high, daily_low) - 1.0,
            "close_location": safe_divide(closing - daily_low, daily_high - daily_low),
            "vwap": vwap,
            "close_vwap_deviation": safe_divide(closing, vwap) - 1.0,
            "above_running_vwap_ratio": float(np.nanmean(close > running_vwap)),
            "vwap_cross_count": vwap_cross_count,
            "opening_30m_return": safe_divide(float(close[opening_bars - 1]), opening) - 1.0,
            "closing_30m_return": safe_divide(closing, float(open_values[-closing_bars])) - 1.0,
            "first_hour_return": first_hour_return,
            "last_hour_return": last_hour_return,
            "morning_return": morning_return,
            "afternoon_return": afternoon_return,
            "lunch_gap": lunch_gap,
            "afternoon_minus_morning": afternoon_return - morning_return,
            "first_hour_volume_share": safe_divide(volume[:first_n].sum(), total_volume),
            "last_hour_volume_share": safe_divide(volume[-last_n:].sum(), total_volume),
            "afternoon_volume_share": safe_divide(
                group.loc[afternoon_mask, "vol"].sum(), total_volume
            ),
            "auction_volume_share": safe_divide(volume[0], total_volume),
            "largest_bar_volume_share": safe_divide(volume.max(), total_volume),
            "largest_bar_amount_share": safe_divide(amount.max(), total_amount),
            "volume_entropy": normalized_entropy(volume),
            "amount_entropy": normalized_entropy(amount),
            "signed_volume_imbalance": safe_divide(signed_volume, total_volume),
            "signed_amount_imbalance": safe_divide(signed_amount, total_amount),
            "up_volume_share": safe_divide(volume[simple_returns > 0].sum(), total_volume),
            "up_amount_share": safe_divide(amount[simple_returns > 0].sum(), total_amount),
            "price_volume_corr": price_volume_corr,
            "range_volume_corr": range_volume_corr,
            "return_autocorr_1": return_autocorr_1,
            "volume_autocorr_1": volume_autocorr_1,
            "return_leads_volume_corr": return_leads_volume_corr,
            "volume_profile_slope": volume_profile_slope,
            "amihud_intraday": float(
                np.nanmean(np.abs(simple_returns) / np.maximum(amount / 1_000_000.0, EPS))
            ),
            "up_bar_ratio": float((simple_returns > 0).mean()),
            "zero_volume_ratio": float((volume <= 0).mean()),
            "max_bar_return": float(np.nanmax(simple_returns)),
            "min_bar_return": float(np.nanmin(simple_returns)),
            "max_return_time": float(np.nanargmax(simple_returns) / max(len(simple_returns) - 1, 1)),
            "min_return_time": float(np.nanargmin(simple_returns) / max(len(simple_returns) - 1, 1)),
            "top_abs_return_share": top_abs_return_share,
            "path_efficiency": path_efficiency,
            "trend_log_return": trend_slope,
            "trend_r2": trend_r2,
            "total_volume": total_volume,
            "total_amount": total_amount,
        }
    )


ROLLING_BASES = (
    "intraday_return",
    "realized_volatility",
    "intraday_max_drawdown",
    "intraday_range",
    "close_vwap_deviation",
    "first_hour_return",
    "last_hour_return",
    "afternoon_minus_morning",
    "last_hour_volume_share",
    "signed_volume_imbalance",
    "signed_amount_imbalance",
    "downside_variance_share",
    "jump_variation_ratio",
    "above_running_vwap_ratio",
    "vwap_cross_count",
    "path_efficiency",
    "volume_profile_slope",
    "price_volume_corr",
    "amihud_intraday",
    "trend_log_return",
    "trend_r2",
)


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("datetime").copy()
    frame["overnight_gap"] = frame["session_open"] / frame["session_close"].shift(1) - 1.0
    frame["gap_intraday_reversal"] = frame["overnight_gap"] * frame["intraday_return"]
    frame["late_minus_early_return"] = frame["closing_30m_return"] - frame["opening_30m_return"]
    frame["late_volume_surge"] = frame["last_hour_volume_share"] - frame["first_hour_volume_share"]
    frame = frame.drop(columns=["session_open", "session_close"])
    for column in ROLLING_BASES:
        for window in (5, 20):
            rolling = frame[column].rolling(window, min_periods=max(3, window // 2))
            frame[f"{column}_mean_{window}d"] = rolling.mean()
            frame[f"{column}_std_{window}d"] = rolling.std()
    return frame


def build_symbol(symbol_dir: Path, output: Path, force: bool) -> dict:
    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        return {"status": "empty", "instrument": symbol_dir.name}
    newest_source = max(path.stat().st_mtime for path in files)
    if output.exists() and not force and output.stat().st_mtime >= newest_source:
        return {"status": "existing", "instrument": symbol_dir.name, "path": str(output)}
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if raw.empty:
        daily = pd.DataFrame()
    else:
        raw["trade_time"] = pd.to_datetime(raw["trade_time"])
        raw["datetime"] = raw["trade_time"].dt.normalize()
        daily = raw.groupby("datetime", sort=True, group_keys=False).apply(
            aggregate_day,
            include_groups=False,
        ).reset_index()
        daily.insert(1, "instrument", symbol_dir.name)
        daily = add_rolling_features(daily)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    daily.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    return {
        "status": "built",
        "instrument": symbol_dir.name,
        "days": len(daily),
        "features": max(0, len(daily.columns) - 2),
        "first": daily["datetime"].min().date().isoformat() if len(daily) else None,
        "last": daily["datetime"].max().date().isoformat() if len(daily) else None,
        "path": str(output),
    }


def build_symbol_task(task: tuple[Path, Path, bool]) -> dict:
    symbol_dir, output, force = task
    return build_symbol(symbol_dir, output, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    symbol_dirs = sorted(path for path in args.input.iterdir() if path.is_dir())
    results = []
    failures = []
    tasks = [
        (symbol_dir, args.output / f"{symbol_dir.name}.parquet", args.force)
        for symbol_dir in symbol_dirs
    ]
    if args.workers <= 1:
        iterator = ((task, None) for task in tasks)
        for index, (task, _) in enumerate(iterator, 1):
            try:
                results.append(build_symbol_task(task))
            except Exception as exc:
                failures.append({"instrument": task[0].name, "error": repr(exc)[:500]})
            if index % 25 == 0 or index == len(tasks):
                print(f"factors {index}/{len(tasks)} built={sum(r['status']=='built' for r in results)} failed={len(failures)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(build_symbol_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), 1):
                task = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append({"instrument": task[0].name, "error": repr(exc)[:500]})
                if index % 25 == 0 or index == len(tasks):
                    print(f"factors {index}/{len(tasks)} built={sum(r['status']=='built' for r in results)} failed={len(failures)}", flush=True)
    built = [result for result in results if result["status"] == "built"]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "symbols": len(symbol_dirs),
        "built": len(built),
        "existing": sum(result["status"] == "existing" for result in results),
        "total_daily_rows_built": int(sum(result.get("days", 0) for result in built)),
        "feature_count": max((result.get("features", 0) for result in built), default=0),
        "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "signal_timing": "All T features use bars through the T close; eligible for T+1-or-later execution only.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "build_report_latest.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} symbols failed")


if __name__ == "__main__":
    main()
