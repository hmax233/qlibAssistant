#!/usr/bin/env python3
"""Download resumable Tushare A-share minute bars by symbol and calendar year.

The token is read from ``~/.config/tushare_token`` and never logged.  Output is
partitioned by frequency/symbol/year so interrupted runs can resume without
redownloading completed partitions.  The default pilot uses a deterministic
hash sample from the historical CSI1000-mainboard union.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://fastapic.stockai888.top"
DEFAULT_UNIVERSE = Path.home() / ".qlib/qlib_data/cn_data/instruments/csi1000_mainboard.txt"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/supplemental/minute_bars"
REQUIRED_FIELDS = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}


def to_ts_code(instrument: str) -> str:
    instrument = str(instrument).upper()
    if instrument.startswith("SH"):
        return f"{instrument[2:]}.SH"
    if instrument.startswith("SZ"):
        return f"{instrument[2:]}.SZ"
    if instrument.startswith("BJ"):
        return f"{instrument[2:]}.BJ"
    raise ValueError(f"Unsupported instrument: {instrument}")


def stable_order(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_universe(path: Path, start: str, end: str, sample_size: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["instrument", "member_start", "member_end"],
        dtype=str,
    )
    frame["member_start"] = pd.to_datetime(frame["member_start"])
    frame["member_end"] = pd.to_datetime(frame["member_end"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    frame = frame.loc[
        frame["member_end"].ge(start_ts) & frame["member_start"].le(end_ts)
    ].copy()
    symbols = (
        frame.groupby("instrument", as_index=False)
        .agg(member_start=("member_start", "min"), member_end=("member_end", "max"))
    )
    symbols["stable_order"] = symbols["instrument"].map(stable_order)
    symbols = symbols.sort_values("stable_order").drop(columns="stable_order")
    if sample_size > 0:
        symbols = symbols.head(sample_size)
    symbols["ts_code"] = symbols["instrument"].map(to_ts_code)
    return symbols.reset_index(drop=True)


def request_bars(
    base: str,
    token: str,
    ts_code: str,
    frequency: str,
    year: int,
    timeout: float,
) -> pd.DataFrame:
    payload = {
        "api_name": "stk_mins",
        "token": token,
        "params": {
            "ts_code": ts_code,
            "freq": frequency,
            "start_date": f"{year}-01-01 00:00:00",
            "end_date": f"{year}-12-31 23:59:59",
        },
    }
    response = requests.post(base, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(str(body.get("msg") or f"stk_mins failed: {ts_code} {year}"))
    data = body.get("data") or {}
    return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])


def validate_bars(frame: pd.DataFrame, ts_code: str, year: int) -> tuple[pd.DataFrame, dict]:
    if frame.empty:
        return frame, {"rows": 0, "days": 0, "duplicates_removed": 0, "bad_ohlc": 0}
    missing = REQUIRED_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fields {sorted(missing)} for {ts_code} {year}")
    frame = frame[list(REQUIRED_FIELDS)].copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    numeric = ["open", "high", "low", "close", "vol", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["trade_time", "open", "high", "low", "close"])
    before = len(frame)
    frame = frame.drop_duplicates(["ts_code", "trade_time"], keep="last")
    duplicates = before - len(frame)
    bad_ohlc = ~(
        frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["high"].ge(frame["low"])
    )
    bad_count = int(bad_ohlc.sum())
    frame = frame.loc[~bad_ohlc & frame["vol"].ge(0) & frame["amount"].ge(0)]
    frame = frame.sort_values("trade_time").reset_index(drop=True)
    return frame, {
        "rows": len(frame),
        "days": int(frame["trade_time"].dt.normalize().nunique()),
        "duplicates_removed": duplicates,
        "bad_ohlc": bad_count,
        "first": frame["trade_time"].min().isoformat() if len(frame) else None,
        "last": frame["trade_time"].max().isoformat() if len(frame) else None,
    }


def download_partition(
    task: tuple[str, str, int, Path],
    base: str,
    token: str,
    frequency: str,
    timeout: float,
    attempts: int,
    force: bool,
) -> dict:
    instrument, ts_code, year, target = task
    if target.exists() and not force:
        return {"status": "existing", "instrument": instrument, "year": year, "path": str(target)}
    error = None
    for attempt in range(1, attempts + 1):
        try:
            frame = request_bars(base, token, ts_code, frequency, year, timeout)
            frame, quality = validate_bars(frame, ts_code, year)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(target)
            return {
                "status": "downloaded",
                "instrument": instrument,
                "year": year,
                "path": str(target),
                **quality,
            }
        except Exception as exc:
            error = repr(exc)
            if attempt < attempts:
                time.sleep(min(10.0, 0.75 * 2 ** (attempt - 1) + random.random() * 0.25))
    return {
        "status": "failed",
        "instrument": instrument,
        "year": year,
        "path": str(target),
        "error": str(error)[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--freq", choices=("1min", "5min", "15min", "30min", "60min"), default="15min")
    parser.add_argument("--sample-size", type=int, default=300, help="0 downloads the full historical union")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    token_path = Path.home() / ".config/tushare_token"
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Empty token file: {token_path}")
    universe = load_universe(args.universe, args.start, args.end, args.sample_size)
    start_year, end_year = pd.Timestamp(args.start).year, pd.Timestamp(args.end).year
    root = args.output / args.freq
    tasks = []
    for row in universe.itertuples(index=False):
        first_year = max(start_year, row.member_start.year)
        last_year = min(end_year, row.member_end.year)
        for year in range(first_year, last_year + 1):
            target = root / row.instrument / f"{year}.parquet"
            tasks.append((row.instrument, row.ts_code, year, target))

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                download_partition,
                task,
                args.base,
                token,
                args.freq,
                args.timeout,
                args.attempts,
                args.force,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(future_map), 1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or index == len(tasks):
                counts = pd.Series([item["status"] for item in results]).value_counts().to_dict()
                elapsed = time.monotonic() - started
                print(
                    f"minute {index}/{len(tasks)} elapsed={elapsed:.1f}s status={counts}",
                    flush=True,
                )

    status_counts = pd.Series([item["status"] for item in results]).value_counts().to_dict()
    failures = [item for item in results if item["status"] == "failed"]
    downloaded = [item for item in results if item["status"] == "downloaded"]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "start": args.start,
        "end": args.end,
        "frequency": args.freq,
        "sample_size_requested": args.sample_size,
        "symbols": len(universe),
        "partitions": len(tasks),
        "status_counts": status_counts,
        "downloaded_rows": int(sum(item.get("rows", 0) for item in downloaded)),
        "downloaded_days": int(sum(item.get("days", 0) for item in downloaded)),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "failures": failures,
        "output": str(root.resolve()),
        "token_in_report": False,
        "universe_instruments": universe["instrument"].tolist(),
    }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "download_report_latest.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({key: value for key, value in report.items() if key != "universe_instruments"}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} partitions failed; rerun to resume")


if __name__ == "__main__":
    main()
