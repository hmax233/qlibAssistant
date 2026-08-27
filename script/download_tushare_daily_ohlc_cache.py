#!/usr/bin/env python3
"""Download a resumable, market-wide Tushare raw daily OHLC cache by date."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/cache/tushare_daily_ohlc.parquet"
DEFAULT_BASE = "https://fastapic.stockai888.top"
FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"


def request_date(base: str, token: str, date: pd.Timestamp, timeout: float) -> pd.DataFrame:
    response = requests.post(
        base,
        json={
            "api_name": "daily",
            "token": token,
            "params": {"trade_date": date.strftime("%Y%m%d")},
            "fields": FIELDS,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(str(body.get("msg") or f"daily failed for {date.date()}"))
    data = body.get("data") or {}
    frame = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
    if frame.empty:
        raise RuntimeError(f"daily returned no rows for {date.date()}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument"] = frame["ts_code"].str.replace(
        r"^(\d+)\.(SH|SZ|BJ)$", r"\2\1", regex=True
    )
    numeric = [column for column in FIELDS.split(",") if column not in {"ts_code", "trade_date"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    token = (Path.home() / ".config/tushare_token").read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("empty Tushare token file")
    index = pd.read_csv(args.index_cache, parse_dates=["datetime"])
    dates = pd.DatetimeIndex(
        sorted(index.loc[index["index"].eq("CSI1000"), "datetime"].drop_duplicates())
    )
    dates = dates[(dates >= pd.Timestamp(args.start)) & (dates <= pd.Timestamp(args.end))]

    existing = pd.DataFrame()
    if args.output.exists():
        existing = pd.read_parquet(args.output)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"])
    completed = set(existing["trade_date"].drop_duplicates()) if not existing.empty else set()
    pending = [date for date in dates if date not in completed]
    print(f"dates={len(dates)} existing={len(completed)} pending={len(pending)}", flush=True)

    frames = [existing] if not existing.empty else []
    for number, date in enumerate(pending, 1):
        error = None
        for attempt in range(1, args.attempts + 1):
            try:
                frame = request_date(args.base, token, date, args.timeout)
                frames.append(frame)
                break
            except Exception as exc:
                error = exc
                if attempt < args.attempts:
                    time.sleep(min(8.0, 0.75 * 2 ** (attempt - 1)))
        else:
            raise RuntimeError(f"failed {date.date()}: {error}")
        if number % 10 == 0 or number == len(pending):
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(["trade_date", "instrument"], keep="last")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp.parquet")
            combined.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(args.output)
            frames = [combined]
            print(f"downloaded={number}/{len(pending)} rows={len(combined)}", flush=True)


if __name__ == "__main__":
    main()
