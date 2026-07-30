#!/usr/bin/env python3
"""Download resumable US/HK daily OHLCV from Yahoo via yahooquery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from yahooquery import Ticker

from common import LOGS, RAW, UNIVERSES, atomic_json, ensure_dirs, safe_symbol


SP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
TUSHARE_PROXY = "https://fastapic.stockai888.top"


def tushare_hk_universe() -> pd.DataFrame:
    token_file = Path.home() / ".config" / "tushare_token"
    if not token_file.exists():
        raise FileNotFoundError(f"missing token file: {token_file}")
    payload = {
        "api_name": "hk_basic",
        "token": token_file.read_text(encoding="utf-8").strip(),
        "params": {"list_status": "L"},
        "fields": "ts_code,name,enname,list_date,market",
    }
    response = requests.post(TUSHARE_PROXY, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(f"hk_basic failed: {body.get('msg')}")
    data = body["data"]
    frame = pd.DataFrame(data["items"], columns=data["fields"])
    frame = frame[frame["market"].eq("主板")].copy()
    frame["yahoo_symbol"] = frame["ts_code"].str.replace(
        r"^0(?=\d{4}\.HK$)", "", regex=True
    )
    frame["source"] = "tushare_hk_basic+yahoo_history"
    return frame.sort_values("yahoo_symbol").reset_index(drop=True)


def sp500_universe() -> pd.DataFrame:
    frame = pd.read_csv(SP500_URL)
    frame["yahoo_symbol"] = frame["Symbol"].str.replace(".", "-", regex=False)
    frame["source"] = "datasets/sp500+yahoo_history"
    return frame.sort_values("yahoo_symbol").reset_index(drop=True)


def _history(symbols: list[str], start: str, end: str, workers: int):
    return Ticker(
        symbols,
        asynchronous=True,
        max_workers=workers,
        retry=3,
        timeout=30,
    ).history(start=start, end=end, interval="1d")


def download_market(
    market: str,
    universe: pd.DataFrame,
    start: str,
    end: str,
    chunk_size: int,
    workers: int,
    limit: int,
) -> dict:
    output = RAW / market.lower()
    output.mkdir(parents=True, exist_ok=True)
    if limit > 0:
        universe = universe.head(limit)
    universe.to_csv(UNIVERSES / f"{market.lower()}_universe.csv", index=False)
    symbols = universe["yahoo_symbol"].dropna().astype(str).drop_duplicates().tolist()
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)

    def has_coverage(symbol: str) -> bool:
        path = output / f"{safe_symbol(symbol)}.parquet"
        if not path.exists():
            return False
        try:
            dates = pd.read_parquet(path, columns=["date"])["date"]
            if dates.empty:
                return False
            # New listings naturally cannot cover the requested start. Requiring
            # either an early start or 5 years of observations distinguishes a
            # deliberate full pull from a short pilot file.
            start_ok = dates.min() <= requested_start + pd.Timedelta(days=45)
            long_history = (dates.max() - dates.min()).days >= 365 * 5
            end_ok = dates.max() >= requested_end - pd.Timedelta(days=10)
            return bool((start_ok or long_history) and end_ok)
        except Exception:
            return False

    pending = [s for s in symbols if not has_coverage(s)]
    succeeded = skipped = failed = 0
    failures: dict[str, str] = {}
    for offset in range(0, len(pending), chunk_size):
        batch = pending[offset : offset + chunk_size]
        try:
            history = _history(batch, start, end, workers)
        except Exception as exc:
            failures.update({s: f"batch error: {exc}" for s in batch})
            failed += len(batch)
            time.sleep(2)
            continue
        if not isinstance(history, pd.DataFrame) or history.empty:
            failures.update({s: "empty batch" for s in batch})
            failed += len(batch)
            continue
        if not isinstance(history.index, pd.MultiIndex):
            only = batch[0]
            history = history.copy()
            history["symbol"] = only
            history = history.reset_index().set_index(["symbol", "date"])
        present = set(history.index.get_level_values("symbol").astype(str))
        for symbol in batch:
            if symbol not in present:
                failures[symbol] = "symbol absent from Yahoo response"
                failed += 1
                continue
            part = history.xs(symbol, level="symbol").reset_index()
            part["symbol"] = symbol
            if len(part) < 60:
                failures[symbol] = f"too few rows: {len(part)}"
                failed += 1
                continue
            destination = output / f"{safe_symbol(symbol)}.parquet"
            if destination.exists():
                old = pd.read_parquet(destination)
                part = (
                    pd.concat([old, part], ignore_index=True)
                    .sort_values("date")
                    .drop_duplicates("date", keep="last")
                )
            part.to_parquet(
                destination,
                index=False,
                compression="zstd",
            )
            succeeded += 1
        print(
            f"{market}: {min(offset + chunk_size, len(pending))}/{len(pending)} "
            f"downloaded={succeeded} failed={failed}",
            flush=True,
        )
    skipped = len(symbols) - len(pending)
    report = {
        "market": market,
        "requested": len(symbols),
        "downloaded": succeeded,
        "resumed_existing": skipped,
        "failed": failed,
        "start": start,
        "end": end,
        "raw_dir": str(output),
        "failures": failures,
    }
    atomic_json(LOGS / f"download_{market.lower()}_latest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=["US", "HK"], default=["US", "HK"])
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--us-limit", type=int, default=0)
    parser.add_argument("--hk-limit", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    reports = []
    if "US" in args.markets:
        reports.append(
            download_market(
                "US", sp500_universe(), args.start, args.end,
                args.chunk_size, args.workers, args.us_limit,
            )
        )
    if "HK" in args.markets:
        reports.append(
            download_market(
                "HK", tushare_hk_universe(), args.start, args.end,
                args.chunk_size, args.workers, args.hk_limit,
            )
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
