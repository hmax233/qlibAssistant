#!/usr/bin/env python3
"""Download resumable exact A-share daily up/down limits from Tushare."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


PROXY = "https://fastapic.stockai888.top"
DEFAULT_OUTPUT = Path("/Users/hmax/investment_data/supplemental/stk_limit")
DEFAULT_CALENDAR = Path.home() / ".qlib/qlib_data/cn_data/calendars/day.txt"


def normalize_symbol(ts_code: str) -> str:
    code, exchange = str(ts_code).split(".", 1)
    return f"{exchange.upper()}{code}"


def request_limit(token: str, date: pd.Timestamp) -> pd.DataFrame:
    text = f"{date:%Y%m%d}"
    payload = {
        "api_name": "stk_limit",
        "token": token,
        "params": {"trade_date": text},
        "fields": "trade_date,ts_code,pre_close,up_limit,down_limit",
    }
    response = requests.post(PROXY, json=payload, timeout=90)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(body.get("msg") or f"stk_limit failed for {text}")
    data = body["data"]
    frame = pd.DataFrame(data["items"], columns=data["fields"])
    if frame.empty:
        raise RuntimeError(f"empty stk_limit response for {text}")
    frame["date"] = pd.to_datetime(frame.pop("trade_date"))
    frame["symbol"] = frame.pop("ts_code").map(normalize_symbol)
    for column in ("up_limit", "down_limit"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # A small number of reseller snapshots omit pre_close even when explicitly
    # requested. Daily upper/lower limits are symmetric around pre-close, so
    # their midpoint is the correct ratio denominator up to exchange rounding.
    if "pre_close" not in frame:
        frame["pre_close"] = (frame["up_limit"] + frame["down_limit"]) / 2
    else:
        frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
        frame["pre_close"] = frame["pre_close"].fillna(
            (frame["up_limit"] + frame["down_limit"]) / 2
        )
    frame["up_return"] = frame["up_limit"] / frame["pre_close"] - 1
    frame["down_return"] = frame["down_limit"] / frame["pre_close"] - 1
    return frame[
        [
            "date",
            "symbol",
            "pre_close",
            "up_limit",
            "down_limit",
            "up_return",
            "down_return",
        ]
    ]


def fetch_one(token: str, date: pd.Timestamp, output: Path) -> tuple[str, str | None]:
    text = f"{date:%Y%m%d}"
    target = output / f"{text}.parquet"
    if target.exists():
        return text, None
    error = None
    for attempt in range(5):
        try:
            frame = request_limit(token, date)
            temp = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temp, index=False, compression="zstd")
            temp.replace(target)
            return text, None
        except Exception as exc:
            error = repr(exc)
            time.sleep(1.5 * (attempt + 1))
    return text, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-06-13")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip rebuilding the date-filterable consolidated Parquet.",
    )
    args = parser.parse_args()
    token = (Path.home() / ".config" / "tushare_token").read_text(
        encoding="utf-8"
    ).strip()
    calendar = pd.to_datetime(
        [
            line.strip()
            for line in args.calendar.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    end = pd.Timestamp(args.end) if args.end else calendar.max()
    dates = list(
        calendar[
            (calendar >= pd.Timestamp(args.start))
            & (calendar <= end)
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [
        date for date in dates if not (args.output / f"{date:%Y%m%d}.parquet").exists()
    ]
    failures: dict[str, str] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_one, token, date, args.output): date
            for date in pending
        }
        for future in as_completed(futures):
            text, error = future.result()
            completed += 1
            if error:
                failures[text] = error
            if completed % 20 == 0 or completed == len(pending):
                print(
                    f"stk_limit {completed}/{len(pending)} "
                    f"failed={len(failures)}",
                    flush=True,
                )
    report = {
        "start": args.start,
        "end": f"{end:%Y-%m-%d}",
        "trade_dates": len(dates),
        "downloaded": len(pending) - len(failures),
        "resumed_existing": len(dates) - len(pending),
        "failed": failures,
        "output": str(args.output),
        "purpose": (
            "Exact historical price-limit ratios for ST-universe exclusion and "
            "strict close-execution backtests."
        ),
    }
    consolidated = args.output / "stk_limit_all.parquet"
    if not args.no_consolidate and not failures:
        daily_files = sorted(
            path
            for path in args.output.glob("*.parquet")
            if len(path.stem) == 8 and path.stem.isdigit()
        )
        combined = pd.concat(
            [pd.read_parquet(path) for path in daily_files],
            ignore_index=True,
        ).sort_values(["date", "symbol"])
        combined["pre_close"] = pd.to_numeric(
            combined["pre_close"], errors="coerce"
        ).fillna((combined["up_limit"] + combined["down_limit"]) / 2)
        combined["up_return"] = combined["up_limit"] / combined["pre_close"] - 1
        combined["down_return"] = (
            combined["down_limit"] / combined["pre_close"] - 1
        )
        temp_consolidated = consolidated.with_suffix(".parquet.tmp")
        combined.to_parquet(
            temp_consolidated,
            index=False,
            compression="zstd",
            row_group_size=100_000,
        )
        temp_consolidated.replace(consolidated)
        report["consolidated"] = str(consolidated)
        report["consolidated_rows"] = len(combined)
    report_path = args.output.parent / "stk_limit_download_latest.json"
    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
