#!/usr/bin/env python3
"""Download resumable Tushare A-share daily money-flow tables.

The current 15,000-point proxy account has been verified against this endpoint.
Files are intentionally stored outside Qlib binaries because money-flow fields
are supplementary features, not OHLCV.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


PROXY = "https://fastapic.stockai888.top"
DEFAULT_OUTPUT = Path("/Users/hmax/investment_data/supplemental/moneyflow")
DEFAULT_CALENDAR = Path.home() / ".qlib/qlib_data/cn_data/calendars/day.txt"


def request_moneyflow(token: str, date: str) -> pd.DataFrame:
    payload = {
        "api_name": "moneyflow",
        "token": token,
        "params": {"trade_date": date.replace("-", "")},
    }
    response = requests.post(PROXY, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(body.get("msg") or f"moneyflow failed for {date}")
    data = body["data"]
    return pd.DataFrame(data["items"], columns=data["fields"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-07-31")
    parser.add_argument("--end", default="2026-07-30")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--delay", type=float, default=0.65)
    args = parser.parse_args()
    token = (Path.home() / ".config/tushare_token").read_text(encoding="utf-8").strip()
    calendar = pd.to_datetime(
        [
            line.strip()
            for line in args.calendar.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    dates = calendar[(calendar >= pd.Timestamp(args.start)) & (calendar <= pd.Timestamp(args.end))]
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [
        date for date in dates
        if not (args.output / f"{date:%Y%m%d}.parquet").exists()
    ]
    failures = {}
    downloaded = 0
    for idx, date in enumerate(pending, 1):
        date_text = f"{date:%Y%m%d}"
        for attempt in range(5):
            try:
                frame = request_moneyflow(token, date_text)
                if frame.empty:
                    raise RuntimeError("empty response")
                frame.to_parquet(
                    args.output / f"{date_text}.parquet",
                    index=False,
                    compression="zstd",
                )
                downloaded += 1
                break
            except Exception as exc:
                if attempt == 4:
                    failures[date_text] = repr(exc)
                else:
                    time.sleep(max(2.0, args.delay * (attempt + 1)))
        if idx % 20 == 0 or idx == len(pending):
            print(
                f"moneyflow {idx}/{len(pending)} downloaded={downloaded} "
                f"failed={len(failures)}",
                flush=True,
            )
        time.sleep(args.delay)
    report = {
        "start": args.start,
        "end": args.end,
        "trade_dates": len(dates),
        "downloaded": downloaded,
        "resumed_existing": len(dates) - len(pending),
        "failed": failures,
        "output": str(args.output),
        "definition": (
            "Tushare L2-derived active buy/sell buckets; net_mf_amount is a "
            "vendor-derived measure, not the identity of an actual institution."
        ),
    }
    report_path = args.output.parent / "moneyflow_download_latest.json"
    temp = report_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
