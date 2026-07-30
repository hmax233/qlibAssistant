#!/usr/bin/env python3
"""Build resumable per-symbol Alpha158-compatible factor Parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import FACTORS, LOGS, RAW, alpha158_frame, atomic_json, ensure_dirs


def build_market(market: str, limit: int) -> dict:
    source = RAW / market.lower()
    output = FACTORS / market.lower()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    pending = [
        p
        for p in files
        if not (output / p.name).exists()
        or (output / p.name).stat().st_mtime < p.stat().st_mtime
    ]
    success = failed = 0
    failures = {}
    for idx, path in enumerate(pending, 1):
        try:
            raw = pd.read_parquet(path)
            symbol = str(raw["symbol"].dropna().iloc[0])
            factors = alpha158_frame(raw, symbol=symbol, market=market)
            factors.to_parquet(output / path.name, index=False, compression="zstd")
            success += 1
        except Exception as exc:
            failures[path.name] = repr(exc)
            failed += 1
        if idx % 25 == 0 or idx == len(pending):
            print(
                f"{market}: {idx}/{len(pending)} built={success} failed={failed}",
                flush=True,
            )
    report = {
        "market": market,
        "raw_files": len(files),
        "built": success,
        "resumed_existing": len(files) - len(pending),
        "failed": failed,
        "factor_dir": str(output),
        "failures": failures,
    }
    atomic_json(LOGS / f"factors_{market.lower()}_latest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=["A", "US", "HK"], default=["A", "US", "HK"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    reports = [build_market(market, args.limit) for market in args.markets]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
