#!/usr/bin/env python3
"""Export A-share main-board adjusted OHLCV from the existing Qlib provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from common import LOGS, RAW, UNIVERSES, atomic_json, ensure_dirs, safe_symbol


MAINBOARD_PREFIXES = ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")


def mainboard_symbols(provider: Path) -> list[str]:
    source = provider / "instruments" / "all.txt"
    symbols = {
        line.split("\t", 1)[0].upper()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line and line.split("\t", 1)[0].upper().startswith(MAINBOARD_PREFIXES)
    }
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2099-12-31")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    provider = Path(args.provider).expanduser().resolve()
    symbols = mainboard_symbols(provider)
    if args.limit > 0:
        symbols = symbols[: args.limit]
    pd.DataFrame({"yahoo_symbol": symbols, "market": "A", "source": "local_qlib"}).to_csv(
        UNIVERSES / "a_mainboard_universe.csv", index=False
    )
    output = RAW / "a"
    output.mkdir(parents=True, exist_ok=True)
    requested_start = pd.Timestamp(args.start)

    def has_coverage(symbol: str) -> bool:
        path = output / f"{safe_symbol(symbol)}.parquet"
        if not path.exists():
            return False
        try:
            dates = pd.read_parquet(path, columns=["datetime"])["datetime"]
            if dates.empty:
                return False
            return bool(
                dates.min() <= requested_start + pd.Timedelta(days=45)
                or (dates.max() - dates.min()).days >= 365 * 5
            )
        except Exception:
            return False

    pending = [s for s in symbols if not has_coverage(s)]
    qlib.init(provider_uri=str(provider), region=REG_CN)
    success = failed = 0
    failures = {}
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        try:
            frame = D.features(
                batch, fields, start_time=args.start, end_time=args.end, freq="day"
            )
        except Exception as exc:
            failures.update({s: str(exc) for s in batch})
            failed += len(batch)
            continue
        frame.columns = [c.lstrip("$") for c in fields]
        present = set(frame.index.get_level_values("instrument"))
        for symbol in batch:
            if symbol not in present:
                failures[symbol] = "no rows in Qlib provider"
                failed += 1
                continue
            part = frame.xs(symbol, level="instrument").dropna(how="all").reset_index()
            if len(part) < 60:
                failures[symbol] = f"too few rows: {len(part)}"
                failed += 1
                continue
            part["symbol"] = symbol
            destination = output / f"{safe_symbol(symbol)}.parquet"
            if destination.exists():
                old = pd.read_parquet(destination)
                part = (
                    pd.concat([old, part], ignore_index=True)
                    .sort_values("datetime")
                    .drop_duplicates("datetime", keep="last")
                )
            part.to_parquet(
                destination,
                index=False,
                compression="zstd",
            )
            success += 1
        print(
            f"A: {min(offset + args.batch_size, len(pending))}/{len(pending)} "
            f"exported={success} failed={failed}",
            flush=True,
        )
    report = {
        "requested": len(symbols),
        "exported": success,
        "resumed_existing": len(symbols) - len(pending),
        "failed": failed,
        "provider": str(provider),
        "start": args.start,
        "end": args.end,
        "failures": failures,
    }
    atomic_json(LOGS / "export_a_latest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
