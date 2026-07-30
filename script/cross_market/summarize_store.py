#!/usr/bin/env python3
"""Create a compact inventory of cross-market raw and factor Parquet stores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from common import DATA_STORE, RAW, FACTORS, REPORTS


def parquet_stats(path: Path) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    date_name = "date" if "date" in parquet.schema.names else "datetime"
    index = parquet.schema.names.index(date_name)
    minimums, maximums = [], []
    for group_index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(group_index).column(index).statistics
        if stats is not None and stats.has_min_max:
            minimums.append(pd.Timestamp(stats.min))
            maximums.append(pd.Timestamp(stats.max))
    return rows, min(minimums) if minimums else None, max(maximums) if maximums else None


def summarize(kind: str, market: str, folder: Path) -> dict:
    files = sorted(folder.glob("*.parquet"))
    rows = 0
    starts, ends = [], []
    for path in files:
        count, start, end = parquet_stats(path)
        rows += count
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    return {
        "kind": kind,
        "market": market,
        "files": len(files),
        "rows": rows,
        "start": min(starts).date().isoformat() if starts else None,
        "end": max(ends).date().isoformat() if ends else None,
        "bytes": sum(path.stat().st_size for path in files),
        "folder": str(folder),
    }


def main() -> None:
    rows = []
    for kind, root in (("raw", RAW), ("factors", FACTORS)):
        for market in ("a", "hk", "us"):
            rows.append(summarize(kind, market.upper(), root / market))
    inventory = pd.DataFrame(rows)
    inventory["size_gib"] = inventory["bytes"] / 1024**3
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(DATA_STORE / "inventory.csv", index=False)
    payload = {
        "data_store": str(DATA_STORE),
        "total_size_gib": float(inventory["size_gib"].sum()),
        "inventory": inventory.drop(columns="bytes").to_dict(orient="records"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(inventory.drop(columns="bytes").to_string(index=False))
    print(f"saved: {DATA_STORE / 'inventory.csv'}")


if __name__ == "__main__":
    main()
