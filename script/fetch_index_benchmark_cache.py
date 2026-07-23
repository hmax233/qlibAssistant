#!/usr/bin/env python3
"""从Tushare分段拉取官方指数收盘价，生成评估用本地缓存。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_CODES = {"CSI1000": "000852.SH", "CSI300": "000300.SH"}


def load_update_module():
    path = PROJECT_ROOT / "update-predict.py"
    spec = importlib.util.spec_from_file_location("update_predict", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fetch_weekly(module, token, ts_code, start, end):
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=6), end)
        payload = module.tushare_curl(
            "index_daily",
            {
                "ts_code": ts_code,
                "start_date": cursor.strftime("%Y%m%d"),
                "end_date": chunk_end.strftime("%Y%m%d"),
            },
            token,
        )
        frames.append(pd.DataFrame(payload["items"], columns=payload["fields"]))
        cursor = chunk_end + pd.Timedelta(days=1)
    result = pd.concat(frames, ignore_index=True).drop_duplicates("trade_date", keep="last")
    result["datetime"] = pd.to_datetime(result["trade_date"])
    result["close"] = result["close"].astype(float)
    return result[["datetime", "close"]].sort_values("datetime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--output",
        default=".qlibAssistant/cache/tushare_index_daily.csv",
    )
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    module = load_update_module()
    token = module.get_token()
    frames = []
    for index_name, ts_code in INDEX_CODES.items():
        frame = fetch_weekly(module, token, ts_code, start, end)
        frame.insert(0, "index", index_name)
        frame.insert(1, "ts_code", ts_code)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.groupby("index")["datetime"].agg(["min", "max", "count"]).to_string())
    print(f"指数缓存: {output}")


if __name__ == "__main__":
    main()
