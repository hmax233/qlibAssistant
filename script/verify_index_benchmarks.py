#!/usr/bin/env python3
"""用 Tushare index_daily 交叉验证 Qlib 中 CSI1000/CSI300 的1日基准收益。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_CODES = {"CSI1000": ("SH000852", "000852.SH"), "CSI300": ("SH000300", "000300.SH")}


def load_update_module():
    path = PROJECT_ROOT / "update-predict.py"
    spec = importlib.util.spec_from_file_location("update_predict", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tushare_close(module, token, ts_code, start, end):
    # 当前代理在跨度较长的 index_daily 请求中偶尔漏交易日；按7个自然日分段可避免
    # 日期缺口导致 shift(-1/-2) 错位，并便于后续检查和去重。
    frames = []
    cursor = start.normalize()
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
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("trade_date", keep="last")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date")["close"].astype(float).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-04-15")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--output", default=".qlibAssistant/analysis/index_benchmark_verification_20260720.csv")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    module = load_update_module()
    token = module.get_token()
    qlib.init(provider_uri=str(Path("~/.qlib/qlib_data/cn_data").expanduser()), region=REG_CN)
    rows = []
    daily_rows = []
    for name, (qlib_code, ts_code) in INDEX_CODES.items():
        qlib_return = D.features(
            [qlib_code], ["Ref($close,-2)/Ref($close,-1)-1"], start_time=start, end_time=end
        ).iloc[:, 0].droplevel("instrument").rename("qlib_return")
        close = tushare_close(module, token, ts_code, start, end + pd.Timedelta(days=10))
        ts_return = (close.shift(-2) / close.shift(-1) - 1).rename("tushare_return")
        compare = pd.concat([qlib_return, ts_return], axis=1, join="inner").dropna()
        compare["abs_error"] = (compare["qlib_return"] - compare["tushare_return"]).abs()
        compare = compare.reset_index().rename(columns={"index": "datetime"})
        compare.insert(0, "index", name)
        daily_rows.append(compare)
        rows.append(
            {
                "index": name,
                "qlib_code": qlib_code,
                "tushare_code": ts_code,
                "start": compare["datetime"].min(),
                "end": compare["datetime"].max(),
                "matched_days": len(compare),
                "qlib_cumulative": (1 + compare["qlib_return"]).prod() - 1,
                "tushare_cumulative": (1 + compare["tushare_return"]).prod() - 1,
                "cumulative_diff": (1 + compare["qlib_return"]).prod() - (1 + compare["tushare_return"]).prod(),
                "mean_abs_daily_error": compare["abs_error"].mean(),
                "max_abs_daily_error": compare["abs_error"].max(),
                "daily_correlation": compare["qlib_return"].corr(compare["tushare_return"]),
                "allclose_1e-6": np.allclose(compare["qlib_return"], compare["tushare_return"], atol=1e-6),
            }
        )
    result = pd.DataFrame(rows)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    daily_output = output.with_name(output.stem + "_daily.csv")
    daily = pd.concat(daily_rows, ignore_index=True)
    daily.to_csv(daily_output, index=False)
    print(result.to_string(index=False))
    print("\n最大差异日期：")
    print(daily.nlargest(10, "abs_error").to_string(index=False))
    print(f"验证结果: {output}")
    print(f"逐日明细: {daily_output}")


if __name__ == "__main__":
    main()
