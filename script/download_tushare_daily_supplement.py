#!/usr/bin/env python3
"""Download resumable Tushare daily-basic, money-flow, chip and pro factors."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from download_tushare_minutes import DEFAULT_BASE, DEFAULT_UNIVERSE, load_universe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/supplemental/tushare_daily"

ENDPOINT_FIELDS = {
    "daily_basic": (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
        "free_share,total_mv,circ_mv"
    ),
    "moneyflow": (
        "ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,"
        "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,"
        "buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,"
        "sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount"
    ),
    "cyq_perf": (
        "ts_code,trade_date,his_low,his_high,cost_5pct,cost_15pct,cost_50pct,"
        "cost_85pct,cost_95pct,weight_avg,winner_rate"
    ),
    # Keep one adjusted-price variant and indicators that are not direct
    # duplicates of the ordinary Alpha158 moving-average/ROC family.
    "stk_factor_pro": (
        "ts_code,trade_date,close_qfq,amount,turnover_rate_f,volume_ratio,"
        "atr_qfq,asi_qfq,asit_qfq,brar_ar_qfq,brar_br_qfq,cr_qfq,"
        "dmi_adx_qfq,dmi_adxr_qfq,dmi_mdi_qfq,dmi_pdi_qfq,emv_qfq,maemv_qfq,"
        "mass_qfq,ma_mass_qfq,mfi_qfq,obv_qfq,psy_qfq,psyma_qfq,"
        "trix_qfq,trma_qfq,vr_qfq,wr_qfq,wr1_qfq"
    ),
}


def request_endpoint(base, token, endpoint, ts_code, start, end, fields, timeout):
    response = requests.post(
        base,
        json={
            "api_name": endpoint,
            "token": token,
            "params": {"ts_code": ts_code, "start_date": start, "end_date": end},
            "fields": fields,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(str(body.get("msg") or f"{endpoint} failed for {ts_code}"))
    data = body.get("data") or {}
    return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])


def normalize(frame: pd.DataFrame, endpoint: str) -> tuple[pd.DataFrame, dict]:
    if frame.empty:
        return frame, {"rows": 0, "first": None, "last": None, "duplicates_removed": 0}
    required = {"ts_code", "trade_date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{endpoint}: missing {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    before = len(frame)
    frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
    numeric = [column for column in frame if column not in {"ts_code", "trade_date"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame, {
        "rows": len(frame),
        "first": frame["trade_date"].min().date().isoformat(),
        "last": frame["trade_date"].max().date().isoformat(),
        "duplicates_removed": before - len(frame),
    }


def run_task(task, args, token):
    endpoint, instrument, ts_code, target = task
    if target.exists() and not args.force:
        return {"status": "existing", "endpoint": endpoint, "instrument": instrument}
    error = None
    for attempt in range(1, args.attempts + 1):
        try:
            frame = request_endpoint(
                args.base, token, endpoint, ts_code,
                pd.Timestamp(args.start).strftime("%Y%m%d"),
                pd.Timestamp(args.end).strftime("%Y%m%d"),
                ENDPOINT_FIELDS[endpoint], args.timeout,
            )
            frame, quality = normalize(frame, endpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(target)
            return {"status": "downloaded", "endpoint": endpoint,
                    "instrument": instrument, **quality}
        except Exception as exc:
            error = repr(exc)
            if attempt < args.attempts:
                time.sleep(min(10.0, 0.75 * 2 ** (attempt - 1) + random.random() * 0.3))
    return {"status": "failed", "endpoint": endpoint, "instrument": instrument,
            "error": str(error)[:500]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-08-13")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--endpoints", nargs="+", choices=sorted(ENDPOINT_FIELDS),
                        default=list(ENDPOINT_FIELDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    token = (Path.home() / ".config/tushare_token").read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("empty Tushare token file")
    universe = load_universe(args.universe, args.start, args.end, args.sample_size)
    tasks = [
        (endpoint, row.instrument, row.ts_code, args.output / endpoint / f"{row.instrument}.parquet")
        for endpoint in args.endpoints
        for row in universe.itertuples(index=False)
    ]
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_task, task, args, token): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 25 == 0 or index == len(tasks):
                counts = pd.Series([item["status"] for item in results]).value_counts().to_dict()
                print(f"daily {index}/{len(tasks)} elapsed={time.monotonic()-started:.1f}s {counts}", flush=True)
    failures = [item for item in results if item["status"] == "failed"]
    by_endpoint = {}
    for endpoint in args.endpoints:
        selected = [item for item in results if item["endpoint"] == endpoint]
        by_endpoint[endpoint] = {
            "status_counts": pd.Series([item["status"] for item in selected]).value_counts().to_dict(),
            "downloaded_rows": int(sum(item.get("rows", 0) for item in selected)),
        }
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end, "symbols": len(universe),
        "endpoints": by_endpoint, "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "output": str(args.output.resolve()), "token_in_report": False,
        "universe_instruments": universe["instrument"].tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "download_report_latest.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "download_report_latest.json")
    print(json.dumps({key: value for key, value in report.items() if key != "universe_instruments"},
                     ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} tasks failed; rerun to resume")


if __name__ == "__main__":
    main()
