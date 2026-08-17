#!/usr/bin/env python3
"""Probe Tushare-compatible API permissions without printing the token.

Each endpoint receives one deliberately small request.  The report stores only
status, fields, row count, date coverage in the returned sample, and a
sanitized error message.  It never serializes request headers, payload tokens,
or raw response rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/audits/tushare_permission_latest.json"
DEFAULT_BASE = "https://fastapic.stockai888.top"


PROBES: dict[str, dict] = {
    "trade_cal": {"exchange": "SSE", "start_date": "20260801", "end_date": "20260814"},
    "daily": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "daily_basic": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "adj_factor": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "moneyflow": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "stk_factor": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "stk_factor_pro": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "limit_list_d": {"trade_date": "20260813"},
    "limit_step": {"trade_date": "20260813"},
    "cyq_perf": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "cyq_chips": {"ts_code": "000001.SZ", "trade_date": "20260813"},
    "top_list": {"trade_date": "20260813"},
    "top_inst": {"trade_date": "20260813"},
    "margin_detail": {"ts_code": "000001.SZ", "start_date": "20260801", "end_date": "20260814"},
    "fina_indicator": {"ts_code": "000001.SZ", "start_date": "20250101", "end_date": "20260814"},
    "forecast": {"ts_code": "000001.SZ", "start_date": "20250101", "end_date": "20260814"},
    "express": {"ts_code": "000001.SZ", "start_date": "20250101", "end_date": "20260814"},
    "stk_mins": {
        "ts_code": "000001.SZ",
        "freq": "1min",
        "start_date": "2026-08-13 09:30:00",
        "end_date": "2026-08-13 15:00:00",
    },
    "rt_min": {"ts_code": "000001.SZ", "freq": "1MIN"},
    "hk_mins": {
        "ts_code": "00700.HK",
        "freq": "1min",
        "start_date": "2026-08-13 09:30:00",
        "end_date": "2026-08-13 16:00:00",
    },
    "us_mins": {
        "ts_code": "AAPL",
        "freq": "1min",
        "start_date": "2026-08-12 09:30:00",
        "end_date": "2026-08-12 16:00:00",
    },
}


def sanitize_message(value: object, token: str) -> str:
    message = str(value or "")
    if token:
        message = message.replace(token, "<redacted>")
    return message[:500]


def probe(session: requests.Session, base: str, token: str, api_name: str, params: dict) -> dict:
    try:
        response = session.post(
            base,
            json={"api_name": api_name, "token": token, "params": params},
            timeout=45,
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or {}
        fields = list(data.get("fields") or [])
        items = list(data.get("items") or [])
        date_fields = [
            field
            for field in fields
            if field in {"trade_date", "cal_date", "ann_date", "end_date", "trade_time"}
            or "time" in field
        ]
        coverage = {}
        for field in date_fields:
            index = fields.index(field)
            values = sorted(str(row[index]) for row in items if row[index] not in (None, ""))
            if values:
                coverage[field] = {"min": values[0], "max": values[-1]}
        return {
            "status": "ok" if body.get("code") == 0 else "denied_or_error",
            "api_code": body.get("code"),
            "message": sanitize_message(body.get("msg"), token),
            "row_count": len(items),
            "fields": fields,
            "sample_coverage": coverage,
        }
    except Exception as exc:
        return {
            "status": "request_failed",
            "message": sanitize_message(repr(exc), token),
            "row_count": 0,
            "fields": [],
            "sample_coverage": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apis", nargs="*", choices=sorted(PROBES))
    parser.add_argument(
        "--minute-coverage",
        action="store_true",
        help="Probe small historical samples and supported minute frequencies.",
    )
    args = parser.parse_args()
    token_path = Path.home() / ".config/tushare_token"
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Empty token file: {token_path}")
    selected = args.apis or list(PROBES)
    session = requests.Session()
    results = {
        api_name: probe(session, args.base, token, api_name, PROBES[api_name])
        for api_name in selected
    }
    minute_coverage = {}
    if args.minute_coverage:
        for date in (
            "2005-01-04",
            "2008-08-13",
            "2010-08-13",
            "2015-08-13",
            "2016-08-15",
            "2017-08-14",
            "2018-08-13",
            "2019-08-13",
            "2020-08-13",
            "2025-08-13",
            "2026-08-13",
        ):
            key = f"1min_{date}"
            minute_coverage[key] = probe(
                session,
                args.base,
                token,
                "stk_mins",
                {
                    "ts_code": "000001.SZ",
                    "freq": "1min",
                    "start_date": f"{date} 09:00:00",
                    "end_date": f"{date} 16:00:00",
                },
            )
        for frequency in ("1min", "5min", "15min", "30min", "60min"):
            key = f"frequency_{frequency}"
            minute_coverage[key] = probe(
                session,
                args.base,
                token,
                "stk_mins",
                {
                    "ts_code": "000001.SZ",
                    "freq": frequency,
                    "start_date": "2026-08-13 09:00:00",
                    "end_date": "2026-08-13 16:00:00",
                },
            )
        for symbol in ("000001.SZ", "000002.SZ", "600000.SH", "600519.SH"):
            for date in ("2018-08-13", "2019-08-13"):
                key = f"cross_symbol_{symbol}_{date}"
                minute_coverage[key] = probe(
                    session,
                    args.base,
                    token,
                    "stk_mins",
                    {
                        "ts_code": symbol,
                        "freq": "1min",
                        "start_date": f"{date} 09:00:00",
                        "end_date": f"{date} 16:00:00",
                    },
                )
        minute_coverage["one_month_row_cap"] = probe(
            session,
            args.base,
            token,
            "stk_mins",
            {
                "ts_code": "000001.SZ",
                "freq": "1min",
                "start_date": "2026-07-01 09:00:00",
                "end_date": "2026-07-31 16:00:00",
            },
        )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "api_base": args.base,
        "token_source": str(token_path),
        "token_in_report": False,
        "small_read_only_probes": True,
        "results": results,
        "minute_coverage": minute_coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary = [
        {
            "api": name,
            "status": value["status"],
            "rows": value["row_count"],
            "fields": len(value["fields"]),
            "message": value["message"],
        }
        for name, value in results.items()
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if minute_coverage:
        coverage_summary = [
            {
                "probe": name,
                "status": value["status"],
                "rows": value["row_count"],
                "coverage": value["sample_coverage"],
                "message": value["message"],
            }
            for name, value in minute_coverage.items()
        ]
        print(json.dumps(coverage_summary, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
