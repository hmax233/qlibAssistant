#!/usr/bin/env python3
"""Audit small, read-only Tushare US-market API permissions.

The audit deliberately requests only one symbol and a very small historical
date/time range.  It never stores response rows and never serializes the API
token.  The token is accepted only from ``TUSHARE_TOKEN`` or
``~/.config/tushare_token``.

The official Tushare endpoint is used by default. ``--base-url`` can override
it only when the operator has explicitly accepted forwarding the credential to
that endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.tushare.pro"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/audits/tushare_us_permission_latest.json"
DEFAULT_TOKEN_FILE = Path.home() / ".config/tushare_token"

STATUSES = {
    "success",
    "permission_denied",
    "empty",
    "api_error",
    "transport_error",
}

PERMISSION_PATTERNS = (
    "permission",
    "privilege",
    "access denied",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token invalid",
    "权限",
    "无权",
    "未授权",
    "未开通",
    "请开通",
    "单独开通",
    "积分不足",
    "token无效",
    "token不对",
    "token为空",
)


@dataclass(frozen=True)
class Attempt:
    """One bounded API attempt in a logical permission probe."""

    api_name: str
    params: Mapping[str, Any]
    fields: str
    reason: str
    retry_on: frozenset[str] = frozenset({"api_error"})


# us_daily_adj is documented by Tushare.  us_basic and us_mins follow the
# existing project usage and Tushare's us_*/plural-minutes naming convention.
# Compatibility aliases are attempted only when the preferred endpoint reports
# an API-level error, so a denied permission never creates extra requests.
PROBES: Mapping[str, Sequence[Attempt]] = {
    "us_basic": (
        Attempt(
            api_name="us_basic",
            params={"ts_code": "AAPL", "limit": 1},
            fields="ts_code,name,enname,exchange,list_date",
            reason="Preferred Tushare US basic-information convention; one symbol only.",
            retry_on=frozenset({"api_error", "empty"}),
        ),
        Attempt(
            api_name="us_basic",
            params={"exchange": "NAS", "limit": 1},
            fields="ts_code,name,enname,exchange,list_date",
            reason="Fallback if the proxy does not accept an exact AAPL basic query.",
            retry_on=frozenset({"api_error"}),
        ),
        Attempt(
            api_name="us_stock_basic",
            params={"ts_code": "AAPL", "limit": 1},
            fields="ts_code,name,enname,exchange,list_date",
            reason="Compatibility alias for proxies exposing an older/nonstandard name.",
        ),
    ),
    "us_daily_adj": (
        Attempt(
            api_name="us_daily_adj",
            params={
                "ts_code": "AAPL",
                "start_date": "20240722",
                "end_date": "20240722",
                "limit": 1,
            },
            fields="ts_code,trade_date,open,close,adj_factor",
            reason="Official Tushare adjusted US daily endpoint; one symbol and one day.",
        ),
        Attempt(
            api_name="us_daily",
            params={
                "ts_code": "AAPL",
                "start_date": "20240722",
                "end_date": "20240722",
                "limit": 1,
            },
            fields="ts_code,trade_date,open,close",
            reason="Compatibility fallback if us_daily_adj is unavailable on the proxy.",
        ),
    ),
    "us_mins": (
        Attempt(
            api_name="us_mins",
            params={
                "ts_code": "AAPL",
                "freq": "1min",
                "start_date": "2024-07-22 09:30:00",
                "end_date": "2024-07-22 09:32:00",
            },
            fields="ts_code,trade_time,open,close",
            reason="Existing project convention, matching Tushare's hk_mins-style API.",
        ),
        Attempt(
            api_name="us_min",
            params={
                "ts_code": "AAPL",
                "freq": "1min",
                "start_date": "2024-07-22 09:30:00",
                "end_date": "2024-07-22 09:32:00",
            },
            fields="ts_code,trade_time,open,close",
            reason="Compatibility alias tried only after an API-level endpoint error.",
        ),
    ),
}


def load_token(
    environ: Mapping[str, str] | None = None,
    token_file: Path = DEFAULT_TOKEN_FILE,
) -> tuple[str, str]:
    """Load a token without returning its path contents in metadata."""

    environment = os.environ if environ is None else environ
    env_token = environment.get("TUSHARE_TOKEN", "").strip()
    if env_token:
        return env_token, "environment:TUSHARE_TOKEN"

    try:
        file_token = token_file.expanduser().read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Tushare token not found; set TUSHARE_TOKEN or create ~/.config/tushare_token"
        ) from exc
    if not file_token:
        raise RuntimeError("Tushare token source is empty")
    return file_token, "file:~/.config/tushare_token"


def sanitize_message(value: object, token: str, limit: int = 500) -> str:
    """Redact the exact credential and common token assignments from text."""

    message = str(value or "")
    if token:
        message = message.replace(token, "<redacted>")
    message = re.sub(
        r"(?i)(token\s*[=:]\s*)[\"']?[^\s,;\"'}]+",
        r"\1<redacted>",
        message,
    )
    return message[:limit]


def _sanitize_tree(value: Any, token: str) -> Any:
    if isinstance(value, str):
        return sanitize_message(value, token, limit=max(500, len(value) + 16))
    if isinstance(value, list):
        return [_sanitize_tree(item, token) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_tree(item, token) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_message(key, token, limit=max(500, len(str(key)) + 16))
            if isinstance(key, str)
            else key: _sanitize_tree(item, token)
            for key, item in value.items()
        }
    return value


def _is_permission_denied(api_code: object, message: str) -> bool:
    normalized = message.casefold().replace(" ", "")
    return any(pattern.casefold().replace(" ", "") in normalized for pattern in PERMISSION_PATTERNS)


def _extract_data(body: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    data = body.get("data")
    if not isinstance(data, Mapping):
        return [], []
    raw_fields = data.get("fields")
    raw_items = data.get("items")
    fields = [str(field) for field in raw_fields] if isinstance(raw_fields, list) else []
    items = raw_items if isinstance(raw_items, list) else []
    return fields, items


def probe_attempt(
    session: requests.Session,
    base_url: str,
    token: str,
    attempt: Attempt,
    timeout: float,
) -> dict[str, Any]:
    """Execute one small probe and classify its result."""

    started = time.perf_counter()
    payload = {
        "api_name": attempt.api_name,
        "token": token,
        "params": dict(attempt.params),
        "fields": attempt.fields,
    }
    try:
        response = session.post(base_url, json=payload, timeout=timeout)
        response.raise_for_status()
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "api_name": attempt.api_name,
                "status": "transport_error",
                "api_code": None,
                "fields": [],
                "row_count": 0,
                "message": sanitize_message(f"Invalid JSON response: {exc}", token),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "reason": attempt.reason,
                "request": {
                    "params": dict(attempt.params),
                    "fields": attempt.fields.split(",") if attempt.fields else [],
                },
            }
        if not isinstance(body, Mapping):
            body = {"code": None, "msg": "API response root is not an object"}

        api_code = body.get("code")
        message = sanitize_message(body.get("msg"), token)
        fields, items = _extract_data(body)
        if api_code == 0:
            status = "success" if items else "empty"
        elif _is_permission_denied(api_code, message):
            status = "permission_denied"
        else:
            status = "api_error"
        return {
            "api_name": attempt.api_name,
            "status": status,
            "api_code": api_code,
            "fields": fields,
            "row_count": len(items),
            "message": message,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "reason": attempt.reason,
            "request": {
                "params": dict(attempt.params),
                "fields": attempt.fields.split(",") if attempt.fields else [],
            },
        }
    except requests.RequestException as exc:
        return {
            "api_name": attempt.api_name,
            "status": "transport_error",
            "api_code": None,
            "fields": [],
            "row_count": 0,
            "message": sanitize_message(exc, token),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "reason": attempt.reason,
            "request": {
                "params": dict(attempt.params),
                "fields": attempt.fields.split(",") if attempt.fields else [],
            },
        }


def probe_logical_api(
    session: requests.Session,
    base_url: str,
    token: str,
    logical_api: str,
    attempts: Sequence[Attempt],
    timeout: float,
) -> dict[str, Any]:
    """Probe a preferred endpoint and bounded compatibility fallbacks."""

    results: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        result = probe_attempt(session, base_url, token, attempt, timeout)
        results.append(result)
        has_fallback = index + 1 < len(attempts)
        if not has_fallback or result["status"] not in attempt.retry_on:
            break

    # Preserve a valid-but-empty preferred endpoint if only compatibility
    # fallbacks fail.  A fallback error must not overwrite stronger evidence.
    selected = next((item for item in results if item["status"] == "success"), None)
    if selected is None:
        selected = next(
            (item for item in results if item["status"] in {"permission_denied", "empty"}),
            results[-1],
        )
    return {
        "logical_api": logical_api,
        "status": selected["status"],
        "selected_api_name": selected["api_name"],
        "api_code": selected["api_code"],
        "fields": selected["fields"],
        "row_count": selected["row_count"],
        "message": selected["message"],
        "elapsed_ms": round(sum(item["elapsed_ms"] for item in results), 3),
        "attempt_count": len(results),
        "attempts": results,
    }


def run_audit(
    session: requests.Session,
    base_url: str,
    token: str,
    token_source: str,
    selected_apis: Iterable[str],
    timeout: float = 20.0,
) -> dict[str, Any]:
    results = {
        logical_api: probe_logical_api(
            session=session,
            base_url=base_url,
            token=token,
            logical_api=logical_api,
            attempts=PROBES[logical_api],
            timeout=timeout,
        )
        for logical_api in selected_apis
    }
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "api_base": base_url,
        "token_source": token_source,
        "token_in_report": False,
        "read_only": True,
        "bounded_small_probes": True,
        "status_values": sorted(STATUSES),
        "results": results,
    }
    return _sanitize_tree(report, token)


def write_report(report: Mapping[str, Any], output: Path, token: str) -> None:
    safe_report = _sanitize_tree(dict(report), token)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", "--base", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--apis", nargs="+", choices=sorted(PROBES), default=list(PROBES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token, token_source = load_token()
    with requests.Session() as session:
        report = run_audit(
            session=session,
            base_url=args.base_url,
            token=token,
            token_source=token_source,
            selected_apis=args.apis,
            timeout=args.timeout,
        )
    write_report(report, args.output, token)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
