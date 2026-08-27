from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/audit_tushare_us_permissions.py"
SPEC = importlib.util.spec_from_file_location("audit_tushare_us_permissions", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class FakeResponse:
    def __init__(self, body, status_error: Exception | None = None):
        self.body = body
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def make_attempt():
    return audit.Attempt(
        api_name="us_daily_adj",
        params={"ts_code": "AAPL", "start_date": "20240722", "end_date": "20240722"},
        fields="ts_code,trade_date,close",
        reason="test",
    )


def test_load_token_prefers_environment_and_reports_only_source(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-secret", encoding="utf-8")

    token, source = audit.load_token({"TUSHARE_TOKEN": "env-secret"}, token_file)

    assert token == "env-secret"
    assert source == "environment:TUSHARE_TOKEN"
    assert "env-secret" not in source


def test_load_token_falls_back_to_standard_file_source(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-secret\n", encoding="utf-8")

    token, source = audit.load_token({}, token_file)

    assert token == "file-secret"
    assert source == "file:~/.config/tushare_token"
    assert "file-secret" not in source


@pytest.mark.parametrize(
    ("body", "expected_status", "expected_rows"),
    [
        (
            {
                "code": 0,
                "msg": None,
                "data": {"fields": ["ts_code", "close"], "items": [["AAPL", 224.82]]},
            },
            "success",
            1,
        ),
        ({"code": 0, "msg": "", "data": {"fields": ["ts_code"], "items": []}}, "empty", 0),
        ({"code": -2001, "msg": "抱歉，您没有接口访问权限", "data": None}, "permission_denied", 0),
        ({"code": -1000, "msg": "接口不存在", "data": None}, "api_error", 0),
    ],
)
def test_probe_attempt_classifies_api_outcomes(body, expected_status, expected_rows):
    session = Mock()
    session.post.return_value = FakeResponse(body)

    result = audit.probe_attempt(session, "https://proxy.invalid", "secret", make_attempt(), 1.0)

    assert result["status"] == expected_status
    assert result["row_count"] == expected_rows
    assert result["status"] in audit.STATUSES


def test_probe_attempt_classifies_transport_error_and_redacts_token():
    secret = "super-secret-token"
    session = Mock()
    session.post.side_effect = requests.ConnectionError(f"failed token={secret}")

    result = audit.probe_attempt(session, "https://proxy.invalid", secret, make_attempt(), 1.0)

    assert result["status"] == "transport_error"
    assert secret not in json.dumps(result)
    assert "<redacted>" in result["message"]


def test_invalid_chinese_token_message_is_auth_denial_without_fallback():
    session = Mock()
    session.post.return_value = FakeResponse(
        {"code": 40101, "msg": "您的token不对，请确认。", "data": None}
    )
    result = audit.probe_logical_api(
        session,
        "https://api.tushare.pro",
        "secret",
        "us_daily_adj",
        audit.PROBES["us_daily_adj"],
        1.0,
    )
    assert result["status"] == "permission_denied"
    assert result["attempt_count"] == 1


def test_api_message_and_entire_report_are_sanitized():
    secret = "token-value-that-must-not-leak"
    session = Mock()
    session.post.return_value = FakeResponse(
        {"code": -2001, "msg": f"permission denied token={secret}", "data": None}
    )

    report = audit.run_audit(
        session=session,
        base_url="https://proxy.invalid",
        token=secret,
        token_source="environment:TUSHARE_TOKEN",
        selected_apis=["us_daily_adj"],
        timeout=1.0,
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert secret not in serialized
    assert report["token_in_report"] is False
    assert report["results"]["us_daily_adj"]["status"] == "permission_denied"


def test_api_error_uses_bounded_fallback_but_permission_denied_does_not():
    session = Mock()
    session.post.side_effect = [
        FakeResponse({"code": -1000, "msg": "接口不存在", "data": None}),
        FakeResponse(
            {
                "code": 0,
                "msg": "",
                "data": {"fields": ["ts_code", "trade_date"], "items": [["AAPL", "20240722"]]},
            }
        ),
    ]

    result = audit.probe_logical_api(
        session,
        "https://proxy.invalid",
        "secret",
        "us_daily_adj",
        audit.PROBES["us_daily_adj"],
        1.0,
    )

    assert result["status"] == "success"
    assert result["selected_api_name"] == "us_daily"
    assert result["attempt_count"] == 2
    assert session.post.call_count == 2

    denied_session = Mock()
    denied_session.post.return_value = FakeResponse(
        {"code": -2001, "msg": "需要单独开通权限", "data": None}
    )
    denied = audit.probe_logical_api(
        denied_session,
        "https://proxy.invalid",
        "secret",
        "us_daily_adj",
        audit.PROBES["us_daily_adj"],
        1.0,
    )
    assert denied["status"] == "permission_denied"
    assert denied["attempt_count"] == 1
    assert denied_session.post.call_count == 1


def test_empty_preferred_endpoint_is_not_overwritten_by_fallback_errors():
    session = Mock()
    session.post.side_effect = [
        FakeResponse({"code": 0, "msg": "", "data": {"fields": ["ts_code"], "items": []}}),
        FakeResponse({"code": -1000, "msg": "参数不兼容", "data": None}),
        FakeResponse({"code": -1000, "msg": "接口不存在", "data": None}),
    ]

    result = audit.probe_logical_api(
        session,
        "https://proxy.invalid",
        "secret",
        "us_basic",
        audit.PROBES["us_basic"],
        1.0,
    )

    assert result["status"] == "empty"
    assert result["selected_api_name"] == "us_basic"
    assert result["attempt_count"] == 3


def test_write_report_contains_no_token(tmp_path):
    secret = "never-write-this-token"
    report = {"message": f"error token={secret}", "token_in_report": False}
    output = tmp_path / "audit.json"

    audit.write_report(report, output, secret)

    content = output.read_text(encoding="utf-8")
    assert secret not in content
    assert json.loads(content)["token_in_report"] is False
