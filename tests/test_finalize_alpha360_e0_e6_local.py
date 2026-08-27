from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "script/finalize_alpha360_e0_e6_local.py"


def load_finalizer_module():
    spec = importlib.util.spec_from_file_location("alpha360_e0_e6_finalizer", FINALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observed(
    key: str,
    task_name: str,
    *,
    status: str | None = "training",
    task_state: str = "Ready",
    matching_python_count: int | None = 0,
    status_error: str | None = None,
    process_query_error: str | None = None,
) -> dict:
    return {
        "key": key,
        "task_name": task_name,
        "pipeline_status": status,
        "pipeline_updated": "2026-08-28 04:00:00",
        "status_error": status_error,
        "task_exists": True,
        "task_state": task_state,
        "task_error": None,
        "matching_python_count": matching_python_count,
        "matching_python_pids": [] if not matching_python_count else [1234],
        "process_query_error": process_query_error,
        "status_body": {"status": status},
    }


def test_finalizer_is_one_shot_and_never_creates_a_polling_process() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "while True" not in source
    assert "time.sleep(" not in source
    assert "nohup" not in source
    assert "Popen(" not in source
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 3
        for node in ast.walk(tree)
    )


def test_finalizer_runs_both_strict_variants_and_read_only_report() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    assert 'run_backtest(staging, "mainboard", True)' in source
    assert 'run_backtest(staging, "all", True)' in source
    assert 'run_backtest(staging, "mainboard", False)' in source
    assert 'run_backtest(staging, "all", False)' in source
    assert '"--fallback" if fallback else "--no-fallback"' in source
    assert '"--commission-rate", "0.000235"' in source
    assert '"--minimum-commission", "5"' in source
    assert '"--slippage-bps", "0", "5"' in source
    assert 'report_alpha360_probabilistic_experiments.py' in source
    assert 'report_alpha360_training_curves.py' in source
    assert '"E6_a_us_four_head"' in source
    assert '"--expected-epochs", "50"' in source
    assert "staging.replace(output)" in source


def test_watchdog_has_exactly_two_authorized_tasks_and_no_termination_command() -> None:
    module = load_finalizer_module()
    assert module.WATCHDOG_TASK_WHITELIST == {
        "Qlib_Alpha360_Probabilistic_Matrix_260828",
        "Qlib_Alpha360_E6_Combined_260828",
    }
    source = FINALIZER.read_text(encoding="utf-8").casefold()
    assert '"schtasks", "/run", "/tn", task_name' in source
    for forbidden in ("taskkill", "stop-process", "terminateprocess", "pkill", "kill -"):
        assert forbidden not in source


def test_watchdog_restarts_only_when_all_three_gates_pass() -> None:
    module = load_finalizer_module()
    base_name = "Qlib_Alpha360_Probabilistic_Matrix_260828"
    combined_name = "Qlib_Alpha360_E6_Combined_260828"
    decisions = module.watchdog_decisions([
        observed("base", base_name),
        observed("combined", combined_name, status="test_ready"),
    ])
    assert decisions[0]["restart_eligible"] is True
    assert decisions[0]["reason"] == "incomplete_task_stopped_no_matching_python"
    assert decisions[1]["restart_eligible"] is False
    assert decisions[1]["reason"] == "pipeline_complete"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"task_state": "Running"}, "scheduled_task_running"),
        ({"matching_python_count": 1}, "matching_pipeline_python_running"),
        ({"status": "selection_ready"}, "pipeline_complete"),
        ({"status": "failed"}, "pipeline_status_unrecognized"),
        ({"status": "unexpected_state"}, "pipeline_status_unrecognized"),
        ({"status": None, "status_error": "missing"}, "pipeline_status_unavailable"),
        (
            {"matching_python_count": None, "process_query_error": "CIM unavailable"},
            "python_process_query_unavailable",
        ),
    ],
)
def test_watchdog_fails_closed_when_any_gate_is_not_proven(changes, reason) -> None:
    module = load_finalizer_module()
    base_name = "Qlib_Alpha360_Probabilistic_Matrix_260828"
    item = observed("base", base_name, **changes)
    combined = observed(
        "combined", "Qlib_Alpha360_E6_Combined_260828", status="test_ready"
    )
    decision = module.watchdog_decisions([item, combined])[0]
    assert decision["restart_eligible"] is False
    assert decision["reason"] == reason


def test_authorized_run_uses_only_schtasks_run_and_rejects_other_names(monkeypatch) -> None:
    module = load_finalizer_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": b"ok", "stderr": b""})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    task_name = "Qlib_Alpha360_E6_Combined_260828"
    module.run_authorized_scheduled_task(task_name)
    assert calls[0][0][-4:] == ["schtasks", "/Run", "/TN", task_name]
    assert calls[0][1]["check"] is True
    with pytest.raises(ValueError, match="not watchdog-authorized"):
        module.run_authorized_scheduled_task("Unrelated_Task")
    assert len(calls) == 1


def test_offline_watchdog_logs_and_defers_without_running_task(monkeypatch, tmp_path) -> None:
    module = load_finalizer_module()
    audit = []
    monkeypatch.setattr(
        module,
        "query_watchdog_snapshot",
        lambda: (_ for _ in ()).throw(module.RemoteWatchdogUnavailable("offline")),
    )
    monkeypatch.setattr(module, "append_watchdog_audit", lambda path, record: audit.append(record))
    monkeypatch.setattr(
        module,
        "run_authorized_scheduled_task",
        lambda task: pytest.fail("offline watchdog must not run a task"),
    )
    assert module.run_training_watchdog(tmp_path / "audit.jsonl", allow_restart=True) is None
    assert len(audit) == 1
    assert audit[0]["event"] == "watchdog_offline_or_unavailable"
    assert audit[0]["action"] == "retry_next_launchd_interval"


def test_status_only_check_never_runs_an_eligible_task(monkeypatch, tmp_path) -> None:
    module = load_finalizer_module()
    snapshot = [
        observed("base", "Qlib_Alpha360_Probabilistic_Matrix_260828"),
        observed("combined", "Qlib_Alpha360_E6_Combined_260828"),
    ]
    audit = []
    monkeypatch.setattr(module, "query_watchdog_snapshot", lambda: snapshot)
    monkeypatch.setattr(module, "append_watchdog_audit", lambda path, record: audit.append(record))
    monkeypatch.setattr(
        module,
        "run_authorized_scheduled_task",
        lambda task: pytest.fail("check-only watchdog must not run a task"),
    )
    assert module.run_training_watchdog(
        tmp_path / "audit.jsonl", allow_restart=False
    ) == snapshot
    assert audit[0]["event"] == "watchdog_check"
    assert audit[0]["allow_restart"] is False


def test_pre_run_recheck_suppresses_restart_when_task_becomes_running(
    monkeypatch, tmp_path
) -> None:
    module = load_finalizer_module()
    base_name = "Qlib_Alpha360_Probabilistic_Matrix_260828"
    combined_name = "Qlib_Alpha360_E6_Combined_260828"
    first = [
        observed("base", base_name),
        observed("combined", combined_name, status="test_ready"),
    ]
    second = [
        observed("base", base_name, task_state="Running"),
        observed("combined", combined_name, status="test_ready"),
    ]
    snapshots = iter([first, second])
    audit = []
    monkeypatch.setattr(module, "query_watchdog_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(module, "append_watchdog_audit", lambda path, record: audit.append(record))
    monkeypatch.setattr(
        module,
        "run_authorized_scheduled_task",
        lambda task: pytest.fail("changed preconditions must suppress /Run"),
    )
    assert module.run_training_watchdog(
        tmp_path / "audit.jsonl", allow_restart=True
    ) == second
    assert [record["event"] for record in audit] == [
        "watchdog_check",
        "watchdog_pre_run_recheck",
    ]


def test_watchdog_submits_only_still_eligible_whitelisted_task(monkeypatch, tmp_path) -> None:
    module = load_finalizer_module()
    base_name = "Qlib_Alpha360_Probabilistic_Matrix_260828"
    combined_name = "Qlib_Alpha360_E6_Combined_260828"
    snapshot = [
        observed("base", base_name),
        observed("combined", combined_name, status="test_ready"),
        observed("unrelated", "Unrelated_Task"),
    ]
    submitted = []
    audit = []
    monkeypatch.setattr(module, "query_watchdog_snapshot", lambda: snapshot)
    monkeypatch.setattr(module, "append_watchdog_audit", lambda path, record: audit.append(record))

    def fake_submit(task_name):
        submitted.append(task_name)
        return type("Completed", (), {"returncode": 0, "stdout": b"ok", "stderr": b""})()

    monkeypatch.setattr(module, "run_authorized_scheduled_task", fake_submit)
    assert module.run_training_watchdog(
        tmp_path / "audit.jsonl", allow_restart=True
    ) == snapshot
    assert submitted == [base_name]
    assert [record["event"] for record in audit] == [
        "watchdog_check",
        "watchdog_pre_run_recheck",
        "watchdog_schtasks_run",
    ]
