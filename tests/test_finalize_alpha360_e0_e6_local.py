from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "script/finalize_alpha360_e0_e6_local.py"


def load_finalizer_module():
    spec = importlib.util.spec_from_file_location("alpha360_e0_e6_finalizer", FINALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, width: int = 2, height: int = 3) -> None:
    """Write a tiny standards-compliant RGB PNG without third-party helpers."""

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)

    raw = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_publishable_staging(module, staging: Path) -> None:
    staging.mkdir()
    candidate_prediction = staging / "evidence/selection_valid_predictions/E1.csv"
    checkpoint = staging / "evidence/checkpoints/E1/close1_close2.pt"
    write_csv(candidate_prediction, "datetime,instrument,score", ["2026-01-01,SH600000,0.1"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    protocol = staging / "evidence/frozen_protocol.json"
    candidate_manifest = staging / "evidence/candidate_manifests/E1.json"
    configuration = staging / "evidence/configurations/E1.json"
    data_manifest = staging / "evidence/data_manifests/data.json"
    for path, value in (
        (protocol, {"schema_version": 1}),
        (candidate_manifest, {"status": "selection_complete"}),
        (configuration, {"epochs": 50}),
        (data_manifest, {"schema_version": 1}),
    ):
        write_json(path, value)

    def reference(path: Path) -> dict:
        return {
            "local_path": path.relative_to(staging).as_posix(),
            "sha256": module.file_sha256(path),
        }

    prediction_reference = reference(candidate_prediction)
    checkpoint_reference = reference(checkpoint)
    selection = {
        "candidate_freeze": {
            "E1": {
                "selection_valid_predictions": {
                    "sha256": prediction_reference["sha256"]
                },
                "checkpoints": {
                    "close1_close2": {"sha256": checkpoint_reference["sha256"]}
                },
            }
        }
    }
    write_json(staging / "selection_manifest.json", selection)
    write_json(staging / "evidence/evidence_index.json", {
        "status": "verified",
        "protocol": reference(protocol),
        "data_manifests": {module.file_sha256(data_manifest): reference(data_manifest)},
        "candidates": {
            "E1": {
                "candidate_manifest": reference(candidate_manifest),
                "configuration": reference(configuration),
                "selection_valid_predictions": prediction_reference,
                "all_checkpoints": {"close1_close2": checkpoint_reference},
            }
        },
    })
    for filename in (
        "selection_valid_ensemble_predictions.csv",
        "test_summary.csv",
        "test_predictions.csv",
        module.FIXED_REFERENCE_STAGING_NAME,
    ):
        write_csv(staging / filename, "value", ["1"])
    write_json(staging / "test_completion_audit.json", {"status": "test_complete"})
    write_json(staging / "evaluated_selection_manifest.json", {"test_files_read": True})
    write_json(staging / "local_lockbox_validation.json", {"status": "verified"})
    write_json(staging / "local_completion.json", {"status": "complete"})

    for experiment in module.REMOTE_EPOCH_METRICS:
        write_csv(
            staging / "epoch_metrics" / f"{experiment}.csv",
            "epoch,valid_nll",
            [f"{epoch},{1 / epoch}" for epoch in range(1, 51)],
        )

    horizons = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")
    for variant in ("mainboard", "all"):
        for fallback in (True, False):
            policy = "fallback" if fallback else "leave_cash"
            directory = staging / f"strict_backtest_{variant}_{policy}"
            write_json(directory / "method.json", {
                "board_variant": variant,
                "fallback": fallback,
                "capital": 100000.0,
                "commission_rate_each_side": 0.000235,
                "minimum_commission": 5.0,
                "lot_size": 100,
                "diagnostic_test_grid_generated": False,
            })
            pretest = directory / "chosen_rule_manifest_pre_test.json"
            write_json(pretest, {"test_opened": False})
            write_json(directory / "evaluated_rule_manifest.json", {
                "test_opened": True,
                "pretest_rule_manifest_sha256": module.file_sha256(pretest),
            })
            write_csv(directory / "selection_valid_chosen_rules.csv", "rule", ["mean_all"])
            write_csv(directory / "selection_valid_uncertainty_grid.csv", "rule", ["mean_all"])
            baseline_rows = [
                f"{horizon},{topk},{slippage},0"
                for horizon in horizons
                for topk in (1, 3, 5, 10)
                for slippage in (0, 5)
            ]
            write_csv(
                directory / "test_baseline_four_horizons.csv",
                "horizon,topk,slippage_bps_each_side,unresolved_exit",
                baseline_rows,
            )
            selected_rows = [
                f"{horizon},{slippage},0"
                for horizon in horizons
                for slippage in (0, 5)
            ]
            write_csv(
                directory / "test_selected_uncertainty_rules.csv",
                "horizon,slippage_bps_each_side,unresolved_exit",
                selected_rows,
            )
            write_csv(directory / "label_alignment.csv", "split,aligned", ["test,1"])
            write_csv(
                directory / "uncertainty_selection_test_comparison.csv",
                "horizon,value",
                ["close1_close2,1"],
            )

    for policy in ("fallback", "leave_cash"):
        report = staging / f"report_{policy}"
        for filename in module.REPORT_REQUIRED_FILES:
            path = report / filename
            if path.suffix == ".csv":
                if filename == "strict_execution_summary.csv":
                    write_csv(path, "row,unresolved_exit", [f"{index},0" for index in range(16)])
                else:
                    write_csv(path, "value", ["1"])
            elif path.suffix == ".png":
                write_png(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("evidence\n", encoding="utf-8")

    write_csv(
        staging / "training_curves/training_curve_summary.csv",
        "experiment,value",
        [f"E{index},{index}" for index in range(7)],
    )
    write_png(staging / "training_curves/training_curves.png", width=4, height=5)
    (staging / "training_curves/training_curves.md").write_text(
        "training evidence\n", encoding="utf-8"
    )


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
    assert '"--fixed-reference-summary", str(staging / FIXED_REFERENCE_STAGING_NAME)' in source
    assert '"--execution-policy", execution_policy' in source
    assert "staging.replace(output)" in source
    assert "FIXED_REFERENCE_EXPECTED_SHA256" in source


def test_fixed_reference_source_is_exactly_the_authorized_history() -> None:
    module = load_finalizer_module()
    assert module.FIXED_REFERENCE_SOURCE == (
        ROOT
        / ".qlibAssistant/analysis/fixed_fold3_full_intraday_exit_20260821/summary.csv"
    )


def test_stage_fixed_reference_copies_and_records_sha(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_finalizer_module()
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(module, "FIXED_REFERENCE_SOURCE", source)
    monkeypatch.setattr(
        module,
        "FIXED_REFERENCE_EXPECTED_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    audit = module.stage_fixed_reference(staging)
    copied = staging / module.FIXED_REFERENCE_STAGING_NAME
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert copied.read_bytes() == source.read_bytes()
    assert audit == {
        "source": str(source.resolve()),
        "staged_path": module.FIXED_REFERENCE_STAGING_NAME,
        "sha256": expected_sha,
        "size": source.stat().st_size,
    }


def test_stage_fixed_reference_missing_or_duplicate_destination_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_finalizer_module()
    staging = tmp_path / "staging"
    staging.mkdir()
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(module, "FIXED_REFERENCE_SOURCE", missing)
    with pytest.raises(FileNotFoundError, match="Missing Fixed Ensemble reference"):
        module.stage_fixed_reference(staging)

    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(module, "FIXED_REFERENCE_SOURCE", source)
    with pytest.raises(RuntimeError, match="differs from the frozen historical artifact"):
        module.stage_fixed_reference(staging)

    monkeypatch.setattr(
        module,
        "FIXED_REFERENCE_EXPECTED_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    destination = staging / module.FIXED_REFERENCE_STAGING_NAME
    destination.write_text("preexisting", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.stage_fixed_reference(staging)
    assert destination.read_text(encoding="utf-8") == "preexisting"


def test_stage_evidence_bundle_copies_and_authenticates_selected_checkpoint_audits(
    monkeypatch, tmp_path: Path,
) -> None:
    module = load_finalizer_module()
    staging = tmp_path / "staging"
    staging.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"schema_version":1}', encoding="utf-8")
    monkeypatch.setattr(module, "PROTOCOL_SOURCE", protocol)
    horizons = ["open1_close2", "close1_open2", "open1_open2", "close1_close2"]
    remote_payloads = {
        "E:/candidate/selection_candidate_manifest.json": b'{"status":"selection_complete"}',
        "E:/candidate/configuration.json": b'{"epochs":50}',
        "E:/candidate/selection_valid_predictions.csv": b'datetime,instrument,score\n2026-01-01,SH600000,0.1\n',
        "E:/unselected/selection_candidate_manifest.json": b'{"status":"selection_complete"}',
        "E:/unselected/configuration.json": b'{"epochs":50}',
        "E:/unselected/selection_valid_predictions.csv": b'datetime,instrument,score\n2026-01-01,SZ000001,0.2\n',
        "E:/data/manifest.json": b'{"schema_version":1}',
    }
    for horizon in horizons:
        remote_payloads[f"E:/candidate/best_{horizon}.pt"] = f"weights-{horizon}".encode()
    for horizon in horizons[:2]:
        remote_payloads[f"E:/unselected/best_{horizon}.pt"] = (
            f"unselected-weights-{horizon}".encode()
        )
    checkpoint_hashes = {
        horizon: hashlib.sha256(remote_payloads[f"E:/candidate/best_{horizon}.pt"]).hexdigest()
        for horizon in horizons
    }
    remote_payloads["E:/candidate/test_completion_audit.json"] = json.dumps({
        "status": "test_complete",
        "test_read": True,
        "candidate_name": "candidate",
        "selected_horizons": horizons,
        "checkpoint_sha256": checkpoint_hashes,
    }).encode()

    def reference(path: str) -> dict:
        return {"path": path, "sha256": hashlib.sha256(remote_payloads[path]).hexdigest()}

    selection = {
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "candidates": {
            "candidate": "E:\\candidate",
            "unselected": "E:\\unselected",
        },
        "candidate_freeze": {
            "candidate": {
                "candidate_manifest": reference("E:/candidate/selection_candidate_manifest.json"),
                "configuration": reference("E:/candidate/configuration.json"),
                "selection_valid_predictions": reference(
                    "E:/candidate/selection_valid_predictions.csv"
                ),
                "data_manifest": reference("E:/data/manifest.json"),
                "checkpoints": {
                    horizon: reference(f"E:/candidate/best_{horizon}.pt")
                    for horizon in horizons
                },
            },
            "unselected": {
                "candidate_manifest": reference(
                    "E:/unselected/selection_candidate_manifest.json"
                ),
                "configuration": reference("E:/unselected/configuration.json"),
                "selection_valid_predictions": reference(
                    "E:/unselected/selection_valid_predictions.csv"
                ),
                "data_manifest": reference("E:/data/manifest.json"),
                "checkpoints": {
                    horizon: reference(f"E:/unselected/best_{horizon}.pt")
                    for horizon in horizons[:2]
                },
            },
        },
        "selections": {
            horizon: {
                "selected_components": ["candidate"],
                "selected_checkpoint_sha256": {"candidate": checkpoint_hashes[horizon]},
            }
            for horizon in horizons
        },
    }
    (staging / "selection_manifest.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )

    def fake_copy(remote_path: str, destination: Path) -> None:
        key = remote_path.replace("\\", "/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote_payloads[key])

    monkeypatch.setattr(module, "copy_remote_file", fake_copy)
    bundle = module.stage_evidence_bundle(
        staging, {"selected_candidates": ["candidate"]}
    )
    assert bundle["status"] == "verified"
    assert bundle["selected_candidates"] == ["candidate"]
    assert set(bundle["candidates"]) == {"candidate", "unselected"}
    assert set(bundle["candidates"]["candidate"]["all_checkpoints"]) == set(horizons)
    assert set(bundle["candidates"]["candidate"]["selected_checkpoints"]) == set(horizons)
    assert set(bundle["candidates"]["unselected"]["all_checkpoints"]) == set(horizons[:2])
    assert "selected_checkpoints" not in bundle["candidates"]["unselected"]
    assert (
        staging / bundle["candidates"]["unselected"]["selection_valid_predictions"]["local_path"]
    ).is_file()
    assert (
        staging
        / bundle["candidates"]["unselected"]["all_checkpoints"][horizons[0]]["local_path"]
    ).is_file()
    assert (staging / "evidence/evidence_index.json").is_file()
    assert (staging / "evidence/test_audits/candidate/test_completion_audit.json").is_file()


def test_fixed_reference_is_staged_only_after_test_ready_gate_and_recorded() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    gate = 'if status.get("status") != "test_ready" or status.get("test_read") is not True:'
    stage = "fixed_reference = stage_fixed_reference(staging)"
    completion = '"fixed_reference": fixed_reference'
    assert source.index(gate) < source.index(stage) < source.index(completion)


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


def test_epoch_validation_requires_all_seven_histories_and_exact_epochs(
    tmp_path: Path,
) -> None:
    module = load_finalizer_module()
    staging = tmp_path / "staging"
    (staging / "epoch_metrics").mkdir(parents=True)
    for experiment in module.REMOTE_EPOCH_METRICS:
        write_csv(
            staging / "epoch_metrics" / f"{experiment}.csv",
            "epoch,value",
            [f"{epoch},1" for epoch in range(1, 51)],
        )
    assert len(module.validate_epoch_histories(staging)) == 7

    malformed = staging / "epoch_metrics/E6_a_us_four_head.csv"
    write_csv(malformed, "epoch,value", [f"{epoch},1" for epoch in range(2, 52)])
    with pytest.raises(ValueError, match="contiguous 1..50"):
        module.validate_epoch_histories(staging)


def test_final_staging_validation_accepts_complete_delivery_and_rejects_unresolved_exit(
    tmp_path: Path,
) -> None:
    module = load_finalizer_module()
    staging = tmp_path / "staging"
    build_publishable_staging(module, staging)
    validation = module.validate_final_staging(staging)
    assert validation["status"] == "verified"
    assert validation["epoch_histories"] == 7
    assert validation["strict_backtests"] == 4
    assert validation["reports"] == 2
    assert validation["png_files"] == 5

    strict_summary = staging / "report_fallback/strict_execution_summary.csv"
    write_csv(strict_summary, "row,unresolved_exit", ["0,1"] + [f"{index},0" for index in range(1, 16)])
    with pytest.raises(RuntimeError, match="unresolved exits"):
        module.validate_final_staging(staging)


def test_artifact_index_covers_directory_with_hashes_metadata_and_core_flags(
    tmp_path: Path,
) -> None:
    module = load_finalizer_module()
    staging = tmp_path / "staging"
    staging.mkdir()
    write_csv(staging / "test_predictions.csv", "value", ["1", "2"])
    write_png(staging / "report_fallback/plot.png", width=7, height=9)
    note = staging / "evidence/note.md"
    note.parent.mkdir(parents=True)
    note.write_text("offline evidence\n", encoding="utf-8")

    value = module.build_artifact_index(staging, {"test_predictions.csv"})
    verification = module.validate_artifact_index(staging)
    records = {record["relative_path"]: record for record in value["artifacts"]}
    assert set(records) == {
        "test_predictions.csv",
        "report_fallback/plot.png",
        "evidence/note.md",
    }
    assert records["test_predictions.csv"]["csv_rows"] == 2
    assert records["test_predictions.csv"]["core"] is True
    assert records["test_predictions.csv"]["generated_stage"] == "remote_lockbox"
    assert records["report_fallback/plot.png"]["png_width"] == 7
    assert records["report_fallback/plot.png"]["png_height"] == 9
    assert records["report_fallback/plot.png"]["generated_stage"] == "report"
    assert records["evidence/note.md"]["generated_stage"] == "offline_evidence"
    assert verification["status"] == "verified"
    assert verification["indexed_files"] == 3
    assert module.ARTIFACT_INDEX_NAME not in records


def test_main_validates_then_indexes_then_atomically_publishes() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    validate = "delivery_validation = validate_final_staging(staging)"
    index = "build_artifact_index(staging, core_artifacts)"
    verify_index = "validate_artifact_index(staging)"
    publish = "staging.replace(output)"
    assert source.index(validate) < source.index(index) < source.index(verify_index) < source.index(publish)
    assert '"prepublish_validation": "local_prepublish_validation.json"' in source
    assert '"artifact_index": ARTIFACT_INDEX_NAME' in source
