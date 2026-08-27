#!/usr/bin/env python3
"""Fetch the completed E0--E6 lockbox and build strict local reports.

This is deliberately a one-shot finalizer, not a resident polling process.  It
exits with status 3 while the remote pipeline is still running, which avoids a
hidden background Python process on the Mac.  Once the remote status is
``test_ready``, artifacts are copied into a staging directory, both strict
board variants are evaluated, the read-only report is generated, and one
directory rename publishes the completed local result.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
HOST = "100.76.140.38"
USER = "12600"
PORT = "22"
HOST_KEY_ALIAS = "192.168.1.7"
REMOTE_ROOT = "E:/qlibAssistant/.qlibAssistant/remote_runs/alpha360_e6_full_260828"
REMOTE_BASE_ROOT = "E:/qlibAssistant/.qlibAssistant/remote_runs/alpha360_probabilistic_matrix_260828"
REMOTE_E0_ROOT = "E:/qlibAssistant/.qlibAssistant/remote_runs/alpha360_cross_stock_fold3_120m_v2_260828/run"
WATCHDOG_AUDIT_LOG = (
    ROOT / ".qlibAssistant/analysis/alpha360_e0_e6_260828_watchdog.jsonl"
)
FIXED_REFERENCE_SOURCE = (
    ROOT
    / ".qlibAssistant/analysis/fixed_fold3_full_intraday_exit_20260821/summary.csv"
)
FIXED_REFERENCE_STAGING_NAME = "fixed_reference_summary.csv"
FIXED_REFERENCE_EXPECTED_SHA256 = (
    "32db848d16978ff6a3f275931c22d85bd86790ba4510ed1044436d4fd30b67c0"
)
PROTOCOL_SOURCE = (
    ROOT / "script/alpha360_experiments/fixed_fold3_probabilistic_cross_market_v1.json"
)
WATCHDOG_TARGETS = (
    {
        "key": "base",
        "task_name": "Qlib_Alpha360_Probabilistic_Matrix_260828",
        "status_path": f"{REMOTE_BASE_ROOT}/pipeline_status.json",
        "process_marker": REMOTE_BASE_ROOT.replace("/", "\\"),
        "complete_statuses": frozenset({"selection_ready", "test_ready"}),
        "incomplete_statuses": frozenset({
            "waiting_for_e0", "materializing_e0_selection", "skipped_selection_ready",
            "training", "selecting_ensemble",
        }),
    },
    {
        "key": "combined",
        "task_name": "Qlib_Alpha360_E6_Combined_260828",
        "status_path": f"{REMOTE_ROOT}/pipeline_status.json",
        "process_marker": REMOTE_ROOT.replace("/", "\\"),
        "complete_statuses": frozenset({"test_ready"}),
        "incomplete_statuses": frozenset({
            "waiting_for_e0_e5_selection", "training_e6",
            "selecting_e0_e6_ensemble", "materializing_selected_test",
            "selected_candidate_test_complete",
        }),
    },
)
WATCHDOG_TASK_WHITELIST = frozenset(value["task_name"] for value in WATCHDOG_TARGETS)
REMOTE_FILES = {
    "pipeline_status.json": "pipeline_status.json",
    "pipeline.log": "pipeline.log",
    "selection_manifest.json": "selection_e0_e6/selection_manifest.json",
    "selection_valid_ensemble_predictions.csv": (
        "selection_e0_e6/selection_valid_ensemble_predictions.csv"
    ),
    "test_summary.csv": "test_evaluation_e0_e6/test_summary.csv",
    "test_predictions.csv": "test_evaluation_e0_e6/test_predictions.csv",
    "test_completion_audit.json": "test_evaluation_e0_e6/test_completion_audit.json",
    "evaluated_selection_manifest.json": (
        "test_evaluation_e0_e6/evaluated_selection_manifest.json"
    ),
}


class RemoteWatchdogUnavailable(RuntimeError):
    """The remote watchdog snapshot could not be authenticated or retrieved."""


REMOTE_EPOCH_METRICS = {
    "E0_joint_three_leg": f"{REMOTE_E0_ROOT}/epoch_metrics.csv",
    "E1_shared_four_head": f"{REMOTE_BASE_ROOT}/E1_shared_four_head/epoch_metrics.csv",
    "E2_single_open1_close2": f"{REMOTE_BASE_ROOT}/E2_single_open1_close2/epoch_metrics.csv",
    "E3_single_close1_open2": f"{REMOTE_BASE_ROOT}/E3_single_close1_open2/epoch_metrics.csv",
    "E4_single_open1_open2": f"{REMOTE_BASE_ROOT}/E4_single_open1_open2/epoch_metrics.csv",
    "E5_single_close1_close2": f"{REMOTE_BASE_ROOT}/E5_single_close1_close2/epoch_metrics.csv",
    "E6_a_us_four_head": f"{REMOTE_ROOT}/E6_a_us_four_head/epoch_metrics.csv",
}


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-o", f"HostKeyAlias={HOST_KEY_ALIAS}",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10",
        "-p", PORT,
        f"{USER}@{HOST}",
    ]


def scp_base() -> list[str]:
    return [
        "scp",
        "-o", f"HostKeyAlias={HOST_KEY_ALIAS}",
        "-o", "StrictHostKeyChecking=yes",
        "-P", PORT,
    ]


def _watchdog_powershell() -> str:
    targets = [
        {
            "key": value["key"],
            "task_name": value["task_name"],
            "status_path": value["status_path"].replace("/", "\\"),
            "process_marker": value["process_marker"],
        }
        for value in WATCHDOG_TARGETS
    ]
    target_json = json.dumps(targets, ensure_ascii=True).replace("'", "''")
    return rf"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
[Console]::OutputEncoding = $utf8NoBom
$targets = ConvertFrom-Json @'
{target_json}
'@
$processError = $null
$pythonProcesses = @()
try {{
    $pythonProcesses = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction Stop)
}} catch {{
    $processError = $_.Exception.Message
}}
$result = @()
foreach ($target in $targets) {{
    $statusBody = $null
    $statusError = $null
    try {{
        $statusBody = Get-Content -LiteralPath $target.status_path -Raw -ErrorAction Stop | ConvertFrom-Json
    }} catch {{
        $statusError = $_.Exception.Message
    }}
    $taskExists = $false
    $taskState = $null
    $taskError = $null
    try {{
        $task = Get-ScheduledTask -TaskName $target.task_name -ErrorAction Stop
        $taskExists = $true
        $taskState = [string]$task.State
    }} catch {{
        $taskError = $_.Exception.Message
    }}
    $matching = @()
    if ($null -eq $processError) {{
        $matching = @($pythonProcesses | Where-Object {{
            $_.CommandLine -and
            $_.CommandLine.IndexOf($target.process_marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        }})
    }}
    $result += [pscustomobject]@{{
        key = [string]$target.key
        task_name = [string]$target.task_name
        pipeline_status = if ($null -ne $statusBody) {{ [string]$statusBody.status }} else {{ $null }}
        pipeline_updated = if ($null -ne $statusBody) {{ [string]$statusBody.updated }} else {{ $null }}
        status_body = $statusBody
        status_error = $statusError
        task_exists = $taskExists
        task_state = $taskState
        task_error = $taskError
        matching_python_count = if ($null -eq $processError) {{ $matching.Count }} else {{ $null }}
        matching_python_pids = @($matching | ForEach-Object {{ [int]$_.ProcessId }})
        process_query_error = $processError
    }}
}}
$result | ConvertTo-Json -Depth 12 -Compress
""".strip()


def _decode_json_output(raw: bytes) -> object:
    text = raw.decode("utf-8-sig", errors="strict").strip()
    candidates = [text, *reversed([line.strip() for line in text.splitlines()])]
    for candidate in candidates:
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Remote watchdog returned no JSON payload")


def query_watchdog_snapshot() -> list[dict]:
    encoded = base64.b64encode(_watchdog_powershell().encode("utf-16le")).decode("ascii")
    command = [
        *ssh_base(),
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
        "-EncodedCommand", encoded,
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, timeout=30,
        )
        value = _decode_json_output(completed.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
        raise RemoteWatchdogUnavailable(str(error)) from error
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RemoteWatchdogUnavailable("Remote watchdog snapshot is not a JSON object list")
    expected_keys = {target["key"] for target in WATCHDOG_TARGETS}
    observed_keys = [item.get("key") for item in value]
    if len(value) != len(WATCHDOG_TARGETS) or set(observed_keys) != expected_keys:
        raise RemoteWatchdogUnavailable(
            f"Remote watchdog snapshot identity mismatch: {observed_keys}"
        )
    return value


def watchdog_decisions(snapshot: list[dict]) -> list[dict]:
    """Return fail-closed decisions for the two explicitly authorized tasks."""

    by_key = {item.get("key"): item for item in snapshot if isinstance(item, dict)}
    decisions: list[dict] = []
    for target in WATCHDOG_TARGETS:
        observed = by_key.get(target["key"])
        decision = {
            "key": target["key"],
            "task_name": target["task_name"],
            "restart_eligible": False,
            "reason": "snapshot_missing",
        }
        if observed is None:
            decisions.append(decision)
            continue
        if observed.get("task_name") != target["task_name"]:
            decision["reason"] = "snapshot_identity_mismatch"
        elif observed.get("status_error") or not isinstance(
            observed.get("pipeline_status"), str
        ):
            decision["reason"] = "pipeline_status_unavailable"
        elif observed["pipeline_status"] in target["complete_statuses"]:
            decision["reason"] = "pipeline_complete"
        elif observed["pipeline_status"] not in target["incomplete_statuses"]:
            decision["reason"] = "pipeline_status_unrecognized"
        elif not observed.get("task_exists") or observed.get("task_error"):
            decision["reason"] = "scheduled_task_unavailable"
        elif str(observed.get("task_state", "")).casefold() == "running":
            decision["reason"] = "scheduled_task_running"
        elif observed.get("process_query_error"):
            decision["reason"] = "python_process_query_unavailable"
        elif observed.get("matching_python_count") != 0:
            decision["reason"] = "matching_pipeline_python_running"
        else:
            decision["restart_eligible"] = True
            decision["reason"] = "incomplete_task_stopped_no_matching_python"
        decision.update({
            "pipeline_status": observed.get("pipeline_status"),
            "pipeline_updated": observed.get("pipeline_updated"),
            "task_state": observed.get("task_state"),
            "matching_python_count": observed.get("matching_python_count"),
            "matching_python_pids": observed.get("matching_python_pids", []),
        })
        decisions.append(decision)
    return decisions


def append_watchdog_audit(path: Path, record: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_authorized_scheduled_task(task_name: str) -> subprocess.CompletedProcess:
    if task_name not in WATCHDOG_TASK_WHITELIST:
        raise ValueError(f"Scheduled task is not watchdog-authorized: {task_name}")
    return subprocess.run(
        [*ssh_base(), "schtasks", "/Run", "/TN", task_name],
        check=True,
        capture_output=True,
        timeout=30,
    )


def run_training_watchdog(audit_log: Path, *, allow_restart: bool) -> list[dict] | None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        snapshot = query_watchdog_snapshot()
    except RemoteWatchdogUnavailable as error:
        append_watchdog_audit(audit_log, {
            "timestamp": timestamp,
            "event": "watchdog_offline_or_unavailable",
            "host": HOST,
            "error": str(error),
            "action": "retry_next_launchd_interval",
        })
        print(f"Remote watchdog unavailable; retry next interval: {error}")
        return None

    decisions = watchdog_decisions(snapshot)
    append_watchdog_audit(audit_log, {
        "timestamp": timestamp,
        "event": "watchdog_check",
        "host": HOST,
        "allow_restart": allow_restart,
        "decisions": decisions,
    })
    if not allow_restart:
        return snapshot

    eligible = [decision for decision in decisions if decision["restart_eligible"]]
    if not eligible:
        return snapshot

    # Close the obvious observation/action race: immediately re-read all three
    # gates before issuing any authorized /Run.  If the host drops offline or a
    # task/process state changes, fail closed and let the next interval retry.
    try:
        recheck_snapshot = query_watchdog_snapshot()
    except RemoteWatchdogUnavailable as error:
        append_watchdog_audit(audit_log, {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": "watchdog_pre_run_recheck_unavailable",
            "host": HOST,
            "error": str(error),
            "action": "no_restart_retry_next_launchd_interval",
        })
        return None
    recheck_decisions = watchdog_decisions(recheck_snapshot)
    append_watchdog_audit(audit_log, {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": "watchdog_pre_run_recheck",
        "host": HOST,
        "decisions": recheck_decisions,
    })
    initially_eligible = {decision["task_name"] for decision in eligible}
    for decision in recheck_decisions:
        if decision["task_name"] not in initially_eligible:
            continue
        if not decision["restart_eligible"]:
            continue
        task_name = decision["task_name"]
        result_record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": "watchdog_schtasks_run",
            "host": HOST,
            "task_name": task_name,
            "preconditions": decision,
        }
        try:
            completed = run_authorized_scheduled_task(task_name)
            result_record.update({
                "result": "submitted",
                "returncode": completed.returncode,
                "stdout": completed.stdout.decode("utf-8", errors="replace").strip()[:2000],
                "stderr": completed.stderr.decode("utf-8", errors="replace").strip()[:2000],
            })
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            result_record.update({
                "result": "failed_retry_next_interval",
                "error": str(error),
            })
        append_watchdog_audit(audit_log, result_record)
    return recheck_snapshot


def combined_status_from_snapshot(snapshot: list[dict]) -> dict:
    for item in snapshot:
        if (
            item.get("key") == "combined"
            and item.get("task_name") == "Qlib_Alpha360_E6_Combined_260828"
            and isinstance(item.get("status_body"), dict)
        ):
            return item["status_body"]
    raise RemoteWatchdogUnavailable("Combined pipeline status is absent from watchdog snapshot")


def remote_status() -> dict:
    command = [
        *ssh_base(),
        "cmd", "/c", "type", REMOTE_ROOT.replace("/", "\\") + "\\pipeline_status.json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    text = completed.stdout.decode("utf-8-sig").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Remote pipeline status is not a JSON object")
    return value


def checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_downloaded_reference(reference: dict | None, path: Path, label: str) -> None:
    if not isinstance(reference, dict) or not isinstance(reference.get("sha256"), str):
        raise ValueError(f"Missing authenticated reference for {label}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if reference["sha256"] != actual:
        raise RuntimeError(f"Downloaded {label} hash mismatch")
    expected_size = reference.get("size")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise RuntimeError(f"Downloaded {label} size mismatch")


def validate_downloaded_lockbox(staging: Path, expected_status: dict) -> dict:
    """Authenticate the copied pre-Test freeze and completed Test artifacts."""

    copied_status = read_json(staging / "pipeline_status.json")
    for key, expected in {
        "status": "test_ready", "test_access_authorized": True, "test_read": True,
    }.items():
        if expected_status.get(key) != expected or copied_status.get(key) != expected:
            raise RuntimeError(f"Pipeline status does not authorize completed Test access: {key}")
    for key in ("selected_candidates", "selection_manifest", "test_predictions", "test_summary"):
        if copied_status.get(key) != expected_status.get(key):
            raise RuntimeError(f"Pipeline status changed while artifacts were copied: {key}")

    selection_path = staging / "selection_manifest.json"
    selection_predictions = staging / "selection_valid_ensemble_predictions.csv"
    test_predictions = staging / "test_predictions.csv"
    test_summary = staging / "test_summary.csv"
    completion_path = staging / "test_completion_audit.json"
    evaluated_path = staging / "evaluated_selection_manifest.json"
    selection = read_json(selection_path)
    completion = read_json(completion_path)
    evaluated = read_json(evaluated_path)

    if selection.get("selection_split") != "selection_valid":
        raise ValueError("Downloaded selection manifest is not a Selection-valid freeze")
    if selection.get("test_files_read") is not False:
        raise ValueError("Downloaded selection manifest is not pre-Test")
    if selection.get("selection_valid_ensemble_predictions_sha256") != file_sha256(
        selection_predictions
    ):
        raise RuntimeError("Frozen Selection prediction hash mismatch")
    if completion.get("status") != "test_complete" or completion.get("test_read") is not True:
        raise ValueError("Downloaded aggregate Test completion audit is incomplete")
    verify_downloaded_reference(
        completion.get("selection_manifest"), selection_path, "selection manifest"
    )
    for name, path in {
        "test_predictions.csv": test_predictions,
        "test_summary.csv": test_summary,
        "evaluated_selection_manifest.json": evaluated_path,
    }.items():
        verify_downloaded_reference(
            completion.get("artifacts", {}).get(name), path, name
        )
    if evaluated.get("test_files_read") is not True:
        raise ValueError("Evaluated manifest does not record Test access")
    immutable_fields = (
        "protocol_sha256", "selection_split", "candidate_freeze", "selections",
        "selection_scoring_keys", "selection_valid_ensemble_predictions_sha256",
    )
    for field in immutable_fields:
        if evaluated.get(field) != selection.get(field):
            raise RuntimeError(f"Evaluated manifest changed frozen field: {field}")
    if (
        completion.get("protocol_segments") != evaluated.get("protocol_segments")
        or evaluated.get("protocol_segments") != selection.get("protocol_segments")
    ):
        raise RuntimeError("Test completion segments differ from the frozen evaluation")
    return {
        "status": "verified",
        "selection_manifest_sha256": file_sha256(selection_path),
        "selection_predictions_sha256": file_sha256(selection_predictions),
        "test_predictions_sha256": file_sha256(test_predictions),
        "test_summary_sha256": file_sha256(test_summary),
        "test_completion_audit_sha256": file_sha256(completion_path),
        "evaluated_selection_manifest_sha256": file_sha256(evaluated_path),
    }


def fetch_artifacts(staging: Path) -> None:
    for local_name, remote_relative in REMOTE_FILES.items():
        source = f"{USER}@{HOST}:{REMOTE_ROOT}/{remote_relative}"
        checked([*scp_base(), source, str(staging / local_name)])

    epoch_directory = staging / "epoch_metrics"
    epoch_directory.mkdir()
    for experiment, remote_path in REMOTE_EPOCH_METRICS.items():
        source = f"{USER}@{HOST}:{remote_path}"
        checked([*scp_base(), source, str(epoch_directory / f"{experiment}.csv")])


def copy_remote_file(remote_path: str, destination: Path) -> None:
    """Copy one authenticated Windows path without weakening SSH host checks."""

    normalized = str(remote_path).replace("\\", "/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked([*scp_base(), f"{USER}@{HOST}:{normalized}", str(destination)])


def copy_authenticated_reference(reference: dict, destination: Path, label: str) -> dict:
    if not isinstance(reference, dict):
        raise ValueError(f"Missing evidence reference for {label}")
    path = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ValueError(f"Incomplete evidence reference for {label}")
    copy_remote_file(path, destination)
    actual = file_sha256(destination)
    if actual != expected:
        raise RuntimeError(f"Evidence hash mismatch for {label}")
    evidence_root = next(
        (parent for parent in destination.parents if parent.name == "evidence"), None
    )
    local_path = (
        str(Path("evidence") / destination.relative_to(evidence_root))
        if evidence_root is not None else str(destination)
    )
    return {
        "remote_path": path,
        "local_path": local_path,
        "sha256": actual,
        "size": destination.stat().st_size,
    }


def selected_component_names(selection: dict) -> set[str]:
    selections = selection.get("selections")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("Selection manifest has no horizon selections")
    names: set[str] = set()
    for horizon, body in selections.items():
        components = body.get("selected_components") if isinstance(body, dict) else None
        if not isinstance(components, list) or not components:
            raise ValueError(f"Selection manifest has no components for {horizon}")
        names.update(str(value) for value in components)
    return names


def stage_evidence_bundle(staging: Path, status: dict) -> dict:
    """Build a locally auditable protocol/config/checkpoint/Test-audit bundle."""

    selection = read_json(staging / "selection_manifest.json")
    evidence = staging / "evidence"
    evidence.mkdir()
    protocol_destination = evidence / "frozen_protocol.json"
    shutil.copy2(PROTOCOL_SOURCE, protocol_destination)
    protocol_hash = file_sha256(protocol_destination)
    if protocol_hash != selection.get("protocol_sha256"):
        raise RuntimeError("Local frozen protocol differs from the remote Selection freeze")

    candidate_freeze = selection.get("candidate_freeze")
    candidate_paths = selection.get("candidates")
    if not isinstance(candidate_freeze, dict) or not isinstance(candidate_paths, dict):
        raise ValueError("Selection manifest is missing candidate evidence")
    selected = selected_component_names(selection)
    status_selected = {str(value) for value in status.get("selected_candidates", [])}
    if selected != status_selected:
        raise RuntimeError("Selected candidates differ between status and Selection manifest")

    records: dict[str, dict] = {}
    data_manifest_records: dict[str, dict] = {}
    for name, freeze in sorted(candidate_freeze.items()):
        if not isinstance(freeze, dict):
            raise ValueError(f"Invalid candidate freeze for {name}")
        candidate_record = {
            "candidate_manifest": copy_authenticated_reference(
                freeze.get("candidate_manifest"),
                evidence / "candidate_manifests" / f"{name}.json",
                f"{name} candidate manifest",
            ),
            "configuration": copy_authenticated_reference(
                freeze.get("configuration"),
                evidence / "configurations" / f"{name}.json",
                f"{name} configuration",
            ),
        }
        data_reference = freeze.get("data_manifest")
        data_hash = data_reference.get("sha256") if isinstance(data_reference, dict) else None
        if not isinstance(data_hash, str):
            raise ValueError(f"Missing data manifest hash for {name}")
        if data_hash not in data_manifest_records:
            data_manifest_records[data_hash] = copy_authenticated_reference(
                data_reference,
                evidence / "data_manifests" / f"{data_hash}.json",
                f"{name} data manifest",
            )
        candidate_record["data_manifest_sha256"] = data_hash

        if name in selected:
            candidate_directory = candidate_paths.get(name)
            if not isinstance(candidate_directory, str):
                raise ValueError(f"Missing candidate directory for selected component {name}")
            candidate_directory = candidate_directory.rstrip("/\\")
            audit_directory = evidence / "test_audits" / name
            audit_filenames = (
                ("test_materialization_manifest.json",)
                if name == "E0_joint_three_leg"
                else ("test_completion_audit.json",)
            )
            for filename in audit_filenames:
                destination = audit_directory / filename
                copy_remote_file(candidate_directory + "\\" + filename, destination)
            test_audit = read_json(audit_directory / audit_filenames[0])
            expected_horizons = sorted(
                horizon for horizon, body in selection["selections"].items()
                if name in body["selected_components"]
            )
            actual_horizons = sorted(
                test_audit.get("selected_horizons", test_audit.get("horizons", []))
            )
            if name == "E0_joint_three_leg":
                if (
                    test_audit.get("command") != "evaluate-test"
                    or test_audit.get("test_read") is not True
                ):
                    raise ValueError("E0 Test materialization audit is incomplete")
            elif (
                test_audit.get("status") != "test_complete"
                or test_audit.get("test_read") is not True
                or test_audit.get("candidate_name") != name
            ):
                raise ValueError(f"Test completion audit is incomplete for {name}")
            if actual_horizons != expected_horizons:
                raise RuntimeError(f"Test audit horizons differ from Selection for {name}")
            if name == "E0_joint_three_leg":
                audit_checkpoints = {
                    horizon: body.get("sha256")
                    for horizon, body in test_audit.get("horizon_checkpoints", {}).items()
                }
            else:
                audit_checkpoints = test_audit.get("checkpoint_sha256", {})
            expected_checkpoint_hashes = {
                horizon: selection["selections"][horizon]["selected_checkpoint_sha256"][name]
                for horizon in expected_horizons
            }
            if audit_checkpoints != expected_checkpoint_hashes:
                raise RuntimeError(f"Test audit checkpoints differ from Selection for {name}")
            checkpoints = freeze.get("checkpoints")
            if not isinstance(checkpoints, dict):
                raise ValueError(f"Missing checkpoint freeze for {name}")
            candidate_record["selected_checkpoints"] = {}
            for horizon, body in selection["selections"].items():
                if name not in body["selected_components"]:
                    continue
                candidate_record["selected_checkpoints"][horizon] = copy_authenticated_reference(
                    checkpoints.get(horizon),
                    evidence / "checkpoints" / name / f"{horizon}.pt",
                    f"{name}/{horizon} checkpoint",
                )
            candidate_record["test_audits"] = {
                filename: {
                    "local_path": str(
                        Path("evidence/test_audits") / name / filename
                    ),
                    "sha256": file_sha256(audit_directory / filename),
                    "size": (audit_directory / filename).stat().st_size,
                }
                for filename in audit_filenames
            }
        records[name] = candidate_record

    bundle = {
        "status": "verified",
        "protocol": {
            "local_path": "evidence/frozen_protocol.json",
            "sha256": protocol_hash,
            "size": protocol_destination.stat().st_size,
        },
        "selected_candidates": sorted(selected),
        "candidates": records,
        "data_manifests": data_manifest_records,
    }
    (evidence / "evidence_index.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle


def stage_fixed_reference(staging: Path) -> dict:
    """Copy and authenticate the fixed external history after Test completion."""

    source = FIXED_REFERENCE_SOURCE.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing Fixed Ensemble reference: {source}")
    destination = staging / FIXED_REFERENCE_STAGING_NAME
    if destination.exists():
        raise FileExistsError(destination)
    source_sha_before = file_sha256(source)
    if source_sha_before != FIXED_REFERENCE_EXPECTED_SHA256:
        raise RuntimeError(
            "Fixed Ensemble reference differs from the frozen historical artifact: "
            f"expected {FIXED_REFERENCE_EXPECTED_SHA256}, got {source_sha_before}"
        )
    shutil.copy2(source, destination)
    source_sha_after = file_sha256(source)
    copied_sha = file_sha256(destination)
    if source_sha_before != source_sha_after:
        raise RuntimeError("Fixed Ensemble reference changed while it was copied")
    if copied_sha != source_sha_before:
        raise RuntimeError("Copied Fixed Ensemble reference hash mismatch")
    return {
        "source": str(source),
        "staged_path": FIXED_REFERENCE_STAGING_NAME,
        "sha256": copied_sha,
        "size": destination.stat().st_size,
    }


def run_backtest(staging: Path, variant: str, fallback: bool) -> Path:
    execution_policy = "fallback" if fallback else "leave_cash"
    output = staging / f"strict_backtest_{variant}_{execution_policy}"
    command = [
        sys.executable,
        str(ROOT / "script/evaluate_alpha360_uncertainty_horizons.py"),
        "--selection-predictions", str(staging / "selection_valid_ensemble_predictions.csv"),
        "--test-predictions", str(staging / "test_predictions.csv"),
        "--ensemble-selection-manifest", str(staging / "selection_manifest.json"),
        "--daily-ohlc-cache", str(ROOT / ".qlibAssistant/cache/tushare_daily_ohlc.parquet"),
        "--exact-limits", "/Users/hmax/investment_data/supplemental/stk_limit/stk_limit_all.parquet",
        "--index-cache", str(ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"),
        "--capital", "100000",
        "--commission-rate", "0.000235",
        "--minimum-commission", "5",
        "--slippage-bps", "0", "5",
        "--selection-slippage-bps", "5",
        "--minimum-active-days", "30",
        "--board-variant", variant,
        "--topks", "1", "3", "5", "10",
        "--fallback" if fallback else "--no-fallback",
        "--output", str(output),
    ]
    checked(command)
    return output


def build_report(
    staging: Path,
    mainboard: Path,
    all_board: Path,
    execution_policy: str,
) -> Path:
    output = staging / f"report_{execution_policy}"
    checked([
        sys.executable,
        str(ROOT / "script/report_alpha360_probabilistic_experiments.py"),
        "--selection-manifest", str(staging / "selection_manifest.json"),
        "--selection-predictions", str(staging / "selection_valid_ensemble_predictions.csv"),
        "--test-summary", str(staging / "test_summary.csv"),
        "--test-predictions", str(staging / "test_predictions.csv"),
        "--mainboard-backtest-dir", str(mainboard),
        "--all-backtest-dir", str(all_board),
        "--index-cache", str(ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"),
        "--fixed-reference-summary", str(staging / FIXED_REFERENCE_STAGING_NAME),
        "--execution-policy", execution_policy,
        "--output", str(output),
    ])
    return output


def build_training_curves(staging: Path) -> Path:
    output = staging / "training_curves"
    command = [
        sys.executable,
        str(ROOT / "script/report_alpha360_training_curves.py"),
    ]
    for experiment in REMOTE_EPOCH_METRICS:
        command.extend([
            "--metrics",
            f"{experiment}={staging / 'epoch_metrics' / f'{experiment}.csv'}",
        ])
    command.extend(["--expected-epochs", "50", "--output", str(output)])
    checked(command)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".qlibAssistant/analysis/alpha360_e0_e6_260828",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument(
        "--watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check and, when all strict gates pass, restart only the two authorized tasks.",
    )
    parser.add_argument(
        "--watchdog-audit-log",
        type=Path,
        default=WATCHDOG_AUDIT_LOG,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.watchdog:
        snapshot = run_training_watchdog(
            args.watchdog_audit_log,
            allow_restart=not args.status_only,
        )
        if snapshot is None:
            return 3
        try:
            status = combined_status_from_snapshot(snapshot)
        except RemoteWatchdogUnavailable as error:
            append_watchdog_audit(args.watchdog_audit_log, {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "event": "watchdog_combined_status_unavailable",
                "host": HOST,
                "error": str(error),
                "action": "retry_next_launchd_interval",
            })
            print(f"Combined pipeline status unavailable; retry next interval: {error}")
            return 3
    else:
        try:
            status = remote_status()
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            print(f"Remote status unavailable; retry next interval: {error}")
            return 3
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.status_only:
        return 0
    if status.get("status") == "failed":
        raise RuntimeError(f"Remote E0-E6 pipeline failed: {status.get('error')}")
    if status.get("status") != "test_ready" or status.get("test_read") is not True:
        print("Remote E0-E6 lockbox is not ready; no local artifact was changed.")
        return 3

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{int(time.time())}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        fetch_artifacts(staging)
        lockbox_audit = validate_downloaded_lockbox(staging, status)
        (staging / "local_lockbox_validation.json").write_text(
            json.dumps(lockbox_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        evidence_bundle = stage_evidence_bundle(staging, status)
        fixed_reference = stage_fixed_reference(staging)
        mainboard_fallback = run_backtest(staging, "mainboard", True)
        all_board_fallback = run_backtest(staging, "all", True)
        mainboard_leave_cash = run_backtest(staging, "mainboard", False)
        all_board_leave_cash = run_backtest(staging, "all", False)
        reports = {
            "fallback": build_report(
                staging, mainboard_fallback, all_board_fallback, "fallback"
            ),
            "leave_cash": build_report(
                staging, mainboard_leave_cash, all_board_leave_cash, "leave_cash"
            ),
        }
        training_curves = build_training_curves(staging)
        completion = {
            "status": "complete",
            "remote_status": status,
            "lockbox_validation": lockbox_audit,
            "evidence_bundle": evidence_bundle,
            "fixed_reference": fixed_reference,
            "reports": {
                name: str(path.relative_to(staging)) for name, path in reports.items()
            },
            "strict_backtests": {
                "mainboard_fallback": str(mainboard_fallback.relative_to(staging)),
                "all_fallback": str(all_board_fallback.relative_to(staging)),
                "mainboard_leave_cash": str(mainboard_leave_cash.relative_to(staging)),
                "all_leave_cash": str(all_board_leave_cash.relative_to(staging)),
            },
            "training_curves": str(training_curves.relative_to(staging)),
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (staging / "local_completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(output)
    except Exception:
        failed = output.parent / f"{output.name}_failed_{time.strftime('%Y%m%d_%H%M%S')}"
        if staging.exists():
            shutil.move(str(staging), str(failed))
        raise
    print(f"result={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
