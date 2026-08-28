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
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zlib


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

ARTIFACT_INDEX_NAME = "artifact_index.json"
EXPECTED_EPOCHS = 50
REPORT_REQUIRED_FILES = (
    "concise_summary.csv",
    "model_selection.csv",
    "strict_execution_summary.csv",
    "topk_slippage_sensitivity.csv",
    "fixed_reference_comparison.csv",
    "prediction_metrics_comparison.png",
    "strategy_equity_curves.png",
    "method_and_findings.md",
)
STRICT_BACKTEST_REQUIRED_FILES = (
    "method.json",
    "chosen_rule_manifest_pre_test.json",
    "evaluated_rule_manifest.json",
    "selection_valid_chosen_rules.csv",
    "selection_valid_uncertainty_grid.csv",
    "test_baseline_four_horizons.csv",
    "test_selected_uncertainty_rules.csv",
    "label_alignment.csv",
    "uncertainty_selection_test_comparison.csv",
)


class RemoteWatchdogUnavailable(RuntimeError):
    """The remote watchdog snapshot could not be authenticated or retrieved."""


REMOTE_EPOCH_METRICS = {
    "E0_joint_three_leg": f"{REMOTE_E0_ROOT}/epoch_metrics.csv",
    "E1_shared_four_head": f"{REMOTE_BASE_ROOT}/E1_shared_four_head/epoch_metrics.csv",
    "E2_single_open1_close2": f"{REMOTE_BASE_ROOT}/E2_single_open1_close2/epoch_metrics.csv",
    "E3_single_close1_open2": f"{REMOTE_BASE_ROOT}/E3_single_close1_open2/epoch_metrics.csv",
    "E4_single_open1_open2": f"{REMOTE_BASE_ROOT}/E4_single_open1_open2/epoch_metrics.csv",
    "E5_single_close1_close2": f"{REMOTE_BASE_ROOT}/E5_single_close1_close2/epoch_metrics.csv",
    "E6_a_us_four_head": f"{REMOTE_ROOT}/run/epoch_metrics.csv",
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


def csv_row_count(path: Path) -> int:
    """Count data records with the standard CSV parser, excluding the header."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV has no header: {path}") from error
        if not header or not any(str(value).strip() for value in header):
            raise ValueError(f"CSV has an empty header: {path}")
        return sum(1 for row in reader if row)


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def png_dimensions(path: Path) -> tuple[int, int]:
    """Validate PNG framing/CRC and return non-zero IHDR dimensions."""

    signature = b"\x89PNG\r\n\x1a\n"
    width = height = None
    saw_idat = saw_iend = False
    with path.open("rb") as stream:
        if stream.read(len(signature)) != signature:
            raise ValueError(f"Invalid PNG signature: {path}")
        first_chunk = True
        while length_bytes := stream.read(4):
            if len(length_bytes) != 4:
                raise ValueError(f"Truncated PNG chunk length: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = stream.read(4)
            data = stream.read(length)
            checksum = stream.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(checksum) != 4:
                raise ValueError(f"Truncated PNG chunk: {path}")
            expected_crc = struct.unpack(">I", checksum)[0]
            actual_crc = zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(f"PNG CRC mismatch: {path}")
            if first_chunk:
                if chunk_type != b"IHDR" or length != 13:
                    raise ValueError(f"PNG does not begin with IHDR: {path}")
                width, height = struct.unpack(">II", data[:8])
                if width <= 0 or height <= 0:
                    raise ValueError(f"PNG has zero dimensions: {path}")
                first_chunk = False
            elif chunk_type == b"IDAT":
                saw_idat = True
            elif chunk_type == b"IEND":
                if length != 0:
                    raise ValueError(f"Invalid PNG IEND: {path}")
                saw_iend = True
                if stream.read(1):
                    raise ValueError(f"PNG has trailing bytes after IEND: {path}")
                break
    if width is None or height is None or not saw_idat or not saw_iend:
        raise ValueError(f"PNG is incomplete or unreadable: {path}")
    return width, height


def generated_stage(relative_path: str) -> str:
    first = Path(relative_path).parts[0]
    if first == "evidence":
        return "offline_evidence"
    if first == "epoch_metrics":
        return "training_history"
    if first.startswith("strict_backtest_"):
        return "strict_backtest"
    if first.startswith("report_"):
        return "report"
    if first == "training_curves":
        return "training_curves"
    if relative_path == FIXED_REFERENCE_STAGING_NAME:
        return "fixed_reference"
    if relative_path in {
        "local_lockbox_validation.json",
        "local_completion.json",
        "local_prepublish_validation.json",
    }:
        return "finalization"
    return "remote_lockbox"


def artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".png": "png",
        ".md": "markdown",
        ".log": "log",
        ".pt": "pytorch_checkpoint",
    }.get(suffix, suffix.lstrip(".") or "binary")


def build_artifact_index(staging: Path, core_artifacts: set[str]) -> dict:
    """Index every published file except this self-referential index itself."""

    staging = staging.expanduser().resolve()
    index_path = staging / ARTIFACT_INDEX_NAME
    if index_path.exists():
        raise FileExistsError(index_path)
    records: list[dict] = []
    for path in sorted(value for value in staging.rglob("*") if value.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative == ARTIFACT_INDEX_NAME:
            continue
        record = {
            "relative_path": relative,
            "type": artifact_type(path),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
            "core": relative in core_artifacts,
            "generated_stage": generated_stage(relative),
        }
        if path.suffix.lower() == ".csv":
            record["csv_rows"] = csv_row_count(path)
        elif path.suffix.lower() == ".png":
            width, height = png_dimensions(path)
            record["png_width"] = width
            record["png_height"] = height
        records.append(record)
    missing_core = sorted(core_artifacts - {record["relative_path"] for record in records})
    if missing_core:
        raise RuntimeError(f"Core artifacts are absent from artifact index: {missing_core}")
    value = {
        "schema_version": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": ".",
        "coverage": (
            "Every file recursively below the final directory at index creation time; "
            "artifact_index.json excludes itself because a file cannot contain its own SHA256."
        ),
        "self_excluded": ARTIFACT_INDEX_NAME,
        "file_count": len(records),
        "core_artifact_count": sum(bool(record["core"]) for record in records),
        "artifacts": records,
    }
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(index_path)
    return value


def validate_artifact_index(staging: Path) -> dict:
    staging = staging.expanduser().resolve()
    index_path = staging / ARTIFACT_INDEX_NAME
    value = read_json(index_path)
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact_index.json has no artifacts list")
    expected = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path != index_path
    }
    indexed = {record.get("relative_path") for record in artifacts if isinstance(record, dict)}
    if indexed != expected or len(indexed) != len(artifacts):
        raise RuntimeError("artifact_index.json does not exactly cover the final directory")
    for record in artifacts:
        relative = record["relative_path"]
        path = (staging / relative).resolve()
        if staging not in path.parents or not path.is_file():
            raise RuntimeError(f"Unsafe or missing artifact-index path: {relative}")
        if record.get("size") != path.stat().st_size:
            raise RuntimeError(f"Artifact size changed after indexing: {relative}")
        if record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"Artifact hash changed after indexing: {relative}")
        if path.suffix.lower() == ".csv" and record.get("csv_rows") != csv_row_count(path):
            raise RuntimeError(f"Artifact CSV row count changed after indexing: {relative}")
        if path.suffix.lower() == ".png":
            width, height = png_dimensions(path)
            if (record.get("png_width"), record.get("png_height")) != (width, height):
                raise RuntimeError(f"Artifact PNG dimensions changed after indexing: {relative}")
    if value.get("file_count") != len(artifacts):
        raise RuntimeError("artifact_index.json file_count is incorrect")
    return {
        "status": "verified",
        "artifact_index_sha256": file_sha256(index_path),
        "indexed_files": len(artifacts),
        "core_artifacts": sum(bool(record.get("core")) for record in artifacts),
    }


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
            "selection_valid_predictions": copy_authenticated_reference(
                freeze.get("selection_valid_predictions"),
                evidence / "selection_valid_predictions" / f"{name}.csv",
                f"{name} Selection-valid predictions",
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

        checkpoints = freeze.get("checkpoints")
        if not isinstance(checkpoints, dict) or not checkpoints:
            raise ValueError(f"Missing checkpoint freeze for {name}")
        candidate_record["all_checkpoints"] = {}
        for horizon, reference in sorted(checkpoints.items()):
            candidate_record["all_checkpoints"][horizon] = copy_authenticated_reference(
                reference,
                evidence / "checkpoints" / name / f"{horizon}.pt",
                f"{name}/{horizon} checkpoint",
            )

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
            candidate_record["selected_checkpoints"] = {}
            for horizon, body in selection["selections"].items():
                if name not in body["selected_components"]:
                    continue
                candidate_record["selected_checkpoints"][horizon] = (
                    candidate_record["all_checkpoints"][horizon]
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


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty {label}: {path}")


def validate_epoch_histories(staging: Path) -> set[str]:
    epoch_directory = staging / "epoch_metrics"
    expected_names = {f"{name}.csv" for name in REMOTE_EPOCH_METRICS}
    actual_names = {path.name for path in epoch_directory.glob("*.csv") if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Epoch histories differ from the seven registered experiments: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    core: set[str] = set()
    expected_epochs = list(range(1, EXPECTED_EPOCHS + 1))
    for name in sorted(expected_names):
        path = epoch_directory / name
        records = read_csv_records(path)
        if len(records) != EXPECTED_EPOCHS or "epoch" not in (records[0] if records else {}):
            raise ValueError(f"{name} must contain exactly {EXPECTED_EPOCHS} epoch rows")
        epochs: list[int] = []
        for record in records:
            value = float(record["epoch"])
            if not value.is_integer():
                raise ValueError(f"{name} contains a non-integral epoch: {record['epoch']}")
            epochs.append(int(value))
        if epochs != expected_epochs:
            raise ValueError(f"{name} epochs must be contiguous 1..{EXPECTED_EPOCHS}")
        core.add(path.relative_to(staging).as_posix())
    return core


def _require_zero_unresolved(path: Path, label: str) -> list[dict[str, str]]:
    records = read_csv_records(path)
    if not records:
        raise ValueError(f"Core strict summary is empty: {path}")
    if "unresolved_exit" not in records[0]:
        raise ValueError(f"{label} lacks unresolved_exit")
    for row_number, record in enumerate(records, start=2):
        try:
            value = float(record["unresolved_exit"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} has invalid unresolved_exit at row {row_number}") from error
        if value != 0.0:
            raise RuntimeError(
                f"{label} has unresolved exits at row {row_number}: {record['unresolved_exit']}"
            )
    return records


def validate_strict_backtest(
    staging: Path,
    directory: Path,
    board_variant: str,
    fallback: bool,
) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing strict backtest directory: {directory}")
    core: set[str] = set()
    for filename in STRICT_BACKTEST_REQUIRED_FILES:
        path = directory / filename
        require_nonempty_file(path, f"strict backtest artifact {filename}")
        if path.suffix.lower() == ".csv" and csv_row_count(path) <= 0:
            raise ValueError(f"Core strict backtest CSV has no records: {path}")
        core.add(path.relative_to(staging).as_posix())

    method = read_json(directory / "method.json")
    expected_parameters = {
        "board_variant": board_variant,
        "fallback": fallback,
        "capital": 100000.0,
        "commission_rate_each_side": 0.000235,
        "minimum_commission": 5.0,
        "lot_size": 100,
        "diagnostic_test_grid_generated": False,
    }
    for key, expected in expected_parameters.items():
        if method.get(key) != expected:
            raise RuntimeError(
                f"{directory.name} method mismatch for {key}: "
                f"expected {expected!r}, got {method.get(key)!r}"
            )
    pretest_path = directory / "chosen_rule_manifest_pre_test.json"
    evaluated_path = directory / "evaluated_rule_manifest.json"
    pretest = read_json(pretest_path)
    evaluated = read_json(evaluated_path)
    if pretest.get("test_opened") is not False or evaluated.get("test_opened") is not True:
        raise RuntimeError(f"{directory.name} does not preserve the pre-Test rule freeze")
    if evaluated.get("pretest_rule_manifest_sha256") != file_sha256(pretest_path):
        raise RuntimeError(f"{directory.name} evaluated manifest does not authenticate pre-Test rules")

    baseline_path = directory / "test_baseline_four_horizons.csv"
    selected_path = directory / "test_selected_uncertainty_rules.csv"
    baseline = _require_zero_unresolved(baseline_path, f"{directory.name} baseline")
    selected = _require_zero_unresolved(selected_path, f"{directory.name} selected rules")
    expected_baseline = {
        (horizon, topk, slippage)
        for horizon in (
            "open1_close2", "close1_open2", "open1_open2", "close1_close2"
        )
        for topk in (1, 3, 5, 10)
        for slippage in (0.0, 5.0)
    }
    actual_baseline = {
        (row["horizon"], int(float(row["topk"])), float(row["slippage_bps_each_side"]))
        for row in baseline
    }
    if actual_baseline != expected_baseline or len(baseline) != len(expected_baseline):
        raise RuntimeError(f"{directory.name} lacks the complete TopK x 0/5bps strict matrix")
    expected_selected = {
        (horizon, slippage)
        for horizon in (
            "open1_close2", "close1_open2", "open1_open2", "close1_close2"
        )
        for slippage in (0.0, 5.0)
    }
    actual_selected = {
        (row["horizon"], float(row["slippage_bps_each_side"])) for row in selected
    }
    if actual_selected != expected_selected or len(selected) != len(expected_selected):
        raise RuntimeError(f"{directory.name} lacks the complete selected-rule 0/5bps matrix")
    return core


def validate_report(staging: Path, directory: Path, execution_policy: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {execution_policy} report directory: {directory}")
    core: set[str] = set()
    for filename in REPORT_REQUIRED_FILES:
        path = directory / filename
        require_nonempty_file(path, f"{execution_policy} report artifact {filename}")
        if path.suffix.lower() == ".csv" and csv_row_count(path) <= 0:
            raise ValueError(f"Core report CSV has no records: {path}")
        if path.suffix.lower() == ".png":
            png_dimensions(path)
        core.add(path.relative_to(staging).as_posix())
    strict_summary = _require_zero_unresolved(
        directory / "strict_execution_summary.csv",
        f"{execution_policy} report strict execution summary",
    )
    if len(strict_summary) != 16:
        raise RuntimeError(f"{execution_policy} strict execution summary must contain 16 rows")
    return core


def validate_offline_evidence(staging: Path) -> set[str]:
    selection = read_json(staging / "selection_manifest.json")
    evidence_path = staging / "evidence/evidence_index.json"
    evidence = read_json(evidence_path)
    if evidence.get("status") != "verified":
        raise ValueError("Offline evidence bundle is not verified")
    frozen = selection.get("candidate_freeze")
    records = evidence.get("candidates")
    if not isinstance(frozen, dict) or not isinstance(records, dict) or set(records) != set(frozen):
        raise RuntimeError("Offline evidence does not contain every frozen candidate")
    core = {evidence_path.relative_to(staging).as_posix()}

    def verify_reference(reference: object, label: str) -> Path:
        if not isinstance(reference, dict):
            raise ValueError(f"Offline evidence lacks {label}")
        local_path = reference.get("local_path")
        expected_hash = reference.get("sha256")
        if not isinstance(local_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Offline evidence reference is incomplete for {label}")
        path = (staging / local_path).resolve()
        if staging not in path.parents:
            raise RuntimeError(f"Offline evidence path escapes final directory for {label}")
        require_nonempty_file(path, f"offline evidence {label}")
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Offline evidence hash mismatch for {label}")
        core.add(path.relative_to(staging).as_posix())
        return path

    verify_reference(evidence.get("protocol"), "frozen protocol")
    data_manifests = evidence.get("data_manifests")
    if not isinstance(data_manifests, dict) or not data_manifests:
        raise ValueError("Offline evidence lacks data manifests")
    for data_hash, reference in data_manifests.items():
        path = verify_reference(reference, f"data manifest {data_hash}")
        if file_sha256(path) != data_hash:
            raise RuntimeError(f"Offline data manifest key/hash mismatch: {data_hash}")

    for name, freeze in frozen.items():
        record = records[name]
        verify_reference(record.get("candidate_manifest"), f"{name} candidate manifest")
        verify_reference(record.get("configuration"), f"{name} configuration")
        prediction_path = verify_reference(
            record.get("selection_valid_predictions"),
            f"{name} Selection-valid predictions",
        )
        if csv_row_count(prediction_path) <= 0:
            raise ValueError(f"Offline Selection-valid predictions are empty for {name}")
        if file_sha256(prediction_path) != freeze["selection_valid_predictions"]["sha256"]:
            raise RuntimeError(f"Offline Selection-valid prediction hash mismatch for {name}")
        checkpoints = record.get("all_checkpoints")
        frozen_checkpoints = freeze.get("checkpoints")
        if not isinstance(checkpoints, dict) or not isinstance(frozen_checkpoints, dict):
            raise ValueError(f"Offline evidence lacks complete checkpoints for {name}")
        if set(checkpoints) != set(frozen_checkpoints):
            raise RuntimeError(f"Offline checkpoint set differs from Selection freeze for {name}")
        for horizon, reference in checkpoints.items():
            checkpoint_path = verify_reference(reference, f"{name}/{horizon} checkpoint")
            if file_sha256(checkpoint_path) != frozen_checkpoints[horizon]["sha256"]:
                raise RuntimeError(f"Offline checkpoint hash mismatch for {name}/{horizon}")
        test_audits = record.get("test_audits", {})
        if not isinstance(test_audits, dict):
            raise ValueError(f"Offline Test audit map is invalid for {name}")
        for filename, reference in test_audits.items():
            verify_reference(reference, f"{name} Test audit {filename}")
    return core


def validate_final_staging(staging: Path) -> dict:
    """Fail closed before atomic publication of any incomplete final delivery."""

    staging = staging.expanduser().resolve()
    core: set[str] = set()
    top_level_core = (
        "selection_manifest.json",
        "selection_valid_ensemble_predictions.csv",
        "test_summary.csv",
        "test_predictions.csv",
        "test_completion_audit.json",
        "evaluated_selection_manifest.json",
        "local_lockbox_validation.json",
        "local_completion.json",
        FIXED_REFERENCE_STAGING_NAME,
    )
    for filename in top_level_core:
        path = staging / filename
        require_nonempty_file(path, f"core final artifact {filename}")
        if path.suffix.lower() == ".csv" and csv_row_count(path) <= 0:
            raise ValueError(f"Core CSV has no records: {path}")
        core.add(filename)

    lockbox = read_json(staging / "local_lockbox_validation.json")
    completion = read_json(staging / "local_completion.json")
    if lockbox.get("status") != "verified" or completion.get("status") != "complete":
        raise RuntimeError("Local lockbox/completion state is not publishable")
    core.update(validate_offline_evidence(staging))
    core.update(validate_epoch_histories(staging))

    strict_specs = (
        ("strict_backtest_mainboard_fallback", "mainboard", True),
        ("strict_backtest_all_fallback", "all", True),
        ("strict_backtest_mainboard_leave_cash", "mainboard", False),
        ("strict_backtest_all_leave_cash", "all", False),
    )
    for directory_name, variant, fallback in strict_specs:
        core.update(validate_strict_backtest(
            staging, staging / directory_name, variant, fallback
        ))

    core.update(validate_report(staging, staging / "report_fallback", "fallback"))
    core.update(validate_report(staging, staging / "report_leave_cash", "leave_cash"))
    training_required = (
        "training_curve_summary.csv", "training_curves.png", "training_curves.md"
    )
    for filename in training_required:
        path = staging / "training_curves" / filename
        require_nonempty_file(path, f"training-curve artifact {filename}")
        if path.suffix.lower() == ".csv" and csv_row_count(path) != 7:
            raise RuntimeError("Training curve summary must contain exactly seven experiments")
        if path.suffix.lower() == ".png":
            png_dimensions(path)
        core.add(path.relative_to(staging).as_posix())

    png_files = sorted(staging.rglob("*.png"))
    if not png_files:
        raise RuntimeError("Final delivery contains no PNG evidence")
    for path in png_files:
        png_dimensions(path)
    return {
        "status": "verified",
        "epoch_histories": len(REMOTE_EPOCH_METRICS),
        "strict_backtests": len(strict_specs),
        "reports": 2,
        "png_files": len(png_files),
        "core_artifacts": sorted(core),
    }


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
            "prepublish_validation": "local_prepublish_validation.json",
            "artifact_index": ARTIFACT_INDEX_NAME,
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (staging / "local_completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        delivery_validation = validate_final_staging(staging)
        validation_path = staging / "local_prepublish_validation.json"
        validation_path.write_text(
            json.dumps(delivery_validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        core_artifacts = set(delivery_validation["core_artifacts"])
        core_artifacts.add(validation_path.relative_to(staging).as_posix())
        build_artifact_index(staging, core_artifacts)
        validate_artifact_index(staging)
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
