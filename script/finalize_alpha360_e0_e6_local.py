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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = remote_status()
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
