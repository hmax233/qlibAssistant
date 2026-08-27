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


def fetch_artifacts(staging: Path) -> None:
    for local_name, remote_relative in REMOTE_FILES.items():
        source = f"{USER}@{HOST}:{REMOTE_ROOT}/{remote_relative}"
        checked([*scp_base(), source, str(staging / local_name)])

    epoch_directory = staging / "epoch_metrics"
    epoch_directory.mkdir()
    for experiment, remote_path in REMOTE_EPOCH_METRICS.items():
        source = f"{USER}@{HOST}:{remote_path}"
        checked([*scp_base(), source, str(epoch_directory / f"{experiment}.csv")])


def run_backtest(staging: Path, variant: str) -> Path:
    output = staging / f"strict_backtest_{variant}"
    command = [
        sys.executable,
        str(ROOT / "script/evaluate_alpha360_uncertainty_horizons.py"),
        "--selection-predictions", str(staging / "selection_valid_ensemble_predictions.csv"),
        "--test-predictions", str(staging / "test_predictions.csv"),
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
        "--fallback",
        "--output", str(output),
    ]
    checked(command)
    return output


def build_report(staging: Path, mainboard: Path, all_board: Path) -> Path:
    output = staging / "report"
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
        mainboard = run_backtest(staging, "mainboard")
        all_board = run_backtest(staging, "all")
        report = build_report(staging, mainboard, all_board)
        training_curves = build_training_curves(staging)
        completion = {
            "status": "complete",
            "remote_status": status,
            "report": str(report.relative_to(staging)),
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
