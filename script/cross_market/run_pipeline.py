#!/usr/bin/env python3
"""Resumable end-to-end cross-market download, factor, and transfer-training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import LOGS, ROOT, atomic_json, ensure_dirs


PYTHON = Path(sys.executable)


def run(stage: str, command: list[str], log_file: Path, state_file: Path) -> None:
    payload = {
        "stage": stage,
        "status": "running",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
    }
    atomic_json(state_file, payload)
    started = time.monotonic()
    with log_file.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    payload.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "exit_code": result.returncode,
        }
    )
    atomic_json(state_file, payload)
    if result.returncode:
        raise SystemExit(f"{stage} failed; inspect {log_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=f"global_to_a_{datetime.now():%y%m%d_%H%M%S}")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--start-at", choices=["download", "export_a", "factors", "train"], default="download")
    args = parser.parse_args()
    ensure_dirs()
    log_file = LOGS / f"{args.run_tag}.log"
    state_file = LOGS / f"{args.run_tag}_state.json"
    stages: list[tuple[str, list[str]]] = []
    yahoo_cmd = [
        str(PYTHON),
        "script/cross_market/download_yahoo_daily.py",
        "--start",
        "2004-01-01",
    ]
    export_cmd = [
        str(PYTHON),
        "script/cross_market/export_a_mainboard_daily.py",
        "--start",
        "2004-01-01",
    ]
    factor_cmd = [str(PYTHON), "script/cross_market/build_factor_store.py"]
    train_cmd = [
        str(PYTHON),
        "script/cross_market/train_global_then_a.py",
        "--run-tag",
        args.run_tag,
    ]
    if args.pilot:
        yahoo_cmd += ["--us-limit", "40", "--hk-limit", "40"]
        export_cmd += ["--limit", "300"]
        factor_cmd += ["--limit", "300"]
        train_cmd += [
            "--pretrain-start", "2018-01-01",
            "--fine-start", "2020-01-01",
            "--max-symbols-a", "200",
            "--max-symbols-us", "40",
            "--max-symbols-hk", "40",
            "--pretrain-rows-per-market", "200000",
            "--fine-rows", "300000",
            "--pretrain-rounds", "80",
            "--fine-rounds", "40",
            "--baseline-rounds", "80",
            "--early-stopping-rounds", "10",
        ]
    stages.extend(
        [
            ("download", yahoo_cmd),
            ("export_a", export_cmd),
            ("factors", factor_cmd),
            ("train", train_cmd),
        ]
    )
    start_index = [name for name, _ in stages].index(args.start_at)
    for stage, command in stages[start_index:]:
        run(stage, command, log_file, state_file)
    atomic_json(
        state_file,
        {
            "stage": "all",
            "status": "completed",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_tag": args.run_tag,
            "log": str(log_file),
        },
    )
    print(f"completed: {args.run_tag}\nlog: {log_file}")


if __name__ == "__main__":
    main()
