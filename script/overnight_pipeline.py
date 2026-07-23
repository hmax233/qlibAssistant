#!/usr/bin/env python3
"""本地低-token夜间流水线：等待已有训练，再完成预测、报告及下一股票池。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = PROJECT_ROOT / "roll"
PYTHON = Path(sys.executable)
STATE_FILE = PROJECT_ROOT / ".qlibAssistant" / "overnight_pipeline_state.json"
LOG_FILE = PROJECT_ROOT / ".qlibAssistant" / "overnight_pipeline.log"
LOCK_FILE = PROJECT_ROOT / ".qlibAssistant" / "overnight_pipeline.lock"


def record(stage, status="running", **details):
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        **details,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run(stage, cmd, cwd=PROJECT_ROOT):
    record(stage, command=cmd)
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".qlibAssistant" / "matplotlib")
    started = time.time()
    with LOG_FILE.open("a", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = round(time.time() - started, 1)
    if result.returncode:
        record(stage, status="failed", exit_code=result.returncode, elapsed_seconds=elapsed)
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")
    record(stage, status="completed", elapsed_seconds=elapsed)


def train(pool, run_tag):
    run(
        f"train_{pool}",
        [str(PYTHON), "script/run.py", "--pool", pool, "--run-tag", run_tag],
    )


def predict(pool, run_tag):
    pattern = f".*{pool}.*{run_tag}.*"
    run(
        f"predict_{pool}",
        [
            str(PYTHON),
            "./roll.py",
            f"--stock_pool={pool}",
            f"--model_filter=[\"{pattern}\"]",
            "model",
            "selection",
        ],
        cwd=ROLL_DIR,
    )


def evaluate(pool, run_tag):
    run(
        f"evaluate_{pool}",
        [
            str(PYTHON),
            "script/evaluate_batch.py",
            "--experiment-pattern",
            f"{pool}_custom_step0_{run_tag}",
        ],
    )


def refresh_index():
    run("refresh_experiment_index", [str(PYTHON), "script/build_experiment_index.py"])


def export_readable(pattern):
    run(
        "export_readable_artifacts",
        [
            str(PYTHON),
            "script/export_readable_artifacts.py",
            "--experiment-pattern",
            pattern,
        ],
    )


def main():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another overnight pipeline instance is already running")
    parser = argparse.ArgumentParser(description="本地夜间量化流水线")
    parser.add_argument("--wait-pid", type=int, help="先等待这个已有训练进程退出")
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--first-pool", default="csi1000")
    parser.add_argument("--first-tag", default="retrain260718")
    parser.add_argument("--next-pool", default="all")
    parser.add_argument("--next-tag", default="retrain260719")
    parser.add_argument("--skip-next", action="store_true")
    args = parser.parse_args()

    if args.wait_pid:
        while pid_alive(args.wait_pid):
            record("waiting_existing_training", pid=args.wait_pid, poll_seconds=args.poll_seconds)
            time.sleep(args.poll_seconds)
        record("existing_training_exited", status="completed", pid=args.wait_pid)

    # 重跑是完整性检查兼断点续训：已存在窗口会跳过，失败/缺失窗口会补齐。
    train(args.first_pool, args.first_tag)
    refresh_index()
    export_readable(f"{args.first_pool}.*{args.first_tag}")
    predict(args.first_pool, args.first_tag)
    evaluate(args.first_pool, args.first_tag)

    if not args.skip_next:
        train(args.next_pool, args.next_tag)
        refresh_index()
        export_readable(f"{args.next_pool}.*{args.next_tag}")
        predict(args.next_pool, args.next_tag)
        evaluate(args.next_pool, args.next_tag)

    record("pipeline", status="completed")


if __name__ == "__main__":
    main()
