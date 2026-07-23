#!/usr/bin/env python3
"""Wait for long-window training, evaluate it, and rebuild the formal matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Linear": "LinearModel",
    "LightGBM": "LGBModel",
    "XGBoost": "XGBModel",
    "CatBoost": "CatBoostModel",
}


def run(command, log, env):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {' '.join(command)}", file=log, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", default=".qlibAssistant/logs/extended_windows_20260720.log")
    parser.add_argument("--evaluation-log", default=".qlibAssistant/logs/extended_windows_evaluation_20260720.log")
    parser.add_argument("--months", nargs="+", type=int, default=[180, 240])
    parser.add_argument("--date-tag", default="260720")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    training_log = ROOT / args.training_log
    expected = f"END train_months={args.months[-1]} exit_code=0"
    while expected not in training_log.read_text(errors="replace"):
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MPLCONFIGDIR"] = str(ROOT / ".qlibAssistant" / "matplotlib")
    evaluation_log = ROOT / args.evaluation_log
    evaluation_log.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_log.open("a", encoding="utf-8", buffering=1) as log:
        run([sys.executable, "script/build_experiment_index.py"], log, env)
        for months in args.months:
            for fold in ("fold1", "fold2", "fold3"):
                pattern = f"windowcmp_{fold}_train{months}m_{args.date_tag}"
                for model_class in MODELS.values():
                    run([
                        sys.executable, "script/evaluate_batch.py",
                        "--experiment-pattern", pattern,
                        "--model-class", model_class,
                        "--cost-rate", "0.0015",
                    ], log, env)
        run([
            sys.executable, "script/summarize_window_experiment.py",
            "--logs",
            ".qlibAssistant/logs/window_length_baseline45_evaluation_with_cost_20260720.log",
            ".qlibAssistant/logs/window_length_evaluation_with_cost_20260720.log",
            args.evaluation_log,
            "--output-dir",
            ".qlibAssistant/analysis/window_length_comparison_extended_20260720",
        ], log, env)
    print(f"completed={evaluation_log}")


if __name__ == "__main__":
    main()
