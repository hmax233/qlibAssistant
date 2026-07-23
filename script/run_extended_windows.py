#!/usr/bin/env python3
"""Run longer fixed-window experiments with a persistent, inspectable log."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-months", nargs="+", type=int, default=[180, 240])
    parser.add_argument("--models", nargs="+", default=["Linear", "LightGBM", "XGBoost", "CatBoost"])
    parser.add_argument("--pool", default="csi1000")
    parser.add_argument("--date-tag", default=datetime.now().strftime("%y%m%d"))
    parser.add_argument(
        "--log",
        default=f".qlibAssistant/logs/extended_windows_{datetime.now():%Y%m%d_%H%M%S}.log",
    )
    args = parser.parse_args()
    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MPLCONFIGDIR"] = str(ROOT / ".qlibAssistant" / "matplotlib")

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for months in args.train_months:
            command = [
                sys.executable,
                "script/run_fixed_folds.py",
                "--models", *args.models,
                "--train-months", str(months),
                "--pool", args.pool,
                "--tag-prefix", "windowcmp",
                "--date-tag", args.date_tag,
            ]
            marker = f"[{datetime.now().isoformat(timespec='seconds')}] START train_months={months}"
            print(marker, flush=True)
            print(marker, file=log)
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            marker = (
                f"[{datetime.now().isoformat(timespec='seconds')}] END "
                f"train_months={months} exit_code={result.returncode}"
            )
            print(marker, flush=True)
            print(marker, file=log)
            if result.returncode:
                raise SystemExit(result.returncode)
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
