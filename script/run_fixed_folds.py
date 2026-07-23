#!/usr/bin/env python3
"""在固定 Valid/Selection-valid/Test 上比较模型或实际 Train 长度。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDS = [
    {
        "name": "fold1",
        "valid": ("2024-06-15", "2024-11-14"),
        "selection_valid": ("2024-11-15", "2025-04-14"),
        "test": ("2025-04-15", "2025-09-15"),
    },
    {
        "name": "fold2",
        "valid": ("2024-11-16", "2025-04-15"),
        "selection_valid": ("2025-04-16", "2025-09-15"),
        "test": ("2025-09-16", "2026-02-16"),
    },
    {
        "name": "fold3",
        "valid": ("2025-04-17", "2025-09-16"),
        "selection_valid": ("2025-09-17", "2026-02-16"),
        "test": ("2026-02-17", "2026-07-17"),
    },
]


def fixed_segments(fold, train_months):
    valid_start = datetime.strptime(fold["valid"][0], "%Y-%m-%d")
    train_start = valid_start - relativedelta(months=train_months)
    train_end = valid_start - timedelta(days=1)
    return {
        "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        "valid": fold["valid"],
        "selection_valid": fold["selection_valid"],
        "test": fold["test"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--train-months", type=int, default=45)
    parser.add_argument("--pool", default="csi1000")
    parser.add_argument("--date-tag", default=datetime.now().strftime("%y%m%d"))
    parser.add_argument("--tag-prefix", default="fixed")
    parser.add_argument("--model-preset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for fold in FOLDS:
        segments = fixed_segments(fold, args.train_months)
        tag = f"{args.tag_prefix}_{fold['name']}_train{args.train_months}m_{args.date_tag}"
        command = [
            sys.executable,
            "script/run.py",
            "--pool",
            args.pool,
            "--run-tag",
            tag,
            "--models",
            *args.models,
            "--split-selection-valid",
            "--segments-json",
            json.dumps(segments, separators=(",", ":")),
        ]
        if args.model_preset:
            command.extend(["--model-preset", args.model_preset])
        if args.dry_run:
            command.append("--dry-run")
        print(f"[fixed-fold] {fold['name']} segments={segments}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
