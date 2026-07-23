#!/usr/bin/env python3
"""顺序运行互不重叠的 CSI1000 时间滚动诊断 folds。"""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDS = [
    ("fold1_260719", "2025-09-15"),
    ("fold2_260719", "2026-02-16"),
    ("fold3_260719", "2026-07-17"),
]


def run(command):
    print(f"[rolling-folds] RUN: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="运行三个非重叠时间滚动 folds")
    parser.add_argument("--model", default="Linear")
    parser.add_argument("--tag-prefix", default="fold")
    parser.add_argument("--date-tag", default="260719")
    parser.add_argument("--label-horizon", type=int, default=1, choices=[1, 3, 5])
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--raw-label", action="store_true")
    args = parser.parse_args()

    folds = [
        (f"{args.tag_prefix}{index}_{args.date_tag}", end_date)
        for index, (_, end_date) in enumerate(FOLDS, 1)
    ]
    for index, (tag, end_date) in enumerate(folds, 1):
        print(f"[fold {index}/{len(FOLDS)}] start tag={tag} end={end_date}", flush=True)
        run(
            [
                sys.executable,
                "script/run.py",
                "--pool",
                "csi1000",
                "--run-tag",
                tag,
                "--models",
                args.model,
                "--split-selection-valid",
                "--window-months",
                "60",
                "--end-date",
                end_date,
                "--label-horizon",
                str(args.label_horizon),
            ]
            + (["--normalize-features"] if args.normalize_features else [])
            + (["--raw-label"] if args.raw_label else [])
        )
        run(
            [
                sys.executable,
                "script/evaluate_batch.py",
                "--experiment-pattern",
                tag,
                "--threshold",
                "-999",
                "--topk",
                "10",
                "--weighting",
                "equal",
            ]
        )
        print(f"[fold {index}/{len(FOLDS)}] complete tag={tag}", flush=True)
    print("[rolling-folds] all folds complete", flush=True)


if __name__ == "__main__":
    main()
