#!/usr/bin/env python3
"""为数字 MLflow experiment 目录生成可读软链接和 CSV 索引。"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / ".qlibAssistant" / "mlruns"
LINK_ROOT = PROJECT_ROOT / ".qlibAssistant" / "mlruns_by_name"
CSV_PATH = PROJECT_ROOT / ".qlibAssistant" / "experiment_index.csv"


def parse_name(name: str):
    models = {
        "XGBModel": "XGBoost",
        "LinearModel": "Linear",
        "DEnsembleModel": "DoubleEnsemble",
        "LGBModel": "LightGBM",
        "CatBoostModel": "CatBoost",
    }
    model = next((label for token, label in models.items() if f"_{token}_" in name), "")
    pool_match = re.search(r"_(csi\d+|all|csiall)_", name, flags=re.IGNORECASE)
    tag_match = re.search(r"_step[^_]+_(.+?)_\d{8}_\d{2}$", name)
    return {
        "model": model,
        "stock_pool": pool_match.group(1).lower() if pool_match else "",
        "run_tag": tag_match.group(1) if tag_match else "",
    }


def recorder_counts(experiment_dir: Path):
    total = completed = 0
    completed_segments = set()
    for child in experiment_dir.iterdir():
        if not child.is_dir() or not (child / "meta.yaml").exists():
            continue
        total += 1
        artifacts = child / "artifacts"
        if (
            (artifacts / "params.pkl").exists()
            and (artifacts / "sig_analysis" / "ic.pkl").exists()
            and (artifacts / "valid_sig_analysis" / "metrics.pkl").exists()
        ):
            completed += 1
            task_path = artifacts / "task"
            if task_path.exists():
                with task_path.open("rb") as stream:
                    task = pickle.load(stream)
                segments = task["dataset"]["kwargs"]["segments"]
                completed_segments.add(tuple((key, tuple(value)) for key, value in sorted(segments.items())))
    return total, completed, len(completed_segments)


def main():
    LINK_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_links = set()

    for meta_path in sorted(MLRUNS.glob("*/meta.yaml")):
        metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        experiment_id = str(metadata.get("experiment_id", meta_path.parent.name))
        name = str(metadata.get("name", "")).strip()
        if not name:
            continue
        total, completed, unique_completed = recorder_counts(meta_path.parent)
        canonical_completed = min(5, unique_completed) if "_custom_" in name else unique_completed
        parsed = parse_name(name)
        rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_name": name,
                **parsed,
                "recorder_count": total,
                "completed_recorder_count": completed,
                "unique_completed_windows": canonical_completed,
                "duplicate_or_incomplete_count": total - canonical_completed,
                "path": str(meta_path.parent.resolve()),
            }
        )

        link = LINK_ROOT / name
        expected_links.add(link.name)
        target = Path("..") / "mlruns" / experiment_id
        if link.is_symlink() and link.readlink() == target:
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)

    # 只清理本脚本管理目录中的陈旧软链接，不触碰任何 MLflow 数据。
    for link in LINK_ROOT.iterdir():
        if link.is_symlink() and link.name not in expected_links:
            link.unlink()

    fieldnames = [
        "experiment_id",
        "experiment_name",
        "model",
        "stock_pool",
        "run_tag",
        "recorder_count",
        "completed_recorder_count",
        "unique_completed_windows",
        "duplicate_or_incomplete_count",
        "path",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Indexed {len(rows)} experiments")
    print(f"Links: {LINK_ROOT}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
