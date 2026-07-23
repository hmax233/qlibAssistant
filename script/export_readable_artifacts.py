#!/usr/bin/env python3
"""将 MLflow recorder 的 pkl 指标/预测导出成人类可读的 Markdown 和 CSV。"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / ".qlibAssistant" / "mlruns"
OUTPUT_ROOT = PROJECT_ROOT / ".qlibAssistant" / "readable_artifacts"


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def as_series(obj, name):
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    result = obj.copy()
    result.name = name
    return result


def metrics(ic, ric):
    ic_mean, ric_mean = float(ic.mean()), float(ric.mean())
    ic_std, ric_std = float(ic.std()), float(ric.std())
    return {
        "IC": ic_mean,
        "ICIR": ic_mean / ic_std if ic_std else np.nan,
        "Rank IC": ric_mean,
        "Rank ICIR": ric_mean / ric_std if ric_std else np.nan,
        "IC Positive Ratio": float((ic > 0).mean()),
        "Rank IC Positive Ratio": float((ric > 0).mean()),
        "Date Count": int(pd.concat([ic, ric], axis=1).dropna().shape[0]),
    }


def segment_text(task, segment):
    values = task["dataset"]["kwargs"]["segments"][segment]
    return f"{values[0]} ~ {values[1]}"


def export_recorder(experiment_name, recorder_dir, topn, full_predictions):
    artifacts = recorder_dir / "artifacts"
    required = [
        artifacts / "task",
        artifacts / "pred.pkl",
        artifacts / "label.pkl",
        artifacts / "sig_analysis" / "ic.pkl",
        artifacts / "sig_analysis" / "ric.pkl",
        artifacts / "valid_sig_analysis" / "ic.pkl",
        artifacts / "valid_sig_analysis" / "ric.pkl",
    ]
    if not all(path.exists() for path in required):
        return None

    task = load_pickle(artifacts / "task")
    output = OUTPUT_ROOT / experiment_name / recorder_dir.name
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, folder in (("validation", "valid_sig_analysis"), ("test", "sig_analysis")):
        ic = as_series(load_pickle(artifacts / folder / "ic.pkl"), "IC")
        ric = as_series(load_pickle(artifacts / folder / "ric.pkl"), "Rank IC")
        daily = pd.concat([ic, ric], axis=1)
        daily.index.name = "datetime"
        daily.to_csv(output / f"{split}_daily_ic.csv")
        row = {"split": split, **metrics(ic, ric)}
        rows.append(row)

    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pred = as_series(load_pickle(artifacts / "pred.pkl"), "score")
    label = as_series(load_pickle(artifacts / "label.pkl"), "label")
    predictions = pd.concat([pred, label], axis=1).dropna(subset=["score"])
    top = (
        predictions.reset_index()
        .sort_values(["datetime", "score"], ascending=[True, False])
        .groupby("datetime", as_index=False, group_keys=False)
        .head(topn)
    )
    top["rank"] = top.groupby("datetime").cumcount() + 1
    top.to_csv(output / f"test_top{topn}_predictions.csv", index=False)
    if full_predictions:
        predictions.reset_index().to_csv(output / "test_all_predictions.csv", index=False)

    model = task["model"]["class"]
    handler = task["dataset"]["kwargs"]["handler"]
    pool = handler["kwargs"].get("instruments", "")
    summary = [
        f"# Recorder {recorder_dir.name}",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Model: `{model}`",
        f"- Stock pool: `{pool}`",
        f"- Train: {segment_text(task, 'train')}",
        f"- Validation: {segment_text(task, 'valid')}",
    ]
    if "selection_valid" in task["dataset"]["kwargs"]["segments"]:
        summary.append(f"- Selection validation: {segment_text(task, 'selection_valid')}")
    summary.extend(
        [
        f"- Test: {segment_text(task, 'test')}",
        "",
        "| Split | IC | ICIR | Rank IC | Rank ICIR | Rank IC positive ratio | Days |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        summary.append(
            f"| {row['split']} | {row['IC']:.6f} | {row['ICIR']:.6f} | "
            f"{row['Rank IC']:.6f} | {row['Rank ICIR']:.6f} | "
            f"{row['Rank IC Positive Ratio']:.2%} | {row['Date Count']} |"
        )
    summary.extend(
        [
            "",
            f"- `test_top{topn}_predictions.csv`: 每个测试日预测分数最高的 {topn} 只股票及真实标签。",
            "- `validation_daily_ic.csv` / `test_daily_ic.csv`: 逐日 IC 和 Rank IC。",
            "- `metrics.csv`: validation 与 test 汇总指标。",
        ]
    )
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"recorder_id": recorder_dir.name, "model": model, "pool": pool, **rows[0]}


def main():
    parser = argparse.ArgumentParser(description="导出可读的 recorder 指标和预测")
    parser.add_argument("--experiment-pattern", default=".*")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--full-predictions", action="store_true")
    args = parser.parse_args()
    pattern = re.compile(args.experiment_pattern)
    exported = []

    for meta_path in sorted(MLRUNS.glob("*/meta.yaml")):
        metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        name = str(metadata.get("name", ""))
        if not pattern.search(name):
            continue
        candidates = []
        for recorder_dir in meta_path.parent.iterdir():
            if not recorder_dir.is_dir():
                continue
            rec_meta = recorder_dir / "meta.yaml"
            start_time = 0
            if rec_meta.exists():
                start_time = int((yaml.safe_load(rec_meta.read_text(encoding="utf-8")) or {}).get("start_time", 0))
            candidates.append((start_time, recorder_dir))

        seen_segments = set()
        exported_for_experiment = 0
        for _, recorder_dir in sorted(candidates):
            if "_custom_" in name and exported_for_experiment >= 5:
                break
            task_path = recorder_dir / "artifacts" / "task"
            if not task_path.exists():
                continue
            task = load_pickle(task_path)
            segments = task["dataset"]["kwargs"]["segments"]
            segment_key = tuple((key, tuple(value)) for key, value in sorted(segments.items()))
            if segment_key in seen_segments:
                continue
            result = export_recorder(name, recorder_dir, args.topn, args.full_predictions)
            if result:
                seen_segments.add(segment_key)
                exported_for_experiment += 1
                result["experiment_name"] = name
                exported.append(result)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exported).to_csv(OUTPUT_ROOT / "export_index.csv", index=False)
    print(f"Exported {len(exported)} completed recorders to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
