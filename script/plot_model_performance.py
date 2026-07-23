#!/usr/bin/env python3
"""绘制一个训练批次全部 recorder 的 validation/test Rank IC 性能图。"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / ".qlibAssistant" / "mlruns"
ANALYSIS = PROJECT_ROOT / ".qlibAssistant" / "analysis"

MODEL_LABELS = {
    "XGBModel": "XGBoost",
    "LinearModel": "Linear",
    "DEnsembleModel": "DoubleEnsemble",
    "LGBModel": "LightGBM",
    "CatBoostModel": "CatBoost",
}


def load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def metric_row(ic, ric):
    return {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()),
        "Rank IC": float(ric.mean()),
        "Rank ICIR": float(ric.mean() / ric.std()),
    }


def total_months(task):
    segments = task["dataset"]["kwargs"]["segments"]
    start = pd.Timestamp(segments["train"][0])
    end = pd.Timestamp(segments["test"][1])
    return round((end - start).days / 30.4375)


def collect(pattern):
    regex = re.compile(pattern)
    rows = []
    for meta in sorted(MLRUNS.glob("*/meta.yaml")):
        info = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        exp_name = str(info.get("name", ""))
        if not regex.search(exp_name):
            continue
        candidates = []
        for rec in meta.parent.iterdir():
            rec_meta = rec / "meta.yaml"
            start_time = 0
            if rec_meta.exists():
                start_time = int((yaml.safe_load(rec_meta.read_text(encoding="utf-8")) or {}).get("start_time", 0))
            candidates.append((start_time, rec))
        seen_segments = set()
        collected_for_experiment = 0
        for _, rec in sorted(candidates):
            if "_custom_" in exp_name and collected_for_experiment >= 5:
                break
            artifacts = rec / "artifacts"
            required = [
                artifacts / "task",
                artifacts / "sig_analysis" / "ic.pkl",
                artifacts / "sig_analysis" / "ric.pkl",
                artifacts / "valid_sig_analysis" / "ic.pkl",
                artifacts / "valid_sig_analysis" / "ric.pkl",
            ]
            if not rec.is_dir() or not all(path.exists() for path in required):
                continue
            task = load(artifacts / "task")
            segments = task["dataset"]["kwargs"]["segments"]
            segment_key = tuple((key, tuple(value)) for key, value in sorted(segments.items()))
            if segment_key in seen_segments:
                continue
            seen_segments.add(segment_key)
            collected_for_experiment += 1
            model_class = task["model"]["class"]
            base = {
                "experiment_name": exp_name,
                "recorder_id": rec.name,
                "model": MODEL_LABELS.get(model_class, model_class),
                "window_months": total_months(task),
            }
            for split, folder in (("Validation", "valid_sig_analysis"), ("Test", "sig_analysis")):
                ic = load(artifacts / folder / "ic.pkl")
                ric = load(artifacts / folder / "ric.pkl")
                rows.append({**base, "split": split, **metric_row(ic, ric)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="绘制 Rank IC / Rank ICIR 散点图")
    parser.add_argument("--experiment-pattern", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    data = collect(args.experiment_pattern)
    if data.empty:
        raise SystemExit("No completed recorders matched")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / f"{args.output_prefix}.csv"
    png_path = ANALYSIS / f"{args.output_prefix}.png"
    data.to_csv(csv_path, index=False)

    colors = {
        "XGBoost": "#4C78A8",
        "Linear": "#F58518",
        "DoubleEnsemble": "#54A24B",
        "LightGBM": "#E45756",
        "CatBoost": "#B279A2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, split in zip(axes, ["Validation", "Test"]):
        part = data[data["split"] == split]
        for model, group in part.groupby("model"):
            ax.scatter(
                group["Rank IC"],
                group["Rank ICIR"],
                s=55 + group["window_months"] * 1.4,
                alpha=0.82,
                label=model,
                color=colors.get(model),
                edgecolor="white",
                linewidth=0.7,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    f"{int(row['window_months'])}m",
                    (row["Rank IC"], row["Rank ICIR"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_title(f"{split}: Rank IC vs Rank ICIR")
        ax.set_xlabel("Rank IC")
        ax.set_ylabel("Rank ICIR")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5, frameon=False)
    fig.suptitle(f"{args.experiment_pattern} — recorder performance", y=0.995)
    fig.subplots_adjust(top=0.84, wspace=0.12)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
