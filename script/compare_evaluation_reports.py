#!/usr/bin/env python3
"""汇总多模型、多 fold 的 evaluate_batch 报告并绘制统一比较图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="MODEL,FOLD,DIR",
        help="可重复；例如 LightGBM,fold1,.qlibAssistant/analysis/evaluation_xxx",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.report:
        model, fold, directory = spec.split(",", 2)
        source = Path(directory) / "summary.csv"
        row = pd.read_csv(source).iloc[0].to_dict()
        row.update({"model_name": model, "fold": fold, "report_dir": str(Path(directory))})
        rows.append(row)

    detail = pd.DataFrame(rows).sort_values(["fold", "model_name"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "model_fold_comparison.csv", index=False)

    metrics = [
        "mean_Rank_IC",
        "Rank_ICIR",
        "Top3_win_rate",
        "Top3_cumulative",
        "Top3_excess_vs_csi1000",
        "Top3_return_diff_vs_csi1000",
        "Top3_excess_vs_csi300",
        "Top3_return_diff_vs_csi300",
        "Top5_win_rate",
        "Top5_cumulative",
        "Top5_excess_vs_csi1000",
        "Top5_return_diff_vs_csi1000",
        "Top5_excess_vs_csi300",
        "Top5_return_diff_vs_csi300",
        "Top10_win_rate",
        "Top10_cumulative",
        "Top10_excess_vs_csi1000",
        "Top10_return_diff_vs_csi1000",
        "Top10_excess_vs_csi300",
        "Top10_return_diff_vs_csi300",
        "executable_topk_cumulative",
        "excess_vs_csi1000",
        "excess_vs_csi300",
    ]
    metrics = [metric for metric in metrics if metric in detail.columns]
    average = detail.groupby("model_name")[metrics].mean().reset_index()
    worst = detail.groupby("model_name")[metrics].min().reset_index()
    average.to_csv(output / "model_average.csv", index=False)
    worst.to_csv(output / "model_worst_fold.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for model, group in detail.groupby("model_name"):
        axes[0].scatter(group["mean_Rank_IC"], group["Rank_ICIR"], s=65, label=model)
        for _, row in group.iterrows():
            axes[0].annotate(row["fold"], (row["mean_Rank_IC"], row["Rank_ICIR"]), fontsize=8)
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].axvline(0, color="grey", linewidth=0.8)
    axes[0].set(xlabel="Mean Rank IC", ylabel="Rank ICIR", title="Ranking quality by fold")
    axes[0].legend()

    pivot = detail.pivot(index="fold", columns="model_name", values="excess_vs_csi1000")
    pivot.plot(kind="bar", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="Cumulative excess", title="Top10 gross excess vs CSI1000")
    axes[1].tick_params(axis="x", rotation=0)
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)

    identity = [
        column
        for column in ["model_name", "fold", "train_months", "train_start", "train_end", "test_start", "test_end"]
        if column in detail.columns
    ]
    print(detail[[*identity, *metrics]].to_string(index=False))
    print(f"汇总目录: {output.resolve()}")


if __name__ == "__main__":
    main()
