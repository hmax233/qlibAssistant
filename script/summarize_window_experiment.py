#!/usr/bin/env python3
"""汇总固定Fold下的多种训练窗口实验。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_NAMES = {
    "LinearModel": "Linear",
    "LGBModel": "LightGBM",
    "XGBModel": "XGBoost",
    "CatBoostModel": "CatBoost",
}


def report_dirs(log_path):
    text = Path(log_path).read_text(errors="replace")
    return [Path(value.strip()) for value in re.findall(r"报告目录:\s*(.+)", text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = []
    for log in args.logs:
        for directory in report_dirs(log):
            row = pd.read_csv(directory / "summary.csv").iloc[0].to_dict()
            daily_path = directory / "daily_metrics.csv"
            if daily_path.exists():
                daily = pd.read_csv(daily_path, parse_dates=["datetime"]).set_index("datetime")
                for topk in (3, 5, 10):
                    for benchmark in ("CSI1000", "CSI300"):
                        last_key = f"{benchmark.lower()}_last_valid_date"
                        if last_key not in row or pd.isna(row[last_key]):
                            continue
                        last_date = pd.Timestamp(row[last_key])
                        strategy_equity = daily.loc[last_date, f"Top{topk} Net Equity"]
                        benchmark_equity = daily.loc[last_date, f"{benchmark} Equity"]
                        row[f"Top{topk}_net_comparable_cumulative"] = strategy_equity - 1
                        row[f"Top{topk}_net_return_diff_vs_{benchmark.lower()}"] = strategy_equity - benchmark_equity
                        row[f"Top{topk}_net_excess_vs_{benchmark.lower()}"] = strategy_equity / benchmark_equity - 1
            recorder_path = directory / "recorders.csv"
            if recorder_path.exists():
                recorder = pd.read_csv(recorder_path)
                if "selected_for_ensemble" in recorder.columns:
                    selected = recorder[recorder["selected_for_ensemble"].astype(str).str.lower() == "true"]
                    if not selected.empty:
                        recorder = selected
                rec = recorder.iloc[0]
                for column in [
                    "experiment_id", "experiment_name", "recorder_id",
                    "valid_IC", "valid_ICIR", "valid_Rank IC", "valid_Rank ICIR",
                    "original_test_IC", "original_test_ICIR",
                    "original_test_Rank IC", "original_test_Rank ICIR",
                ]:
                    if column in rec.index:
                        row[column] = rec[column]
            pattern = str(row["experiment_pattern"])
            fold_match = re.search(r"fold([123])", pattern)
            row["fold"] = f"fold{fold_match.group(1)}" if fold_match else "unknown"
            row["model_name"] = MODEL_NAMES.get(row["model"], row["model"])
            row["report_dir"] = str(directory)
            rows.append(row)
    detail = pd.DataFrame(rows).drop_duplicates(
        ["train_months", "fold", "model_name"], keep="last"
    ).sort_values(["model_name", "train_months", "fold"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "window_fold_detail.csv", index=False)

    metrics = [
        "mean_Rank_IC", "Rank_ICIR",
        "Top3_cumulative", "Top3_excess_vs_csi1000", "Top3_excess_vs_csi300",
        "Top3_average_turnover", "Top3_net_cumulative",
        "Top3_net_comparable_cumulative", "Top3_net_return_diff_vs_csi1000", "Top3_net_excess_vs_csi1000",
        "Top3_net_return_diff_vs_csi300", "Top3_net_excess_vs_csi300",
        "Top5_cumulative", "Top5_excess_vs_csi1000", "Top5_excess_vs_csi300",
        "Top5_average_turnover", "Top5_net_cumulative",
        "Top5_net_comparable_cumulative", "Top5_net_return_diff_vs_csi1000", "Top5_net_excess_vs_csi1000",
        "Top5_net_return_diff_vs_csi300", "Top5_net_excess_vs_csi300",
        "Top10_cumulative", "Top10_excess_vs_csi1000", "Top10_excess_vs_csi300",
        "Top10_average_turnover", "Top10_net_cumulative",
        "Top10_net_comparable_cumulative", "Top10_net_return_diff_vs_csi1000", "Top10_net_excess_vs_csi1000",
        "Top10_net_return_diff_vs_csi300", "Top10_net_excess_vs_csi300",
    ]
    average = detail.groupby(["model_name", "train_months"])[metrics].mean().reset_index()
    worst = detail.groupby(["model_name", "train_months"])[metrics].min().reset_index()
    average.to_csv(output / "window_model_average.csv", index=False)
    worst.to_csv(output / "window_model_worst_fold.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plots = [
        ("mean_Rank_IC", "Mean Rank IC"),
        ("Rank_ICIR", "Rank ICIR"),
        ("Top3_excess_vs_csi1000", "Top3 excess vs CSI1000"),
        ("Top10_excess_vs_csi1000", "Top10 excess vs CSI1000"),
    ]
    for ax, (metric, title) in zip(axes.flat, plots):
        for model, group in average.groupby("model_name"):
            ax.plot(group["train_months"], group[metric], marker="o", label=model)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set(title=title, xlabel="Train months", ylabel=metric)
        ax.legend()
    fig.savefig(output / "window_comparison.png", dpi=180)
    plt.close(fig)

    def pct(value):
        return "" if pd.isna(value) else f"{value * 100:.2f}%"

    lines = [
        f"# {'/'.join(map(str, sorted(detail.train_months.unique())))}个月训练窗口实验完整报告",
        "",
        "## 实验设计",
        "",
        f"- 唯一变量：Train历史长度（{'、'.join(map(str, sorted(detail.train_months.unique())))}个月）。",
        "- 控制变量：CSI1000、Alpha158、1日标签、Linear/LightGBM/XGBoost/CatBoost默认参数。",
        "- 数据职责：Train拟合；Valid早停；Selection-valid选模；Test仅做最终比较。",
        "- 三个Fold的Valid、Selection-valid、Test固定；改变月份时只向历史方向移动Train起点。",
        "- 收益口径：TopK理论等资金、无费用、无滑点、允许碎股；不是可直接执行的实盘回测。",
        "",
        "## Fold固定区间",
        "",
        "| Fold | Valid | Selection-valid | Test |",
        "|---|---|---|---|",
        "| Fold1 | 2024-06-15～2024-11-14 | 2024-11-15～2025-04-14 | 2025-04-15～2025-09-15 |",
        "| Fold2 | 2024-11-16～2025-04-15 | 2025-04-16～2025-09-15 | 2025-09-16～2026-02-16 |",
        "| Fold3 | 2025-04-17～2025-09-16 | 2025-09-17～2026-02-16 | 2026-02-17～2026-07-17 |",
        "",
        "## 48组逐Fold完整结果",
        "",
    ]
    for model in ["Linear", "LightGBM", "XGBoost", "CatBoost"]:
        lines.extend([
            f"### {model}", "",
            "| Train月 | Fold | Train区间 | Test区间 | Valid Rank ICIR | Test Rank IC | Test Rank ICIR | Top3毛/净 | Top3换手 | Top3超额/CSI1000 | Top5毛/净 | Top5换手 | Top5超额/CSI1000 | Top10毛/净 | Top10换手 | Top10超额/CSI1000 | CSI1000 | CSI300 | Experiment ID | Recorder |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        subset = detail[detail["model_name"] == model]
        for _, row in subset.iterrows():
            lines.append(
                f"| {int(row['train_months'])} | {row['fold']} | {row['train_start']}～{row['train_end']} | "
                f"{str(row['test_start'])[:10]}～{str(row['test_end'])[:10]} | "
                f"{row.get('valid_Rank ICIR', float('nan')):.4f} | {row['mean_Rank_IC']:.4f} | {row['Rank_ICIR']:.4f} | "
                f"{pct(row['Top3_cumulative'])}/{pct(row.get('Top3_net_cumulative'))} | {pct(row.get('Top3_average_turnover'))} | {pct(row['Top3_excess_vs_csi1000'])} | "
                f"{pct(row['Top5_cumulative'])}/{pct(row.get('Top5_net_cumulative'))} | {pct(row.get('Top5_average_turnover'))} | {pct(row['Top5_excess_vs_csi1000'])} | "
                f"{pct(row['Top10_cumulative'])}/{pct(row.get('Top10_net_cumulative'))} | {pct(row.get('Top10_average_turnover'))} | {pct(row['Top10_excess_vs_csi1000'])} | "
                f"{pct(row['csi1000_cumulative'])} | {pct(row['csi300_cumulative'])} | "
                f"{row.get('experiment_id', '')} | {row.get('recorder_id', '')} |"
            )
        lines.append("")

    lines.extend([
        "## 三Fold平均与最差Fold",
        "",
        "| 模型 | Train月 | 平均Rank IC | 平均Rank ICIR | Top3毛/净 | Top3平均换手 | 平均/最差Top3超额 | Top5毛/净 | Top5平均换手 | Top10毛/净 | Top10平均换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    merged = average.merge(worst, on=["model_name", "train_months"], suffixes=("_avg", "_worst"))
    for _, row in merged.iterrows():
        lines.append(
            f"| {row['model_name']} | {int(row['train_months'])} | {row['mean_Rank_IC_avg']:.4f} | "
            f"{row['Rank_ICIR_avg']:.4f} | {pct(row['Top3_cumulative_avg'])}/{pct(row['Top3_net_cumulative_avg'])} | "
            f"{pct(row['Top3_average_turnover_avg'])} | {pct(row['Top3_excess_vs_csi1000_avg'])}/{pct(row['Top3_excess_vs_csi1000_worst'])} | "
            f"{pct(row['Top5_cumulative_avg'])}/{pct(row['Top5_net_cumulative_avg'])} | {pct(row['Top5_average_turnover_avg'])} | "
            f"{pct(row['Top10_cumulative_avg'])}/{pct(row['Top10_net_cumulative_avg'])} | {pct(row['Top10_average_turnover_avg'])} |"
        )
    lines.extend([
        "",
        "## 可复现信息",
        "",
        "```bash",
        "PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python",
        "$PY script/run_fixed_folds.py --models Linear LightGBM XGBoost CatBoost --train-months <45|60|84|120> --pool csi1000 --tag-prefix windowcmp --date-tag 260720",
        "```",
        "",
        "- 训练日志：`.qlibAssistant/logs/window_length_experiment_20260720.log`",
        "- 新窗口评估日志：`.qlibAssistant/logs/window_length_evaluation_20260720.log`",
        "- 45个月重算日志：`.qlibAssistant/logs/window_length_baseline45_evaluation_20260720.log`",
        "- 完整CSV：`window_fold_detail.csv`",
        "- 平均结果：`window_model_average.csv`",
        "- 最差Fold：`window_model_worst_fold.csv`",
        "- 图：`window_comparison.png`",
        "",
        "## 异常与限制",
        "",
        "- Linear训练出现矩阵乘法overflow/divide-by-zero警告；其结果需作为数值稳定性风险单独处理。",
        "- Fold3指数由Qlib加Tushare缓存补齐，完整可比的最后信号日为2026-07-15。",
        "- 当前收益未计手续费、滑点、整数手、涨跌停和真实换手率。",
        "- 净收益列使用相邻TopK重合度计算换手，并按完整换仓0.15%的简化综合成本扣减；连续持仓不重复收费。",
        "- 本轮已经观察Test，因此只能形成研究结论；不能继续在同一Test上无限调参后声称无偏泛化。",
        "- 模型窗口最终选择应优先依据Selection-valid，并通过新的前向模拟期确认。",
    ])
    (output / "window_experiment_detailed_report.md").write_text("\n".join(lines) + "\n")
    expected = detail.model_name.nunique() * detail.train_months.nunique() * detail.fold.nunique()
    print(f"detail rows={len(detail)}, expected={expected}")
    print(average.to_string(index=False))
    print(f"汇总目录: {output.resolve()}")


if __name__ == "__main__":
    main()
