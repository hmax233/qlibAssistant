#!/usr/bin/env python3
"""按 validation 选模，在所有 recorder 的共同 test 日期上生成统一评估报告。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".qlibAssistant" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.config import C
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R


ROLL_DIR = PROJECT_ROOT / "roll"
TUSHARE_INDEX_CACHE = PROJECT_ROOT / ".qlibAssistant" / "cache" / "tushare_index_daily.csv"
sys.path.insert(0, str(ROLL_DIR))

from validation_analysis import load_validation_analysis, metrics_from_ic  # noqa: E402
from evaluation_variants import evaluate_board_variants  # noqa: E402


def init_qlib():
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(
        (PROJECT_ROOT / ".qlibAssistant" / "mlruns").resolve()
    )
    qlib.init(
        provider_uri=str(Path("~/.qlib/qlib_data/cn_data").expanduser()),
        region=REG_CN,
        exp_manager=exp_manager,
    )


def as_series(obj, name):
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    result = obj.copy()
    result.name = name
    return result


def main():
    parser = argparse.ArgumentParser(description="统一共同 test 区间的 validation 选模报告")
    parser.add_argument("--experiment-pattern", required=True, help="实验名称包含的文本")
    parser.add_argument(
        "--exact-experiment-name",
        action="store_true",
        help="要求实验名与 experiment-pattern 完全一致，避免 fold1 同时匹配 h3/h5/raw 实验",
    )
    parser.add_argument("--model-class", help="可选：仅评估指定 task model class，如 XGBModel")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.0015,
        help="完整替换全部持仓时扣除的简化综合成本，默认0.15%%；按TopK实际换手比例缩放",
    )
    parser.add_argument("--test-start", help="可选：限制统一测试起始日期 YYYY-MM-DD")
    parser.add_argument("--test-end", help="可选：限制统一测试结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--holding-period",
        type=int,
        default=1,
        help="标签持有交易日数；大于1时用H组错峰组合计算非重叠收益",
    )
    parser.add_argument(
        "--weighting",
        choices=["validation_rank_icir", "equal"],
        default="validation_rank_icir",
        help="集成权重：selection-validation Rank ICIR 或等权",
    )
    parser.add_argument(
        "--top-models",
        type=int,
        default=None,
        help="按 validation Rank ICIR 排序后仅集成前 N 个 recorder",
    )
    args = parser.parse_args()

    init_qlib()
    selected = []
    recorder_rows = []
    for exp_name in R.list_experiments():
        matched = (
            exp_name == args.experiment_pattern
            if args.exact_experiment_name
            else args.experiment_pattern in exp_name
        )
        if not matched:
            continue
        exp = R.get_exp(experiment_name=exp_name)
        recorder_pairs = []
        for rid in exp.list_recorders():
            candidate = exp.get_recorder(recorder_id=rid)
            start_time = candidate.info.get("start_time", 0) if candidate.info else 0
            recorder_pairs.append((start_time, rid, candidate))
        if "_custom_" in exp_name:
            recorder_pairs = sorted(recorder_pairs)[:5]
        for _, rid, rec in recorder_pairs:
            valid_metrics, valid_values = load_validation_analysis(rec)
            test_ic = rec.load_object("sig_analysis/ic.pkl")
            test_ric = rec.load_object("sig_analysis/ric.pkl")
            test_metrics, _ = metrics_from_ic(test_ic, test_ric)
            task = rec.load_object("task")
            if args.model_class and task["model"]["class"] != args.model_class:
                continue
            segments = task["dataset"]["kwargs"]["segments"]
            passed = all(np.isfinite(v) and v >= args.threshold for v in valid_values)
            row = {
                "experiment_id": exp.id,
                "experiment_name": exp_name,
                "recorder_id": rid,
                "model": task["model"]["class"],
                "selected_by_validation": passed,
                "train_start": segments["train"][0],
                "train_end": segments["train"][1],
                "valid_start": segments["valid"][0],
                "valid_end": segments["valid"][1],
                "test_start": segments["test"][0],
                "test_end": segments["test"][1],
            }
            row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
            row.update({f"original_test_{k}": v for k, v in test_metrics.items()})
            recorder_rows.append(row)
            if passed:
                selected.append((rec, float(valid_metrics["Rank ICIR"]), row))

    if not selected:
        raise SystemExit("没有符合 validation 阈值的 recorder")

    selected.sort(key=lambda item: item[1], reverse=True)
    if args.top_models is not None:
        selected = selected[: args.top_models]
    selected_ids = {item[0].id for item in selected}
    for row in recorder_rows:
        row["selected_for_ensemble"] = row["recorder_id"] in selected_ids

    if args.weighting == "equal":
        weights = [1.0 / len(selected)] * len(selected)
    else:
        total_weight = sum(item[1] for item in selected)
        if total_weight <= 0:
            raise SystemExit("validation Rank ICIR 权重之和非正，请改用 --weighting equal")
        weights = [item[1] / total_weight for item in selected]
    prediction_columns = []
    labels = []
    common_dates = None
    for (rec, rank_icir, row), weight in zip(selected, weights):
        pred = as_series(rec.load_object("pred.pkl"), rec.id)
        label = as_series(rec.load_object("label.pkl"), "label")
        dates = pd.Index(pred.index.get_level_values("datetime").unique())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
        prediction_columns.append(pred * weight)
        labels.append(label)
        row["ensemble_weight"] = weight

    common_dates = common_dates.sort_values()
    if args.test_start:
        common_dates = common_dates[common_dates >= pd.Timestamp(args.test_start)]
    if args.test_end:
        common_dates = common_dates[common_dates <= pd.Timestamp(args.test_end)]
    if common_dates.empty:
        raise SystemExit("指定范围内没有共同 test 日期")
    predictions = pd.concat(prediction_columns, axis=1)
    predictions = predictions[
        predictions.index.get_level_values("datetime").isin(common_dates)
    ]
    score = predictions.sum(axis=1, min_count=len(prediction_columns)).rename("score")
    label = labels[0]
    frame = pd.concat([score, label], axis=1).dropna()

    # 对 raw-label 模型，这些指标可直接衡量“预测收益率”的校准程度。
    # 对默认 CSZScoreNorm 标签模型也保留输出，便于直观看到其 score 并非收益率单位。
    residual = frame["score"] - frame["label"]
    absolute_metrics = {
        "prediction_mean": frame["score"].mean(),
        "prediction_std": frame["score"].std(),
        "realized_return_mean": frame["label"].mean(),
        "realized_return_std": frame["label"].std(),
        "absolute_MAE": residual.abs().mean(),
        "absolute_RMSE": np.sqrt(np.mean(np.square(residual))),
        "prediction_bias": residual.mean(),
        "sample_correlation": frame["score"].corr(frame["label"]),
    }

    bin_count = min(10, frame["score"].nunique())
    if bin_count >= 2:
        calibration = (
            frame.assign(score_bin=pd.qcut(frame["score"], q=bin_count, duplicates="drop"))
            .groupby("score_bin", observed=True)
            .agg(
                samples=("label", "size"),
                predicted_mean=("score", "mean"),
                predicted_min=("score", "min"),
                predicted_max=("score", "max"),
                realized_mean=("label", "mean"),
                realized_win_rate=("label", lambda value: (value > 0).mean()),
            )
            .reset_index()
        )
        calibration["score_bin"] = calibration["score_bin"].astype(str)
    else:
        calibration = pd.DataFrame()

    report_topks = sorted({1, 3, 5, 10, args.topk})

    def daily_metrics(group):
        ranked = group.sort_values("score", ascending=False)
        result = {
            "IC": group["score"].corr(group["label"]),
            "Rank IC": group["score"].corr(group["label"], method="spearman"),
            "Universe Mean Return": group["label"].mean(),
        }
        for topk in report_topks:
            result[f"Top{topk} Mean Return"] = ranked.head(topk)["label"].mean()
        result["Long Excess Return"] = (
            result[f"Top{args.topk} Mean Return"] - result["Universe Mean Return"]
        )
        result["Top1 Return"] = ranked.iloc[0]["label"]
        return pd.Series(result)

    daily = frame.groupby(level="datetime").apply(daily_metrics)

    # 每日滚动组合的换手：连续留在TopK的股票不卖出重买，仅替换部分扣成本。
    topk_members = {}
    for date, group in frame.groupby(level="datetime"):
        ranked_instruments = list(
            group.sort_values("score", ascending=False).index.get_level_values("instrument")
        )
        for topk in report_topks:
            topk_members.setdefault(topk, {})[date] = set(ranked_instruments[:topk])
    for topk in report_topks:
        previous = None
        turnover = []
        for date in daily.index:
            current = topk_members[topk][date]
            if previous is None:
                value = 1.0
            else:
                value = 1.0 - len(current & previous) / max(topk, 1)
            turnover.append(value)
            previous = current
        daily[f"Top{topk} Turnover"] = turnover
        daily[f"Top{topk} Net Return"] = (
            daily[f"Top{topk} Mean Return"]
            - daily[f"Top{topk} Turnover"] * args.cost_rate
        )
        daily[f"Top{topk} Net Equity"] = (1 + daily[f"Top{topk} Net Return"]).cumprod()
    daily["TopK Cumulative"] = (1 + daily[f"Top{args.topk} Mean Return"]).cumprod() - 1
    daily["Universe Cumulative"] = (1 + daily["Universe Mean Return"]).cumprod() - 1
    daily["Excess Cumulative"] = (1 + daily["Long Excess Return"]).cumprod() - 1

    def staggered_equity(returns: pd.Series, holding_period: int) -> pd.Series:
        """H个等资金 cohort，每个 cohort 每隔H日调仓，避免多日标签重叠复利。"""
        curves = []
        for offset in range(holding_period):
            growth = pd.Series(1.0, index=returns.index, dtype=float)
            positions = np.arange(offset, len(returns), holding_period)
            if len(positions):
                growth.iloc[positions] = 1.0 + returns.iloc[positions].fillna(0.0)
            curves.append(growth.cumprod())
        return pd.concat(curves, axis=1).mean(axis=1)

    holding_period = args.holding_period
    if holding_period < 1:
        raise SystemExit("holding-period 必须 >= 1")
    daily["Executable TopK Equity"] = staggered_equity(
        daily[f"Top{args.topk} Mean Return"], holding_period
    )
    for topk in report_topks:
        daily[f"Top{topk} Equity"] = staggered_equity(
            daily[f"Top{topk} Mean Return"], holding_period
        )
    daily["Executable Universe Equity"] = staggered_equity(
        daily["Universe Mean Return"], holding_period
    )
    daily["Executable Excess"] = (
        daily["Executable TopK Equity"] / daily["Executable Universe Equity"] - 1
    )

    # 官方指数基准使用与模型标签一致的时点：信号日之后第一个可交易收盘价买入，
    # 持有 H 个交易日后按收盘价卖出。中证1000是本实验主基准，沪深300作风格参照。
    benchmark_expression = (
        f"Ref($close,-{holding_period + 1})/Ref($close,-1)-1"
    )
    benchmark = D.features(
        ["SH000852", "SH000300"],
        [benchmark_expression],
        start_time=common_dates.min(),
        end_time=common_dates.max(),
        freq="day",
    ).iloc[:, 0]
    benchmark_names = {"SH000852": "CSI1000", "SH000300": "CSI300"}
    tushare_returns = {}
    if TUSHARE_INDEX_CACHE.exists():
        cache = pd.read_csv(TUSHARE_INDEX_CACHE, parse_dates=["datetime"])
        for display_name, group in cache.groupby("index"):
            close = group.drop_duplicates("datetime").set_index("datetime")["close"].sort_index()
            tushare_returns[display_name] = close.shift(-(holding_period + 1)) / close.shift(-1) - 1
    benchmark_coverage = {}
    for instrument, display_name in benchmark_names.items():
        try:
            index_return = benchmark.xs(instrument, level="instrument")
        except KeyError:
            index_return = pd.Series(dtype=float)
        index_return = index_return.reindex(daily.index)
        if display_name in tushare_returns:
            index_return = index_return.combine_first(
                tushare_returns[display_name].reindex(daily.index)
            )
        benchmark_coverage[display_name] = {
            "days": int(index_return.notna().sum()),
            "last_date": index_return.dropna().index.max() if index_return.notna().any() else pd.NaT,
        }
        daily[f"{display_name} Return"] = index_return
        daily[f"{display_name} Equity"] = staggered_equity(
            index_return, holding_period
        )
        daily[f"Excess vs {display_name}"] = (
            daily["Executable TopK Equity"] / daily[f"{display_name} Equity"] - 1
        )
        for topk in report_topks:
            daily[f"Top{topk} Excess vs {display_name}"] = (
                daily[f"Top{topk} Equity"] / daily[f"{display_name} Equity"] - 1
            )
            daily[f"Top{topk} Return Diff vs {display_name}"] = (
                daily[f"Top{topk} Equity"] - daily[f"{display_name} Equity"]
            )

    def benchmark_last_value(display_name, column):
        last_date = benchmark_coverage[display_name]["last_date"]
        return daily.loc[last_date, column] if pd.notna(last_date) else np.nan

    selected_rows = [item[2] for item in selected]

    def common_or_mixed(key):
        values = sorted({str(row[key]) for row in selected_rows})
        return values[0] if len(values) == 1 else "mixed:" + "|".join(values)

    train_start_value = common_or_mixed("train_start")
    train_end_value = common_or_mixed("train_end")
    if not train_start_value.startswith("mixed:") and not train_end_value.startswith("mixed:"):
        train_delta = relativedelta(
            pd.Timestamp(train_end_value) + pd.Timedelta(days=1),
            pd.Timestamp(train_start_value),
        )
        train_months = train_delta.years * 12 + train_delta.months
    else:
        train_months = "mixed"

    summary = pd.DataFrame(
        [
            {
                "experiment_pattern": args.experiment_pattern,
                "model": common_or_mixed("model"),
                "train_months": train_months,
                "train_start": train_start_value,
                "train_end": train_end_value,
                "selected_recorders": len(selected),
                "top_models": args.top_models or "all_passed",
                "weighting": args.weighting,
                "holding_period": holding_period,
                "cost_rate": args.cost_rate,
                "test_start": daily.index.min(),
                "test_end": daily.index.max(),
                "test_trading_days": len(daily),
                "mean_IC": daily["IC"].mean(),
                "ICIR": daily["IC"].mean() / daily["IC"].std(),
                "mean_Rank_IC": daily["Rank IC"].mean(),
                "Rank_ICIR": daily["Rank IC"].mean() / daily["Rank IC"].std(),
                "Rank_IC_positive_ratio": (daily["Rank IC"] > 0).mean(),
                f"Top{args.topk}_win_rate": (daily[f"Top{args.topk} Mean Return"] > 0).mean(),
                "Top1_win_rate": (daily["Top1 Return"] > 0).mean(),
                "gross_topk_cumulative": daily["TopK Cumulative"].iloc[-1],
                "gross_universe_cumulative": daily["Universe Cumulative"].iloc[-1],
                "gross_excess_cumulative": daily["Excess Cumulative"].iloc[-1],
                "executable_topk_cumulative": daily["Executable TopK Equity"].iloc[-1] - 1,
                "executable_universe_cumulative": daily["Executable Universe Equity"].iloc[-1] - 1,
                "executable_excess_cumulative": daily["Executable Excess"].iloc[-1],
                "csi1000_cumulative": benchmark_last_value("CSI1000", "CSI1000 Equity") - 1,
                "csi1000_covered_days": benchmark_coverage["CSI1000"]["days"],
                "csi1000_last_valid_date": benchmark_coverage["CSI1000"]["last_date"],
                "excess_vs_csi1000": benchmark_last_value("CSI1000", "Excess vs CSI1000"),
                "csi300_cumulative": benchmark_last_value("CSI300", "CSI300 Equity") - 1,
                "csi300_covered_days": benchmark_coverage["CSI300"]["days"],
                "csi300_last_valid_date": benchmark_coverage["CSI300"]["last_date"],
                "excess_vs_csi300": benchmark_last_value("CSI300", "Excess vs CSI300"),
            }
        ]
    )
    for key, value in absolute_metrics.items():
        summary[key] = value
    for topk in report_topks:
        summary[f"Top{topk}_win_rate"] = (
            daily[f"Top{topk} Mean Return"] > 0
        ).mean()
        summary[f"Top{topk}_cumulative"] = daily[f"Top{topk} Equity"].iloc[-1] - 1
        summary[f"Top{topk}_average_turnover"] = daily[f"Top{topk} Turnover"].mean()
        summary[f"Top{topk}_net_cumulative"] = daily[f"Top{topk} Net Equity"].iloc[-1] - 1
        summary[f"Top{topk}_excess_vs_csi1000"] = daily[
            f"Top{topk} Excess vs CSI1000"
        ].loc[benchmark_coverage["CSI1000"]["last_date"]]
        summary[f"Top{topk}_return_diff_vs_csi1000"] = daily[
            f"Top{topk} Return Diff vs CSI1000"
        ].loc[benchmark_coverage["CSI1000"]["last_date"]]
        summary[f"Top{topk}_excess_vs_csi300"] = daily[
            f"Top{topk} Excess vs CSI300"
        ].loc[benchmark_coverage["CSI300"]["last_date"]]
        summary[f"Top{topk}_return_diff_vs_csi300"] = daily[
            f"Top{topk} Return Diff vs CSI300"
        ].loc[benchmark_coverage["CSI300"]["last_date"]]

    output = PROJECT_ROOT / ".qlibAssistant" / "analysis" / f"evaluation_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(recorder_rows).to_csv(output / "recorders.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    daily.to_csv(output / "daily_metrics.csv")
    frame.reset_index().to_csv(output / "ensemble_test_predictions.csv", index=False)
    calibration.to_csv(output / "absolute_return_calibration.csv", index=False)
    board_summary, board_daily = evaluate_board_variants(frame, cost_rate=args.cost_rate)
    board_summary.to_csv(output / "board_variant_summary.csv", index=False)
    board_daily.to_csv(output / "board_variant_daily.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    daily[
        [
            "Executable TopK Equity",
            f"Top{args.topk} Net Equity",
            "Executable Universe Equity",
            "CSI1000 Equity",
            "CSI300 Equity",
        ]
    ].plot(ax=axes[0])
    axes[0].set_title(
        f"{args.experiment_pattern}: gross and cost-adjusted test (Top {args.topk}, hold {holding_period}d)"
    )
    axes[0].set_ylabel("Equity / excess")
    daily["Rank IC"].plot(kind="bar", ax=axes[1], color=np.where(daily["Rank IC"] >= 0, "#2ca02c", "#d62728"))
    axes[1].axhline(daily["Rank IC"].mean(), color="black", linestyle="--", label="Mean Rank IC")
    axes[1].set_title("Daily Rank IC")
    axes[1].tick_params(axis="x", labelrotation=60)
    axes[1].legend()
    fig.savefig(output / "test_report.png", dpi=160)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"报告目录: {output}")
    print(
        f"注意：Executable 毛收益采用 {holding_period} 组错峰模拟；TopK Net 按实际换手率扣除"
        f"简化综合成本 {args.cost_rate:.4%}。仍未逐笔模拟最低佣金、涨跌停、成交失败和真实冲击成本。"
    )


if __name__ == "__main__":
    main()
