#!/usr/bin/env python3
"""顺序训练一批 Qlib 模型，并用明确的股票池/批次标签命名实验。"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


DEFAULT_MODELS = ["XGBoost", "Linear", "LightGBM", "CatBoost"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = PROJECT_ROOT / "roll"


def run_batch_experiments(
    models,
    pool,
    dataset,
    run_tag,
    dry_run=False,
    split_selection_valid=False,
    window_months=None,
    end_date=None,
    label_horizon=1,
    normalize_features=False,
    raw_label=False,
    model_preset=None,
    segments_json=None,
):
    combinations = list(itertools.product(models, [dataset], [pool], ["custom"]))
    recorder_count = 1 if segments_json else (len(window_months) if window_months else 5)
    logger.info(
        f"总共 {len(combinations)} 个算法任务；每个任务生成 {recorder_count} 个时间窗口 recorder"
    )

    failures = []
    for index, (model, dataset_name, stock_pool, rolling_type) in enumerate(combinations, 1):
        cmd = [
            sys.executable,
            "./roll.py",
            "--pfx_name=EXP",
            f"--sfx_name={run_tag}",
            f"--model_name={model}",
            f"--dataset_name={dataset_name}",
            f"--stock_pool={stock_pool}",
            f"--rolling_type={rolling_type}",
            f"--label_horizon={label_horizon}",
        ]
        if split_selection_valid:
            cmd.append("--split_selection_valid=true")
        if normalize_features:
            cmd.append("--normalize_features=true")
        if raw_label:
            cmd.append("--raw_label=true")
        if model_preset:
            cmd.append(f"--model_preset={model_preset}")
        if window_months:
            cmd.append(f"--custom_months={','.join(str(value) for value in window_months)}")
        if end_date:
            cmd.append(f"--custom_end_date={end_date}")
        if segments_json:
            # 先解析以便在入口尽早发现格式错误，再原样交给 TrainCLI。
            json.loads(segments_json)
            cmd.append(f"--fixed_segments={segments_json}")
        cmd.extend(["train", "start_custom"])
        logger.info(f"[{index}/{len(combinations)}] {' '.join(cmd)}")
        if dry_run:
            continue

        started = time.time()
        result = subprocess.run(cmd, cwd=ROLL_DIR, check=False)
        elapsed = time.time() - started
        if result.returncode:
            failures.append((model, result.returncode))
            logger.error(f"{model} 失败，exit={result.returncode}，耗时 {elapsed:.1f}s")
        else:
            logger.info(f"{model} 完成，耗时 {elapsed:.1f}s")
        subprocess.run(
            [sys.executable, "script/build_experiment_index.py"],
            cwd=PROJECT_ROOT,
            check=False,
        )

    subprocess.run(
        [
            sys.executable,
            "script/export_readable_artifacts.py",
            "--experiment-pattern",
            f"{pool}.*{run_tag}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if failures:
        raise SystemExit(f"训练失败: {failures}")


def main():
    parser = argparse.ArgumentParser(description="批量训练 Qlib custom 时间窗口")
    parser.add_argument("--pool", default="csi300", help="例如 csi300/csi500/csi1000")
    parser.add_argument("--dataset", default="Alpha158")
    parser.add_argument("--run-tag", default=time.strftime("retrain%y%m%d"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--split-selection-valid",
        action="store_true",
        help="使用独立 selection_valid 选模，valid 仅供模型训练监控",
    )
    parser.add_argument(
        "--model-preset",
        help=(
            "模型参数预设；LightGBM读取roll/model_params.yaml，"
            "TRA可选tra_smoke或tra_official_full"
        ),
    )
    parser.add_argument("--window-months", nargs="+", type=int)
    parser.add_argument("--end-date", help="自定义窗口截止日期 YYYY-MM-DD")
    parser.add_argument(
        "--segments-json",
        help="固定 train/valid/selection_valid/test 的 JSON；用于公平对照实验",
    )
    parser.add_argument("--label-horizon", type=int, default=1, choices=[1, 3, 5])
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument(
        "--raw-label",
        action="store_true",
        help="取消标签横截面Z-score，直接预测未来绝对收益率",
    )
    args = parser.parse_args()
    run_batch_experiments(
        args.models,
        args.pool,
        args.dataset,
        args.run_tag,
        args.dry_run,
        args.split_selection_valid,
        args.window_months,
        args.end_date,
        args.label_horizon,
        args.normalize_features,
        args.raw_label,
        args.model_preset,
        args.segments_json,
    )


if __name__ == "__main__":
    main()
