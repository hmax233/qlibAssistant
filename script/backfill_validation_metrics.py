#!/usr/bin/env python3
"""为历史 MLflow recorder 补算 validation 信号指标。

默认跳过已有 ``valid_sig_analysis`` 的 recorder，支持中断后继续运行。
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import qlib
from loguru import logger
from qlib.config import C
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.utils import init_instance_by_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = PROJECT_ROOT / "roll"
sys.path.insert(0, str(ROLL_DIR))

from validation_analysis import ensure_validation_analysis  # noqa: E402


def init_qlib():
    mlruns = PROJECT_ROOT / ".qlibAssistant" / "mlruns"
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(mlruns.resolve())
    qlib.init(
        provider_uri=str(Path("~/.qlib/qlib_data/cn_data").expanduser()),
        region=REG_CN,
        exp_manager=exp_manager,
    )


def main():
    parser = argparse.ArgumentParser(description="补算 recorder validation 指标")
    parser.add_argument("--force", action="store_true", help="覆盖已有 validation 指标")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 recorder（测试用）")
    parser.add_argument("--recorder-id", help="只处理指定 recorder")
    args = parser.parse_args()

    init_qlib()
    jobs = []
    visited = 0
    for exp_name in R.list_experiments():
        if exp_name == "Default":
            continue
        exp = R.get_exp(experiment_name=exp_name)
        for rid in exp.list_recorders():
            if args.recorder_id and rid != args.recorder_id:
                continue
            rec = exp.get_recorder(recorder_id=rid)
            artifacts = rec.list_artifacts()
            if "params.pkl" not in artifacts or "task" not in artifacts:
                continue
            if args.limit is not None and visited >= args.limit:
                break
            visited += 1
            task = rec.load_object("task")
            jobs.append((exp_name, rec, task))
        if args.limit is not None and visited >= args.limit:
            break

    success = 0
    failed = 0
    grouped = defaultdict(list)
    for exp_name, rec, task in jobs:
        rid = rec.id
        if not args.force and "valid_sig_analysis" in rec.list_artifacts():
            try:
                metrics, _ = ensure_validation_analysis(rec)
                print(
                    f"OK {rid} | {exp_name} | "
                    f"Valid Rank IC={metrics['Rank IC']:.6f} "
                    f"Rank ICIR={metrics['Rank ICIR']:.6f}"
                )
                success += 1
            except Exception as exc:
                logger.exception(f"FAILED {rid} | {exp_name}: {exc}")
                failed += 1
            continue

        dataset_config = task["dataset"]
        key = json.dumps(dataset_config, sort_keys=True, default=str)
        grouped[key].append((exp_name, rec, dataset_config))

    for group_index, group_jobs in enumerate(grouped.values(), start=1):
        dataset_config = copy.deepcopy(group_jobs[0][2])
        valid_segment = dataset_config["kwargs"]["segments"]["valid"]
        print(
            f"加载 validation Dataset {group_index}/{len(grouped)}: "
            f"{valid_segment}，供 {len(group_jobs)} 个模型复用"
        )
        try:
            dataset = init_instance_by_config(dataset_config)
        except Exception as exc:
            logger.exception(f"Dataset 加载失败 {valid_segment}: {exc}")
            failed += len(group_jobs)
            continue

        for exp_name, rec, _ in group_jobs:
            rid = rec.id
            try:
                metrics, _ = ensure_validation_analysis(rec, force=args.force, dataset=dataset)
                print(
                    f"OK {rid} | {exp_name} | "
                    f"Valid Rank IC={metrics['Rank IC']:.6f} "
                    f"Rank ICIR={metrics['Rank ICIR']:.6f}"
                )
                success += 1
            except Exception as exc:
                logger.exception(f"FAILED {rid} | {exp_name}: {exc}")
                failed += 1
        del dataset
        gc.collect()

    print(f"完成：成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()
