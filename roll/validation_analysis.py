"""为 Qlib recorder 生成并缓存 validation 信号评价产物。"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from qlib.contrib.eva.alpha import calc_ic
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from utils import restore_model_runtime_state


VALID_ANALYSIS_DIR = "valid_sig_analysis"
VALID_REQUIRED_FILES = ("pred.pkl", "label.pkl", "ic.pkl", "ric.pkl", "metrics.pkl")


def save_readable_validation_artifacts(recorder, pred, label, ic, ric, metrics, segment):
    """在保留 Qlib PKL 的同时，将 validation 产物双写为可直接查看的 CSV。"""

    with tempfile.TemporaryDirectory(prefix="qlib-valid-csv-") as temp_dir:
        output = Path(temp_dir)
        pred.reset_index().to_csv(output / "pred.csv", index=False)
        label.reset_index().to_csv(output / "label.csv", index=False)
        pd.concat(
            [ic.rename("IC"), ric.rename("Rank IC")], axis=1
        ).rename_axis("datetime").reset_index().to_csv(output / "daily_ic.csv", index=False)
        pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(
            [{"segment": segment["name"], "range": json.dumps(segment["range"])}]
        ).to_csv(output / "segment.csv", index=False)
        for csv_path in output.glob("*.csv"):
            recorder.log_artifact(str(csv_path), artifact_path=VALID_ANALYSIS_DIR)


def save_readable_test_artifacts(recorder):
    """为 Qlib 默认的 test 预测和 sig_analysis 追加可读 CSV。"""

    pred = recorder.load_object("pred.pkl")
    label = recorder.load_object("label.pkl")
    ic = recorder.load_object("sig_analysis/ic.pkl")
    ric = recorder.load_object("sig_analysis/ric.pkl")
    test_metrics, _ = metrics_from_ic(ic, ric)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if isinstance(label, pd.Series):
        label = label.to_frame("label")

    with tempfile.TemporaryDirectory(prefix="qlib-test-csv-") as temp_dir:
        output = Path(temp_dir)
        pred.reset_index().to_csv(output / "pred.csv", index=False)
        label.reset_index().to_csv(output / "label.csv", index=False)
        recorder.log_artifact(str(output / "pred.csv"))
        recorder.log_artifact(str(output / "label.csv"))

        pd.concat(
            [ic.rename("IC"), ric.rename("Rank IC")], axis=1
        ).rename_axis("datetime").reset_index().to_csv(output / "daily_ic.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv(output / "metrics.csv", index=False)
        recorder.log_artifact(str(output / "daily_ic.csv"), artifact_path="sig_analysis")
        recorder.log_artifact(str(output / "metrics.csv"), artifact_path="sig_analysis")


def metrics_from_ic(ic: pd.Series, ric: pd.Series) -> Tuple[Dict[str, float], list[float]]:
    """从逐日 IC/Rank IC 计算汇总指标，并统一处理零方差。"""

    ic_mean = float(ic.mean())
    ric_mean = float(ric.mean())
    ic_std = float(ic.std())
    ric_std = float(ric.std())
    icir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
    rank_icir = ric_mean / ric_std if np.isfinite(ric_std) and ric_std > 0 else np.nan
    values = [ic_mean, icir, ric_mean, rank_icir]
    metrics = {
        "IC": ic_mean,
        "ICIR": icir,
        "Rank IC": ric_mean,
        "Rank ICIR": rank_icir,
        "IC Positive Ratio": float((ic > 0).mean()),
        "Rank IC Positive Ratio": float((ric > 0).mean()),
        "Date Count": int(pd.concat([ic, ric], axis=1).dropna().shape[0]),
    }
    return metrics, values


def load_validation_analysis(recorder):
    """读取已缓存的 validation 指标。"""

    metrics = recorder.load_object(f"{VALID_ANALYSIS_DIR}/metrics.pkl")
    ic = recorder.load_object(f"{VALID_ANALYSIS_DIR}/ic.pkl")
    ric = recorder.load_object(f"{VALID_ANALYSIS_DIR}/ric.pkl")
    _, values = metrics_from_ic(ic, ric)
    return metrics, values


def ensure_validation_analysis(recorder, force: bool = False, dataset=None):
    """确保 recorder 存在 validation 预测和评价产物。

    新任务优先使用独立的 ``selection_valid`` segment；旧任务回退到 ``valid``。
    test 预测和 test 指标不会参与计算。
    """

    if not force:
        try:
            return load_validation_analysis(recorder)
        except Exception:
            pass

    task = recorder.load_object("task")
    model = restore_model_runtime_state(recorder.load_object("params.pkl"))
    dataset_config = copy.deepcopy(task["dataset"])
    segments = dataset_config["kwargs"].get("segments", {})
    segment = "selection_valid" if "selection_valid" in segments else "valid"
    if segment not in segments:
        raise ValueError(f"Recorder {recorder.id} 的 task 不包含可用于选模的 validation segment")

    logger.info(f"为 recorder {recorder.id} 使用 {segment} 计算选模指标: {segments[segment]}")
    if dataset is None:
        dataset = init_instance_by_config(dataset_config)
    pred = model.predict(dataset, segment=segment)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    # TRAModel returns routing diagnostics together with ``score`` and embeds
    # the matching label in its prediction frame.  Its MTSDatasetH.prepare()
    # deliberately returns an iterable segment object, not a DataFrame, so the
    # ordinary DatasetH label-fetching path below is not applicable.
    if "label" in pred.columns:
        label = pred[["label"]].copy()
        pred = pred[["score"]].copy()
    else:
        label = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R)
        if isinstance(label, pd.Series):
            label = label.to_frame("label")
    if pred.empty or label.empty:
        raise ValueError(f"Recorder {recorder.id} 的 validation 预测或标签为空")

    common_index = pred.index.intersection(label.index)
    pred = pred.loc[common_index].sort_index()
    label = label.loc[common_index].sort_index()
    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0], dropna=True)
    metrics, values = metrics_from_ic(ic, ric)
    if not all(np.isfinite(v) for v in values):
        raise ValueError(f"Recorder {recorder.id} validation 指标包含非有限值: {metrics}")

    segment_info = {"name": segment, "range": segments[segment]}
    recorder.save_objects(
        artifact_path=VALID_ANALYSIS_DIR,
        **{
            "pred.pkl": pred,
            "label.pkl": label,
            "ic.pkl": ic,
            "ric.pkl": ric,
            "metrics.pkl": metrics,
            "segment.pkl": segment_info,
        },
    )
    save_readable_validation_artifacts(
        recorder, pred, label, ic, ric, metrics, segment_info
    )
    logger.info(f"Recorder {recorder.id} validation 指标已缓存: {metrics}")
    return metrics, values
