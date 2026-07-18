"""为 Qlib recorder 生成并缓存 validation 信号评价产物。"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from qlib.contrib.eva.alpha import calc_ic
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


VALID_ANALYSIS_DIR = "valid_sig_analysis"
VALID_REQUIRED_FILES = ("pred.pkl", "label.pkl", "ic.pkl", "ric.pkl", "metrics.pkl")


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

    只使用 task 中的 ``valid`` segment。test 预测和 test 指标不会参与计算。
    """

    if not force:
        try:
            return load_validation_analysis(recorder)
        except Exception:
            pass

    task = recorder.load_object("task")
    model = recorder.load_object("params.pkl")
    dataset_config = copy.deepcopy(task["dataset"])
    segments = dataset_config["kwargs"].get("segments", {})
    if "valid" not in segments:
        raise ValueError(f"Recorder {recorder.id} 的 task 不包含 valid segment")

    logger.info(f"为 recorder {recorder.id} 计算 validation 指标: {segments['valid']}")
    if dataset is None:
        dataset = init_instance_by_config(dataset_config)
    pred = model.predict(dataset, segment="valid")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    label = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R)
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

    recorder.save_objects(
        artifact_path=VALID_ANALYSIS_DIR,
        **{
            "pred.pkl": pred,
            "label.pkl": label,
            "ic.pkl": ic,
            "ric.pkl": ric,
            "metrics.pkl": metrics,
        },
    )
    logger.info(f"Recorder {recorder.id} validation 指标已缓存: {metrics}")
    return metrics, values
