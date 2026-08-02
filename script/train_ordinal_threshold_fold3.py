#!/usr/bin/env python3
"""Train ordered binary GBDT models on the fixed CSI1000 Fold3 split.

Each configuration (algorithm x training-window) owns one binary model per
return threshold.  Their probabilities are calibrated on the first part of
selection_valid, selected on its later part, and evaluated once on Test.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = ROOT / "roll"
if str(ROLL_DIR) not in sys.path:
    sys.path.insert(0, str(ROLL_DIR))

from myconfig import get_dataset_config  # noqa: E402
from ordinal_threshold import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    class_representatives,
    cumulative_to_class_probabilities,
    make_threshold_targets,
    monotonic_violation_rate,
    project_monotonic,
    validate_thresholds,
)


FOLD3 = {
    "train_end": "2025-04-16",
    "valid": ("2025-04-17", "2025-09-16"),
    "selection_valid": ("2025-09-17", "2026-02-16"),
    "test": ("2026-02-17", "2026-07-17"),
}
MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost")


@dataclass(frozen=True)
class Task:
    model: str
    train_months: int
    threshold: float

    @property
    def threshold_tag(self) -> str:
        sign = "p" if self.threshold >= 0 else "m"
        return f"{sign}{abs(self.threshold):.4f}".replace(".", "")

    @property
    def key(self) -> str:
        return f"{self.model.lower()}_train{self.train_months}m_gt_{self.threshold_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="csi1000")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--train-months", nargs="+", type=int, default=[45, 60, 84, 120])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--provider-uri", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    parser.add_argument("--run-tag", default=f"ordinal_csi1000_fold3_{datetime.now():%y%m%d}")
    parser.add_argument("--output-root", default=str(ROOT / ".qlibAssistant/ordinal_runs"))
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--selection-calibration-ratio",
        type=float,
        default=0.60,
        help="selection_valid按日期前段用于校准的比例；后段仅用于选模",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已有预测文件的任务")
    parser.add_argument("--smoke", action="store_true", help="运行一个快速端到端冒烟测试")
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ordinal-fold3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def fixed_train_range(months: int) -> tuple[str, str]:
    valid_start = datetime.strptime(FOLD3["valid"][0], "%Y-%m-%d")
    return (
        (valid_start - relativedelta(months=months)).strftime("%Y-%m-%d"),
        (valid_start - timedelta(days=1)).strftime("%Y-%m-%d"),
    )


def build_dataset(pool: str, max_train_months: int):
    train = fixed_train_range(max_train_months)
    config = get_dataset_config(
        train=train,
        valid=FOLD3["valid"],
        test=FOLD3["test"],
        handler_kwargs={
            "instruments": pool,
            "start_time": train[0],
            "end_time": FOLD3["test"][1],
            "fit_start_time": train[0],
            "fit_end_time": train[1],
            "raw_label": True,
        },
    )
    config["kwargs"]["segments"]["selection_valid"] = FOLD3["selection_valid"]
    return init_instance_by_config(config), config


def prepare_xy(dataset, segment) -> tuple[pd.DataFrame, pd.Series]:
    frame = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    if frame.empty:
        raise RuntimeError(f"empty Qlib segment: {segment}")
    if not isinstance(frame.columns, pd.MultiIndex):
        raise RuntimeError("expected Qlib feature/label MultiIndex columns")
    features = frame["feature"]
    label = frame["label"].iloc[:, 0].rename("raw_return")
    valid = label.notna()
    return features.loc[valid].sort_index(), label.loc[valid].sort_index()


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32, order="C")


def prediction_frame(index: pd.Index, raw_return: pd.Series, probability: np.ndarray, split: str):
    result = pd.DataFrame(
        {
            "datetime": index.get_level_values("datetime"),
            "instrument": index.get_level_values("instrument"),
            "split": split,
            "raw_return": raw_return.reindex(index).to_numpy(dtype=float),
            "raw_probability": np.asarray(probability, dtype=float),
        }
    )
    return result


def xgboost_train(task: Task, arrays: dict[str, Any], args, model_path: Path, logger):
    y_train = make_threshold_targets(arrays["y_train"], [task.threshold]).ravel()
    y_valid = make_threshold_targets(arrays["y_valid"], [task.threshold]).ravel()
    dtrain = xgb.DMatrix(arrays["x_train"], label=y_train, feature_names=arrays["feature_names"])
    dvalid = xgb.DMatrix(arrays["x_valid"], label=y_valid, feature_names=arrays["feature_names"])
    dselection = xgb.DMatrix(arrays["x_selection"], feature_names=arrays["feature_names"])
    dtest = xgb.DMatrix(arrays["x_test"], feature_names=arrays["feature_names"])
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "eta": 0.0421,
        "max_depth": 8,
        "subsample": 0.8789,
        "colsample_bytree": 0.8879,
        "nthread": args.threads,
        "seed": 0,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )
    booster.save_model(model_path)
    iteration_range = (0, int(booster.best_iteration) + 1)
    logger.info("%s best_iteration=%s", task.key, booster.best_iteration)
    return (
        booster.predict(dselection, iteration_range=iteration_range),
        booster.predict(dtest, iteration_range=iteration_range),
    )


def lightgbm_train(task: Task, arrays: dict[str, Any], args, model_path: Path, logger):
    y_train = make_threshold_targets(arrays["y_train"], [task.threshold]).ravel()
    y_valid = make_threshold_targets(arrays["y_valid"], [task.threshold]).ravel()
    train_set = lgb.Dataset(arrays["x_train"], label=y_train, feature_name=arrays["feature_names"])
    valid_set = lgb.Dataset(
        arrays["x_valid"], label=y_valid, feature_name=arrays["feature_names"], reference=train_set
    )
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
        "learning_rate": 0.0421,
        "max_depth": 8,
        "num_leaves": 210,
        "subsample": 0.8789,
        "feature_fraction": 0.8879,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "num_threads": args.threads,
        "verbosity": -1,
        "seed": 0,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(50),
        ],
    )
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    logger.info("%s best_iteration=%s", task.key, booster.best_iteration)
    return (
        booster.predict(arrays["x_selection"], num_iteration=booster.best_iteration),
        booster.predict(arrays["x_test"], num_iteration=booster.best_iteration),
    )


def catboost_train(task: Task, arrays: dict[str, Any], args, model_path: Path, logger):
    y_train = make_threshold_targets(arrays["y_train"], [task.threshold]).ravel()
    y_valid = make_threshold_targets(arrays["y_valid"], [task.threshold]).ravel()
    model = cb.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=0.0421,
        depth=6,
        grow_policy="Lossguide",
        max_leaves=100,
        bootstrap_type="MVS",
        subsample=0.8789,
        iterations=args.num_boost_round,
        thread_count=args.threads,
        random_seed=0,
        allow_writing_files=False,
    )
    model.fit(
        arrays["x_train"],
        y_train,
        eval_set=(arrays["x_valid"], y_valid),
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=50,
    )
    model.save_model(str(model_path))
    logger.info("%s best_iteration=%s", task.key, model.get_best_iteration())
    return (
        model.predict_proba(arrays["x_selection"])[:, 1],
        model.predict_proba(arrays["x_test"])[:, 1],
    )


TRAINERS = {
    "XGBoost": (xgboost_train, ".json"),
    "LightGBM": (lightgbm_train, ".txt"),
    "CatBoost": (catboost_train, ".cbm"),
}


def safe_binary_metrics(y_true, probability) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.uint8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prevalence = float(y.mean())
    result = {
        "n": int(y.size),
        "positive_rate": prevalence,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": np.nan,
        "average_precision": np.nan,
    }
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    baseline_brier = prevalence * (1.0 - prevalence)
    result["brier_skill"] = (
        float(1.0 - result["brier"] / baseline_brier) if baseline_brier > 0 else np.nan
    )
    return result


def split_selection_dates(frame: pd.DataFrame, ratio: float) -> tuple[np.ndarray, np.ndarray, str]:
    dates = np.sort(pd.to_datetime(frame["datetime"]).unique())
    if dates.size < 4:
        raise RuntimeError("selection_valid has too few dates for calibration/model selection")
    cut = min(max(int(np.floor(dates.size * ratio)), 1), dates.size - 1)
    cutoff = pd.Timestamp(dates[cut - 1])
    calibration = pd.to_datetime(frame["datetime"]) <= cutoff
    selection = ~calibration
    return calibration.to_numpy(), selection.to_numpy(), cutoff.strftime("%Y-%m-%d")


def daily_rank_ic(frame: pd.DataFrame, score_column: str) -> tuple[float, float, int]:
    values = frame.groupby("datetime", sort=True).apply(
        lambda group: group[score_column].corr(group["raw_return"], method="spearman"),
        include_groups=False,
    ).dropna()
    mean = float(values.mean()) if not values.empty else np.nan
    std = float(values.std()) if not values.empty else np.nan
    return mean, (mean / std if std and np.isfinite(std) else np.nan), int(values.size)


def topk_metrics(frame: pd.DataFrame, score_column: str, topk: int) -> dict[str, float]:
    selected = (
        frame.sort_values(["datetime", score_column], ascending=[True, False])
        .groupby("datetime", sort=True)
        .head(topk)
    )
    daily = selected.groupby("datetime")["raw_return"].mean().dropna()
    return {
        f"top{topk}_day_count": int(daily.size),
        f"top{topk}_win_rate": float((daily > 0).mean()) if not daily.empty else np.nan,
        f"top{topk}_mean_return": float(daily.mean()) if not daily.empty else np.nan,
        f"top{topk}_gross_cumulative": float((1.0 + daily).prod() - 1.0) if not daily.empty else np.nan,
    }


def configuration_metrics(frame: pd.DataFrame, threshold_rows: list[dict[str, Any]], split: str):
    relevant = [row for row in threshold_rows if row["split"] == split]
    ric, ricir, dates = daily_rank_ic(frame, "ordinal_score")
    result = {
        "split": split,
        "mean_brier": float(np.nanmean([row["brier"] for row in relevant])),
        "mean_brier_skill": float(np.nanmean([row["brier_skill"] for row in relevant])),
        "mean_auc": float(np.nanmean([row["auc"] for row in relevant])),
        "mean_average_precision": float(
            np.nanmean([row["average_precision"] for row in relevant])
        ),
        "daily_rank_ic": ric,
        "daily_rank_icir": ricir,
        "rank_ic_dates": dates,
    }
    result.update(topk_metrics(frame, "ordinal_score", 1))
    result.update(topk_metrics(frame, "ordinal_score", 3))
    return result


def calibrate_configuration(
    model: str,
    train_months: int,
    thresholds: np.ndarray,
    run_dir: Path,
    ratio: float,
    logger,
):
    merged = None
    raw_probabilities = []
    calibration_masks = None
    selection_masks = None
    cutoff = None
    calibrators = {}
    for threshold in thresholds:
        task = Task(model, train_months, float(threshold))
        source = pd.read_parquet(run_dir / "raw_predictions" / f"{task.key}.parquet")
        source["datetime"] = pd.to_datetime(source["datetime"])
        source = source.sort_values(["split", "datetime", "instrument"]).reset_index(drop=True)
        identity = source[["datetime", "instrument", "split", "raw_return"]]
        if merged is None:
            merged = identity.copy()
        elif not identity.equals(merged[["datetime", "instrument", "split", "raw_return"]]):
            raise RuntimeError(f"prediction index mismatch for {task.key}")
        if calibration_masks is None:
            selection_frame = source[source["split"].eq("selection_valid")].reset_index(drop=True)
            calibration_masks, selection_masks, cutoff = split_selection_dates(selection_frame, ratio)
        selection_frame = source[source["split"].eq("selection_valid")].reset_index(drop=True)
        calibration_y = (selection_frame.loc[calibration_masks, "raw_return"] > threshold).astype(int)
        calibration_p = selection_frame.loc[calibration_masks, "raw_probability"].to_numpy()
        if calibration_y.nunique() >= 2 and np.unique(calibration_p).size >= 2:
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(calibration_p, calibration_y)
            calibrated = calibrator.predict(source["raw_probability"].to_numpy())
        else:
            calibrator = None
            calibrated = source["raw_probability"].to_numpy()
        calibrators[float(threshold)] = calibrator
        raw_probabilities.append(calibrated)

    matrix_values = np.column_stack(raw_probabilities)
    violation_before = monotonic_violation_rate(matrix_values)
    projected = project_monotonic(matrix_values)
    violation_after = monotonic_violation_rate(projected)
    class_probability = cumulative_to_class_probabilities(projected)
    representatives = class_representatives(thresholds)
    merged["ordinal_score"] = np.sum(
        np.nan_to_num(class_probability, nan=0.0) * representatives[None, :], axis=1
    )
    for index, threshold in enumerate(thresholds):
        tag = Task(model, train_months, float(threshold)).threshold_tag
        merged[f"p_gt_{tag}"] = projected[:, index]
    for index in range(class_probability.shape[1]):
        merged[f"class_{index}_probability"] = class_probability[:, index]
    merged["predicted_class"] = class_probability.argmax(axis=1)
    merged["model"] = model
    merged["train_months"] = train_months

    output_path = run_dir / "configuration_predictions" / f"{model.lower()}_train{train_months}m.parquet"
    merged.to_parquet(output_path, index=False)
    with (run_dir / "calibrators" / f"{model.lower()}_train{train_months}m.pkl").open("wb") as stream:
        pickle.dump(calibrators, stream)

    threshold_rows = []
    selection_indices = merged["split"].eq("selection_valid").to_numpy()
    selection_row_positions = np.flatnonzero(selection_indices)
    selection_eval_positions = selection_row_positions[selection_masks]
    split_positions = {
        "selection_eval": selection_eval_positions,
        "test": np.flatnonzero(merged["split"].eq("test").to_numpy()),
    }
    for split, positions in split_positions.items():
        for index, threshold in enumerate(thresholds):
            y = (merged.iloc[positions]["raw_return"].to_numpy() > threshold).astype(int)
            metrics = safe_binary_metrics(y, projected[positions, index])
            threshold_rows.append(
                {
                    "model": model,
                    "train_months": train_months,
                    "split": split,
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    configuration_rows = []
    for split, positions in split_positions.items():
        row = configuration_metrics(merged.iloc[positions].copy(), threshold_rows, split)
        configuration_rows.append(
            {
                "model": model,
                "train_months": train_months,
                "calibration_cutoff": cutoff,
                "monotonic_violation_before": violation_before,
                "monotonic_violation_after": violation_after,
                **row,
            }
        )
    logger.info(
        "%s train%sm calibrated; violations %.2f%% -> %.2f%%",
        model,
        train_months,
        violation_before * 100,
        violation_after * 100,
    )
    return merged, threshold_rows, configuration_rows


def ensemble_reports(configurations, configuration_table: pd.DataFrame, thresholds, run_dir: Path):
    selection = configuration_table[configuration_table["split"].eq("selection_eval")].copy()
    selection = selection.sort_values(
        ["mean_brier_skill", "daily_rank_ic"], ascending=[False, False]
    ).reset_index(drop=True)
    selection["selection_rank"] = np.arange(1, len(selection) + 1)
    selection.to_csv(run_dir / "selection_ranking.csv", index=False)
    ordered_keys = list(zip(selection["model"], selection["train_months"].astype(int)))
    ensemble_rows = []
    requested_sizes = {1, 2, 4, 6, 8, len(ordered_keys)}
    for topn in sorted(size for size in requested_sizes if size <= len(ordered_keys)):
        keys = ordered_keys[:topn]
        frames = [configurations[key] for key in keys]
        base = frames[0][["datetime", "instrument", "split", "raw_return"]].copy()
        probability_columns = [
            f"p_gt_{Task(keys[0][0], keys[0][1], float(threshold)).threshold_tag}"
            for threshold in thresholds
        ]
        values = np.mean([frame[probability_columns].to_numpy() for frame in frames], axis=0)
        values = project_monotonic(values)
        class_probability = cumulative_to_class_probabilities(values)
        representatives = class_representatives(np.asarray(thresholds, dtype=float))
        base["ordinal_score"] = np.sum(
            np.nan_to_num(class_probability, nan=0.0) * representatives[None, :], axis=1
        )
        for index, column in enumerate(probability_columns):
            base[column] = values[:, index]
        base["ensemble_topn"] = topn
        test = base[base["split"].eq("test")].copy()
        selection_eval_start = pd.Timestamp(
            selection["calibration_cutoff"].iloc[0]
        ) + pd.Timedelta(days=1)
        selection_eval = base[
            base["split"].eq("selection_valid")
            & (pd.to_datetime(base["datetime"]) >= selection_eval_start)
        ].copy()
        threshold_rows = []
        for split, frame in (("selection_eval", selection_eval), ("test", test)):
            for index, threshold in enumerate(thresholds):
                metrics = safe_binary_metrics(
                    (frame["raw_return"].to_numpy() > threshold).astype(int),
                    values[frame.index.to_numpy(), index],
                )
                threshold_rows.append({"split": split, "threshold": threshold, **metrics})
            config_metric = configuration_metrics(frame, threshold_rows, split)
            ensemble_rows.append(
                {
                    "ensemble_topn": topn,
                    "members": ";".join(f"{model}-{months}m" for model, months in keys),
                    **config_metric,
                }
            )
        if topn in (4, len(ordered_keys)):
            base.to_parquet(run_dir / f"ensemble_top{topn}_predictions.parquet", index=False)
    ensemble_table = pd.DataFrame(ensemble_rows)
    ensemble_table.to_csv(run_dir / "ensemble_metrics.csv", index=False)
    return selection, ensemble_table


def write_report(run_dir: Path, manifest: dict[str, Any], config_table, selection, ensemble):
    test_configs = config_table[config_table["split"].eq("test")].sort_values(
        "mean_brier_skill", ascending=False
    )
    test_ensembles = ensemble[ensemble["split"].eq("test")].sort_values("ensemble_topn")
    lines = [
        "# CSI1000 Fold3 有序阈值模型报告",
        "",
        "本报告的Test从未用于早停、概率校准或模型排序。`selection_valid`前60%按日期用于",
        "Isotonic概率校准，后40%用于配置排序。所有收益为标签对应的原始未来一日收益；",
        "Top1/Top3仅是无手续费、无涨跌停执行器、无event_guard的诊断值，不是严格回测。",
        "",
        "## 实验配置",
        "",
        f"- 股票池：{manifest['pool']}",
        f"- Valid：{manifest['fold3']['valid']}",
        f"- Selection-valid：{manifest['fold3']['selection_valid']}",
        f"- Test：{manifest['fold3']['test']}",
        f"- 算法：{', '.join(manifest['models'])}",
        f"- Train窗口（月）：{manifest['train_months']}",
        f"- 阈值：{manifest['thresholds']}",
        f"- 二分类模型总数：{manifest['task_count']}",
        "",
        "## Selection-valid后段配置排名（前12）",
        "",
        selection.head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Test单配置结果",
        "",
        test_configs.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Test集成结果",
        "",
        test_ensembles.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = validate_thresholds(args.thresholds)
    if not 0 < args.selection_calibration_ratio < 1:
        raise ValueError("selection-calibration-ratio must be between 0 and 1")
    if args.smoke:
        args.models = ["XGBoost"]
        args.train_months = [3]
        args.thresholds = [0.0]
        thresholds = validate_thresholds(args.thresholds)
        args.num_boost_round = min(args.num_boost_round, 5)
        args.early_stopping_rounds = min(args.early_stopping_rounds, 2)
        args.run_tag = f"{args.run_tag}_smoke"

    run_dir = Path(args.output_root).expanduser() / args.run_tag
    for folder in ("models", "raw_predictions", "configuration_predictions", "calibrators"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    started = time.time()
    tasks = [
        Task(model, months, float(threshold))
        for months in sorted(set(args.train_months))
        for model in args.models
        for threshold in thresholds
    ]
    manifest = {
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pool": args.pool,
        "models": args.models,
        "train_months": sorted(set(args.train_months)),
        "thresholds": thresholds.tolist(),
        "task_count": len(tasks),
        "fold3": FOLD3,
        "selection_calibration_ratio": args.selection_calibration_ratio,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "threads": args.threads,
        "provider_uri": str(Path(args.provider_uri).expanduser()),
        "status": "running",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("run_dir=%s tasks=%s", run_dir, len(tasks))
    logger.info("loading Qlib Alpha158 data; this is the first long step")
    qlib.init(provider_uri=str(Path(args.provider_uri).expanduser()), region=REG_CN, kernels=1)
    dataset, dataset_config = build_dataset(args.pool, max(args.train_months))
    (run_dir / "dataset_config.json").write_text(
        json.dumps(dataset_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    valid_x, valid_y = prepare_xy(dataset, FOLD3["valid"])
    selection_x, selection_y = prepare_xy(dataset, FOLD3["selection_valid"])
    test_x, test_y = prepare_xy(dataset, FOLD3["test"])
    common_features = list(valid_x.columns)
    logger.info(
        "prepared features=%s valid=%s selection=%s test=%s",
        len(common_features), len(valid_x), len(selection_x), len(test_x),
    )
    x_valid = matrix(valid_x)
    x_selection = matrix(selection_x)
    x_test = matrix(test_x)

    completed = 0
    for train_months in sorted(set(args.train_months)):
        train_range = fixed_train_range(train_months)
        train_x, train_y = prepare_xy(dataset, train_range)
        arrays = {
            "x_train": matrix(train_x),
            "y_train": train_y.to_numpy(dtype=float),
            "x_valid": x_valid,
            "y_valid": valid_y.to_numpy(dtype=float),
            "x_selection": x_selection,
            "x_test": x_test,
            "feature_names": [str(column) for column in common_features],
        }
        logger.info("train%sm range=%s rows=%s", train_months, train_range, len(train_y))
        for model in args.models:
            trainer, suffix = TRAINERS[model]
            for threshold in thresholds:
                task = Task(model, train_months, float(threshold))
                prediction_path = run_dir / "raw_predictions" / f"{task.key}.parquet"
                model_path = run_dir / "models" / f"{task.key}{suffix}"
                completed += 1
                if args.resume and prediction_path.exists() and model_path.exists():
                    logger.info("[%s/%s] resume skip %s", completed, len(tasks), task.key)
                    continue
                task_started = time.time()
                logger.info("[%s/%s] start %s", completed, len(tasks), task.key)
                selection_probability, test_probability = trainer(
                    task, arrays, args, model_path, logger
                )
                output = pd.concat(
                    [
                        prediction_frame(
                            selection_x.index, selection_y, selection_probability, "selection_valid"
                        ),
                        prediction_frame(test_x.index, test_y, test_probability, "test"),
                    ],
                    ignore_index=True,
                )
                output.to_parquet(prediction_path, index=False)
                logger.info(
                    "[%s/%s] done %s elapsed=%.1fs positive_rates(train=%.4f valid=%.4f)",
                    completed,
                    len(tasks),
                    task.key,
                    time.time() - task_started,
                    float((arrays["y_train"] > threshold).mean()),
                    float((arrays["y_valid"] > threshold).mean()),
                )
                del selection_probability, test_probability, output
                gc.collect()
        del train_x, train_y, arrays
        gc.collect()

    logger.info("all binary models completed; calibrating and evaluating configurations")
    configurations = {}
    threshold_rows = []
    configuration_rows = []
    for model in args.models:
        for months in sorted(set(args.train_months)):
            frame, threshold_metrics, config_metrics = calibrate_configuration(
                model,
                months,
                thresholds,
                run_dir,
                args.selection_calibration_ratio,
                logger,
            )
            configurations[(model, months)] = frame
            threshold_rows.extend(threshold_metrics)
            configuration_rows.extend(config_metrics)
    threshold_table = pd.DataFrame(threshold_rows)
    configuration_table = pd.DataFrame(configuration_rows)
    threshold_table.to_csv(run_dir / "threshold_metrics.csv", index=False)
    configuration_table.to_csv(run_dir / "configuration_metrics.csv", index=False)
    selection, ensemble = ensemble_reports(
        configurations, configuration_table, thresholds, run_dir
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(run_dir, manifest, configuration_table, selection, ensemble)
    logger.info("complete elapsed=%.1fs report=%s", time.time() - started, run_dir / "REPORT.md")


if __name__ == "__main__":
    main()
