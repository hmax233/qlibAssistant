#!/usr/bin/env python3
"""Screen new Tushare factors, train a Fixed-style model matrix and select an ensemble.

The workflow has two explicit stages:

1. ``screen`` uses only train/valid data on a deterministic pilot universe to
   select stable new factors.
2. ``train`` fits multiple model families and training windows on the full
   universe.  Ensemble architecture and weights are selected only on
   selection-valid; the frozen test is reported afterwards.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qlibAssistant/matplotlib"))

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

sys.path.insert(0, str(ROOT / "roll"))
from myconfig import get_dataset_config  # noqa: E402

sys.path.insert(0, str(ROOT / "script"))
from run_intraday_factor_experiment import metrics  # noqa: E402


@dataclass(frozen=True)
class Split:
    train_end: str = "2024-08-13"
    valid_start: str = "2024-08-14"
    valid_end: str = "2025-02-13"
    selection_start: str = "2025-02-14"
    selection_end: str = "2025-08-13"
    test_start: str = "2025-08-14"
    test_end: str = "2026-08-11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "train"))
    parser.add_argument("--pool", default="csi1000_mainboard")
    parser.add_argument("--intraday-root", type=Path, required=True)
    parser.add_argument("--daily-source", type=Path, required=True)
    parser.add_argument("--selected-features", type=Path)
    parser.add_argument("--top-new-features", type=int, default=100)
    parser.add_argument("--windows", nargs="+", type=int, default=[24, 36, 60])
    parser.add_argument("--models", nargs="+", choices=("XGBoost", "LightGBM", "CatBoost"),
                        default=["XGBoost", "LightGBM", "CatBoost"])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument(
        "--target-mode", choices=("raw", "drop_unavailable", "zero_unavailable"),
        default="raw",
        help="optionally train on an execution-aware T+1 buyability target",
    )
    parser.add_argument(
        "--feature-mode", choices=("combined", "alpha_only", "new_only"),
        default="combined",
        help="feature ablation while keeping the same factor-covered stock-day sample",
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--provider-uri", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def logger_for(output: Path) -> logging.Logger:
    logger = logging.getLogger("multisource-fixed-v2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(path).names


def load_intraday(root: Path, selected: set[str] | None = None) -> pd.DataFrame:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no intraday factors under {root}")
    files = [path for path in files if {"datetime", "instrument"}.issubset(parquet_columns(path))]
    if not files:
        raise RuntimeError(f"all intraday factor partitions are empty under {root}")
    available = [c for c in parquet_columns(files[0]) if c not in {"datetime", "instrument"}]
    chosen = available if selected is None else [c for c in available if f"min__{c}" in selected]
    frames = [pd.read_parquet(path, columns=["datetime", "instrument", *chosen]) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.drop_duplicates(["datetime", "instrument"], keep="last")
    frame = frame.rename(columns={column: f"min__{column}" for column in chosen})
    return frame.set_index(["datetime", "instrument"]).sort_index().astype("float32")


def load_daily(source: Path, selected: set[str] | None = None) -> pd.DataFrame:
    files = [source] if source.is_file() else sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no daily factors under {source}")
    files = [path for path in files if {"datetime", "instrument"}.issubset(parquet_columns(path))]
    if not files:
        raise RuntimeError(f"all daily factor partitions are empty under {source}")
    available = [
        c for c in parquet_columns(files[0])
        if c not in {"datetime", "instrument"} and not c.startswith("csrank_")
    ]
    chosen = available if selected is None else [c for c in available if f"daily__{c}" in selected]
    frames = [pd.read_parquet(path, columns=["datetime", "instrument", *chosen]) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.drop_duplicates(["datetime", "instrument"], keep="last")
    frame = frame.rename(columns={column: f"daily__{column}" for column in chosen})
    return frame.set_index(["datetime", "instrument"]).sort_index().astype("float32")


def load_new_factors(intraday_root: Path, daily_source: Path,
                     selected: set[str] | None = None) -> pd.DataFrame:
    intraday = load_intraday(intraday_root, selected)
    daily = load_daily(daily_source, selected)
    common = intraday.index.intersection(daily.index)
    result = intraday.loc[common].join(daily.loc[common], how="inner")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def make_dataset(args: argparse.Namespace, start: str, end: str, split: Split):
    config = get_dataset_config(
        train=(start, split.train_end), valid=(split.valid_start, split.valid_end),
        test=(split.test_start, end),
        handler_kwargs={
            "instruments": args.pool,
            "start_time": start,
            "end_time": end,
            "fit_start_time": start,
            "fit_end_time": split.train_end,
            "raw_label": True,
        },
    )
    return init_instance_by_config(config)


def qlib_slice(dataset, interval: tuple[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    frame = dataset.prepare(interval, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    features = frame["feature"].copy()
    features.columns = [f"alpha__{column}" for column in features.columns]
    label = frame["label"].iloc[:, 0].rename("label")
    valid = label.notna()
    return features.loc[valid].astype("float32"), label.loc[valid].astype("float32")


def cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.groupby(level="datetime", sort=False).rank(pct=True)
    ranked.columns = [f"newrank__{column}" for column in frame.columns]
    return ranked.astype("float32")


def cs_target(label: pd.Series) -> np.ndarray:
    grouped = label.groupby(level="datetime")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((label - mean) / std).fillna(0.0).to_numpy(dtype="float32")


def load_tplus1_unavailable(index: pd.MultiIndex) -> pd.Series:
    """Whether a mainboard signal-T sample is unavailable at the T+1 close."""
    start = str(index.get_level_values("datetime").min().date())
    end = str(index.get_level_values("datetime").max().date())
    instruments = sorted(index.get_level_values("instrument").unique())
    data = D.features(
        instruments,
        ["Ref($change,-1)", "Ref($close,-1)"],
        start_time=start,
        end_time=end,
        freq="day",
    )
    data.columns = ["next_change", "next_close"]
    if set(data.index.names) == {"datetime", "instrument"}:
        data = data.reorder_levels(["datetime", "instrument"]).sort_index()
    unavailable = data["next_close"].isna() | data["next_change"].ge(0.095)
    return unavailable.reindex(index).fillna(True).astype(bool)


def join_split(alpha: pd.DataFrame, label: pd.Series, new_raw: pd.DataFrame,
               feature_mode: str = "combined"):
    # Rank only inside the date-specific Qlib universe.  Ranking over the full
    # historical union would quietly include stocks that were not CSI1000
    # members on that date and would make the feature definition inconsistent.
    common = alpha.index.intersection(new_raw.index)
    new_rank = cs_rank(new_raw.loc[common])
    if feature_mode == "combined":
        features = alpha.loc[common].join(new_rank, how="inner")
    elif feature_mode == "alpha_only":
        features = alpha.loc[common]
    elif feature_mode == "new_only":
        features = new_rank
    else:
        raise ValueError(f"unknown feature_mode={feature_mode}")
    label = label.reindex(features.index)
    valid = label.notna()
    return features.loc[valid].astype("float32"), label.loc[valid].astype("float32")


def screen_features(args: argparse.Namespace, output: Path, logger: logging.Logger, split: Split) -> None:
    logger.info("loading pilot factors")
    new = load_new_factors(args.intraday_root, args.daily_source)
    start = "2019-01-01"
    dataset = make_dataset(args, start, split.valid_end, split)
    alpha_train, label_train = qlib_slice(dataset, (start, split.train_end))
    alpha_valid, label_valid = qlib_slice(dataset, (split.valid_start, split.valid_end))
    x_train, y_train = join_split(alpha_train, label_train, new)
    x_valid, y_valid = join_split(alpha_valid, label_valid, new)
    logger.info("screen rows train=%d valid=%d features=%d", len(x_train), len(x_valid), x_train.shape[1])

    rank_y_train = y_train.groupby(level="datetime").rank(pct=True)
    rank_y_valid = y_valid.groupby(level="datetime").rank(pct=True)
    new_columns = [column for column in x_train if column.startswith("newrank__")]
    train_corr = x_train[new_columns].corrwith(rank_y_train)
    valid_corr = x_valid[new_columns].corrwith(rank_y_valid)

    names = list(x_train.columns)
    dtrain = xgb.DMatrix(x_train, label=cs_target(y_train), feature_names=names)
    dvalid = xgb.DMatrix(x_valid, label=cs_target(y_valid), feature_names=names)
    booster = xgb.train(
        {
            "objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.04,
            "max_depth": 7, "min_child_weight": 20, "subsample": 0.82,
            "colsample_bytree": 0.75, "lambda": 8.0, "alpha": 0.1,
            "nthread": args.threads, "seed": 20260818,
        },
        dtrain, num_boost_round=min(args.rounds, 600), evals=[(dvalid, "valid")],
        early_stopping_rounds=args.early_stopping, verbose_eval=50,
    )
    booster.save_model(output / "screening_xgboost.json")
    gain = pd.Series(booster.get_score(importance_type="gain"), dtype=float)
    diagnostics = pd.DataFrame({
        "feature": new_columns,
        "train_rank_corr": train_corr.reindex(new_columns).to_numpy(),
        "valid_rank_corr": valid_corr.reindex(new_columns).to_numpy(),
        "gain": gain.reindex(new_columns).fillna(0.0).to_numpy(),
    })
    diagnostics["sign_stable"] = (
        diagnostics["train_rank_corr"] * diagnostics["valid_rank_corr"] > 0
    )
    gain_scale = diagnostics["gain"].rank(pct=True)
    corr_scale = (
        diagnostics["train_rank_corr"].abs().rank(pct=True)
        + diagnostics["valid_rank_corr"].abs().rank(pct=True)
    ) / 2.0
    diagnostics["selection_score"] = gain_scale + 0.5 * corr_scale
    diagnostics.loc[~diagnostics["sign_stable"], "selection_score"] *= 0.35
    diagnostics = diagnostics.sort_values("selection_score", ascending=False)
    selected = diagnostics.head(args.top_new_features)["feature"].str.removeprefix("newrank__").tolist()
    diagnostics.to_csv(output / "feature_diagnostics.csv", index=False)
    payload = {
        "selected_features": selected,
        "count": len(selected),
        "pilot_symbols": int(new.index.get_level_values("instrument").nunique()),
        "train": [start, split.train_end],
        "valid": [split.valid_start, split.valid_end],
        "selection_uses_test": False,
        "method": "XGBoost gain plus train/valid cross-sectional rank-correlation stability",
    }
    (output / "selected_new_features.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("selected %d new factors -> %s", len(selected), output / "selected_new_features.json")


def train_one(model_name: str, x_train, y_train, x_valid, y_valid,
              x_selection, x_test, args, model_path: Path):
    names = list(x_train.columns)
    target_train, target_valid = cs_target(y_train), cs_target(y_valid)
    if model_name == "XGBoost":
        dtrain = xgb.DMatrix(x_train, label=target_train, feature_names=names)
        dvalid = xgb.DMatrix(x_valid, label=target_valid, feature_names=names)
        model = xgb.train(
            {
                "objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.035,
                "max_depth": 7, "min_child_weight": 20, "subsample": 0.82,
                "colsample_bytree": 0.75, "lambda": 8.0, "alpha": 0.1,
                "nthread": args.threads, "seed": 20260818,
            },
            dtrain, num_boost_round=args.rounds, evals=[(dvalid, "valid")],
            early_stopping_rounds=args.early_stopping, verbose_eval=50,
        )
        model.save_model(model_path)
        iteration = (0, model.best_iteration + 1)
        return (
            model.predict(xgb.DMatrix(x_selection, feature_names=names), iteration_range=iteration),
            model.predict(xgb.DMatrix(x_test, feature_names=names), iteration_range=iteration),
            int(model.best_iteration),
        )
    if model_name == "LightGBM":
        train_set = lgb.Dataset(x_train, label=target_train, feature_name=names)
        valid_set = lgb.Dataset(x_valid, label=target_valid, reference=train_set, feature_name=names)
        model = lgb.train(
            {
                "objective": "regression", "metric": "l2", "learning_rate": 0.03,
                "num_leaves": 63, "max_depth": 8, "min_data_in_leaf": 150,
                "feature_fraction": 0.75, "bagging_fraction": 0.82, "bagging_freq": 1,
                "lambda_l1": 0.2, "lambda_l2": 8.0, "num_threads": args.threads,
                "verbosity": -1, "seed": 20260818,
            },
            train_set, num_boost_round=args.rounds, valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
        )
        model.save_model(str(model_path), num_iteration=model.best_iteration)
        return (
            model.predict(x_selection, num_iteration=model.best_iteration),
            model.predict(x_test, num_iteration=model.best_iteration),
            int(model.best_iteration),
        )
    model = cb.CatBoostRegressor(
        iterations=args.rounds, depth=7, learning_rate=0.035, loss_function="RMSE",
        l2_leaf_reg=8.0, random_strength=0.5, random_seed=20260818,
        thread_count=args.threads, allow_writing_files=False, verbose=50,
    )
    model.fit(
        x_train, target_train, eval_set=(x_valid, target_valid),
        early_stopping_rounds=args.early_stopping, use_best_model=True,
    )
    model.save_model(model_path)
    return model.predict(x_selection), model.predict(x_test), int(model.get_best_iteration())


def score_frame(index, label, prediction) -> pd.DataFrame:
    return pd.DataFrame({"score": prediction, "label": label.reindex(index).to_numpy()}, index=index).dropna()


def daily_zscore(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level="datetime")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def ensemble_frame(paths: list[Path], weights: list[float]) -> pd.DataFrame:
    scores, labels = [], []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index(["datetime", "instrument"])
        scores.append(daily_zscore(frame["score"]))
        labels.append(frame["label"])
    score_frame_all = pd.concat(scores, axis=1, join="inner")
    label_frame = pd.concat(labels, axis=1, join="inner")
    normalized = np.asarray(weights, dtype=float)
    normalized = normalized / normalized.sum()
    score = score_frame_all.mul(normalized, axis=1).sum(axis=1)
    return pd.DataFrame({"score": score, "label": label_frame.iloc[:, 0]}).dropna()


def select_ensemble(output: Path, candidate_metrics: pd.DataFrame, args, logger):
    selection = candidate_metrics[candidate_metrics["split"].eq("selection")].copy()
    selection = selection.sort_values(["Rank_ICIR", "Rank_IC"], ascending=False)
    positive = selection[selection["Rank_ICIR"].gt(0)].copy()
    if positive.empty:
        positive = selection.head(3)
    definitions = {}
    family_best = positive.groupby("model", sort=False).head(1)
    definitions["family_best_icir"] = family_best
    for size in (4, 6):
        definitions[f"top{size}_icir"] = positive.head(min(size, len(positive)))
    definitions["all_positive_icir"] = positive

    rows, frames = [], {}
    for name, members in definitions.items():
        if members.empty:
            continue
        weights = members["Rank_ICIR"].clip(lower=0.01).tolist()
        for split_name in ("selection", "test"):
            paths = [output / f"{key}_{split_name}.parquet" for key in members["key"]]
            frame = ensemble_frame(paths, weights)
            target = output / f"ensemble_{name}_{split_name}.parquet"
            frame.reset_index().to_parquet(target, index=False)
            row = {"ensemble": name, "split": split_name, "members": len(members),
                   "components": ",".join(members["key"]), **metrics(frame, args.cost_rate)}
            rows.append(row)
            frames[(name, split_name)] = target
    result = pd.DataFrame(rows)
    result.to_csv(output / "ensemble_metrics.csv", index=False)
    selection_result = result[result["split"].eq("selection")].sort_values(
        ["Rank_ICIR", "Rank_IC"], ascending=False
    )
    winner = selection_result.iloc[0]
    selected_name = winner["ensemble"]
    selected_test = frames[(selected_name, "test")]
    selected_selection = frames[(selected_name, "selection")]
    (output / "selected_ensemble.json").write_text(json.dumps({
        "ensemble": selected_name,
        "selection_rank_ic": float(winner["Rank_IC"]),
        "selection_rank_icir": float(winner["Rank_ICIR"]),
        "components": winner["components"].split(","),
        "selection_prediction": str(selected_selection),
        "test_prediction": str(selected_test),
        "selection_rule": "maximum selection-valid Rank ICIR, tie-break Rank IC; test unused",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("selected ensemble=%s selection RankICIR=%.4f", selected_name, winner["Rank_ICIR"])


def plot_model_diagnostics(output: Path, candidate_metrics: pd.DataFrame) -> None:
    """Write a compact model-selection chart without changing selection logic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"XGBoost": "#1f77b4", "LightGBM": "#2ca02c", "CatBoost": "#d62728"}
    for split_name, axis in zip(("selection", "test"), axes, strict=True):
        frame = candidate_metrics[candidate_metrics["split"].eq(split_name)]
        for model_name, group in frame.groupby("model", sort=False):
            axis.scatter(
                group["Rank_IC"], group["Rank_ICIR"], s=75,
                color=colors.get(model_name), label=model_name, alpha=0.85,
            )
            for row in group.itertuples(index=False):
                axis.annotate(
                    f"{row.train_months}m", (row.Rank_IC, row.Rank_ICIR),
                    xytext=(4, 4), textcoords="offset points", fontsize=8,
                )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(f"{split_name}: Rank IC vs Rank ICIR")
        axis.set_xlabel("Rank IC")
        axis.set_ylabel("Rank ICIR")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "candidate_rank_ic_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_matrix(args: argparse.Namespace, output: Path, logger: logging.Logger, split: Split) -> None:
    if not args.selected_features:
        raise SystemExit("train mode requires --selected-features")
    selected = set(json.loads(args.selected_features.read_text(encoding="utf-8"))["selected_features"])
    logger.info("loading %d selected new factors from full universe", len(selected))
    new = load_new_factors(args.intraday_root, args.daily_source, selected)
    earliest = (pd.Timestamp(split.train_end) - pd.DateOffset(months=max(args.windows))).strftime("%Y-%m-%d")
    dataset = make_dataset(args, earliest, split.test_end, split)
    intervals = {
        "train": (earliest, split.train_end),
        "valid": (split.valid_start, split.valid_end),
        "selection": (split.selection_start, split.selection_end),
        "test": (split.test_start, split.test_end),
    }
    prepared = {}
    for name, interval in intervals.items():
        alpha, label = qlib_slice(dataset, interval)
        prepared[name] = join_split(alpha, label, new, args.feature_mode)
        logger.info("prepared %s rows=%d features=%d", name, len(prepared[name][0]), prepared[name][0].shape[1])
    unavailable = {}
    if args.target_mode != "raw":
        for name, (features, _) in prepared.items():
            unavailable[name] = load_tplus1_unavailable(features.index)
            logger.info("%s unavailable rate=%.3f%%", name, unavailable[name].mean() * 100)
    del new, dataset
    gc.collect()

    rows = []
    for model_name in args.models:
        for months in args.windows:
            key = f"{model_name.lower()}_{months}m"
            selection_path = output / f"{key}_selection.parquet"
            test_path = output / f"{key}_test.parquet"
            metrics_path = output / f"{key}_metrics.json"
            if args.resume and selection_path.exists() and test_path.exists() and metrics_path.exists():
                logger.info("resume skip %s", key)
                rows.extend(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            train_start = (pd.Timestamp(split.train_end) - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
            train_x, train_y = prepared["train"]
            dates = train_x.index.get_level_values("datetime")
            mask = dates >= pd.Timestamp(train_start)
            x_train, y_train = train_x.loc[mask], train_y.loc[mask]
            x_valid, y_valid = prepared["valid"]
            x_selection, y_selection = prepared["selection"]
            x_test, y_test = prepared["test"]
            if args.target_mode == "drop_unavailable":
                train_keep = ~unavailable["train"].reindex(x_train.index).fillna(True)
                valid_keep = ~unavailable["valid"].reindex(x_valid.index).fillna(True)
                x_train, y_train = x_train.loc[train_keep], y_train.loc[train_keep]
                x_valid, y_valid = x_valid.loc[valid_keep], y_valid.loc[valid_keep]
            elif args.target_mode == "zero_unavailable":
                train_bad = unavailable["train"].reindex(x_train.index).fillna(True)
                valid_bad = unavailable["valid"].reindex(x_valid.index).fillna(True)
                y_train = y_train.mask(train_bad, 0.0)
                y_valid = y_valid.mask(valid_bad, 0.0)
            suffix = {"XGBoost": ".json", "LightGBM": ".txt", "CatBoost": ".cbm"}[model_name]
            logger.info("training %s train=%s~%s rows=%d", key, train_start, split.train_end, len(x_train))
            started = time.monotonic()
            selection_pred, test_pred, best_iteration = train_one(
                model_name, x_train, y_train, x_valid, y_valid, x_selection, x_test,
                args, output / f"{key}{suffix}",
            )
            model_rows = []
            for split_name, index, label, pred, path in (
                ("selection", x_selection.index, y_selection, selection_pred, selection_path),
                ("test", x_test.index, y_test, test_pred, test_path),
            ):
                frame = score_frame(index, label, pred)
                frame.reset_index().to_parquet(path, index=False)
                row = {
                    "key": key, "model": model_name, "train_months": months,
                    "train_start": train_start, "train_end": split.train_end,
                    "valid_start": split.valid_start, "valid_end": split.valid_end,
                    "selection_start": split.selection_start, "selection_end": split.selection_end,
                    "test_start": split.test_start, "test_end": split.test_end,
                    "split": split_name, "best_iteration": best_iteration,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "samples": len(frame), "features": x_train.shape[1],
                    **metrics(frame, args.cost_rate),
                }
                rows.append(row)
                model_rows.append(row)
                logger.info("%s %s RankIC=%.4f RankICIR=%.4f Top1Net=%.2f%%",
                            key, split_name, row["Rank_IC"], row["Rank_ICIR"],
                            row["Top1_net_cumulative"] * 100)
            metrics_path.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            gc.collect()
    candidate_metrics = pd.DataFrame(rows)
    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    plot_model_diagnostics(output, candidate_metrics)
    select_ensemble(output, candidate_metrics, args, logger)
    (output / "metadata.json").write_text(json.dumps({
        "pool": args.pool, "split": asdict(split), "windows": args.windows,
        "models": args.models, "selected_feature_file": str(args.selected_features),
        "new_feature_count": len(selected), "total_model_feature_count": int(prepared["train"][0].shape[1]),
        "feature_mode": args.feature_mode,
        "target": "daily cross-sectional z-score of T+1-to-T+2 return",
        "target_mode": args.target_mode,
        "timing": "signal T close -> buy T+1 close -> sell T+2 close",
        "ensemble_selection_uses_test": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = ROOT / ".qlibAssistant/experiments" / args.run_tag
    output.mkdir(parents=True, exist_ok=args.resume)
    logger = logger_for(output)
    split = Split()
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    if args.mode == "screen":
        screen_features(args, output, logger, split)
    else:
        train_matrix(args, output, logger, split)
    logger.info("complete output=%s", output)


if __name__ == "__main__":
    main()
