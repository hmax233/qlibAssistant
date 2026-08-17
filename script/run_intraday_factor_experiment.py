#!/usr/bin/env python3
"""Compare Alpha158 with leakage-safe intraday factors on frozen time splits.

The development stage may use only folds A/B.  The final stage evaluates the
configuration selected by A/B once on fold C.  Features dated T predict the
close-to-close return from T+1 to T+2, matching the project's signal timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qlibAssistant/matplotlib"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
import xgboost as xgb
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


ROLL = ROOT / "roll"
if str(ROLL) not in sys.path:
    sys.path.insert(0, str(ROLL))
from myconfig import get_dataset_config  # noqa: E402


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[str, str]
    valid: tuple[str, str]
    selection: tuple[str, str]
    test: tuple[str, str]


FOLDS = {
    "A": Fold(
        "A", ("2019-01-01", "2022-12-31"), ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30"),
    ),
    "B": Fold(
        "B", ("2019-01-01", "2023-12-31"), ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-12-31"),
    ),
    "C": Fold(
        "C", ("2019-01-01", "2024-12-31"), ("2025-01-01", "2025-06-30"),
        ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-07-17"),
    ),
}
FEATURE_SETS = (
    "alpha158", "intraday", "daily", "combined", "alpha_daily",
    "intraday_daily", "all",
)
MODELS = ("XGBoost", "LightGBM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "final"), default="dev")
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS,
                        default=["alpha158", "intraday", "combined"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--final-feature-set", choices=FEATURE_SETS)
    parser.add_argument("--final-model", choices=MODELS)
    parser.add_argument("--final-test-start", help="override fold-C test start")
    parser.add_argument("--final-test-end", help="override fold-C test end")
    parser.add_argument("--pool", default="csi1000_mainboard")
    parser.add_argument("--factor-root", type=Path, default=ROOT / ".qlibAssistant/supplemental/intraday_daily_factors_15m")
    parser.add_argument(
        "--daily-factor-file", type=Path,
        default=ROOT / ".qlibAssistant/supplemental/tushare_daily_factors/all_factors.parquet",
    )
    parser.add_argument("--provider-uri", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument(
        "--target", choices=("raw", "executable"), default="raw",
        help="executable sets the learning target to zero when T+1 close cannot be bought",
    )
    parser.add_argument("--run-tag", default=f"intraday_factor_{time.strftime('%y%m%d_%H%M%S')}")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def logger_for(output: Path) -> logging.Logger:
    logger = logging.getLogger("intraday-factor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_intraday(root: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no intraday factor parquet files under {root}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.drop_duplicates(["datetime", "instrument"], keep="last")
    feature_names = [column for column in frame.columns if column not in {"datetime", "instrument"}]
    frame = frame.set_index(["datetime", "instrument"]).sort_index()
    return frame[feature_names].astype("float32"), feature_names


def load_factor_file(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.drop_duplicates(["datetime", "instrument"], keep="last")
    names = [column for column in frame if column not in {"datetime", "instrument"}]
    return frame.set_index(["datetime", "instrument"])[names].sort_index().astype("float32"), names


def load_buyability(instruments: list[str], start: str, end: str) -> pd.Series:
    """Whether a signal-T candidate can be bought at the T+1 close."""
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
    # This experiment is mainboard-only. 9.5% is deliberately conservative
    # and also catches rounded 10% upper-limit closes.
    return (data["next_close"].notna() & data["next_change"].lt(0.095)).rename("buyable")


def make_dataset(args: argparse.Namespace, end_time: str = "2026-07-17"):
    config = get_dataset_config(
        train=FOLDS["C"].train,
        valid=FOLDS["C"].valid,
        test=FOLDS["C"].test,
        handler_kwargs={
            "instruments": args.pool,
            "start_time": "2019-01-01",
            "end_time": end_time,
            "fit_start_time": "2019-01-01",
            "fit_end_time": "2024-12-31",
            "raw_label": True,
        },
    )
    return init_instance_by_config(config)


def qlib_slice(dataset, interval: tuple[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    frame = dataset.prepare(interval, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    features = frame["feature"].copy()
    features.columns = [f"alpha_{column}" for column in features.columns]
    label = frame["label"].iloc[:, 0].rename("raw_return")
    valid = label.notna()
    return features.loc[valid].astype("float32"), label.loc[valid].astype("float32")


def merge_features(
    alpha: pd.DataFrame,
    label: pd.Series,
    intraday: pd.DataFrame,
    daily: pd.DataFrame | None,
    feature_set: str,
    buyable: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    sources = {
        "alpha158": (alpha,),
        "intraday": (intraday,),
        "daily": (daily,),
        "combined": (alpha, intraday),
        "alpha_daily": (alpha, daily),
        "intraday_daily": (intraday, daily),
        "all": (alpha, intraday, daily),
    }[feature_set]
    if any(source is None for source in sources):
        raise ValueError(f"feature set {feature_set} requires --daily-factor-file")
    # Every ablation must use the exact same sampled stock-days.  The minute
    # store defines the frozen 300-stock research sample even for Alpha-only
    # and daily-only baselines; otherwise sample size/universe would confound
    # the feature comparison.
    common_index = alpha.index.intersection(intraday.index)
    for source in sources:
        common_index = common_index.intersection(source.index)
    selected = sources[0].loc[common_index]
    for source in sources[1:]:
        selected = selected.join(source.loc[common_index], how="inner")
    selected = selected.replace([np.inf, -np.inf], np.nan).sort_index()
    raw_label = label.reindex(selected.index)
    if buyable is None:
        learning_label = raw_label
    else:
        learning_label = raw_label.where(buyable.reindex(selected.index).fillna(False), 0.0)
    return selected, raw_label, learning_label.rename("learning_target")


def cs_zscore(label: pd.Series) -> np.ndarray:
    grouped = label.groupby(level="datetime")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((label - mean) / std).fillna(0.0).to_numpy(dtype="float32")


def train_predict(model_name: str, train, valid, selection, test, args, model_path: Path):
    x_train, _, y_train = train
    x_valid, _, y_valid = valid
    x_selection, _, _ = selection
    x_test, _, _ = test
    names = list(x_train.columns)
    if model_name == "XGBoost":
        dtrain = xgb.DMatrix(x_train, label=cs_zscore(y_train), feature_names=names)
        dvalid = xgb.DMatrix(x_valid, label=cs_zscore(y_valid), feature_names=names)
        booster = xgb.train(
            {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "eta": 0.0421, "max_depth": 8, "min_child_weight": 5,
                "subsample": 0.88, "colsample_bytree": 0.89,
                "lambda": 1.0, "alpha": 0.0, "nthread": args.threads, "seed": 20260817,
            },
            dtrain, num_boost_round=args.rounds, evals=[(dvalid, "valid")],
            early_stopping_rounds=args.early_stopping, verbose_eval=50,
        )
        booster.save_model(model_path)
        iteration = (0, booster.best_iteration + 1)
        return (
            booster.predict(xgb.DMatrix(x_selection, feature_names=names), iteration_range=iteration),
            booster.predict(xgb.DMatrix(x_test, feature_names=names), iteration_range=iteration),
            int(booster.best_iteration),
        )
    train_set = lgb.Dataset(x_train, label=cs_zscore(y_train), feature_name=names)
    valid_set = lgb.Dataset(x_valid, label=cs_zscore(y_valid), feature_name=names, reference=train_set)
    booster = lgb.train(
        {
            "objective": "regression", "metric": "l2", "learning_rate": 0.03,
            "max_depth": 8, "num_leaves": 127, "min_data_in_leaf": 100,
            "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1,
            "lambda_l1": 1.0, "lambda_l2": 5.0, "num_threads": args.threads,
            "verbosity": -1, "seed": 20260817,
        },
        train_set, num_boost_round=args.rounds, valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
    )
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    return (
        booster.predict(x_selection, num_iteration=booster.best_iteration),
        booster.predict(x_test, num_iteration=booster.best_iteration),
        int(booster.best_iteration),
    )


def score_frame(index, label, prediction) -> pd.DataFrame:
    return pd.DataFrame({"score": prediction, "label": label.reindex(index).to_numpy()}, index=index).dropna()


def metrics(frame: pd.DataFrame, cost_rate: float) -> dict[str, float]:
    grouped = frame.groupby(level="datetime", sort=True)
    daily_ic = grouped.apply(lambda x: x["score"].corr(x["label"]), include_groups=False)
    daily_ric = grouped.apply(
        lambda x: x["score"].corr(x["label"], method="spearman"), include_groups=False
    )
    result = {
        "days": int(len(daily_ic)), "IC": float(daily_ic.mean()),
        "ICIR": float(daily_ic.mean() / daily_ic.std()),
        "Rank_IC": float(daily_ric.mean()),
        "Rank_ICIR": float(daily_ric.mean() / daily_ric.std()),
        "Rank_IC_positive_ratio": float((daily_ric > 0).mean()),
    }
    for topk in (1, 3, 5, 10):
        members, gross = [], []
        for _, group in grouped:
            top = group.nlargest(topk, "score")
            members.append(set(top.index.get_level_values("instrument")))
            gross.append(float(top["label"].mean()))
        turnover = [1.0] + [1 - len(members[i] & members[i - 1]) / topk for i in range(1, len(members))]
        gross = pd.Series(gross, dtype=float)
        net = gross - np.asarray(turnover) * cost_rate
        result[f"Top{topk}_win_rate"] = float((gross > 0).mean())
        result[f"Top{topk}_gross_cumulative"] = float((1 + gross).prod() - 1)
        result[f"Top{topk}_net_cumulative"] = float((1 + net).prod() - 1)
        result[f"Top{topk}_average_turnover"] = float(np.mean(turnover))
    return result


def main() -> None:
    args = parse_args()
    if args.stage == "final" and (not args.final_feature_set or not args.final_model):
        raise SystemExit("final stage requires --final-feature-set and --final-model")
    output = ROOT / ".qlibAssistant/experiments" / args.run_tag
    output.mkdir(parents=True, exist_ok=False)
    logger = logger_for(output)
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    logger.info("loading intraday factors")
    intraday, intraday_names = load_intraday(args.factor_root)
    needs_daily = any(name in {"daily", "alpha_daily", "intraday_daily", "all"}
                      for name in (args.feature_sets if args.stage == "dev" else [args.final_feature_set]))
    daily, daily_names = load_factor_file(args.daily_factor_file) if needs_daily else (None, [])
    buyable = (
        load_buyability(
            sorted(intraday.index.get_level_values("instrument").unique()),
            "2019-01-01", "2026-07-17",
        )
        if args.target == "executable"
        else None
    )
    final_fold = FOLDS["C"]
    if args.stage == "final" and (args.final_test_start or args.final_test_end):
        final_fold = Fold(
            final_fold.name, final_fold.train, final_fold.valid, final_fold.selection,
            (args.final_test_start or final_fold.test[0], args.final_test_end or final_fold.test[1]),
        )
    dataset = make_dataset(args, end_time=final_fold.test[1])
    folds = [FOLDS["A"], FOLDS["B"]] if args.stage == "dev" else [final_fold]
    feature_sets = args.feature_sets if args.stage == "dev" else [args.final_feature_set]
    model_names = args.models if args.stage == "dev" else [args.final_model]
    if args.smoke:
        folds, feature_sets, model_names = folds[:1], feature_sets[:1], model_names[:1]
        args.rounds, args.early_stopping = 10, 3
    rows = []
    for fold in folds:
        logger.info("preparing fold %s", fold.name)
        raw = {}
        for name, interval in (("train", fold.train), ("valid", fold.valid),
                               ("selection", fold.selection), ("test", fold.test)):
            raw[name] = qlib_slice(dataset, interval)
        for feature_set in feature_sets:
            prepared = {
                name: merge_features(features, label, intraday, daily, feature_set, buyable)
                for name, (features, label) in raw.items()
            }
            for model_name in model_names:
                key = f"fold{fold.name}_{model_name.lower()}_{feature_set}"
                logger.info("training %s rows=%s features=%s", key,
                            len(prepared["train"][0]), prepared["train"][0].shape[1])
                suffix = ".json" if model_name == "XGBoost" else ".txt"
                selection_pred, test_pred, best_iteration = train_predict(
                    model_name, prepared["train"], prepared["valid"], prepared["selection"],
                    prepared["test"], args, output / f"{key}{suffix}",
                )
                for split, prediction in (("selection", selection_pred), ("test", test_pred)):
                    features, label, _ = prepared[split]
                    frame = score_frame(features.index, label, prediction)
                    frame.reset_index().to_parquet(output / f"{key}_{split}.parquet", index=False)
                    row = {"fold": fold.name, "model": model_name,
                           "feature_set": feature_set, "split": split,
                           "best_iteration": best_iteration, "samples": len(frame),
                           "features": features.shape[1]}
                    row.update(metrics(frame, args.cost_rate))
                    rows.append(row)
                    logger.info("%s %s RankIC=%.4f RankICIR=%.4f Top1Net=%.2f%%",
                                key, split, row["Rank_IC"], row["Rank_ICIR"],
                                row["Top1_net_cumulative"] * 100)
                gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics.csv", index=False)
    metadata = {
        "stage": args.stage, "pool": args.pool, "factor_root": str(args.factor_root),
        "daily_factor_file": str(args.daily_factor_file) if needs_daily else None,
        "target": args.target,
        "intraday_features": intraday_names, "daily_features": daily_names,
        "folds": [fold.__dict__ for fold in folds],
        "selection_rule": "Choose on folds A/B selection splits; fold C remains unopened until final.",
        "timing": "T features -> buy T+1 close -> sell T+2 close",
        "cost_rate": args.cost_rate,
        "strict_execution": "Not applied here; winning final predictions must pass strict evaluator.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("complete: %s", output)


if __name__ == "__main__":
    main()
