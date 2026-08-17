#!/usr/bin/env python3
"""Run a compact CPU TRA on stable Tushare daily-factor features."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qlibAssistant/matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.constant import REG_CN
from qlib.contrib.data.dataset import MTSDatasetH
from qlib.contrib.model.pytorch_tra import TRAModel
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader

from run_intraday_factor_experiment import FOLDS, metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-file", type=Path, default=ROOT / ".qlibAssistant/supplemental/tushare_daily_factors/all_factors.parquet")
    parser.add_argument("--importance-file", type=Path, default=ROOT / ".qlibAssistant/experiments/multisource_raw_dev_fair_260818_0034/xgboost_daily_feature_importance.csv")
    parser.add_argument("--folds", nargs="+", choices=["A", "B", "C"], default=["A", "B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--run-tag", default=f"cpu_tra_daily_{time.strftime('%y%m%d_%H%M%S')}")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def selected_features(path: Path, count: int) -> list[str]:
    importance = pd.read_csv(path)
    if "feature" not in importance:
        importance = importance.rename(columns={importance.columns[0]: "feature"})
    importance = importance.sort_values(["folds_used", "mean_rank"], ascending=[False, True])
    return importance["feature"].head(count).tolist()


def load_frame(path: Path, features: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["datetime", "instrument", *features])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.set_index(["datetime", "instrument"]).sort_index()
    instruments = sorted(frame.index.get_level_values("instrument").unique())
    label = D.features(
        instruments, ["Ref($close,-2)/Ref($close,-1)-1"],
        start_time=str(frame.index.get_level_values("datetime").min().date()),
        end_time=str(frame.index.get_level_values("datetime").max().date()), freq="day",
    ).iloc[:, 0].rename("label")
    if set(label.index.names) == {"datetime", "instrument"}:
        label = label.reorder_levels(["datetime", "instrument"]).sort_index()
    return frame.join(label, how="inner")


def normalize_for_fold(frame: pd.DataFrame, features: list[str], train: tuple[str, str]) -> pd.DataFrame:
    train_mask = frame.index.get_level_values("datetime").to_series(index=frame.index).between(*map(pd.Timestamp, train))
    train_values = frame.loc[train_mask, features]
    median = train_values.median()
    mad = (train_values - median).abs().median().replace(0.0, np.nan)
    normalized = ((frame[features] - median) / (mad * 1.4826)).clip(-10, 10).fillna(0.0).astype("float32")
    rank_label = frame["label"].groupby(level="datetime").rank(pct=True).sub(0.5).mul(3.46).astype("float32")
    result = pd.concat({"feature": normalized, "label": rank_label.to_frame("LABEL0")}, axis=1)
    return result.sort_index()


def make_dataset(data: pd.DataFrame, fold, args) -> MTSDatasetH:
    handler = DataHandlerLP(
        instruments=None, start_time=fold.train[0], end_time=fold.test[1],
        data_loader=StaticDataLoader(data), infer_processors=[], learn_processors=[],
    )
    return MTSDatasetH(
        handler=handler,
        segments={"train": fold.train, "valid": fold.valid,
                  "selection": fold.selection, "test": fold.test},
        seq_len=args.seq_len, horizon=1, num_states=3, memory_mode="sample",
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        input_size=len(data["feature"].columns),
    )


def make_model(input_size: int, seed: int, args) -> TRAModel:
    return TRAModel(
        model_config={"input_size": input_size, "hidden_size": args.hidden_size,
                      "num_layers": 1, "rnn_arch": "LSTM", "use_attn": True,
                      "dropout": 0.0},
        tra_config={"num_states": 3, "rnn_arch": "LSTM", "hidden_size": 16,
                    "num_layers": 1, "dropout": 0.0, "tau": 1.0,
                    "src_info": "LR_TPE"},
        model_type="RNN", lr=0.001, n_epochs=args.epochs, early_stop=args.early_stop,
        max_steps_per_epoch=args.max_steps, lamb=1.0, rho=0.99, alpha=0.5,
        seed=seed, logdir=None, eval_train=False, eval_test=False, pretrain=True,
        transport_method="router", memory_mode="sample",
    )


def prediction_frame(pred, raw_label: pd.Series) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        score = pred.rename("score")
    elif "score" in pred:
        score = pred["score"].rename("score")
    else:
        score = pred.iloc[:, 0].rename("score")
    return pd.concat([score, raw_label.rename("label")], axis=1).dropna()


def main():
    args = parse_args()
    if args.smoke:
        args.folds, args.seeds, args.top_features = args.folds[:1], args.seeds[:1], min(10, args.top_features)
        args.epochs, args.early_stop, args.max_steps = 2, 1, 5
    output = ROOT / ".qlibAssistant/experiments" / args.run_tag
    output.mkdir(parents=True, exist_ok=False)
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN)
    features = selected_features(args.importance_file, args.top_features)
    raw = load_frame(args.factor_file, features)
    rows = []
    started = time.monotonic()
    for fold_name in args.folds:
        fold = FOLDS[fold_name]
        data = normalize_for_fold(raw, features, fold.train)
        dataset = make_dataset(data, fold, args)
        for seed in args.seeds:
            run_start = time.monotonic()
            model = make_model(len(features), seed, args)
            evaluations = {}
            model.fit(dataset, evals_result=evaluations)
            torch.save({"model": model.model.state_dict(), "tra": model.tra.state_dict()},
                       output / f"fold{fold_name}_seed{seed}.pt")
            for split, interval in (("selection", fold.selection), ("test", fold.test)):
                pred = model.predict(dataset, segment=split)
                dates = raw.index.get_level_values("datetime")
                label = raw.loc[dates.to_series(index=raw.index).between(*map(pd.Timestamp, interval)), "label"]
                frame = prediction_frame(pred, label)
                frame.reset_index().to_parquet(output / f"fold{fold_name}_seed{seed}_{split}.parquet", index=False)
                row = {"fold": fold_name, "seed": seed, "split": split,
                       "features": len(features), "samples": len(frame),
                       "elapsed_seconds": round(time.monotonic() - run_start, 2)}
                row.update(metrics(frame, 0.0015))
                rows.append(row)
                print(f"TRA fold={fold_name} seed={seed} {split} RankIC={row['Rank_IC']:.4f} RankICIR={row['Rank_ICIR']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "device": "cpu", "features": features, "folds": args.folds, "seeds": args.seeds,
        "seq_len": args.seq_len, "epochs": args.epochs, "early_stop": args.early_stop,
        "max_steps_per_epoch": args.max_steps, "elapsed_seconds": round(time.monotonic()-started, 2),
        "note": "Compact CPU TRA; labels use official CSRankNorm formula.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
