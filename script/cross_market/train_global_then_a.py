#!/usr/bin/env python3
"""Pretrain XGBoost on US+HK+A daily factors, then continue training on A main board."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FACTORS, MODELS, RAW, REPORTS, atomic_json, ensure_dirs, factor_columns


MARKETS = ("A", "US", "HK")
MARKET_FEATURES = [f"MARKET_{market}" for market in MARKETS]


def available_files(market: str) -> list[Path]:
    return sorted((FACTORS / market.lower()).glob("*.parquet"))


def liquidity_table(market: str, files: list[Path]) -> pd.DataFrame:
    cache = REPORTS / f"{market.lower()}_liquidity.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        cached_files = set(cached["file"])
        if cached_files.issuperset({p.name for p in files}):
            return cached
    rows = []
    raw_dir = RAW / market.lower()
    for idx, path in enumerate(files, 1):
        raw_path = raw_dir / path.name
        try:
            frame = pd.read_parquet(raw_path, columns=["symbol", "close", "volume"]).tail(252)
            turnover = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(
                frame["volume"], errors="coerce"
            )
            rows.append(
                {
                    "file": path.name,
                    "symbol": str(frame["symbol"].dropna().iloc[-1]),
                    "median_turnover_252": float(turnover.median()),
                    "rows": len(pd.read_parquet(path, columns=["date"])),
                }
            )
        except Exception:
            rows.append(
                {
                    "file": path.name,
                    "symbol": path.stem,
                    "median_turnover_252": np.nan,
                    "rows": 0,
                }
            )
        if idx % 250 == 0:
            print(f"{market} liquidity scan {idx}/{len(files)}", flush=True)
    result = pd.DataFrame(rows).sort_values("median_turnover_252", ascending=False)
    result.to_csv(cache, index=False)
    return result


def select_files(market: str, max_symbols: int) -> list[Path]:
    files = available_files(market)
    if max_symbols <= 0 or len(files) <= max_symbols:
        return files
    ranking = liquidity_table(market, files).head(max_symbols)
    wanted = set(ranking["file"])
    return [p for p in files if p.name in wanted]


def load_period(
    market: str,
    start: str,
    end: str,
    max_symbols: int,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    files = select_files(market, max_symbols)
    pieces = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for idx, path in enumerate(files, 1):
        try:
            part = pd.read_parquet(
                path,
                filters=[("date", ">=", start_ts), ("date", "<=", end_ts)],
            )
        except Exception:
            part = pd.read_parquet(path)
            part = part[part["date"].between(start_ts, end_ts)]
        if not part.empty:
            pieces.append(part)
        if idx % 250 == 0:
            print(f"{market} load {idx}/{len(files)} pieces={len(pieces)}", flush=True)
    if not pieces:
        raise RuntimeError(f"no {market} data between {start} and {end}")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.dropna(subset=["label_abs"])
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame["date"] = pd.to_datetime(frame["date"])
    print(
        f"loaded {market}: symbols={frame['symbol'].nunique()} rows={len(frame):,} "
        f"dates={frame['date'].min().date()}..{frame['date'].max().date()}",
        flush=True,
    )
    return frame


def add_learning_target(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby(["market", "date"])["label_abs"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    frame["label_z"] = (frame["label_abs"] - mean) / std
    return frame.dropna(subset=["label_z"])


def matrix(frame: pd.DataFrame, features: list[str], label: bool = True) -> xgb.DMatrix:
    data = frame.reindex(columns=features).astype(np.float32)
    data = data.clip(-20, 20)
    kwargs = {"feature_names": features, "missing": np.nan}
    if label:
        kwargs["label"] = frame["label_z"].astype(np.float32)
    return xgb.DMatrix(data, **kwargs)


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for market in MARKETS:
        frame[f"MARKET_{market}"] = (frame["market"] == market).astype(np.float32)
    return frame


def daily_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict:
    work = frame[["date", "symbol", "label_abs"]].copy()
    work["score"] = score
    by_date = work.groupby("date", sort=True)
    pearson = by_date.apply(
        lambda x: x["score"].corr(x["label_abs"]) if len(x) >= 10 else np.nan
    ).dropna()
    rank = by_date.apply(
        lambda x: x["score"].corr(x["label_abs"], method="spearman")
        if len(x) >= 10
        else np.nan
    ).dropna()

    def summarize(series: pd.Series, prefix: str) -> dict:
        std = series.std()
        return {
            prefix: float(series.mean()) if len(series) else np.nan,
            f"{prefix}IR": float(series.mean() / std) if len(series) and std else np.nan,
            f"{prefix}_days": int(len(series)),
        }

    result = {}
    result.update(summarize(pearson, "IC"))
    result.update(summarize(rank, "RankIC"))
    for topk in (1, 3, 5, 10):
        selected = (
            work.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=True)
            .head(topk)
        )
        daily = selected.groupby("date")["label_abs"].mean().sort_index()
        gross_equity = (1 + daily).cumprod()
        # User's current commission is 0.0235% each side; stamp duty is excluded
        # here at the user's request. Minimum CNY 5 depends on position size.
        net_daily = daily - 0.00047
        net_equity = (1 + net_daily).cumprod()
        running_max = net_equity.cummax()
        result.update(
            {
                f"Top{topk}_gross_cumulative": float(gross_equity.iloc[-1] - 1),
                f"Top{topk}_net_cumulative": float(net_equity.iloc[-1] - 1),
                f"Top{topk}_day_win_rate": float((daily > 0).mean()),
                f"Top{topk}_stock_win_rate": float((selected["label_abs"] > 0).mean()),
                f"Top{topk}_net_max_drawdown": float(
                    (net_equity / running_max - 1).min()
                ),
                f"Top{topk}_days": int(len(daily)),
            }
        )
    return result


def topk_equity(frame: pd.DataFrame, score: np.ndarray, topk: int = 10) -> pd.Series:
    work = frame[["date", "label_abs"]].copy()
    work["score"] = score
    daily = (
        work.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(topk)
        .groupby("date")["label_abs"]
        .mean()
        .sort_index()
    )
    return (1 + daily - 0.00047).cumprod()


def save_plots(
    run_dir: Path,
    selection_metrics: dict,
    test_metrics: dict,
    test: pd.DataFrame,
    test_scores: dict[str, np.ndarray],
) -> None:
    labels = list(selection_metrics)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, [selection_metrics[k]["RankIC"] for k in labels])
    axes[0].set_title("Selection Rank IC")
    test_labels = [k for k in labels if k in test_metrics]
    test_x = np.arange(len(test_labels))
    axes[1].bar(test_x, [test_metrics[k]["RankIC"] for k in test_labels])
    axes[1].set_title("Test Rank IC")
    axes[1].set_xticks(test_x, test_labels, rotation=35, ha="right")
    axes[2].bar(
        test_x,
        [test_metrics[k]["Top10_net_cumulative"] for k in test_labels],
    )
    axes[2].set_title("Test Top10 net cumulative")
    axes[2].set_xticks(test_x, test_labels, rotation=35, ha="right")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, score in test_scores.items():
        topk_equity(test, score).plot(ax=ax, label=name)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_title("Test Top10 equity, commission 0.0235% each side")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "test_top10_equity.png", dpi=180)
    plt.close(fig)


def predict(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    best = getattr(booster, "best_iteration", None)
    if best is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, best + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=f"global_to_a_{datetime.now():%y%m%d_%H%M%S}")
    parser.add_argument("--pretrain-start", default="2004-01-01")
    parser.add_argument("--fine-start", default="2014-07-31")
    parser.add_argument("--train-end", default="2024-07-26")
    parser.add_argument("--valid-start", default="2024-07-31")
    parser.add_argument("--valid-end", default="2025-01-28")
    parser.add_argument("--select-start", default="2025-01-31")
    parser.add_argument("--select-end", default="2025-07-28")
    parser.add_argument("--test-start", default="2025-07-31")
    parser.add_argument("--test-end", default="2026-07-28")
    parser.add_argument("--max-symbols-a", type=int, default=0)
    parser.add_argument("--max-symbols-us", type=int, default=500)
    parser.add_argument("--max-symbols-hk", type=int, default=500)
    parser.add_argument("--pretrain-rows-per-market", type=int, default=1_500_000)
    parser.add_argument("--fine-rows", type=int, default=3_000_000)
    parser.add_argument("--pretrain-rounds", type=int, default=400)
    parser.add_argument("--fine-rounds", type=int, default=160)
    parser.add_argument("--baseline-rounds", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    ensure_dirs()
    started = time.monotonic()
    run_dir = REPORTS / args.run_tag
    model_dir = MODELS / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", vars(args))

    pretrain_parts = [
        load_period(
            market,
            args.pretrain_start,
            args.train_end,
            getattr(args, f"max_symbols_{market.lower()}"),
            args.pretrain_rows_per_market,
            args.seed + idx,
        )
        for idx, market in enumerate(MARKETS)
    ]
    pretrain = add_market_features(add_learning_target(pd.concat(pretrain_parts, ignore_index=True)))
    pretrain_rows = len(pretrain)
    pretrain_symbols = pretrain.groupby("market")["symbol"].nunique().to_dict()
    del pretrain_parts
    gc.collect()
    valid_parts = [
        load_period(
            market,
            args.valid_start,
            args.valid_end,
            getattr(args, f"max_symbols_{market.lower()}"),
            0,
            args.seed,
        )
        for market in MARKETS
    ]
    valid_global = add_market_features(
        add_learning_target(pd.concat(valid_parts, ignore_index=True))
    )
    valid_global_rows = len(valid_global)
    fine = load_period(
        "A", args.fine_start, args.train_end, args.max_symbols_a, args.fine_rows, args.seed
    )
    fine = add_market_features(add_learning_target(fine))
    fine_symbols = int(fine["symbol"].nunique())
    valid_a = add_market_features(add_learning_target(valid_parts[0]))
    del valid_parts
    gc.collect()
    selection = add_market_features(
        add_learning_target(
            load_period("A", args.select_start, args.select_end, args.max_symbols_a, 0, args.seed)
        )
    )
    test = add_market_features(
        add_learning_target(
            load_period("A", args.test_start, args.test_end, args.max_symbols_a, 0, args.seed)
        )
    )

    factors = factor_columns(pretrain)
    features = factors + MARKET_FEATURES
    (run_dir / "feature_names.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "max_bin": 256,
        "nthread": args.threads,
        "seed": args.seed,
    }
    print("training global pretrain model", flush=True)
    dpre = matrix(pretrain, features)
    dvalid_global = matrix(valid_global, features)
    global_model = xgb.train(
        params,
        dpre,
        num_boost_round=args.pretrain_rounds,
        evals=[(dvalid_global, "global_valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=20,
    )
    global_model.save_model(model_dir / "global_pretrain.ubj")
    del dpre, dvalid_global, pretrain, valid_global
    gc.collect()

    print("continuing on A main board", flush=True)
    dfine = matrix(fine, features)
    dvalid_a = matrix(valid_a, features)
    fine_model = xgb.train(
        params,
        dfine,
        num_boost_round=args.fine_rounds,
        evals=[(dvalid_a, "a_valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        xgb_model=global_model,
        verbose_eval=20,
    )
    fine_model.save_model(model_dir / "global_then_a_finetuned.ubj")

    print("training A-only control model", flush=True)
    baseline_model = xgb.train(
        params,
        dfine,
        num_boost_round=args.baseline_rounds,
        evals=[(dvalid_a, "a_valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=20,
    )
    baseline_model.save_model(model_dir / "a_only_control.ubj")
    fine_rows = len(fine)
    valid_a_rows = len(valid_a)
    del dfine, dvalid_a, fine, valid_a
    gc.collect()

    dselect = matrix(selection, features, label=False)
    dtest = matrix(test, features, label=False)
    selection_scores = {
        "global": predict(global_model, dselect),
        "finetuned": predict(fine_model, dselect),
        "a_only": predict(baseline_model, dselect),
    }
    selection_metrics = {
        name: daily_metrics(selection, score) for name, score in selection_scores.items()
    }
    blends = {}
    for weight in (0.25, 0.50, 0.75):
        name = f"blend_finetuned_{weight:.2f}"
        score = (
            weight * selection_scores["finetuned"]
            + (1 - weight) * selection_scores["global"]
        )
        blends[name] = score
        selection_metrics[name] = daily_metrics(selection, score)
    chosen = max(
        selection_metrics,
        key=lambda name: (
            -math.inf
            if not np.isfinite(selection_metrics[name]["RankIC"])
            else selection_metrics[name]["RankIC"]
        ),
    )
    print(f"selection chose {chosen}", flush=True)

    test_base = {
        "global": predict(global_model, dtest),
        "finetuned": predict(fine_model, dtest),
        "a_only": predict(baseline_model, dtest),
    }
    if chosen.startswith("blend_finetuned_"):
        weight = float(chosen.rsplit("_", 1)[-1])
        test_score = weight * test_base["finetuned"] + (1 - weight) * test_base["global"]
    else:
        test_score = test_base[chosen]
    test_metrics = {
        name: daily_metrics(test, score) for name, score in test_base.items()
    }
    test_metrics[chosen] = daily_metrics(test, test_score)
    plotted_scores = dict(test_base)
    plotted_scores[chosen] = test_score
    save_plots(run_dir, selection_metrics, test_metrics, test, plotted_scores)

    predictions = test[["date", "symbol", "label_abs"]].copy()
    predictions["score"] = test_score
    predictions["rank_pct"] = predictions.groupby("date")["score"].rank(
        pct=True, ascending=True
    )
    predictions.sort_values(["date", "score"], ascending=[True, False]).to_csv(
        run_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame(selection_metrics).T.to_csv(run_dir / "selection_metrics.csv")
    pd.DataFrame(test_metrics).T.to_csv(run_dir / "test_metrics.csv")

    importance = pd.Series(
        fine_model.get_score(importance_type="gain"), name="gain"
    ).sort_values(ascending=False)
    importance.to_csv(run_dir / "feature_importance_gain.csv", header=True)
    summary = {
        "run_tag": args.run_tag,
        "chosen_on_selection": chosen,
        "selection_metrics": selection_metrics,
        "test_metrics": test_metrics,
        "rows": {
            "pretrain": pretrain_rows,
            "fine": fine_rows,
            "valid_global": valid_global_rows,
            "valid_a": valid_a_rows,
            "selection": len(selection),
            "test": len(test),
        },
        "symbols": {
            "pretrain": pretrain_symbols,
            "fine_a": fine_symbols,
            "test_a": int(test["symbol"].nunique()),
        },
        "factor_count": len(factors),
        "features_with_market_flags": len(features),
        "model_dir": str(model_dir),
        "report_dir": str(run_dir),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "method_notes": {
            "label": "close(T+2)/close(T+1)-1, CS z-score by market/day for learning",
            "vwap": "OHLC4 proxy used consistently across A/HK/US",
            "selection": "blend/model choice made only on selection-valid; test untouched",
            "cost": "0.0235% buy + 0.0235% sell; minimum CNY 5 and stamp duty excluded",
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
