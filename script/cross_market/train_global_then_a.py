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
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FACTORS, MODELS, RAW, REPORTS, atomic_json, ensure_dirs, factor_columns


MARKETS = ("A", "US", "HK")
MARKET_FEATURES = [f"MARKET_{market}" for market in MARKETS]
LIMIT_ROOT = Path("/Users/hmax/investment_data/supplemental/stk_limit")


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
    min_listed_days: int,
    max_abs_label: float,
    training_filter: bool = False,
) -> pd.DataFrame:
    files = select_files(market, max_symbols)
    pieces = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    read_end_ts = end_ts + pd.Timedelta(days=14)
    for idx, path in enumerate(files, 1):
        try:
            part = pd.read_parquet(
                path,
                filters=[
                    ("date", ">=", start_ts),
                    ("date", "<=", read_end_ts if market == "A" else end_ts),
                ],
            )
        except Exception:
            part = pd.read_parquet(path)
            part = part[
                part["date"].between(
                    start_ts, read_end_ts if market == "A" else end_ts
                )
            ]
        if market == "A" and not part.empty:
            part = part.sort_values("date")
            part["_next_buy_date"] = part["date"].shift(-1)
            part["_next_sell_date"] = part["date"].shift(-2)
            part = part[part["date"].between(start_ts, end_ts)]
        if not part.empty:
            pieces.append(part)
        if idx % 250 == 0:
            print(f"{market} load {idx}/{len(files)} pieces={len(pieces)}", flush=True)
    if not pieces:
        raise RuntimeError(f"no {market} data between {start} and {end}")
    frame = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    raw_rows = len(frame)
    frame = frame.dropna(subset=["label_abs"])
    missing_label_rows = raw_rows - len(frame)
    young_rows = int((frame["listed_days"] < min_listed_days).sum())
    frame = frame[frame["listed_days"] >= min_listed_days]
    outlier_rows = int((frame["label_abs"].abs() > max_abs_label).sum())
    frame = frame[frame["label_abs"].abs() <= max_abs_label]
    # Sampling before the three exact-limit joins avoids a very large transient
    # copy of the full 10M-row A history. The seed makes this deterministic.
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    if market == "A":
        frame = enrich_a_execution(frame, start_ts, read_end_ts)
    st_rows = int(frame["is_st_signal"].fillna(False).sum()) if market == "A" else 0
    if market == "A":
        frame = frame[~frame["is_st_signal"].fillna(False)]
    blocked_train_rows = 0
    if market == "A" and training_filter:
        blocked = frame["buy_blocked_proxy"].fillna(False).astype(bool)
        blocked_train_rows = int(blocked.sum())
        frame = frame[~blocked]
    frame["date"] = pd.to_datetime(frame["date"])
    print(
        f"loaded {market}: symbols={frame['symbol'].nunique()} rows={len(frame):,} "
        f"dates={frame['date'].min().date()}..{frame['date'].max().date()} "
        f"dropped(missing_label={missing_label_rows:,}, listed<{min_listed_days}="
        f"{young_rows:,}, ST={st_rows:,}, abs_label>{max_abs_label:.3f}="
        f"{outlier_rows:,}, "
        f"buy_blocked_train={blocked_train_rows:,})",
        flush=True,
    )
    return frame


def load_limit_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    consolidated = LIMIT_ROOT / "stk_limit_all.parquet"
    if consolidated.exists():
        return pd.read_parquet(
            consolidated,
            columns=["date", "symbol", "up_return", "down_return"],
            filters=[("date", ">=", start), ("date", "<=", end)],
        )
    files = [
        path
        for path in sorted(LIMIT_ROOT.glob("*.parquet"))
        if len(path.stem) == 8
        and path.stem.isdigit()
        and start.strftime("%Y%m%d") <= path.stem <= end.strftime("%Y%m%d")
    ]
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "up_return", "down_return"])
    pieces = [
        pd.read_parquet(
            path, columns=["date", "symbol", "up_return", "down_return"]
        )
        for path in files
    ]
    return pd.concat(pieces, ignore_index=True)


def enrich_a_execution(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Attach exact known-at-date ST status and next-session limit execution."""
    limits = load_limit_prices(start, end)
    frame = frame.copy()
    # Ordinary-mainboard close-return fallback remains available outside exact
    # cache coverage. It is intentionally conservative only at the 10% limit.
    frame["buy_blocked_proxy"] = frame["next_buy_return"] >= 0.095
    frame["sell_blocked_proxy"] = frame["label_abs"] <= -0.095
    frame["is_st_signal"] = False
    frame["limit_data_available"] = False
    if limits.empty:
        return frame

    signal = limits.rename(columns={"up_return": "signal_up_return"})[
        ["date", "symbol", "signal_up_return"]
    ]
    frame = frame.merge(signal, on=["date", "symbol"], how="left")
    frame["is_st_signal"] = frame["signal_up_return"].between(0.03, 0.08)
    frame["limit_data_available"] = frame["signal_up_return"].notna()

    buy = limits.rename(
        columns={"date": "_next_buy_date", "up_return": "buy_up_return"}
    )[["_next_buy_date", "symbol", "buy_up_return"]]
    frame = frame.merge(buy, on=["_next_buy_date", "symbol"], how="left")
    exact_buy = frame["buy_up_return"].notna()
    frame.loc[exact_buy, "buy_blocked_proxy"] = (
        frame.loc[exact_buy, "next_buy_return"]
        >= frame.loc[exact_buy, "buy_up_return"] - 0.001
    )

    sell = limits.rename(
        columns={"date": "_next_sell_date", "down_return": "sell_down_return"}
    )[["_next_sell_date", "symbol", "sell_down_return"]]
    frame = frame.merge(sell, on=["_next_sell_date", "symbol"], how="left")
    exact_sell = frame["sell_down_return"].notna()
    frame.loc[exact_sell, "sell_blocked_proxy"] = (
        frame.loc[exact_sell, "label_abs"]
        <= frame.loc[exact_sell, "sell_down_return"] + 0.001
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


def daily_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    benchmarks: pd.DataFrame | None = None,
) -> dict:
    columns = ["date", "symbol", "label_abs"]
    for column in ("buy_blocked_proxy", "sell_blocked_proxy"):
        if column in frame:
            columns.append(column)
    work = frame[columns].copy()
    work["score"] = score
    work["buy_blocked_proxy"] = (
        work.get("buy_blocked_proxy", False)
        if "buy_blocked_proxy" in work
        else False
    )
    work["sell_blocked_proxy"] = (
        work.get("sell_blocked_proxy", False)
        if "sell_blocked_proxy" in work
        else False
    )
    work["buy_blocked_proxy"] = work["buy_blocked_proxy"].fillna(False).astype(bool)
    work["sell_blocked_proxy"] = work["sell_blocked_proxy"].fillna(False).astype(bool)
    ideal_by_date = work.groupby("date", sort=True)
    executable = work[~work["buy_blocked_proxy"]]
    executable_by_date = executable.groupby("date", sort=True)
    pearson = executable_by_date[["score", "label_abs"]].apply(
        lambda x: x["score"].corr(x["label_abs"]) if len(x) >= 10 else np.nan
    ).dropna()
    rank = executable_by_date[["score", "label_abs"]].apply(
        lambda x: x["score"].corr(x["label_abs"], method="spearman")
        if len(x) >= 10
        else np.nan
    ).dropna()
    ideal_pearson = ideal_by_date[["score", "label_abs"]].apply(
        lambda x: x["score"].corr(x["label_abs"]) if len(x) >= 10 else np.nan
    ).dropna()
    ideal_rank = ideal_by_date[["score", "label_abs"]].apply(
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
    result.update(summarize(ideal_pearson, "IdealIC"))
    result.update(summarize(ideal_rank, "IdealRankIC"))
    result["evaluation_rows"] = int(len(work))
    result["buy_blocked_rows"] = int(work["buy_blocked_proxy"].sum())
    result["buy_blocked_rate"] = float(work["buy_blocked_proxy"].mean())
    result["sell_blocked_rows"] = int(work["sell_blocked_proxy"].sum())
    result["sell_blocked_rate"] = float(work["sell_blocked_proxy"].mean())
    equal_weight = ideal_by_date["label_abs"].mean().sort_index()
    result["A_mainboard_equal_weight_cumulative"] = float(
        (1 + equal_weight).cumprod().iloc[-1] - 1
    )
    if benchmarks is not None:
        for name in benchmarks:
            aligned = benchmarks[name].reindex(equal_weight.index).dropna()
            if not aligned.empty:
                result[f"{name}_cumulative"] = float((1 + aligned).cumprod().iloc[-1] - 1)
    for topk in (1, 3, 5, 10):
        selected = (
            work.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=True)
            .head(topk)
        )
        # Strict execution convention: choose Top-K first. A slot whose stock is
        # already locked at the next-day buy limit stays in cash; it is not
        # replaced by Top-(K+1). A next-day sell-limit observation remains
        # marked to market, but only the buy-side commission is charged because
        # the position could not be closed on schedule.
        selected["strict_gross_return"] = selected["label_abs"].where(
            ~selected["buy_blocked_proxy"], 0.0
        )
        commission = np.where(
            selected["buy_blocked_proxy"],
            0.0,
            np.where(selected["sell_blocked_proxy"], 0.000235, 0.00047),
        )
        selected["strict_net_return"] = selected["strict_gross_return"] - commission
        strict_daily = (
            selected.groupby("date")["strict_gross_return"].mean().sort_index()
        )
        net_daily = (
            selected.groupby("date")["strict_net_return"].mean().sort_index()
        )
        ideal_daily = selected.groupby("date")["label_abs"].mean().sort_index()
        gross_equity = (1 + strict_daily).cumprod()
        # User's current commission is 0.0235% each side; stamp duty is excluded
        # here at the user's request. Minimum CNY 5 depends on position size.
        net_equity = (1 + net_daily).cumprod()
        ideal_equity = (1 + ideal_daily - 0.00047).cumprod()
        running_max = net_equity.cummax()
        result.update(
            {
                f"Top{topk}_gross_cumulative": float(gross_equity.iloc[-1] - 1),
                f"Top{topk}_net_cumulative": float(net_equity.iloc[-1] - 1),
                f"Top{topk}_ideal_net_cumulative": float(ideal_equity.iloc[-1] - 1),
                f"Top{topk}_day_win_rate": float((strict_daily > 0).mean()),
                f"Top{topk}_stock_win_rate": float(
                    (selected["strict_gross_return"] > 0).mean()
                ),
                f"Top{topk}_net_max_drawdown": float(
                    (net_equity / running_max - 1).min()
                ),
                f"Top{topk}_buy_blocked_slots": int(
                    selected["buy_blocked_proxy"].sum()
                ),
                f"Top{topk}_sell_blocked_slots": int(
                    selected["sell_blocked_proxy"].sum()
                ),
                f"Top{topk}_days": int(len(strict_daily)),
            }
        )
        if benchmarks is not None:
            for name in benchmarks:
                aligned = benchmarks[name].reindex(net_daily.index).dropna()
                if aligned.empty:
                    continue
                common = net_daily.index.intersection(aligned.index)
                strategy_value = float((1 + net_daily.loc[common]).prod() - 1)
                benchmark_value = float((1 + aligned.loc[common]).prod() - 1)
                result[f"Top{topk}_net_diff_vs_{name}"] = strategy_value - benchmark_value
    return result


def fetch_index_benchmarks(start: str, end: str, run_dir: Path) -> pd.DataFrame | None:
    token_file = Path.home() / ".config" / "tushare_token"
    if not token_file.exists():
        return None
    token = token_file.read_text(encoding="utf-8").strip()
    returns = {}
    for name, code in {"CSI300": "000300.SH", "CSI1000": "000852.SH"}.items():
        payload = {
            "api_name": "index_daily",
            "token": token,
            "params": {
                "ts_code": code,
                "start_date": pd.Timestamp(start).strftime("%Y%m%d"),
                "end_date": (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y%m%d"),
            },
            "fields": "ts_code,trade_date,close",
        }
        try:
            body = requests.post(
                "https://fastapic.stockai888.top", json=payload, timeout=60
            ).json()
            if body.get("code") != 0:
                raise RuntimeError(body.get("msg"))
            data = body["data"]
            frame = pd.DataFrame(data["items"], columns=data["fields"])
            frame["date"] = pd.to_datetime(frame["trade_date"])
            close = frame.sort_values("date").set_index("date")["close"].astype(float)
            returns[name] = close.shift(-2) / close.shift(-1) - 1
        except Exception as exc:
            print(f"benchmark {name} unavailable: {exc}", flush=True)
    if not returns:
        return None
    result = pd.DataFrame(returns).sort_index()
    result.to_csv(run_dir / "benchmark_daily_returns.csv", index_label="date")
    return result


def topk_equity(frame: pd.DataFrame, score: np.ndarray, topk: int = 10) -> pd.Series:
    work = frame[["date", "label_abs", "buy_blocked_proxy", "sell_blocked_proxy"]].copy()
    work["score"] = score
    selected = (
        work.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(topk)
    )
    selected["gross"] = selected["label_abs"].where(
        ~selected["buy_blocked_proxy"].fillna(False), 0.0
    )
    commission = np.where(
        selected["buy_blocked_proxy"].fillna(False),
        0.0,
        np.where(selected["sell_blocked_proxy"].fillna(False), 0.000235, 0.00047),
    )
    selected["net"] = selected["gross"] - commission
    daily = selected.groupby("date")["net"].mean().sort_index()
    return (1 + daily).cumprod()


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


def write_readable_report(run_dir: Path, summary: dict) -> None:
    selection = summary["selection_metrics"]
    test = summary["test_metrics"]
    rows = []
    for name in ("global", "finetuned", "a_only"):
        metrics = test[name]
        rows.append(
            {
                "model": name,
                "selected": name == summary["chosen_on_selection"],
                "selection_rank_ic": selection[name]["RankIC"],
                "test_ic": metrics["IC"],
                "test_rank_ic": metrics["RankIC"],
                "top1_net": metrics["Top1_net_cumulative"],
                "top3_net": metrics["Top3_net_cumulative"],
                "top5_net": metrics["Top5_net_cumulative"],
                "top10_net": metrics["Top10_net_cumulative"],
                "top10_win_rate": metrics["Top10_stock_win_rate"],
                "top10_max_drawdown": metrics["Top10_net_max_drawdown"],
                "csi300": metrics.get("CSI300_cumulative"),
                "csi1000": metrics.get("CSI1000_cumulative"),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(run_dir / "transfer_comparison.csv", index=False)
    pct_columns = [
        "top1_net",
        "top3_net",
        "top5_net",
        "top10_net",
        "top10_win_rate",
        "top10_max_drawdown",
        "csi300",
        "csi1000",
    ]
    display = comparison.copy()
    for column in pct_columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    for column in ("selection_rank_ic", "test_ic", "test_rank_ic"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    lines = [
        f"# {summary['run_tag']}",
        "",
        "## Outcome",
        "",
        f"Selection-valid chose `{summary['chosen_on_selection']}` by executable RankIC.",
        "A model is not deployable merely because IC is positive; strict Top-K net "
        "returns and drawdown must also pass.",
        "",
        display.to_markdown(index=False),
        "",
        "## Method",
        "",
        f"- Pretraining rows: {summary['rows']['pretrain']:,}",
        f"- A-mainboard fine-tuning rows: {summary['rows']['fine']:,}",
        f"- Blind-test rows: {summary['rows']['test']:,}",
        "- Historical ST stocks are excluded using exact daily limit ratios.",
        "- Top-K is fixed before execution; a next-day close-limit buy remains cash "
        "without rank fallback.",
        "- Commission is 0.0235% per side; minimum CNY 5 and stamp duty are not included.",
        "",
        "## Interpretation",
        "",
        "This is a one-year out-of-sample research result, not a trading recommendation. "
        "Inspect `selection_metrics.csv`, `test_metrics.csv`, `test_predictions.csv`, "
        "`model_comparison.png`, and `test_top10_equity.png` for full evidence.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--min-listed-days", type=int, default=120)
    parser.add_argument("--max-abs-label", type=float, default=0.20)
    parser.add_argument(
        "--drop-a-buy-blocked-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not reward historically unbuyable next-day limit-up A-share samples.",
    )
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
            args.min_listed_days,
            args.max_abs_label,
            training_filter=(
                market == "A" and args.drop_a_buy_blocked_train
            ),
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
            args.min_listed_days,
            args.max_abs_label,
            training_filter=(
                market == "A" and args.drop_a_buy_blocked_train
            ),
        )
        for market in MARKETS
    ]
    valid_global = add_market_features(
        add_learning_target(pd.concat(valid_parts, ignore_index=True))
    )
    valid_global_rows = len(valid_global)
    fine = load_period(
        "A",
        args.fine_start,
        args.train_end,
        args.max_symbols_a,
        args.fine_rows,
        args.seed,
        args.min_listed_days,
        args.max_abs_label,
        training_filter=args.drop_a_buy_blocked_train,
    )
    fine = add_market_features(add_learning_target(fine))
    fine_symbols = int(fine["symbol"].nunique())
    valid_a = add_market_features(add_learning_target(valid_parts[0]))
    del valid_parts
    gc.collect()
    selection = add_market_features(
        add_learning_target(
            load_period(
                "A",
                args.select_start,
                args.select_end,
                args.max_symbols_a,
                0,
                args.seed,
                args.min_listed_days,
                args.max_abs_label,
                training_filter=False,
            )
        )
    )
    test = add_market_features(
        add_learning_target(
            load_period(
                "A",
                args.test_start,
                args.test_end,
                args.max_symbols_a,
                0,
                args.seed,
                args.min_listed_days,
                args.max_abs_label,
                training_filter=False,
            )
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
    benchmarks = fetch_index_benchmarks(args.test_start, args.test_end, run_dir)
    test_metrics = {
        name: daily_metrics(test, score, benchmarks) for name, score in test_base.items()
    }
    test_metrics[chosen] = daily_metrics(test, test_score, benchmarks)
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
            "quality_filter": (
                f"listed_days>={args.min_listed_days}; abs(label)<="
                f"{args.max_abs_label}; historical A buy-limit-locked rows excluded "
                "from learning"
            ),
            "strict_execution": (
                "Top-K is selected before execution; next-day buy-limit-locked "
                "slots remain cash without fallback. Sell-limit rows are marked "
                "to market and charged buy-side commission only."
            ),
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_readable_report(run_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
