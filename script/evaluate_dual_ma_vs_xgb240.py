#!/usr/bin/env python3
"""在固定三Fold上比较双均线与XGBoost-240 Top3。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = Path("~/.qlib/qlib_data/cn_data").expanduser()
INDEX_CACHE = ROOT / ".qlibAssistant" / "cache" / "tushare_index_daily.csv"
XGB_REPORT = ROOT / ".qlibAssistant" / "analysis" / "selective_xgb240_de240_20260721_23_33_08" / "fold_summary.csv"
FOLDS = {
    "fold1": ("2025-04-15", "2025-09-15"),
    "fold2": ("2025-09-16", "2026-02-13"),
    "fold3": ("2026-02-24", "2026-07-15"),
}
WINDOWS = ((5, 20), (10, 30), (20, 60))
STAR = ("SH688", "SH689")
COST = 0.0015
SLOTS = 3


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def load_benchmarks() -> dict[str, pd.Series]:
    cache = pd.read_csv(INDEX_CACHE, parse_dates=["datetime"])
    result = {}
    for name, group in cache.groupby("index"):
        close = group.drop_duplicates("datetime").set_index("datetime").close.sort_index()
        result[name] = close.shift(-2) / close.shift(-1) - 1
    return result


def load_stock_features() -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(PROVIDER), region=REG_CN)
    fields = ["Ref($close, -2)/Ref($close, -1)-1"]
    names = ["return"]
    for short, long in WINDOWS:
        fields.append(f"Mean($close,{short})/Mean($close,{long})-1")
        names.append(f"ma{short}_{long}")
    data = D.features(
        D.instruments("csi1000"), fields,
        start_time=min(value[0] for value in FOLDS.values()),
        end_time=max(value[1] for value in FOLDS.values()), freq="day",
    )
    data.columns = names
    return data


def add_benchmarks(summary: dict, daily: pd.DataFrame, benchmark: dict[str, pd.Series]) -> None:
    for name in ("CSI1000", "CSI300"):
        values = benchmark[name].reindex(daily.index).dropna()
        end = values.index.max()
        strategy = float((1 + daily.loc[:end, "net_return"]).prod() - 1)
        base = float((1 + values).prod() - 1)
        summary[f"{name}_cumulative"] = base
        summary[f"net_diff_vs_{name}"] = strategy - base
        summary[f"net_excess_vs_{name}"] = (1 + strategy) / (1 + base) - 1


def simulate_stock_ma(frame: pd.DataFrame, score_col: str, fold: str, universe: str) -> tuple[dict, pd.DataFrame]:
    holdings, rows = set(), []
    for date, group in frame.groupby(level="datetime", sort=True):
        ranked = group.dropna(subset=[score_col, "return"])
        ranked = ranked[ranked[score_col] > 0].sort_values(score_col, ascending=False)
        target_list = list(ranked.head(SLOTS).index.get_level_values("instrument"))
        target = set(target_list)
        turnover = max(len(target - holdings), len(holdings - target)) / SLOTS
        gross = float(ranked.head(SLOTS)["return"].mean()) * (len(target) / SLOTS) if target else 0.0
        rows.append({"datetime": date, "holdings": ",".join(target_list), "holding_count": len(target),
                     "gross_return": gross, "turnover": turnover, "net_return": gross - turnover * COST})
        holdings = target
    daily = pd.DataFrame(rows).set_index("datetime")
    equity = (1 + daily.net_return).cumprod()
    active = daily.holding_count > 0
    summary = {
        "fold": fold, "strategy": f"stock_{score_col}_top3", "universe_variant": universe,
        "days": len(daily), "active_ratio": float(active.mean()),
        "average_holdings": float(daily.holding_count.mean()),
        "average_turnover": float(daily.turnover.mean()),
        "gross_cumulative": float((1 + daily.gross_return).prod() - 1),
        "net_cumulative": float(equity.iloc[-1] - 1),
        "net_max_drawdown": max_drawdown(equity),
        "active_win_rate": float((daily.loc[active, "gross_return"] > 0).mean()),
    }
    return summary, daily


def simulate_index_ma(short: int, long: int, fold: str, dates: tuple[str, str], benchmark: dict[str, pd.Series]) -> tuple[dict, pd.DataFrame]:
    cache = pd.read_csv(INDEX_CACHE, parse_dates=["datetime"])
    close = cache[cache["index"] == "CSI1000"].drop_duplicates("datetime").set_index("datetime").close.sort_index()
    signal = close.rolling(short).mean() > close.rolling(long).mean()
    returns = benchmark["CSI1000"].loc[dates[0]:dates[1]].dropna()
    active = signal.reindex(returns.index).fillna(False)
    turnover = active.astype(int).diff().abs().fillna(float(active.iloc[0]))
    gross = returns.where(active, 0.0)
    net = gross - turnover * COST
    daily = pd.DataFrame({"gross_return": gross, "turnover": turnover, "net_return": net,
                          "holding_count": active.astype(int)})
    equity = (1 + net).cumprod()
    summary = {
        "fold": fold, "strategy": f"index_ma{short}_{long}_timing", "universe_variant": "index",
        "days": len(daily), "active_ratio": float(active.mean()), "average_holdings": float(active.mean()),
        "average_turnover": float(turnover.mean()), "gross_cumulative": float((1 + gross).prod() - 1),
        "net_cumulative": float(equity.iloc[-1] - 1), "net_max_drawdown": max_drawdown(equity),
        "active_win_rate": float((gross[active] > 0).mean()),
    }
    return summary, daily


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["strategy", "universe_variant"]):
        row = {"strategy": keys[0], "universe_variant": keys[1]}
        for col in ("net_cumulative", "net_max_drawdown", "active_ratio", "average_holdings", "average_turnover",
                    "active_win_rate", "net_excess_vs_CSI1000", "net_excess_vs_CSI300"):
            row[f"mean_{col}"] = group[col].mean()
        row["worst_net_cumulative"] = group.net_cumulative.min()
        row["worst_net_excess_vs_CSI1000"] = group.net_excess_vs_CSI1000.min()
        row["worst_net_excess_vs_CSI300"] = group.net_excess_vs_CSI300.min()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("worst_net_excess_vs_CSI1000", ascending=False)


def pct(value) -> str:
    return "—" if pd.isna(value) else f"{value:.2%}"


def write_report(output: Path, folds: pd.DataFrame, agg: pd.DataFrame) -> None:
    lines = ["# 双均线与XGBoost-240三折对比", "", "## 口径", "",
             "- 固定5/20、10/30、20/60，不使用Test选参数。",
             "- 个股双均线：短均线高于长均线才入选，按短/长均线比率持有CSI1000 Top3。",
             "- 指数双均线：短均线高于长均线时持有CSI1000，否则现金。",
             "- XGBoost使用已有240月Fold recorder的每日Top3结果。",
             "- 成本均为实际换手×0.15%；个股策略同时报告含/剔除科创板。", "",
             "## 三折汇总", "",
             "| 策略 | 范围 | 平均/最差净收益 | 平均/最差超额CSI1000 | 平均/最差超额CSI300 | 换手 | 活跃率 | 回撤 | 胜率 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in agg.iterrows():
        lines.append(f"| {r.strategy} | {r.universe_variant} | {pct(r.mean_net_cumulative)}/{pct(r.worst_net_cumulative)} | "
                     f"{pct(r.mean_net_excess_vs_CSI1000)}/{pct(r.worst_net_excess_vs_CSI1000)} | "
                     f"{pct(r.mean_net_excess_vs_CSI300)}/{pct(r.worst_net_excess_vs_CSI300)} | "
                     f"{pct(r.mean_average_turnover)} | {pct(r.mean_active_ratio)} | {pct(r.mean_net_max_drawdown)} | {pct(r.mean_active_win_rate)} |")
    lines += ["", "## 逐Fold", "",
              "| Fold | 策略 | 范围 | 净收益 | CSI1000 | 超额CSI1000 | CSI300 | 超额CSI300 | 换手 | 活跃率 | 回撤 | 胜率 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in folds.sort_values(["fold", "strategy", "universe_variant"]).iterrows():
        lines.append(f"| {r.fold} | {r.strategy} | {r.universe_variant} | {pct(r.net_cumulative)} | {pct(r.CSI1000_cumulative)} | "
                     f"{pct(r.net_excess_vs_CSI1000)} | {pct(r.CSI300_cumulative)} | {pct(r.net_excess_vs_CSI300)} | "
                     f"{pct(r.average_turnover)} | {pct(r.active_ratio)} | {pct(r.net_max_drawdown)} | {pct(r.active_win_rate)} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    output = ROOT / ".qlibAssistant" / "analysis" / f"dual_ma_vs_xgb240_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    benchmark = load_benchmarks()
    features = load_stock_features()
    summaries, daily_rows = [], []
    for fold, dates in FOLDS.items():
        current = features.loc[pd.IndexSlice[:, dates[0]:dates[1]], :]
        for universe in ("all", "ex_star"):
            frame = current
            if universe == "ex_star":
                mask = ~frame.index.get_level_values("instrument").astype(str).str.startswith(STAR)
                frame = frame.loc[mask]
            for short, long in WINDOWS:
                summary, daily = simulate_stock_ma(frame, f"ma{short}_{long}", fold, universe)
                add_benchmarks(summary, daily, benchmark)
                summaries.append(summary)
                daily_rows.append(daily.reset_index().assign(fold=fold, strategy=summary["strategy"], universe_variant=universe))
        for short, long in WINDOWS:
            summary, daily = simulate_index_ma(short, long, fold, dates, benchmark)
            add_benchmarks(summary, daily, benchmark)
            summaries.append(summary)
            daily_rows.append(daily.reset_index().assign(fold=fold, strategy=summary["strategy"], universe_variant="index"))

    xgb = pd.read_csv(XGB_REPORT)
    xgb = xgb[(xgb.model_mode == "xgb240") & (xgb.rule == "daily_top3")].copy()
    xgb["strategy"] = "xgb240_daily_top3"
    summaries.extend(xgb[["fold", "strategy", "universe_variant", "days", "active_ratio", "average_holdings",
                          "average_turnover", "gross_cumulative", "net_cumulative", "net_max_drawdown", "active_win_rate",
                          "CSI1000_cumulative", "net_diff_vs_CSI1000", "net_excess_vs_CSI1000",
                          "CSI300_cumulative", "net_diff_vs_CSI300", "net_excess_vs_CSI300"]].to_dict("records"))
    folds = pd.DataFrame(summaries)
    agg = aggregate(folds)
    folds.to_csv(output / "fold_summary.csv", index=False)
    agg.to_csv(output / "aggregate_summary.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(output / "dual_ma_daily.csv", index=False)
    write_report(output, folds, agg)
    print(output)


if __name__ == "__main__":
    main()
