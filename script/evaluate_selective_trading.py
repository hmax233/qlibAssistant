#!/usr/bin/env python3
"""XGBoost-240 × DoubleEnsemble-240 三折选择性交易交叉评测。

本脚本固定只使用这两个模型的 240 月 recorder。模型分数用于排名，真实交易
收益通过 Qlib 原始收盘价表达式重新读取，避免把训练用 CSZScoreNorm 标签当收益率。
所有阈值只从各 Fold 的 valid_sig_analysis（selection-valid）确定。
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ROOT / ".qlibAssistant" / "mlruns"
INDEX_CACHE = ROOT / ".qlibAssistant" / "cache" / "tushare_index_daily.csv"
STAR_PREFIXES = ("SH688", "SH689")
MODEL_PATTERNS = {
    "xgb240": "XGBModel_Alpha158_csi1000_custom_step0_windowcmp_fold{fold}_train240m_",
    "de240": "DEnsembleModel_Alpha158_csi1000_custom_step0_hmx_learning_DoubleEnsemble_fold{fold}_train240m_",
}


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def as_series(value, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    result = value.copy()
    result.name = name
    return result


def discover_recorder(model: str, fold: int) -> tuple[str, str, str]:
    pattern = MODEL_PATTERNS[model].format(fold=fold)
    candidates = []
    for exp_meta in MLRUNS.glob("*/meta.yaml"):
        meta = yaml.safe_load(exp_meta.read_text()) or {}
        name = str(meta.get("name", ""))
        if pattern not in name:
            continue
        exp_id = exp_meta.parent.name
        for rec in exp_meta.parent.iterdir():
            if not rec.is_dir():
                continue
            required = (
                rec / "artifacts" / "pred.pkl",
                rec / "artifacts" / "valid_sig_analysis" / "pred.pkl",
            )
            if not all(path.exists() for path in required):
                continue
            rec_meta_path = rec / "meta.yaml"
            rec_meta = yaml.safe_load(rec_meta_path.read_text()) if rec_meta_path.exists() else {}
            candidates.append((int((rec_meta or {}).get("start_time", 0)), exp_id, rec.name, name))
    if not candidates:
        raise FileNotFoundError(f"找不到 {model} fold{fold} train240m recorder")
    _, exp_id, rec_id, exp_name = sorted(candidates)[-1]
    return exp_id, rec_id, exp_name


def load_predictions(exp_id: str, rec_id: str, split: str, name: str) -> pd.Series:
    folder = MLRUNS / exp_id / rec_id / "artifacts"
    if split == "selection_valid":
        folder /= "valid_sig_analysis"
    return as_series(load_pickle(folder / "pred.pkl"), name)


def load_validation_label(exp_id: str, rec_id: str) -> pd.Series:
    path = MLRUNS / exp_id / rec_id / "artifacts" / "valid_sig_analysis" / "label.pkl"
    return as_series(load_pickle(path), "label")


def daily_zscore(series: pd.Series) -> pd.Series:
    def normalize(values):
        std = values.std()
        return (values - values.mean()) / std if np.isfinite(std) and std > 0 else values * np.nan
    return series.groupby(level="datetime", group_keys=False).apply(normalize)


def rank_metrics(prediction: pd.Series, label: pd.Series) -> tuple[float, float]:
    frame = pd.concat([prediction, label], axis=1).dropna()
    daily = frame.groupby(level="datetime").apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], method="spearman"))
    mean = float(daily.mean())
    icir = float(mean / daily.std()) if daily.std() > 0 else np.nan
    return mean, icir


def model_frame(xgb: pd.Series, de: pd.Series, mode: str, weights: dict[str, float]) -> pd.DataFrame:
    frame = pd.concat([xgb.rename("xgb"), de.rename("de")], axis=1, join="inner").dropna()
    frame["z_xgb"] = daily_zscore(frame.xgb)
    frame["z_de"] = daily_zscore(frame.de)
    if mode == "xgb240":
        frame["score"] = frame.z_xgb
    elif mode == "de240":
        frame["score"] = frame.z_de
    elif mode == "equal_ensemble":
        frame["score"] = (frame.z_xgb + frame.z_de) / 2
    elif mode == "valid_icir_weighted":
        total = max(weights["xgb"], 0) + max(weights["de"], 0)
        if total <= 0:
            frame["score"] = (frame.z_xgb + frame.z_de) / 2
        else:
            frame["score"] = (frame.z_xgb * max(weights["xgb"], 0) + frame.z_de * max(weights["de"], 0)) / total
    else:
        raise ValueError(mode)
    frame["rank_xgb"] = frame.xgb.groupby(level="datetime").rank(ascending=False, method="min")
    frame["rank_de"] = frame.de.groupby(level="datetime").rank(ascending=False, method="min")
    return frame.dropna(subset=["score"])


def raw_returns_for(index: pd.MultiIndex, provider_uri: str) -> pd.Series:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=REG_CN)
    dates = index.get_level_values("datetime")
    instruments = sorted(set(index.get_level_values("instrument").astype(str)))
    data = D.features(
        instruments,
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=str(pd.Timestamp(dates.min()).date()),
        end_time=str(pd.Timestamp(dates.max()).date()),
        freq="day",
    )
    return data.iloc[:, 0].rename("raw_return")


def benchmarks() -> dict[str, pd.Series]:
    cache = pd.read_csv(INDEX_CACHE, parse_dates=["datetime"])
    output = {}
    for name, group in cache.groupby("index"):
        close = group.drop_duplicates("datetime").set_index("datetime")["close"].sort_index()
        output[name] = (close.shift(-2) / close.shift(-1) - 1).rename(name)
    return output


def validation_cuts(frame: pd.DataFrame, slots: int) -> dict[str, float]:
    rows, previous = [], set()
    for _, group in frame.groupby(level="datetime", sort=True):
        ranked = group.sort_values("score", ascending=False)
        top = ranked.head(slots)
        members = set(top.index.get_level_values("instrument"))
        rows.append({
            "strength": float(top.score.mean()),
            "persistence": len(members & previous) / slots if previous else np.nan,
            "replacement_gap": float(ranked.iloc[0].score - ranked.iloc[slots - 1].score),
        })
        previous = members
    daily = pd.DataFrame(rows)
    return {
        "strength_q50": float(daily.strength.quantile(0.5)),
        "strength_q70": float(daily.strength.quantile(0.7)),
        "persistence_q50": float(daily.persistence.quantile(0.5)),
        "persistence_q70": float(daily.persistence.quantile(0.7)),
        "gap_q50": float(daily.replacement_gap.quantile(0.5)),
        "gap_q70": float(daily.replacement_gap.quantile(0.7)),
    }


RULES = (
    "daily_top3",
    "strength_q50_cash", "strength_q70_cash",
    "persistence_q50_cash", "persistence_q70_cash",
    "buffer_top5", "buffer_top10",
    "margin_q50", "margin_q70",
    "combined_q70_buffer10",
    "consensus_top10", "consensus_top20", "consensus_top10_persist",
)


def target_holdings(
    ranked: pd.DataFrame, holdings: list[str], previous_top: set[str], rule: str,
    cuts: dict[str, float], slots: int,
) -> list[str]:
    instruments = list(ranked.index.get_level_values("instrument"))
    scores = dict(zip(instruments, ranked.score))
    ranks = {instrument: idx + 1 for idx, instrument in enumerate(instruments)}
    current = [item for item in holdings if item in ranks]
    top = instruments[:slots]
    strength = float(ranked.head(slots).score.mean())
    persistence = len(set(top) & previous_top) / slots if previous_top else np.nan

    if rule == "daily_top3":
        return top
    if rule.startswith("strength_"):
        q = "strength_q50" if "q50" in rule else "strength_q70"
        return top if strength >= cuts[q] else []
    if rule.startswith("persistence_"):
        q = "persistence_q50" if "q50" in rule else "persistence_q70"
        return top if np.isfinite(persistence) and persistence >= cuts[q] else []
    if rule.startswith("buffer_top"):
        exit_rank = int(rule.replace("buffer_top", ""))
        kept = [item for item in current if ranks[item] <= exit_rank]
        return kept + [item for item in instruments if item not in kept][: slots - len(kept)]
    if rule.startswith("margin_"):
        gap = cuts["gap_q50" if rule == "margin_q50" else "gap_q70"]
        result = current[:slots]
        result += [item for item in instruments if item not in result][: slots - len(result)]
        challenger = next((item for item in instruments if item not in result), None)
        if challenger and result:
            weakest = min(result, key=scores.get)
            if scores[challenger] - scores[weakest] >= gap:
                result[result.index(weakest)] = challenger
        return result
    if rule == "combined_q70_buffer10":
        result = [item for item in current if ranks[item] <= 10]
        if strength >= cuts["strength_q70"]:
            result += [item for item in instruments if item not in result][: slots - len(result)]
        result = result[:slots]
        if len(result) == slots:
            challenger = next((item for item in instruments if item not in result), None)
            if challenger:
                weakest = min(result, key=scores.get)
                if scores[challenger] - scores[weakest] >= cuts["gap_q70"]:
                    result[result.index(weakest)] = challenger
        return result
    if rule.startswith("consensus_"):
        limit = 10 if "top10" in rule else 20
        eligible = ranked[(ranked.rank_xgb <= limit) & (ranked.rank_de <= limit)]
        candidates = list(eligible.sort_values("score", ascending=False).index.get_level_values("instrument"))
        if rule.endswith("persist"):
            candidates = [item for item in candidates if item in previous_top]
        return candidates[:slots]
    raise ValueError(rule)


def simulate(
    frame: pd.DataFrame, rule: str, cuts: dict[str, float], slots: int,
    cost_rate: float, fold: str, model_mode: str, universe: str,
) -> tuple[dict, pd.DataFrame]:
    holdings, previous_top, rows = [], set(), []
    for date, group in frame.groupby(level="datetime", sort=True):
        ranked = group.dropna(subset=["score", "raw_return"]).sort_values("score", ascending=False)
        if ranked.empty:
            continue
        target = target_holdings(ranked, holdings, previous_top, rule, cuts, slots)
        labels = ranked.raw_return.droplevel("datetime")
        gross = float(labels.reindex(target).mean()) * (len(target) / slots) if target else 0.0
        buys = len(set(target) - set(holdings))
        sells = len(set(holdings) - set(target))
        turnover = max(buys, sells) / slots
        cost = turnover * cost_rate
        net = gross - cost
        rows.append({
            "datetime": date, "fold": fold, "model_mode": model_mode, "universe_variant": universe,
            "rule": rule, "holdings": ",".join(target), "holding_count": len(target),
            "gross_return": gross, "turnover": turnover, "cost": cost, "net_return": net,
        })
        previous_top = set(ranked.head(slots).index.get_level_values("instrument"))
        holdings = target
    daily = pd.DataFrame(rows).set_index("datetime")
    daily["gross_equity"] = (1 + daily.gross_return).cumprod()
    daily["net_equity"] = (1 + daily.net_return).cumprod()
    drawdown = daily.net_equity / daily.net_equity.cummax() - 1
    active = daily.holding_count > 0
    positive_sum = daily.net_return.clip(lower=0).sum()
    best5 = daily.net_return.nlargest(min(5, len(daily))).sum()
    summary = {
        "fold": fold, "model_mode": model_mode, "universe_variant": universe, "rule": rule,
        "days": len(daily), "active_days": int(active.sum()), "active_ratio": float(active.mean()),
        "average_holdings": float(daily.holding_count.mean()), "average_turnover": float(daily.turnover.mean()),
        "gross_cumulative": float(daily.gross_equity.iloc[-1] - 1),
        "net_cumulative": float(daily.net_equity.iloc[-1] - 1),
        "net_max_drawdown": float(drawdown.min()),
        "active_win_rate": float((daily.loc[active, "gross_return"] > 0).mean()) if active.any() else np.nan,
        "best_5_days_net_sum": float(best5),
        "best_5_days_share_of_positive_sum": float(best5 / positive_sum) if positive_sum > 0 else np.nan,
    }
    return summary, daily.reset_index()


def add_benchmarks(summary: dict, daily: pd.DataFrame, benchmark_map: dict[str, pd.Series]) -> None:
    indexed = daily.set_index("datetime")
    for name in ("CSI1000", "CSI300"):
        aligned = benchmark_map[name].reindex(indexed.index).dropna()
        if aligned.empty:
            continue
        last = aligned.index.max()
        benchmark_return = float((1 + aligned).prod() - 1)
        strategy_return = float((1 + indexed.loc[:last, "net_return"]).prod() - 1)
        summary[f"{name}_cumulative"] = benchmark_return
        summary[f"net_diff_vs_{name}"] = strategy_return - benchmark_return
        summary[f"net_excess_vs_{name}"] = (1 + strategy_return) / (1 + benchmark_return) - 1


def aggregate(fold_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_mode", "universe_variant", "rule"]
    numeric = [
        "net_cumulative", "gross_cumulative", "net_max_drawdown", "active_ratio", "average_holdings",
        "average_turnover", "active_win_rate", "best_5_days_share_of_positive_sum",
        "CSI1000_cumulative", "net_diff_vs_CSI1000", "net_excess_vs_CSI1000",
        "CSI300_cumulative", "net_diff_vs_CSI300", "net_excess_vs_CSI300",
    ]
    rows = []
    for group_key, group in fold_summary.groupby(keys):
        row = dict(zip(keys, group_key))
        for column in numeric:
            row[f"mean_{column}"] = group[column].mean()
        row["worst_net_cumulative"] = group.net_cumulative.min()
        row["worst_net_excess_vs_CSI1000"] = group.net_excess_vs_CSI1000.min()
        row["worst_net_excess_vs_CSI300"] = group.net_excess_vs_CSI300.min()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["worst_net_excess_vs_CSI1000", "mean_net_excess_vs_CSI1000"], ascending=False
    )


def pct(value) -> str:
    return "—" if pd.isna(value) else f"{value:.2%}"


def write_report(output: Path, manifest: pd.DataFrame, folds: pd.DataFrame, agg: pd.DataFrame) -> None:
    top = agg.head(20)
    lines = [
        "# XGBoost-240 × DoubleEnsemble-240 三折选择性交易交叉实验", "",
        "## 口径", "",
        "- 只使用XGBoost-240和DoubleEnsemble-240；每个Fold各一个recorder。",
        "- 排名使用模型预测；交易收益从Qlib原始收盘价重新计算，不使用标准化训练标签。",
        "- 阈值只由各Fold的selection-valid确定，再原样用于Test。",
        "- 成本按实际换手 × 0.15%；尚未模拟最低佣金、滑点、涨跌停、停牌和整数手。",
        "- all与ex_star分别评测；ex_star排除SH688/SH689。", "",
        "## Recorder", "",
        "| Fold | 模型 | Experiment | Recorder | Valid Rank ICIR |",
        "|---|---|---|---|---:|",
    ]
    for _, row in manifest.iterrows():
        lines.append(f"| {row.fold} | {row.model} | {row.experiment_id} | `{row.recorder_id}` | {row.valid_rank_icir:.4f} |")
    lines += ["", "## 按最差Fold CSI1000超额排序的前20项", "",
              "| 模型组合 | 股票范围 | 规则 | 平均净收益 | 最差净收益 | 平均/最差超额CSI1000 | 平均/最差超额CSI300 | 换手 | 活跃率 | 回撤 | 胜率 | Top5盈利日集中度 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in top.iterrows():
        lines.append(
            f"| {row.model_mode} | {row.universe_variant} | {row.rule} | {pct(row.mean_net_cumulative)} | "
            f"{pct(row.worst_net_cumulative)} | {pct(row.mean_net_excess_vs_CSI1000)}/{pct(row.worst_net_excess_vs_CSI1000)} | "
            f"{pct(row.mean_net_excess_vs_CSI300)}/{pct(row.worst_net_excess_vs_CSI300)} | "
            f"{pct(row.mean_average_turnover)} | {pct(row.mean_active_ratio)} | {pct(row.mean_net_max_drawdown)} | "
            f"{pct(row.mean_active_win_rate)} | {pct(row.mean_best_5_days_share_of_positive_sum)} |"
        )
    lines += ["", "## 逐Fold完整结果", "",
              "下表包含全部模型组合、规则和含/剔除科创板口径。详细日频持仓见 `daily_results.csv`。", "",
              "| Fold | 模型 | 范围 | 规则 | 净收益 | CSI1000 | 超额CSI1000 | CSI300 | 超额CSI300 | 换手 | 活跃率 | 回撤 | 胜率 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in folds.sort_values(["fold", "universe_variant", "model_mode", "rule"]).iterrows():
        lines.append(
            f"| {row.fold} | {row.model_mode} | {row.universe_variant} | {row.rule} | {pct(row.net_cumulative)} | "
            f"{pct(row.CSI1000_cumulative)} | {pct(row.net_excess_vs_CSI1000)} | {pct(row.CSI300_cumulative)} | "
            f"{pct(row.net_excess_vs_CSI300)} | {pct(row.average_turnover)} | {pct(row.active_ratio)} | "
            f"{pct(row.net_max_drawdown)} | {pct(row.active_win_rate)} |"
        )
    lines += ["", "## 限制", "",
              "结果仍是历史Test研究，不是新的无偏前向证据。规则数量较多，必须优先看最差Fold和参数邻域，不能只挑最高收益。",
              "投票规则可能长期无候选或候选不足3只；未投资部分按现金处理。"]
    (output / "detailed_report.md").write_text("\n".join(lines) + "\n")


def plot_results(output: Path, agg: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    selected = agg[(agg.universe_variant == "ex_star")].head(16).iloc[::-1]
    labels = selected.model_mode + "\n" + selected.rule
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].barh(labels, selected.mean_net_cumulative, color="#3B82F6")
    axes[0].scatter(selected.worst_net_cumulative, range(len(selected)), color="#DC2626", label="worst fold")
    axes[0].set_title("Ex-STAR mean / worst net cumulative")
    axes[0].legend()
    axes[1].barh(labels, selected.mean_net_excess_vs_CSI1000, color="#10B981")
    axes[1].scatter(selected.worst_net_excess_vs_CSI1000, range(len(selected)), color="#DC2626", label="worst fold")
    axes[1].set_title("Ex-STAR mean / worst excess vs CSI1000")
    axes[1].legend()
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "strategy_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument("--slots", type=int, default=3)
    args = parser.parse_args()

    output = ROOT / ".qlibAssistant" / "analysis" / f"selective_xgb240_de240_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    benchmark_map = benchmarks()
    manifest_rows, threshold_rows, summary_rows, daily_rows = [], [], [], []

    for fold_num in (1, 2, 3):
        fold = f"fold{fold_num}"
        discovered = {name: discover_recorder(name, fold_num) for name in MODEL_PATTERNS}
        valid_pred, test_pred, valid_labels, icirs = {}, {}, {}, {}
        for name, (exp_id, rec_id, exp_name) in discovered.items():
            valid_pred[name] = load_predictions(exp_id, rec_id, "selection_valid", name)
            test_pred[name] = load_predictions(exp_id, rec_id, "test", name)
            valid_labels[name] = load_validation_label(exp_id, rec_id)
            rank_ic, rank_icir = rank_metrics(valid_pred[name], valid_labels[name])
            icirs["xgb" if name == "xgb240" else "de"] = rank_icir
            manifest_rows.append({
                "fold": fold, "model": name, "experiment_id": exp_id, "experiment_name": exp_name,
                "recorder_id": rec_id, "valid_rank_ic": rank_ic, "valid_rank_icir": rank_icir,
            })

        raw_returns = raw_returns_for(pd.concat(test_pred.values(), axis=1, join="inner").index, args.provider_uri)
        for universe in ("all", "ex_star"):
            for mode in ("xgb240", "de240", "equal_ensemble", "valid_icir_weighted"):
                valid = model_frame(valid_pred["xgb240"], valid_pred["de240"], mode, icirs)
                test = model_frame(test_pred["xgb240"], test_pred["de240"], mode, icirs)
                if universe == "ex_star":
                    valid = valid.loc[~valid.index.get_level_values("instrument").astype(str).str.startswith(STAR_PREFIXES)]
                    test = test.loc[~test.index.get_level_values("instrument").astype(str).str.startswith(STAR_PREFIXES)]
                test = test.join(raw_returns, how="left").dropna(subset=["raw_return"])
                cuts = validation_cuts(valid, args.slots)
                threshold_rows.append({"fold": fold, "model_mode": mode, "universe_variant": universe, **cuts})
                for rule in RULES:
                    summary, daily = simulate(test, rule, cuts, args.slots, args.cost_rate, fold, mode, universe)
                    add_benchmarks(summary, daily, benchmark_map)
                    summary_rows.append(summary)
                    daily_rows.append(daily)

    manifest = pd.DataFrame(manifest_rows)
    thresholds = pd.DataFrame(threshold_rows)
    fold_summary = pd.DataFrame(summary_rows)
    aggregate_summary = aggregate(fold_summary)
    daily = pd.concat(daily_rows, ignore_index=True)
    manifest.to_csv(output / "recorder_manifest.csv", index=False)
    thresholds.to_csv(output / "validation_thresholds.csv", index=False)
    fold_summary.to_csv(output / "fold_summary.csv", index=False)
    aggregate_summary.to_csv(output / "aggregate_summary.csv", index=False)
    daily.to_csv(output / "daily_results.csv", index=False)
    write_report(output, manifest, fold_summary, aggregate_summary)
    plot_results(output, aggregate_summary)
    print(output)


if __name__ == "__main__":
    main()
