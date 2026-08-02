#!/usr/bin/env python3
"""Strictly backtest saved ordered-threshold Fold3 predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN


ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = ROOT / "roll"
SCRIPT_DIR = ROOT / "script"
for path in (ROLL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_hard_risk_filters import (  # noqa: E402
    RULES,
    risk_execution_features,
    simulate,
    summarize,
)
from ordinal_threshold import (  # noqa: E402
    class_representatives,
    cumulative_to_class_probabilities,
    project_monotonic,
    validate_thresholds,
)
from report_mainboard_matrix import benchmark_returns  # noqa: E402


BOARD_PREFIXES = ("SH688", "SH689", "SZ300", "SZ301")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / ".qlibAssistant/ordinal_runs/ordinal_csi1000_fold3_260802"),
    )
    parser.add_argument("--ensemble-sizes", default="1,2,4,6,8,12")
    parser.add_argument("--topks", default="1,3")
    parser.add_argument(
        "--provider-uri", default=str(Path.home() / ".qlib/qlib_data/cn_data")
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def threshold_tag(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.4f}".replace(".", "")


def load_configuration(run_dir: Path, model: str, months: int) -> pd.DataFrame:
    path = run_dir / "configuration_predictions" / f"{model.lower()}_train{months}m.parquet"
    frame = pd.read_parquet(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame[frame["split"].eq("test")].sort_values(
        ["datetime", "instrument"]
    ).reset_index(drop=True)


def build_ensemble(
    configurations: dict[tuple[str, int], pd.DataFrame],
    members: list[tuple[str, int]],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    frames = [configurations[key] for key in members]
    identity = frames[0][["datetime", "instrument", "raw_return"]].reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame[["datetime", "instrument", "raw_return"]].reset_index(drop=True)
        if not candidate.equals(identity):
            raise RuntimeError(f"prediction identity mismatch for members={members}")
    probability_columns = [f"p_gt_{threshold_tag(float(value))}" for value in thresholds]
    cumulative = project_monotonic(
        np.mean([frame[probability_columns].to_numpy() for frame in frames], axis=0)
    )
    classes = cumulative_to_class_probabilities(cumulative)
    score = np.sum(
        np.nan_to_num(classes, nan=0.0) * class_representatives(thresholds)[None, :],
        axis=1,
    )
    identity["score"] = score
    identity["p_up"] = cumulative[:, int(np.where(np.isclose(thresholds, 0.0))[0][0])]
    mainboard = ~identity["instrument"].astype(str).str.startswith(BOARD_PREFIXES)
    identity = identity.loc[mainboard].copy()
    return identity.set_index(["datetime", "instrument"]).sort_index()


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    percent_columns = [
        "win_rate",
        "gross_cumulative",
        "net_cumulative",
        "net_max_drawdown",
        "average_cash_ratio",
        "CSI1000_cumulative",
        "net_excess_vs_CSI1000",
        "CSI300_cumulative",
        "net_excess_vs_CSI300",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
    if "net_sharpe_rf0" in display:
        display["net_sharpe_rf0"] = display["net_sharpe_rf0"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    return display.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    thresholds = validate_thresholds(manifest["thresholds"])
    ranking = pd.read_csv(run_dir / "selection_ranking.csv").sort_values("selection_rank")
    ordered_keys = list(zip(ranking["model"], ranking["train_months"].astype(int)))
    requested_sizes = [int(value) for value in args.ensemble_sizes.split(",") if value]
    sizes = [size for size in requested_sizes if 0 < size <= len(ordered_keys)]
    topks = [int(value) for value in args.topks.split(",") if value]

    qlib.init(
        provider_uri=str(Path(args.provider_uri).expanduser()),
        region=REG_CN,
        kernels=1,
    )
    configurations = {
        key: load_configuration(run_dir, key[0], key[1]) for key in ordered_keys
    }
    ensembles = {
        size: build_ensemble(configurations, ordered_keys[:size], thresholds)
        for size in sizes
    }
    starts = [frame.index.get_level_values("datetime").min() for frame in ensembles.values()]
    ends = [frame.index.get_level_values("datetime").max() for frame in ensembles.values()]
    execution = risk_execution_features(str(min(starts).date()), str(max(ends).date()))
    benchmarks = benchmark_returns(str(min(starts).date()), str(max(ends).date()))
    test_dates = pd.DatetimeIndex(
        sorted(ensembles[sizes[0]].index.get_level_values("datetime").unique())
    )
    benchmark_coverage = {
        name: int(values.reindex(test_dates).notna().sum())
        for name, values in benchmarks.items()
    }

    rows = []
    daily_rows = []
    for size, frame in ensembles.items():
        members = ";".join(f"{model}-{months}m" for model, months in ordered_keys[:size])
        for topk in topks:
            for rule_name in ("baseline", "event_guard"):
                for fallback in (False, True):
                    daily = simulate(frame, execution, topk, RULES[rule_name], fallback)
                    summary = summarize(daily, benchmarks)
                    for name, available_days in benchmark_coverage.items():
                        summary[f"{name}_available_days"] = available_days
                        if available_days < len(daily):
                            # A partial benchmark can make the full-period net
                            # return and aligned excess appear contradictory.
                            summary[f"{name}_cumulative"] = np.nan
                            summary[f"net_excess_vs_{name}"] = np.nan
                    rows.append(
                        {
                            "ensemble_size": size,
                            "members": members,
                            "topk": topk,
                            "rule": rule_name,
                            "fallback": fallback,
                            **summary,
                        }
                    )
                    daily_rows.append(
                        daily.reset_index().assign(
                            ensemble_size=size,
                            topk=topk,
                            rule=rule_name,
                            fallback=fallback,
                        )
                    )

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT
        / ".qlibAssistant/analysis"
        / f"ordinal_strict_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(rows).sort_values(
        ["topk", "fallback", "rule", "ensemble_size"]
    )
    results.to_csv(output / "strict_by_ensemble.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(output / "strict_daily.csv", index=False)
    selection = ranking[["selection_rank", "model", "train_months"]].copy()
    selection.to_csv(output / "selection_members.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark": name,
                "available_days": days,
                "required_days": len(test_dates),
                "complete": days == len(test_dates),
            }
            for name, days in benchmark_coverage.items()
        ]
    ).to_csv(output / "benchmark_coverage.csv", index=False)
    focus = results[
        results["topk"].eq(1) & ~results["fallback"]
    ][
        [
            "ensemble_size",
            "rule",
            "test_days",
            "win_rate",
            "gross_cumulative",
            "net_cumulative",
            "net_max_drawdown",
            "net_sharpe_rf0",
            "average_cash_ratio",
            "risk_blocked_days",
            "limit_buy_blocked_orders",
            "limit_sell_blocked_orders",
            "CSI1000_cumulative",
            "net_excess_vs_CSI1000",
            "CSI300_cumulative",
            "net_excess_vs_CSI300",
        ]
    ]
    report = f"""# CSI1000 Fold3 有序分类严格回测

## 口径

- Test：{manifest['fold3']['test']}
- Selection排名完全来自Test之前的selection-valid后段；Test不参与成员排序。
- 剔除科创板、创业板；T信号、T+1收盘买、T+2收盘卖。
- 涨跌停执行约束、0.15%换手成本；`fallback=False`时拦截后留现金。
- `event_guard`使用T+1完整日线，仍有同Bar理想化；它是事后探索规则。
- 基准必须覆盖全部{len(test_dates)}个Test交易日才计算累计和超额；当前覆盖：
  {benchmark_coverage}。覆盖不足时相关列留空，不用局部日期冒充全区间。

## Top1、不向下替补

{markdown_table(focus)}

## Selection成员顺序

{selection.to_markdown(index=False)}

完整逐日路径见`strict_daily.csv`，全部Top1/Top3和替补对照见
`strict_by_ensemble.csv`。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "output": str(output),
                "ensemble_sizes": sizes,
                "topks": topks,
                "thresholds": thresholds.tolist(),
                "board_filter": "STAR and ChiNext removed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
