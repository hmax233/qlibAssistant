#!/usr/bin/env python3
"""Strictly evaluate external intraday-factor prediction parquet files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qlibAssistant/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN


SCRIPT = ROOT / "script"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

from evaluate_hard_risk_filters import RULES, risk_execution_features, simulate, summarize  # noqa: E402
from report_mainboard_matrix import COST_RATE, benchmark_returns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=path/to/foldC_*_test.parquet; repeat for comparisons",
    )
    parser.add_argument("--topks", default="1,3,5,10")
    parser.add_argument(
        "--include-fallback", action="store_true",
        help="also fill blocked buy slots with the next currently buyable ranked stock",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = {}
    for specification in args.prediction:
        name, separator, value = specification.partition("=")
        if not separator:
            raise SystemExit(f"invalid --prediction {specification!r}; expected NAME=PATH")
        frame = pd.read_parquet(value)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        predictions[name] = frame.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()
    start = min(frame.index.get_level_values("datetime").min() for frame in predictions.values())
    end = max(frame.index.get_level_values("datetime").max() for frame in predictions.values())
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN)
    execution = risk_execution_features(str(start.date()), str(end.date()))
    benchmarks = benchmark_returns(str(start.date()), str(end.date()))
    topks = [int(value) for value in args.topks.split(",") if value]
    rows, daily_rows = [], []
    fallback_values = (False, True) if args.include_fallback else (False,)
    for name, frame in predictions.items():
        for topk in topks:
            for rule_name in ("baseline", "event_guard"):
                for fallback in fallback_values:
                    daily = simulate(frame, execution, topk, RULES[rule_name], fallback=fallback)
                    rows.append({"source": name, "topk": topk, "rule": rule_name,
                                 "fallback": fallback, **summarize(daily, benchmarks)})
                    daily_rows.append(daily.reset_index().assign(
                        source=name, topk=topk, rule=rule_name, fallback=fallback
                    ))
    output = args.output_dir or (
        ROOT / ".qlibAssistant/analysis" / f"intraday_factor_strict_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(rows)
    daily_all = pd.concat(daily_rows, ignore_index=True)
    summary.to_csv(output / "strict_summary.csv", index=False)
    daily_all.to_csv(output / "strict_daily.csv", index=False)
    (output / "method.json").write_text(json.dumps({
        "timing": "signal T, buy T+1 close, sell/mark T+2 close",
        "cost_rate": COST_RATE,
        "fallback_rule": (
            "both no-fallback (blocked slot remains cash) and executable lower-rank fallback"
            if args.include_fallback else "blocked slot remains cash"
        ),
        "unsellable": "holding remains in portfolio",
        "event_guard_warning": "uses the complete T+1 daily bar and is exploratory/same-bar idealized",
        "board_scope": "CSI1000-mainboard sample; STAR/GEM excluded by the universe",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    display = summary[[
        "source", "topk", "rule", "fallback", "test_days", "win_rate", "net_cumulative",
        "net_max_drawdown", "net_sharpe_rf0", "average_turnover", "average_cash_ratio",
        "limit_buy_blocked_orders", "limit_sell_blocked_orders", "CSI1000_cumulative",
        "net_excess_vs_CSI1000", "CSI300_cumulative", "net_excess_vs_CSI300",
    ]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    baseline_top1 = daily_all[
        daily_all["topk"].eq(1) & daily_all["rule"].eq("baseline")
    ]
    for (source, fallback), group in baseline_top1.groupby(["source", "fallback"], sort=False):
        group = group.sort_values("datetime")
        axes[0].plot(
            pd.to_datetime(group["datetime"]), (1.0 + group["net"]).cumprod(),
            label=f"{source}{' + fallback' if fallback else ''}", linewidth=1.6,
        )
    for benchmark_name, values in benchmarks.items():
        aligned = values.loc[(values.index >= start) & (values.index <= end)].dropna()
        axes[0].plot(aligned.index, (1.0 + aligned).cumprod(), label=benchmark_name,
                     linestyle="--", linewidth=1.3)
    axes[0].axhline(1.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_title("Strict baseline Top1 equity")
    axes[0].set_ylabel("Equity (start=1)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    baseline = summary[summary["rule"].eq("baseline")].copy()
    baseline["source_variant"] = baseline["source"] + np.where(
        baseline["fallback"], " + fallback", ""
    )
    pivot = baseline.pivot(index="topk", columns="source_variant", values="net_cumulative")
    pivot.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_title("Strict baseline net cumulative return")
    axes[1].set_xlabel("TopK")
    axes[1].set_ylabel("Net cumulative return")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "strict_backtest_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(display.to_string(index=False))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
