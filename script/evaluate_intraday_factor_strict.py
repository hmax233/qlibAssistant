#!/usr/bin/env python3
"""Strictly evaluate external intraday-factor prediction parquet files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN


ROOT = Path(__file__).resolve().parents[1]
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
    for name, frame in predictions.items():
        for topk in topks:
            for rule_name in ("baseline", "event_guard"):
                daily = simulate(frame, execution, topk, RULES[rule_name], fallback=False)
                rows.append({"source": name, "topk": topk, "rule": rule_name,
                             "fallback": False, **summarize(daily, benchmarks)})
                daily_rows.append(daily.reset_index().assign(
                    source=name, topk=topk, rule=rule_name, fallback=False
                ))
    output = args.output_dir or (
        ROOT / ".qlibAssistant/analysis" / f"intraday_factor_strict_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_summary.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(output / "strict_daily.csv", index=False)
    (output / "method.json").write_text(json.dumps({
        "timing": "signal T, buy T+1 close, sell/mark T+2 close",
        "cost_rate": COST_RATE,
        "fallback": False,
        "unbuyable": "slot remains cash",
        "unsellable": "holding remains in portfolio",
        "event_guard_warning": "uses the complete T+1 daily bar and is exploratory/same-bar idealized",
        "board_scope": "CSI1000-mainboard sample; STAR/GEM excluded by the universe",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    display = summary[[
        "source", "topk", "rule", "test_days", "win_rate", "net_cumulative",
        "net_max_drawdown", "net_sharpe_rf0", "average_turnover", "average_cash_ratio",
        "limit_buy_blocked_orders", "limit_sell_blocked_orders", "CSI1000_cumulative",
        "net_excess_vs_CSI1000", "CSI300_cumulative", "net_excess_vs_CSI300",
    ]]
    print(display.to_string(index=False))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
