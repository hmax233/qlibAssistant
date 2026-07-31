#!/usr/bin/env python3
"""Decompose the historical Fixed Fold3 event_guard cumulative return."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from evaluate_source_hard_filters import DEFAULT_WINDOW_DETAIL, build_fixed_sources


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".qlibAssistant" / "analysis"
DAILY = (
    ANALYSIS
    / "source_hard_filter_20260728_225020"
    / "source_hard_filter_daily.csv"
)


def main() -> None:
    daily = pd.read_csv(DAILY, parse_dates=["datetime"])
    selected = daily[
        daily["source"].eq("FixedArchitecture")
        & daily["fold"].eq("fold3")
        & daily["topk"].eq(1)
        & daily["rule"].eq("event_guard")
        & daily["fallback"].eq(False)
    ].sort_values("datetime")

    recorder_detail = pd.read_csv(
        DEFAULT_WINDOW_DETAIL,
        dtype={"experiment_id": str, "recorder_id": str},
    )
    fixed = next(
        item
        for item in build_fixed_sources(recorder_detail)
        if item["fold"] == "fold3"
    )
    score = fixed["score"]
    names = pd.read_csv(ROOT / ".qlibAssistant/cache/stock_basic.csv").set_index("code")[
        "name"
    ]

    top = selected.nlargest(15, "net").copy()
    instruments = []
    for date in top["datetime"]:
        ranked = score.xs(date, level="datetime").sort_values(ascending=False)
        instruments.append(str(ranked.index[0]))
    top["instrument"] = instruments
    top["name"] = top["instrument"].map(names)

    positive_sum = selected.loc[selected["net"] > 0, "net"].sum()
    exclusions = []
    for count in (0, 1, 3, 5, 10):
        kept = selected if count == 0 else selected.drop(
            selected.nlargest(count, "net").index
        )
        exclusions.append(
            {
                "excluded_best_days": count,
                "remaining_net_cumulative": float((1 + kept["net"]).prod() - 1),
                "removed_positive_sum_share": (
                    0.0
                    if count == 0
                    else float(
                        selected.nlargest(count, "net")["net"].sum() / positive_sum
                    )
                ),
            }
        )
    exclusion_frame = pd.DataFrame(exclusions)
    overview = {
        "days": len(selected),
        "positive_days": int((selected["gross"] > 0).sum()),
        "negative_days": int((selected["gross"] < 0).sum()),
        "zero_days": int((selected["gross"] == 0).sum()),
        "gross_cumulative": float((1 + selected["gross"]).prod() - 1),
        "net_cumulative": float((1 + selected["net"]).prod() - 1),
        "average_cash_ratio": float(selected["cash_ratio"].mean()),
        "risk_blocked_days": int((selected["risk_blocked"] > 0).sum()),
        "limit_buy_blocked_days": int((selected["limit_buy_blocked"] > 0).sum()),
    }

    output = ANALYSIS / f"fixed_event_guard_decomposition_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True)
    top.to_csv(output / "largest_return_days.csv", index=False)
    exclusion_frame.to_csv(output / "best_day_exclusion.csv", index=False)
    (output / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    display_top = top[
        ["datetime", "instrument", "name", "gross", "turnover", "net"]
    ].head(10).copy()
    for column in ("gross", "net"):
        display_top[column] = display_top[column].map(lambda value: f"{value:.2%}")
    display_exclusion = exclusion_frame.copy()
    for column in ("remaining_net_cumulative", "removed_positive_sum_share"):
        display_exclusion[column] = display_exclusion[column].map(
            lambda value: f"{value:.2%}"
        )
    report = f"""# Fixed Fold3 event_guard 90.68%归因

```json
{json.dumps(overview, ensure_ascii=False, indent=2)}
```

## 最大盈利日

{display_top.to_markdown(index=False)}

## 删除最好交易日后的累计收益

{display_exclusion.to_markdown(index=False)}

这是看过历史数据后形成的Fixed组件和event_guard组合，并使用同日完整K线过滤，
只能作为探索性历史结果，不能当作严格前向收益。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
