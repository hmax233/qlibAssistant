#!/usr/bin/env python3
"""Audit saved daily Top1 predictions after their labels become observable."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from evaluate_hard_risk_filters import RULES, passes_rule
from report_mainboard_matrix import benchmark_returns


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".qlibAssistant" / "analysis"
CALENDAR = Path.home() / ".qlib/qlib_data/cn_data/calendars/day.txt"
LIMITS = Path(
    "/Users/hmax/investment_data/supplemental/stk_limit/stk_limit_all.parquet"
)
COMMISSION_ONE_SIDE = 0.000235

SOURCE_SPECS = {
    "XGBoost240": {
        "glob": "recorder_prediction_*",
        "files": (
            "ranking_ex_star_chinext.csv",
            "ranking_ex_star.csv",
            "ranking.csv",
        ),
    },
    "FixedArchitecture": {
        "glob": "fixed_ensemble_*",
        "files": (
            "ensemble_ranking_ex_star_chinext.csv",
            "ensemble_ranking_ex_star.csv",
            "ensemble_ranking.csv",
        ),
    },
    "Mainboard20": {
        "glob": "mainboard_top20_ensemble_*",
        "files": (
            "ensemble_ranking_ex_star_chinext.csv",
            "ensemble_ranking_ex_star.csv",
            "ensemble_ranking.csv",
        ),
    },
}


def is_mainboard(code: str) -> bool:
    return str(code).startswith(
        ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")
    )


def choose_csv(folder: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = folder / name
        if path.exists():
            return path
    return None


def infer_signal_date(folder: Path, frame: pd.DataFrame) -> pd.Timestamp | None:
    configs = list(folder.glob("*config*.json")) + list(folder.glob("metadata.json"))
    for path in configs:
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("signal_date")
        except (json.JSONDecodeError, OSError):
            value = None
        if value:
            return pd.Timestamp(value)
    for column in ("datetime", "signal_date", "date"):
        if column in frame and frame[column].notna().any():
            return pd.Timestamp(frame[column].dropna().iloc[0])
    suffix = folder.name.rsplit("_", 2)
    if len(suffix) >= 2 and len(suffix[-2]) == 8 and suffix[-2].isdigit():
        return pd.Timestamp(suffix[-2])
    return None


def discover_top1() -> pd.DataFrame:
    latest: dict[tuple[str, pd.Timestamp], dict] = {}
    for source, spec in SOURCE_SPECS.items():
        for folder in sorted(ANALYSIS.glob(spec["glob"])):
            path = choose_csv(folder, spec["files"])
            if path is None:
                continue
            frame = pd.read_csv(path)
            if frame.empty or "instrument" not in frame:
                continue
            signal_date = infer_signal_date(folder, frame)
            if signal_date is None or signal_date < pd.Timestamp("2026-07-17"):
                continue
            eligible = frame[frame["instrument"].map(is_mainboard)].copy()
            if eligible.empty:
                continue
            rank_column = next(
                (
                    column
                    for column in (
                        "rank",
                        "ensemble_rank",
                        "model_rank",
                    )
                    if column in eligible
                ),
                None,
            )
            if rank_column:
                eligible = eligible.sort_values(rank_column)
            top = eligible.iloc[0]
            key = (source, signal_date)
            latest[key] = {
                "source": source,
                "signal_date": signal_date,
                "instrument": str(top["instrument"]),
                "name": str(top.get("name", "")),
                "original_rank": int(top[rank_column]) if rank_column else 1,
                "ranking_file": str(path),
                "output_folder": folder.name,
            }
    return pd.DataFrame(latest.values()).sort_values(["signal_date", "source"])


def load_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(candidates["instrument"].unique())
    expressions = [
        "Ref($change,-1)",
        "Ref($close,-1)",
        "Ref($close,-2)",
        "Ref($close,-2)/Ref($close,-1)-1",
        "Ref(Sum($change<=-0.095,10),-1)",
        "Ref($close,-1)/Ref(Max($close,20),-1)-1",
        "Ref($close,-1)/Ref($open,-1)-1",
        "Ref($low,-1)/$close-1",
    ]
    data = D.features(
        symbols,
        expressions,
        start_time=candidates["signal_date"].min(),
        end_time=candidates["signal_date"].max(),
        freq="day",
    )
    data.columns = [
        "buy_day_change",
        "buy_close",
        "sell_close",
        "gross_return",
        "limit_down_count_10d",
        "drawdown_20d",
        "close_open_return",
        "low_vs_prev_close",
    ]
    if set(data.index.names) == {"datetime", "instrument"}:
        data = data.reorder_levels(["datetime", "instrument"]).sort_index()
    return data


def calendar_dates(signal: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    calendar = pd.to_datetime(
        [line.strip() for line in CALENDAR.read_text().splitlines() if line.strip()]
    )
    positions = np.flatnonzero(calendar == signal)
    if len(positions) != 1 or positions[0] + 2 >= len(calendar):
        return None, None
    return calendar[positions[0] + 1], calendar[positions[0] + 2]


def limit_lookup(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    limits = pd.read_parquet(
        LIMITS,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    return limits.set_index(["date", "symbol"]).sort_index()


def main() -> None:
    qlib.init(
        provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"),
        region=REG_CN,
    )
    candidates = discover_top1()
    features = load_candidate_features(candidates)
    latest_data = pd.Timestamp(
        CALENDAR.read_text(encoding="utf-8").splitlines()[-1].strip()
    )
    limits = limit_lookup(candidates["signal_date"].min(), latest_data)
    benchmarks = benchmark_returns(
        str(candidates["signal_date"].min().date()),
        str(candidates["signal_date"].max().date()),
    )

    rows = []
    for item in candidates.itertuples(index=False):
        buy_date, sell_date = calendar_dates(item.signal_date)
        mature = sell_date is not None and sell_date <= latest_data
        try:
            values = features.loc[(item.signal_date, item.instrument)]
        except KeyError:
            values = pd.Series(dtype=float)
        gross = values.get("gross_return", np.nan) if mature else np.nan
        buy_close = values.get("buy_close", np.nan) if mature else np.nan
        sell_close = values.get("sell_close", np.nan) if mature else np.nan
        try:
            buy_limit = limits.loc[(buy_date, item.instrument)] if mature else None
        except KeyError:
            buy_limit = None
        try:
            sell_limit = limits.loc[(sell_date, item.instrument)] if mature else None
        except KeyError:
            sell_limit = None
        buy_blocked = bool(
            mature
            and buy_limit is not None
            and pd.notna(values.get("buy_day_change", np.nan))
            and float(values["buy_day_change"])
            >= float(buy_limit["up_return"]) - 1e-4
        )
        sell_blocked = bool(
            mature
            and sell_limit is not None
            and pd.notna(gross)
            and float(gross) <= float(sell_limit["down_return"]) + 1e-4
        )
        guard_values = pd.Series(
            {
                "trade_close": buy_close,
                "trade_change": values.get("buy_day_change", np.nan),
                "limit_down_count_10d": values.get(
                    "limit_down_count_10d", np.nan
                ),
                "drawdown_20d": values.get("drawdown_20d", np.nan),
                "close_open_return": values.get("close_open_return", np.nan),
                "low_vs_prev_close": values.get("low_vs_prev_close", np.nan),
            }
        )
        event_pass = bool(
            mature and passes_rule(guard_values, RULES["event_guard"])
        )
        if not mature or pd.isna(gross):
            baseline_net = np.nan
            event_net = np.nan
        elif buy_blocked:
            baseline_net = 0.0
            event_net = 0.0
        else:
            cost = COMMISSION_ONE_SIDE * (1 if sell_blocked else 2)
            baseline_net = float(gross) - cost
            event_net = baseline_net if event_pass else 0.0
        row = {
            **item._asdict(),
            "buy_date": buy_date,
            "sell_date": sell_date,
            "latest_data_date": latest_data,
            "mature": mature,
            "gross_return": gross,
            "baseline_net_return": baseline_net,
            "event_guard_pass": event_pass if mature else np.nan,
            "event_guard_net_return": event_net,
            "buy_limit_blocked": buy_blocked if mature else np.nan,
            "sell_limit_blocked": sell_blocked if mature else np.nan,
        }
        for name, values_series in benchmarks.items():
            row[f"{name}_return"] = values_series.get(item.signal_date, np.nan)
        rows.append(row)
    detail = pd.DataFrame(rows)

    summary_rows = []
    for source, group in detail[detail["mature"]].groupby("source"):
        baseline = group["baseline_net_return"].dropna()
        guarded = group["event_guard_net_return"].dropna()
        summary_rows.append(
            {
                "source": source,
                "mature_predictions": len(group),
                "baseline_win_rate": float((baseline > 0).mean()),
                "baseline_cumulative": float((1 + baseline).prod() - 1),
                "event_guard_passes": int(group["event_guard_pass"].sum()),
                "event_guard_cumulative": float((1 + guarded).prod() - 1),
                "average_net_return": float(baseline.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    output = ANALYSIS / f"recent_live_review_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True)
    detail.to_csv(output / "prediction_realization_detail.csv", index=False)
    summary.to_csv(output / "source_summary.csv", index=False)
    metadata = {
        "latest_local_data": str(latest_data.date()),
        "timing": "signal T; buy T+1 close; sell T+2 close",
        "universe": "first main-board stock in each saved ranking",
        "commission": "0.0235% per completed side; minimum CNY 5 excluded",
        "event_guard_warning": (
            "Exploratory same-bar check using the complete T+1 daily bar."
        ),
        "account_pnl_warning": (
            "This audits model recommendations, not the user's actual fills."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readable_detail = detail[
        [
            "source",
            "signal_date",
            "buy_date",
            "sell_date",
            "instrument",
            "name",
            "mature",
            "baseline_net_return",
            "event_guard_pass",
            "event_guard_net_return",
            "buy_limit_blocked",
            "sell_limit_blocked",
            "CSI1000_return",
            "CSI300_return",
        ]
    ].copy()
    readable_summary = summary.copy()
    percent_columns = [
        "baseline_win_rate",
        "baseline_cumulative",
        "event_guard_cumulative",
        "average_net_return",
    ]
    for column in percent_columns:
        readable_summary[column] = readable_summary[column].map(
            lambda value: f"{value:.2%}"
        )
    for column in (
        "baseline_net_return",
        "event_guard_net_return",
        "CSI1000_return",
        "CSI300_return",
    ):
        readable_detail[column] = readable_detail[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    report = f"""# 近期每日Top1预测兑现审计

- 本地行情截止：{latest_data:%Y-%m-%d}
- 口径：T日信号，T+1收盘买入，T+2收盘卖出
- 股票：每份历史榜单中第一只主板股票
- 成本：正常完成买卖扣0.047%，未计最低5元
- 这是模型推荐的协议收益，不是用户真实账户收益

## 按预测源汇总

{readable_summary.to_markdown(index=False)}

## 每日明细

{readable_detail.to_markdown(index=False)}

`event_guard`使用完整T+1日线检查并假设T+1收盘成交，属于同日理想化研究
条件；预测尚未到T+2收盘时标记为未成熟，不计入汇总。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(detail.to_string(index=False))
    print("\nSummary\n", summary.to_string(index=False))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
