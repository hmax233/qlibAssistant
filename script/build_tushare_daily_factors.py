#!/usr/bin/env python3
"""Build point-in-time valuation, money-flow, chip and professional factors."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / ".qlibAssistant/supplemental/tushare_daily"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/supplemental/tushare_daily_factors/all_factors.parquet"
EPS = 1e-12


def safe_ratio(left, right):
    return left.astype(float) / right.astype(float).replace(0.0, np.nan)


def read_endpoint(root: Path, endpoint: str, instrument: str) -> pd.DataFrame:
    path = root / endpoint / f"{instrument}.parquet"
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))
    frame = pd.read_parquet(path).drop(columns=["ts_code"], errors="ignore")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()


def build_symbol(root: Path, instrument: str) -> pd.DataFrame:
    basic = read_endpoint(root, "daily_basic", instrument)
    flow = read_endpoint(root, "moneyflow", instrument)
    chips = read_endpoint(root, "cyq_perf", instrument)
    pro = read_endpoint(root, "stk_factor_pro", instrument)
    index = basic.index.union(flow.index).union(chips.index).union(pro.index).sort_values()
    if index.empty:
        return pd.DataFrame()
    b, m, c, p = (value.reindex(index) for value in (basic, flow, chips, pro))
    out = pd.DataFrame(index=index)

    # Valuation, liquidity and size. Percent-valued vendor fields are put on
    # decimal scale; valuation inverses are easier to compare through time.
    out["db_turnover"] = b["turnover_rate"] / 100.0
    out["db_free_turnover"] = b["turnover_rate_f"] / 100.0
    out["db_volume_ratio"] = b["volume_ratio"]
    out["db_log_total_mv"] = np.log1p(b["total_mv"].clip(lower=0))
    out["db_log_circ_mv"] = np.log1p(b["circ_mv"].clip(lower=0))
    out["db_circ_share_ratio"] = safe_ratio(b["float_share"], b["total_share"])
    out["db_free_share_ratio"] = safe_ratio(b["free_share"], b["total_share"])
    out["db_earnings_yield"] = 1.0 / b["pe_ttm"].where(b["pe_ttm"] > 0)
    out["db_book_yield"] = 1.0 / b["pb"].where(b["pb"] > 0)
    out["db_sales_yield"] = 1.0 / b["ps_ttm"].where(b["ps_ttm"] > 0)
    out["db_dividend_yield"] = b["dv_ttm"] / 100.0

    amount_columns = [f"{side}_{size}_amount" for side in ("buy", "sell")
                      for size in ("sm", "md", "lg", "elg")]
    volume_columns = [f"{side}_{size}_vol" for side in ("buy", "sell")
                      for size in ("sm", "md", "lg", "elg")]
    total_amount = m[amount_columns].sum(axis=1, min_count=1)
    total_volume = m[volume_columns].sum(axis=1, min_count=1)
    for size in ("sm", "md", "lg", "elg"):
        out[f"mf_{size}_amount_imbalance"] = safe_ratio(
            m[f"buy_{size}_amount"] - m[f"sell_{size}_amount"], total_amount
        )
        out[f"mf_{size}_volume_imbalance"] = safe_ratio(
            m[f"buy_{size}_vol"] - m[f"sell_{size}_vol"], total_volume
        )
    buy_amount = m[[f"buy_{size}_amount" for size in ("sm", "md", "lg", "elg")]].sum(axis=1)
    out["mf_buy_pressure"] = safe_ratio(buy_amount, total_amount)
    out["mf_net_amount_ratio"] = safe_ratio(m["net_mf_amount"], total_amount)
    out["mf_net_volume_ratio"] = safe_ratio(m["net_mf_vol"], total_volume)
    out["mf_large_minus_small"] = (
        out["mf_lg_amount_imbalance"] + out["mf_elg_amount_imbalance"]
        - out["mf_sm_amount_imbalance"]
    )
    large_buy_amount = m[["buy_lg_amount", "buy_elg_amount"]].sum(axis=1, min_count=1)
    large_sell_amount = m[["sell_lg_amount", "sell_elg_amount"]].sum(axis=1, min_count=1)
    out["mf_large_amount_share"] = safe_ratio(large_buy_amount + large_sell_amount, total_amount)
    out["mf_large_buy_pressure"] = safe_ratio(large_buy_amount, large_buy_amount + large_sell_amount)
    out["mf_elg_minus_lg"] = out["mf_elg_amount_imbalance"] - out["mf_lg_amount_imbalance"]
    out["mf_amount_volume_divergence"] = out["mf_net_amount_ratio"] - out["mf_net_volume_ratio"]
    out["mf_net_amount_to_circ_mv"] = safe_ratio(m["net_mf_amount"], b["circ_mv"])
    out["mf_net_amount_to_free_share"] = safe_ratio(m["net_mf_amount"], b["free_share"])

    close = b["close"].combine_first(p["close_qfq"])
    out["cyq_winner_rate"] = c["winner_rate"] / 100.0
    out["cyq_weight_avg_deviation"] = safe_ratio(close, c["weight_avg"]) - 1.0
    out["cyq_cost50_deviation"] = safe_ratio(close, c["cost_50pct"]) - 1.0
    out["cyq_cost90_width"] = safe_ratio(c["cost_95pct"], c["cost_5pct"]) - 1.0
    out["cyq_cost70_width"] = safe_ratio(c["cost_85pct"], c["cost_15pct"]) - 1.0
    out["cyq_position_in_history"] = safe_ratio(close - c["his_low"], c["his_high"] - c["his_low"])
    for percentile in (5, 15, 50, 85, 95):
        out[f"cyq_close_to_cost{percentile}"] = safe_ratio(close, c[f"cost_{percentile}pct"]) - 1.0
    upper_chip_width = c["cost_95pct"] - c["cost_50pct"]
    lower_chip_width = c["cost_50pct"] - c["cost_5pct"]
    out["cyq_cost_asymmetry"] = safe_ratio(upper_chip_width - lower_chip_width,
                                           upper_chip_width + lower_chip_width)
    out["cyq_core_cost_asymmetry"] = safe_ratio(
        (c["cost_85pct"] - c["cost_50pct"]) - (c["cost_50pct"] - c["cost_15pct"]),
        c["cost_85pct"] - c["cost_15pct"],
    )

    out["pro_atr_to_close"] = safe_ratio(p["atr_qfq"], p["close_qfq"])
    out["pro_asi"] = p["asi_qfq"]
    out["pro_asit"] = p["asit_qfq"]
    for source, target in (
        ("brar_ar_qfq", "pro_brar_ar"), ("brar_br_qfq", "pro_brar_br"),
        ("cr_qfq", "pro_cr"), ("dmi_adx_qfq", "pro_dmi_adx"),
        ("dmi_adxr_qfq", "pro_dmi_adxr"), ("dmi_mdi_qfq", "pro_dmi_mdi"),
        ("dmi_pdi_qfq", "pro_dmi_pdi"), ("mfi_qfq", "pro_mfi"),
        ("psy_qfq", "pro_psy"), ("psyma_qfq", "pro_psyma"),
        ("vr_qfq", "pro_vr"), ("wr_qfq", "pro_wr"), ("wr1_qfq", "pro_wr1"),
    ):
        out[target] = p[source] / 100.0
    for source, target in (
        ("emv_qfq", "pro_emv"), ("maemv_qfq", "pro_maemv"),
        ("mass_qfq", "pro_mass"), ("ma_mass_qfq", "pro_ma_mass"),
        ("trix_qfq", "pro_trix"), ("trma_qfq", "pro_trma"),
    ):
        out[target] = p[source]
    # Scale OBV by current traded volume proxy to avoid a listing-age trend.
    out["pro_obv_scaled"] = safe_ratio(p["obv_qfq"], total_volume.rolling(20, min_periods=5).mean())
    out["pro_dmi_spread"] = out["pro_dmi_pdi"] - out["pro_dmi_mdi"]
    out["pro_dmi_strength"] = out["pro_dmi_adx"] * out["pro_dmi_spread"]
    out["pro_mfi_psy_divergence"] = out["pro_mfi"] - out["pro_psy"]
    out["pro_brar_spread"] = out["pro_brar_ar"] - out["pro_brar_br"]
    out["pro_wr_reversal"] = -(out["pro_wr"] + out["pro_wr1"]) / 2.0

    rolling_columns = [
        "db_turnover", "db_free_turnover", "db_volume_ratio",
        "mf_net_amount_ratio", "mf_large_minus_small", "mf_buy_pressure",
        "cyq_winner_rate", "cyq_weight_avg_deviation", "cyq_cost90_width",
        "mf_large_buy_pressure", "mf_large_amount_share",
        "cyq_cost70_width", "cyq_cost_asymmetry",
        "pro_atr_to_close", "pro_mfi", "pro_dmi_adx", "pro_dmi_spread", "pro_vr",
    ]
    rolling_features = {}
    for column in rolling_columns:
        for window in (5, 20):
            rolling = out[column].rolling(window, min_periods=max(3, window // 2))
            rolling_features[f"{column}_mean_{window}d"] = rolling.mean()
            rolling_features[f"{column}_std_{window}d"] = rolling.std()
        rolling_features[f"{column}_change_5d"] = out[column] - out[column].shift(5)
    out = pd.concat([out, pd.DataFrame(rolling_features, index=out.index)], axis=1)
    # Point-in-time surprises, persistence and economically motivated interactions.
    for column in (
        "mf_net_amount_ratio", "mf_large_minus_small", "mf_elg_amount_imbalance",
        "db_turnover", "db_free_turnover", "db_volume_ratio", "cyq_winner_rate",
    ):
        mean20 = out[column].rolling(20, min_periods=10).mean()
        std20 = out[column].rolling(20, min_periods=10).std().replace(0.0, np.nan)
        out[f"{column}_surprise_20d"] = (out[column] - mean20) / std20
        out[f"{column}_positive_ratio_5d"] = (
            out[column].gt(0).rolling(5, min_periods=3).mean()
        )
    out["mf_net_amount_persistence_3d"] = out["mf_net_amount_ratio"].rolling(3, min_periods=2).sum()
    out["mf_net_amount_persistence_10d"] = out["mf_net_amount_ratio"].rolling(10, min_periods=5).sum()
    out["mf_large_flow_persistence_5d"] = out["mf_large_minus_small"].rolling(5, min_periods=3).sum()
    out["cyq_winner_change_1d"] = out["cyq_winner_rate"].diff()
    out["cyq_winner_change_5d"] = out["cyq_winner_rate"].diff(5)
    out["cyq_width_change_5d"] = out["cyq_cost90_width"].diff(5)
    out["flow_turnover_interaction"] = out["mf_net_amount_ratio"] * out["db_free_turnover"]
    out["large_flow_trend_interaction"] = out["mf_large_minus_small"] * out["pro_dmi_strength"]
    out["crowding_outflow_risk"] = out["cyq_winner_rate"] * (-out["mf_net_amount_ratio"])
    out["chip_breakout_strength"] = out["cyq_close_to_cost85"] * out["mf_large_buy_pressure"]
    out["illiquid_flow_pressure"] = out["mf_net_amount_ratio"] / np.sqrt(
        out["db_free_turnover"].clip(lower=1e-6)
    )
    out = out.replace([np.inf, -np.inf], np.nan).astype("float32")
    out.insert(0, "instrument", instrument)
    out.index.name = "datetime"
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-cross-sectional-ranks", action="store_true")
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="write one raw-factor parquet per instrument; ranks can be computed after feature selection",
    )
    args = parser.parse_args()
    started = time.monotonic()
    instruments = sorted(path.stem for path in (args.input / "daily_basic").glob("*.parquet"))
    frames, failures, rows_written, feature_count = [], [], 0, 0
    for index, instrument in enumerate(instruments, 1):
        try:
            frame = build_symbol(args.input, instrument)
            if not frame.empty:
                feature_count = max(feature_count, len(frame.columns) - 2)
                if args.partitioned:
                    args.output.mkdir(parents=True, exist_ok=True)
                    target = args.output / f"{instrument}.parquet"
                    temporary = target.with_suffix(".tmp.parquet")
                    frame.to_parquet(temporary, index=False, compression="zstd")
                    temporary.replace(target)
                    rows_written += len(frame)
                else:
                    frames.append(frame)
        except Exception as exc:
            failures.append({"instrument": instrument, "error": repr(exc)[:500]})
        if index % 25 == 0 or index == len(instruments):
            current_rows = rows_written if args.partitioned else sum(len(x) for x in frames)
            print(f"factor {index}/{len(instruments)} rows={current_rows} failures={len(failures)}", flush=True)
    if args.partitioned:
        report = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input": str(args.input.resolve()), "output": str(args.output.resolve()),
            "symbols": len(instruments), "rows": rows_written,
            "features": feature_count,
            "partitioned": True, "cross_sectional_ranks": False,
            "failures": failures, "elapsed_seconds": round(time.monotonic() - started, 2),
            "timing": "All date-T fields are published daily market fields known after the T close.",
        }
        report_path = args.output / "build_report_latest.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(f"{len(failures)} instruments failed")
        return
    if not frames:
        raise RuntimeError("no daily supplement factors built")
    result = pd.concat(frames, ignore_index=True).sort_values(["datetime", "instrument"])
    feature_columns = [column for column in result if column not in {"datetime", "instrument"}]
    if not args.skip_cross_sectional_ranks:
        # Same-date ranks are observable after the T close and reduce scale and
        # regime drift. Keep raw transformed values as well for tree models.
        ranks = result.groupby("datetime", sort=False)[feature_columns].rank(pct=True)
        ranks.columns = [f"csrank_{column}" for column in ranks.columns]
        result = pd.concat([result, ranks.astype("float32")], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(args.output)
    features = [column for column in result if column not in {"datetime", "instrument"}]
    missing = result[features].isna().mean().sort_values(ascending=False)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()), "output": str(args.output.resolve()),
        "symbols": int(result["instrument"].nunique()), "rows": len(result),
        "first": result["datetime"].min().date().isoformat(),
        "last": result["datetime"].max().date().isoformat(),
        "features": len(features), "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "highest_missing_rates": missing.head(20).to_dict(),
        "timing": "All date-T fields are published daily market fields known after the T close.",
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} instruments failed")


if __name__ == "__main__":
    main()
