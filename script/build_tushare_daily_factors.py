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

    close = b["close"].combine_first(p["close_qfq"])
    out["cyq_winner_rate"] = c["winner_rate"] / 100.0
    out["cyq_weight_avg_deviation"] = safe_ratio(close, c["weight_avg"]) - 1.0
    out["cyq_cost50_deviation"] = safe_ratio(close, c["cost_50pct"]) - 1.0
    out["cyq_cost90_width"] = safe_ratio(c["cost_95pct"], c["cost_5pct"]) - 1.0
    out["cyq_cost70_width"] = safe_ratio(c["cost_85pct"], c["cost_15pct"]) - 1.0
    out["cyq_position_in_history"] = safe_ratio(close - c["his_low"], c["his_high"] - c["his_low"])

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

    rolling_columns = [
        "db_turnover", "db_free_turnover", "db_volume_ratio",
        "mf_net_amount_ratio", "mf_large_minus_small", "mf_buy_pressure",
        "cyq_winner_rate", "cyq_weight_avg_deviation", "cyq_cost90_width",
        "pro_atr_to_close", "pro_mfi", "pro_dmi_adx", "pro_vr",
    ]
    rolling_features = {}
    for column in rolling_columns:
        for window in (5, 20):
            rolling = out[column].rolling(window, min_periods=max(3, window // 2))
            rolling_features[f"{column}_mean_{window}d"] = rolling.mean()
            rolling_features[f"{column}_std_{window}d"] = rolling.std()
        rolling_features[f"{column}_change_5d"] = out[column] - out[column].shift(5)
    out = pd.concat([out, pd.DataFrame(rolling_features, index=out.index)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).astype("float32")
    out.insert(0, "instrument", instrument)
    out.index.name = "datetime"
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-cross-sectional-ranks", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    instruments = sorted(path.stem for path in (args.input / "daily_basic").glob("*.parquet"))
    frames, failures = [], []
    for index, instrument in enumerate(instruments, 1):
        try:
            frame = build_symbol(args.input, instrument)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            failures.append({"instrument": instrument, "error": repr(exc)[:500]})
        if index % 25 == 0 or index == len(instruments):
            print(f"factor {index}/{len(instruments)} rows={sum(len(x) for x in frames)} failures={len(failures)}", flush=True)
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
