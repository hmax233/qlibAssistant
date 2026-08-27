from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from evaluate_alpha360_overnight import commission, load_prices, mainboard, raw_trade


def test_mainboard_and_minimum_commission():
    assert mainboard("SH600000")
    assert mainboard("SZ000001")
    assert not mainboard("SH688001")
    assert not mainboard("SZ300001")
    assert commission(10_000, 0.000235, 5.0) == 5.0
    assert commission(100_000, 0.000235, 5.0) == 23.5


def test_direct_daily_cache_overrides_reconstructed_prices(tmp_path):
    raw_root, basic_root = tmp_path / "raw", tmp_path / "basic"
    raw_root.mkdir()
    basic_root.mkdir()
    pd.DataFrame({"datetime": ["2026-01-05"], "open": [10.0], "close": [10.0]}).to_parquet(
        raw_root / "SH600000.parquet", index=False
    )
    pd.DataFrame({"trade_date": ["2026-01-05"], "close": [20.0]}).to_parquet(
        basic_root / "SH600000.parquet", index=False
    )
    cache = tmp_path / "daily.parquet"
    pd.DataFrame({
        "trade_date": ["2026-01-05"], "instrument": ["SH600000"],
        "open": [19.5], "close": [20.5],
    }).to_parquet(cache, index=False)
    prices = load_prices({"SH600000"}, raw_root, basic_root, cache)
    row = prices.loc[(pd.Timestamp("2026-01-05"), "SH600000")]
    assert row["raw_open"] == 19.5
    assert row["raw_close"] == 20.5


def test_raw_trade_rejects_exact_up_limit_and_accepts_normal_exit():
    index = pd.MultiIndex.from_tuples([
        (pd.Timestamp("2026-01-05"), "SH600000"),
        (pd.Timestamp("2026-01-06"), "SH600000"),
    ], names=["datetime", "instrument"])
    prices = pd.DataFrame({"raw_open": [10.8, 11.2], "raw_close": [11.0, 11.1]}, index=index)
    limits = pd.DataFrame(
        {"up_limit": [11.0, 12.1], "down_limit": [9.0, 9.9]}, index=index
    )
    trade = raw_trade(prices, limits, pd.Timestamp("2026-01-05"),
                      pd.Timestamp("2026-01-06"), "SH600000")
    assert not trade["buyable"]
    assert trade["sellable"]
    assert trade["reason"] == "buy_at_up_limit"
