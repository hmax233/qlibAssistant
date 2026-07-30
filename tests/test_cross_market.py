from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / "script" / "cross_market"
sys.path.insert(0, str(MODULE_DIR))

from common import alpha158_frame, factor_columns  # noqa: E402
from train_global_then_a import daily_metrics  # noqa: E402


def sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=rows)
    close = pd.Series(np.linspace(10.0, 20.0, rows))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "adjclose": close,
            "volume": np.linspace(1_000_000, 2_000_000, rows),
        }
    )


def test_alpha158_layout_and_label_semantics() -> None:
    factors = alpha158_frame(sample_ohlcv(), symbol="TEST", market="US")
    assert len(factor_columns(factors)) == 158
    expected = factors["next_sell_close"] / factors["next_buy_close"] - 1
    pd.testing.assert_series_equal(
        factors["label_abs"], expected, check_names=False
    )
    assert factors["symbol"].eq("TEST").all()
    assert factors["market"].eq("US").all()


def test_market_flags_are_not_counted_as_alpha_factors() -> None:
    factors = alpha158_frame(sample_ohlcv(), symbol="TEST", market="A")
    factors["MARKET_A"] = 1.0
    assert len(factor_columns(factors)) == 158


def test_strict_topk_keeps_blocked_slot_in_cash_without_fallback() -> None:
    rows = []
    for date in pd.bdate_range("2026-01-01", periods=3):
        for rank in range(20):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{rank:02d}",
                    "label_abs": 0.10 if rank == 0 else 0.001,
                    "buy_blocked_proxy": rank == 0,
                    "sell_blocked_proxy": False,
                }
            )
    frame = pd.DataFrame(rows)
    score = np.tile(np.arange(20, 0, -1), 3)
    metrics = daily_metrics(frame, score)
    assert metrics["Top1_buy_blocked_slots"] == 3
    assert metrics["Top1_gross_cumulative"] == 0.0
    assert metrics["Top1_ideal_net_cumulative"] > 0.30
