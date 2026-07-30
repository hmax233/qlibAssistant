from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / "script" / "cross_market"
sys.path.insert(0, str(MODULE_DIR))

from common import alpha158_frame, factor_columns  # noqa: E402


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
