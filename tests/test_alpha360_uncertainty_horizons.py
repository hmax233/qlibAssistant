from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from evaluate_alpha360_uncertainty_horizons import (
    benchmark_cumulative,
    find_exit,
    prepare_rule,
)


def prediction_frame():
    return pd.DataFrame({
        "instrument": ["SH600000", "SH600001"],
        "close1_open2_expected_return": [0.02, 0.01],
        "close1_open2_return_std": [0.03, 0.01],
        "close1_open2_probability_positive": [0.54, 0.90],
    })


def test_day_gate_does_not_substitute_a_lower_ranked_candidate():
    frame = prediction_frame()
    assert prepare_rule(frame, "close1_open2", "gate_mean_p55").empty
    candidate_filtered = prepare_rule(frame, "close1_open2", "mean_p55")
    assert candidate_filtered["instrument"].tolist() == ["SH600001"]


def test_find_exit_carries_a_limit_down_position_to_next_open():
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-05", "2026-01-06"]))
    index = pd.MultiIndex.from_product(
        [calendar, ["SH600000"]], names=["trade_date", "instrument"]
    )
    prices = pd.DataFrame({"open": [9.0, 9.5]}, index=index)
    limits = pd.DataFrame({"down_limit": [9.0, 8.1]}, index=index)
    result = find_exit(
        prices, limits, calendar, {date: i for i, date in enumerate(calendar)},
        calendar[0], "SH600000", "open", 0.0, 0.000235, 5.0, 100,
    )
    assert result is not None
    exit_date, sell_price, sell_fee, delayed = result
    assert exit_date == calendar[1]
    assert sell_price == 9.5
    assert sell_fee == 5.0
    assert delayed == 1


def test_overlapping_horizon_benchmark_uses_two_half_capital_sleeves():
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    ]))
    index = pd.DataFrame({
        "datetime": calendar,
        "index": ["CSI1000"] * 4,
        "open": [100.0, 100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 110.0, 120.0],
    })
    result = benchmark_cumulative(index, calendar[:2], calendar, "open1_close2", "CSI1000")
    # Sleeve A earns 10%, sleeve B earns 20%; total initial capital earns 15%.
    assert abs(result - 0.15) < 1e-12
