from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from evaluate_alpha360_uncertainty_horizons import (
    benchmark_cumulative,
    prepare_rule,
    simulate,
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


def execution_inputs():
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"
    ]))
    index = pd.MultiIndex.from_product(
        [calendar, ["SH600000"]], names=["trade_date", "instrument"]
    )
    prices = pd.DataFrame({
        "open": [10.0, 10.0, 9.0, 9.5],
        "close": [10.0, 10.0, 9.0, 9.5],
    }, index=index)
    limits = pd.DataFrame({
        "up_limit": [11.0, 11.0, 11.0, 11.0],
        "down_limit": [9.0, 9.0, 9.0, 8.1],
    }, index=index)
    predictions = pd.DataFrame({
        "datetime": [calendar[0]], "instrument": ["SH600000"],
        "close1_open2_expected_return": [0.02],
        "close1_open2_return_std": [0.03],
        "close1_open2_probability_positive": [0.60],
    })
    return calendar, prices, limits, predictions


def test_state_machine_carries_limit_down_position_without_future_entry_scan():
    calendar, prices, limits, predictions = execution_inputs()
    daily, result = simulate(
        predictions, prices, limits, calendar, "close1_open2", "mean_all",
        1, False, 0.0, 100000.0, 0.000235, 5.0,
    )
    assert daily.iloc[0]["entries"] == 1
    assert result["completed_trades"] == 1
    assert result["delayed_exit_trades"] == 1
    assert result["blocked_sell_down_limit_attempts"] == 1
    assert result["unresolved_exit"] == 0


def test_unresolved_future_exit_does_not_cancel_the_historical_buy():
    calendar, prices, limits, predictions = execution_inputs()
    limits.loc[(calendar[3], "SH600000"), "down_limit"] = 9.5
    daily, result = simulate(
        predictions, prices, limits, calendar, "close1_open2", "mean_all",
        1, False, 0.0, 100000.0, 0.000235, 5.0,
    )
    assert daily.iloc[0]["entries"] == 1
    assert result["completed_trades"] == 0
    assert result["unresolved_exit"] == 1


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
