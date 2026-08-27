from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from evaluate_alpha360_uncertainty_horizons import (
    benchmark_cumulative,
    commission,
    excludes_star_and_chinext,
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
    # Scheduled open exit is blocked; the chronological state machine retries
    # again at that day's close before succeeding on the next event.
    assert result["blocked_sell_down_limit_attempts"] == 2
    assert result["unresolved_exit"] == 0


def test_flat_market_marked_equity_conserves_cash_after_buy():
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07",
    ]))
    index = pd.MultiIndex.from_product(
        [calendar, ["SH600000"]], names=["trade_date", "instrument"]
    )
    prices = pd.DataFrame({"open": 10.0, "close": 10.0}, index=index)
    limits = pd.DataFrame({"up_limit": 11.0, "down_limit": 9.0}, index=index)
    predictions = pd.DataFrame({
        "datetime": [calendar[0]],
        "instrument": ["SH600000"],
        "close1_close2_expected_return": [0.02],
        "close1_close2_return_std": [0.03],
        "close1_close2_probability_positive": [0.60],
    })
    capital = 100000.0
    rate = 0.000235
    daily, result = simulate(
        predictions, prices, limits, calendar, "close1_close2", "mean_all",
        1, False, 0.0, capital, rate, 5.0,
    )

    shares = 9900
    one_side_fee = commission(shares * 10.0, rate, 5.0)
    assert abs(daily.iloc[0]["equity_mark"] - (capital - one_side_fee)) < 1e-9
    assert abs(result["final_equity"] - (capital - 2.0 * one_side_fee)) < 1e-9
    assert daily["equity_mark"].max() <= capital
    assert result["max_drawdown_marked"] > -0.001


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


def test_fallback_walks_the_complete_ranking_beyond_top50():
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07",
    ]))
    instruments = [f"SH60{number:04d}" for number in range(60)]
    index = pd.MultiIndex.from_product(
        [calendar, instruments], names=["trade_date", "instrument"]
    )
    prices = pd.DataFrame({"open": 10.0, "close": 10.0}, index=index)
    limits = pd.DataFrame({"up_limit": 11.0, "down_limit": 9.0}, index=index)
    # The first 55 ranked names are unbuyable at their exact upper limit.
    prices.loc[(calendar[1], instruments[:55]), "close"] = 11.0
    predictions = pd.DataFrame({
        "datetime": calendar[0],
        "instrument": instruments,
        "close1_close2_expected_return": list(reversed(range(60))),
        "close1_close2_return_std": 0.02,
        "close1_close2_probability_positive": 0.7,
    })
    _, fallback = simulate(
        predictions, prices, limits, calendar, "close1_close2", "mean_all",
        1, True, 0.0, 100000.0, 0.000235, 5.0,
    )
    _, leave_cash = simulate(
        predictions, prices, limits, calendar, "close1_close2", "mean_all",
        1, False, 0.0, 100000.0, 0.000235, 5.0,
    )
    assert fallback["completed_trades"] == 1
    assert fallback["fallback_replacements"] == 1
    assert fallback["blocked_buy_up_limit"] == 55
    assert leave_cash["completed_trades"] == 0
    assert leave_cash["filtered_cash_slots"] == 1


def test_mainboard_variant_excludes_only_star_and_chinext():
    assert not excludes_star_and_chinext("SH688001")
    assert not excludes_star_and_chinext("SZ300001")
    assert excludes_star_and_chinext("SH600000")
    assert excludes_star_and_chinext("SZ000001")
    assert excludes_star_and_chinext("BJ430001")
