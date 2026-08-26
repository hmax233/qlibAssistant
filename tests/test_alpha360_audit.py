import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from script.audit_alpha360_cross_stock import MATRIX, NAMES, gaussian_baseline_nll, validate_probability_columns


def example_predictions():
    means = np.asarray([[0.01, -0.02, 0.03], [-0.01, 0.01, 0.02]]) @ MATRIX.T
    covariance = MATRIX @ (np.eye(3) * 0.001) @ MATRIX.T
    frame = pd.DataFrame()
    for i, name in enumerate(NAMES):
        variance = covariance[i, i]
        frame[name + "_log_mean"] = means[:, i]
        frame[name + "_log_variance"] = variance
        frame[name + "_probability_positive"] = ndtr(means[:, i] / np.sqrt(variance))
        frame[name + "_expected_return"] = np.expm1(means[:, i] + variance / 2)
        frame[name + "_return_std"] = np.sqrt(np.expm1(variance) * np.exp(2 * means[:, i] + variance))
    return frame


def test_distribution_output_audit_passes_valid_parameters():
    validate_probability_columns(example_predictions())


def test_distribution_output_audit_rejects_invalid_probability():
    frame = example_predictions()
    frame.loc[0, "open1_close2_probability_positive"] = 1.1
    with pytest.raises(AssertionError):
        validate_probability_columns(frame)


def test_distribution_output_audit_rejects_inconsistent_mean():
    frame = example_predictions()
    frame.loc[0, "open1_close2_expected_return"] += 0.05
    with pytest.raises(AssertionError):
        validate_probability_columns(frame)


def test_constant_gaussian_baseline_nll_matches_standard_normal():
    labels = [np.zeros((2, 3)), np.ones((1, 3))]
    baseline = {"leg_mean": [0, 0, 0], "leg_covariance": np.eye(3).tolist()}
    expected = 0.5 * (3 * np.log(2 * np.pi) + 1.5)
    assert gaussian_baseline_nll(labels, baseline, scale=1.0) == pytest.approx(expected)


def test_prediction_loader_requests_no_prices_after_signal_date():
    from script.predict_alpha360_cross_stock import signal_features

    calendar = pd.bdate_range("2025-01-01", periods=70)
    signal = calendar[63]
    prices = pd.DataFrame(np.arange(1, 421, dtype="float32").reshape(70, 6), index=calendar,
                          columns=["$close", "$open", "$high", "$low", "$vwap", "$volume"])

    class FakeData:
        def calendar(self, future=False):
            assert not future
            return calendar

        def instruments(self, pool):
            return pool

        def list_instruments(self, pool, start_time, end_time, as_list):
            assert start_time == signal and end_time == signal
            return {"SH600000": [(calendar[0], calendar[-1])]}

        def features(self, codes, fields, start_time, end_time):
            assert end_time == signal
            assert all("Ref" not in field for field in fields)
            result = prices.loc[start_time:end_time, fields].copy()
            result["instrument"] = codes[0]
            result.index.name = "datetime"
            return result.reset_index().set_index(["instrument", "datetime"])

    codes, features, valid = signal_features(FakeData(), str(signal.date()))
    assert codes == ["SH600000"]
    assert features.shape == (1, 360)
    assert valid.tolist() == [True]
    first = features.copy()
    prices.loc[calendar[64]:] = -999999
    _, changed, _ = signal_features(FakeData(), str(signal.date()))
    np.testing.assert_array_equal(first, changed)
