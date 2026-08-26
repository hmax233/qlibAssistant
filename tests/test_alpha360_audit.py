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
