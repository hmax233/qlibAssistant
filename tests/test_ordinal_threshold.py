import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = ROOT / "roll"
if str(ROLL_DIR) not in sys.path:
    sys.path.insert(0, str(ROLL_DIR))

from ordinal_threshold import (  # noqa: E402
    class_representatives,
    cumulative_to_class_probabilities,
    make_threshold_targets,
    monotonic_violation_rate,
    project_monotonic,
    validate_thresholds,
)


def test_threshold_targets_are_ordered():
    targets = make_threshold_targets([-0.04, -0.01, 0.01, 0.05, 0.10])
    assert targets.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_projection_and_class_conversion_are_valid_probabilities():
    raw = np.array([[0.8, 0.9, 0.4, 0.5], [1.2, 0.7, -0.1, 0.2]])
    assert monotonic_violation_rate(raw) == 1.0
    projected = project_monotonic(raw)
    assert np.all(np.diff(projected, axis=1) <= 0)
    assert monotonic_violation_rate(projected) == 0.0

    classes = cumulative_to_class_probabilities(raw)
    assert classes.shape == (2, 5)
    assert np.all(classes >= 0)
    np.testing.assert_allclose(classes.sum(axis=1), 1.0)


def test_threshold_validation_rejects_duplicates():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_thresholds([-0.03, 0.0, 0.0, 0.07])


def test_class_representatives_cover_open_intervals():
    representatives = class_representatives([-0.03, 0.0, 0.03, 0.07])
    np.testing.assert_allclose(representatives, [-0.06, -0.015, 0.015, 0.05, 0.11])
