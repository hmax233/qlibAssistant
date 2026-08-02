"""Utilities for ordered return-threshold classification experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_THRESHOLDS = (-0.03, 0.0, 0.03, 0.07)


def validate_thresholds(thresholds: Iterable[float]) -> np.ndarray:
    """Return strictly increasing finite thresholds as a float array."""

    values = np.asarray(tuple(thresholds), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("thresholds must all be finite")
    if np.any(np.diff(values) <= 0):
        raise ValueError("thresholds must be strictly increasing")
    return values


def make_threshold_targets(returns, thresholds=DEFAULT_THRESHOLDS) -> np.ndarray:
    """Create one binary target ``return > threshold`` for every threshold."""

    values = np.asarray(returns, dtype=float).reshape(-1)
    cuts = validate_thresholds(thresholds)
    return (values[:, None] > cuts[None, :]).astype(np.uint8)


def project_monotonic(probabilities) -> np.ndarray:
    """Project cumulative probabilities onto p(y>t1) >= ... >= p(y>tk).

    The cumulative-minimum projection is deterministic and conservative: a
    rarer, higher-threshold event can never receive a larger probability than
    a lower-threshold event for the same sample.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (samples, thresholds)")
    clipped = np.clip(values, 0.0, 1.0)
    return np.minimum.accumulate(clipped, axis=1)


def cumulative_to_class_probabilities(probabilities) -> np.ndarray:
    """Convert cumulative threshold probabilities into ordered class probs.

    For four thresholds this yields five classes: below the first threshold,
    the three intervals between adjacent thresholds, and above the last one.
    """

    cumulative = project_monotonic(probabilities)
    result = np.empty((cumulative.shape[0], cumulative.shape[1] + 1), dtype=float)
    result[:, 0] = 1.0 - cumulative[:, 0]
    if cumulative.shape[1] > 1:
        result[:, 1:-1] = cumulative[:, :-1] - cumulative[:, 1:]
    result[:, -1] = cumulative[:, -1]
    result = np.clip(result, 0.0, 1.0)
    totals = result.sum(axis=1, keepdims=True)
    return np.divide(result, totals, out=np.zeros_like(result), where=totals > 0)


def monotonic_violation_rate(probabilities) -> float:
    """Fraction of rows with at least one increasing cumulative probability."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (samples, thresholds)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        return 0.0
    return float(np.any(np.diff(values, axis=1) > 0, axis=1).mean())


def class_representatives(thresholds=DEFAULT_THRESHOLDS) -> np.ndarray:
    """Return finite return proxies for the open-ended ordered classes."""

    cuts = validate_thresholds(thresholds)
    if cuts.size == 1:
        width = max(abs(float(cuts[0])), 0.03)
        return np.asarray([cuts[0] - width, cuts[0] + width], dtype=float)
    gaps = np.diff(cuts)
    return np.concatenate(
        [
            [cuts[0] - gaps[0]],
            (cuts[:-1] + cuts[1:]) / 2.0,
            [cuts[-1] + gaps[-1]],
        ]
    )
