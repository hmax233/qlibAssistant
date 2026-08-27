from __future__ import annotations

import pandas as pd
import pytest
import torch

from roll.alpha360_cross_stock import HORIZON_NAMES
from script.train_alpha360_decoupled import horizon_targets, merge_prediction_columns


def test_horizon_targets_match_four_price_interval_identities() -> None:
    legs = torch.tensor([[[1.0, 2.0, 3.0]]])
    actual = horizon_targets(legs, HORIZON_NAMES, torch)
    torch.testing.assert_close(actual, torch.tensor([[[6.0, 2.0, 3.0, 5.0]]]))
    close_only = horizon_targets(legs, ("close1_close2",), torch)
    torch.testing.assert_close(close_only, torch.tensor([[[5.0]]]))


def test_merge_prediction_columns_preserves_one_row_per_stock_date() -> None:
    keys = {"datetime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "instrument": ["A", "B"]}
    first = pd.DataFrame({**keys, "one_log_mean": [0.1, 0.2]})
    second = pd.DataFrame({**keys, "two_log_mean": [0.3, 0.4]})
    merged = merge_prediction_columns([first, second])
    assert list(merged.columns) == ["datetime", "instrument", "one_log_mean", "two_log_mean"]
    assert len(merged) == 2


def test_merge_prediction_columns_rejects_overlap_and_misalignment() -> None:
    keys = {"datetime": pd.to_datetime(["2026-01-01"]), "instrument": ["A"]}
    with pytest.raises(ValueError, match="duplicate prediction columns"):
        merge_prediction_columns([
            pd.DataFrame({**keys, "same": [1]}),
            pd.DataFrame({**keys, "same": [2]}),
        ])
    with pytest.raises(ValueError, match="no common rows"):
        merge_prediction_columns([
            pd.DataFrame({**keys, "one": [1]}),
            pd.DataFrame({"datetime": pd.to_datetime(["2026-01-02"]),
                          "instrument": ["A"], "two": [2]}),
        ])
