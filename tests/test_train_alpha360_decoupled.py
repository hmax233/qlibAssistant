from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from roll.alpha360_cross_stock import HORIZON_NAMES
from script.train_alpha360_decoupled import (
    frozen_file,
    horizon_targets,
    merge_prediction_columns,
    write_selection_candidate_manifest,
)
from script.train_alpha360_cross_stock import file_hash


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


def test_selection_candidate_manifest_freezes_all_required_inputs(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    data = tmp_path / "data"
    output.mkdir()
    data.mkdir()
    data_manifest = data / "manifest.json"
    data_manifest.write_text('{"version": 1}', encoding="utf-8")
    configuration = {
        "model_mode": "single_horizon",
        "horizon_names": ["close1_close2"],
        "segments": {
            "train": ["2015-04-17", "2025-04-16"],
            "valid": ["2025-04-17", "2025-09-16"],
            "selection_valid": ["2025-09-17", "2026-02-16"],
            "test": ["2026-02-17", "2026-07-17"],
        },
        "data_manifest_sha256": file_hash(data_manifest),
    }
    configuration_path = output / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    predictions = output / "selection_valid_predictions.csv"
    summary = output / "selection_valid_summary.csv"
    predictions.write_text("datetime,instrument\n", encoding="utf-8")
    summary.write_text("horizon\n", encoding="utf-8")
    checkpoint = output / "best_close1_close2_rank_ic_model.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    checkpoints = {
        "close1_close2": {
            "path": str(checkpoint.resolve()),
            "sha256": file_hash(checkpoint),
            "epoch": 12,
            "selection_metric": "close1_close2_rank_ic",
            "selection_value": 0.03,
        }
    }
    manifest = write_selection_candidate_manifest(
        output, data, configuration, checkpoints
    )
    persisted = json.loads((output / "selection_candidate_manifest.json").read_text())
    assert persisted == manifest
    assert manifest["status"] == "selection_complete"
    assert manifest["selection_split"] == "selection_valid"
    assert manifest["test_files_read"] is False
    assert manifest["configuration"] == frozen_file(configuration_path)
    assert manifest["data_manifest"] == frozen_file(data_manifest)
    assert manifest["selection_valid_predictions"] == frozen_file(predictions)
    assert manifest["checkpoints"]["close1_close2"]["sha256"] == file_hash(checkpoint)


def test_selection_candidate_manifest_rejects_changed_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    data = tmp_path / "data"
    output.mkdir()
    data.mkdir()
    data_manifest = data / "manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    configuration = {
        "segments": {},
        "data_manifest_sha256": file_hash(data_manifest),
    }
    (output / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
    (output / "selection_valid_predictions.csv").write_text("x\n", encoding="utf-8")
    (output / "selection_valid_summary.csv").write_text("x\n", encoding="utf-8")
    checkpoint = output / "best_close1_close2_rank_ic_model.pt"
    checkpoint.write_bytes(b"current")
    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        write_selection_candidate_manifest(
            output,
            data,
            configuration,
            {"close1_close2": {
                "path": str(checkpoint),
                "sha256": "0" * 64,
                "epoch": 1,
                "selection_metric": "close1_close2_rank_ic",
                "selection_value": 0.01,
            }},
        )
