from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from script.select_alpha360_probabilistic_ensemble import (
    exact_mixture_nll,
    mixture_predictions,
    select,
)


HORIZONS = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")


def candidate_frame(signal: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3)
    result = pd.DataFrame({"datetime": dates, "instrument": ["A", "B", "C"] * 2})
    for horizon in HORIZONS:
        result[f"{horizon}_log_mean"] = signal
        result[f"{horizon}_log_variance"] = 0.01
        result[f"{horizon}_expected_return"] = np.exp(signal + 0.005) - 1
        result[f"{horizon}_probability_positive"] = 0.5
        result[f"{horizon}_actual_return"] = actual
    return result


def write_run(path: Path, signal: np.ndarray, actual: np.ndarray) -> None:
    path.mkdir()
    frame = candidate_frame(signal, actual)
    frame.to_csv(path / "selection_valid_predictions.csv", index=False)
    frame.to_csv(path / "test_predictions.csv", index=False)


def test_mixture_uses_law_of_total_variance_and_exact_mixture_nll() -> None:
    source = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01"]), "instrument": ["A"],
        "one__mean": [0.0], "one__variance": [1.0], "one__positive": [0.5],
        "two__mean": [2.0], "two__variance": [1.0], "two__positive": [0.9],
        "actual_return": [0.0],
    })
    result = mixture_predictions(source, ["one", "two"])
    assert result.loc[0, "log_mean"] == 1.0
    assert result.loc[0, "log_variance"] == 2.0
    assert result.loc[0, "probability_positive"] == pytest.approx(0.7)
    expected_density = 0.5 * (1 / np.sqrt(2 * np.pi)) + 0.5 * (
        np.exp(-2.0) / np.sqrt(2 * np.pi)
    )
    assert exact_mixture_nll(source, ["one", "two"])[0] == pytest.approx(-np.log(expected_density))


def test_selection_never_reads_test_before_manifest(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    good, weak = tmp_path / "good", tmp_path / "weak"
    write_run(good, actual, actual)
    write_run(weak, actual[::-1], actual)
    # If select accidentally opens test, this invalid CSV must make it fail.
    (good / "test_predictions.csv").write_text("forbidden", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"horizons": list(HORIZONS)}), encoding="utf-8")
    output = tmp_path / "selection.json"
    select(protocol, {"good": good, "weak": weak}, output)
    manifest = json.loads(output.read_text())
    assert manifest["test_files_read"] is False
    assert all(value["selected_components"] == ["good"] for value in manifest["selections"].values())
    selection_predictions = pd.read_csv(tmp_path / "selection_valid_ensemble_predictions.csv")
    assert all(f"{horizon}_expected_return" in selection_predictions for horizon in HORIZONS)


def test_selection_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        select(tmp_path / "missing.json", {}, output)


def test_selection_supports_single_horizon_candidate_files(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    shared, single = tmp_path / "shared", tmp_path / "single"
    write_run(shared, actual * 0.5, actual)
    single.mkdir()
    frame = candidate_frame(actual, actual)
    columns = [column for column in frame if column in {"datetime", "instrument"}
               or column.startswith("close1_close2_")]
    frame[columns].to_csv(single / "selection_valid_predictions.csv", index=False)
    frame[columns].to_csv(single / "test_predictions.csv", index=False)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"horizons": list(HORIZONS)}), encoding="utf-8")
    output = tmp_path / "selection.json"
    select(protocol, {"shared": shared, "single": single}, output)
    manifest = json.loads(output.read_text())
    assert "single" not in manifest["selections"]["open1_close2"]["individual_metrics"]
    assert "single" in manifest["selections"]["close1_close2"]["individual_metrics"]
