from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from roll.alpha360_cross_stock import (
    Alpha360CrossStockTransformer,
    Alpha360TransformerConfig,
    HORIZON_NAMES,
)
from script.materialize_alpha360_joint_checkpoints import (
    REPORT_COLUMNS,
    evaluate_test,
    materialize_selection,
    sha256,
)
from script.select_alpha360_probabilistic_ensemble import (
    evaluate as evaluate_ensemble,
    prediction_key_sha256,
    select as select_ensemble,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_npy(path: Path, value: np.ndarray) -> str:
    np.save(path, value)
    return sha256(path)


def make_fixture(tmp_path: Path, include_test_files: bool = True) -> tuple[Path, Path]:
    data = tmp_path / "data"
    run = tmp_path / "run"
    data.mkdir()
    run.mkdir()
    stock_ids = {"SH600001": 1, "SZ000002": 2}
    write_json(data / "stock_ids.json", stock_ids)
    write_json(run / "stock_ids.json", stock_ids)
    np.savez(
        data / "normalizer.npz",
        mean=np.zeros(360, dtype="float32"),
        std=np.ones(360, dtype="float32"),
        count=np.full(360, 2, dtype="int64"),
    )
    parts = []
    for split, date in (("selection_valid", "2026-01-02"), ("test", "2026-02-18")):
        prefix = f"part_000_{split}"
        files = {
            f"{prefix}_features.npy": np.zeros((2, 360), dtype="float32"),
            f"{prefix}_labels.npy": np.array(
                [[0.01, 0.02, 0.03], [-0.01, 0.01, -0.02]], dtype="float32"
            ),
            f"{prefix}_stock_ids.npy": np.array([1, 2], dtype="int64"),
            f"{prefix}_offsets.npy": np.array([0, 2], dtype="int64"),
        }
        hashes = {}
        for filename, value in files.items():
            if split != "test" or include_test_files:
                hashes[filename] = write_npy(data / filename, value)
            else:
                hashes[filename] = "0" * 64
        parts.append({"prefix": prefix, "split": split, "dates": [date], "sha256": hashes})
    segments = {
        "train": ["2015-04-17", "2025-04-16"],
        "valid": ["2025-04-17", "2025-09-16"],
        "selection_valid": ["2025-09-17", "2026-02-16"],
        "test": ["2026-02-17", "2026-07-17"],
    }
    manifest = {
        "schema_version": 1,
        "segments": segments,
        "stock_count": 2,
        "parts": parts,
        "normalizer_sha256": sha256(data / "normalizer.npz"),
        "stock_ids_sha256": sha256(data / "stock_ids.json"),
    }
    write_json(data / "manifest.json", manifest)
    config = Alpha360TransformerConfig(
        model_width=4,
        temporal_layers=1,
        cross_section_layers=1,
        attention_heads=1,
        feedforward_width=8,
        stock_embedding_width=2,
        output_head_width=4,
        dropout=0.0,
        target_scale=100.0,
    )
    configuration = {
        "model": asdict(config),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "segments": segments,
        "seed": 20260827,
        "epochs": 50,
        "early_stopping": False,
        "date_batch_size": 4,
        "learning_rate": 0.0003,
        "min_learning_rate": 0.000001,
        "warmup_epochs": 3,
        "warmup_start_factor": 1 / 3,
        "feature_count": 360,
        "target_scale": 100.0,
        "model_code_sha256": "model-code-test-hash",
        "autocast_dtype": "torch.bfloat16",
    }
    write_json(run / "configuration.json", configuration)
    for index, horizon in enumerate(HORIZON_NAMES):
        model = Alpha360CrossStockTransformer(2, config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # The selected horizon gets a checkpoint-specific log mean.
            model.distribution_head[-1].bias[:3].fill_(float(index + 1))
        torch.save(
            {
                "model": model.state_dict(),
                "configuration": configuration,
                "epoch": index + 10,
                "selection_metric": f"{horizon}_rank_ic",
                "selection_value": 0.1 + index,
            },
            run / f"best_{horizon}_rank_ic_model.pt",
        )
    return run, data


def expected_columns(horizons: tuple[str, ...] | list[str]) -> list[str]:
    columns = ["datetime", "instrument"]
    for horizon in HORIZON_NAMES:
        if horizon in horizons:
            columns.extend(f"{horizon}_{suffix}" for suffix in REPORT_COLUMNS)
    return columns


def test_selection_uses_a_distinct_checkpoint_for_each_horizon_and_audits_hashes(
    tmp_path: Path,
) -> None:
    run, data = make_fixture(tmp_path)
    output = tmp_path / "candidate"
    audit = materialize_selection(run, data, output, threads=1)
    frame = pd.read_csv(output / "selection_valid_predictions.csv")

    assert list(frame.columns) == expected_columns(HORIZON_NAMES)
    assert audit["prediction_output"]["sha256"] == sha256(
        output / "selection_valid_predictions.csv"
    )
    paths = [audit["horizon_checkpoints"][name]["path"] for name in HORIZON_NAMES]
    hashes = [audit["horizon_checkpoints"][name]["sha256"] for name in HORIZON_NAMES]
    assert len(set(paths)) == 4
    assert len(set(hashes)) == 4
    assert [audit["horizon_checkpoints"][name]["epoch"] for name in HORIZON_NAMES] == [
        10,
        11,
        12,
        13,
    ]
    # Each checkpoint has a distinct constant leg mean; selected columns prove
    # that the materializer did not accidentally reuse one loaded model.
    assert frame.loc[0, "open1_close2_log_mean"] == pytest.approx(0.03)
    assert frame.loc[0, "close1_open2_log_mean"] == pytest.approx(0.02)
    assert frame.loc[0, "open1_open2_log_mean"] == pytest.approx(0.06)
    assert frame.loc[0, "close1_close2_log_mean"] == pytest.approx(0.08)
    persisted = json.loads((output / "materialization_manifest.json").read_text())
    assert persisted["test_read"] is False
    assert persisted["data_manifest"]["sha256"] == sha256(data / "manifest.json")


def test_selection_never_accesses_test_partition(tmp_path: Path) -> None:
    run, data = make_fixture(tmp_path, include_test_files=False)
    output = tmp_path / "selection-only"
    materialize_selection(run, data, output, threads=1)
    assert (output / "selection_valid_predictions.csv").is_file()


def test_evaluate_test_rejects_candidate_not_selected_before_test_access(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.json"
    write_json(
        manifest,
        {
            "selection_split": "selection_valid",
            "test_files_read": False,
            "candidates": {"joint": str(tmp_path / "missing-candidate")},
            "input_sha256": {},
            "selections": {
                horizon: {"selected_components": ["some-other-candidate"]}
                for horizon in HORIZON_NAMES
            },
        },
    )
    with pytest.raises(RuntimeError, match="not selected for any horizon"):
        evaluate_test(
            tmp_path / "missing-run",
            tmp_path / "missing-data",
            tmp_path / "must-not-exist",
            manifest,
            "joint",
            threads=1,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_evaluate_test_reads_only_manifest_selected_horizons_and_preserves_format(
    tmp_path: Path,
) -> None:
    run, data = make_fixture(tmp_path)
    candidate = tmp_path / "joint-candidate"
    materialize_selection(run, data, candidate, threads=1)
    candidate_csv = candidate / "selection_valid_predictions.csv"
    selected = ["open1_close2", "close1_close2"]
    manifest = tmp_path / "frozen_selection.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "selection_split": "selection_valid",
            "test_files_read": False,
            "candidates": {"joint": str(candidate)},
            "input_sha256": {
                "joint:selection_valid_predictions": sha256(candidate_csv),
            },
            "selections": {
                horizon: {
                    "selected_components": ["joint"] if horizon in selected else ["other"]
                }
                for horizon in HORIZON_NAMES
            },
        },
    )
    audit = evaluate_test(run, data, candidate, manifest, "joint", threads=1)
    frame = pd.read_csv(candidate / "test_predictions.csv")
    assert list(frame.columns) == expected_columns(selected)
    assert audit["selected_horizons"] == selected
    assert set(audit["horizon_checkpoints"]) == set(selected)
    assert audit["selection_manifest"]["sha256"] == sha256(manifest)
    assert audit["prediction_output"]["sha256"] == sha256(
        candidate / "test_predictions.csv"
    )
    assert (candidate / "selection_valid_predictions.csv").is_file()
    assert (candidate / "materialization_manifest.json").is_file()
    assert (candidate / "test_materialization_manifest.json").is_file()


def test_existing_selector_evaluate_reads_test_from_same_candidate_directory(
    tmp_path: Path,
) -> None:
    run, data = make_fixture(tmp_path)
    candidate = tmp_path / "joint-candidate"
    materialize_selection(run, data, candidate, threads=1)
    candidate_csv = candidate / "selection_valid_predictions.csv"
    # Give the tiny deterministic fixture a nonconstant, correctly ordered
    # Selection score so the real selector can freeze the E0 candidate.
    frame = pd.read_csv(candidate_csv)
    for horizon in HORIZON_NAMES:
        variance = frame[f"{horizon}_log_variance"]
        frame[f"{horizon}_log_mean"] = (
            np.log1p(frame[f"{horizon}_actual_return"]) - 0.5 * variance
        )
        frame[f"{horizon}_expected_return"] = frame[f"{horizon}_actual_return"]
    frame.to_csv(candidate_csv, index=False)
    candidate_audit_path = candidate / "materialization_manifest.json"
    candidate_audit = json.loads(candidate_audit_path.read_text())
    candidate_audit["prediction_output"]["sha256"] = sha256(candidate_csv)
    write_json(candidate_audit_path, candidate_audit)

    protocol = tmp_path / "protocol.json"
    configuration = json.loads((run / "configuration.json").read_text())
    write_json(protocol, {
        "segments": configuration["segments"],
        "data_manifest_sha256": configuration["data_manifest_sha256"],
        "horizons": list(HORIZON_NAMES),
        "experiments": [{"id": "joint"}],
        "optimization": {
            "epochs": configuration["epochs"],
            "early_stopping": configuration["early_stopping"],
            "learning_rate": configuration["learning_rate"],
            "minimum_learning_rate": configuration["min_learning_rate"],
            "warmup_epochs": configuration["warmup_epochs"],
            "warmup_start_factor": configuration["warmup_start_factor"],
            "date_batch_size": configuration["date_batch_size"],
            "seed": configuration["seed"],
            "target_scale": configuration["target_scale"],
        },
        "selection_scoring": {
            "signal_start": str(pd.to_datetime(frame["datetime"]).min().date()),
            "signal_end": str(pd.to_datetime(frame["datetime"]).max().date()),
            "expected_days": int(pd.to_datetime(frame["datetime"]).nunique()),
            "expected_rows": int(len(frame)),
            "canonical_key_sha256": prediction_key_sha256(frame),
        },
    })
    manifest = tmp_path / "selection" / "frozen_selection.json"
    select_ensemble(protocol, {"joint": candidate}, manifest)
    evaluate_test(run, data, candidate, manifest, "joint", threads=1)
    ensemble_output = tmp_path / "ensemble-evaluation"
    # The tiny fixture intentionally emits a constant score across two stocks;
    # suppress the corresponding synthetic Spearman diagnostic warning only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        evaluate_ensemble(manifest, ensemble_output)
    assert (ensemble_output / "test_predictions.csv").is_file()
    assert set(pd.read_csv(ensemble_output / "test_summary.csv")["horizon"]) == set(
        HORIZON_NAMES
    )


def test_materialization_never_overwrites_an_existing_output(tmp_path: Path) -> None:
    run, data = make_fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        materialize_selection(run, data, output, threads=1)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_evaluate_test_rejects_checkpoint_changed_after_selection(tmp_path: Path) -> None:
    run, data = make_fixture(tmp_path)
    candidate = tmp_path / "candidate"
    materialize_selection(run, data, candidate, threads=1)
    candidate_csv = candidate / "selection_valid_predictions.csv"
    manifest = tmp_path / "selection.json"
    write_json(
        manifest,
        {
            "selection_split": "selection_valid",
            "test_files_read": False,
            "candidates": {"joint": str(candidate)},
            "input_sha256": {
                "joint:selection_valid_predictions": sha256(candidate_csv),
            },
            "selections": {
                horizon: {"selected_components": ["joint"]} for horizon in HORIZON_NAMES
            },
        },
    )
    checkpoint = run / "best_open1_close2_rank_ic_model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed-after-selection")
    with pytest.raises(RuntimeError, match="Checkpoint changed after selection"):
        evaluate_test(run, data, candidate, manifest, "joint", threads=1)
    assert not (candidate / "test_predictions.csv").exists()
    assert not (candidate / "test_materialization_manifest.json").exists()


@pytest.mark.parametrize(
    "existing_name", ["test_predictions.csv", "test_materialization_manifest.json"]
)
def test_evaluate_test_refuses_existing_test_artifact_before_reading_test(
    tmp_path: Path, existing_name: str
) -> None:
    run, data = make_fixture(tmp_path, include_test_files=False)
    candidate = tmp_path / "candidate"
    materialize_selection(run, data, candidate, threads=1)
    candidate_csv = candidate / "selection_valid_predictions.csv"
    manifest = tmp_path / "selection.json"
    write_json(
        manifest,
        {
            "selection_split": "selection_valid",
            "test_files_read": False,
            "candidates": {"joint": str(candidate)},
            "input_sha256": {
                "joint:selection_valid_predictions": sha256(candidate_csv),
            },
            "selections": {
                horizon: {"selected_components": ["joint"]} for horizon in HORIZON_NAMES
            },
        },
    )
    existing = candidate / existing_name
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing test artifact"):
        evaluate_test(run, data, candidate, manifest, "joint", threads=1)
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_cpu_bf16_selection_inference_is_supported(tmp_path: Path) -> None:
    run, data = make_fixture(tmp_path)
    output = tmp_path / "cpu-bf16"
    audit = materialize_selection(run, data, output, bf16=True, threads=1)
    assert audit["device"] == "cpu"
    assert audit["bf16"] is True
    assert pd.read_csv(output / "selection_valid_predictions.csv").shape[0] == 2
