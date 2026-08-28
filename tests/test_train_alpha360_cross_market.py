from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest
import torch

from roll.alpha360_cross_stock import HORIZON_NAMES
from script.train_alpha360_cross_market import (
    FORMAL_EPOCHS,
    TRAIN_SPLITS,
    CrossMarketDateStore,
    Trainer,
    build_learning_rate_scheduler,
    file_hash,
    merge_prediction_columns,
    pad_cross_market_date_batches,
    parse_args,
    validate_frozen_selection_manifest,
)


def test_merge_prediction_columns_keeps_identical_cross_market_metadata() -> None:
    first = pd.DataFrame({
        "datetime": ["2026-01-05", "2026-01-05"],
        "instrument": ["SH600000", "SZ000001"],
        "us_asof_date": ["2026-01-02", "2026-01-02"],
        "open1_close2_expected_return": [0.01, 0.02],
    })
    second = pd.DataFrame({
        "datetime": ["2026-01-05", "2026-01-05"],
        "instrument": ["SH600000", "SZ000001"],
        "us_asof_date": ["2026-01-02", "2026-01-02"],
        "close1_open2_expected_return": [0.03, 0.04],
    })
    merged = merge_prediction_columns([first, second])
    assert merged.columns.tolist().count("us_asof_date") == 1
    assert merged["us_asof_date"].eq("2026-01-02").all()
    assert "open1_close2_expected_return" in merged
    assert "close1_open2_expected_return" in merged


def test_merge_prediction_columns_rejects_metadata_or_key_mismatch() -> None:
    first = pd.DataFrame({
        "datetime": ["2026-01-05"],
        "instrument": ["SH600000"],
        "us_asof_date": ["2026-01-02"],
        "open1_close2_expected_return": [0.01],
    })
    metadata_mismatch = pd.DataFrame({
        "datetime": ["2026-01-05"],
        "instrument": ["SH600000"],
        "us_asof_date": ["2026-01-01"],
        "close1_open2_expected_return": [0.02],
    })
    with pytest.raises(ValueError, match="shared prediction metadata differs"):
        merge_prediction_columns([first, metadata_mismatch])

    key_mismatch = metadata_mismatch.assign(
        instrument="SZ000001", us_asof_date="2026-01-02"
    )
    with pytest.raises(ValueError, match="do not have identical keys"):
        merge_prediction_columns([first, key_mismatch])


def _write_npy(path: Path, values: np.ndarray) -> str:
    np.save(path, values, allow_pickle=False)
    return file_hash(path)


def _source(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def make_cross_store(root: Path) -> Path:
    root.mkdir(parents=True)
    a_source = root / "a_source"
    a_source.mkdir()
    a_normalizer = root / "a_normalizer.npz"
    us_normalizer = root / "us_normalizer.npz"
    np.savez(a_normalizer, mean=np.zeros(360, dtype="float32"), std=np.ones(360, dtype="float32"))
    np.savez(us_normalizer, mean=np.zeros(360, dtype="float32"), std=np.ones(360, dtype="float32"))
    dictionaries = root / "stock_dictionaries.json"
    dictionaries.write_text(
        json.dumps({"a": {"SH600000": 1, "SZ000001": 2}, "us": {"SPY": 1}}),
        encoding="utf-8",
    )

    dates = {
        "train": "2025-01-02",
        "valid": "2025-01-03",
        "selection_valid": "2025-01-06",
        "test": "2025-01-07",
    }
    parts = []
    for number, (split, date) in enumerate(dates.items()):
        prefix = f"part_{number:03d}_{split}"
        a_values = {
            "features": np.full((2, 360), number + 1, dtype="float32"),
            "labels": np.asarray(
                [[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]], dtype="float32"
            ),
            "stock_ids": np.asarray([1, 2], dtype="int32"),
            "offsets": np.asarray([0, 2], dtype="int64"),
        }
        references = {}
        for name, values in a_values.items():
            path = a_source / f"{prefix}_{name}.npy"
            _write_npy(path, values)
            references[name] = _source(path)
        us_values = {
            "a_dates": np.asarray([date], dtype="datetime64[D]"),
            "a_signal_times_utc": np.asarray(
                [f"{date}T07:00:00"], dtype="datetime64[ns]"
            ),
            "us_asof_dates": np.asarray(
                [(pd.Timestamp(date) - pd.Timedelta(days=1)).date()], dtype="datetime64[D]"
            ),
            "us_close_times_utc": np.asarray(
                [pd.Timestamp(date) - pd.Timedelta(hours=12)], dtype="datetime64[ns]"
            ),
            "us_features": np.full((1, 360), number + 0.5, dtype="float32"),
            "us_stock_ids": np.asarray([1], dtype="int32"),
            "us_offsets": np.asarray([0, 1], dtype="int64"),
        }
        hashes = {
            f"{prefix}_{name}.npy": _write_npy(root / f"{prefix}_{name}.npy", values)
            for name, values in us_values.items()
        }
        parts.append(
            {
                "prefix": prefix,
                "split": split,
                "dates": [date],
                "a_day_indices": [0],
                "a_reference": references,
                "sha256": hashes,
            }
        )

    manifest = {
        "schema_version": 2,
        "status": "complete",
        "input_fingerprint": "synthetic-e6-store",
        "segments": {name: [date, date] for name, date in dates.items()},
        "feature_layout": {"width": 360},
        "alignment": {
            "same_calendar_day_us_close_allowed": False,
            "strictly_prior_close_required": True,
            "comparison_timezone": "UTC",
        },
        "normalizers": {
            "independent_markets": True,
            "a": "a_normalizer.npz",
            "us": {"path": "us_normalizer.npz", "sha256": file_hash(us_normalizer)},
        },
        "a_source": {
            "normalizer_output": {
                "path": str(a_normalizer.resolve()),
                "sha256": file_hash(a_normalizer),
            }
        },
        "stock_dictionaries": {
            "path": "stock_dictionaries.json",
            "sha256": file_hash(dictionaries),
            "a_count": 2,
            "us_count": 1,
        },
        "parts": parts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_padding_is_a_true_variable_size_date_batch_with_two_masks() -> None:
    batches = [
        {
            "a_features": np.ones((2, 360), dtype="float32"),
            "a_labels": np.ones((2, 3), dtype="float32"),
            "a_stock_ids": np.asarray([1, 2]),
            "us_features": np.ones((3, 360), dtype="float32"),
            "us_stock_ids": np.asarray([1, 2, 3]),
        },
        {
            "a_features": np.ones((4, 360), dtype="float32"),
            "a_labels": np.ones((4, 3), dtype="float32"),
            "a_stock_ids": np.asarray([3, 4, 5, 6]),
            "us_features": np.empty((0, 360), dtype="float32"),
            "us_stock_ids": np.asarray([], dtype="int64"),
        },
    ]
    padded = pad_cross_market_date_batches(batches)
    assert padded["a_features"].shape == (2, 4, 360)
    assert padded["us_features"].shape == (2, 3, 360)
    assert padded["a_mask"].tolist() == [[True, True, False, False], [True] * 4]
    assert padded["us_mask"].tolist() == [[True, True, True], [False, False, False]]
    assert np.isnan(padded["a_labels"][0, 2:]).all()
    assert not padded["us_features"][1].any()


def test_padding_rejects_empty_a_dates_and_mismatched_rows() -> None:
    empty_a = {
        "a_features": np.empty((0, 360), dtype="float32"),
        "a_labels": np.empty((0, 3), dtype="float32"),
        "a_stock_ids": np.empty((0,), dtype="int64"),
        "us_features": np.ones((1, 360), dtype="float32"),
        "us_stock_ids": np.asarray([1]),
    }
    with pytest.raises(ValueError, match="at least one|must contain positive"):
        pad_cross_market_date_batches([empty_a])

    mismatched = dict(empty_a)
    mismatched["a_features"] = np.ones((2, 360), dtype="float32")
    mismatched["a_labels"] = np.ones((1, 3), dtype="float32")
    mismatched["a_stock_ids"] = np.asarray([1, 2])
    with pytest.raises(ValueError, match="labels/IDs"):
        pad_cross_market_date_batches([mismatched])


def test_training_scoped_verification_and_iteration_never_touch_test(tmp_path: Path) -> None:
    store = CrossMarketDateStore(make_cross_store(tmp_path / "store"))
    store.verify_parts(TRAIN_SPLITS)
    assert all("test" not in prefix for prefix in store._verified_parts)
    assert store.accessed_splits == []
    rows = list(store.iterate("selection_valid"))
    assert len(rows) == 1
    assert rows[0]["a_features"].shape == (2, 360)
    assert rows[0]["us_features"].shape == (1, 360)
    assert np.datetime64(rows[0]["us_asof_date"]) < np.datetime64(rows[0]["date"])
    assert _test_split_not_accessed(store)


def _test_split_not_accessed(store: CrossMarketDateStore) -> bool:
    return "test" not in store.accessed_splits and all(
        "test" not in prefix for prefix in store._verified_parts
    )


def test_protocol_defaults_use_benchmarked_batch_and_lower_cosine_floor(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["train", "--data", str(tmp_path / "data"), "--output", str(tmp_path / "run")]
    )
    assert FORMAL_EPOCHS == 50
    assert args.learning_rate == pytest.approx(3e-4)
    assert args.minimum_learning_rate == pytest.approx(1e-7)
    assert args.warmup_epochs == 3
    assert args.date_batch_size == 12
    with pytest.raises(SystemExit):
        parse_args(
            [
                "train",
                "--data",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "run"),
                "--learning-rate",
                "0.001",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "train",
                "--data",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "run"),
                "--warmup-epochs",
                "4",
            ]
        )


def test_schedule_uses_three_warmup_epochs_and_eta_min_during_epoch_50(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["train", "--data", str(tmp_path / "data"), "--output", str(tmp_path / "run")]
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
    scheduler = build_learning_rate_scheduler(torch, optimizer, args)
    used = []
    for epoch in range(FORMAL_EPOCHS):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if epoch + 1 < FORMAL_EPOCHS:
            scheduler.step()
    assert used[:4] == pytest.approx([1e-4, 2e-4, 3e-4, 3e-4])
    assert used[-1] == pytest.approx(1e-7)
    assert min(used) == pytest.approx(1e-7)


def test_training_source_has_one_update_per_batch_and_no_forbidden_operations() -> None:
    source = Path(__file__).parents[1].joinpath(
        "script/train_alpha360_cross_market.py"
    ).read_text(encoding="utf-8")
    assert "clip_grad_norm_" not in source
    assert "max_norm" not in source
    assert source.count("self.scaler.step(self.optimizer)") == 1
    assert source.count("self.scaler.scale(loss).backward()") == 1
    assert "scaled_loss" not in source
    assert "len(usable) / self.args.date_batch_size" not in source
    assert "group_date_batches(iterator, self.args.date_batch_size)" in source


def _selection_frame(path: Path) -> None:
    payload = {"datetime": ["2025-01-06"], "instrument": ["SH600000"]}
    for horizon in HORIZON_NAMES:
        payload[f"{horizon}_log_mean"] = [0.01]
        payload[f"{horizon}_log_variance"] = [0.001]
        payload[f"{horizon}_expected_return"] = [0.011]
        payload[f"{horizon}_probability_positive"] = [0.7]
        payload[f"{horizon}_actual_return"] = [0.02]
    pd.DataFrame(payload).to_csv(path, index=False)


def make_selection_manifest(
    path: Path, candidate: Path, candidate_name: str, selected: tuple[str, ...]
) -> Path:
    prediction = candidate / "selection_valid_predictions.csv"
    _selection_frame(prediction)
    selections = {
        horizon: {
            "selected_components": [candidate_name] if horizon in selected else ["another"]
        }
        for horizon in HORIZON_NAMES
    }
    body = {
        "schema_version": 1,
        "selection_split": "selection_valid",
        "test_files_read": False,
        "candidates": {candidate_name: str(candidate.resolve()), "another": "/tmp/another"},
        "input_sha256": {
            f"{candidate_name}:selection_valid_predictions": file_hash(prediction)
        },
        "selections": selections,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_frozen_manifest_is_validated_before_test_and_pins_selection_hash(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest_path = make_selection_manifest(
        tmp_path / "selection.json", candidate, "e6", ("close1_open2",)
    )
    _, selected = validate_frozen_selection_manifest(manifest_path, "e6", candidate)
    assert selected == ("close1_open2",)

    with (candidate / "selection_valid_predictions.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_frozen_selection_manifest(manifest_path, "e6", candidate)


def test_evaluate_test_reads_and_exports_only_selected_horizons(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    selected = ("open1_open2",)
    manifest_path = make_selection_manifest(tmp_path / "selection.json", candidate, "e6", selected)
    manifest, selected = validate_frozen_selection_manifest(manifest_path, "e6", candidate)

    run_configuration = {
        "model": {"target_scale": 100.0},
        "model_invariants": {"independent_a_us_temporal_encoders": True},
        "data_manifest_sha256": "data-hash",
        "data_input_fingerprint": "fingerprint",
        "segments": {"test": ["2025-01-07", "2025-01-07"]},
        "epochs": 50,
        "date_batch_size": 4,
        "gradient_accumulation": False,
        "gradient_clipping": False,
        "learning_rate": 3e-4,
        "minimum_learning_rate": 1e-6,
        "warmup_epochs": 3,
        "model_code_sha256": "model-code-hash",
    }
    (candidate / "configuration.json").write_text(
        json.dumps(run_configuration), encoding="utf-8"
    )
    checkpoint = candidate / "best_open1_open2_rank_ic_model.pt"
    torch.save(
        {
            "model": {},
            "epoch": 17,
            "selection_metric": "open1_open2_rank_ic",
            "configuration": run_configuration,
        },
        checkpoint,
    )
    (candidate / "status.json").write_text(
        json.dumps(
            {
                "status": "selection_ready",
                "test_read": False,
                "checkpoint_sha256": {"open1_open2": file_hash(checkpoint)},
            }
        ),
        encoding="utf-8",
    )
    (candidate / "selection_candidate_manifest.json").write_text(
        json.dumps({"status": "selection_complete"}), encoding="utf-8"
    )

    class StoreSpy:
        def __init__(self):
            self.verified: list[tuple[str, ...]] = []

        def verify_parts(self, splits):
            self.verified.append(tuple(splits))

    class ModelSpy:
        def __init__(self):
            self.loaded = 0

        def load_state_dict(self, state):
            assert state == {}
            self.loaded += 1

    trainer = Trainer.__new__(Trainer)
    trainer.args = types.SimpleNamespace(output=candidate)
    trainer.torch = torch
    trainer.device = torch.device("cpu")
    trainer.store = StoreSpy()
    trainer.model = ModelSpy()
    trainer.configuration = run_configuration
    calls: list[tuple[str, str]] = []

    def fake_evaluate(self, split, *, collect, only_horizon, max_days=None):
        del max_days
        calls.append((split, only_horizon))
        assert collect is True
        daily = pd.DataFrame(
            {"datetime": ["2025-01-07"], f"{only_horizon}_rank_ic": [0.1]}
        )
        predictions = pd.DataFrame(
            {
                "datetime": ["2025-01-07"],
                "instrument": ["SH600000"],
                f"{only_horizon}_log_mean": [0.01],
                f"{only_horizon}_log_variance": [0.001],
                f"{only_horizon}_expected_return": [0.011],
                f"{only_horizon}_return_std": [0.03],
                f"{only_horizon}_probability_positive": [0.7],
                f"{only_horizon}_actual_return": [0.02],
            }
        )
        return {f"{only_horizon}_rank_ic": 0.1}, daily, predictions

    trainer.evaluate_current = types.MethodType(fake_evaluate, trainer)
    trainer.evaluate_test(manifest_path, "e6", manifest, selected)

    assert trainer.store.verified == [("test",)]
    assert calls == [("test", "open1_open2")]
    assert trainer.model.loaded == 1
    columns = set(pd.read_csv(candidate / "test_predictions.csv", nrows=0).columns)
    assert "open1_open2_log_mean" in columns
    assert "close1_close2_log_mean" not in columns
    access = json.loads((candidate / "test_access.json").read_text(encoding="utf-8"))
    assert access["horizons"] == ["open1_open2"]
    assert access["selection_manifest_sha256"] == file_hash(manifest_path)
    completion = json.loads((candidate / "test_completion_audit.json").read_text())
    assert completion["status"] == "test_complete"
    assert completion["test_read"] is True


def test_cpu_benchmark_smoke_records_protocol_and_never_accesses_test(
    tmp_path: Path,
) -> None:
    data = make_cross_store(tmp_path / "store")
    output = tmp_path / "run"
    args = parse_args(
        [
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--benchmark-only",
            "--benchmark-days",
            "1",
            "--threads",
            "1",
        ]
    )
    trainer = Trainer(args, command="train")
    trainer.train()
    configuration = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    assert configuration["epochs"] == 50
    assert configuration["early_stopping"] is False
    assert configuration["date_batch_size"] == 12
    assert configuration["gradient_accumulation"] is False
    assert configuration["gradient_clipping"] is False
    assert configuration["optimizer"] == "AdamW"
    assert configuration["learning_rate"] == pytest.approx(3e-4)
    assert configuration["minimum_learning_rate"] == pytest.approx(1e-7)
    assert configuration["warmup_epochs"] == 3
    assert configuration["model_invariants"]["independent_a_us_temporal_encoders"] is True
    assert configuration["model_invariants"]["decoupled_gaussian_heads"] == list(HORIZON_NAMES)
    assert _test_split_not_accessed(trainer.store)
    assert (output / "benchmark.json").is_file()
