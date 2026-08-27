#!/usr/bin/env python3
"""Train and materialize E6 cross-market Alpha360 predictions.

``train`` always runs the fixed 50-epoch protocol.  It selects one checkpoint
per horizon on ``valid`` and exports only ``selection_valid`` predictions.
The held-out Test arrays are not verified or opened by that command.

``evaluate-test`` first validates an already frozen probabilistic-ensemble
selection manifest.  Only horizons for which this E6 candidate was selected
are then materialized from Test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roll.alpha360_cross_market import (  # noqa: E402
    Alpha360CrossMarketConfig,
    Alpha360CrossMarketTransformer,
)
from roll.alpha360_cross_stock import HORIZON_MATRIX, HORIZON_NAMES  # noqa: E402
from roll.alpha360_decoupled import (  # noqa: E402
    distribution_report,
    independent_gaussian_nll,
)
from script.train_alpha360_cross_stock import (  # noqa: E402
    file_hash,
    group_date_batches,
    replace_with_retry,
    write_json,
)
from script.train_alpha360_decoupled import (  # noqa: E402
    frozen_file,
    write_selection_candidate_manifest,
)


FORMAL_EPOCHS = 50
TRAIN_SPLITS = ("train", "valid", "selection_valid")
TEST_SPLIT = "test"
SCHEDULER_DESCRIPTION = (
    "3 training epochs of linear warmup, then cosine annealing; "
    "the 50th training epoch uses eta_min"
)


def build_learning_rate_scheduler(torch_module, optimizer, args):
    """Build the fixed 50-epoch schedule with eta_min used during epoch 50."""

    # LinearLR exposes its initial start_factor immediately.  With three
    # warmup training epochs, two transitions produce start -> midpoint ->
    # base LR over epochs 1, 2, and 3.
    warmup = torch_module.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup_epochs - 1,
    )
    # Epoch 4 is cosine step 0 and epoch 50 is step 46.  This endpoint choice
    # ensures eta_min is an LR actually used for training, not merely reached
    # by a scheduler.step() after all optimization has finished.
    cosine = torch_module.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FORMAL_EPOCHS - args.warmup_epochs - 1,
        eta_min=args.minimum_learning_rate,
    )
    return torch_module.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    replace_with_retry(temporary, destination)


def horizon_targets(leg_labels, torch_module):
    matrix = HORIZON_MATRIX.to(device=leg_labels.device, dtype=leg_labels.dtype)
    return torch_module.einsum("...c,hc->...h", leg_labels, matrix)


def merge_prediction_columns(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one horizon prediction frame is required")
    result = frames[0]
    for frame in frames[1:]:
        overlap = (set(result.columns) & set(frame.columns)) - {"datetime", "instrument"}
        if overlap:
            raise ValueError(f"duplicate prediction columns: {sorted(overlap)}")
        result = result.merge(
            frame,
            on=["datetime", "instrument"],
            how="inner",
            validate="one_to_one",
        )
    if result.empty:
        raise ValueError("horizon prediction frames have no common rows")
    return result.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _validated_segment_bounds(manifest: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    expected = (*TRAIN_SPLITS, TEST_SPLIT)
    segments = manifest.get("segments", {})
    if set(segments) != set(expected):
        raise ValueError("E6 store must contain train/valid/selection_valid/test metadata")
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    previous_end: pd.Timestamp | None = None
    for split in expected:
        values = segments[split]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"Invalid E6 segment bounds for {split}")
        start, end = (pd.Timestamp(value).normalize() for value in values)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"Invalid E6 segment bounds for {split}")
        if previous_end is not None and start <= previous_end:
            raise ValueError("E6 segments must be strictly ordered and non-overlapping")
        bounds[split] = (start, end)
        previous_end = end
    return bounds


def _validate_manifest_parts(manifest: dict[str, Any]) -> None:
    bounds = _validated_segment_bounds(manifest)
    prefixes: set[str] = set()
    seen_dates: set[pd.Timestamp] = set()
    split_counts = {split: 0 for split in bounds}
    for part in manifest.get("parts", []):
        prefix = str(part.get("prefix", ""))
        split = part.get("split")
        if not prefix or prefix in prefixes:
            raise ValueError(f"Duplicate or empty E6 part prefix: {prefix!r}")
        prefixes.add(prefix)
        if split not in bounds:
            raise ValueError(f"Unknown E6 part split: {split}")
        dates = [pd.Timestamp(value).normalize() for value in part.get("dates", [])]
        if not dates or dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError(f"E6 part {prefix} dates must be non-empty, sorted, and unique")
        start, end = bounds[split]
        if any(pd.isna(value) or value < start or value > end for value in dates):
            raise ValueError(f"E6 part {prefix} contains dates outside {split}")
        if any(value in seen_dates for value in dates):
            raise ValueError(f"E6 part {prefix} overlaps another split or part")
        seen_dates.update(dates)
        split_counts[split] += len(dates)
    missing = [split for split, count in split_counts.items() if count == 0]
    if missing:
        raise ValueError(f"E6 store has no dates for splits: {missing}")


class CrossMarketDateStore:
    """Lazy mmap reader for the builder's E6 manifest and parts.

    Hash verification is split-scoped.  This is intentional: a training run
    must not read Test bytes merely to hash them.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 2:
            raise ValueError("E6 store schema 2 with UTC as-of timestamps is required")
        if self.manifest.get("status") != "complete":
            raise RuntimeError("E6 cross-market store is not complete")
        if self.manifest.get("feature_layout", {}).get("width") != 360:
            raise ValueError("E6 store is not Alpha360")
        if self.manifest.get("normalizers", {}).get("independent_markets") is not True:
            raise ValueError("E6 store must use independent A/US normalizers")
        alignment = self.manifest.get("alignment", {})
        if (
            alignment.get("same_calendar_day_us_close_allowed") is not False
            or alignment.get("strictly_prior_close_required") is not True
            or alignment.get("comparison_timezone") != "UTC"
        ):
            raise ValueError("E6 store does not enforce strict prior-US-close alignment")
        _validate_manifest_parts(self.manifest)

        dictionaries_path = self.directory / self.manifest["stock_dictionaries"]["path"]
        if file_hash(dictionaries_path) != self.manifest["stock_dictionaries"]["sha256"]:
            raise RuntimeError("Input hash mismatch: stock_dictionaries.json")
        dictionaries = json.loads(dictionaries_path.read_text(encoding="utf-8"))
        self.a_id_to_code = {int(value): key for key, value in dictionaries["a"].items()}
        self.us_id_to_code = {int(value): key for key, value in dictionaries["us"].items()}
        for market, mapping, expected_count in (
            ("A", self.a_id_to_code, self.manifest["stock_dictionaries"]["a_count"]),
            ("US", self.us_id_to_code, self.manifest["stock_dictionaries"]["us_count"]),
        ):
            expected_ids = set(range(1, int(expected_count) + 1))
            if set(mapping) != expected_ids:
                raise ValueError(f"{market} stock IDs must be unique and contiguous from 1")

        a_normalizer = Path(self.manifest["a_source"]["normalizer_output"]["path"])
        if file_hash(a_normalizer) != self.manifest["a_source"]["normalizer_output"]["sha256"]:
            raise RuntimeError("Input hash mismatch: A normalizer")
        us_spec = self.manifest["normalizers"]["us"]
        us_normalizer = self.directory / us_spec["path"]
        if file_hash(us_normalizer) != us_spec["sha256"]:
            raise RuntimeError("Input hash mismatch: US normalizer")
        with np.load(a_normalizer, allow_pickle=False) as values:
            self.a_mean = np.asarray(values["mean"], dtype="float32")
            self.a_std = np.asarray(values["std"], dtype="float32")
        with np.load(us_normalizer, allow_pickle=False) as values:
            self.us_mean = np.asarray(values["mean"], dtype="float32")
            self.us_std = np.asarray(values["std"], dtype="float32")
        if self.a_mean.shape != (360,) or self.us_mean.shape != (360,):
            raise ValueError("A and US normalizers must each contain 360 features")
        self._verified_parts: set[str] = set()
        self.accessed_splits: list[str] = []

    @property
    def a_stock_count(self) -> int:
        return int(self.manifest["stock_dictionaries"]["a_count"])

    @property
    def us_stock_count(self) -> int:
        return int(self.manifest["stock_dictionaries"]["us_count"])

    def _parts(self, split: str) -> list[dict[str, Any]]:
        if split not in {*TRAIN_SPLITS, TEST_SPLIT}:
            raise ValueError(f"Unknown split: {split}")
        return [part for part in self.manifest["parts"] if part["split"] == split]

    def days(self, split: str) -> int:
        return sum(len(part["dates"]) for part in self._parts(split))

    def verify_parts(self, splits: Iterable[str]) -> None:
        for split in splits:
            for part in self._parts(split):
                prefix = part["prefix"]
                if prefix in self._verified_parts:
                    continue
                for name, expected in part["sha256"].items():
                    if file_hash(self.directory / name) != expected:
                        raise RuntimeError(f"Input hash mismatch: {name}")
                for source in part["a_reference"].values():
                    path = Path(source["path"])
                    if file_hash(path) != source["sha256"]:
                        raise RuntimeError(f"Input hash mismatch: {path.name}")
                self._verified_parts.add(prefix)

    @staticmethod
    def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype="float32") - mean) / std
        result[~np.isfinite(result)] = 0.0
        return result

    def iterate(self, split: str, seed: int | None = None, max_days: int | None = None):
        if split not in self.accessed_splits:
            self.accessed_splits.append(split)
        self.verify_parts((split,))
        parts = self._parts(split)
        generator = random.Random(seed)
        if seed is not None:
            generator.shuffle(parts)
        emitted = 0
        for part in parts:
            a_arrays = {
                name: np.load(Path(source["path"]), mmap_mode="r", allow_pickle=False)
                for name, source in part["a_reference"].items()
            }
            us_arrays = {
                name: np.load(
                    self.directory / f"{part['prefix']}_{name}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in (
                    "a_dates",
                    "a_signal_times_utc",
                    "us_asof_dates",
                    "us_close_times_utc",
                    "us_features",
                    "us_stock_ids",
                    "us_offsets",
                )
            }
            day_count = len(part["dates"])
            for name in (
                "a_dates", "a_signal_times_utc", "us_asof_dates", "us_close_times_utc"
            ):
                if us_arrays[name].shape != (day_count,):
                    raise ValueError(
                        f"Invalid {name} shape in {part['prefix']}: {us_arrays[name].shape}"
                    )
            if (
                us_arrays["us_offsets"].shape != (day_count + 1,)
                or int(us_arrays["us_offsets"][0]) != 0
                or int(us_arrays["us_offsets"][-1]) != len(us_arrays["us_stock_ids"])
                or len(us_arrays["us_features"]) != len(us_arrays["us_stock_ids"])
            ):
                raise ValueError(f"Invalid US offsets/rows in {part['prefix']}")
            local_days = list(range(len(part["dates"])))
            if seed is not None:
                generator.shuffle(local_days)
            for local_day in local_days:
                if max_days is not None and emitted >= max_days:
                    return
                source_day = int(part["a_day_indices"][local_day])
                a_begin, a_end = (int(value) for value in a_arrays["offsets"][source_day:source_day + 2])
                us_begin, us_end = (int(value) for value in us_arrays["us_offsets"][local_day:local_day + 2])
                a_date = np.datetime64(us_arrays["a_dates"][local_day], "D")
                us_asof = np.datetime64(us_arrays["us_asof_dates"][local_day], "D")
                a_signal = np.datetime64(
                    us_arrays["a_signal_times_utc"][local_day], "ns"
                )
                us_close = np.datetime64(
                    us_arrays["us_close_times_utc"][local_day], "ns"
                )
                if not np.isnat(us_close) and not us_close < a_signal:
                    raise RuntimeError(f"US as-of leakage in {part['prefix']} day {local_day}")
                if np.isnat(us_asof) != np.isnat(us_close):
                    raise RuntimeError(f"US as-of date/time mismatch in {part['prefix']} day {local_day}")
                if str(a_date) != str(part["dates"][local_day]):
                    raise RuntimeError(f"A date mismatch in {part['prefix']} day {local_day}")
                yield {
                    "date": part["dates"][local_day],
                    "us_asof_date": str(us_asof),
                    "a_signal_time_utc": str(a_signal),
                    "us_close_time_utc": str(us_close),
                    "a_features": self._normalize(
                        np.array(a_arrays["features"][a_begin:a_end]), self.a_mean, self.a_std
                    ),
                    "a_labels": np.array(a_arrays["labels"][a_begin:a_end], dtype="float32"),
                    "a_stock_ids": np.array(a_arrays["stock_ids"][a_begin:a_end], dtype="int64"),
                    "us_features": self._normalize(
                        np.array(us_arrays["us_features"][us_begin:us_end]), self.us_mean, self.us_std
                    ),
                    "us_stock_ids": np.array(
                        us_arrays["us_stock_ids"][us_begin:us_end], dtype="int64"
                    ),
                }
                emitted += 1


def pad_cross_market_date_batches(batches: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("at least one date batch is required")
    required = {
        "a_features", "a_labels", "a_stock_ids", "us_features", "us_stock_ids"
    }
    for row, batch in enumerate(batches):
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"date batch {row} is missing fields: {sorted(missing)}")
        a_features = np.asarray(batch["a_features"])
        a_labels = np.asarray(batch["a_labels"])
        a_ids = np.asarray(batch["a_stock_ids"])
        us_features = np.asarray(batch["us_features"])
        us_ids = np.asarray(batch["us_stock_ids"])
        if a_features.ndim != 2 or a_features.shape[1] != 360:
            raise ValueError(f"date batch {row} A features must have shape [N, 360]")
        if a_labels.shape != (len(a_features), 3) or a_ids.shape != (len(a_features),):
            raise ValueError(f"date batch {row} A labels/IDs do not match A features")
        if len(a_ids) < 1 or np.any(a_ids <= 0):
            raise ValueError(f"date batch {row} must contain positive A stock IDs")
        if us_features.ndim != 2 or us_features.shape[1] != 360:
            raise ValueError(f"date batch {row} US features must have shape [N, 360]")
        if us_ids.shape != (len(us_features),) or np.any(us_ids <= 0):
            raise ValueError(f"date batch {row} US IDs do not match features or are nonpositive")
    batch_size = len(batches)
    max_a = max(len(batch["a_stock_ids"]) for batch in batches)
    max_us = max(1, max(len(batch["us_stock_ids"]) for batch in batches))
    if max_a < 1:
        raise ValueError("each date must contain at least one A-share")
    result = {
        "a_features": np.zeros((batch_size, max_a, 360), dtype="float32"),
        "a_labels": np.full((batch_size, max_a, 3), np.nan, dtype="float32"),
        "a_stock_ids": np.zeros((batch_size, max_a), dtype="int64"),
        "a_mask": np.zeros((batch_size, max_a), dtype=bool),
        "us_features": np.zeros((batch_size, max_us, 360), dtype="float32"),
        "us_stock_ids": np.zeros((batch_size, max_us), dtype="int64"),
        "us_mask": np.zeros((batch_size, max_us), dtype=bool),
    }
    for row, batch in enumerate(batches):
        a_count = len(batch["a_stock_ids"])
        us_count = len(batch["us_stock_ids"])
        result["a_features"][row, :a_count] = batch["a_features"]
        result["a_labels"][row, :a_count] = batch["a_labels"]
        result["a_stock_ids"][row, :a_count] = batch["a_stock_ids"]
        result["a_mask"][row, :a_count] = True
        if us_count:
            result["us_features"][row, :us_count] = batch["us_features"]
            result["us_stock_ids"][row, :us_count] = batch["us_stock_ids"]
            result["us_mask"][row, :us_count] = True
    return result


def validate_frozen_selection_manifest(
    manifest_path: Path,
    candidate_name: str,
    candidate_directory: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_split") != "selection_valid":
        raise RuntimeError("Selection manifest was not frozen from selection_valid")
    if manifest.get("test_files_read") is not False:
        raise RuntimeError("Selection manifest is not a pre-Test frozen manifest")
    candidates = manifest.get("candidates", {})
    if candidate_name not in candidates:
        raise RuntimeError(f"Candidate absent from selection manifest: {candidate_name}")
    expected = Path(candidates[candidate_name]).expanduser().resolve()
    if expected != Path(candidate_directory).expanduser().resolve():
        raise RuntimeError(f"Selection manifest candidate path mismatch: {expected}")
    selected = tuple(
        horizon
        for horizon in HORIZON_NAMES
        if candidate_name in manifest.get("selections", {}).get(horizon, {}).get(
            "selected_components", []
        )
    )
    if not selected:
        raise RuntimeError(f"Candidate {candidate_name} was not selected for any horizon")
    selection_path = expected / "selection_valid_predictions.csv"
    expected_hash = manifest.get("input_sha256", {}).get(
        f"{candidate_name}:selection_valid_predictions"
    )
    if not expected_hash or file_hash(selection_path) != expected_hash:
        raise RuntimeError("Frozen selection prediction hash mismatch")
    return manifest, selected


class Trainer:
    def __init__(self, args: argparse.Namespace, *, command: str):
        # Import qlib before torch to avoid DLL ordering problems in the Windows env.
        import qlib  # noqa: F401
        import torch

        self.args = args
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
        if args.bf16 and self.device.type != "cuda":
            raise RuntimeError("--bf16 requires CUDA")
        if args.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("--bf16 requested but this CUDA device does not support BF16")
        torch.set_num_threads(args.threads)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.store = CrossMarketDateStore(args.data)
        verified_splits = TRAIN_SPLITS if command == "train" else ()
        if verified_splits:
            print(f"Verifying dataset hashes for {verified_splits} (Test remains locked)...", flush=True)
            self.store.verify_parts(verified_splits)
        self.model_config = Alpha360CrossMarketConfig(target_scale=args.target_scale)
        self.model = Alpha360CrossMarketTransformer(
            self.store.a_stock_count,
            self.store.us_stock_count,
            self.model_config,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = build_learning_rate_scheduler(
            torch, self.optimizer, args
        )
        self.amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and not args.bf16
        )
        self.configuration = {
            "experiment_family": "E6 A+US cross-market four-head Gaussian",
            "model": asdict(self.model_config),
            "model_invariants": {
                "independent_a_us_temporal_encoders": True,
                "cross_market_attention": True,
                "a_share_outputs_only": True,
                "decoupled_gaussian_heads": list(HORIZON_NAMES),
            },
            "data_manifest_sha256": file_hash(args.data / "manifest.json"),
            "data_input_fingerprint": self.store.manifest["input_fingerprint"],
            "segments": self.store.manifest["segments"],
            "seed": args.seed,
            "epochs": FORMAL_EPOCHS,
            "early_stopping": False,
            "date_batch_size": args.date_batch_size,
            "target_scale": args.target_scale,
            "gradient_accumulation": False,
            "gradient_clipping": False,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "warmup_start_factor": args.warmup_start_factor,
            "scheduler": SCHEDULER_DESCRIPTION,
            "checkpoint_split": "valid",
            "checkpoint_metric": "per-horizon RankIC",
            "selection_export_split": "selection_valid",
            "test_policy": "locked; train never verifies or opens Test arrays",
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "script_sha256": file_hash(Path(__file__)),
            "model_code_sha256": file_hash(ROOT / "roll/alpha360_cross_market.py"),
            "torch": torch.__version__,
            "autocast_dtype": str(self.amp_dtype) if self.device.type == "cuda" else "float32",
        }

    def tensor_date_batch(self, batches: list[dict[str, Any]]):
        values = pad_cross_market_date_batches(batches)
        torch = self.torch
        tensors = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in values.items()
        }
        targets = horizon_targets(tensors.pop("a_labels"), torch) * self.model_config.target_scale
        return tensors, targets

    def forward(self, tensors: dict[str, Any]):
        with self.torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self.model(
                tensors["a_features"],
                tensors["us_features"],
                tensors["a_stock_ids"],
                tensors["us_stock_ids"],
                tensors["a_mask"],
                tensors["us_mask"],
            )

    def train_epoch(self, epoch: int, max_days: int | None = None):
        torch = self.torch
        self.model.train()
        loss_date_total = 0.0
        loss_date_count = 0
        durations: list[float] = []
        processed = 0
        total_days = min(self.store.days("train"), max_days) if max_days else self.store.days("train")
        iterator = self.store.iterate("train", seed=self.args.seed + epoch, max_days=max_days)
        for batches in group_date_batches(iterator, self.args.date_batch_size):
            usable = [batch for batch in batches if np.isfinite(batch["a_labels"]).any()]
            if not usable:
                continue
            started = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            tensors, targets = self.tensor_date_batch(usable)
            output = self.forward(tensors)
            loss = independent_gaussian_nll(
                targets, output["mean"], output["std"], tensors["a_mask"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training NLL: {[item['date'] for item in usable]}")
            # One optimizer update per real padded date batch.
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            processed += len(usable)
            loss_date_total += float(loss.detach()) * len(usable)
            loss_date_count += len(usable)
            durations.append(time.monotonic() - started)
            if processed % 100 == 0 or processed == total_days:
                status = {
                    "status": "training",
                    "pid": os.getpid(),
                    "epoch": epoch + 1,
                    "processed_dates": processed,
                    "training_dates_total": total_days,
                    "date_batch_size": self.args.date_batch_size,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                write_json(self.args.output / "status.json", status)
                print("PROGRESS " + json.dumps(status), flush=True)
        if not loss_date_count:
            raise RuntimeError("No usable training labels")
        return loss_date_total / loss_date_count, durations

    def evaluate_current(
        self,
        split: str,
        *,
        collect: bool = False,
        only_horizon: str | None = None,
        max_days: int | None = None,
    ):
        torch = self.torch
        self.model.eval()
        names = (only_horizon,) if only_horizon else HORIZON_NAMES
        indices = [HORIZON_NAMES.index(name) for name in names]
        rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        with torch.no_grad():
            for batch in self.store.iterate(split, max_days=max_days):
                tensors, all_targets = self.tensor_date_batch([batch])
                output = self.forward(tensors)
                targets = all_targets[..., indices]
                if not torch.isfinite(targets).any():
                    continue
                mean_scaled = output["mean"][..., indices]
                std_scaled = output["std"][..., indices]
                report = distribution_report(
                    mean_scaled / self.model_config.target_scale,
                    std_scaled / self.model_config.target_scale,
                )
                targets_log = targets / self.model_config.target_scale
                day_row: dict[str, Any] = {
                    "datetime": batch["date"],
                    "us_asof_date": batch["us_asof_date"],
                    "a_stocks": len(batch["a_stock_ids"]),
                    "us_stocks": len(batch["us_stock_ids"]),
                }
                prediction: dict[str, Any] = {
                    "datetime": batch["date"],
                    "instrument": [
                        self.store.a_id_to_code[int(value)] for value in batch["a_stock_ids"]
                    ],
                    "us_asof_date": batch["us_asof_date"],
                }
                for column, name in enumerate(names):
                    actual_log = targets_log[0, :, column].float().cpu().numpy()
                    actual_return = np.expm1(actual_log)
                    mean = report["log_mean"][0, :, column].cpu().numpy()
                    std = report["log_std"][0, :, column].cpu().numpy()
                    expected = report["expected_return"][0, :, column].cpu().numpy()
                    return_std = report["return_std"][0, :, column].cpu().numpy()
                    probability = report["probability_positive"][0, :, column].cpu().numpy()
                    usable = np.isfinite(expected) & np.isfinite(actual_return)
                    rank_ic = (
                        pd.Series(expected[usable]).corr(
                            pd.Series(actual_return[usable]), method="spearman"
                        )
                        if usable.sum() > 1
                        else np.nan
                    )
                    component_nll = independent_gaussian_nll(
                        targets[..., column:column + 1],
                        mean_scaled[..., column:column + 1],
                        std_scaled[..., column:column + 1],
                        tensors["a_mask"],
                    )
                    day_row[f"{name}_rank_ic"] = rank_ic
                    day_row[f"{name}_nll_log_return"] = (
                        float(component_nll) - math.log(self.model_config.target_scale)
                    )
                    day_row[f"{name}_mae"] = float(
                        np.mean(np.abs(expected[usable] - actual_return[usable]))
                    )
                    day_row[f"{name}_brier"] = float(
                        np.mean((probability[usable] - (actual_return[usable] > 0)) ** 2)
                    )
                    if collect:
                        prediction[f"{name}_log_mean"] = mean
                        prediction[f"{name}_log_variance"] = std**2
                        prediction[f"{name}_expected_return"] = expected
                        prediction[f"{name}_return_std"] = return_std
                        prediction[f"{name}_probability_positive"] = probability
                        prediction[f"{name}_actual_return"] = actual_return
                rows.append(day_row)
                if collect:
                    prediction_rows.append(pd.DataFrame(prediction))
        daily = pd.DataFrame(rows)
        if daily.empty:
            raise RuntimeError(f"No usable labels in {split}")
        metrics: dict[str, Any] = {"days": len(daily)}
        for name in names:
            rank_ic = daily[f"{name}_rank_ic"].dropna()
            rank_std = rank_ic.std(ddof=0)
            metrics[f"{name}_rank_ic"] = float(rank_ic.mean())
            metrics[f"{name}_rank_icir"] = (
                float(rank_ic.mean() / rank_std) if rank_std > 0 else None
            )
            for metric in ("nll_log_return", "mae", "brier"):
                metrics[f"{name}_{metric}"] = float(daily[f"{name}_{metric}"].mean())
        predictions = pd.concat(prediction_rows, ignore_index=True) if collect else None
        return metrics, daily, predictions

    def _save_model_payload(self, horizon: str, epoch: int, value: float) -> None:
        destination = self.args.output / f"best_{horizon}_rank_ic_model.pt"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "configuration": self.configuration,
                "epoch": epoch,
                "selection_metric": f"{horizon}_rank_ic",
                "selection_value": value,
            },
            temporary,
        )
        replace_with_retry(temporary, destination)

    def _save_last_checkpoint(
        self,
        epoch: int,
        history: list[dict[str, Any]],
        best_rank_ic: dict[str, float],
    ) -> None:
        torch = self.torch
        destination = self.args.output / "last_checkpoint.pt"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": history,
            "best_rank_ic": best_rank_ic,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }
        torch.save(payload, temporary)
        replace_with_retry(temporary, destination)

    def _load_resume(self):
        old = json.loads((self.args.output / "configuration.json").read_text(encoding="utf-8"))
        immutable = (
            "model",
            "model_invariants",
            "data_manifest_sha256",
            "data_input_fingerprint",
            "segments",
            "seed",
            "epochs",
            "date_batch_size",
            "gradient_accumulation",
            "gradient_clipping",
            "optimizer",
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "warmup_epochs",
            "warmup_start_factor",
            "scheduler",
            "script_sha256",
            "model_code_sha256",
        )
        for key in immutable:
            if old.get(key) != self.configuration.get(key):
                raise RuntimeError(f"Resume configuration mismatch: {key}")
        state = self.torch.load(
            self.args.output / "last_checkpoint.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            self.torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        return state["epoch"], state["history"], state["best_rank_ic"]

    def train(self) -> None:
        args = self.args
        if args.output.exists() and not args.resume:
            permitted = {args.log_file.resolve()} if args.log_file else set()
            unexpected = [path for path in args.output.iterdir() if path.resolve() not in permitted]
            if unexpected:
                raise FileExistsError(
                    f"Output contains an existing run ({unexpected[0]}); use --resume or a new directory"
                )
        args.output.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        first_epoch = 0
        best_rank_ic = {name: -float("inf") for name in HORIZON_NAMES}
        if args.resume:
            first_epoch, history, best_rank_ic = self._load_resume()
        write_json(args.output / "configuration.json", self.configuration)
        print("CONFIGURATION " + json.dumps(self.configuration), flush=True)

        if args.benchmark_only:
            loss, durations = self.train_epoch(0, args.benchmark_days)
            elapsed = float(np.median(durations[2:] or durations))
            estimate = elapsed * math.ceil(self.store.days("train") / args.date_batch_size)
            write_json(
                args.output / "benchmark.json",
                {
                    "train_loss": loss,
                    "date_batch_size": args.date_batch_size,
                    "peak_cuda_memory_bytes": (
                        self.torch.cuda.max_memory_allocated()
                        if self.device.type == "cuda"
                        else None
                    ),
                    "estimated_train_epoch_seconds_excluding_valid": estimate,
                    "estimated_50_epoch_train_hours_excluding_valid": estimate * 50 / 3600,
                },
            )
            return

        for epoch in range(first_epoch, FORMAL_EPOCHS):
            started = time.monotonic()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_loss, _ = self.train_epoch(epoch)
            valid_metrics, _, _ = self.evaluate_current("valid")
            for horizon in HORIZON_NAMES:
                value = valid_metrics[f"{horizon}_rank_ic"]
                if np.isfinite(value) and value > best_rank_ic[horizon]:
                    best_rank_ic[horizon] = value
                    self._save_model_payload(horizon, epoch + 1, value)
            row = {
                "epoch": epoch + 1,
                "train_nll_scaled": train_loss,
                **valid_metrics,
                "learning_rate": learning_rate,
                "epoch_seconds": time.monotonic() - started,
            }
            history.append(row)
            if epoch + 1 < FORMAL_EPOCHS:
                self.scheduler.step()
            atomic_csv(pd.DataFrame(history), args.output / "epoch_metrics.csv")
            self._save_last_checkpoint(epoch + 1, history, best_rank_ic)
            print("EPOCH " + json.dumps(row), flush=True)

        frames: list[pd.DataFrame] = []
        summary: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        checkpoint_freeze: dict[str, dict[str, Any]] = {}
        for horizon in HORIZON_NAMES:
            checkpoint_path = args.output / f"best_{horizon}_rank_ic_model.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"No finite valid RankIC checkpoint for {horizon}")
            checkpoint = self.torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model"])
            metrics, daily, predictions = self.evaluate_current(
                "selection_valid", collect=True, only_horizon=horizon
            )
            atomic_csv(
                daily,
                args.output / f"selection_valid_{horizon}_daily_metrics.csv",
            )
            frames.append(predictions)
            summary.append(
                {
                    "horizon": horizon,
                    "valid_best_epoch": checkpoint["epoch"],
                    "valid_selection_value": checkpoint["selection_value"],
                    **metrics,
                }
            )
            checkpoint_hashes[horizon] = file_hash(checkpoint_path)
            checkpoint_freeze[horizon] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_hashes[horizon],
                "epoch": checkpoint["epoch"],
                "selection_metric": checkpoint["selection_metric"],
                "selection_value": checkpoint["selection_value"],
            }
        selection_predictions = args.output / "selection_valid_predictions.csv"
        selection_summary = args.output / "selection_valid_summary.csv"
        atomic_csv(merge_prediction_columns(frames), selection_predictions)
        atomic_csv(pd.DataFrame(summary), selection_summary)
        candidate_manifest = write_selection_candidate_manifest(
            args.output, args.data, self.configuration, checkpoint_freeze
        )
        candidate_manifest_path = args.output / "selection_candidate_manifest.json"
        write_json(
            args.output / "status.json",
            {
                "status": "selection_ready",
                "epochs": FORMAL_EPOCHS,
                "test_read": False,
                "accessed_splits": self.store.accessed_splits,
                "checkpoint_sha256": checkpoint_hashes,
                "selection_valid_predictions_sha256": file_hash(selection_predictions),
                "selection_valid_summary_sha256": file_hash(selection_summary),
                "selection_candidate_manifest": str(candidate_manifest_path.resolve()),
                "selection_candidate_manifest_sha256": file_hash(candidate_manifest_path),
                "selection_candidate_status": candidate_manifest["status"],
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print("E6 SELECTION_VALID READY; TEST REMAINS LOCKED", flush=True)

    def evaluate_test(
        self,
        manifest_path: Path,
        candidate_name: str,
        manifest: dict[str, Any],
        selected_horizons: tuple[str, ...],
        frozen_candidate: dict[str, Any] | None = None,
    ) -> None:
        # Detect a manifest replacement in the interval between the shared
        # selector validation and this first potential Test access.
        manifest_path = Path(manifest_path).expanduser().resolve()
        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted_manifest != manifest:
            raise RuntimeError("Selection manifest changed after validation")
        destination = self.args.output / "test_predictions.csv"
        summary_path = self.args.output / "test_summary.csv"
        access_path = self.args.output / "test_access.json"
        completion_path = self.args.output / "test_completion_audit.json"
        destinations = [
            destination,
            summary_path,
            access_path,
            completion_path,
            *[
                self.args.output / f"test_{horizon}_daily_metrics.csv"
                for horizon in selected_horizons
            ],
        ]
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Test artifact already exists: {path}")
        status = json.loads((self.args.output / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "selection_ready" or status.get("test_read") is not False:
            raise RuntimeError("Candidate is not in the frozen pre-Test selection-ready state")
        run_configuration = json.loads(
            (self.args.output / "configuration.json").read_text(encoding="utf-8")
        )
        for key in (
            "model",
            "model_invariants",
            "data_manifest_sha256",
            "data_input_fingerprint",
            "segments",
            "epochs",
            "date_batch_size",
            "target_scale",
            "gradient_accumulation",
            "gradient_clipping",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_epochs",
            "model_code_sha256",
        ):
            if run_configuration.get(key) != self.configuration.get(key):
                raise RuntimeError(f"Test configuration mismatch: {key}")
        checkpoints: dict[str, dict[str, Any]] = {}
        checkpoint_hashes: dict[str, str] = {}
        for horizon in selected_horizons:
            checkpoint_path = self.args.output / f"best_{horizon}_rank_ic_model.pt"
            checkpoint_hash = file_hash(checkpoint_path)
            if checkpoint_hash != status["checkpoint_sha256"][horizon]:
                raise RuntimeError(f"Checkpoint hash mismatch: {horizon}")
            if frozen_candidate is not None:
                frozen_reference = frozen_candidate["checkpoints"][horizon]
                expected_hash = manifest["selections"][horizon][
                    "selected_checkpoint_sha256"
                ].get(candidate_name)
                if (
                    Path(frozen_reference["path"]).resolve() != checkpoint_path.resolve()
                    or frozen_reference["sha256"] != checkpoint_hash
                    or expected_hash != checkpoint_hash
                ):
                    raise RuntimeError(f"Frozen checkpoint hash mismatch: {horizon}")
            checkpoint = self.torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if checkpoint.get("selection_metric") != f"{horizon}_rank_ic":
                raise RuntimeError(f"Checkpoint horizon mismatch: {horizon}")
            checkpoint_configuration = checkpoint.get("configuration", {})
            if checkpoint_configuration.get("data_manifest_sha256") != run_configuration.get(
                "data_manifest_sha256"
            ):
                raise RuntimeError(f"Checkpoint data hash mismatch: {horizon}")
            if checkpoint_configuration.get("model") != run_configuration.get("model"):
                raise RuntimeError(f"Checkpoint model configuration mismatch: {horizon}")
            checkpoints[horizon] = checkpoint
            checkpoint_hashes[horizon] = checkpoint_hash

        # Every manifest, run, selection prediction, and selected checkpoint
        # invariant has now been checked. This is the first operation allowed
        # to hash or open Test arrays.
        self.store.verify_parts((TEST_SPLIT,))
        frames: list[pd.DataFrame] = []
        summary: list[dict[str, Any]] = []
        for horizon in selected_horizons:
            checkpoint = checkpoints[horizon]
            self.model.load_state_dict(checkpoint["model"])
            metrics, daily, predictions = self.evaluate_current(
                TEST_SPLIT, collect=True, only_horizon=horizon
            )
            atomic_csv(daily, self.args.output / f"test_{horizon}_daily_metrics.csv")
            frames.append(predictions)
            summary.append(
                {"horizon": horizon, "checkpoint_epoch": checkpoint["epoch"], **metrics}
            )
        atomic_csv(merge_prediction_columns(frames), destination)
        atomic_csv(pd.DataFrame(summary), summary_path)
        write_json(
            access_path,
            {
                "selection_manifest": str(manifest_path),
                "selection_manifest_sha256": file_hash(manifest_path),
                "candidate_name": candidate_name,
                "horizons": list(selected_horizons),
                "test_read": True,
                "test_predictions_sha256": file_hash(destination),
                "test_summary_sha256": file_hash(summary_path),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        artifact_paths = [
            destination,
            summary_path,
            access_path,
            *[
                self.args.output / f"test_{horizon}_daily_metrics.csv"
                for horizon in selected_horizons
            ],
        ]
        completion = {
            "schema_version": 1,
            "status": "test_complete",
            "test_read": True,
            "selection_manifest": frozen_file(manifest_path),
            "candidate_manifest": (
                frozen_candidate["candidate_manifest"] if frozen_candidate is not None
                else frozen_file(self.args.output / "selection_candidate_manifest.json")
            ),
            "candidate_name": candidate_name,
            "selected_horizons": list(selected_horizons),
            "checkpoint_sha256": checkpoint_hashes,
            "artifacts": {path.name: frozen_file(path) for path in artifact_paths},
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(completion_path, completion)
        print(f"E6 TEST MATERIALIZED FOR {selected_horizons}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate-test"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-factor", type=float, default=1 / 3)
    parser.add_argument("--date-batch-size", type=int, default=4)
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-days", type=int, default=16)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "train" and (args.selection_manifest or args.candidate_name):
        parser.error("selection manifest arguments are only valid for evaluate-test")
    if args.command == "evaluate-test" and (
        not args.selection_manifest or not args.candidate_name
    ):
        parser.error("evaluate-test requires --selection-manifest and --candidate-name")
    if args.resume and args.command != "train":
        parser.error("--resume is only valid for train")
    if args.warmup_epochs != 3:
        parser.error("E6 formal protocol fixes warmup at exactly 3 epochs")
    if args.date_batch_size < 1 or args.benchmark_days < 1 or args.threads < 1:
        parser.error("batch size, benchmark days, and threads must be positive")
    if args.learning_rate != 3e-4 or args.minimum_learning_rate != 1e-6:
        parser.error("E6 formal protocol fixes lr=3e-4 and cosine eta_min=1e-6")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.log_file:
        args.log_file = args.log_file.expanduser().resolve()
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.command == "train":
            trainer = Trainer(args, command="train")
            trainer.train()
        else:
            # The shared selector authenticates the protocol, every candidate,
            # Selection predictions, and frozen checkpoints before Test access.
            from script.select_alpha360_probabilistic_ensemble import (
                validate_frozen_selection_manifest as validate_complete_freeze,
            )

            manifest, _, candidates, frozen_candidates = validate_complete_freeze(
                args.selection_manifest, require_test_artifacts=False
            )
            if candidates.get(args.candidate_name) != args.output.resolve():
                raise RuntimeError("Selection manifest candidate path mismatch")
            selected = tuple(
                horizon for horizon in HORIZON_NAMES
                if args.candidate_name in manifest["selections"][horizon]["selected_components"]
            )
            if not selected:
                raise RuntimeError(f"Candidate {args.candidate_name} was not selected")
            trainer = Trainer(args, command="evaluate-test")
            trainer.evaluate_test(
                args.selection_manifest,
                args.candidate_name,
                manifest,
                selected,
                frozen_candidates[args.candidate_name],
            )
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "last_failure.json",
            {"error": repr(error), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        )
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
