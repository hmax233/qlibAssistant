#!/usr/bin/env python3
"""Train E1--E5 Alpha360 decoupled probabilistic experiments.

The ordinary ``train`` command deliberately stops after validation checkpoint
selection and selection-validation prediction export.  Held-out test data can
only be materialized with ``evaluate-test`` and an already frozen ensemble
selection manifest.
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

import numpy as np
import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roll.alpha360_cross_stock import HORIZON_MATRIX, HORIZON_NAMES  # noqa: E402
from roll.alpha360_decoupled import (  # noqa: E402
    Alpha360DecoupledConfig,
    Alpha360DecoupledTransformer,
    distribution_report,
    independent_gaussian_nll,
)
from script.train_alpha360_cross_stock import (  # noqa: E402
    DateStore,
    file_hash,
    group_date_batches,
    pad_date_batches,
    replace_with_retry,
    write_json,
)


def horizon_targets(leg_labels, horizon_names, torch_module):
    matrix = HORIZON_MATRIX.to(device=leg_labels.device, dtype=leg_labels.dtype)
    all_targets = torch_module.einsum("...c,hc->...h", leg_labels, matrix)
    indices = [HORIZON_NAMES.index(name) for name in horizon_names]
    return all_targets[..., indices]


def merge_prediction_columns(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one horizon prediction frame is required")
    result = frames[0]
    for frame in frames[1:]:
        overlap = (set(result.columns) & set(frame.columns)) - {"datetime", "instrument"}
        if overlap:
            raise ValueError(f"duplicate prediction columns: {sorted(overlap)}")
        result = result.merge(frame, on=["datetime", "instrument"], how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("horizon prediction frames have no common rows")
    return result.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def frozen_file(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_hash(path)}


def write_selection_candidate_manifest(
    output: Path,
    data: Path,
    configuration: dict,
    checkpoints: dict[str, dict],
) -> dict:
    """Publish the complete pre-Test freeze for one E1--E5 candidate."""

    output = output.expanduser().resolve()
    configuration_path = output / "configuration.json"
    data_manifest_path = data.expanduser().resolve() / "manifest.json"
    predictions_path = output / "selection_valid_predictions.csv"
    summary_path = output / "selection_valid_summary.csv"
    if configuration != json.loads(configuration_path.read_text(encoding="utf-8-sig")):
        raise RuntimeError("In-memory and persisted candidate configurations disagree")
    if configuration.get("data_manifest_sha256") != file_hash(data_manifest_path):
        raise RuntimeError("Candidate configuration/data manifest hash mismatch")
    normalized_checkpoints: dict[str, dict] = {}
    for horizon, metadata in checkpoints.items():
        path = Path(metadata["path"]).expanduser().resolve()
        reference = frozen_file(path)
        if reference["sha256"] != metadata["sha256"]:
            raise RuntimeError(f"Checkpoint changed before candidate freeze: {horizon}")
        normalized_checkpoints[horizon] = {
            **metadata,
            **reference,
        }
    manifest = {
        "schema_version": 1,
        "status": "selection_complete",
        "candidate_directory": str(output),
        "selection_split": "selection_valid",
        "test_files_read": False,
        "configuration": frozen_file(configuration_path),
        "data_manifest": frozen_file(data_manifest_path),
        "segments": configuration["segments"],
        "selection_valid_predictions": frozen_file(predictions_path),
        "selection_valid_summary": frozen_file(summary_path),
        "checkpoints": normalized_checkpoints,
        "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output / "selection_candidate_manifest.json", manifest)
    return manifest


class Trainer:
    def __init__(self, args):
        # Qlib before torch avoids DLL ordering trouble in the Windows conda env.
        import qlib  # noqa: F401
        import torch

        self.args = args
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
        torch.set_num_threads(args.threads)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.store = DateStore(args.data)
        if args.command == "train":
            print("Verifying authorized pre-Test dataset hashes...", flush=True)
            self.store.verify_parts({"train", "valid", "selection_valid"})
        self.config = Alpha360DecoupledConfig(stock_embedding_width=64, target_scale=args.target_scale)
        self.horizon_names = (HORIZON_NAMES if args.model_mode == "shared_four_head"
                              else (args.horizon,))
        self.model = Alpha360DecoupledTransformer(
            stock_count=self.store.manifest["stock_count"],
            mode=args.model_mode,
            horizon=args.horizon,
            config=self.config,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=args.warmup_start_factor, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
        )
        self.amp_dtype = (torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16
        )
        self.configuration = {
            "experiment_family": "E1-E5 decoupled Gaussian",
            "model_mode": args.model_mode,
            "horizon_names": list(self.horizon_names),
            "model": asdict(self.config),
            "data_manifest_sha256": file_hash(args.data / "manifest.json"),
            "segments": self.store.manifest["segments"],
            "seed": args.seed,
            "epochs": args.epochs,
            "early_stopping": False,
            "date_batch_size": args.date_batch_size,
            "target_scale": args.target_scale,
            "gradient_accumulation": False,
            "gradient_clipping": False,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "warmup_start_factor": args.warmup_start_factor,
            "scheduler": "linear warmup followed by cosine annealing",
            "checkpoint_split": "valid",
            "selection_export_split": "selection_valid",
            "test_policy": "locked; train command never reads test",
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "script_sha256": file_hash(Path(__file__)),
            "model_code_sha256": file_hash(ROOT / "roll/alpha360_decoupled.py"),
            "torch": torch.__version__,
            "autocast_dtype": str(self.amp_dtype) if self.device.type == "cuda" else "float32",
        }

    def tensor_date_batch(self, batches):
        torch = self.torch
        features, ids, labels, mask = pad_date_batches(batches)
        features = torch.from_numpy(features).to(self.device)
        ids = torch.from_numpy(ids).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)
        legs = torch.from_numpy(labels).to(self.device)
        targets = horizon_targets(legs, self.horizon_names, torch) * self.config.target_scale
        return features, ids, targets, mask

    def forward(self, features, ids, mask):
        torch = self.torch
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self.model(features, ids, mask)

    def train_epoch(self, epoch: int, max_days: int | None = None):
        torch = self.torch
        self.model.train()
        losses, durations, processed = [], [], 0
        total_days = min(self.store.days("train"), max_days) if max_days else self.store.days("train")
        iterator = self.store.iterate("train", seed=self.args.seed + epoch, max_days=max_days)
        for batches in group_date_batches(iterator, self.args.date_batch_size):
            usable = [batch for batch in batches if np.isfinite(batch["labels"]).all(axis=1).any()]
            if not usable:
                continue
            tick = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            features, ids, targets, mask = self.tensor_date_batch(usable)
            output = self.forward(features, ids, mask)
            loss = independent_gaussian_nll(targets, output["mean"], output["std"], mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training NLL: {[batch['date'] for batch in usable]}")
            # This is a true padded date batch. There is no gradient accumulation
            # and no gradient clipping.
            scaled_loss = loss * (len(usable) / self.args.date_batch_size)
            self.scaler.scale(scaled_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            processed += len(usable)
            losses.append(float(loss.detach()))
            durations.append(time.monotonic() - tick)
            if processed % 100 == 0:
                state = {
                    "status": "training", "pid": os.getpid(), "epoch": epoch + 1,
                    "processed_dates": processed, "training_dates_total": total_days,
                    "date_batch_size": self.args.date_batch_size,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                write_json(self.args.output / "status.json", state)
                print("PROGRESS " + json.dumps(state), flush=True)
        if not losses:
            raise RuntimeError("No usable training labels")
        return float(np.mean(losses)), durations

    def evaluate_current(self, split: str, collect: bool = False, only_horizon: str | None = None,
                         max_days: int | None = None):
        torch = self.torch
        self.model.eval()
        names = (only_horizon,) if only_horizon else self.horizon_names
        model_indices = [self.horizon_names.index(name) for name in names]
        rows, prediction_rows = [], []
        with torch.no_grad():
            for batch in self.store.iterate(split, max_days=max_days):
                features, ids, targets, mask = self.tensor_date_batch([batch])
                output = self.forward(features, ids, mask)
                target_subset = targets[..., model_indices]
                if not torch.isfinite(target_subset).any():
                    # Purged boundary dates intentionally have no label. They
                    # must not contribute a zero NLL or an empty MAE.
                    continue
                mean_scaled = output["mean"][..., model_indices]
                std_scaled = output["std"][..., model_indices]
                report = distribution_report(
                    mean_scaled / self.config.target_scale,
                    std_scaled / self.config.target_scale,
                )
                targets_log = target_subset / self.config.target_scale
                loss_scaled = independent_gaussian_nll(
                    target_subset, mean_scaled, std_scaled, mask
                )
                day_row = {"datetime": batch["date"], "stocks": len(batch["stock_ids"])}
                prediction = {
                    "datetime": batch["date"],
                    "instrument": [self.store.id_to_code[int(value)] for value in batch["stock_ids"]],
                }
                for column, name in enumerate(names):
                    actual_log = targets_log[0, :, column].float().cpu().numpy()
                    actual_return = np.expm1(actual_log)
                    mean = report["log_mean"][0, :, column].cpu().numpy()
                    std = report["log_std"][0, :, column].cpu().numpy()
                    expected = report["expected_return"][0, :, column].cpu().numpy()
                    probability = report["probability_positive"][0, :, column].cpu().numpy()
                    usable = np.isfinite(expected) & np.isfinite(actual_return)
                    rank_ic = (pd.Series(expected[usable]).corr(pd.Series(actual_return[usable]), method="spearman")
                               if usable.sum() > 1 else np.nan)
                    component_loss = independent_gaussian_nll(
                        target_subset[..., column:column + 1],
                        mean_scaled[..., column:column + 1],
                        std_scaled[..., column:column + 1], mask,
                    )
                    day_row[f"{name}_rank_ic"] = rank_ic
                    day_row[f"{name}_nll_log_return"] = float(component_loss) - math.log(self.config.target_scale)
                    day_row[f"{name}_mae"] = float(np.mean(np.abs(expected[usable] - actual_return[usable])))
                    day_row[f"{name}_brier"] = float(np.mean(
                        (probability[usable] - (actual_return[usable] > 0)) ** 2
                    ))
                    if collect:
                        prediction[f"{name}_log_mean"] = mean
                        prediction[f"{name}_log_variance"] = std**2
                        prediction[f"{name}_expected_return"] = expected
                        prediction[f"{name}_return_std"] = report["return_std"][0, :, column].cpu().numpy()
                        prediction[f"{name}_probability_positive"] = probability
                        prediction[f"{name}_actual_return"] = actual_return
                day_row["nll_scaled"] = float(loss_scaled)
                rows.append(day_row)
                if collect:
                    prediction_rows.append(pd.DataFrame(prediction))
        daily = pd.DataFrame(rows)
        metrics = {"days": len(daily), "nll_scaled": float(daily["nll_scaled"].mean())}
        for name in names:
            rank_ic = daily[f"{name}_rank_ic"].dropna()
            std = rank_ic.std(ddof=0)
            metrics[f"{name}_rank_ic"] = float(rank_ic.mean())
            metrics[f"{name}_rank_icir"] = float(rank_ic.mean() / std) if std > 0 else None
            for metric in ("nll_log_return", "mae", "brier"):
                metrics[f"{name}_{metric}"] = float(daily[f"{name}_{metric}"].mean())
        predictions = pd.concat(prediction_rows, ignore_index=True) if collect else None
        return metrics, daily, predictions

    def save_payload(self, filename: str, epoch: int, metric: str, value: float):
        payload = {
            "model": self.model.state_dict(), "configuration": self.configuration,
            "epoch": epoch, "selection_metric": metric, "selection_value": value,
        }
        temporary = self.args.output / (filename + ".tmp")
        self.torch.save(payload, temporary)
        replace_with_retry(temporary, self.args.output / filename)

    def train(self):
        args, torch = self.args, self.torch
        if args.output.exists() and not args.resume:
            # --log-file commonly lives inside the output directory and is
            # opened before Trainer is constructed. That empty/new log alone
            # does not make this a pre-existing experiment.
            permitted = {args.log_file.resolve()} if args.log_file else set()
            unexpected = [path for path in args.output.iterdir() if path.resolve() not in permitted]
            if unexpected:
                raise FileExistsError(
                    f"Output contains an existing run ({unexpected[0]}); use --resume or a new directory"
                )
        args.output.mkdir(parents=True, exist_ok=True)
        history, first_epoch = [], 0
        best_nll = float("inf")
        best_rank_ic = {name: -float("inf") for name in self.horizon_names}
        if args.resume:
            checkpoint = torch.load(args.output / "last_checkpoint.pt", map_location=self.device,
                                    weights_only=False)
            old = json.loads((args.output / "configuration.json").read_text())
            immutable = [
                "model_mode", "horizon_names", "model", "data_manifest_sha256", "seed",
                "epochs", "date_batch_size", "learning_rate", "minimum_learning_rate",
                "weight_decay", "warmup_epochs", "warmup_start_factor",
                "gradient_accumulation", "gradient_clipping", "script_sha256", "model_code_sha256",
            ]
            for key in immutable:
                if old[key] != self.configuration[key]:
                    raise RuntimeError(f"Resume configuration mismatch: {key}")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            first_epoch = checkpoint["epoch"]
            best_nll = checkpoint["best_nll"]
            best_rank_ic = checkpoint["best_rank_ic"]
        write_json(args.output / "configuration.json", self.configuration)
        write_json(args.output / "stock_ids.json", {code: idx for idx, code in self.store.id_to_code.items()})
        np.savez(args.output / "normalizer.npz", mean=self.store.mean, std=self.store.std)
        print("CONFIGURATION " + json.dumps(self.configuration), flush=True)

        if args.benchmark_only:
            loss, durations = self.train_epoch(0, args.benchmark_days)
            started = time.monotonic()
            self.evaluate_current("valid", max_days=5)
            valid_seconds = (time.monotonic() - started) / min(5, self.store.days("valid"))
            step_seconds = float(np.median(durations[2:] or durations))
            estimated = (step_seconds * math.ceil(self.store.days("train") / args.date_batch_size)
                         + valid_seconds * self.store.days("valid"))
            write_json(args.output / "benchmark.json", {
                "train_loss": loss, "date_batch_size": args.date_batch_size,
                "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                           if self.device.type == "cuda" else None),
                "estimated_epoch_seconds": estimated,
                "estimated_50_epochs_hours": estimated * args.epochs / 3600,
            })
            return

        for epoch in range(first_epoch, args.epochs):
            started = time.monotonic()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_loss, _ = self.train_epoch(epoch)
            valid, _, _ = self.evaluate_current("valid")
            if valid["nll_scaled"] < best_nll:
                best_nll = valid["nll_scaled"]
                self.save_payload("best_nll_model.pt", epoch + 1, "nll_scaled", best_nll)
            for name in self.horizon_names:
                value = valid[f"{name}_rank_ic"]
                if np.isfinite(value) and value > best_rank_ic[name]:
                    best_rank_ic[name] = value
                    self.save_payload(f"best_{name}_rank_ic_model.pt", epoch + 1,
                                      f"{name}_rank_ic", value)
            row = {
                "epoch": epoch + 1, "train_nll_scaled": train_loss, **valid,
                "learning_rate": learning_rate,
                "epoch_seconds": time.monotonic() - started,
                "best_valid_nll": best_nll,
            }
            history.append(row)
            self.scheduler.step()
            pd.DataFrame(history).to_csv(args.output / "epoch_metrics.csv", index=False)
            snapshot = {
                "epoch": epoch + 1, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "history": history,
                "best_nll": best_nll, "best_rank_ic": best_rank_ic,
            }
            temporary = args.output / "last_checkpoint.pt.tmp"
            torch.save(snapshot, temporary)
            replace_with_retry(temporary, args.output / "last_checkpoint.pt")
            print("EPOCH " + json.dumps(row), flush=True)

        prediction_frames, selection_rows, checkpoint_hashes, checkpoint_freeze = [], [], {}, {}
        for name in self.horizon_names:
            checkpoint_path = args.output / f"best_{name}_rank_ic_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if checkpoint.get("configuration") != self.configuration:
                raise RuntimeError(f"Checkpoint/configuration mismatch before Selection freeze: {name}")
            if checkpoint.get("selection_metric") != f"{name}_rank_ic":
                raise RuntimeError(f"Checkpoint has the wrong Selection metric: {name}")
            self.model.load_state_dict(checkpoint["model"])
            metrics, daily, predictions = self.evaluate_current(
                "selection_valid", collect=True, only_horizon=name
            )
            daily.to_csv(args.output / f"selection_valid_{name}_daily_metrics.csv", index=False)
            prediction_frames.append(predictions)
            selection_rows.append({
                "horizon": name, "valid_best_epoch": checkpoint["epoch"],
                "valid_selection_value": checkpoint["selection_value"], **metrics,
            })
            checkpoint_hashes[name] = file_hash(checkpoint_path)
            checkpoint_freeze[name] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_hashes[name],
                "epoch": checkpoint["epoch"],
                "selection_metric": checkpoint["selection_metric"],
                "selection_value": checkpoint["selection_value"],
            }
        merge_prediction_columns(prediction_frames).to_csv(
            args.output / "selection_valid_predictions.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(args.output / "selection_valid_summary.csv", index=False)
        candidate_manifest = write_selection_candidate_manifest(
            args.output, args.data, self.configuration, checkpoint_freeze
        )
        candidate_manifest_path = args.output / "selection_candidate_manifest.json"
        write_json(args.output / "status.json", {
            "status": "selection_ready", "epochs": args.epochs,
            "test_read": False, "checkpoint_sha256": checkpoint_hashes,
            "selection_candidate_manifest": str(candidate_manifest_path.resolve()),
            "selection_candidate_manifest_sha256": file_hash(candidate_manifest_path),
            "selection_candidate_status": candidate_manifest["status"],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print("SELECTION_VALID READY; TEST REMAINS LOCKED", flush=True)

    def evaluate_test(self, manifest_path: Path, candidate_name: str):
        from script.select_alpha360_probabilistic_ensemble import (
            validate_frozen_selection_manifest,
        )

        # Authenticate the protocol, every candidate manifest, all Selection
        # inputs, and all selected checkpoint hashes before touching Test.
        manifest_path = manifest_path.expanduser().resolve()
        manifest, _, candidates, frozen_candidates = validate_frozen_selection_manifest(
            manifest_path, require_test_artifacts=False
        )
        if candidate_name not in candidates:
            raise RuntimeError(f"Candidate {candidate_name!r} is absent from the frozen manifest")
        expected = candidates[candidate_name]
        if expected != self.args.output.resolve():
            raise RuntimeError(f"Selection manifest candidate path mismatch: {expected}")
        selected_horizons = [
            name for name, value in manifest["selections"].items()
            if candidate_name in value["selected_components"]
        ]
        if not selected_horizons:
            raise RuntimeError(f"Candidate {candidate_name} was not selected for any horizon")
        frozen_candidate = frozen_candidates[candidate_name]
        frozen_configuration_path = Path(
            frozen_candidate["configuration"]["path"]
        ).resolve()
        frozen_configuration = json.loads(
            frozen_configuration_path.read_text(encoding="utf-8-sig")
        )
        stable_configuration_keys = (
            "model_mode",
            "horizon_names",
            "model",
            "data_manifest_sha256",
            "segments",
            "seed",
            "epochs",
            "date_batch_size",
            "target_scale",
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "warmup_epochs",
            "warmup_start_factor",
            "gradient_accumulation",
            "gradient_clipping",
            "model_code_sha256",
        )
        for key in stable_configuration_keys:
            if frozen_configuration.get(key) != self.configuration.get(key):
                raise RuntimeError(f"Frozen/current candidate configuration mismatch: {key}")

        destinations = [
            self.args.output / "test_predictions.csv",
            self.args.output / "test_summary.csv",
            self.args.output / "test_access.json",
            self.args.output / "test_completion_audit.json",
            *[
                self.args.output / f"test_{name}_daily_metrics.csv"
                for name in selected_horizons
            ],
        ]
        for destination in destinations:
            if destination.exists():
                raise FileExistsError(destination)

        checkpoints: dict[str, dict] = {}
        checkpoint_hashes: dict[str, str] = {}
        for name in selected_horizons:
            checkpoint_reference = frozen_candidate["checkpoints"][name]
            checkpoint_path = Path(checkpoint_reference["path"]).resolve()
            checkpoint_hash = file_hash(checkpoint_path)
            expected_hash = manifest["selections"][name]["selected_checkpoint_sha256"].get(
                candidate_name
            )
            if checkpoint_hash != checkpoint_reference["sha256"] or checkpoint_hash != expected_hash:
                raise RuntimeError(f"Frozen checkpoint hash mismatch before Test: {name}")
            checkpoint = self.torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if checkpoint.get("configuration") != frozen_configuration:
                raise RuntimeError(f"Frozen checkpoint configuration mismatch before Test: {name}")
            if checkpoint.get("selection_metric") != f"{name}_rank_ic":
                raise RuntimeError(f"Frozen checkpoint metric mismatch before Test: {name}")
            checkpoints[name] = checkpoint
            checkpoint_hashes[name] = checkpoint_hash

        # The full freeze and selected-checkpoint gates above must succeed
        # before any held-out Test array is hashed or memory-mapped.
        print("Selection manifest frozen; verifying held-out Test hashes...", flush=True)
        self.store.verify_parts("test")
        frames, rows = [], []
        for name in selected_horizons:
            checkpoint = checkpoints[name]
            self.model.load_state_dict(checkpoint["model"])
            metrics, daily, predictions = self.evaluate_current("test", collect=True, only_horizon=name)
            daily.to_csv(self.args.output / f"test_{name}_daily_metrics.csv", index=False)
            frames.append(predictions)
            rows.append({"horizon": name, "checkpoint_epoch": checkpoint["epoch"], **metrics})
        predictions_path = self.args.output / "test_predictions.csv"
        summary_path = self.args.output / "test_summary.csv"
        merge_prediction_columns(frames).to_csv(predictions_path, index=False)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        test_access_path = self.args.output / "test_access.json"
        access = {
            "schema_version": 1,
            "status": "test_access_complete",
            "selection_manifest": frozen_file(manifest_path),
            "candidate_manifest": frozen_candidate["candidate_manifest"],
            "candidate_name": candidate_name,
            "selected_horizons": selected_horizons,
            "checkpoint_sha256": checkpoint_hashes,
            "test_read": True,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(test_access_path, access)
        artifact_paths = [
            predictions_path,
            summary_path,
            test_access_path,
            *[self.args.output / f"test_{name}_daily_metrics.csv" for name in selected_horizons],
        ]
        completion = {
            "schema_version": 1,
            "status": "test_complete",
            "test_read": True,
            "selection_manifest": frozen_file(manifest_path),
            "candidate_manifest": frozen_candidate["candidate_manifest"],
            "candidate_name": candidate_name,
            "selected_horizons": selected_horizons,
            "checkpoint_sha256": checkpoint_hashes,
            "artifacts": {path.name: frozen_file(path) for path in artifact_paths},
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(self.args.output / "test_completion_audit.json", completion)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate-test"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-mode", choices=["shared_four_head", "single_horizon"], required=True)
    parser.add_argument("--horizon", choices=HORIZON_NAMES)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-factor", type=float, default=1 / 3)
    parser.add_argument("--date-batch-size", type=int, default=4)
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-days", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.model_mode == "single_horizon" and args.horizon is None:
        parser.error("single_horizon requires --horizon")
    if args.model_mode == "shared_four_head" and args.horizon is not None:
        parser.error("shared_four_head does not accept --horizon")
    if args.command == "evaluate-test" and (not args.selection_manifest or not args.candidate_name):
        parser.error("evaluate-test requires --selection-manifest and --candidate-name")
    if args.command == "train" and (args.selection_manifest or args.candidate_name):
        parser.error("selection manifest arguments are only valid for evaluate-test")
    if args.warmup_epochs >= args.epochs or min(
        args.epochs, args.warmup_epochs, args.date_batch_size, args.benchmark_days
    ) < 1:
        parser.error("invalid positive count or warmup >= epochs")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error("minimum learning rate must be in (0, learning rate]")
    return args


def main():
    args = parse_args()
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        trainer = Trainer(args)
        if args.command == "train":
            trainer.train()
        else:
            trainer.evaluate_test(args.selection_manifest, args.candidate_name)
    except Exception as error:
        write_json(args.output / "last_failure.json", {
            "error": repr(error), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
