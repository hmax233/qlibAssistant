#!/usr/bin/env python3
"""Materialize horizon-specific E0 checkpoints without leaking held-out test data.

``selection`` evaluates each horizon with its own best validation-RankIC
checkpoint and writes one candidate ``selection_valid_predictions.csv``.
``evaluate-test`` is deliberately separate and requires a frozen ensemble
selection manifest.  It reads test data only for horizons that actually chose
the requested candidate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

from train_alpha360_cross_stock import DateStore  # noqa: E402
from roll.alpha360_cross_stock import (  # noqa: E402
    Alpha360CrossStockTransformer,
    Alpha360TransformerConfig,
    HORIZON_MATRIX,
    HORIZON_NAMES,
    distribution_report,
)


KEYS = ["datetime", "instrument"]
REPORT_COLUMNS = (
    "log_mean",
    "log_variance",
    "expected_return",
    "return_std",
    "probability_positive",
    "actual_return",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def horizon_columns(horizon: str) -> list[str]:
    return [*KEYS, *[f"{horizon}_{suffix}" for suffix in REPORT_COLUMNS]]


def _same_configuration(checkpoint: dict[str, Any], source: dict[str, Any], path: Path) -> None:
    checkpoint_configuration = checkpoint.get("configuration")
    if not isinstance(checkpoint_configuration, dict):
        raise ValueError(f"Checkpoint has no configuration object: {path}")
    required = (
        "model",
        "data_manifest_sha256",
        "segments",
        "feature_count",
        "target_scale",
        "model_code_sha256",
    )
    for key in required:
        if key not in source or key not in checkpoint_configuration:
            raise ValueError(f"Missing source/checkpoint configuration key {key!r}: {path}")
        if checkpoint_configuration[key] != source[key]:
            raise ValueError(f"Checkpoint configuration mismatch for {key!r}: {path}")


def _verify_split_parts(store: DateStore, split: str) -> list[dict[str, Any]]:
    """Verify only the requested partition; selection must never open test arrays."""

    parts = [part for part in store.manifest.get("parts", []) if part.get("split") == split]
    if not parts:
        raise ValueError(f"Dataset has no parts for split {split!r}")
    verified: list[dict[str, Any]] = []
    for part in parts:
        hashes = part.get("sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"Part has no file hashes: {part.get('prefix')}")
        for filename, expected in hashes.items():
            path = store.directory / filename
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Input hash mismatch: {path}")
            verified.append({"path": str(path.resolve()), "sha256": actual})
    return verified


def validate_source(source_run: Path, data: Path, split: str) -> tuple[DateStore, dict[str, Any]]:
    source_run = source_run.expanduser().resolve()
    data = data.expanduser().resolve()
    configuration_path = source_run / "configuration.json"
    data_manifest_path = data / "manifest.json"
    run_stock_ids_path = source_run / "stock_ids.json"
    data_stock_ids_path = data / "stock_ids.json"
    for path in (configuration_path, data_manifest_path, run_stock_ids_path, data_stock_ids_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration = read_json(configuration_path)
    data_manifest_hash = sha256(data_manifest_path)
    if configuration.get("data_manifest_sha256") != data_manifest_hash:
        raise RuntimeError("Source configuration and --data manifest hash disagree")
    if configuration.get("feature_count") != 360:
        raise ValueError("Source configuration is not an Alpha360 run")
    if configuration.get("segments") != read_json(data_manifest_path).get("segments"):
        raise RuntimeError("Source configuration and data segments disagree")

    store = DateStore(data)
    run_vocabulary = read_json(run_stock_ids_path)
    data_vocabulary = read_json(data_stock_ids_path)
    if run_vocabulary != data_vocabulary:
        raise RuntimeError("Source-run and dataset stock dictionaries disagree")
    expected_ids = set(range(1, int(store.manifest["stock_count"]) + 1))
    actual_ids = {int(value) for value in run_vocabulary.values()}
    if actual_ids != expected_ids:
        raise RuntimeError("Stock dictionary IDs are not the expected contiguous 1..stock_count range")
    split_parts = _verify_split_parts(store, split)
    audit = {
        "source_run": str(source_run),
        "source_configuration": {
            "path": str(configuration_path.resolve()),
            "sha256": sha256(configuration_path),
        },
        "data": str(data),
        "data_manifest": {
            "path": str(data_manifest_path.resolve()),
            "sha256": data_manifest_hash,
        },
        "source_stock_dictionary": {
            "path": str(run_stock_ids_path.resolve()),
            "sha256": sha256(run_stock_ids_path),
        },
        "data_stock_dictionary": {
            "path": str(data_stock_ids_path.resolve()),
            "sha256": sha256(data_stock_ids_path),
        },
        "verified_split_parts": split_parts,
    }
    return store, {"configuration": configuration, "audit": audit}


def validate_device(device_name: str, bf16: bool) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    if bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 requested but unsupported by this GPU")
    if bf16 and device.type not in {"cpu", "cuda"}:
        raise ValueError("BF16 inference is supported only on CPU or CUDA")
    return device


def load_horizon_model(
    source_run: Path,
    horizon: str,
    source_configuration: dict[str, Any],
    stock_count: int,
    device: torch.device,
) -> tuple[Alpha360CrossStockTransformer, dict[str, Any]]:
    checkpoint_path = source_run / f"best_{horizon}_rank_ic_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    _same_configuration(checkpoint, source_configuration, checkpoint_path)
    expected_metric = f"{horizon}_rank_ic"
    if checkpoint.get("selection_metric") != expected_metric:
        raise ValueError(
            f"Checkpoint selection metric is {checkpoint.get('selection_metric')!r}, "
            f"expected {expected_metric!r}: {checkpoint_path}"
        )
    config = Alpha360TransformerConfig(**source_configuration["model"])
    if asdict(config) != source_configuration["model"]:
        raise ValueError("Source model configuration contains unsupported or noncanonical values")
    model = Alpha360CrossStockTransformer(stock_count, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.stock_identity.num_embeddings != stock_count + 1:
        raise RuntimeError("Checkpoint stock embedding does not match the validated dictionary")
    model.to(device).eval()
    provenance = {
        "path": str(checkpoint_path.resolve()),
        "sha256": checkpoint_hash,
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
    }
    return model, provenance


def _autocast(device: torch.device, bf16: bool):
    if not bf16:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def infer_horizon(
    store: DateStore,
    model: Alpha360CrossStockTransformer,
    source_configuration: dict[str, Any],
    split: str,
    horizon: str,
    device: torch.device,
    bf16: bool,
) -> pd.DataFrame:
    horizon_index = HORIZON_NAMES.index(horizon)
    target_scale = float(source_configuration["target_scale"])
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    matrix = HORIZON_MATRIX.detach().cpu().numpy().T
    rows: list[pd.DataFrame] = []
    model.eval()
    with torch.inference_mode():
        for batch in store.iterate(split):
            features = torch.from_numpy(np.asarray(batch["features"], dtype="float32"))[None].to(device)
            stock_ids = torch.from_numpy(np.asarray(batch["stock_ids"], dtype="int64"))[None].to(device)
            stock_mask = torch.ones_like(stock_ids, dtype=torch.bool)
            with _autocast(device, bf16):
                output = model(features, stock_ids, stock_mask)
            horizon_mean = output["horizon_mean"].float()[0] / target_scale
            horizon_covariance = output["horizon_covariance"].float()[0] / target_scale**2
            report = distribution_report(horizon_mean, horizon_covariance)
            labels = np.asarray(batch["labels"], dtype="float64")
            actual = np.expm1(labels @ matrix)[:, horizon_index]
            try:
                instruments = [store.id_to_code[int(value)] for value in batch["stock_ids"]]
            except KeyError as error:
                raise RuntimeError(f"Unknown stock ID in {split}: {error.args[0]}") from error
            values: dict[str, Any] = {
                "datetime": [batch["date"]] * len(instruments),
                "instrument": instruments,
            }
            for suffix in REPORT_COLUMNS[:-1]:
                values[f"{horizon}_{suffix}"] = (
                    report[suffix][:, horizon_index].float().cpu().numpy()
                )
            values[f"{horizon}_actual_return"] = actual
            rows.append(pd.DataFrame(values))
    if not rows:
        raise RuntimeError(f"No predictions were materialized for {split}/{horizon}")
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(KEYS).any():
        raise RuntimeError(f"Duplicate prediction keys for {split}/{horizon}")
    return result[horizon_columns(horizon)].sort_values(KEYS).reset_index(drop=True)


def merge_horizons(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one horizon prediction is required")
    result: pd.DataFrame | None = None
    reference_keys: pd.DataFrame | None = None
    for horizon in HORIZON_NAMES:
        if horizon not in frames:
            continue
        frame = frames[horizon]
        keys = frame[KEYS].reset_index(drop=True)
        if reference_keys is None:
            reference_keys = keys
            result = frame
        else:
            if not keys.equals(reference_keys):
                raise RuntimeError(f"Prediction key set/order differs for horizon {horizon}")
            assert result is not None
            result = result.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    assert result is not None
    expected = [*KEYS]
    for horizon in HORIZON_NAMES:
        if horizon in frames:
            expected.extend(f"{horizon}_{suffix}" for suffix in REPORT_COLUMNS)
    return result[expected]


def _write_materialization(
    output: Path,
    filename: str,
    frame: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    csv_path = output / filename
    frame.to_csv(csv_path, index=False)
    audit["prediction_output"] = {
        "path": str(csv_path.resolve()),
        "sha256": sha256(csv_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    audit_path = output / "materialization_manifest.json"
    write_json(audit_path, audit)
    return audit


def _append_test_materialization(
    candidate_directory: Path,
    frame: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Add test artifacts to an authenticated candidate without replacing files."""

    if not candidate_directory.is_dir():
        raise FileNotFoundError(f"Candidate output directory is absent: {candidate_directory}")
    csv_path = candidate_directory / "test_predictions.csv"
    audit_path = candidate_directory / "test_materialization_manifest.json"
    for path in (csv_path, audit_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing test artifact: {path}")
    # mode='x' preserves the no-overwrite guarantee if another process races us.
    frame.to_csv(csv_path, index=False, mode="x")
    audit["prediction_output"] = {
        "path": str(csv_path.resolve()),
        "sha256": sha256(csv_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    write_json(audit_path, audit)
    return audit


def materialize_selection(
    source_run: Path,
    data: Path,
    output: Path,
    device_name: str = "cpu",
    bf16: bool = False,
    threads: int = 4,
) -> dict[str, Any]:
    if output.expanduser().exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    device = validate_device(device_name, bf16)
    torch.set_num_threads(threads)
    source_run = source_run.expanduser().resolve()
    store, source = validate_source(source_run, data, "selection_valid")
    frames: dict[str, pd.DataFrame] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for horizon in HORIZON_NAMES:
        model, provenance = load_horizon_model(
            source_run,
            horizon,
            source["configuration"],
            int(store.manifest["stock_count"]),
            device,
        )
        frames[horizon] = infer_horizon(
            store, model, source["configuration"], "selection_valid", horizon, device, bf16
        )
        checkpoints[horizon] = provenance
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = merge_horizons(frames)
    audit = {
        "schema_version": 1,
        "command": "selection",
        "split": "selection_valid",
        "test_read": False,
        "device": str(device),
        "bf16": bool(bf16),
        **source["audit"],
        "horizon_checkpoints": checkpoints,
    }
    return _write_materialization(
        output.expanduser().resolve(), "selection_valid_predictions.csv", frame, audit
    )


def _candidate_path(value: Any) -> Path:
    if isinstance(value, str):
        return Path(value).expanduser().resolve()
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return Path(value["path"]).expanduser().resolve()
    raise ValueError("Candidate entry must be a path string or an object containing 'path'")


def selected_horizons_for_candidate(manifest: dict[str, Any], candidate_name: str) -> list[str]:
    candidates = manifest.get("candidates")
    selections = manifest.get("selections")
    if not isinstance(candidates, dict) or candidate_name not in candidates:
        raise RuntimeError(f"Candidate {candidate_name!r} is absent from the selection manifest")
    if not isinstance(selections, dict):
        raise ValueError("Selection manifest has no selections object")
    selected: list[str] = []
    for horizon in HORIZON_NAMES:
        value = selections.get(horizon)
        if not isinstance(value, dict):
            continue
        components = value.get("selected_components")
        if isinstance(components, list) and candidate_name in components:
            selected.append(horizon)
    if not selected:
        raise RuntimeError(f"Candidate {candidate_name!r} was not selected for any horizon")
    return selected


def validate_frozen_manifest(
    manifest_path: Path,
    candidate_name: str,
    source_run: Path,
    data: Path,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("selection_split") != "selection_valid":
        raise ValueError("Manifest was not frozen from selection_valid")
    if manifest.get("test_files_read") is not False:
        raise ValueError("Manifest is not a pre-test frozen selection manifest")
    selected = selected_horizons_for_candidate(manifest, candidate_name)
    candidate_path = _candidate_path(manifest["candidates"][candidate_name])
    candidate_audit_path = candidate_path / "materialization_manifest.json"
    candidate_predictions = candidate_path / "selection_valid_predictions.csv"
    if not candidate_audit_path.is_file() or not candidate_predictions.is_file():
        raise FileNotFoundError("Candidate lacks selection materialization audit or predictions")
    candidate_audit = read_json(candidate_audit_path)
    if candidate_audit.get("command") != "selection" or candidate_audit.get("test_read") is not False:
        raise ValueError("Candidate directory is not a test-locked selection materialization")
    if Path(candidate_audit["source_run"]).resolve() != source_run.expanduser().resolve():
        raise RuntimeError("Candidate source-run path does not match --source-run")
    if Path(candidate_audit["data"]).resolve() != data.expanduser().resolve():
        raise RuntimeError("Candidate data path does not match --data")
    actual_prediction_hash = sha256(candidate_predictions)
    recorded_prediction = candidate_audit.get("prediction_output", {})
    if Path(recorded_prediction.get("path", "")).resolve() != candidate_predictions.resolve():
        raise RuntimeError("Candidate audit points at a different selection prediction file")
    if recorded_prediction.get("sha256") != actual_prediction_hash:
        raise RuntimeError("Candidate selection predictions changed after materialization")
    manifest_hashes = manifest.get("input_sha256", {})
    expected_prediction_hash = manifest_hashes.get(
        f"{candidate_name}:selection_valid_predictions"
    )
    if expected_prediction_hash != actual_prediction_hash:
        raise RuntimeError("Frozen selection manifest does not authenticate candidate predictions")
    return manifest, selected, {
        "selection_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "candidate_name": candidate_name,
        "candidate_selection_directory": str(candidate_path),
        "candidate_materialization_manifest": {
            "path": str(candidate_audit_path.resolve()),
            "sha256": sha256(candidate_audit_path),
        },
        "candidate_selection_predictions": {
            "path": str(candidate_predictions.resolve()),
            "sha256": actual_prediction_hash,
        },
        "selection_time_provenance": {
            "source_configuration_sha256": candidate_audit.get(
                "source_configuration", {}
            ).get("sha256"),
            "data_manifest_sha256": candidate_audit.get("data_manifest", {}).get("sha256"),
            "source_stock_dictionary_sha256": candidate_audit.get(
                "source_stock_dictionary", {}
            ).get("sha256"),
            "data_stock_dictionary_sha256": candidate_audit.get(
                "data_stock_dictionary", {}
            ).get("sha256"),
            "horizon_checkpoint_sha256": {
                horizon: candidate_audit.get("horizon_checkpoints", {})
                .get(horizon, {})
                .get("sha256")
                for horizon in HORIZON_NAMES
            },
        },
    }


def evaluate_test(
    source_run: Path,
    data: Path,
    output: Path,
    selection_manifest: Path,
    candidate_name: str,
    device_name: str = "cpu",
    bf16: bool = False,
    threads: int = 4,
) -> dict[str, Any]:
    # This gate intentionally runs before DateStore construction or any test access.
    _, selected_horizons, selection_audit = validate_frozen_manifest(
        selection_manifest, candidate_name, source_run, data
    )
    output = output.expanduser().resolve()
    authenticated_candidate = Path(
        selection_audit["candidate_selection_directory"]
    ).resolve()
    if output != authenticated_candidate:
        raise RuntimeError(
            "--output must be the candidate directory authenticated by the frozen manifest: "
            f"{authenticated_candidate}"
        )
    # Refuse collisions after manifest authentication but before opening Test.
    for path in (
        output / "test_predictions.csv",
        output / "test_materialization_manifest.json",
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing test artifact: {path}")
    device = validate_device(device_name, bf16)
    torch.set_num_threads(threads)
    source_run = source_run.expanduser().resolve()
    store, source = validate_source(source_run, data, "test")
    frozen = selection_audit["selection_time_provenance"]
    current_hashes = {
        "source_configuration_sha256": source["audit"]["source_configuration"]["sha256"],
        "data_manifest_sha256": source["audit"]["data_manifest"]["sha256"],
        "source_stock_dictionary_sha256": source["audit"]["source_stock_dictionary"]["sha256"],
        "data_stock_dictionary_sha256": source["audit"]["data_stock_dictionary"]["sha256"],
    }
    for key, actual in current_hashes.items():
        if frozen.get(key) != actual:
            raise RuntimeError(f"Selection-time provenance changed before test: {key}")
    frames: dict[str, pd.DataFrame] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for horizon in selected_horizons:
        model, provenance = load_horizon_model(
            source_run,
            horizon,
            source["configuration"],
            int(store.manifest["stock_count"]),
            device,
        )
        expected_checkpoint_hash = frozen["horizon_checkpoint_sha256"].get(horizon)
        if provenance["sha256"] != expected_checkpoint_hash:
            raise RuntimeError(
                f"Checkpoint changed after selection for horizon {horizon}: "
                f"expected {expected_checkpoint_hash}, got {provenance['sha256']}"
            )
        frames[horizon] = infer_horizon(
            store, model, source["configuration"], "test", horizon, device, bf16
        )
        checkpoints[horizon] = provenance
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = merge_horizons(frames)
    audit = {
        "schema_version": 1,
        "command": "evaluate-test",
        "split": "test",
        "test_read": True,
        "device": str(device),
        "bf16": bool(bf16),
        "selected_horizons": selected_horizons,
        **selection_audit,
        **source["audit"],
        "horizon_checkpoints": checkpoints,
    }
    return _append_test_materialization(output, frame, audit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("selection", "evaluate-test"):
        command = commands.add_parser(name)
        command.add_argument("--source-run", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--bf16", action="store_true")
        command.add_argument("--threads", type=int, default=4)
        if name == "evaluate-test":
            command.add_argument("--selection-manifest", type=Path, required=True)
            command.add_argument("--candidate-name", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.command == "selection":
        result = materialize_selection(
            args.source_run, args.data, args.output, args.device, args.bf16, args.threads
        )
    else:
        result = evaluate_test(
            args.source_run,
            args.data,
            args.output,
            args.selection_manifest,
            args.candidate_name,
            args.device,
            args.bf16,
            args.threads,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
