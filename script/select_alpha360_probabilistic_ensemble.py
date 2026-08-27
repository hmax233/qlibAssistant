#!/usr/bin/env python3
"""Select Alpha360 probabilistic ensembles without touching the held-out test.

Selection reads only ``selection_valid_predictions.csv``.  Test evaluation is a
separate command that refuses to run until the immutable selection manifest has
already been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd


HORIZONS = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")
KEYS = ["datetime", "instrument"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_reference(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def verify_reference(reference: dict, label: str, expected_path: Path | None = None) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} is not a file reference")
    raw_path, expected_hash = reference.get("path"), reference.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label} must contain path and sha256")
    path = Path(raw_path).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise RuntimeError(f"{label} path mismatch: expected {expected_path}, got {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}")
    return path


def load_protocol(protocol: Path) -> tuple[dict, str]:
    protocol = protocol.expanduser().resolve()
    body = read_json(protocol)
    required = {"segments", "data_manifest_sha256", "horizons", "experiments", "optimization"}
    missing = required - set(body)
    if missing:
        raise ValueError(f"Protocol missing required fields: {sorted(missing)}")
    if tuple(body["horizons"]) != HORIZONS:
        raise ValueError(f"Protocol horizons must be exactly {HORIZONS}")
    if not isinstance(body["segments"], dict) or not isinstance(body["optimization"], dict):
        raise ValueError("Protocol segments and optimization must be objects")
    experiment_ids = [value.get("id") for value in body["experiments"] if isinstance(value, dict)]
    if len(experiment_ids) != len(body["experiments"]) or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Protocol experiment IDs are missing or duplicated")
    return body, sha256(protocol)


def _configuration_value(configuration: dict, *names: str):
    for name in names:
        if name in configuration:
            return configuration[name]
    raise ValueError(f"Candidate configuration lacks all aliases {names}")


def validate_configuration_against_protocol(configuration: dict, protocol_body: dict) -> None:
    if configuration.get("segments") != protocol_body["segments"]:
        raise RuntimeError("Candidate segments do not match the frozen protocol")
    if configuration.get("data_manifest_sha256") != protocol_body["data_manifest_sha256"]:
        raise RuntimeError("Candidate data manifest hash does not match the frozen protocol")
    optimization = protocol_body["optimization"]
    aliases = {
        "epochs": ("epochs",),
        "early_stopping": ("early_stopping",),
        "learning_rate": ("learning_rate",),
        "minimum_learning_rate": ("minimum_learning_rate", "min_learning_rate"),
        "warmup_epochs": ("warmup_epochs",),
        "warmup_start_factor": ("warmup_start_factor",),
        "date_batch_size": ("date_batch_size",),
        "seed": ("seed",),
        "target_scale": ("target_scale",),
    }
    for protocol_key, candidate_keys in aliases.items():
        if protocol_key not in optimization:
            raise ValueError(f"Protocol optimization lacks {protocol_key}")
        actual = _configuration_value(configuration, *candidate_keys)
        if actual != optimization[protocol_key]:
            raise RuntimeError(
                f"Candidate optimization mismatch for {protocol_key}: "
                f"expected {optimization[protocol_key]!r}, got {actual!r}"
            )
    for optional in ("optimizer", "weight_decay", "gradient_accumulation", "gradient_clipping"):
        if optional in configuration and configuration[optional] != optimization.get(optional):
            raise RuntimeError(f"Candidate optimization mismatch for {optional}")


def validate_candidate_manifest(
    candidate_name: str,
    run: Path,
    protocol_body: dict,
) -> dict:
    """Authenticate one pre-Test candidate and normalize E0/E1-E5 audit schemas."""

    run = run.expanduser().resolve()
    decoupled_manifest = run / "selection_candidate_manifest.json"
    joint_manifest = run / "materialization_manifest.json"
    if decoupled_manifest.is_file():
        manifest_path, kind = decoupled_manifest, "decoupled"
    elif joint_manifest.is_file():
        manifest_path, kind = joint_manifest, "joint_e0"
    else:
        raise FileNotFoundError(
            f"Candidate {candidate_name!r} has no selection candidate manifest: {run}"
        )
    manifest = read_json(manifest_path)
    if kind == "decoupled":
        if manifest.get("status") != "selection_complete":
            raise ValueError(f"Candidate {candidate_name!r} is not selection_complete")
        if manifest.get("selection_split") != "selection_valid":
            raise ValueError(f"Candidate {candidate_name!r} has the wrong selection split")
        if manifest.get("test_files_read") is not False:
            raise ValueError(f"Candidate {candidate_name!r} is not pre-Test locked")
        if Path(manifest.get("candidate_directory", "")).resolve() != run:
            raise RuntimeError(f"Candidate {candidate_name!r} directory mismatch")
        configuration_ref = manifest.get("configuration")
        data_manifest_ref = manifest.get("data_manifest")
        prediction_ref = manifest.get("selection_valid_predictions")
        checkpoints = manifest.get("checkpoints")
        segments = manifest.get("segments")
    else:
        if manifest.get("command") != "selection" or manifest.get("split") != "selection_valid":
            raise ValueError(f"E0 candidate {candidate_name!r} is not a Selection materialization")
        if manifest.get("test_read") is not False:
            raise ValueError(f"E0 candidate {candidate_name!r} is not pre-Test locked")
        configuration_ref = manifest.get("source_configuration")
        data_manifest_ref = manifest.get("data_manifest")
        prediction_ref = manifest.get("prediction_output")
        checkpoints = manifest.get("horizon_checkpoints")
        segments = None

    configuration_path = verify_reference(configuration_ref, f"{candidate_name} configuration")
    data_manifest_path = verify_reference(data_manifest_ref, f"{candidate_name} data manifest")
    prediction_path = verify_reference(
        prediction_ref,
        f"{candidate_name} Selection predictions",
        run / "selection_valid_predictions.csv",
    )
    configuration = read_json(configuration_path)
    if segments is not None and segments != configuration.get("segments"):
        raise RuntimeError(f"Candidate {candidate_name!r} manifest/configuration segments disagree")
    if sha256(data_manifest_path) != configuration.get("data_manifest_sha256"):
        raise RuntimeError(f"Candidate {candidate_name!r} configuration/data manifest disagree")
    validate_configuration_against_protocol(configuration, protocol_body)
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError(f"Candidate {candidate_name!r} has no frozen checkpoints")
    normalized_checkpoints: dict[str, dict] = {}
    for horizon, reference in checkpoints.items():
        if horizon not in HORIZONS or not isinstance(reference, dict):
            raise ValueError(f"Candidate {candidate_name!r} has an invalid checkpoint entry")
        checkpoint_path = verify_reference(reference, f"{candidate_name}/{horizon} checkpoint")
        normalized_checkpoints[horizon] = {
            **reference,
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
        }
    return {
        "kind": kind,
        "candidate_directory": str(run),
        "candidate_manifest": file_reference(manifest_path),
        "configuration": file_reference(configuration_path),
        "data_manifest": file_reference(data_manifest_path),
        "segments": configuration["segments"],
        "selection_valid_predictions": file_reference(prediction_path),
        "checkpoints": normalized_checkpoints,
    }


def _same_frozen_candidate(expected: dict, actual: dict, name: str) -> None:
    for key in (
        "kind",
        "candidate_directory",
        "candidate_manifest",
        "configuration",
        "data_manifest",
        "segments",
        "selection_valid_predictions",
        "checkpoints",
    ):
        if expected.get(key) != actual.get(key):
            raise RuntimeError(f"Frozen candidate {name!r} changed after ensemble selection: {key}")


def _validate_candidate_test_artifacts(
    manifest_path: Path,
    manifest: dict,
    candidates: dict[str, Path],
    normalized: dict[str, dict],
    name: str,
) -> dict:
    if name not in candidates or name not in normalized:
        raise RuntimeError(f"Candidate {name!r} is absent from the frozen selection manifest")
    expected_horizons = [
        horizon for horizon in HORIZONS
        if name in manifest["selections"][horizon]["selected_components"]
    ]
    if not expected_horizons:
        raise RuntimeError(f"Candidate {name!r} was not selected for any horizon")
    run = candidates[name]
    kind = normalized[name]["kind"]
    audit_path = run / (
        "test_completion_audit.json" if kind == "decoupled"
        else "test_materialization_manifest.json"
    )
    audit = read_json(audit_path)
    if kind == "decoupled":
        if audit.get("status") != "test_complete" or audit.get("test_read") is not True:
            raise ValueError(f"Candidate {name!r} Test audit is incomplete")
        manifest_ref = audit.get("selection_manifest")
        prediction_ref = audit.get("artifacts", {}).get("test_predictions.csv")
        checkpoint_hashes = audit.get("checkpoint_sha256", {})
        audited_horizons = audit.get("selected_horizons")
        for required_name in ("test_predictions.csv", "test_summary.csv", "test_access.json"):
            verify_reference(
                audit.get("artifacts", {}).get(required_name),
                f"{name} Test artifact {required_name}",
                run / required_name,
            )
    else:
        if audit.get("command") != "evaluate-test" or audit.get("test_read") is not True:
            raise ValueError(f"E0 candidate {name!r} Test audit is incomplete")
        manifest_ref = audit.get("selection_manifest")
        prediction_ref = audit.get("prediction_output")
        checkpoint_hashes = {
            horizon: value.get("sha256")
            for horizon, value in audit.get("horizon_checkpoints", {}).items()
        }
        audited_horizons = audit.get("selected_horizons")
    verify_reference(manifest_ref, f"{name} Test selection manifest", manifest_path)
    selection_manifest_hash = sha256(manifest_path)
    if manifest_ref.get("sha256") != selection_manifest_hash:
        raise RuntimeError(f"Candidate {name!r} Test audit used another selection manifest")
    verify_reference(prediction_ref, f"{name} Test predictions", run / "test_predictions.csv")
    if audited_horizons != expected_horizons:
        raise RuntimeError(f"Candidate {name!r} Test horizons differ from the freeze")
    for horizon in expected_horizons:
        expected_hash = manifest["selections"][horizon]["selected_checkpoint_sha256"][name]
        if checkpoint_hashes.get(horizon) != expected_hash:
            raise RuntimeError(f"Candidate {name!r} Test audit checkpoint mismatch: {horizon}")
    return {
        "candidate": name,
        "kind": kind,
        "selected_horizons": expected_horizons,
        "test_audit": file_reference(audit_path),
        "selection_manifest_sha256": selection_manifest_hash,
    }


def validate_frozen_selection_manifest(
    manifest_path: Path,
    *,
    require_test_artifacts: bool = False,
) -> tuple[dict, dict, dict[str, Path], dict[str, dict]]:
    """Validate the complete pre-Test freeze before any Test prediction is opened."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = read_json(manifest_path)
    if manifest.get("selection_split") != "selection_valid":
        raise ValueError("Selection manifest was not frozen from selection_valid")
    if manifest.get("test_files_read") is not False:
        raise ValueError("Selection manifest is not a pre-Test freeze: test_files_read must be false")
    protocol_path = Path(manifest.get("protocol", "")).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_body, protocol_hash = load_protocol(protocol_path)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Frozen protocol hash changed or is not authenticated")
    candidates_raw = manifest.get("candidates")
    frozen_candidates = manifest.get("candidate_freeze")
    selections = manifest.get("selections")
    if not isinstance(candidates_raw, dict) or not isinstance(frozen_candidates, dict):
        raise ValueError("Selection manifest has no candidate freeze")
    if not isinstance(selections, dict) or set(selections) != set(HORIZONS):
        raise ValueError("Selection manifest does not contain exactly the four horizons")
    candidates = {name: Path(path).expanduser().resolve() for name, path in candidates_raw.items()}
    expected_experiments = {value["id"] for value in protocol_body["experiments"]}
    if set(candidates) != expected_experiments:
        raise RuntimeError("Selection manifest candidates differ from the frozen experiment matrix")
    normalized: dict[str, dict] = {}
    for name, run in candidates.items():
        current = validate_candidate_manifest(name, run, protocol_body)
        if name not in frozen_candidates:
            raise ValueError(f"Candidate {name!r} is absent from candidate_freeze")
        _same_frozen_candidate(frozen_candidates[name], current, name)
        expected_prediction_hash = manifest.get("input_sha256", {}).get(
            f"{name}:selection_valid_predictions"
        )
        if expected_prediction_hash != current["selection_valid_predictions"]["sha256"]:
            raise RuntimeError(f"Candidate {name!r} Selection prediction hash is not frozen")
        normalized[name] = current
    selected_candidates: set[str] = set()
    for horizon in HORIZONS:
        selection = selections[horizon]
        names = selection.get("selected_components") if isinstance(selection, dict) else None
        frozen_hashes = selection.get("selected_checkpoint_sha256") if isinstance(selection, dict) else None
        if not isinstance(names, list) or not names or not isinstance(frozen_hashes, dict):
            raise ValueError(f"Invalid selected components/checkpoint freeze for {horizon}")
        for name in names:
            if name not in normalized or horizon not in normalized[name]["checkpoints"]:
                raise RuntimeError(f"Candidate {name!r} cannot supply selected horizon {horizon}")
            actual_hash = normalized[name]["checkpoints"][horizon]["sha256"]
            if frozen_hashes.get(name) != actual_hash:
                raise RuntimeError(f"Selected checkpoint hash changed for {name}/{horizon}")
            selected_candidates.add(name)
    if require_test_artifacts:
        for name in selected_candidates:
            _validate_candidate_test_artifacts(
                manifest_path, manifest, candidates, normalized, name
            )
    return manifest, protocol_body, candidates, normalized


def validate_one_candidate(protocol: Path, candidate_name: str, run: Path) -> dict:
    protocol_body, protocol_hash = load_protocol(protocol)
    expected_experiments = {value["id"] for value in protocol_body["experiments"]}
    if candidate_name not in expected_experiments:
        raise RuntimeError(f"Candidate {candidate_name!r} is not registered in the protocol")
    result = validate_candidate_manifest(candidate_name, run, protocol_body)
    return {"protocol_sha256": protocol_hash, "candidate": candidate_name, **result}


def validate_one_candidate_test(manifest_path: Path, candidate_name: str) -> dict:
    manifest, _, candidates, normalized = validate_frozen_selection_manifest(
        manifest_path, require_test_artifacts=False
    )
    return _validate_candidate_test_artifacts(
        manifest_path.expanduser().resolve(), manifest, candidates, normalized, candidate_name
    )


def validate_test_evaluation(manifest_path: Path, output_directory: Path) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    validate_frozen_selection_manifest(manifest_path, require_test_artifacts=True)
    completion_path = output_directory / "test_completion_audit.json"
    completion = read_json(completion_path)
    if completion.get("status") != "test_complete" or completion.get("test_read") is not True:
        raise ValueError("Aggregate Test completion audit is incomplete")
    verify_reference(
        completion.get("selection_manifest"),
        "aggregate Test selection manifest",
        manifest_path,
    )
    required = (
        "test_predictions.csv",
        "test_summary.csv",
        "evaluated_selection_manifest.json",
    )
    for filename in required:
        verify_reference(
            completion.get("artifacts", {}).get(filename),
            f"aggregate Test artifact {filename}",
            output_directory / filename,
        )
    evaluated = read_json(output_directory / "evaluated_selection_manifest.json")
    if evaluated.get("test_files_read") is not True:
        raise ValueError("Evaluated selection manifest does not record completed Test access")
    if evaluated.get("protocol_sha256") != read_json(manifest_path).get("protocol_sha256"):
        raise RuntimeError("Aggregate Test evaluation used another frozen protocol")
    return {
        "status": "test_complete",
        "test_read": True,
        "selection_manifest_sha256": sha256(manifest_path),
        "completion_audit": file_reference(completion_path),
        "artifacts": {filename: file_reference(output_directory / filename) for filename in required},
    }


def parse_candidates(values: list[str]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be NAME=RUN_DIRECTORY: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in candidates:
            raise ValueError(f"Missing or duplicate candidate name: {name!r}")
        candidates[name] = Path(raw_path).expanduser().resolve()
    if len(candidates) < 1:
        raise ValueError("At least one candidate is required")
    return candidates


def required_columns(horizon: str) -> list[str]:
    return [
        *KEYS,
        f"{horizon}_log_mean",
        f"{horizon}_log_variance",
        f"{horizon}_expected_return",
        f"{horizon}_probability_positive",
        f"{horizon}_actual_return",
    ]


def load_prediction(run: Path, split: str, horizon: str) -> tuple[pd.DataFrame, Path]:
    path = run / f"{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=lambda column: column in required_columns(horizon))
    missing = set(required_columns(horizon)) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate date/instrument rows")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values(KEYS).reset_index(drop=True), path


def supports_horizon(run: Path, split: str, horizon: str) -> bool:
    path = run / f"{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = set(pd.read_csv(path, nrows=0).columns)
    return set(required_columns(horizon)).issubset(columns)


def align_components(
    candidates: dict[str, Path], split: str, horizon: str, component_names: list[str]
) -> tuple[pd.DataFrame, dict[str, Path]]:
    merged: pd.DataFrame | None = None
    paths: dict[str, Path] = {}
    actual_column = f"{horizon}_actual_return"
    for name in component_names:
        frame, path = load_prediction(candidates[name], split, horizon)
        paths[name] = path
        renamed = frame.rename(columns={
            f"{horizon}_log_mean": f"{name}__mean",
            f"{horizon}_log_variance": f"{name}__variance",
            f"{horizon}_expected_return": f"{name}__expected",
            f"{horizon}_probability_positive": f"{name}__positive",
            actual_column: f"{name}__actual",
        })
        merged = renamed if merged is None else merged.merge(
            renamed, on=KEYS, how="inner", validate="one_to_one"
        )
    assert merged is not None
    if merged.empty:
        raise ValueError(f"No common rows for {split}/{horizon}/{component_names}")
    reference = merged[f"{component_names[0]}__actual"].to_numpy(float)
    for name in component_names[1:]:
        actual = merged[f"{name}__actual"].to_numpy(float)
        if not np.allclose(reference, actual, rtol=1e-7, atol=1e-9, equal_nan=True):
            raise ValueError(f"Realized labels disagree between candidates for {horizon}")
    merged["actual_return"] = reference
    return merged, paths


def mixture_predictions(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    means = np.column_stack([frame[f"{name}__mean"].to_numpy(float) for name in names])
    variances = np.column_stack([frame[f"{name}__variance"].to_numpy(float) for name in names])
    probabilities = np.column_stack([frame[f"{name}__positive"].to_numpy(float) for name in names])
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Mixture components contain invalid means or variances")
    mixture_mean = means.mean(axis=1)
    mixture_variance = (variances + means**2).mean(axis=1) - mixture_mean**2
    mixture_variance = np.maximum(mixture_variance, 1e-12)
    # Expected ordinary return of a Gaussian mixture is the weighted average of
    # component log-normal expectations, not exp(mixture mean).
    expected = np.mean(np.exp(means + 0.5 * variances) - 1.0, axis=1)
    return pd.DataFrame({
        "datetime": frame["datetime"],
        "instrument": frame["instrument"],
        "log_mean": mixture_mean,
        "log_variance": mixture_variance,
        "expected_return": expected,
        "probability_positive": probabilities.mean(axis=1),
        "actual_return": frame["actual_return"].to_numpy(float),
    })


def standardize_horizon(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    variance = frame["log_variance"].to_numpy(float)
    mean = frame["log_mean"].to_numpy(float)
    ordinary_variance = (np.exp(variance) - 1.0) * np.exp(2.0 * mean + variance)
    return frame[["datetime", "instrument"]].assign(**{
        f"{horizon}_log_mean": mean,
        f"{horizon}_log_variance": variance,
        f"{horizon}_expected_return": frame["expected_return"].to_numpy(float),
        f"{horizon}_return_std": np.sqrt(np.maximum(ordinary_variance, 0.0)),
        f"{horizon}_probability_positive": frame["probability_positive"].to_numpy(float),
        f"{horizon}_actual_return": frame["actual_return"].to_numpy(float),
    })


def merge_standardized_horizons(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("Selected horizons do not share prediction rows")
    return result.sort_values(KEYS).reset_index(drop=True)


def exact_mixture_nll(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    actual = frame["actual_return"].to_numpy(float)
    valid = np.isfinite(actual) & (actual > -1.0)
    target = np.full_like(actual, np.nan)
    target[valid] = np.log1p(actual[valid])
    log_probabilities = []
    for name in names:
        mean = frame[f"{name}__mean"].to_numpy(float)
        variance = frame[f"{name}__variance"].to_numpy(float)
        log_probabilities.append(
            -0.5 * ((target - mean) ** 2 / variance + np.log(variance) + math.log(2 * math.pi))
        )
    values = np.column_stack(log_probabilities)
    maximum = np.nanmax(values, axis=1)
    log_mixture = maximum + np.log(np.exp(values - maximum[:, None]).mean(axis=1))
    result = -log_mixture
    result[~valid] = np.nan
    return result


def score(frame: pd.DataFrame, component_frame: pd.DataFrame, names: list[str]) -> dict[str, float | int | None]:
    daily = []
    for _, day in frame.groupby("datetime", sort=True):
        usable = np.isfinite(day["expected_return"]) & np.isfinite(day["actual_return"])
        selected = day.loc[usable]
        rank_ic = (selected["expected_return"].corr(selected["actual_return"], method="spearman")
                   if len(selected) > 1 else np.nan)
        daily.append(rank_ic)
    rank_ics = np.asarray(daily, dtype=float)
    rank_ics = rank_ics[np.isfinite(rank_ics)]
    mean_rank_ic = float(rank_ics.mean()) if len(rank_ics) else float("nan")
    std_rank_ic = float(rank_ics.std(ddof=0)) if len(rank_ics) else float("nan")
    nll = exact_mixture_nll(component_frame, names)
    return {
        "components": len(names),
        "days": int(frame["datetime"].nunique()),
        "rows": int(len(frame)),
        "rank_ic": mean_rank_ic,
        "rank_icir": mean_rank_ic / std_rank_ic if std_rank_ic > 0 else None,
        "nll": float(np.nanmean(nll)),
        "mae": float(np.nanmean(np.abs(frame["expected_return"] - frame["actual_return"]))),
        "brier": float(np.nanmean((frame["probability_positive"] - (frame["actual_return"] > 0)) ** 2)),
    }


def ranked_candidate_names(candidates: dict[str, Path], horizon: str) -> tuple[list[str], dict[str, dict]]:
    metrics: dict[str, dict] = {}
    for name in candidates:
        if not supports_horizon(candidates[name], "selection_valid", horizon):
            continue
        source, _ = align_components(candidates, "selection_valid", horizon, [name])
        prediction = mixture_predictions(source, [name])
        metrics[name] = score(prediction, source, [name])
    eligible = [name for name, metric in metrics.items()
                if np.isfinite(metric["rank_ic"]) and metric["rank_ic"] > 0]
    eligible.sort(key=lambda name: (metrics[name]["rank_ic"], metrics[name]["rank_icir"] or -np.inf), reverse=True)
    return eligible, metrics


def combination_key(item: dict) -> tuple[float, float, float, int]:
    metric = item["metrics"]
    return (
        metric["rank_ic"],
        metric["rank_icir"] if metric["rank_icir"] is not None else -np.inf,
        -metric["nll"],
        -metric["components"],
    )


def select(protocol: Path, candidates: dict[str, Path], output: Path) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Selection manifest already exists: {output}")
    if output.parent.exists():
        raise FileExistsError(
            f"Selection publication directory already exists without a manifest: {output.parent}"
        )
    protocol = protocol.expanduser().resolve()
    protocol_body, protocol_hash = load_protocol(protocol)
    expected_experiments = {value["id"] for value in protocol_body["experiments"]}
    if set(candidates) != expected_experiments:
        raise RuntimeError(
            "Candidate names must exactly match the frozen protocol experiments: "
            f"expected {sorted(expected_experiments)}, got {sorted(candidates)}"
        )
    candidate_freeze = {
        name: validate_candidate_manifest(name, path, protocol_body)
        for name, path in candidates.items()
    }
    selections: dict[str, dict] = {}
    file_hashes: dict[str, str] = {
        f"{name}:selection_valid_predictions": value["selection_valid_predictions"]["sha256"]
        for name, value in candidate_freeze.items()
    }
    selected_prediction_frames: list[pd.DataFrame] = []
    for horizon in protocol_body["horizons"]:
        eligible, individual = ranked_candidate_names(candidates, horizon)
        if not eligible:
            raise RuntimeError(f"No positive selection_valid RankIC candidate for {horizon}")
        groups = [eligible[:1]]
        if len(eligible) >= 2:
            groups.append(eligible[:2])
        if len(eligible) >= 3:
            groups.append(eligible)
        unique_groups = []
        for group in groups:
            if group not in unique_groups:
                unique_groups.append(group)
        groups = unique_groups
        alternatives = []
        for names in groups:
            source, paths = align_components(candidates, "selection_valid", horizon, names)
            prediction = mixture_predictions(source, names)
            metrics = score(prediction, source, names)
            alternatives.append({"components": names, "metrics": metrics})
            for name, path in paths.items():
                file_hashes[f"{name}:selection_valid_predictions"] = sha256(path)
        selected = max(alternatives, key=combination_key)
        selected_source, _ = align_components(
            candidates, "selection_valid", horizon, selected["components"]
        )
        selected_prediction_frames.append(standardize_horizon(
            mixture_predictions(selected_source, selected["components"]), horizon
        ))
        selections[horizon] = {
            "individual_metrics": individual,
            "alternatives": alternatives,
            "selected_components": selected["components"],
            "selected_checkpoint_sha256": {
                name: candidate_freeze[name]["checkpoints"][horizon]["sha256"]
                for name in selected["components"]
            },
            "selection_valid_metrics": selected["metrics"],
        }
    final_predictions = output.parent / "selection_valid_ensemble_predictions.csv"
    staging = output.parent.parent / f".{output.parent.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        staging_predictions = staging / final_predictions.name
        staging_manifest = staging / output.name
        merge_standardized_horizons(selected_prediction_frames).to_csv(
            staging_predictions, index=False
        )
        manifest = {
            "schema_version": 1,
            "protocol": str(protocol),
            "protocol_sha256": protocol_hash,
            "protocol_segments": protocol_body["segments"],
            "protocol_data_manifest_sha256": protocol_body["data_manifest_sha256"],
            "selection_split": "selection_valid",
            "test_files_read": False,
            "candidates": {name: str(path) for name, path in candidates.items()},
            "candidate_freeze": candidate_freeze,
            "input_sha256": file_hashes,
            "selection_valid_ensemble_predictions": str(final_predictions),
            "selection_valid_ensemble_predictions_sha256": sha256(staging_predictions),
            "selections": selections,
        }
        staging_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # One directory rename publishes the CSV and immutable manifest
        # together.  A failed write leaves no final directory and is retryable.
        staging.replace(output.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def evaluate(manifest_path: Path, output_directory: Path) -> None:
    # This complete preflight authenticates the protocol, candidate manifests,
    # Selection inputs, selected checkpoints, and candidate Test completion
    # audits before load_prediction opens any held-out Test CSV.
    manifest_path = manifest_path.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    manifest, protocol_body, candidates, _ = validate_frozen_selection_manifest(
        manifest_path, require_test_artifacts=True
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    summary, standardized = [], []
    for horizon, selection in manifest["selections"].items():
        names = selection["selected_components"]
        source, paths = align_components(candidates, "test", horizon, names)
        prediction = mixture_predictions(source, names)
        metrics = score(prediction, source, names)
        prediction.to_csv(output_directory / f"{horizon}_test_predictions.csv", index=False)
        standardized.append(standardize_horizon(prediction, horizon))
        summary.append({"horizon": horizon, "components": "+".join(names), **metrics})
        for name, path in paths.items():
            manifest.setdefault("test_input_sha256", {})[f"{name}:test_predictions"] = sha256(path)
    summary_path = output_directory / "test_summary.csv"
    predictions_path = output_directory / "test_predictions.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    merge_standardized_horizons(standardized).to_csv(predictions_path, index=False)
    evaluated = dict(manifest)
    evaluated["test_files_read"] = True
    evaluated["test_evaluation_directory"] = str(output_directory.resolve())
    evaluated["test_input_sha256"] = manifest.get("test_input_sha256", {})
    evaluated_path = output_directory / "evaluated_selection_manifest.json"
    evaluated_path.write_text(
        json.dumps(evaluated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts = {
        path.name: file_reference(path)
        for path in [
            *sorted(output_directory.glob("*_test_predictions.csv")),
            summary_path,
            predictions_path,
            evaluated_path,
        ]
    }
    completion = {
        "schema_version": 1,
        "status": "test_complete",
        "test_read": True,
        "selection_manifest": file_reference(manifest_path),
        "protocol": file_reference(Path(manifest["protocol"])),
        "protocol_segments": protocol_body["segments"],
        "selected_checkpoint_sha256": {
            horizon: manifest["selections"][horizon]["selected_checkpoint_sha256"]
            for horizon in HORIZONS
        },
        "artifacts": artifacts,
    }
    (output_directory / "test_completion_audit.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    selection = subparsers.add_parser("select")
    selection.add_argument("--protocol", type=Path, required=True)
    selection.add_argument("--candidate", action="append", default=[], help="NAME=RUN_DIRECTORY")
    selection.add_argument("--output", type=Path, required=True)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--output-directory", type=Path, required=True)
    candidate_validation = subparsers.add_parser("validate-candidate")
    candidate_validation.add_argument("--protocol", type=Path, required=True)
    candidate_validation.add_argument("--candidate-name", required=True)
    candidate_validation.add_argument("--candidate-directory", type=Path, required=True)
    selection_validation = subparsers.add_parser("validate-selection")
    selection_validation.add_argument("--manifest", type=Path, required=True)
    candidate_test_validation = subparsers.add_parser("validate-candidate-test")
    candidate_test_validation.add_argument("--manifest", type=Path, required=True)
    candidate_test_validation.add_argument("--candidate-name", required=True)
    test_validation = subparsers.add_parser("validate-test-evaluation")
    test_validation.add_argument("--manifest", type=Path, required=True)
    test_validation.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select(args.protocol, parse_candidates(args.candidate), args.output)
    elif args.command == "evaluate":
        evaluate(args.manifest, args.output_directory)
    elif args.command == "validate-candidate":
        print(json.dumps(validate_one_candidate(
            args.protocol, args.candidate_name, args.candidate_directory
        ), ensure_ascii=False, indent=2))
    elif args.command == "validate-selection":
        manifest, _, _, _ = validate_frozen_selection_manifest(
            args.manifest, require_test_artifacts=False
        )
        print(json.dumps({
            "status": "selection_frozen",
            "test_files_read": manifest["test_files_read"],
            "selection_manifest_sha256": sha256(args.manifest),
        }, ensure_ascii=False, indent=2))
    elif args.command == "validate-candidate-test":
        print(json.dumps(validate_one_candidate_test(
            args.manifest, args.candidate_name
        ), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(validate_test_evaluation(
            args.manifest, args.output_directory
        ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
