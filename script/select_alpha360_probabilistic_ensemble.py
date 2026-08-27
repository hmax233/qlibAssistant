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
    if output.exists():
        raise FileExistsError(f"Selection manifest already exists: {output}")
    protocol_body = json.loads(protocol.read_text(encoding="utf-8"))
    selections: dict[str, dict] = {}
    file_hashes: dict[str, str] = {}
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
            "selection_valid_metrics": selected["metrics"],
        }
    selection_predictions = output.parent / "selection_valid_ensemble_predictions.csv"
    if selection_predictions.exists():
        raise FileExistsError(selection_predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    merge_standardized_horizons(selected_prediction_frames).to_csv(
        selection_predictions, index=False
    )
    manifest = {
        "schema_version": 1,
        "protocol": str(protocol.resolve()),
        "protocol_sha256": sha256(protocol),
        "selection_split": "selection_valid",
        "test_files_read": False,
        "candidates": {name: str(path) for name, path in candidates.items()},
        "input_sha256": file_hashes,
        "selection_valid_ensemble_predictions": str(selection_predictions.resolve()),
        "selection_valid_ensemble_predictions_sha256": sha256(selection_predictions),
        "selections": selections,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def evaluate(manifest_path: Path, output_directory: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = {name: Path(path) for name, path in manifest["candidates"].items()}
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
    pd.DataFrame(summary).to_csv(output_directory / "test_summary.csv", index=False)
    merge_standardized_horizons(standardized).to_csv(
        output_directory / "test_predictions.csv", index=False
    )
    manifest["test_files_read"] = True
    manifest["test_evaluation_directory"] = str(output_directory.resolve())
    (output_directory / "evaluated_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
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
    args = parser.parse_args()
    if args.command == "select":
        select(args.protocol, parse_candidates(args.candidate), args.output)
    else:
        evaluate(args.manifest, args.output_directory)


if __name__ == "__main__":
    main()
