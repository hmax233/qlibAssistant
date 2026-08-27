from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from script.select_alpha360_probabilistic_ensemble import (
    evaluate,
    exact_mixture_nll,
    mixture_predictions,
    prediction_key_sha256,
    select,
    sha256,
)


HORIZONS = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")
SEGMENTS = {
    "train": ["2015-04-17", "2025-04-16"],
    "valid": ["2025-04-17", "2025-09-16"],
    "selection_valid": ["2025-09-17", "2026-02-16"],
    "test": ["2026-02-17", "2026-07-17"],
}
OPTIMIZATION = {
    "epochs": 50,
    "early_stopping": False,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "warmup_epochs": 3,
    "warmup_start_factor": 1 / 3,
    "minimum_learning_rate": 0.000001,
    "date_batch_size": 4,
    "gradient_accumulation": False,
    "gradient_clipping": False,
    "seed": 20260827,
    "target_scale": 100.0,
}


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


def write_protocol(path: Path, candidate_names: list[str], data_hash: str) -> None:
    keys = candidate_frame(np.zeros(6), np.zeros(6))[["datetime", "instrument"]]
    path.write_text(json.dumps({
        "segments": SEGMENTS,
        "data_manifest_sha256": data_hash,
        "horizons": list(HORIZONS),
        "experiments": [{"id": name} for name in candidate_names],
        "optimization": OPTIMIZATION,
        "selection_scoring": {
            "signal_start": "2026-01-01",
            "signal_end": "2026-01-02",
            "expected_days": 2,
            "expected_rows": 6,
            "canonical_key_sha256": prediction_key_sha256(keys),
        },
    }), encoding="utf-8")


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def write_run(
    path: Path,
    signal: np.ndarray,
    actual: np.ndarray,
    data_manifest: Path,
    horizons: tuple[str, ...] = HORIZONS,
) -> None:
    path.mkdir()
    frame = candidate_frame(signal, actual)
    columns = [
        column for column in frame
        if column in {"datetime", "instrument"}
        or any(column.startswith(f"{horizon}_") for horizon in horizons)
    ]
    selection_path = path / "selection_valid_predictions.csv"
    frame[columns].to_csv(selection_path, index=False)
    frame[columns].to_csv(path / "test_predictions.csv", index=False)
    configuration = {
        "model_mode": "shared_four_head" if len(horizons) > 1 else "single_horizon",
        "horizon_names": list(horizons),
        "model": {"target_scale": 100.0},
        "segments": SEGMENTS,
        "data_manifest_sha256": sha256(data_manifest),
        **OPTIMIZATION,
    }
    configuration_path = path / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    checkpoints = {}
    for horizon in horizons:
        checkpoint = path / f"best_{horizon}_rank_ic_model.pt"
        checkpoint.write_bytes(f"checkpoint:{path.name}:{horizon}".encode())
        checkpoints[horizon] = {
            **reference(checkpoint),
            "epoch": 7,
            "selection_metric": f"{horizon}_rank_ic",
            "selection_value": 0.1,
        }
    (path / "selection_valid_summary.csv").write_text("horizon\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "selection_complete",
        "candidate_directory": str(path.resolve()),
        "selection_split": "selection_valid",
        "test_files_read": False,
        "configuration": reference(configuration_path),
        "data_manifest": reference(data_manifest),
        "segments": SEGMENTS,
        "selection_valid_predictions": reference(selection_path),
        "selection_valid_summary": reference(path / "selection_valid_summary.csv"),
        "checkpoints": checkpoints,
    }
    (path / "selection_candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_test_completion(candidate: Path, selection_manifest: Path, horizons: list[str]) -> None:
    test_predictions = candidate / "test_predictions.csv"
    test_summary = candidate / "test_summary.csv"
    test_access = candidate / "test_access.json"
    test_summary.write_text("horizon\n", encoding="utf-8")
    test_access.write_text(json.dumps({
        "status": "test_access_complete", "test_read": True,
        "selection_manifest": reference(selection_manifest),
    }), encoding="utf-8")
    audit = {
        "schema_version": 1,
        "status": "test_complete",
        "test_read": True,
        "selection_manifest": reference(selection_manifest),
        "candidate_manifest": reference(candidate / "selection_candidate_manifest.json"),
        "selected_horizons": horizons,
        "checkpoint_sha256": {
            horizon: sha256(candidate / f"best_{horizon}_rank_ic_model.pt")
            for horizon in horizons
        },
        "artifacts": {
            "test_predictions.csv": reference(test_predictions),
            "test_summary.csv": reference(test_summary),
            "test_access.json": reference(test_access),
        },
    }
    (candidate / "test_completion_audit.json").write_text(json.dumps(audit), encoding="utf-8")


def write_joint_e0_run(
    path: Path,
    signal: np.ndarray,
    actual: np.ndarray,
    data_manifest: Path,
) -> None:
    path.mkdir()
    frame = candidate_frame(signal, actual)
    selection_path = path / "selection_valid_predictions.csv"
    frame.to_csv(selection_path, index=False)
    frame.to_csv(path / "test_predictions.csv", index=False)
    configuration = {
        "segments": SEGMENTS,
        "data_manifest_sha256": sha256(data_manifest),
        **{key: value for key, value in OPTIMIZATION.items() if key != "minimum_learning_rate"},
        "min_learning_rate": OPTIMIZATION["minimum_learning_rate"],
    }
    configuration_path = path / "source_configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    checkpoints = {}
    for horizon in HORIZONS:
        checkpoint = path / f"source_best_{horizon}.pt"
        checkpoint.write_bytes(f"e0:{horizon}".encode())
        checkpoints[horizon] = {
            **reference(checkpoint),
            "epoch": 9,
            "selection_metric": f"{horizon}_rank_ic",
            "selection_value": 0.2,
        }
    audit = {
        "schema_version": 1,
        "command": "selection",
        "split": "selection_valid",
        "test_read": False,
        "source_configuration": reference(configuration_path),
        "data_manifest": reference(data_manifest),
        "prediction_output": reference(selection_path),
        "horizon_checkpoints": checkpoints,
    }
    (path / "materialization_manifest.json").write_text(json.dumps(audit), encoding="utf-8")


def write_joint_e0_test_completion(
    candidate: Path,
    selection_manifest: Path,
    horizons: list[str],
) -> None:
    selection_audit = json.loads((candidate / "materialization_manifest.json").read_text())
    audit = {
        "schema_version": 1,
        "command": "evaluate-test",
        "split": "test",
        "test_read": True,
        "selection_manifest": reference(selection_manifest),
        "selected_horizons": horizons,
        "prediction_output": reference(candidate / "test_predictions.csv"),
        "horizon_checkpoints": {
            horizon: selection_audit["horizon_checkpoints"][horizon]
            for horizon in horizons
        },
    }
    (candidate / "test_materialization_manifest.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )


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
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(good, actual, actual, data_manifest)
    write_run(weak, actual[::-1], actual, data_manifest)
    # If select accidentally opens test, this invalid CSV must make it fail.
    (good / "test_predictions.csv").write_text("forbidden", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["good", "weak"], sha256(data_manifest))
    output = tmp_path / "selection" / "selection.json"
    select(protocol, {"good": good, "weak": weak}, output)
    manifest = json.loads(output.read_text())
    assert manifest["test_files_read"] is False
    assert all(value["selected_components"] == ["good"] for value in manifest["selections"].values())
    metrics = manifest["selections"]["close1_close2"]["selection_valid_metrics"]
    for name in (
        "coverage_50", "coverage_80", "coverage_95", "direction_accuracy",
        "top1_win_rate", "top3_mean_return", "top5_cumulative", "top10_stock_win_rate",
    ):
        assert name in metrics
    selection_predictions = pd.read_csv(output.parent / "selection_valid_ensemble_predictions.csv")
    assert all(f"{horizon}_expected_return" in selection_predictions for horizon in HORIZONS)


def test_selection_excludes_legacy_rows_crossing_the_test_label_boundary(
    tmp_path: Path,
) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    e0, strict = tmp_path / "E0_joint_three_leg", tmp_path / "strict"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_joint_e0_run(e0, actual, actual, data_manifest)
    write_run(strict, actual, actual, data_manifest)

    selection_path = e0 / "selection_valid_predictions.csv"
    frame = pd.read_csv(selection_path)
    extra = frame.iloc[:3].copy()
    extra["datetime"] = "2026-02-12"
    pd.concat([frame, extra], ignore_index=True).to_csv(selection_path, index=False)
    audit_path = e0 / "materialization_manifest.json"
    audit = json.loads(audit_path.read_text())
    audit["prediction_output"] = reference(selection_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["E0_joint_three_leg", "strict"], sha256(data_manifest))
    output = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"E0_joint_three_leg": e0, "strict": strict}, output)
    manifest = json.loads(output.read_text())
    e0_audit = manifest["selection_scoring_keys"]["candidate_audit"]["E0_joint_three_leg"]
    assert e0_audit["raw_rows"] == 9
    assert e0_audit["scored_rows"] == 6
    assert e0_audit["excluded_rows_outside_scoring_range"] == 3
    selected = pd.read_csv(output.parent / "selection_valid_ensemble_predictions.csv")
    assert len(selected) == 6
    assert selected["datetime"].max() == "2026-01-02"


def test_selection_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "selection" / "selection.json"
    output.parent.mkdir()
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        select(tmp_path / "missing.json", {}, output)


def test_selection_supports_single_horizon_candidate_files(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    shared, single = tmp_path / "shared", tmp_path / "single"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(shared, actual * 0.5, actual, data_manifest)
    write_run(single, actual, actual, data_manifest, ("close1_close2",))
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["shared", "single"], sha256(data_manifest))
    output = tmp_path / "selection" / "selection.json"
    select(protocol, {"shared": shared, "single": single}, output)
    manifest = json.loads(output.read_text())
    assert "single" not in manifest["selections"]["open1_close2"]["individual_metrics"]
    assert "single" in manifest["selections"]["close1_close2"]["individual_metrics"]


def test_selection_freezes_candidate_and_selected_checkpoint_hashes(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], sha256(data_manifest))
    output = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"candidate": candidate}, output)
    manifest = json.loads(output.read_text())
    assert manifest["candidate_freeze"]["candidate"]["candidate_manifest"] == reference(
        candidate / "selection_candidate_manifest.json"
    )
    for horizon in HORIZONS:
        assert manifest["selections"][horizon]["selected_checkpoint_sha256"] == {
            "candidate": sha256(candidate / f"best_{horizon}_rank_ic_model.pt")
        }


def test_selection_rejects_protocol_data_hash_mismatch(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], "wrong-data-hash")
    with pytest.raises(RuntimeError, match="data manifest hash"):
        select(protocol, {"candidate": candidate}, tmp_path / "selection" / "selection.json")


def test_protocol_can_freeze_a_candidate_specific_cross_market_data_hash(
    tmp_path: Path,
) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "e6"
    data_manifest = tmp_path / "e6_data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["e6"], "base-a-share-data-hash")
    body = json.loads(protocol.read_text())
    body["experiments"][0]["data_manifest_sha256"] = sha256(data_manifest)
    protocol.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"e6": candidate}, output)
    manifest = json.loads(output.read_text())
    assert manifest["candidate_freeze"]["e6"]["data_manifest"]["sha256"] == sha256(
        data_manifest
    )


def test_evaluate_requires_test_completion_audit_and_writes_aggregate_audit(
    tmp_path: Path,
) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], sha256(data_manifest))
    selection_manifest = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"candidate": candidate}, selection_manifest)
    with pytest.raises(FileNotFoundError):
        evaluate(selection_manifest, tmp_path / "missing-audit-evaluation")
    write_test_completion(candidate, selection_manifest, list(HORIZONS))
    output = tmp_path / "evaluation"
    evaluate(selection_manifest, output)
    audit = json.loads((output / "test_completion_audit.json").read_text())
    assert audit["status"] == "test_complete"
    assert audit["selection_manifest"] == reference(selection_manifest)
    assert audit["artifacts"]["test_predictions.csv"] == reference(
        output / "test_predictions.csv"
    )


def test_evaluate_rejects_checkpoint_changed_after_selection_before_test_read(
    tmp_path: Path,
) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], sha256(data_manifest))
    selection_manifest = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"candidate": candidate}, selection_manifest)
    checkpoint = candidate / "best_close1_close2_rank_ic_model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="hash mismatch|changed after ensemble selection"):
        evaluate(selection_manifest, output)
    assert not output.exists()


def test_evaluate_rejects_protocol_changed_after_selection(tmp_path: Path) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], sha256(data_manifest))
    selection_manifest = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"candidate": candidate}, selection_manifest)
    protocol.write_text(protocol.read_text() + "\n", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="protocol hash"):
        evaluate(selection_manifest, output)
    assert not output.exists()


def test_joint_e0_candidate_uses_materializer_audits_in_strict_pipeline(
    tmp_path: Path,
) -> None:
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    candidate = tmp_path / "E0_joint_three_leg"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    write_joint_e0_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["E0_joint_three_leg"], sha256(data_manifest))
    selection_manifest = tmp_path / "selection" / "selection_manifest.json"
    select(protocol, {"E0_joint_three_leg": candidate}, selection_manifest)
    write_joint_e0_test_completion(candidate, selection_manifest, list(HORIZONS))
    output = tmp_path / "evaluation"
    evaluate(selection_manifest, output)
    assert (output / "test_predictions.csv").is_file()
    assert json.loads((output / "test_completion_audit.json").read_text())["status"] == "test_complete"
