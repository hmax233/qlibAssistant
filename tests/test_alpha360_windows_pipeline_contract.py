"""Executable contracts for the resumable Windows Alpha360 pipeline."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from script.select_alpha360_probabilistic_ensemble import evaluate, select, sha256
from tests.test_select_alpha360_probabilistic_ensemble import write_protocol, write_run


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PIPELINE = ROOT / "script/run_alpha360_probabilistic_matrix_windows.ps1"
MATERIALIZER = ROOT / "script/materialize_alpha360_joint_checkpoints.py"
TRAINER = ROOT / "script/train_alpha360_decoupled.py"
SELECTOR = ROOT / "script/select_alpha360_probabilistic_ensemble.py"
HORIZONS = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def function_source(path: Path, class_name: str | None, function_name: str) -> str:
    body = source(path)
    tree = ast.parse(body)
    nodes = tree.body
    if class_name is not None:
        owner = next(
            node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = owner.body
    function = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return ast.get_source_segment(body, function) or ""


def prediction_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02"] * 3 + ["2026-01-05"] * 3)
    actual = np.array([-0.02, 0.00, 0.03, -0.01, 0.01, 0.04])
    frame = pd.DataFrame({
        "datetime": dates,
        "instrument": ["A", "B", "C"] * 2,
    })
    for horizon in HORIZONS:
        frame[f"{horizon}_log_mean"] = actual
        frame[f"{horizon}_log_variance"] = 0.01
        frame[f"{horizon}_expected_return"] = np.exp(actual + 0.005) - 1.0
        frame[f"{horizon}_probability_positive"] = 0.5
        frame[f"{horizon}_actual_return"] = actual
    return frame


def write_candidate(path: Path, *, selection: bool = True, test: bool = True) -> None:
    path.mkdir()
    frame = prediction_frame()
    if selection:
        frame.to_csv(path / "selection_valid_predictions.csv", index=False)
    if test:
        frame.to_csv(path / "test_predictions.csv", index=False)


def write_frozen_manifest(path: Path, candidate: Path, *, test_files_read: bool = False) -> None:
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "selection_split": "selection_valid",
            "test_files_read": test_files_read,
            "candidates": {"chosen": str(candidate.resolve())},
            "input_sha256": {},
            "selections": {
                horizon: {"selected_components": ["chosen"]} for horizon in HORIZONS
            },
        }),
        encoding="utf-8",
    )


def test_powershell_forwards_locked_training_parameters_and_resume_flag() -> None:
    text = source(WINDOWS_PIPELINE)
    for token in (
        "'--epochs', '50'",
        "'--learning-rate', '0.0003'",
        "'--min-learning-rate', '0.000001'",
        "'--warmup-epochs', '3'",
        "'--date-batch-size', '4'",
        "$Arguments += '--resume'",
        "& $Python @Arguments",
        "if ($LASTEXITCODE -ne 0)",
    ):
        assert token in text
    assert "--gradient-clipping" not in text
    assert "--gradient-accumulation" not in text


def test_powershell_uses_one_e0_candidate_directory_before_and_after_freeze() -> None:
    text = source(WINDOWS_PIPELINE)
    assert "$E0Candidate = Join-Path $Root 'E0_joint_three_leg'" in text
    assert '"E0_joint_three_leg=$E0Candidate"' in text
    assert "'--output', $E0Candidate" in text
    assert "'--candidate-name', 'E0_joint_three_leg'" in text


def test_powershell_orders_selection_freeze_before_any_test_materialization() -> None:
    text = source(WINDOWS_PIPELINE)
    freeze = text.index("Invoke-CheckedPython $SelectionArguments")
    manifest_read = text.index("$Frozen = Get-Content $SelectionManifest")
    e0_test = text.index("$Materializer, 'evaluate-test'")
    e1_e5_test = text.index("$TrainEntry, 'evaluate-test'")
    final_test = text.index("$Selector, 'evaluate'")
    assert freeze < manifest_read < min(e0_test, e1_e5_test, final_test)
    assert "$SelectedCandidates -contains 'E0_joint_three_leg'" in text
    assert "$SelectedCandidates -notcontains $Experiment.Id" in text


def test_e0_materializer_authenticates_manifest_and_output_before_test_store_access() -> None:
    body = function_source(MATERIALIZER, None, "evaluate_test")
    gate = body.index("validate_frozen_manifest(")
    output_match = body.index("if output != authenticated_candidate")
    collisions = body.index('output / "test_predictions.csv"')
    test_store = body.index('validate_source(source_run, data, "test")')
    assert gate < output_match < collisions < test_store


def test_selector_evaluate_requests_test_only_for_manifest_selected_components() -> None:
    body = function_source(SELECTOR, None, "evaluate")
    selection = body.index('names = selection["selected_components"]')
    test_alignment = body.index('align_components(candidates, "test", horizon, names)')
    assert selection < test_alignment


def test_decoupled_train_and_evaluate_do_not_touch_test_before_manifest_gate() -> None:
    constructor = function_source(TRAINER, "Trainer", "__init__")
    assert ".verify_parts()" not in constructor


def test_decoupled_test_uses_the_checkpoint_frozen_at_selection_time() -> None:
    body = function_source(TRAINER, "Trainer", "evaluate_test")
    freeze = body.index("validate_frozen_selection_manifest(")
    checkpoint_hash = body.index("checkpoint_hash = file_hash(checkpoint_path)")
    checkpoint_configuration = body.index('checkpoint.get("configuration")')
    test_hashes = body.index('self.store.verify_parts("test")')
    test_evaluation = body.index('evaluate_current("test"')
    assert freeze < checkpoint_hash < checkpoint_configuration < test_hashes < test_evaluation


def test_selector_rejects_non_pretest_manifest_before_opening_test(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    write_candidate(candidate)
    manifest = tmp_path / "manifest.json"
    write_frozen_manifest(manifest, candidate, test_files_read=True)
    with pytest.raises(ValueError, match="pre-test|test_files_read"):
        evaluate(manifest, tmp_path / "evaluation")


def test_selector_evaluate_failure_does_not_poison_rerun_directory(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    write_candidate(candidate, test=False)
    manifest = tmp_path / "manifest.json"
    write_frozen_manifest(manifest, candidate)
    output = tmp_path / "evaluation"
    with pytest.raises(FileNotFoundError):
        evaluate(manifest, output)
    assert not output.exists()


def test_selector_selection_publish_is_transactional_and_rerunnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    actual = np.array([-0.03, 0.00, 0.03, -0.02, 0.01, 0.04])
    write_run(candidate, actual, actual, data_manifest)
    protocol = tmp_path / "protocol.json"
    write_protocol(protocol, ["candidate"], sha256(data_manifest))
    manifest = tmp_path / "selection" / "selection_manifest.json"
    original = Path.write_text
    failed = False

    def fail_manifest_once(self: Path, *args, **kwargs):
        nonlocal failed
        if (
            self.name == manifest.name
            and self.parent.name.startswith(".selection.staging-")
            and not failed
        ):
            failed = True
            raise OSError("simulated manifest publication failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest_once)
    with pytest.raises(OSError, match="publication failure"):
        select(protocol, {"candidate": candidate}, manifest)
    assert not manifest.parent.exists()
    assert not list(tmp_path.glob(".selection.staging-*"))
    # A transaction either publishes both files or permits an automatic retry.
    select(protocol, {"candidate": candidate}, manifest)
    assert manifest.is_file()
    assert (manifest.parent / "selection_valid_ensemble_predictions.csv").is_file()


def test_selector_enforces_frozen_protocol_against_every_candidate() -> None:
    selection = function_source(SELECTOR, None, "select")
    protocol = function_source(SELECTOR, None, "load_protocol")
    candidate = function_source(SELECTOR, None, "validate_candidate_manifest")
    configuration = function_source(SELECTOR, None, "validate_configuration_against_protocol")
    assert "load_protocol(protocol)" in selection
    assert "validate_candidate_manifest(name, path, protocol_body)" in selection
    for required in ("data_manifest_sha256", "segments", "experiments", "optimization"):
        assert required in protocol
    assert "data_manifest_sha256" in candidate
    assert "segments" in configuration
    assert "optimization" in configuration


def test_powershell_selection_ready_resume_validates_candidate_contract() -> None:
    text = source(WINDOWS_PIPELINE)
    begin = text.index("if (Test-Path $Status)")
    end = text.index("$Arguments = @(", begin)
    skip_block = text[begin:end]
    for required in (
        "configuration.json",
        "selection_valid_predictions.csv",
        "selection_valid_summary.csv",
        "data_manifest_sha256",
        "protocol_sha256",
        "test_read",
    ):
        assert required in skip_block


def test_powershell_e0_resume_authenticates_complete_selection_candidate() -> None:
    text = source(WINDOWS_PIPELINE)
    begin = text.index("# E0 resume is accepted")
    end = text.index("foreach ($Experiment in $Experiments)", begin)
    block = text[begin:end]
    assert "materialization_manifest.json" in block
    assert "sha256" in block.lower()


def test_powershell_test_resume_authenticates_all_required_artifacts() -> None:
    text = source(WINDOWS_PIPELINE)
    begin = text.index("foreach ($Experiment in $Experiments)", text.index("materializing_selected_test"))
    end = text.index("$AggregateReady = $false", begin)
    block = text[begin:end]
    for required in ("test_access.json", "test_summary.csv", "selection_manifest_sha256"):
        assert required in block


def test_pipeline_status_distinguishes_authorized_test_access_from_completed_read() -> None:
    text = source(WINDOWS_PIPELINE)
    begin = text.index("status='materializing_selected_test'")
    end = text.index("if ($SelectedCandidates -contains", begin)
    block = text[begin:end]
    assert "test_access_authorized=$true" in block
    assert "test_read=$false" in block


def test_powershell_test_ready_requires_complete_aggregate_artifact_set() -> None:
    text = source(WINDOWS_PIPELINE)
    begin = text.index("$AggregateReady = $false")
    end = text.index("status='test_ready'", begin)
    block = text[begin:end]
    assert "test_summary.csv" in block
    assert "evaluated_selection_manifest.json" in block
    assert "test_completion_audit.json" in block
