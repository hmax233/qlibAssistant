from __future__ import annotations

import json
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "script/run_alpha360_e6_combined_windows.ps1"
BASE_WORKER = ROOT / "script/run_alpha360_probabilistic_matrix_windows.ps1"
TRAINER = ROOT / "script/train_alpha360_cross_market.py"
PROTOCOL = (
    ROOT
    / "script/alpha360_experiments/fixed_fold3_probabilistic_cross_market_v1.json"
)


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_base_pipeline_stops_before_test_for_blind_e6_extension() -> None:
    source = text(BASE_WORKER)
    assert "$SelectionOnly = $true" in source
    stop = source.index("if ($SelectionOnly)")
    test_access = source.index("# Test authorization", stop)
    assert "status='selection_ready'" in source[stop:test_access]
    assert "test_access_authorized=$false" in source[stop:test_access]
    assert "test_read=$false" in source[stop:test_access]


def test_cross_market_protocol_freezes_all_seven_candidates_and_e6_data() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    ids = [item["id"] for item in protocol["experiments"]]
    assert ids == [
        "E0_joint_three_leg",
        "E1_shared_four_head",
        "E2_single_open1_close2",
        "E3_single_close1_open2",
        "E4_single_open1_open2",
        "E5_single_close1_close2",
        "E6_a_us_four_head",
    ]
    e6 = protocol["experiments"][-1]
    assert len(e6["data_manifest_sha256"]) == 64
    optimization = protocol["optimization"]
    assert optimization["epochs"] == 50
    assert optimization["date_batch_size"] == 4
    assert optimization["gradient_accumulation"] is False
    assert optimization["gradient_clipping"] is False
    assert optimization["minimum_learning_rate"] == 1e-6
    scoring = protocol["selection_scoring"]
    assert scoring["canonical_key_sha256"] == (
        "aece545199bca9bd0bb331a160ed390e8577d12e37f8f76be4625c5bc87670e1"
    )
    assert scoring["checksum_amendment"]["previous_erroneous_sha256"] == (
        "50803d9c3c1a56c979766be1a781f4e9e9c712885681857558923cf15473ab9e"
    )


def test_e6_worker_waits_for_selection_and_freezes_before_test() -> None:
    source = text(WORKER)
    wait = source.index("status='waiting_for_e0_e5_selection'")
    train = source.index("Train E6 A+US cross-market model for 50 epochs")
    freeze = source.index("Select and freeze E0-E6 ensemble on Selection-valid")
    authorize = source.index("test_access_authorized=$true", freeze)
    test_materialization = source.index("Materialize selected E0 Test horizons", authorize)
    aggregate = source.index("Evaluate frozen E0-E6 ensemble on Test", test_materialization)
    assert wait < train < freeze < authorize < test_materialization < aggregate
    pretest_reauthentication = source.index(
        "Re-authenticate E0-E6 freeze immediately before Test access", freeze
    )
    assert pretest_reauthentication < authorize < test_materialization
    assert "'--date-batch-size', '4'" in source
    assert "'--minimum-learning-rate', '0.000001'" in source
    assert "--gradient-clipping" not in source
    assert "--gradient-accumulation" not in source
    validation = source[
        source.index("function Test-PythonValidation"):
        source.index("function Move-DirectoryToArchive")
    ]
    assert "$ErrorActionPreference = 'Continue'" in validation
    assert "$ErrorActionPreference = $PreviousErrorActionPreference" in validation


def test_e6_trainer_publishes_standard_selection_and_test_audits() -> None:
    source = text(TRAINER)
    tree = ast.parse(source)
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    evaluate = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_test"
    )
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    evaluate_source = ast.get_source_segment(source, evaluate) or ""
    main_source = ast.get_source_segment(source, main) or ""
    assert main_source.index("validate_complete_freeze(") < main_source.index(
        'Trainer(args, command="evaluate-test")'
    )
    assert evaluate_source.index("self.store.verify_parts((TEST_SPLIT,))") < evaluate_source.index(
        '"status": "test_complete"'
    )
    assert "write_selection_candidate_manifest(" in source
    assert '"selection_candidate_manifest.json"' in source
    assert '"test_completion_audit.json"' in source
    assert '"gradient_accumulation": False' in source
    assert '"gradient_clipping": False' in source
