from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "script/finalize_alpha360_e0_e6_local.py"


def test_finalizer_is_one_shot_and_never_creates_a_polling_process() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "while True" not in source
    assert "time.sleep(" not in source
    assert "nohup" not in source
    assert "Popen(" not in source
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 3
        for node in ast.walk(tree)
    )


def test_finalizer_runs_both_strict_variants_and_read_only_report() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    assert 'run_backtest(staging, "mainboard")' in source
    assert 'run_backtest(staging, "all")' in source
    assert '"--commission-rate", "0.000235"' in source
    assert '"--minimum-commission", "5"' in source
    assert '"--slippage-bps", "0", "5"' in source
    assert 'report_alpha360_probabilistic_experiments.py' in source
    assert 'report_alpha360_training_curves.py' in source
    assert '"E6_a_us_four_head"' in source
    assert '"--expected-epochs", "50"' in source
    assert "staging.replace(output)" in source
