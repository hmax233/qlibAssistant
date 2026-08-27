from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from script.report_alpha360_training_curves import HORIZONS, load_history, render


def history(name: str, epochs: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "train_nll_scaled": np.linspace(3.0, 2.0, epochs),
        "nll_scaled": np.linspace(2.8, 1.9, epochs),
        "learning_rate": np.geomspace(3e-4, 1e-6, epochs),
        "epoch_seconds": np.full(epochs, 10.0),
    })
    for number, horizon_name in enumerate(HORIZONS):
        frame[f"{horizon_name}_rank_ic"] = np.linspace(
            0.01 + number / 100, 0.04 + number / 100, epochs
        )
    return frame


def test_render_writes_auditable_summary_and_nonempty_plot(tmp_path: Path) -> None:
    output = tmp_path / "report"
    summary = render({"E1": history("E1"), "E2": history("E2")}, output)
    assert summary["experiment"].tolist() == ["E1", "E2"]
    assert summary["epochs"].tolist() == [4, 4]
    assert summary["best_valid_nll_epoch"].tolist() == [4, 4]
    assert (output / "training_curve_summary.csv").is_file()
    assert (output / "training_curves.md").is_file()
    markdown = (output / "training_curves.md").read_text(encoding="utf-8")
    assert "Best validation Rank IC by experiment and horizon" in markdown
    for experiment in ("E1", "E2"):
        for horizon_name in HORIZONS:
            assert (
                f"| {experiment} | "
                in markdown
                and f"{float(summary.loc[summary['experiment'].eq(experiment), f'{horizon_name}_best_rank_ic'].iloc[0]):.6f}"
                in markdown
            )
    assert "T+1 open → T+2 close" in markdown
    assert "| 4 |" in markdown
    image = mpimg.imread(output / "training_curves.png")
    assert image.shape[0] > 100 and image.shape[1] > 100


def test_load_history_rejects_incomplete_or_noncontiguous_training(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    history("E1", epochs=3).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 4 epochs"):
        load_history("E1", path, expected_epochs=4)

    broken = history("E1", epochs=4)
    broken.loc[2, "epoch"] = 8
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="contiguous"):
        load_history("E1", path, expected_epochs=4)


def test_training_markdown_marks_non_emitted_horizons(tmp_path: Path) -> None:
    partial = history("E2").drop(
        columns=[f"{horizon_name}_rank_ic" for horizon_name in HORIZONS[1:]]
    )
    output = tmp_path / "partial-report"
    render({"E2": partial}, output)
    markdown = (output / "training_curves.md").read_text(encoding="utf-8")
    assert "| E2 | T+1 open → T+2 close | 0.040000 | 4 |" in markdown
    assert "| E2 | T+1 close → T+2 open | — | — |" in markdown
