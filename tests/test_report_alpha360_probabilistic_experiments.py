from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from script.report_alpha360_probabilistic_experiments import (
    HORIZONS,
    benchmark_equity_curve,
    generate_report,
    observable_prediction_metrics,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_fixture(scale: float) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=4)
    instruments = ["SH600000", "SH600001", "SH600002"]
    actual_by_day = [
        [-0.02, 0.00, 0.03],
        [0.01, -0.02, 0.04],
        [-0.03, 0.02, 0.01],
        [0.04, -0.01, 0.00],
    ]
    signal_by_day = [
        [-0.01, 0.01, 0.03],
        [0.02, -0.01, 0.04],
        [0.01, 0.03, -0.02],
        [0.03, 0.00, 0.01],
    ]
    rows = []
    for date, actuals, signals in zip(dates, actual_by_day, signal_by_day, strict=True):
        for instrument, actual, signal in zip(instruments, actuals, signals, strict=True):
            row = {"datetime": date, "instrument": instrument}
            for horizon_number, horizon in enumerate(HORIZONS):
                expected = scale * signal + 0.0005 * horizon_number
                variance = 0.01 + 0.001 * horizon_number
                row[f"{horizon}_log_mean"] = np.log1p(max(expected, -0.9)) - variance / 2
                row[f"{horizon}_log_variance"] = variance
                row[f"{horizon}_expected_return"] = expected
                row[f"{horizon}_return_std"] = 0.1
                row[f"{horizon}_probability_positive"] = 0.3 + 0.4 * (expected > 0)
                row[f"{horizon}_actual_return"] = actual + 0.0002 * horizon_number
            rows.append(row)
    return pd.DataFrame(rows)


def metric_fixture(predictions: pd.DataFrame, horizon: str, nll: float) -> dict:
    metrics = observable_prediction_metrics(predictions, horizon)
    return {
        "components": 1,
        **metrics,
        "nll": nll,
    }


def index_fixture(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    calendar = pd.bdate_range("2026-01-05", periods=6)
    rows = []
    for benchmark, base, drift in (("CSI1000", 100.0, 0.004), ("CSI300", 200.0, -0.001)):
        previous = base
        for number, date in enumerate(calendar):
            opening = previous * (1 + drift / 3)
            close = opening * (1 + drift + 0.0003 * number)
            rows.append({
                "datetime": date,
                "index": benchmark,
                "open": opening,
                "close": close,
            })
            previous = close
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame, pd.DatetimeIndex(calendar)


def fixed_reference_fixture(path: Path) -> pd.DataFrame:
    rows = []
    for exit_time in ("10:30", "15:00"):
        for fallback in (False, True):
            for slippage in (0.0, 5.0):
                net = (
                    0.15
                    - (0.03 if exit_time == "10:30" else 0.0)
                    - (0.03 if fallback else 0.0)
                    - slippage / 1000
                )
                rows.append({
                    "buy_time": "15:00",
                    "exit_time": exit_time,
                    "fallback": fallback,
                    "slippage_bps_each_side": slippage,
                    "signal_days": 4,
                    "completed_trades": 4 if fallback else 3,
                    "net_cumulative": net,
                    "max_drawdown": -0.20 if fallback else -0.15,
                    "trade_win_rate": 0.60 if fallback else 0.55,
                    "average_net_trade_return": 0.002,
                    "total_fees": 2500.0,
                    "blocked_buy_candidates": 20,
                    "blocked_sell_attempts": 10,
                    "skipped_due_existing_holding": 10,
                    "average_selected_rank": 1.3 if fallback else 1.0,
                    "max_selected_rank": 3 if fallback else 1,
                    "ending_equity": 100_000 * (1 + net),
                    "unclosed_position": False,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def write_strict_directory(
    path: Path,
    variant: str,
    selection_predictions: Path,
    test_predictions: Path,
    test_frame: pd.DataFrame,
    index: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> None:
    path.mkdir()
    capital = 100_000.0
    slippage = 5.0
    chosen_records = []
    chosen_rows = []
    baseline_rows = []
    selected_rows = []
    signal_dates = list(test_frame["datetime"].drop_duplicates().sort_values())
    for number, horizon in enumerate(HORIZONS):
        rule = f"frozen_rule_{number}"
        selection_net = 0.02 + number * 0.01
        selection_win = 0.51 + number * 0.01
        selection_drawdown = -0.08 - number * 0.01
        chosen_records.append({
            "horizon": horizon,
            "rule": rule,
            "selection_net_cumulative": selection_net,
            "selection_active_signal_days": 4,
            "selection_trade_win_rate": selection_win,
            "selection_max_drawdown": selection_drawdown,
        })
        chosen_rows.append(chosen_records[-1])
        baseline_net = 0.01 + number * 0.005 + (0.002 if variant == "all" else 0.0)
        selected_net = 0.03 + number * 0.007 + (0.003 if variant == "all" else 0.0)
        benchmark_values = {
            benchmark: benchmark_equity_curve(
                index, calendar, signal_dates, horizon, benchmark
            ).iloc[-1] - 1.0
            for benchmark in ("CSI300", "CSI1000")
        }
        common = {
            "horizon": horizon,
            "topk": 1,
            "fallback": True,
            "slippage_bps_each_side": slippage,
            "signal_days": 4,
            "active_signal_days": 4,
            "completed_trades": 4,
            "mean_trade_return": 0.01,
            "final_equity": capital,
            "blocked_buy_up_limit": 0,
            "blocked_buy_missing": 0,
            "too_expensive": 0,
            "delayed_exit_trades": 0,
            "skipped_busy_slots": 0,
            "filtered_cash_slots": 0,
            "unresolved_exit": 0,
            "split": "test",
            "CSI1000_gross_cumulative": benchmark_values["CSI1000"],
            "CSI300_gross_cumulative": benchmark_values["CSI300"],
        }
        baseline = {
            **common,
            "rule": "mean_all",
            "net_cumulative": baseline_net,
            "trade_win_rate": 0.50,
            "max_drawdown_marked": -0.10,
            "final_equity": capital * (1 + baseline_net),
            "net_excess_vs_CSI1000": (1 + baseline_net) / (1 + benchmark_values["CSI1000"]) - 1,
            "net_excess_vs_CSI300": (1 + baseline_net) / (1 + benchmark_values["CSI300"]) - 1,
        }
        selected = {
            **common,
            "rule": rule,
            "net_cumulative": selected_net,
            "trade_win_rate": 0.55,
            "max_drawdown_marked": -0.07,
            "final_equity": capital * (1 + selected_net),
            "net_excess_vs_CSI1000": (1 + selected_net) / (1 + benchmark_values["CSI1000"]) - 1,
            "net_excess_vs_CSI300": (1 + selected_net) / (1 + benchmark_values["CSI300"]) - 1,
        }
        baseline_rows.append(baseline)
        selected_rows.append(selected)
        for kind, terminal in (("baseline", baseline_net), ("selected", selected_net)):
            curve = pd.DataFrame({
                "datetime": signal_dates,
                "entry_date": signal_dates,
                "scheduled_exit": signal_dates,
                "equity_mark": capital * (1 + np.linspace(terminal / 4, terminal, 4)),
                "entries": 1,
                "active_positions": 1,
            })
            filename = (
                f"test_{horizon}_top1_5bps_daily.csv"
                if kind == "baseline"
                else f"test_selection_{horizon}_{rule}_5bps_daily.csv"
            )
            curve.to_csv(path / filename, index=False)
    pd.DataFrame(chosen_rows).to_csv(path / "selection_valid_chosen_rules.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(path / "test_baseline_four_horizons.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(path / "test_selected_uncertainty_rules.csv", index=False)
    pre = {
        "selection_predictions": str(selection_predictions.resolve()),
        "selection_predictions_sha256": file_hash(selection_predictions),
        "selection_slippage_bps": slippage,
        "minimum_active_days": 1,
        "chosen": chosen_records,
        "execution_data_inputs": {
            name: {
                "path": str(selection_predictions.resolve()),
                "sha256": file_hash(selection_predictions),
                "size": selection_predictions.stat().st_size,
                "rows": len(test_frame),
                "start": "2026-01-01",
                "end": "2026-12-31",
            }
            for name in ("daily_ohlc", "exact_limits", "index_cache")
        },
        "test_opened": False,
    }
    (path / "chosen_rule_manifest_pre_test.json").write_text(
        json.dumps(pre, indent=2), encoding="utf-8"
    )
    evaluated = {
        **pre,
        "test_opened": True,
        "test_predictions": str(test_predictions.resolve()),
        "test_predictions_sha256": file_hash(test_predictions),
    }
    (path / "evaluated_rule_manifest.json").write_text(
        json.dumps(evaluated, indent=2), encoding="utf-8"
    )
    method = {
        "selection_policy": "selection_valid only",
        "board_variant": variant,
        "capital": capital,
        "commission_rate_each_side": 0.000235,
        "minimum_commission": 5.0,
        "stamp_tax": "omitted per user preference",
        "fallback": True,
        "lot_size": 100,
        "limit_down_exit": "chronological same-phase retry",
        "diagnostic_test_grid_generated": False,
    }
    (path / "method.json").write_text(json.dumps(method, indent=2), encoding="utf-8")


@pytest.fixture
def complete_fixture(tmp_path: Path) -> dict[str, Path]:
    selection_predictions = tmp_path / "selection_valid_ensemble_predictions.csv"
    test_predictions = tmp_path / "test_predictions.csv"
    selection_frame = prediction_fixture(1.0)
    test_frame = prediction_fixture(0.7)
    selection_frame.to_csv(selection_predictions, index=False)
    test_frame.to_csv(test_predictions, index=False)
    selections = {}
    for number, horizon in enumerate(HORIZONS):
        selected_metrics = metric_fixture(selection_frame, horizon, 1.0 + number)
        weak_metrics = {**selected_metrics, "rank_ic": selected_metrics["rank_ic"] - 0.2}
        pair_metrics = {**selected_metrics, "components": 2, "rank_ic": selected_metrics["rank_ic"] - 0.1}
        selections[horizon] = {
            "individual_metrics": {"alpha": selected_metrics, "beta": weak_metrics},
            "alternatives": [
                {"components": ["alpha"], "metrics": selected_metrics},
                {"components": ["alpha", "beta"], "metrics": pair_metrics},
            ],
            "selected_components": ["alpha"],
            "selection_valid_metrics": selected_metrics,
        }
    selection_manifest = tmp_path / "selected_ensemble_manifest.json"
    selection_manifest.write_text(json.dumps({
        "schema_version": 1,
        "selection_split": "selection_valid",
        "test_files_read": False,
        "selection_valid_ensemble_predictions_sha256": file_hash(selection_predictions),
        "selections": selections,
    }, indent=2), encoding="utf-8")
    test_summary = tmp_path / "test_summary.csv"
    test_rows = []
    for number, horizon in enumerate(HORIZONS):
        metrics = metric_fixture(test_frame, horizon, 1.5 + number)
        # Deliberately excellent Test values do not alter the manifest-selected model.
        test_rows.append({"horizon": horizon, **metrics})
    pd.DataFrame(test_rows).to_csv(test_summary, index=False)
    index_cache = tmp_path / "index.csv"
    index, calendar = index_fixture(index_cache)
    fixed_reference = tmp_path / "fixed_reference_summary.csv"
    fixed_reference_fixture(fixed_reference)
    mainboard = tmp_path / "mainboard"
    all_pool = tmp_path / "all"
    write_strict_directory(
        mainboard, "mainboard", selection_predictions, test_predictions,
        test_frame, index, calendar,
    )
    write_strict_directory(
        all_pool, "all", selection_predictions, test_predictions,
        test_frame, index, calendar,
    )
    return {
        "selection_manifest": selection_manifest,
        "selection_predictions": selection_predictions,
        "test_summary": test_summary,
        "test_predictions": test_predictions,
        "mainboard": mainboard,
        "all": all_pool,
        "index": index_cache,
        "fixed_reference": fixed_reference,
        "output": tmp_path / "report",
    }


def run_report(paths: dict[str, Path]) -> Path:
    return generate_report(
        selection_manifest_path=paths["selection_manifest"],
        selection_predictions_path=paths["selection_predictions"],
        test_summary_path=paths["test_summary"],
        test_predictions_path=paths["test_predictions"],
        mainboard_backtest_dir=paths["mainboard"],
        all_backtest_dir=paths["all"],
        index_cache_path=paths["index"],
        fixed_reference_summary_path=paths["fixed_reference"],
        execution_policy="fallback",
        output_directory=paths["output"],
    )


def test_generates_report_with_frozen_mappings_and_pngs(complete_fixture: dict[str, Path]) -> None:
    input_paths = [
        complete_fixture[key]
        for key in (
            "selection_manifest", "selection_predictions", "test_summary",
            "test_predictions", "index", "fixed_reference",
        )
    ]
    before = {path: file_hash(path) for path in input_paths}
    output = run_report(complete_fixture)
    assert {path.name for path in output.iterdir()} == {
        "concise_summary.csv",
        "model_selection.csv",
        "fixed_reference_comparison.csv",
        "prediction_metrics_comparison.png",
        "strategy_equity_curves.png",
        "method_and_findings.md",
    }
    summary = pd.read_csv(output / "concise_summary.csv")
    assert summary["horizon"].tolist() == list(HORIZONS)
    assert summary["selected_components"].eq("alpha").all()
    assert summary.loc[0, "mainboard_selection_rule"] == "frozen_rule_0"
    assert summary.loc[0, "mainboard_selection_rule_test_net_cumulative"] == pytest.approx(0.03)
    assert summary.loc[0, "all_selection_rule_test_net_cumulative"] == pytest.approx(0.033)
    comparison = pd.read_csv(output / "fixed_reference_comparison.csv")
    assert len(comparison) == 1
    assert comparison.loc[0, "horizon"] == "close1_close2"
    assert comparison.loc[0, "board_variant"] == "mainboard"
    assert comparison.loc[0, "topk"] == 1
    assert comparison.loc[0, "execution_policy"] == "fallback"
    assert comparison.loc[0, "comparability"] == (
        "approximate_old_fallback_rank_cap_20_vs_new_full_ranking"
    )
    assert comparison.loc[0, "alpha360_net_cumulative"] == pytest.approx(0.025)
    assert comparison.loc[0, "fixed_reference_net_cumulative"] == pytest.approx(0.115)
    assert comparison.loc[
        0, "delta_alpha360_minus_fixed_net_cumulative"
    ] == pytest.approx(-0.09)
    assert comparison.loc[0, "alpha360_completed_trades"] == 4
    assert comparison.loc[0, "fixed_reference_completed_trades"] == 4
    for column in (
        "test_coverage_50", "test_coverage_80", "test_coverage_95",
        "test_top1_win_rate", "test_top3_mean_return", "test_top5_cumulative",
        "test_top10_stock_win_rate",
    ):
        assert column in summary
    model_selection = pd.read_csv(output / "model_selection.csv")
    selected = model_selection.loc[
        model_selection["record_type"].eq("ensemble_alternative")
        & model_selection["selected"]
    ]
    assert len(selected) == 4
    assert selected["components"].eq("alpha").all()
    for image_name in ("prediction_metrics_comparison.png", "strategy_equity_curves.png"):
        image = mpimg.imread(output / image_name)
        assert image.shape[0] > 500
        assert image.shape[1] > 500
    method = (output / "method_and_findings.md").read_text(encoding="utf-8")
    assert "selection_valid" in method
    assert "Test is opened only after model components and trading rules are frozen" in method
    assert "100" in method
    assert "STAR and ChiNext excluded" in method
    assert "Calibration and frictionless ranking diagnostics" in method
    assert "Top1 win/mean/cum" in method
    assert "does not participate in model Selection" in method
    assert "old Fixed implementation searched at most through rank 20" in method
    assert "close1_close2" in method
    assert {path: file_hash(path) for path in input_paths} == before


def test_missing_prediction_column_fails_atomically(complete_fixture: dict[str, Path]) -> None:
    predictions = pd.read_csv(complete_fixture["test_predictions"])
    predictions.drop(columns=["close1_close2_actual_return"]).to_csv(
        complete_fixture["test_predictions"], index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_rejects_rule_changed_after_test(complete_fixture: dict[str, Path]) -> None:
    path = complete_fixture["all"] / "evaluated_rule_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["chosen"][0]["rule"] = "test_selected_rule"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after opening Test"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_rejects_test_diagnostic_grid(complete_fixture: dict[str, Path]) -> None:
    method_path = complete_fixture["mainboard"] / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["diagnostic_test_grid_generated"] = True
    method_path.write_text(json.dumps(method), encoding="utf-8")
    with pytest.raises(ValueError, match="must not generate a Test diagnostic"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_duplicate_fixed_reference_row_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    path = complete_fixture["fixed_reference"]
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate key rows"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_unresolved_test_position_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    path = complete_fixture["mainboard"] / "test_selected_uncertainty_rules.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "unresolved_exit"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unresolved positions"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_missing_fixed_reference_row_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    path = complete_fixture["fixed_reference"]
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="grid mismatch"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_wrong_fixed_reference_row_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    path = complete_fixture["fixed_reference"]
    frame = pd.read_csv(path)
    frame.loc[0, "buy_time"] = "14:59"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="grid mismatch"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_fixed_reference_signal_day_mismatch_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    path = complete_fixture["fixed_reference"]
    frame = pd.read_csv(path)
    frame["signal_days"] = frame["signal_days"] + 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="signal-day counts differ"):
        run_report(complete_fixture)
    assert not complete_fixture["output"].exists()


def test_leave_cash_uses_exact_matching_reference_and_strongest_label(
    complete_fixture: dict[str, Path],
) -> None:
    for key in ("mainboard", "all"):
        method_path = complete_fixture[key] / "method.json"
        method = json.loads(method_path.read_text(encoding="utf-8"))
        method["fallback"] = False
        method_path.write_text(json.dumps(method), encoding="utf-8")
    output = generate_report(
        selection_manifest_path=complete_fixture["selection_manifest"],
        selection_predictions_path=complete_fixture["selection_predictions"],
        test_summary_path=complete_fixture["test_summary"],
        test_predictions_path=complete_fixture["test_predictions"],
        mainboard_backtest_dir=complete_fixture["mainboard"],
        all_backtest_dir=complete_fixture["all"],
        index_cache_path=complete_fixture["index"],
        fixed_reference_summary_path=complete_fixture["fixed_reference"],
        execution_policy="leave_cash",
        output_directory=complete_fixture["output"],
    )
    comparison = pd.read_csv(output / "fixed_reference_comparison.csv")
    assert comparison.loc[0, "fixed_reference_net_cumulative"] == pytest.approx(0.145)
    assert comparison.loc[0, "fixed_reference_completed_trades"] == 3
    assert comparison.loc[0, "comparability"] == "strongest_same_definition"
    method = (output / "method_and_findings.md").read_text(encoding="utf-8")
    assert "`leave_cash` is the strongest same-definition comparison" in method


def test_execution_policy_mismatch_fails_atomically(
    complete_fixture: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="does not match explicit execution_policy"):
        generate_report(
            selection_manifest_path=complete_fixture["selection_manifest"],
            selection_predictions_path=complete_fixture["selection_predictions"],
            test_summary_path=complete_fixture["test_summary"],
            test_predictions_path=complete_fixture["test_predictions"],
            mainboard_backtest_dir=complete_fixture["mainboard"],
            all_backtest_dir=complete_fixture["all"],
            index_cache_path=complete_fixture["index"],
            fixed_reference_summary_path=complete_fixture["fixed_reference"],
            execution_policy="leave_cash",
            output_directory=complete_fixture["output"],
        )
    assert not complete_fixture["output"].exists()
