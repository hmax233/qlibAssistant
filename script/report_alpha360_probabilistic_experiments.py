#!/usr/bin/env python3
"""Build a read-only report for the four-horizon Alpha360 experiment matrix.

The ensemble components and trading rules are read from immutable, pre-Test
selection artifacts.  Test metrics are descriptive only: this module contains
no ranking or selection operation over Test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_CACHE = ROOT / ".qlibAssistant/cache/tushare_index_daily.csv"
HORIZONS = (
    "open1_close2",
    "close1_open2",
    "open1_open2",
    "close1_close2",
)
HORIZON_LABELS = {
    "open1_close2": "T+1 open → T+2 close",
    "close1_open2": "T+1 close → T+2 open",
    "open1_open2": "T+1 open → T+2 open",
    "close1_close2": "T+1 close → T+2 close",
}
HORIZON_EXECUTION = {
    "open1_close2": ("open", "close", 2),
    "close1_open2": ("close", "open", 1),
    "open1_open2": ("open", "open", 1),
    "close1_close2": ("close", "close", 1),
}
CORE_PREDICTION_METRICS = ("rank_ic", "rank_icir", "nll", "mae", "brier")
CALIBRATION_METRICS = ("direction_accuracy", "coverage_50", "coverage_80", "coverage_95")
TOPK_METRICS = tuple(
    f"top{topk}_{metric}"
    for topk in (1, 3, 5, 10)
    for metric in ("mean_return", "cumulative", "win_rate", "stock_win_rate")
)
PREDICTION_METRICS = (*CORE_PREDICTION_METRICS, *CALIBRATION_METRICS, *TOPK_METRICS)
SUMMARY_METRICS = ("days", "rows", "components", *PREDICTION_METRICS)
SENSITIVITY_TOPKS = (1, 3, 5, 10)
SENSITIVITY_SLIPPAGE_BPS = (0.0, 5.0)
STRICT_EXECUTION_METRICS = (
    "signal_days",
    "active_signal_days",
    "completed_trades",
    "net_cumulative",
    "trade_win_rate",
    "mean_trade_return",
    "max_drawdown_marked",
    "net_sharpe_rf0",
    "total_commission",
    "average_daily_turnover",
    "annualized_turnover",
    "average_gross_exposure",
    "max_gross_exposure",
    "average_max_name_concentration",
    "max_name_concentration",
    "blocked_buy_up_limit",
    "blocked_buy_missing",
    "blocked_buy_suspended",
    "too_expensive",
    "fallback_replacements",
    "filtered_cash_slots",
    "delayed_exit_trades",
    "blocked_sell_down_limit_attempts",
    "blocked_sell_missing_attempts",
    "blocked_sell_suspended_attempts",
    "unresolved_exit",
    "CSI300_gross_cumulative",
    "net_excess_vs_CSI300",
    "CSI1000_gross_cumulative",
    "net_excess_vs_CSI1000",
)
FIXED_REFERENCE_REQUIRED_COLUMNS = {
    "buy_time",
    "exit_time",
    "fallback",
    "slippage_bps_each_side",
    "signal_days",
    "completed_trades",
    "net_cumulative",
    "max_drawdown",
    "trade_win_rate",
    "average_net_trade_return",
    "total_fees",
    "blocked_buy_candidates",
    "blocked_sell_attempts",
    "skipped_due_existing_holding",
    "average_selected_rank",
    "max_selected_rank",
    "ending_equity",
    "unclosed_position",
}
FIXED_REFERENCE_KEY_COLUMNS = (
    "buy_time", "exit_time", "fallback", "slippage_bps_each_side"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON in {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _read_csv(path: Path, label: str, required: set[str]) -> pd.DataFrame:
    path = _require_file(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise ValueError(f"Cannot read {label}: {path}: {error}") from error
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns {sorted(missing)}: {path}")
    return frame


def _exact_horizons(values: pd.Series | list[str], label: str) -> None:
    items = list(values)
    if len(items) != len(set(items)):
        raise ValueError(f"{label} contains duplicate horizons: {items}")
    if set(items) != set(HORIZONS):
        raise ValueError(
            f"{label} horizons must be exactly {list(HORIZONS)}, got {items}"
        )


def _require_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _validate_execution_data_inputs(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"daily_ohlc", "exact_limits", "index_cache"}:
        raise ValueError(f"{label} must freeze daily_ohlc, exact_limits, and index_cache")
    normalized = {}
    for name, reference in value.items():
        if not isinstance(reference, dict):
            raise ValueError(f"{label}/{name} must be an object")
        missing = {"path", "sha256", "size", "rows", "start", "end"} - set(reference)
        if missing:
            raise ValueError(f"{label}/{name} missing {sorted(missing)}")
        path = _require_file(Path(reference["path"]), f"{label}/{name}")
        if sha256(path) != reference["sha256"] or path.stat().st_size != int(reference["size"]):
            raise RuntimeError(f"{label}/{name} file identity changed after rule freeze")
        if int(reference["rows"]) <= 0:
            raise ValueError(f"{label}/{name} rows must be positive")
        if pd.Timestamp(reference["start"]) > pd.Timestamp(reference["end"]):
            raise ValueError(f"{label}/{name} date range is reversed")
        normalized[name] = dict(reference)
    return normalized


def _strict_boolean_series(values: pd.Series, label: str) -> pd.Series:
    """Parse booleans without allowing truthy strings or missing values."""

    parsed: list[bool] = []
    for row_number, value in enumerate(values, start=2):
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            parsed.append(value.strip().casefold() == "true")
            continue
        raise ValueError(f"{label} row {row_number} must be exactly true or false")
    return pd.Series(parsed, index=values.index, dtype=bool)


def load_fixed_reference(path: Path, execution_policy: str) -> dict[str, Any]:
    """Load the immutable external Fixed Ensemble history, failing closed."""

    if execution_policy not in {"fallback", "leave_cash"}:
        raise ValueError(
            "execution_policy must be exactly 'fallback' or 'leave_cash'"
        )
    frame = _read_csv(
        path, "Fixed Ensemble external reference", FIXED_REFERENCE_REQUIRED_COLUMNS
    )
    if frame.empty:
        raise ValueError("Fixed Ensemble external reference is empty")
    frame = frame.copy()
    for column in ("buy_time", "exit_time"):
        if frame[column].isna().any():
            raise ValueError(f"Fixed Ensemble external reference/{column} is missing")
        frame[column] = frame[column].astype(str)
    frame["fallback"] = _strict_boolean_series(
        frame["fallback"], "Fixed Ensemble external reference/fallback"
    )
    frame["unclosed_position"] = _strict_boolean_series(
        frame["unclosed_position"],
        "Fixed Ensemble external reference/unclosed_position",
    )
    numeric_columns = sorted(
        FIXED_REFERENCE_REQUIRED_COLUMNS
        - {"buy_time", "exit_time", "fallback", "unclosed_position"}
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[numeric_columns].to_numpy(float)).all():
        raise ValueError("Fixed Ensemble external reference contains non-finite values")
    integer_columns = (
        "signal_days",
        "completed_trades",
        "blocked_buy_candidates",
        "blocked_sell_attempts",
        "skipped_due_existing_holding",
        "max_selected_rank",
    )
    for column in integer_columns:
        values = frame[column].to_numpy(float)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"Fixed Ensemble external reference/{column} must be non-negative integers"
            )
    if (frame["completed_trades"] > frame["signal_days"]).any():
        raise ValueError(
            "Fixed Ensemble external reference completed_trades exceeds signal_days"
        )
    if (frame["net_cumulative"] <= -1).any():
        raise ValueError("Fixed Ensemble external reference net_cumulative must exceed -1")
    if ((frame["max_drawdown"] < -1) | (frame["max_drawdown"] > 0)).any():
        raise ValueError("Fixed Ensemble external reference max_drawdown must be in [-1, 0]")
    if ((frame["trade_win_rate"] < 0) | (frame["trade_win_rate"] > 1)).any():
        raise ValueError("Fixed Ensemble external reference trade_win_rate must be in [0, 1]")
    if (frame["ending_equity"] <= 0).any() or (frame["total_fees"] < 0).any():
        raise ValueError(
            "Fixed Ensemble external reference equity must be positive and fees non-negative"
        )
    if frame["unclosed_position"].any():
        raise ValueError("Fixed Ensemble external reference contains an unclosed position")
    if frame.duplicated(list(FIXED_REFERENCE_KEY_COLUMNS)).any():
        raise ValueError("Fixed Ensemble external reference contains duplicate key rows")

    expected_keys = {
        ("15:00", exit_time, fallback, slippage)
        for exit_time in ("10:30", "15:00")
        for fallback in (False, True)
        for slippage in (0.0, 5.0)
    }
    observed_keys = {
        (row.buy_time, row.exit_time, bool(row.fallback), float(row.slippage_bps_each_side))
        for row in frame.itertuples(index=False)
    }
    if observed_keys != expected_keys or len(frame) != len(expected_keys):
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            "Fixed Ensemble external reference grid mismatch: "
            f"missing={missing}, unexpected={unexpected}, rows={len(frame)}"
        )

    expected_fallback = execution_policy == "fallback"
    selected = frame.loc[
        frame["buy_time"].eq("15:00")
        & frame["exit_time"].eq("15:00")
        & frame["fallback"].eq(expected_fallback)
        & np.isclose(frame["slippage_bps_each_side"], 5.0)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Fixed Ensemble external reference must contain exactly one matching "
            "buy_time=15:00/exit_time=15:00/slippage=5 row"
        )
    resolved = path.expanduser().resolve()
    return {
        "path": resolved,
        "sha256": sha256(resolved),
        "execution_policy": execution_policy,
        "comparability": (
            "strongest_same_definition"
            if execution_policy == "leave_cash"
            else "approximate_old_fallback_rank_cap_20_vs_new_full_ranking"
        ),
        "row": selected.iloc[0],
    }


def prediction_columns(horizon: str) -> set[str]:
    return {
        "datetime",
        "instrument",
        f"{horizon}_log_mean",
        f"{horizon}_log_variance",
        f"{horizon}_expected_return",
        f"{horizon}_probability_positive",
        f"{horizon}_actual_return",
    }


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    required = set().union(*(prediction_columns(horizon) for horizon in HORIZONS))
    frame = _read_csv(path, label, required)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError(f"{label} contains duplicate datetime/instrument rows: {path}")
    for horizon in HORIZONS:
        actual_column = f"{horizon}_actual_return"
        prediction_fields = prediction_columns(horizon) - {
            "datetime", "instrument", actual_column,
        }
        predictions = frame[list(prediction_fields)].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(predictions.to_numpy(float)).all():
            raise ValueError(f"{label}/{horizon} contains non-finite prediction values")
        if (predictions[f"{horizon}_log_variance"] <= 0).any():
            raise ValueError(f"{label}/{horizon} contains non-positive variance")
        probabilities = predictions[f"{horizon}_probability_positive"]
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{label}/{horizon} probability must be in [0, 1]")
        # A feature row can legitimately lack a realized label when its future
        # open/close is unavailable.  Preserve that row and its prediction for
        # ranking/execution, but fail closed on malformed text or +/- infinity.
        raw_actual = frame[actual_column]
        actual = pd.to_numeric(raw_actual, errors="coerce")
        malformed = raw_actual.notna() & actual.isna()
        if malformed.any() or np.isinf(actual.to_numpy(float)).any():
            raise ValueError(f"{label}/{horizon} contains invalid realized labels")
        finite_actual = actual[np.isfinite(actual)]
        if (finite_actual <= -1.0).any():
            raise ValueError(f"{label}/{horizon} realized return must exceed -1")
        frame[list(prediction_fields)] = predictions
        frame[actual_column] = actual
    return frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def observable_prediction_metrics(frame: pd.DataFrame, horizon: str) -> dict[str, float | int]:
    expected = frame[f"{horizon}_expected_return"].to_numpy(float)
    actual = frame[f"{horizon}_actual_return"].to_numpy(float)
    probability = frame[f"{horizon}_probability_positive"].to_numpy(float)
    daily_ic: list[float] = []
    for _, day in frame.groupby("datetime", sort=True):
        if len(day) < 2:
            continue
        value = day[f"{horizon}_expected_return"].corr(
            day[f"{horizon}_actual_return"], method="spearman"
        )
        if pd.notna(value):
            daily_ic.append(float(value))
    if not daily_ic:
        raise ValueError(f"No finite daily RankIC values for {horizon}")
    daily = np.asarray(daily_ic, dtype=float)
    deviation = float(daily.std(ddof=0))
    result: dict[str, float | int] = {
        "days": int(frame["datetime"].nunique()),
        "rows": int(len(frame)),
        "rank_ic": float(daily.mean()),
        "rank_icir": float(daily.mean() / deviation) if deviation > 0 else math.nan,
        # These definitions intentionally reproduce the already-frozen
        # Selection/Test scorer.  Missing labels are ignored by MAE and daily
        # portfolio means; the frozen scorer's boolean diagnostics map a
        # missing realized value to ``False``.  Strict account backtests use
        # audited OHLC execution data and do not depend on these labels.
        "mae": float(np.nanmean(np.abs(expected - actual))),
        "brier": float(np.nanmean((probability - (actual > 0)) ** 2)),
        "direction_accuracy": float(np.nanmean(
            (probability >= 0.5) == (actual > 0)
        )),
    }
    actual_log = np.log1p(actual)
    log_mean = frame[f"{horizon}_log_mean"].to_numpy(float)
    log_std = np.sqrt(frame[f"{horizon}_log_variance"].to_numpy(float))
    for level, z_value in ((50, 0.6744897501960817), (80, 1.2815515655446004),
                           (95, 1.959963984540054)):
        valid_log = np.isfinite(actual_log)
        covered = valid_log & (
            actual_log >= log_mean - z_value * log_std
        ) & (
            actual_log <= log_mean + z_value * log_std
        )
        result[f"coverage_{level}"] = float(covered[valid_log].mean())
    ordered = frame.sort_values(
        ["datetime", f"{horizon}_expected_return"], ascending=[True, False]
    )
    for topk in (1, 3, 5, 10):
        selected = ordered.groupby("datetime", sort=True).head(topk)
        daily_portfolio = selected.groupby("datetime", sort=True)[
            f"{horizon}_actual_return"
        ].mean()
        result[f"top{topk}_mean_return"] = float(daily_portfolio.mean())
        result[f"top{topk}_cumulative"] = float(np.prod(1.0 + daily_portfolio) - 1.0)
        result[f"top{topk}_win_rate"] = float((daily_portfolio > 0).mean())
        result[f"top{topk}_stock_win_rate"] = float(
            (selected[f"{horizon}_actual_return"] > 0).mean()
        )
    return result


def _assert_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-8) -> None:
    left = _require_finite(actual, f"{label} (reported)")
    right = _require_finite(expected, f"{label} (recomputed)")
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: reported={left}, recomputed={right}")


def validate_metric_mapping(
    frame: pd.DataFrame,
    reported: dict[str, Any] | pd.Series,
    horizon: str,
    label: str,
) -> None:
    observed = observable_prediction_metrics(frame, horizon)
    for metric in observed:
        if metric == "rank_icir":
            continue
        _assert_close(reported[metric], observed[metric], f"{label}/{horizon}/{metric}")
    observed_icir = observed["rank_icir"]
    reported_icir = reported["rank_icir"]
    if math.isfinite(float(observed_icir)):
        _assert_close(reported_icir, observed_icir, f"{label}/{horizon}/rank_icir")
    elif reported_icir is not None and not pd.isna(reported_icir):
        raise ValueError(f"{label}/{horizon}/rank_icir should be undefined")


def load_selection_manifest(path: Path, selection_predictions: Path) -> dict[str, Any]:
    manifest = _read_json(path, "selection manifest")
    required = {
        "selection_split",
        "test_files_read",
        "selection_valid_ensemble_predictions_sha256",
        "selections",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"selection manifest missing keys: {sorted(missing)}")
    if manifest["selection_split"] != "selection_valid":
        raise ValueError("selection manifest must use selection_valid")
    if manifest["test_files_read"] is not False:
        raise ValueError("model selection manifest must have test_files_read=false")
    if manifest["selection_valid_ensemble_predictions_sha256"] != sha256(selection_predictions):
        raise ValueError("selection prediction hash does not match selection manifest")
    selections = manifest["selections"]
    if not isinstance(selections, dict):
        raise ValueError("selection manifest selections must be an object")
    _exact_horizons(list(selections), "selection manifest")
    for horizon in HORIZONS:
        item = selections[horizon]
        required_item = {
            "individual_metrics",
            "alternatives",
            "selected_components",
            "selection_valid_metrics",
        }
        missing_item = required_item - set(item)
        if missing_item:
            raise ValueError(
                f"selection manifest/{horizon} missing keys: {sorted(missing_item)}"
            )
        selected = item["selected_components"]
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ValueError(f"selection manifest/{horizon} has invalid selected_components")
        metrics = item["selection_valid_metrics"]
        missing_metrics = set(SUMMARY_METRICS) - set(metrics)
        if missing_metrics:
            raise ValueError(
                f"selection manifest/{horizon} metrics missing {sorted(missing_metrics)}"
            )
        if int(metrics["components"]) != len(selected):
            raise ValueError(f"selection manifest/{horizon} component count mismatch")
        selected_matches = [
            alternative
            for alternative in item["alternatives"]
            if alternative.get("components") == selected
        ]
        if len(selected_matches) != 1:
            raise ValueError(
                f"selection manifest/{horizon} selected alternative is not unique"
            )
    return manifest


def load_test_summary(path: Path) -> pd.DataFrame:
    required = {"horizon", *SUMMARY_METRICS}
    frame = _read_csv(path, "Test summary", required)
    _exact_horizons(frame["horizon"], "Test summary")
    return frame.set_index("horizon").loc[list(HORIZONS)].reset_index()


def build_model_selection(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        selection = manifest["selections"][horizon]
        selected = selection["selected_components"]
        alternatives = selection["alternatives"]
        # Ranking is Selection-valid-only and is included solely for auditability.
        ordered = sorted(
            alternatives,
            key=lambda item: (
                _require_finite(item["metrics"]["rank_ic"], "selection rank_ic"),
                -math.inf
                if item["metrics"]["rank_icir"] is None
                else _require_finite(item["metrics"]["rank_icir"], "selection rank_icir"),
                -_require_finite(item["metrics"]["nll"], "selection nll"),
                -int(item["metrics"]["components"]),
            ),
            reverse=True,
        )
        for rank, alternative in enumerate(ordered, start=1):
            metrics = alternative["metrics"]
            components = alternative["components"]
            rows.append({
                "horizon": horizon,
                "record_type": "ensemble_alternative",
                "selection_rank": rank,
                "selected": components == selected,
                "component_count": len(components),
                "components": "+".join(components),
                **{metric: metrics[metric] for metric in PREDICTION_METRICS},
                "days": metrics["days"],
                "rows": metrics["rows"],
                "selection_basis": "selection_valid_only",
            })
        for name, metrics in sorted(selection["individual_metrics"].items()):
            missing = set(SUMMARY_METRICS) - set(metrics)
            if missing:
                raise ValueError(
                    f"selection manifest/{horizon}/individual/{name} missing {sorted(missing)}"
                )
            rows.append({
                "horizon": horizon,
                "record_type": "individual_candidate",
                "selection_rank": pd.NA,
                "selected": selected == [name],
                "component_count": 1,
                "components": name,
                **{metric: metrics[metric] for metric in PREDICTION_METRICS},
                "days": metrics["days"],
                "rows": metrics["rows"],
                "selection_basis": "selection_valid_only",
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("selection manifest contains no model-selection records")
    selected_count = result.loc[
        result["record_type"].eq("ensemble_alternative") & result["selected"]
    ].groupby("horizon").size()
    if not (selected_count.reindex(HORIZONS, fill_value=0) == 1).all():
        raise ValueError("each horizon must have exactly one selected ensemble alternative")
    return result


def _canonical_chosen(records: Any, label: str) -> dict[str, str]:
    if not isinstance(records, list):
        raise ValueError(f"{label} chosen must be a list")
    frame = pd.DataFrame(records)
    missing = {"horizon", "rule"} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} chosen missing columns {sorted(missing)}")
    _exact_horizons(frame["horizon"], f"{label} chosen")
    if frame["rule"].isna().any() or frame["rule"].duplicated().all():
        raise ValueError(f"{label} contains invalid rule values")
    return dict(zip(frame["horizon"], frame["rule"], strict=True))


def _format_number(value: float) -> str:
    return f"{value:g}"


def load_strict_backtest(
    directory: Path,
    variant: str,
    selection_predictions: Path,
    test_predictions: Path,
) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {variant} strict backtest directory: {directory}")
    method = _read_json(directory / "method.json", f"{variant} method")
    if method.get("board_variant") != variant:
        raise ValueError(
            f"{variant} method board_variant={method.get('board_variant')!r}, expected {variant!r}"
        )
    if method.get("diagnostic_test_grid_generated") is not False:
        raise ValueError(f"{variant} must not generate a Test diagnostic rule grid")
    if (directory / "test_uncertainty_grid_diagnostic_only.csv").exists():
        raise ValueError(f"{variant} directory contains a forbidden Test rule grid")
    pre = _read_json(
        directory / "chosen_rule_manifest_pre_test.json", f"{variant} pre-Test rule manifest"
    )
    evaluated = _read_json(
        directory / "evaluated_rule_manifest.json", f"{variant} evaluated rule manifest"
    )
    if pre.get("test_opened") is not False:
        raise ValueError(f"{variant} pre-Test manifest must have test_opened=false")
    if evaluated.get("test_opened") is not True:
        raise ValueError(f"{variant} evaluated manifest must have test_opened=true")
    if pre.get("selection_predictions_sha256") != sha256(selection_predictions):
        raise ValueError(f"{variant} selection prediction hash mismatch")
    if evaluated.get("selection_predictions_sha256") != sha256(selection_predictions):
        raise ValueError(f"{variant} evaluated selection prediction hash mismatch")
    if evaluated.get("test_predictions_sha256") != sha256(test_predictions):
        raise ValueError(f"{variant} Test prediction hash mismatch")
    execution_inputs = _validate_execution_data_inputs(
        pre.get("execution_data_inputs"), f"{variant} pre-Test execution inputs"
    )
    if evaluated.get("execution_data_inputs") != pre.get("execution_data_inputs"):
        raise RuntimeError(f"{variant} execution-data freeze changed after Test was opened")
    pre_chosen = _canonical_chosen(pre.get("chosen"), f"{variant} pre-Test manifest")
    evaluated_chosen = _canonical_chosen(
        evaluated.get("chosen"), f"{variant} evaluated manifest"
    )
    if pre_chosen != evaluated_chosen:
        raise ValueError(f"{variant} rule choices changed after opening Test")
    slippage = _require_finite(pre.get("selection_slippage_bps"), f"{variant} slippage")
    chosen = _read_csv(
        directory / "selection_valid_chosen_rules.csv",
        f"{variant} frozen rule table",
        {
            "horizon",
            "rule",
            "selection_net_cumulative",
            "selection_active_signal_days",
            "selection_trade_win_rate",
            "selection_max_drawdown",
        },
    )
    _exact_horizons(chosen["horizon"], f"{variant} frozen rule table")
    chosen_rules = dict(zip(chosen["horizon"], chosen["rule"], strict=True))
    if chosen_rules != pre_chosen:
        raise ValueError(f"{variant} frozen rule table disagrees with pre-Test manifest")
    common_summary_columns = {
        "horizon",
        "rule",
        "topk",
        "slippage_bps_each_side",
        *STRICT_EXECUTION_METRICS,
    }
    baseline_matrix = _read_csv(
        directory / "test_baseline_four_horizons.csv",
        f"{variant} Test baseline",
        common_summary_columns,
    )
    selected_matrix = _read_csv(
        directory / "test_selected_uncertainty_rules.csv",
        f"{variant} Test selection-rule",
        common_summary_columns,
    )
    if (pd.to_numeric(baseline_matrix["unresolved_exit"], errors="coerce") != 0).any():
        raise ValueError(f"{variant} Test baseline contains unresolved positions")
    if (pd.to_numeric(selected_matrix["unresolved_exit"], errors="coerce") != 0).any():
        raise ValueError(f"{variant} Test selection-rule contains unresolved positions")
    sensitivity = baseline_matrix.loc[
        baseline_matrix["topk"].isin(SENSITIVITY_TOPKS)
        & baseline_matrix["slippage_bps_each_side"].isin(SENSITIVITY_SLIPPAGE_BPS)
    ].copy()
    sensitivity_keys = sensitivity[["horizon", "topk", "slippage_bps_each_side"]].copy()
    if sensitivity_keys.duplicated().any():
        raise ValueError(f"{variant} Test baseline sensitivity grid contains duplicate rows")
    expected_sensitivity = {
        (horizon, topk, slippage_bps)
        for horizon in HORIZONS
        for topk in SENSITIVITY_TOPKS
        for slippage_bps in SENSITIVITY_SLIPPAGE_BPS
    }
    actual_sensitivity = {
        (str(row.horizon), int(row.topk), float(row.slippage_bps_each_side))
        for row in sensitivity_keys.itertuples(index=False)
    }
    if actual_sensitivity != expected_sensitivity:
        missing = sorted(expected_sensitivity - actual_sensitivity)
        extra = sorted(actual_sensitivity - expected_sensitivity)
        raise ValueError(
            f"{variant} Test baseline sensitivity grid mismatch: "
            f"missing={missing}, extra={extra}"
        )
    baseline = baseline_matrix.loc[
        baseline_matrix["topk"].eq(1)
        & np.isclose(baseline_matrix["slippage_bps_each_side"], slippage)
    ].copy()
    selected = selected_matrix.loc[
        selected_matrix["topk"].eq(1)
        & np.isclose(selected_matrix["slippage_bps_each_side"], slippage)
    ].copy()
    _exact_horizons(baseline["horizon"], f"{variant} filtered Test Top1 baseline")
    _exact_horizons(selected["horizon"], f"{variant} filtered Test selection-rule")
    selected_rules = dict(zip(selected["horizon"], selected["rule"], strict=True))
    if selected_rules != pre_chosen:
        raise ValueError(f"{variant} Test rows do not use the frozen selection rules")
    baseline = baseline.set_index("horizon").loc[list(HORIZONS)]
    selected = selected.set_index("horizon").loc[list(HORIZONS)]
    chosen = chosen.set_index("horizon").loc[list(HORIZONS)]
    capital = _require_finite(method.get("capital"), f"{variant} capital")
    if capital <= 0:
        raise ValueError(f"{variant} capital must be positive")
    daily: dict[str, dict[str, pd.DataFrame]] = {}
    suffix = _format_number(slippage)
    for horizon in HORIZONS:
        rule = pre_chosen[horizon]
        paths = {
            "baseline": directory / f"test_{horizon}_top1_{suffix}bps_daily.csv",
            "selected": directory
            / f"test_selection_{horizon}_{rule}_{suffix}bps_daily.csv",
        }
        daily[horizon] = {}
        for key, path in paths.items():
            frame = _read_csv(
                path,
                f"{variant} {horizon} {key} daily equity",
                {"datetime", "equity_mark"},
            )
            if frame.empty:
                raise ValueError(f"{variant} {horizon} {key} daily equity is empty")
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
            frame["equity_mark"] = pd.to_numeric(frame["equity_mark"], errors="coerce")
            if not np.isfinite(frame["equity_mark"].to_numpy(float)).all():
                raise ValueError(f"{variant} {horizon} {key} equity has non-finite values")
            if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
                raise ValueError(f"{variant} {horizon} {key} dates must be unique and sorted")
            summary = baseline.loc[horizon] if key == "baseline" else selected.loc[horizon]
            _assert_close(
                frame["equity_mark"].iloc[-1] / capital - 1.0,
                summary["net_cumulative"],
                f"{variant}/{horizon}/{key}/terminal equity",
                tolerance=1e-6,
            )
            daily[horizon][key] = frame
    return {
        "directory": directory,
        "method": method,
        "pre_manifest": pre,
        "rules": pre_chosen,
        "slippage": slippage,
        "capital": capital,
        "execution_data_inputs": execution_inputs,
        "chosen": chosen,
        "baseline": baseline,
        "selected": selected,
        "baseline_matrix": baseline_matrix,
        "selected_matrix": selected_matrix,
        "sensitivity": sensitivity,
        "daily": daily,
    }


def build_fixed_reference_comparison(
    fixed_reference: dict[str, Any],
    strict_mainboard: dict[str, Any],
) -> pd.DataFrame:
    """Compare only executable mainboard Top1 close1→close2 results."""

    current = strict_mainboard["baseline"].loc["close1_close2"]
    reference = fixed_reference["row"]
    current_signal_days = int(_require_finite(current["signal_days"], "current/signal_days"))
    reference_signal_days = int(
        _require_finite(reference["signal_days"], "fixed/signal_days")
    )
    same_signal_days = current_signal_days == reference_signal_days
    metrics = {
        "net_cumulative": ("net_cumulative", "net_cumulative"),
        "max_drawdown": ("max_drawdown_marked", "max_drawdown"),
        "trade_win_rate": ("trade_win_rate", "trade_win_rate"),
        "completed_trades": ("completed_trades", "completed_trades"),
    }
    row: dict[str, Any] = {
        "horizon": "close1_close2",
        "board_variant": "mainboard",
        "topk": 1,
        "buy_time": "15:00",
        "exit_time": "15:00",
        "slippage_bps_each_side": 5.0,
        "execution_policy": fixed_reference["execution_policy"],
        "comparability": (
            fixed_reference["comparability"]
            if same_signal_days
            else fixed_reference["comparability"] + "_different_signal_days"
        ),
        "selection_role": "external_reference_only_not_used_for_selection",
        "fixed_reference_sha256": fixed_reference["sha256"],
        "signal_days": current_signal_days,
        "alpha360_signal_days": current_signal_days,
        "fixed_reference_signal_days": reference_signal_days,
        "same_signal_days": same_signal_days,
    }
    for metric, (current_column, reference_column) in metrics.items():
        current_value = _require_finite(current[current_column], f"current/{metric}")
        reference_value = _require_finite(reference[reference_column], f"fixed/{metric}")
        row[f"alpha360_{metric}"] = current_value
        row[f"fixed_reference_{metric}"] = reference_value
        row[f"delta_alpha360_minus_fixed_{metric}"] = (
            current_value - reference_value if same_signal_days else math.nan
        )
    return pd.DataFrame([row])


def load_index_cache(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    frame = _read_csv(path, "index cache", {"datetime", "index", "open", "close"})
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.loc[frame["index"].isin(["CSI300", "CSI1000"])].copy()
    if frame.duplicated(["datetime", "index"]).any():
        raise ValueError("index cache contains duplicate datetime/index rows")
    for benchmark in ("CSI300", "CSI1000"):
        subset = frame.loc[frame["index"].eq(benchmark)]
        if subset.empty:
            raise ValueError(f"index cache contains no {benchmark} rows")
        if not np.isfinite(subset[["open", "close"]].to_numpy(float)).all():
            raise ValueError(f"index cache contains non-finite {benchmark} prices")
    calendar = pd.DatetimeIndex(
        sorted(frame.loc[frame["index"].eq("CSI1000"), "datetime"].unique())
    )
    return frame, calendar


def benchmark_equity_curve(
    index: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    signal_dates: list[pd.Timestamp],
    horizon: str,
    benchmark: str,
) -> pd.Series:
    entry_field, exit_field, sleeve_count = HORIZON_EXECUTION[horizon]
    prices = (
        index.loc[index["index"].eq(benchmark)]
        .drop_duplicates("datetime")
        .set_index("datetime")
        .sort_index()
    )
    calendar_positions = {date: position for position, date in enumerate(calendar)}
    sleeves = np.ones(sleeve_count, dtype=float)
    values: list[float] = []
    ordered = sorted(pd.Timestamp(date) for date in signal_dates)
    for number, signal in enumerate(ordered):
        if signal not in calendar_positions:
            raise ValueError(f"signal date {signal.date()} is missing from CSI1000 calendar")
        position = calendar_positions[signal]
        if position + 2 >= len(calendar):
            raise ValueError(f"signal date {signal.date()} lacks T+2 benchmark data")
        entry_date, exit_date = calendar[position + 1], calendar[position + 2]
        if entry_date not in prices.index or exit_date not in prices.index:
            raise ValueError(
                f"{benchmark}/{horizon} missing entry or exit price for signal {signal.date()}"
            )
        entry = _require_finite(prices.loc[entry_date, entry_field], "benchmark entry")
        exit_price = _require_finite(prices.loc[exit_date, exit_field], "benchmark exit")
        if entry <= 0 or exit_price <= 0:
            raise ValueError(f"{benchmark}/{horizon} has a non-positive price")
        sleeves[number % sleeve_count] *= exit_price / entry
        values.append(float(sleeves.mean()))
    return pd.Series(values, index=pd.DatetimeIndex(ordered), name=benchmark)


def build_concise_summary(
    manifest: dict[str, Any],
    test_summary: pd.DataFrame,
    strict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    test_by_horizon = test_summary.set_index("horizon")
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        selection = manifest["selections"][horizon]
        selection_metrics = selection["selection_valid_metrics"]
        test_metrics = test_by_horizon.loc[horizon]
        expected_components = len(selection["selected_components"])
        if int(test_metrics["components"]) != expected_components:
            raise ValueError(f"Test summary/{horizon} component count disagrees with selection")
        row: dict[str, Any] = {
            "horizon": horizon,
            "horizon_label": HORIZON_LABELS[horizon],
            "selected_components": "+".join(selection["selected_components"]),
            "component_count": expected_components,
            "selection_days": int(selection_metrics["days"]),
            "selection_rows": int(selection_metrics["rows"]),
            "test_days": int(test_metrics["days"]),
            "test_rows": int(test_metrics["rows"]),
        }
        for metric in PREDICTION_METRICS:
            row[f"selection_{metric}"] = selection_metrics[metric]
            row[f"test_{metric}"] = test_metrics[metric]
        for variant in ("mainboard", "all"):
            data = strict[variant]
            baseline = data["baseline"].loc[horizon]
            selected = data["selected"].loc[horizon]
            chosen = data["chosen"].loc[horizon]
            prefix = f"{variant}_"
            row.update({
                f"{prefix}selection_rule": data["rules"][horizon],
                f"{prefix}selection_net_cumulative": chosen["selection_net_cumulative"],
                f"{prefix}selection_win_rate": chosen["selection_trade_win_rate"],
                f"{prefix}selection_max_drawdown": chosen["selection_max_drawdown"],
                f"{prefix}top1_baseline_net_cumulative": baseline["net_cumulative"],
                f"{prefix}top1_baseline_win_rate": baseline["trade_win_rate"],
                f"{prefix}top1_baseline_max_drawdown": baseline["max_drawdown_marked"],
                f"{prefix}top1_baseline_excess_vs_CSI300": baseline["net_excess_vs_CSI300"],
                f"{prefix}top1_baseline_excess_vs_CSI1000": baseline["net_excess_vs_CSI1000"],
                f"{prefix}selection_rule_test_net_cumulative": selected["net_cumulative"],
                f"{prefix}selection_rule_test_win_rate": selected["trade_win_rate"],
                f"{prefix}selection_rule_test_max_drawdown": selected["max_drawdown_marked"],
                f"{prefix}CSI300_cumulative": selected["CSI300_gross_cumulative"],
                f"{prefix}CSI1000_cumulative": selected["CSI1000_gross_cumulative"],
                f"{prefix}selection_rule_excess_vs_CSI300": selected["net_excess_vs_CSI300"],
                f"{prefix}selection_rule_excess_vs_CSI1000": selected["net_excess_vs_CSI1000"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_strict_execution_summary(
    strict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Return normalized executable Top1 baseline/selected-rule rows."""

    rows: list[dict[str, Any]] = []
    for variant in ("mainboard", "all"):
        data = strict[variant]
        execution_policy = "fallback" if data["method"]["fallback"] else "leave_cash"
        for result_type, frame in (
            ("top1_baseline", data["baseline"]),
            ("selected_rule", data["selected"]),
        ):
            for horizon in HORIZONS:
                source = frame.loc[horizon]
                row = {
                    "board_variant": variant,
                    "execution_policy": execution_policy,
                    "result_type": result_type,
                    "horizon": horizon,
                    "horizon_label": HORIZON_LABELS[horizon],
                    "rule": source["rule"],
                    "topk": int(source["topk"]),
                    "slippage_bps_each_side": float(source["slippage_bps_each_side"]),
                }
                for metric in STRICT_EXECUTION_METRICS:
                    row[metric] = source[metric]
                rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 2 * 2 * len(HORIZONS):
        raise RuntimeError("Strict execution summary does not contain the expected 16 rows")
    return result


def build_topk_slippage_sensitivity(
    strict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Return the complete friction-aware baseline TopK x 0/5 bps grid."""

    rows: list[dict[str, Any]] = []
    for variant in ("mainboard", "all"):
        data = strict[variant]
        execution_policy = "fallback" if data["method"]["fallback"] else "leave_cash"
        ordered = data["sensitivity"].sort_values(
            ["horizon", "topk", "slippage_bps_each_side"]
        )
        for source in ordered.itertuples(index=False):
            row = {
                "board_variant": variant,
                "execution_policy": execution_policy,
                "result_type": "mean_all_baseline",
                "horizon": source.horizon,
                "horizon_label": HORIZON_LABELS[source.horizon],
                "rule": source.rule,
                "topk": int(source.topk),
                "slippage_bps_each_side": float(source.slippage_bps_each_side),
            }
            for metric in STRICT_EXECUTION_METRICS:
                row[metric] = getattr(source, metric)
            rows.append(row)
    result = pd.DataFrame(rows)
    expected_rows = 2 * len(HORIZONS) * len(SENSITIVITY_TOPKS) * len(
        SENSITIVITY_SLIPPAGE_BPS
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"TopK/slippage sensitivity contains {len(result)} rows, expected {expected_rows}"
        )
    return result


def plot_prediction_metrics(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    labels = [HORIZON_LABELS[horizon].replace(" → ", "\n→ ") for horizon in HORIZONS]
    for axis, metric in zip(axes.flat, CORE_PREDICTION_METRICS):
        width = 0.36
        axis.bar(x - width / 2, summary[f"selection_{metric}"], width, label="Selection-valid")
        axis.bar(x + width / 2, summary[f"test_{metric}"], width, label="Test")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(metric.upper())
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    figure.suptitle("Four-horizon predictive metrics (Test is evaluation-only)", fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_equity_curves(
    strict: dict[str, dict[str, Any]],
    index: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    signal_dates: list[pd.Timestamp],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=False, constrained_layout=True)
    for row, variant in enumerate(("mainboard", "all")):
        data = strict[variant]
        for column, horizon in enumerate(HORIZONS):
            axis = axes[row, column]
            for key, label, color in (
                ("baseline", "Top1 baseline", "#64748b"),
                ("selected", "Selection-rule", "#2563eb"),
            ):
                daily = data["daily"][horizon][key]
                axis.plot(
                    daily["datetime"], daily["equity_mark"] / data["capital"],
                    label=label, color=color, linewidth=1.6,
                )
            # Strict-account daily curves can extend beyond the final signal
            # date while an untradeable exit is retried.  Those mark-to-market
            # dates are not additional benchmark entry signals.  The benchmark
            # must use the immutable prediction signal calendar in every board
            # variant, including dates where the strategy leaves cash.
            benchmark_dates = signal_dates
            for benchmark, color in (("CSI300", "#f59e0b"), ("CSI1000", "#16a34a")):
                curve = benchmark_equity_curve(
                    index, calendar, benchmark_dates, horizon, benchmark
                )
                expected = data["selected"].loc[horizon, f"{benchmark}_gross_cumulative"]
                _assert_close(
                    curve.iloc[-1] - 1.0,
                    expected,
                    f"{variant}/{horizon}/{benchmark} terminal benchmark",
                    tolerance=1e-7,
                )
                axis.plot(curve.index, curve.values, label=benchmark, color=color, linewidth=1.3)
            axis.axhline(1.0, color="black", linewidth=0.7, alpha=0.6)
            axis.set_title(f"{variant} | {HORIZON_LABELS[horizon]}", fontsize=10)
            axis.grid(alpha=0.2)
            axis.tick_params(axis="x", rotation=25, labelsize=8)
            if column == 0:
                axis.set_ylabel("Net value")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.suptitle("Strict Test account equity and contemporaneous benchmarks", fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def write_method_and_findings(
    output: Path,
    summary: pd.DataFrame,
    manifest_path: Path,
    selection_predictions: Path,
    test_summary_path: Path,
    test_predictions: Path,
    index_cache: Path,
    strict: dict[str, dict[str, Any]],
    strict_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fixed_comparison: pd.DataFrame,
    fixed_reference: dict[str, Any],
) -> None:
    lines = [
        "# Alpha360 probabilistic experiment report",
        "",
        "## Lockbox protocol",
        "",
        "- Model components were selected only on `selection_valid`; the immutable selection manifest has `test_files_read=false`.",
        "- Trading rules were selected only on `selection_valid` and frozen in each `chosen_rule_manifest_pre_test.json` before Test was opened.",
        "- Test is opened only after model components and trading rules are frozen. The frozen Test artifact may be reread by deterministic aggregation, backtest, and report stages, but no Test metric is fed back into re-ranking, re-selection, threshold tuning, or rule choice.",
        "- The report generator is read-only with respect to every input and fails if a Test diagnostic rule grid is present.",
        "",
        "## Known experiment limitations",
        "",
        "- E6 uses a bounded current-S&P-500 approximation for its US stock tokens. This has universe-level survivorship bias, so E6 is exploratory cross-market evidence rather than a point-in-time, unbiased US OOS experiment.",
        "- E0--E5 record `minimum_learning_rate=1e-6`, but scheduler step ordering means epoch 50 actually trains at about `1.33e-6` and reaches `1e-6` only after that epoch. E6 was corrected before training so epoch 50 itself uses `1e-6`.",
        "- E0 was migrated to the lockbox code after epoch 43. Before migration, the old startup verifier hashed the Test partition bytes. There is no evidence that Test inference, labels, metrics, or backtests were produced or used for selection before the freeze, but this run must not be described as never having read any Test byte. E1--E6 were byte-locked from their training start.",
        "- `all` means every CSI1000 name supplied by the prediction file, including STAR and ChiNext; it does not mean every listed A-share. `mainboard` means CSI1000 after excluding STAR and ChiNext.",
        "- Test is one continuous held-out interval, not several independent market regimes. E0--E6 candidates, ensemble alternatives, and trading rules were compared on one Selection-valid interval, so multiple-comparison/Selection overfitting remains possible.",
        "- Test results are descriptive only. They cannot be used to change components, thresholds, ranking rules, or execution policy without creating a new experiment and a new untouched Test interval.",
        "- Mixture coverage uses moment-matched Gaussian intervals rather than exact Gaussian-mixture quantiles.",
        "- Execution uses daily open/close prices plus fixed slippage. It enforces exact daily limits and suspension, but cannot model queue position, intraday liquidity, partial fills, volume participation, or market impact; a non-one-word limit price is not a guarantee of a real fill.",
        "- Stamp tax is fixed at zero by the requested protocol, which overstates net proceeds relative to an account that pays it.",
        "- The 100,000-yuan, 100-share-lot Top1 accounts are highly concentrated and exposed to single-name event risk.",
        "- The legacy Fixed fallback reference only searches its first 20 names, whereas the new fallback engine walks the complete supplied ranking, so fallback comparisons are approximate; leave-cash is the cleaner execution-policy comparison.",
        "- Frictionless Top-K cumulative returns are diagnostics, not executable account returns, particularly for overlapping horizons.",
        "- No block bootstrap, confidence interval, or formal significance test is claimed; a positive result over this Test window alone is not proof of persistent alpha.",
        "",
        "## Strict execution assumptions",
        "",
    ]
    for variant in ("mainboard", "all"):
        method = strict[variant]["method"]
        lines.extend([
            f"### {variant}",
            "",
            f"- Initial capital: `{method['capital']}`; lot size: `{method['lot_size']}` shares.",
            f"- Commission: `{method['commission_rate_each_side']}` each side, minimum `{method['minimum_commission']}`; stamp tax: `{method['stamp_tax']}`.",
            f"- Slippage used for frozen rule reporting: `{strict[variant]['slippage']}` bps each side.",
            f"- Buy fallback: `{method['fallback']}`; exact daily up/down limits are enforced; limit-down exits follow `{method['limit_down_exit']}`.",
            f"- Board filter: `{'STAR and ChiNext excluded' if variant == 'mainboard' else 'all supplied stocks, including STAR/ChiNext'}`.",
            "- `open1_close2` uses two independent half-capital temporal sleeves; other horizons use one sleeve.",
            "",
        ])
    lines.extend([
        "## Frozen model components and predictive metrics",
        "",
        "| Horizon | Components | Sel days/rows | Test days/rows | Sel/Test RankIC | Sel/Test RankICIR | Sel/Test NLL | Sel/Test MAE | Sel/Test Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.selected_components} | "
            f"{int(row.selection_days)}/{int(row.selection_rows)} | "
            f"{int(row.test_days)}/{int(row.test_rows)} | "
            f"{row.selection_rank_ic:.4f} / {row.test_rank_ic:.4f} | "
            f"{row.selection_rank_icir:.4f} / {row.test_rank_icir:.4f} | "
            f"{row.selection_nll:.4f} / {row.test_nll:.4f} | "
            f"{row.selection_mae:.4f} / {row.test_mae:.4f} | "
            f"{row.selection_brier:.4f} / {row.test_brier:.4f} |"
        )
    lines.extend([
        "",
        "### Calibration and frictionless ranking diagnostics",
        "",
        "These diagnostics use realized labels and therefore never participate in model or rule selection. "
        "Coverage is exact for a single Gaussian and moment-matched for a Gaussian mixture. "
        "Top-K cumulative return below is a frictionless daily diagnostic; the strict account table is the executable result.",
        "",
        "| Horizon | Split | Days | Rows | Direction acc. | Coverage 50/80/95 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        for split, prefix in (("Selection-valid", "selection"), ("Test", "test")):
            lines.append(
                f"| {row.horizon} | {split} | {int(getattr(row, prefix + '_days'))} | "
                f"{int(getattr(row, prefix + '_rows'))} | "
                f"{_percent(getattr(row, prefix + '_direction_accuracy'))} | "
                f"{_percent(getattr(row, prefix + '_coverage_50'))} / "
                f"{_percent(getattr(row, prefix + '_coverage_80'))} / "
                f"{_percent(getattr(row, prefix + '_coverage_95'))} |"
            )
    lines.extend([
        "",
        "### Frictionless Top-K ranking diagnostics",
        "",
        "`Portfolio win` is the fraction of positive daily equal-weight Top-K portfolios; "
        "`stock win` is the fraction of positive individual selected stock returns.",
        "",
        "| Horizon | Split | K | Portfolio win | Stock win | Mean return | Cumulative |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        for split, prefix in (("Selection-valid", "selection"), ("Test", "test")):
            for topk in (1, 3, 5, 10):
                lines.append(
                    f"| {row.horizon} | {split} | {topk} | "
                    f"{_percent(getattr(row, f'{prefix}_top{topk}_win_rate'))} | "
                    f"{_percent(getattr(row, f'{prefix}_top{topk}_stock_win_rate'))} | "
                    f"{_percent(getattr(row, f'{prefix}_top{topk}_mean_return'))} | "
                    f"{_percent(getattr(row, f'{prefix}_top{topk}_cumulative'))} |"
                )
    lines.extend([
        "",
        "## Strict Test observations (descriptive, never used for selection)",
        "",
        "Both the executable Top1 baseline and the uncertainty rule frozen on Selection-valid "
        "are shown at the pre-registered reporting slippage.",
        "",
        "### Performance, costs, and benchmarks",
        "",
        "| Board | Result | Horizon | Rule | Net | Win | DD | Sharpe | Trades | Fees | Daily turnover | CSI300 | Excess | CSI1000 | Excess |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in strict_summary.itertuples(index=False):
        lines.append(
            f"| {row.board_variant} | {row.result_type} | {row.horizon} | {row.rule} | "
            f"{_percent(row.net_cumulative)} | {_percent(row.trade_win_rate)} | "
            f"{_percent(row.max_drawdown_marked)} | {float(row.net_sharpe_rf0):.3f} | "
            f"{int(row.completed_trades)} | {float(row.total_commission):.2f} | "
            f"{_percent(row.average_daily_turnover)} | "
            f"{_percent(row.CSI300_gross_cumulative)} | "
            f"{_percent(row.net_excess_vs_CSI300)} | "
            f"{_percent(row.CSI1000_gross_cumulative)} | "
            f"{_percent(row.net_excess_vs_CSI1000)} |"
        )
    lines.extend([
        "",
        "### Exposure, concentration, and execution diagnostics",
        "",
        "Buy blocks are `up-limit / missing quote-or-limit / suspended / too expensive`; "
        "sell blocks are `down-limit / missing quote / suspended`.",
        "",
        "| Board | Result | Horizon | Active/signal | Exposure avg/max | Name concentration avg/max | Buy blocks | Fallback/cash | Delayed exits | Sell blocks | Unresolved |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in strict_summary.itertuples(index=False):
        lines.append(
            f"| {row.board_variant} | {row.result_type} | {row.horizon} | "
            f"{int(row.active_signal_days)}/{int(row.signal_days)} | "
            f"{_percent(row.average_gross_exposure)} / {_percent(row.max_gross_exposure)} | "
            f"{_percent(row.average_max_name_concentration)} / "
            f"{_percent(row.max_name_concentration)} | "
            f"{int(row.blocked_buy_up_limit)} / {int(row.blocked_buy_missing)} / "
            f"{int(row.blocked_buy_suspended)} / {int(row.too_expensive)} | "
            f"{int(row.fallback_replacements)} / {int(row.filtered_cash_slots)} | "
            f"{int(row.delayed_exit_trades)} | "
            f"{int(row.blocked_sell_down_limit_attempts)} / "
            f"{int(row.blocked_sell_missing_attempts)} / "
            f"{int(row.blocked_sell_suspended_attempts)} | "
            f"{int(row.unresolved_exit)} |"
        )
    lines.extend([
        "",
        "### Executable Top-K x slippage sensitivity (mean-score baseline)",
        "",
        "This is the strict `mean_all` baseline grid, not a post-Test search for a new rule. "
        "It reports every pre-generated Top1/3/5/10 account at 0 and 5 bps each side.",
        "",
        "| Board | Horizon | TopK | bps/side | Net | Win | DD | Sharpe | Trades | Fees | CSI300 | Excess | CSI1000 | Excess |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.board_variant} | {row.horizon} | {int(row.topk)} | "
            f"{float(row.slippage_bps_each_side):g} | {_percent(row.net_cumulative)} | "
            f"{_percent(row.trade_win_rate)} | {_percent(row.max_drawdown_marked)} | "
            f"{float(row.net_sharpe_rf0):.3f} | {int(row.completed_trades)} | "
            f"{float(row.total_commission):.2f} | "
            f"{_percent(row.CSI300_gross_cumulative)} | "
            f"{_percent(row.net_excess_vs_CSI300)} | "
            f"{_percent(row.CSI1000_gross_cumulative)} | "
            f"{_percent(row.net_excess_vs_CSI1000)} |"
        )
    lines.extend([
        "",
        "The differences between Selection-valid and Test are out-of-sample findings, not a reason to switch components or rules. Any follow-up design must be registered and evaluated on a future lockbox.",
        "",
        "## Fixed Ensemble external historical reference",
        "",
        "- This is a descriptive external historical baseline only. It does not participate in model Selection, trading-rule Selection, or Test gating.",
        "- The comparison is restricted to `close1_close2` / `mainboard` / `Top1`, with buy time `15:00`, exit time `15:00`, and `5` bps slippage each side.",
        (
            "- `leave_cash` is the strongest same-definition comparison."
            if fixed_reference["execution_policy"] == "leave_cash"
            else "- `fallback` is approximate: the old Fixed implementation searched at most through rank 20, while the new strict implementation traverses the complete ranking."
        ),
        (
            "- Positive deltas below mean the current Alpha360 result is numerically larger than the Fixed reference; for drawdown, a positive delta means a shallower (better) drawdown."
            if bool(fixed_comparison.iloc[0].same_signal_days)
            else "- The signal-day counts differ, so the two standalone historical results are shown but their deltas are deliberately omitted; this is not an identical-window comparison."
        ),
        "",
        "| Policy | Comparability | Metric | Alpha360 | Fixed reference | Delta (Alpha360 − Fixed) |",
        "|---|---|---|---:|---:|---:|",
    ])
    comparison = fixed_comparison.iloc[0]
    for metric in ("net_cumulative", "max_drawdown", "trade_win_rate"):
        delta = comparison[f"delta_alpha360_minus_fixed_{metric}"]
        lines.append(
            f"| {comparison.execution_policy} | {comparison.comparability} | {metric} | "
            f"{_percent(comparison[f'alpha360_{metric}'])} | "
            f"{_percent(comparison[f'fixed_reference_{metric}'])} | "
            f"{_percent(delta) if pd.notna(delta) else 'N/A'} |"
        )
    completed_delta = comparison.delta_alpha360_minus_fixed_completed_trades
    lines.append(
        f"| {comparison.execution_policy} | {comparison.comparability} | completed_trades | "
        f"{int(comparison.alpha360_completed_trades)} | "
        f"{int(comparison.fixed_reference_completed_trades)} | "
        f"{int(completed_delta) if pd.notna(completed_delta) else 'N/A'} |"
    )
    lines.append(
        f"\nSignal days: Alpha360={int(comparison.alpha360_signal_days)}, "
        f"Fixed reference={int(comparison.fixed_reference_signal_days)}."
    )
    lines.extend([
        "",
        "## Input audit",
        "",
        "| Input | SHA-256 |",
        "|---|---|",
    ])
    audited = {
        "selection manifest": manifest_path,
        "selection-valid ensemble predictions": selection_predictions,
        "Test summary": test_summary_path,
        "Test predictions": test_predictions,
        "index cache": index_cache,
        "mainboard pre-Test rule manifest": strict["mainboard"]["directory"] / "chosen_rule_manifest_pre_test.json",
        "all pre-Test rule manifest": strict["all"]["directory"] / "chosen_rule_manifest_pre_test.json",
        "Fixed Ensemble external historical summary": fixed_reference["path"],
    }
    for variant in ("mainboard", "all"):
        for name, reference in strict[variant]["execution_data_inputs"].items():
            audited[f"{variant} execution input: {name}"] = Path(reference["path"])
    for label, path in audited.items():
        lines.append(f"| {label} | `{sha256(path)}` |")
    lines.extend([
        "",
        "Generated files: `concise_summary.csv`, `model_selection.csv`, "
        "`strict_execution_summary.csv`, `topk_slippage_sensitivity.csv`, "
        "`fixed_reference_comparison.csv`, `prediction_metrics_comparison.png`, "
        "and `strategy_equity_curves.png`.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def generate_report(
    *,
    selection_manifest_path: Path,
    selection_predictions_path: Path,
    test_summary_path: Path,
    test_predictions_path: Path,
    mainboard_backtest_dir: Path,
    all_backtest_dir: Path,
    index_cache_path: Path,
    fixed_reference_summary_path: Path,
    execution_policy: str,
    output_directory: Path,
) -> Path:
    inputs = [
        selection_manifest_path,
        selection_predictions_path,
        test_summary_path,
        test_predictions_path,
        index_cache_path,
        fixed_reference_summary_path,
    ]
    resolved_inputs = [_require_file(path, "report input") for path in inputs]
    (
        selection_manifest_path,
        selection_predictions_path,
        test_summary_path,
        test_predictions_path,
        index_cache_path,
        fixed_reference_summary_path,
    ) = resolved_inputs
    input_hashes_before = {path: sha256(path) for path in resolved_inputs}
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f"Report output already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    try:
        selection_predictions = load_predictions(
            selection_predictions_path, "selection-valid ensemble predictions"
        )
        test_predictions = load_predictions(test_predictions_path, "Test predictions")
        manifest = load_selection_manifest(
            selection_manifest_path, selection_predictions_path
        )
        test_summary = load_test_summary(test_summary_path)
        for horizon in HORIZONS:
            selection_metrics = manifest["selections"][horizon]["selection_valid_metrics"]
            validate_metric_mapping(
                selection_predictions, selection_metrics, horizon, "selection manifest"
            )
            test_row = test_summary.set_index("horizon").loc[horizon]
            validate_metric_mapping(test_predictions, test_row, horizon, "Test summary")
        strict = {
            "mainboard": load_strict_backtest(
                mainboard_backtest_dir,
                "mainboard",
                selection_predictions_path,
                test_predictions_path,
            ),
            "all": load_strict_backtest(
                all_backtest_dir,
                "all",
                selection_predictions_path,
                test_predictions_path,
            ),
        }
        if execution_policy not in {"fallback", "leave_cash"}:
            raise ValueError(
                "execution_policy must be exactly 'fallback' or 'leave_cash'"
            )
        expected_fallback = execution_policy == "fallback"
        for variant, data in strict.items():
            if data["method"].get("fallback") is not expected_fallback:
                raise ValueError(
                    f"{variant} strict backtest fallback does not match explicit "
                    f"execution_policy={execution_policy}"
                )
        fixed_reference = load_fixed_reference(
            fixed_reference_summary_path, execution_policy
        )
        fixed_comparison = build_fixed_reference_comparison(
            fixed_reference, strict["mainboard"]
        )
        summary = build_concise_summary(manifest, test_summary, strict)
        model_selection = build_model_selection(manifest)
        strict_summary = build_strict_execution_summary(strict)
        sensitivity = build_topk_slippage_sensitivity(strict)
        index, calendar = load_index_cache(index_cache_path)
        signal_dates = list(test_predictions["datetime"].drop_duplicates().sort_values())
        summary.to_csv(output_directory / "concise_summary.csv", index=False)
        model_selection.to_csv(output_directory / "model_selection.csv", index=False)
        strict_summary.to_csv(
            output_directory / "strict_execution_summary.csv", index=False
        )
        sensitivity.to_csv(
            output_directory / "topk_slippage_sensitivity.csv", index=False
        )
        fixed_comparison.to_csv(
            output_directory / "fixed_reference_comparison.csv", index=False
        )
        plot_prediction_metrics(
            summary, output_directory / "prediction_metrics_comparison.png"
        )
        plot_equity_curves(
            strict,
            index,
            calendar,
            signal_dates,
            output_directory / "strategy_equity_curves.png",
        )
        write_method_and_findings(
            output_directory / "method_and_findings.md",
            summary,
            selection_manifest_path,
            selection_predictions_path,
            test_summary_path,
            test_predictions_path,
            index_cache_path,
            strict,
            strict_summary,
            sensitivity,
            fixed_comparison,
            fixed_reference,
        )
        input_hashes_after = {path: sha256(path) for path in resolved_inputs}
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("an input file changed while the read-only report was generated")
    except Exception:
        # Keep failures atomic: an invalid report must never look complete.
        for path in output_directory.glob("*"):
            if path.is_file():
                path.unlink()
        output_directory.rmdir()
        raise
    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-predictions", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--mainboard-backtest-dir", type=Path, required=True)
    parser.add_argument("--all-backtest-dir", type=Path, required=True)
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX_CACHE)
    parser.add_argument("--fixed-reference-summary", type=Path, required=True)
    parser.add_argument(
        "--execution-policy", choices=("fallback", "leave_cash"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_report(
        selection_manifest_path=args.selection_manifest,
        selection_predictions_path=args.selection_predictions,
        test_summary_path=args.test_summary,
        test_predictions_path=args.test_predictions,
        mainboard_backtest_dir=args.mainboard_backtest_dir,
        all_backtest_dir=args.all_backtest_dir,
        index_cache_path=args.index_cache,
        fixed_reference_summary_path=args.fixed_reference_summary,
        execution_policy=args.execution_policy,
        output_directory=args.output,
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
