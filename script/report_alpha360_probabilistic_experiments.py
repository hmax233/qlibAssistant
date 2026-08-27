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
PREDICTION_METRICS = ("rank_ic", "rank_icir", "nll", "mae", "brier")
SUMMARY_METRICS = ("days", "rows", "components", *PREDICTION_METRICS)


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
        columns = prediction_columns(horizon) - {"datetime", "instrument"}
        numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{label}/{horizon} contains non-finite prediction values")
        if (numeric[f"{horizon}_log_variance"] <= 0).any():
            raise ValueError(f"{label}/{horizon} contains non-positive variance")
        probabilities = numeric[f"{horizon}_probability_positive"]
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{label}/{horizon} probability must be in [0, 1]")
        frame[list(columns)] = numeric
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
    return {
        "days": int(frame["datetime"].nunique()),
        "rows": int(len(frame)),
        "rank_ic": float(daily.mean()),
        "rank_icir": float(daily.mean() / deviation) if deviation > 0 else math.nan,
        "mae": float(np.mean(np.abs(expected - actual))),
        "brier": float(np.mean((probability - (actual > 0)) ** 2)),
    }


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
    for metric in ("days", "rows", "rank_ic", "mae", "brier"):
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
        "net_cumulative",
        "trade_win_rate",
        "max_drawdown_marked",
        "CSI1000_gross_cumulative",
        "net_excess_vs_CSI1000",
        "CSI300_gross_cumulative",
        "net_excess_vs_CSI300",
    }
    baseline = _read_csv(
        directory / "test_baseline_four_horizons.csv",
        f"{variant} Test baseline",
        common_summary_columns,
    )
    selected = _read_csv(
        directory / "test_selected_uncertainty_rules.csv",
        f"{variant} Test selection-rule",
        common_summary_columns,
    )
    baseline = baseline.loc[
        baseline["topk"].eq(1)
        & np.isclose(baseline["slippage_bps_each_side"], slippage)
    ].copy()
    selected = selected.loc[
        selected["topk"].eq(1)
        & np.isclose(selected["slippage_bps_each_side"], slippage)
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
        "chosen": chosen,
        "baseline": baseline,
        "selected": selected,
        "daily": daily,
    }


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


def plot_prediction_metrics(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    labels = [HORIZON_LABELS[horizon].replace(" → ", "\n→ ") for horizon in HORIZONS]
    for axis, metric in zip(axes.flat, PREDICTION_METRICS):
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
            variant_signal_dates = list(
                data["daily"][horizon]["selected"]["datetime"]
            )
            if variant_signal_dates != signal_dates:
                # A board filter can theoretically leave a date with no eligible
                # stocks.  Use the exact dates evaluated by that strict account.
                benchmark_dates = variant_signal_dates
            else:
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
) -> None:
    lines = [
        "# Alpha360 probabilistic experiment report",
        "",
        "## Lockbox protocol",
        "",
        "- Model components were selected only on `selection_valid`; the immutable selection manifest has `test_files_read=false`.",
        "- Trading rules were selected only on `selection_valid` and frozen in each `chosen_rule_manifest_pre_test.json` before Test was opened.",
        "- Test is evaluated exactly once for the frozen model components and frozen trading rules. No Test metric is used for re-ranking, re-selection, threshold tuning, or rule choice.",
        "- The report generator is read-only with respect to every input and fails if a Test diagnostic rule grid is present.",
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
        "| Horizon | Components | Sel RankIC | Test RankIC | Sel NLL | Test NLL | Sel MAE | Test MAE | Sel Brier | Test Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.selected_components} | {row.selection_rank_ic:.4f} | "
            f"{row.test_rank_ic:.4f} | {row.selection_nll:.4f} | {row.test_nll:.4f} | "
            f"{row.selection_mae:.4f} | {row.test_mae:.4f} | "
            f"{row.selection_brier:.4f} | {row.test_brier:.4f} |"
        )
    lines.extend([
        "",
        "## Strict Test observations (descriptive, never used for selection)",
        "",
        "| Horizon | Mainboard rule/net/win/DD | All-pool rule/net/win/DD | Mainboard excess CSI1000/CSI300 | All-pool excess CSI1000/CSI300 |",
        "|---|---|---|---|---|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.mainboard_selection_rule}: "
            f"{_percent(row.mainboard_selection_rule_test_net_cumulative)} / "
            f"{_percent(row.mainboard_selection_rule_test_win_rate)} / "
            f"{_percent(row.mainboard_selection_rule_test_max_drawdown)} | "
            f"{row.all_selection_rule}: {_percent(row.all_selection_rule_test_net_cumulative)} / "
            f"{_percent(row.all_selection_rule_test_win_rate)} / "
            f"{_percent(row.all_selection_rule_test_max_drawdown)} | "
            f"{_percent(row.mainboard_selection_rule_excess_vs_CSI1000)} / "
            f"{_percent(row.mainboard_selection_rule_excess_vs_CSI300)} | "
            f"{_percent(row.all_selection_rule_excess_vs_CSI1000)} / "
            f"{_percent(row.all_selection_rule_excess_vs_CSI300)} |"
        )
    lines.extend([
        "",
        "The differences between Selection-valid and Test are out-of-sample findings, not a reason to switch components or rules. Any follow-up design must be registered and evaluated on a future lockbox.",
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
    }
    for label, path in audited.items():
        lines.append(f"| {label} | `{sha256(path)}` |")
    lines.extend([
        "",
        "Generated files: `concise_summary.csv`, `model_selection.csv`, `prediction_metrics_comparison.png`, and `strategy_equity_curves.png`.",
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
    output_directory: Path,
) -> Path:
    inputs = [
        selection_manifest_path,
        selection_predictions_path,
        test_summary_path,
        test_predictions_path,
        index_cache_path,
    ]
    resolved_inputs = [_require_file(path, "report input") for path in inputs]
    (
        selection_manifest_path,
        selection_predictions_path,
        test_summary_path,
        test_predictions_path,
        index_cache_path,
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
        summary = build_concise_summary(manifest, test_summary, strict)
        model_selection = build_model_selection(manifest)
        index, calendar = load_index_cache(index_cache_path)
        signal_dates = list(test_predictions["datetime"].drop_duplicates().sort_values())
        summary.to_csv(output_directory / "concise_summary.csv", index=False)
        model_selection.to_csv(output_directory / "model_selection.csv", index=False)
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
        output_directory=args.output,
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
