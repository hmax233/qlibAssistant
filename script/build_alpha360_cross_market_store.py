#!/usr/bin/env python3
"""Build an E6 A-share/US Alpha360 store without modifying either source.

The A-share side is referenced byte-for-byte from an existing E0 DateStore.
Yahoo US parquet files are converted to Qlib's field-major Alpha360 layout and
aligned conservatively: an A-share signal date T can only see a US close whose
calendar date is strictly earlier than T.

The output consists of uncompressed ``.npy`` chunks so consumers can use
``numpy.load(..., mmap_mode="r")``.  A resumable build state pins every input by
content hash and mtime; resuming with changed inputs is rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_US_RAW = Path("/Users/hmax/investment_data/cross_market_daily/raw/us")
DEFAULT_UNIVERSE = ROOT / ".qlibAssistant/cross_market/universes/us_universe.csv"
DEFAULT_OUTPUT = ROOT / ".qlibAssistant/cross_market/alpha360_e6_store"
FIELDS = ("close", "open", "high", "low", "vwap", "volume")
FEATURE_WIDTH = 360
LOOKBACK = 60
A_SIGNAL_TIME = "15:00"
A_SIGNAL_TIMEZONE = "Asia/Shanghai"
US_CLOSE_TIME = "16:00"
US_CLOSE_TIMEZONE = "America/New_York"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def alpha360_feature_names() -> list[str]:
    return [f"{field.upper()}{lag}" for field in FIELDS for lag in range(59, -1, -1)]


def adjust_yahoo_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return split/dividend-adjusted OHLCV and an explicit VWAP proxy.

    Yahoo has no historical VWAP in these files.  OHLC is multiplied by
    ``adjclose / close``.  Volume is divided by that factor, which preserves
    the OHLC4-times-volume scale.  VWAP is the adjusted OHLC4 proxy.
    """

    lowered = frame.rename(columns={column: str(column).lower() for column in frame.columns})
    required = {"date", "open", "high", "low", "close", "volume", "adjclose"}
    missing = sorted(required.difference(lowered.columns))
    if missing:
        raise ValueError(f"Yahoo parquet missing required columns: {missing}")
    data = lowered.loc[:, sorted(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).sort_values("date")
    if data["date"].duplicated().any():
        duplicates = data.loc[data["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate Yahoo dates: {duplicates[:5]}")
    numeric = ["open", "high", "low", "close", "volume", "adjclose"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    factor = data["adjclose"] / data["close"]
    factor = factor.where(np.isfinite(factor) & (factor > 0))
    adjusted = pd.DataFrame({"date": data["date"]})
    for column in ("open", "high", "low", "close"):
        adjusted[column] = data[column] * factor
    # Force exact agreement with the authoritative adjusted close column.
    adjusted["close"] = data["adjclose"]
    adjusted["volume"] = data["volume"] / factor
    adjusted["vwap"] = adjusted[["open", "high", "low", "close"]].mean(axis=1)
    for column in FIELDS:
        adjusted.loc[~np.isfinite(adjusted[column]), column] = np.nan
    return adjusted[["date", *FIELDS]].reset_index(drop=True)


def construct_us_alpha360(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Construct Qlib-compatible field-major 360 features.

    ``values`` must be ``[date, close/open/high/low/vwap/volume]``.  Within each
    field the result runs from lag 59 (oldest) through lag 0 (as-of date).
    Prices and VWAP are divided by current adjusted close; volume is divided by
    current adjusted volume, exactly matching the E0 Alpha360 convention.
    """

    values = np.asarray(values, dtype="float32")
    positions = np.asarray(positions, dtype="int64")
    if values.ndim != 2 or values.shape[1] != len(FIELDS):
        raise ValueError(f"Expected [dates, 6] OHLCV values, got {values.shape}")
    if len(positions) and (positions.min() < LOOKBACK - 1 or positions.max() >= len(values)):
        raise ValueError("Alpha360 positions require 60 available observations")
    indices = positions[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :]
    sequence = values[indices].copy()
    current_close = values[positions, 0]
    current_volume = values[positions, 5]
    with np.errstate(divide="ignore", invalid="ignore"):
        sequence[:, :, :5] /= current_close[:, None, None]
        sequence[:, :, 5] /= current_volume[:, None]
    output = sequence.transpose(0, 2, 1).reshape(len(positions), FEATURE_WIDTH)
    output[~np.isfinite(output)] = np.nan
    return output.astype("float32", copy=False)


def _session_date_index(values: Iterable[pd.Timestamp], *, market: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.hasnans:
        raise ValueError(f"{market} calendar contains NaT")
    # Source dates are exchange-session labels, not instants.  Drop any source
    # timezone while preserving the displayed exchange date before attaching
    # the explicit exchange timezone below.
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _session_times_utc(
    dates: pd.DatetimeIndex,
    *,
    local_time: str,
    timezone: str,
) -> np.ndarray:
    hour, minute = (int(value) for value in local_time.split(":"))
    local = dates.tz_localize(timezone) + pd.Timedelta(hours=hour, minutes=minute)
    return local.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def strict_us_asof_alignment(
    a_dates: Iterable[pd.Timestamp], us_calendar: Iterable[pd.Timestamp]
) -> dict[str, np.ndarray]:
    """Return the latest US close already observable at each China signal.

    A signals are timestamped at 15:00 Asia/Shanghai.  A US session is only
    eligible after its 16:00 America/New_York close, with daylight saving time
    handled by the timezone database.  Comparing UTC instants makes the no-
    future-information rule explicit and auditable.
    """

    a_index = _session_date_index(a_dates, market="A-share")
    us_index = _session_date_index(us_calendar, market="US").unique().sort_values()
    a_signal_times = _session_times_utc(
        a_index, local_time=A_SIGNAL_TIME, timezone=A_SIGNAL_TIMEZONE
    )
    us_close_times = _session_times_utc(
        us_index, local_time=US_CLOSE_TIME, timezone=US_CLOSE_TIMEZONE
    )
    positions = np.searchsorted(us_close_times, a_signal_times, side="left") - 1
    asof_dates = np.full(len(a_index), np.datetime64("NaT"), dtype="datetime64[D]")
    aligned_close_times = np.full(
        len(a_index), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    valid = positions >= 0
    if valid.any():
        asof_dates[valid] = us_index.values[positions[valid]].astype("datetime64[D]")
        aligned_close_times[valid] = us_close_times[positions[valid]]
    if np.any(valid & ~(aligned_close_times < a_signal_times)):
        raise AssertionError("US close must be strictly earlier than the A signal instant")
    return {
        "a_dates": a_index.values.astype("datetime64[D]"),
        "a_signal_times_utc": a_signal_times,
        "us_asof_dates": asof_dates,
        "us_close_times_utc": aligned_close_times,
    }


def strict_us_asof_dates(
    a_dates: Iterable[pd.Timestamp], us_calendar: Iterable[pd.Timestamp]
) -> np.ndarray:
    """Compatibility wrapper returning exchange dates from strict UTC alignment."""

    return strict_us_asof_alignment(a_dates, us_calendar)["us_asof_dates"]


@dataclass
class RunningMoments:
    count: np.ndarray
    total: np.ndarray
    squared: np.ndarray

    @classmethod
    def empty(cls, width: int = FEATURE_WIDTH) -> "RunningMoments":
        return cls(
            np.zeros(width, dtype="int64"),
            np.zeros(width, dtype="float64"),
            np.zeros(width, dtype="float64"),
        )

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0).astype("float64", copy=False)
        self.count += finite.sum(axis=0)
        self.total += safe.sum(axis=0)
        self.squared += np.square(safe).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.maximum(self.count, 1)
        mean = self.total / denominator
        variance = np.maximum(self.squared / denominator - mean * mean, 0.0)
        std = np.sqrt(variance)
        std[(std < 1e-6) | ~np.isfinite(std)] = 1.0
        return mean.astype("float32"), std.astype("float32"), self.count


def _part_paths(a_store: Path, part: dict[str, Any]) -> dict[str, Path]:
    return {
        kind: a_store / f"{part['prefix']}_{kind}.npy"
        for kind in ("features", "labels", "stock_ids", "offsets")
    }


def validate_split_boundaries(manifest: dict[str, Any]) -> None:
    """Reject mislabeled, overlapping, or out-of-bound source parts."""

    expected = ("train", "valid", "selection_valid", "test")
    segments = manifest.get("segments", {})
    if set(segments) != set(expected):
        raise ValueError("E0 manifest must contain train/valid/selection_valid/test")
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    previous_end: pd.Timestamp | None = None
    for split in expected:
        values = segments[split]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"Invalid E0 segment bounds for {split}")
        start, end = (pd.Timestamp(value).normalize() for value in values)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"Invalid E0 segment bounds for {split}")
        if previous_end is not None and start <= previous_end:
            raise ValueError("E0 segments must be strictly ordered and non-overlapping")
        bounds[split] = (start, end)
        previous_end = end

    prefixes: set[str] = set()
    seen_dates: dict[pd.Timestamp, str] = {}
    dates_by_split = {split: [] for split in expected}
    for part in manifest.get("parts", []):
        prefix = str(part.get("prefix", ""))
        split = part.get("split")
        if not prefix or prefix in prefixes:
            raise ValueError(f"Duplicate or empty E0 part prefix: {prefix!r}")
        prefixes.add(prefix)
        if split not in bounds:
            raise ValueError(f"Unknown E0 part split: {split}")
        dates = [pd.Timestamp(value).normalize() for value in part.get("dates", [])]
        if not dates or any(pd.isna(value) for value in dates):
            raise ValueError(f"E0 part {prefix} has no valid dates")
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError(f"E0 part {prefix} dates must be sorted and unique")
        start, end = bounds[split]
        for value in dates:
            if value < start or value > end:
                raise ValueError(f"E0 part {prefix} date {value.date()} is outside {split}")
            if value in seen_dates:
                raise ValueError(
                    f"E0 date {value.date()} appears in both {seen_dates[value]} and {split}"
                )
            seen_dates[value] = split
        dates_by_split[split].extend(dates)
    for split, dates in dates_by_split.items():
        if not dates:
            raise ValueError(f"E0 manifest has no {split} parts")
        if dates != sorted(dates):
            raise ValueError(f"E0 {split} parts are not in chronological order")


def validate_a_store(a_store: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate E0 metadata and every memory-mapped array/hash."""

    manifest_path = a_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_names") != alpha360_feature_names():
        raise ValueError("E0 feature_names are not canonical field-major Alpha360")
    if int(manifest.get("stock_count", 0)) <= 0:
        raise ValueError("E0 manifest has no stocks")
    validate_split_boundaries(manifest)
    sidecars: dict[str, Any] = {}
    for name, key in (
        ("normalizer.npz", "normalizer_sha256"),
        ("stock_ids.json", "stock_ids_sha256"),
    ):
        path = a_store / name
        actual = sha256_file(path)
        if actual != manifest.get(key):
            raise RuntimeError(f"E0 hash mismatch: {name}")
        sidecars[name] = source_record(path)
    part_sources: dict[str, dict[str, Any]] = {}
    for part in manifest.get("parts", []):
        paths = _part_paths(a_store, part)
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
        rows = int(part["rows"])
        if arrays["features"].shape != (rows, FEATURE_WIDTH):
            raise ValueError(f"Invalid E0 feature shape in {part['prefix']}")
        if arrays["labels"].shape != (rows, 3):
            raise ValueError(f"Invalid E0 label shape in {part['prefix']}")
        if arrays["stock_ids"].shape != (rows,):
            raise ValueError(f"Invalid E0 stock-id shape in {part['prefix']}")
        offsets = arrays["offsets"]
        if offsets.shape != (len(part["dates"]) + 1,) or offsets[0] != 0 or offsets[-1] != rows:
            raise ValueError(f"Invalid E0 date offsets in {part['prefix']}")
        for name, path in paths.items():
            actual = sha256_file(path)
            expected = part["sha256"].get(path.name)
            if actual != expected:
                raise RuntimeError(f"E0 hash mismatch: {path.name}")
        part_sources[part["prefix"]] = {
            name: source_record(path) for name, path in paths.items()
        }
    return manifest, {
        "path": str(a_store.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_source": source_record(manifest_path),
        "sidecars": sidecars,
        "parts": part_sources,
    }


def universe_symbols(universe_path: Path, raw_dir: Path, maximum: int | None) -> list[str]:
    frame = pd.read_csv(universe_path)
    candidates = pd.Series(index=frame.index, dtype="object")
    for column in ("yahoo_symbol", "Symbol", "symbol"):
        if column in frame.columns:
            candidates = candidates.fillna(frame[column])
    candidates = candidates.dropna().astype(str).str.strip()
    if candidates is None:
        raise ValueError("US universe needs yahoo_symbol, Symbol, or symbol")
    symbols = sorted(dict.fromkeys(value for value in candidates if value))
    available = [symbol for symbol in symbols if (raw_dir / f"{symbol}.parquet").is_file()]
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("--max-us-stocks must be positive")
        available = available[:maximum]
    if not available:
        raise RuntimeError("No universe symbols have matching Yahoo parquet files")
    return available


def source_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def load_us_histories(
    raw_dir: Path, symbols: list[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DatetimeIndex, dict[str, Any]]:
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calendar_values: list[np.ndarray] = []
    invalid_rows = 0
    for symbol in symbols:
        adjusted = adjust_yahoo_ohlcv(pd.read_parquet(raw_dir / f"{symbol}.parquet"))
        dates = pd.DatetimeIndex(adjusted["date"])
        values = adjusted.loc[:, FIELDS].to_numpy(dtype="float32")
        invalid_rows += int((~np.isfinite(values).all(axis=1)).sum())
        histories[symbol] = (dates.values.astype("datetime64[D]"), values)
        calendar_values.append(dates.values.astype("datetime64[D]"))
    calendar = pd.DatetimeIndex(np.unique(np.concatenate(calendar_values))).sort_values()
    return histories, calendar, {"source_rows_with_any_missing_adjusted_field": invalid_rows}


def _selected_parts(
    manifest: dict[str, Any], maximum_dates_per_split: int | None
) -> list[tuple[dict[str, Any], list[int]]]:
    if maximum_dates_per_split is not None and maximum_dates_per_split <= 0:
        raise ValueError("--max-a-dates must be positive")
    used = {split: 0 for split in manifest["segments"]}
    selected = []
    for part in manifest["parts"]:
        remaining = (
            len(part["dates"])
            if maximum_dates_per_split is None
            else max(maximum_dates_per_split - used[part["split"]], 0)
        )
        indices = list(range(min(len(part["dates"]), remaining)))
        if indices:
            selected.append((part, indices))
            used[part["split"]] += len(indices)
    if not any(part["split"] == "train" for part, _ in selected):
        raise RuntimeError("No A-share train dates selected; US normalizer cannot be fit")
    return selected


def _build_us_part(
    part: dict[str, Any],
    day_indices: list[int],
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    symbols: list[str],
    us_ids: dict[str, int],
    us_calendar: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    a_dates = pd.DatetimeIndex([part["dates"][index] for index in day_indices]).normalize()
    alignment = strict_us_asof_alignment(a_dates, us_calendar)
    asof_dates = alignment["us_asof_dates"]
    rows: list[np.ndarray] = []
    ids: list[int] = []
    offsets = [0]
    missing_asof = insufficient_history = all_nan = 0
    for asof in asof_dates:
        if np.isnat(asof):
            missing_asof += len(symbols)
            offsets.append(len(ids))
            continue
        for symbol in symbols:
            dates, values = histories[symbol]
            position = int(np.searchsorted(dates, asof, side="left"))
            if position >= len(dates) or dates[position] != asof:
                missing_asof += 1
                continue
            if position < LOOKBACK - 1:
                insufficient_history += 1
                continue
            feature = construct_us_alpha360(values, np.asarray([position]))[0]
            if not np.isfinite(feature).any():
                all_nan += 1
                continue
            rows.append(feature)
            ids.append(us_ids[symbol])
        offsets.append(len(ids))
    features = np.asarray(rows, dtype="float32").reshape(-1, FEATURE_WIDTH)
    arrays = {
        **alignment,
        "us_features": features,
        "us_stock_ids": np.asarray(ids, dtype="int32"),
        "us_offsets": np.asarray(offsets, dtype="int64"),
    }
    observed = ~np.isnat(arrays["us_close_times_utc"])
    if np.any(
        observed
        & ~(arrays["us_close_times_utc"] < arrays["a_signal_times_utc"])
    ):
        raise AssertionError("US as-of leakage detected while writing a part")
    sizes = np.diff(arrays["us_offsets"])
    stats = {
        "a_dates": len(a_dates),
        "us_rows": len(features),
        "min_us_stocks": int(sizes.min()) if len(sizes) else 0,
        "max_us_stocks": int(sizes.max()) if len(sizes) else 0,
        "missing_symbol_asof_rows": missing_asof,
        "insufficient_60_session_history_rows": insufficient_history,
        "all_nan_feature_rows": all_nan,
        "finite_feature_fraction": float(np.isfinite(features).mean()) if features.size else 0.0,
    }
    return arrays, stats


def _verify_output_part(output: Path, record: dict[str, Any]) -> None:
    for name, expected in record["sha256"].items():
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Partial output changed; refusing mixed resume: {name}")


def _write_part(output: Path, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    hashes = {}
    for suffix, values in arrays.items():
        path = output / f"{prefix}_{suffix}.npy"
        atomic_npy(path, values)
        hashes[path.name] = sha256_file(path)
    return hashes


def _fit_us_normalizer(output: Path, parts: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moments = RunningMoments.empty()
    for part in parts:
        if part["split"] != "train":
            continue
        features = np.load(output / f"{part['prefix']}_us_features.npy", mmap_mode="r", allow_pickle=False)
        # Keep memory bounded even for large S&P 500 x date chunks.
        for first in range(0, len(features), 8192):
            moments.update(np.asarray(features[first:first + 8192]))
    if not moments.count.any():
        raise RuntimeError("No finite US train features available for normalization")
    return moments.finish()


def build_store(args: argparse.Namespace) -> dict[str, Any]:
    a_store = Path(args.a_store).expanduser().resolve()
    raw_dir = Path(args.us_raw).expanduser().resolve()
    universe_path = Path(args.universe).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == raw_dir or raw_dir in output.parents:
        raise ValueError("Output must not be inside the read-only US source directory")
    if output == a_store or a_store in output.parents:
        raise ValueError("Output must not be inside the read-only E0 source directory")
    protected = Path("/Users/hmax/investment_data").resolve()
    if output == protected or protected in output.parents:
        raise ValueError("Refusing to write anywhere under /Users/hmax/investment_data")

    a_manifest, a_identity = validate_a_store(a_store)
    symbols = universe_symbols(universe_path, raw_dir, args.max_us_stocks)
    universe_source = source_record(universe_path)
    us_sources = [source_record(raw_dir / f"{symbol}.parquet") for symbol in symbols]
    selected = _selected_parts(a_manifest, args.max_a_dates)
    configuration = {
        "schema_version": 2,
        "a_source_identity": a_identity,
        "a_selected_dates": [
            {"prefix": part["prefix"], "indices": indices} for part, indices in selected
        ],
        "universe_source": universe_source,
        "us_sources": us_sources,
        "symbols": symbols,
        "max_us_stocks": args.max_us_stocks,
        "max_a_dates_per_split": args.max_a_dates,
        "asof_rule": (
            "max(US 16:00 America/New_York close timestamp < "
            "A 15:00 Asia/Shanghai signal timestamp), compared in UTC"
        ),
        "adjustment": "OHLC *= adjclose/close; volume /= adjclose/close; VWAP = adjusted OHLC4 proxy",
    }
    fingerprint = canonical_hash(configuration)
    state_path = output / "build_state.json"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("input_fingerprint") != fingerprint:
            raise RuntimeError("Completed output belongs to different inputs; use a new --output")
        for record in existing["parts"]:
            _verify_output_part(output, record)
        return existing

    if output.exists() and any(output.iterdir()) and not state_path.exists():
        raise RuntimeError("Non-empty output has no build_state.json; refusing to mix files")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("input_fingerprint") != fingerprint:
            raise RuntimeError("Input hash/mtime/config changed; resume into a new --output")
        completed = {record["prefix"]: record for record in state.get("parts", [])}
        for record in completed.values():
            _verify_output_part(output, record)
    else:
        state = {
            "schema_version": 2,
            "status": "building",
            "input_fingerprint": fingerprint,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parts": [],
        }
        completed = {}
        atomic_json(state_path, state)

    histories, us_calendar, source_coverage = load_us_histories(raw_dir, symbols)
    us_ids = {symbol: index + 1 for index, symbol in enumerate(symbols)}
    records: list[dict[str, Any]] = []
    for part, day_indices in selected:
        prefix = part["prefix"]
        if prefix in completed:
            records.append(completed[prefix])
            continue
        arrays, coverage = _build_us_part(
            part, day_indices, histories, symbols, us_ids, us_calendar
        )
        hashes = _write_part(output, prefix, arrays)
        a_paths = _part_paths(a_store, part)
        record = {
            "prefix": prefix,
            "split": part["split"],
            "dates": [part["dates"][index] for index in day_indices],
            "a_day_indices": day_indices,
            "a_reference": {
                name: a_identity["parts"][prefix][name]
                for name in a_paths
            },
            "coverage": coverage,
            "sha256": hashes,
        }
        records.append(record)
        state["parts"] = records
        state["last_completed_prefix"] = prefix
        atomic_json(state_path, state)

    mean, std, count = _fit_us_normalizer(output, records)
    us_normalizer_path = output / "us_normalizer.npz"
    temporary = us_normalizer_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, mean=mean, std=std, count=count)
    os.replace(temporary, us_normalizer_path)
    a_normalizer_path = output / "a_normalizer.npz"
    shutil.copyfile(a_store / "normalizer.npz", a_normalizer_path)
    dictionaries = {
        "a": json.loads((a_store / "stock_ids.json").read_text(encoding="utf-8")),
        "us": us_ids,
    }
    atomic_json(output / "stock_dictionaries.json", dictionaries)
    aggregate = {
        "a_dates": sum(record["coverage"]["a_dates"] for record in records),
        "us_rows": sum(record["coverage"]["us_rows"] for record in records),
        "missing_symbol_asof_rows": sum(
            record["coverage"]["missing_symbol_asof_rows"] for record in records
        ),
        "insufficient_60_session_history_rows": sum(
            record["coverage"]["insufficient_60_session_history_rows"] for record in records
        ),
        "all_nan_feature_rows": sum(
            record["coverage"]["all_nan_feature_rows"] for record in records
        ),
        **source_coverage,
    }
    # A long build must not silently combine pre- and post-update source bytes.
    _, final_a_identity = validate_a_store(a_store)
    final_universe_source = source_record(universe_path)
    final_us_sources = [source_record(raw_dir / f"{symbol}.parquet") for symbol in symbols]
    if (
        final_a_identity != a_identity
        or final_universe_source != universe_source
        or final_us_sources != us_sources
    ):
        raise RuntimeError(
            "An A/US/universe source changed during construction; output remains resumable "
            "but is not finalized"
        )
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "input_fingerprint": fingerprint,
        "created_at": state["created_at"],
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": a_manifest["segments"],
        "feature_layout": {
            "width": FEATURE_WIDTH,
            "order": "field-major",
            "fields": list(FIELDS),
            "names": alpha360_feature_names(),
            "lookback_us_sessions": LOOKBACK,
        },
        "alignment": {
            "rule": configuration["asof_rule"],
            "a_signal_local_time": A_SIGNAL_TIME,
            "a_signal_timezone": A_SIGNAL_TIMEZONE,
            "us_close_local_time": US_CLOSE_TIME,
            "us_close_timezone": US_CLOSE_TIMEZONE,
            "comparison_timezone": "UTC",
            "same_calendar_day_us_close_allowed": False,
            "strictly_prior_close_required": True,
            "per_a_date_files": (
                "*_a_dates.npy, *_a_signal_times_utc.npy, *_us_asof_dates.npy, "
                "and *_us_close_times_utc.npy"
            ),
        },
        "a_source": {
            **a_identity,
            "reuse_mode": "A arrays remain in E0 and are referenced by absolute path and verified SHA256",
            "normalizer_output": {
                "path": str(a_normalizer_path),
                "sha256": sha256_file(a_normalizer_path),
                "fit_split": "E0 train (reused without refit)",
            },
        },
        "us_source": {
            "raw_directory": str(raw_dir),
            "universe": universe_source,
            "selected_stock_count": len(symbols),
            "files": us_sources,
            "survivorship_bias": (
                "Current S&P 500 constituent list; historical samples exclude former constituents "
                "and therefore contain current-constituent survivorship bias."
            ),
            "adjustment_and_vwap": configuration["adjustment"],
        },
        "normalizers": {
            "a": "a_normalizer.npz",
            "us": {
                "path": "us_normalizer.npz",
                "sha256": sha256_file(us_normalizer_path),
                "fit_split": "train only",
                "fit_population": "US Alpha360 rows aligned to selected A-share train signal dates",
            },
            "independent_markets": True,
        },
        "stock_dictionaries": {
            "path": "stock_dictionaries.json",
            "sha256": sha256_file(output / "stock_dictionaries.json"),
            "a_count": len(dictionaries["a"]),
            "us_count": len(dictionaries["us"]),
        },
        "parts": records,
        "coverage": aggregate,
        "limits": {
            "max_us_stocks": args.max_us_stocks,
            "max_a_dates_per_split": args.max_a_dates,
        },
        "resume_policy": "all input content hashes, source mtimes and configuration must match",
    }
    atomic_json(manifest_path, manifest)
    state["status"] = "complete"
    state["manifest_sha256"] = sha256_file(manifest_path)
    atomic_json(state_path, state)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a-store", type=Path, required=True,
        help="Existing complete E0 DateStore; its arrays are validated and referenced, not copied.",
    )
    parser.add_argument("--us-raw", type=Path, default=DEFAULT_US_RAW)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-us-stocks", type=int, default=None,
        help="Use the first N sorted universe symbols for a bounded smoke build.",
    )
    parser.add_argument(
        "--max-a-dates", type=int, default=None,
        help="Use at most N A-share dates per split for a bounded smoke build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_store(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "input_fingerprint": manifest["input_fingerprint"],
                "coverage": manifest["coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
