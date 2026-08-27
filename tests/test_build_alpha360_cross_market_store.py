from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from build_alpha360_cross_market_store import (  # noqa: E402
    FIELDS,
    adjust_yahoo_ohlcv,
    alpha360_feature_names,
    build_store,
    construct_us_alpha360,
    sha256_file,
    strict_us_asof_alignment,
    strict_us_asof_dates,
    validate_a_store,
)


def _write_npy(path: Path, values: np.ndarray) -> str:
    np.save(path, values, allow_pickle=False)
    return sha256_file(path)


def _make_e0_store(root: Path) -> Path:
    root.mkdir()
    segments = {
        "train": ["2020-04-01", "2020-04-02"],
        "valid": ["2020-04-06", "2020-04-06"],
        "selection_valid": ["2020-04-07", "2020-04-07"],
        "test": ["2020-04-08", "2020-04-08"],
    }
    dates = {
        "train": ["2020-04-01", "2020-04-02"],
        "valid": ["2020-04-06"],
        "selection_valid": ["2020-04-07"],
        "test": ["2020-04-08"],
    }
    parts = []
    for number, (split, split_dates) in enumerate(dates.items()):
        prefix = f"part_{number:03d}_{split}"
        rows = len(split_dates) * 2
        arrays = {
            "features": np.arange(rows * 360, dtype="float32").reshape(rows, 360),
            "labels": np.full((rows, 3), number, dtype="float32"),
            "stock_ids": np.tile(np.asarray([1, 2], dtype="int32"), len(split_dates)),
            "offsets": np.arange(0, rows + 1, 2, dtype="int64"),
        }
        hashes = {
            f"{prefix}_{name}.npy": _write_npy(root / f"{prefix}_{name}.npy", value)
            for name, value in arrays.items()
        }
        parts.append(
            {
                "prefix": prefix,
                "split": split,
                "rows": rows,
                "dates": split_dates,
                "sha256": hashes,
            }
        )
    normalizer = root / "normalizer.npz"
    np.savez(
        normalizer,
        mean=np.zeros(360, dtype="float32"),
        std=np.ones(360, dtype="float32"),
        count=np.ones(360, dtype="int64"),
    )
    stocks = root / "stock_ids.json"
    stocks.write_text(json.dumps({"SH600000": 1, "SZ000001": 2}), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "pool": "synthetic",
        "segments": segments,
        "feature_names": alpha360_feature_names(),
        "stock_count": 2,
        "normalizer_sha256": sha256_file(normalizer),
        "stock_ids_sha256": sha256_file(stocks),
        "parts": parts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _make_us_source(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    dates = pd.bdate_range("2019-12-02", "2020-04-08")
    index = np.arange(len(dates), dtype="float64")
    close = 100.0 + index
    # Make non-train periods observably different so an all-split normalizer
    # would not accidentally equal the train-only normalizer.
    close[dates >= pd.Timestamp("2020-04-03")] *= 4.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.98,
            "high": close * 1.03,
            "low": close * 0.96,
            "close": close,
            "volume": (1_000 + index * index).astype("int64"),
            "adjclose": close * 0.5,
            "symbol": "AAA",
        }
    )
    frame.to_parquet(raw / "AAA.parquet", index=False)
    universe = root / "us_universe.csv"
    pd.DataFrame(
        {"Symbol": ["AAA"], "yahoo_symbol": ["AAA"], "source": ["current_sp500"]}
    ).to_csv(universe, index=False)
    return raw, universe


def _args(a_store: Path, raw: Path, universe: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        a_store=a_store,
        us_raw=raw,
        universe=universe,
        output=output,
        max_us_stocks=1,
        max_a_dates=None,
    )


def test_alpha360_is_field_major_and_uses_adjusted_ohlcv() -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    base = np.arange(1, 61, dtype="float64")
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": base * 2,
            "high": base * 3,
            "low": base,
            "close": base * 2.5,
            "volume": base * 100,
            "adjclose": base * 1.25,
        }
    )
    adjusted = adjust_yahoo_ohlcv(raw)
    # Constant 0.5 factor applies to OHLC; inverse factor applies to volume.
    np.testing.assert_allclose(adjusted["close"], raw["adjclose"])
    np.testing.assert_allclose(adjusted["open"], raw["open"] * 0.5)
    np.testing.assert_allclose(adjusted["volume"], raw["volume"] / 0.5)
    values = adjusted.loc[:, FIELDS].to_numpy(dtype="float32")
    result = construct_us_alpha360(values, np.asarray([59]))
    assert result.shape == (1, 360)
    current_close = adjusted.loc[59, "close"]
    current_volume = adjusted.loc[59, "volume"]
    np.testing.assert_allclose(result[0, 0:60], adjusted["close"] / current_close)
    np.testing.assert_allclose(result[0, 60:120], adjusted["open"] / current_close)
    np.testing.assert_allclose(result[0, 120:180], adjusted["high"] / current_close)
    np.testing.assert_allclose(result[0, 180:240], adjusted["low"] / current_close)
    np.testing.assert_allclose(result[0, 240:300], adjusted["vwap"] / current_close)
    np.testing.assert_allclose(result[0, 300:360], adjusted["volume"] / current_volume)
    assert result[0, 59] == pytest.approx(1.0)
    assert result[0, 359] == pytest.approx(1.0)


def test_strict_asof_handles_weekends_holidays_and_forbids_same_day() -> None:
    us = pd.to_datetime(["2026-07-02", "2026-07-06", "2026-07-07"])
    a = pd.to_datetime(["2026-07-03", "2026-07-04", "2026-07-06", "2026-07-07", "2026-07-08"])
    aligned = strict_us_asof_dates(a, us)
    assert aligned.astype(str).tolist() == [
        "2026-07-02",
        "2026-07-02",
        "2026-07-02",
        "2026-07-06",
        "2026-07-07",
    ]
    assert np.all(aligned < a.values.astype("datetime64[D]"))


def test_asof_compares_actual_exchange_close_and_signal_instants_in_utc() -> None:
    # July exercises US daylight saving time.  The July 6 US close is 20:00
    # UTC and is available at the July 7 China signal (07:00 UTC); the July 7
    # US close is 20:00 UTC and must not be used at that signal.
    alignment = strict_us_asof_alignment(
        pd.to_datetime(["2026-07-07"]),
        pd.to_datetime(["2026-07-06", "2026-07-07"]),
    )
    assert alignment["us_asof_dates"].astype(str).tolist() == ["2026-07-06"]
    assert alignment["a_signal_times_utc"].astype(str).tolist() == [
        "2026-07-07T07:00:00.000000000"
    ]
    assert alignment["us_close_times_utc"].astype(str).tolist() == [
        "2026-07-06T20:00:00.000000000"
    ]
    assert np.all(
        alignment["us_close_times_utc"] < alignment["a_signal_times_utc"]
    )


def test_builder_uses_train_only_normalization_and_pins_all_inputs(tmp_path: Path) -> None:
    a_store = _make_e0_store(tmp_path / "a_store")
    raw, universe = _make_us_source(tmp_path / "us")
    output = tmp_path / "output"
    manifest = build_store(_args(a_store, raw, universe, output))

    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 2
    assert manifest["normalizers"]["independent_markets"] is True
    assert manifest["normalizers"]["us"]["fit_split"] == "train only"
    assert manifest["alignment"]["same_calendar_day_us_close_allowed"] is False
    assert manifest["alignment"]["strictly_prior_close_required"] is True
    assert manifest["alignment"]["comparison_timezone"] == "UTC"
    assert "survivorship bias" in manifest["us_source"]["survivorship_bias"]
    assert manifest["stock_dictionaries"]["a_count"] == 2
    assert manifest["stock_dictionaries"]["us_count"] == 1

    train_arrays = []
    all_arrays = []
    for part in manifest["parts"]:
        features = np.load(output / f"{part['prefix']}_us_features.npy")
        all_arrays.append(features)
        if part["split"] == "train":
            train_arrays.append(features)
        a_dates = np.load(output / f"{part['prefix']}_a_dates.npy")
        a_signal_times = np.load(output / f"{part['prefix']}_a_signal_times_utc.npy")
        us_dates = np.load(output / f"{part['prefix']}_us_asof_dates.npy")
        us_close_times = np.load(output / f"{part['prefix']}_us_close_times_utc.npy")
        assert np.all(us_close_times < a_signal_times)
        assert np.all(us_dates < a_dates)
        for name, expected in part["sha256"].items():
            assert sha256_file(output / name) == expected

    normalizer = np.load(output / "us_normalizer.npz")
    train = np.concatenate(train_arrays)
    expected_train_mean = np.nanmean(train, axis=0)
    np.testing.assert_allclose(normalizer["mean"], expected_train_mean, rtol=2e-6, atol=2e-6)
    all_split_mean = np.nanmean(np.concatenate(all_arrays), axis=0)
    assert np.max(np.abs(normalizer["mean"] - all_split_mean)) > 1e-3

    source = manifest["us_source"]["files"][0]
    assert source["sha256"] == sha256_file(raw / "AAA.parquet")
    assert source["mtime_ns"] == (raw / "AAA.parquet").stat().st_mtime_ns
    assert manifest["a_source"]["manifest_sha256"] == sha256_file(a_store / "manifest.json")
    assert manifest["a_source"]["manifest_source"]["mtime_ns"] == (
        a_store / "manifest.json"
    ).stat().st_mtime_ns
    assert manifest["input_fingerprint"]

    # A byte-identical resume is idempotent.
    first_manifest_hash = sha256_file(output / "manifest.json")
    resumed = build_store(_args(a_store, raw, universe, output))
    assert resumed["input_fingerprint"] == manifest["input_fingerprint"]
    assert sha256_file(output / "manifest.json") == first_manifest_hash

    # Even an mtime-only input mutation is rejected, preventing mixed resumes.
    source_path = raw / "AAA.parquet"
    stat = source_path.stat()
    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="different inputs"):
        build_store(_args(a_store, raw, universe, output))


def test_small_build_limits_apply_per_split(tmp_path: Path) -> None:
    a_store = _make_e0_store(tmp_path / "a_store")
    raw, universe = _make_us_source(tmp_path / "us")
    args = _args(a_store, raw, universe, tmp_path / "limited")
    args.max_a_dates = 1
    manifest = build_store(args)
    counts = {part["split"]: len(part["dates"]) for part in manifest["parts"]}
    assert counts == {"train": 1, "valid": 1, "selection_valid": 1, "test": 1}
    assert manifest["limits"]["max_us_stocks"] == 1
    assert manifest["limits"]["max_a_dates_per_split"] == 1


def test_partial_resume_verifies_existing_chunks_and_input_identity(tmp_path: Path) -> None:
    a_store = _make_e0_store(tmp_path / "a_store")
    raw, universe = _make_us_source(tmp_path / "us")
    output = tmp_path / "resumable"
    args = _args(a_store, raw, universe, output)
    first = build_store(args)
    (output / "manifest.json").unlink()
    resumed = build_store(args)
    assert resumed["input_fingerprint"] == first["input_fingerprint"]

    (output / "manifest.json").unlink()
    damaged = output / f"{resumed['parts'][0]['prefix']}_us_features.npy"
    damaged.write_bytes(b"not a numpy file")
    with pytest.raises(RuntimeError, match="Partial output changed"):
        build_store(args)


def test_builder_refuses_any_output_under_investment_data(tmp_path: Path) -> None:
    args = argparse.Namespace(
        a_store=tmp_path / "not_read",
        us_raw=tmp_path / "not_read_either",
        universe=tmp_path / "not_read.csv",
        output=Path("/Users/hmax/investment_data/forbidden_e6_output"),
        max_us_stocks=1,
        max_a_dates=1,
    )
    with pytest.raises(ValueError, match="Refusing to write"):
        build_store(args)


def test_a_source_split_dates_cannot_be_mislabeled_into_pretest_data(
    tmp_path: Path,
) -> None:
    a_store = _make_e0_store(tmp_path / "a_store")
    manifest_path = a_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_part = next(
        part for part in manifest["parts"] if part["split"] == "selection_valid"
    )
    selection_part["dates"] = ["2020-04-08"]  # Held-out Test date.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside selection_valid"):
        validate_a_store(a_store)
