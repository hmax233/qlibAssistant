#!/usr/bin/env python3
"""Export Alpha360, train a cross-stock probabilistic Transformer, or benchmark.

Example: python script/train_alpha360_cross_stock.py pipeline --data DATA --output RUN
The default is Fixed Fold3/120m, raw log-return targets, and trainable stock IDs.
This entry point never executes real trades or changes existing recorders.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEGMENTS = {
    "train": ["2015-04-17", "2025-04-16"],
    "valid": ["2025-04-17", "2025-09-16"],
    "selection_valid": ["2025-09-17", "2026-02-16"],
    "test": ["2026-02-17", "2026-07-17"],
}
HORIZON_NAMES_FOR_CLI = (
    "open1_close2", "close1_open2", "open1_open2", "close1_close2",
)


def group_date_batches(iterator, batch_size):
    """Group variable-size date cross-sections without mixing their stocks."""
    group = []
    for batch in iterator:
        group.append(batch)
        if len(group) == batch_size:
            yield group
            group = []
    if group:
        yield group


def pad_date_batches(batches):
    """Pad date cross-sections to [dates, max_stocks, ...] and return a mask."""
    import numpy as np

    if not batches:
        raise ValueError("at least one date batch is required")
    max_stocks = max(len(batch["stock_ids"]) for batch in batches)
    feature_count = batches[0]["features"].shape[1]
    features = np.zeros((len(batches), max_stocks, feature_count), dtype="float32")
    labels = np.full((len(batches), max_stocks, 3), np.nan, dtype="float32")
    stock_ids = np.zeros((len(batches), max_stocks), dtype="int64")
    stock_mask = np.zeros((len(batches), max_stocks), dtype=bool)
    for row, batch in enumerate(batches):
        stocks = len(batch["stock_ids"])
        features[row, :stocks] = batch["features"]
        labels[row, :stocks] = batch["labels"]
        stock_ids[row, :stocks] = batch["stock_ids"]
        stock_mask[row, :stocks] = True
    return features, stock_ids, labels, stock_mask

# Explicit provenance for the Windows status-file sharing repair. This is not a
# general permission to resume after arbitrary training-code changes.
PRE_IO_REPAIR_SCRIPT_SHA256 = "a61606f4d5375914ebaf64e649cdcf65f6f84f634ea91b1eb7d67dd91f7ce7b2"


def replace_with_retry(source: Path, destination: Path, attempts: int = 20) -> None:
    """A short-lived Windows reader can deny an otherwise valid atomic rename."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.25))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    replace_with_retry(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def future_log_legs(open1, close1, open2, close2):
    import numpy as np

    prices = np.column_stack([open1, close1, open2, close2]).astype("float64")
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    result = np.full((len(prices), 3), np.nan, dtype="float32")
    result[valid] = np.column_stack([
        np.log(prices[valid, 1] / prices[valid, 0]),
        np.log(prices[valid, 2] / prices[valid, 1]),
        np.log(prices[valid, 3] / prices[valid, 2]),
    ])
    return result


def purged_signal_dates(calendar, segments):
    """Drop labels from each non-test segment's final two trading dates."""
    import pandas as pd

    calendar = pd.DatetimeIndex(calendar)
    return {
        split: [str(date.date()) for date in calendar[(calendar >= bounds[0]) & (calendar <= bounds[1])][-2:]]
        for split, bounds in segments.items() if split != "test"
    }


def construct_alpha360(values, positions):
    """Vectorized equivalent of Alpha360DL; values are [date, C/O/H/L/VWAP/VOL]."""
    import numpy as np

    values = np.asarray(values, dtype="float32")
    positions = np.asarray(positions, dtype="int64")
    if len(positions) and positions.min() < 59:
        raise ValueError("Alpha360 requires 59 earlier calendar observations")
    sequence = values[positions[:, None] - np.arange(59, -1, -1)[None, :]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sequence[:, :, :5] /= values[positions, 0][:, None, None]
        sequence[:, :, 5] /= (values[positions, 5] + 1e-12)[:, None]
    result = sequence.transpose(0, 2, 1).reshape(len(positions), 360)
    result[~np.isfinite(result)] = np.nan
    return result


def load_alpha360_chunk(data_api, calendar, chunk_dates, pool, names):
    """Read six base fields once, preserving point-in-time membership/history."""
    import numpy as np
    import pandas as pd

    members = data_api.list_instruments(
        data_api.instruments(pool), start_time=chunk_dates[0], end_time=chunk_dates[-1], as_list=False
    )
    first = calendar.get_loc(chunk_dates[0])
    last = calendar.get_loc(chunk_dates[-1])
    if first < 59 or last + 2 >= len(calendar):
        raise RuntimeError("Insufficient warmup or future-label calendar data")
    context_dates = calendar[first - 59:last + 3]
    expressions = ["$close", "$open", "$high", "$low", "$vwap", "$volume"]
    raw = data_api.features(list(members), expressions, start_time=context_dates[0], end_time=context_dates[-1])
    frames = []
    for code, intervals in members.items():
        active = np.zeros(len(chunk_dates), dtype=bool)
        for begin, end in intervals:
            active |= (chunk_dates >= begin) & (chunk_dates <= end)
        selected_dates = chunk_dates[active]
        if not len(selected_dates):
            continue
        stock = raw.xs(code, level="instrument").reindex(context_dates)
        values = stock.to_numpy(dtype="float32")
        positions = context_dates.get_indexer(selected_dates)
        features = construct_alpha360(values, positions)
        # Future prices only enter labels; construct_alpha360 never reads them.
        future = np.column_stack([values[positions + 1, 1], values[positions + 1, 0],
                                  values[positions + 2, 1], values[positions + 2, 0]])
        index = pd.MultiIndex.from_arrays([selected_dates, [code] * len(selected_dates)], names=["datetime", "instrument"])
        frames.append(pd.DataFrame(np.column_stack([features, future]), index=index, columns=names + ["O1", "C1", "O2", "C2"]))
    if not frames:
        raise RuntimeError("No historical constituent samples in this chunk")
    return pd.concat(frames).sort_index()


def export_data(args) -> None:
    import gc
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha360DL

    args.data.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("A completed dataset already exists; use train, not export")
    if list(args.data.glob("part_*")):
        raise FileExistsError("Partial export exists; use a new data directory to avoid mixing datasets")
    provider = Path(args.provider).expanduser()
    qlib.init(provider_uri=str(provider), region="cn", kernels=args.threads)
    segments = json.loads(Path(args.segments_json).read_text()) if args.segments_json else SEGMENTS
    calendar = pd.DatetimeIndex(D.calendar(future=False))
    if len(calendar[calendar > segments["test"][1]]) < 2:
        raise RuntimeError("Provider must contain at least two trading dates after the test signal end")
    purged = purged_signal_dates(calendar, segments)
    fields, names = Alpha360DL.get_feature_config()
    future_fields = ["Ref($open,-1)", "Ref($close,-1)", "Ref($open,-2)", "Ref($close,-2)"]
    stocks, parts, excluded_dates = {}, [], []
    count = np.zeros(360, dtype=np.int64)
    total = np.zeros(360, dtype=np.float64)
    squared = np.zeros(360, dtype=np.float64)
    started = time.monotonic()
    for split, bounds in segments.items():
        dates = calendar[(calendar >= bounds[0]) & (calendar <= bounds[1])]
        for first in range(0, len(dates), args.export_days):
            chunk_dates = dates[first:first + args.export_days]
            number = len(parts)
            label = f"part_{number:03d}_{split}"
            print(f"EXPORT {label} {chunk_dates[0].date()} -> {chunk_dates[-1].date()}", flush=True)
            frame = load_alpha360_chunk(D, calendar, chunk_dates, args.pool, names)
            if frame.empty:
                raise RuntimeError(f"Empty historical universe: {split}, {chunk_dates[0]}")
            frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()
            daily_sizes = frame.groupby(level="datetime").size()
            incomplete = daily_sizes[daily_sizes < args.min_stocks_per_date]
            if len(incomplete):
                if split != "train":
                    raise RuntimeError(f"Incomplete {split} cross-section: {incomplete.to_dict()}")
                excluded_dates.extend({"date": str(date.date()), "stocks": int(size),
                                       "reason": "historical constituent list below minimum"}
                                      for date, size in incomplete.items())
                frame = frame.loc[~frame.index.get_level_values("datetime").isin(incomplete.index)]
                print(f"EXCLUDED {len(incomplete)} incomplete train dates (<{args.min_stocks_per_date} stocks)", flush=True)
                if frame.empty:
                    continue
            if number == 0:
                # Verify against installed Qlib, not just our own reshaping test.
                code = frame.index.get_level_values("instrument")[0]
                reference = D.features([code], fields + future_fields, start_time=chunk_dates[0], end_time=chunk_dates[-1])
                reference = reference.reorder_levels(["datetime", "instrument"]).sort_index()
                actual = frame.loc[frame.index.intersection(reference.index)]
                expected = reference.loc[actual.index].to_numpy(dtype="float32")
                expected[~np.isfinite(expected)] = np.nan
                np.testing.assert_allclose(actual.to_numpy(), expected, rtol=2e-5, atol=2e-5, equal_nan=True)
                print(f"ALPHA360_EQUIVALENCE_OK {code} {len(actual)} stock-days x 360 features", flush=True)
            features = frame.iloc[:, :360].to_numpy(dtype="float32", copy=True)
            features[~np.isfinite(features)] = np.nan
            legs = future_log_legs(*[frame.iloc[:, 360 + offset].to_numpy() for offset in range(4)])
            row_dates = frame.index.get_level_values("datetime")
            if split in purged:
                legs[row_dates.isin(pd.to_datetime(purged[split]))] = np.nan
            codes = frame.index.get_level_values("instrument")
            for code in codes.unique():
                if code not in stocks:
                    stocks[code] = len(stocks) + 1
            stock_ids = np.asarray([stocks[code] for code in codes], dtype="int32")
            unique_dates, rows_per_date = np.unique(row_dates.values, return_counts=True)
            offsets = np.concatenate([[0], np.cumsum(rows_per_date)]).astype("int64")
            if split == "train":
                finite = np.isfinite(features)
                safe = np.where(finite, features, 0.0).astype("float64")
                count += finite.sum(axis=0)
                total += safe.sum(axis=0)
                squared += np.square(safe).sum(axis=0)
                del finite, safe
            arrays = {"features": features, "labels": legs, "stock_ids": stock_ids, "offsets": offsets}
            hashes = {}
            for name, values in arrays.items():
                path = args.data / f"{label}_{name}.npy"
                np.save(path, values, allow_pickle=False)
                hashes[path.name] = file_hash(path)
            parts.append({
                "prefix": label, "split": split, "rows": len(frame),
                "dates": [str(pd.Timestamp(date).date()) for date in unique_dates],
                "usable_labels": int(np.isfinite(legs).all(axis=1).sum()),
                "min_stocks": int(rows_per_date.min()), "max_stocks": int(rows_per_date.max()),
                "sha256": hashes,
            })
            write_json(args.data / "export_status.json", {
                "status": "exporting", "pid": os.getpid(), "part": label,
                "parts_finished": len(parts), "elapsed_seconds": time.monotonic() - started,
            })
            del frame, features, legs, arrays
            gc.collect()
    mean = total / np.maximum(count, 1)
    variance = np.maximum(squared / np.maximum(count, 1) - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    np.savez(args.data / "normalizer.npz", mean=mean.astype("float32"), std=std.astype("float32"), count=count)
    write_json(args.data / "stock_ids.json", stocks)
    manifest = {
        "schema_version": 1, "pool": args.pool, "segments": segments,
        "feature_names": names, "feature_expressions": fields,
        "label_expressions": future_fields,
        "legs": ["log(close1/open1)", "log(open2/close1)", "log(close2/open2)"],
        "label_normalization": "none; fixed target_scale only at training time",
        "feature_normalization": "per-column ZScore fit only on train; fill missing after normalization",
        "purged_signal_dates": purged, "stock_count": len(stocks), "parts": parts,
        "min_stocks_per_date": args.min_stocks_per_date, "excluded_train_dates": excluded_dates,
        "normalizer_sha256": file_hash(args.data / "normalizer.npz"),
        "stock_ids_sha256": file_hash(args.data / "stock_ids.json"),
        "export_seconds": time.monotonic() - started, "provider": str(provider),
        "qlib_version": qlib.__version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest_path, manifest)
    write_json(args.data / "export_status.json", {"status": "completed", "seconds": manifest["export_seconds"]})
    print(f"EXPORT COMPLETE {len(stocks)} historical stocks, {sum(p['rows'] for p in parts)} rows", flush=True)


class DateStore:
    """Memory-mapped date batches; never mixes stock time histories or future dates."""

    def __init__(self, directory: Path):
        import numpy as np

        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        for name, key in [("normalizer.npz", "normalizer_sha256"), ("stock_ids.json", "stock_ids_sha256")]:
            if file_hash(directory / name) != self.manifest[key]:
                raise RuntimeError(f"Input hash mismatch: {name}")
        normalizer = np.load(directory / "normalizer.npz", allow_pickle=False)
        self.mean, self.std = normalizer["mean"], normalizer["std"]
        self.id_to_code = {value: key for key, value in json.loads((directory / "stock_ids.json").read_text()).items()}

    def verify_parts(self):
        for part in self.manifest["parts"]:
            for name, expected in part["sha256"].items():
                if file_hash(self.directory / name) != expected:
                    raise RuntimeError(f"Input hash mismatch: {name}")

    def days(self, split):
        return sum(len(part["dates"]) for part in self.manifest["parts"] if part["split"] == split)

    def iterate(self, split, seed=None, max_days=None):
        import numpy as np

        parts = [part for part in self.manifest["parts"] if part["split"] == split]
        generator = random.Random(seed)
        if seed is not None:
            generator.shuffle(parts)
        emitted = 0
        for part in parts:
            arrays = {
                key: np.load(self.directory / f"{part['prefix']}_{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in ("features", "labels", "stock_ids", "offsets")
            }
            days = list(range(len(part["dates"])))
            if seed is not None:
                generator.shuffle(days)
            for day in days:
                if max_days is not None and emitted >= max_days:
                    return
                begin, end = arrays["offsets"][day:day + 2]
                values = (np.array(arrays["features"][begin:end]) - self.mean) / self.std
                values[~np.isfinite(values)] = 0.0
                yield {
                    "date": part["dates"][day], "features": values,
                    "labels": np.array(arrays["labels"][begin:end]),
                    "stock_ids": np.array(arrays["stock_ids"][begin:end]),
                }
                emitted += 1


def train(args) -> None:
    # Qlib before torch avoids DLL ordering trouble in the Windows conda env.
    import qlib  # noqa: F401
    import numpy as np
    import pandas as pd
    import torch
    from roll.alpha360_cross_stock import (
        Alpha360CrossStockTransformer, Alpha360TransformerConfig, HORIZON_MATRIX,
        HORIZON_NAMES, distribution_report, joint_gaussian_nll,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume or choose a new run")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    store = DateStore(args.data)
    print("Verifying dataset hashes...", flush=True)
    store.verify_parts()
    config = Alpha360TransformerConfig(stock_embedding_width=args.stock_embedding_width)
    model = Alpha360CrossStockTransformer(store.manifest["stock_count"], config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.warmup_start_factor, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs],
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    configuration = {
        "model": asdict(config), "data_manifest_sha256": file_hash(args.data / "manifest.json"),
        "segments": store.manifest["segments"], "seed": args.seed,
        "epochs": args.epochs, "early_stopping": False,
        "date_batch_size": args.date_batch_size,
        "learning_rate": args.learning_rate, "min_learning_rate": args.min_learning_rate,
        "warmup_epochs": args.warmup_epochs, "warmup_start_factor": args.warmup_start_factor,
        "scheduler": "3-epoch linear warmup followed by cosine annealing",
        "selection_metric": args.selection_metric, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "feature_count": 360, "target_scale": config.target_scale,
        "test_policy": "not used for early stopping; evaluate once after best-valid checkpoint",
        "execution": "open/close labels only; no fill/fee/backtest claims",
        "script_sha256": file_hash(Path(__file__)),
        "model_code_sha256": file_hash(ROOT / "roll" / "alpha360_cross_stock.py"),
        "torch": torch.__version__,
        "autocast_dtype": str(amp_dtype) if device.type == "cuda" else "float32",
    }
    history, first_epoch, best, stale = [], 0, float("inf"), 0
    best_rank_ic = {horizon: -float("inf") for horizon in HORIZON_NAMES}
    previous_epoch_seconds = 0.0
    if args.resume:
        old_config = json.loads((args.output / "configuration.json").read_text())
        for key in ("model", "data_manifest_sha256", "seed", "learning_rate",
                    "min_learning_rate", "warmup_epochs", "warmup_start_factor",
                    "scheduler", "selection_metric", "date_batch_size",
                    "script_sha256", "model_code_sha256"):
            if old_config[key] != configuration[key]:
                if (key == "script_sha256" and args.resume_io_repair
                        and old_config[key] == PRE_IO_REPAIR_SCRIPT_SHA256):
                    continue
                raise RuntimeError(f"Resume configuration mismatch: {key}")
        state = torch.load(args.output / "last_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, best, stale = state["epoch"], state["best"], state["stale"]
        best_rank_ic = state["best_rank_ic"]
        history = state["history"]
        previous_epoch_seconds = sum(row["epoch_seconds"] for row in history)
        configuration["resume_events"] = old_config.get("resume_events", []) + [{
            "from_epoch": first_epoch, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "previous_script_sha256": old_config["script_sha256"],
            "current_script_sha256": configuration["script_sha256"],
            "io_repair": bool(args.resume_io_repair),
            "note": "atomic publication retry; model, data, optimizer and loss unchanged",
        }]
        if args.resume_io_repair:
            archive = args.output / "configuration_before_io_repair.json"
            if archive.exists():
                raise FileExistsError("I/O repair was already recorded; use ordinary --resume")
            write_json(archive, old_config)
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    write_json(args.output / "configuration.json", configuration)
    # Store code->ID vocabulary beside weights for portable prediction.
    write_json(args.output / "stock_ids.json", {code: idx for idx, code in store.id_to_code.items()})
    np.savez(args.output / "normalizer.npz", mean=store.mean, std=store.std)
    started = time.monotonic()

    def tensor_batch(batch):
        features, ids, labels, mask = pad_date_batches([batch])
        return (torch.from_numpy(features).to(device), torch.from_numpy(ids).to(device),
                torch.from_numpy(labels).to(device) * config.target_scale,
                torch.from_numpy(mask).to(device))

    def tensor_date_batch(batches):
        features, ids, labels, mask = pad_date_batches(batches)
        return (torch.from_numpy(features).to(device), torch.from_numpy(ids).to(device),
                torch.from_numpy(labels).to(device) * config.target_scale,
                torch.from_numpy(mask).to(device))

    def evaluate(split, max_days=None, collect=False):
        model.eval()
        losses, day_rows, prediction_rows = [], [], []
        with torch.no_grad():
            for batch in store.iterate(split, max_days=max_days):
                features, ids, labels, mask = tensor_batch(batch)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    output = model(features, ids, mask)
                loss = joint_gaussian_nll(labels, output["leg_mean"], output["leg_cholesky"], mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite evaluation NLL: {split} {batch['date']}")
                if np.isfinite(batch["labels"]).all(axis=1).any():
                    losses.append(float(loss))
                horizon_mean = output["horizon_mean"].float().squeeze(0) / config.target_scale
                horizon_covariance = output["horizon_covariance"].float().squeeze(0) / config.target_scale**2
                report = {key: value.cpu().numpy() for key, value in distribution_report(horizon_mean, horizon_covariance).items()}
                actual_log = batch["labels"] @ HORIZON_MATRIX.numpy().T
                actual = np.expm1(actual_log)
                row = {"datetime": batch["date"], "nll_scaled_3leg": float(loss), "stocks": len(actual)}
                prediction = {"datetime": batch["date"], "instrument": [store.id_to_code[int(idx)] for idx in batch["stock_ids"]]}
                for column, horizon in enumerate(HORIZON_NAMES):
                    expected, realized = report["expected_return"][:, column], actual[:, column]
                    usable = np.isfinite(expected) & np.isfinite(realized)
                    predicted = pd.Series(expected[usable])
                    truth = pd.Series(realized[usable])
                    row[horizon + "_rank_ic"] = predicted.corr(truth, method="spearman") if len(truth) > 1 else float("nan")
                    row[horizon + "_mae"] = float(np.mean(np.abs(expected[usable] - realized[usable]))) if usable.any() else float("nan")
                    probabilities = report["probability_positive"][:, column]
                    row[horizon + "_brier"] = float(np.mean((probabilities[usable] - (realized[usable] > 0))**2)) if usable.any() else float("nan")
                    for coverage, quantile in [(50, 0.67448975), (80, 1.28155157), (95, 1.95996398)]:
                        distance = np.abs(actual_log[:, column] - report["log_mean"][:, column])
                        inside = distance <= quantile * np.sqrt(report["log_variance"][:, column])
                        row[f"{horizon}_coverage{coverage}"] = float(inside[usable].mean()) if usable.any() else float("nan")
                    if collect:
                        for key in report:
                            prediction[f"{horizon}_{key}"] = report[key][:, column]
                        prediction[horizon + "_actual_return"] = realized
                day_rows.append(row)
                if collect:
                    prediction_rows.append(pd.DataFrame(prediction))
        daily = pd.DataFrame(day_rows)
        metrics = {"nll_scaled_3leg": float(np.mean(losses)), "days": len(daily)}
        for horizon in HORIZON_NAMES:
            correlations = daily[horizon + "_rank_ic"].dropna()
            metrics[horizon + "_rank_ic"] = float(correlations.mean())
            std = correlations.std(ddof=0)
            metrics[horizon + "_rank_icir"] = float(correlations.mean() / std) if std > 0 else None
            for metric in ("mae", "brier", "coverage50", "coverage80", "coverage95"):
                metrics[horizon + "_" + metric] = float(daily[horizon + "_" + metric].mean())
        if collect:
            daily.to_csv(args.output / f"{split}_daily_metrics.csv", index=False)
            pd.concat(prediction_rows, ignore_index=True).to_csv(args.output / f"{split}_predictions.csv", index=False)
        return metrics

    def train_epoch(epoch, max_days=None):
        model.train()
        losses, duration, count = [], [], 0
        total_days = min(store.days("train"), max_days) if max_days else store.days("train")
        iterator = store.iterate("train", seed=args.seed + epoch, max_days=max_days)
        for batches in group_date_batches(iterator, args.date_batch_size):
            usable = [batch for batch in batches if np.isfinite(batch["labels"]).all(axis=1).any()]
            if not usable:
                continue
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            features, ids, labels, mask = tensor_date_batch(usable)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                output = model(features, ids, mask)
            loss = joint_gaussian_nll(labels, output["leg_mean"], output["leg_cholesky"], mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training NLL at {[batch['date'] for batch in usable]}")
            # Keep every date equally weighted when the final batch is smaller.
            scaled_loss = loss * (len(usable) / args.date_batch_size)
            scaler.scale(scaled_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count += len(usable)
            losses.append(float(loss.detach()))
            duration.append(time.monotonic() - tick)
            if count % 25 == 0 or count == 1:
                state = {"status": "training", "pid": os.getpid(), "epoch": epoch + 1,
                         "processed_dates": count, "training_dates_total": total_days,
                         "date_batch_size": args.date_batch_size,
                         "recent_seconds_per_date": float(np.sum(duration[-20:]) /
                                                          min(count, 20 * args.date_batch_size)),
                         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
                write_json(args.output / "status.json", state)
                print("PROGRESS " + json.dumps(state), flush=True)
        if not losses:
            raise RuntimeError("No usable training labels")
        return float(np.mean(losses)), duration

    print("CONFIGURATION " + json.dumps(configuration), flush=True)
    if args.benchmark_only:
        loss, durations = train_epoch(0, max_days=args.benchmark_days)
        valid_started = time.monotonic()
        evaluate("valid", max_days=5)
        valid_seconds = (time.monotonic() - valid_started) / min(5, store.days("valid"))
        steady = float(np.median(durations[2:] or durations))
        train_steps = math.ceil(store.days("train") / args.date_batch_size)
        estimate = steady * train_steps + valid_seconds * store.days("valid")
        write_json(args.output / "benchmark.json", {
            "train_loss": loss, "timed_optimizer_steps": len(durations),
            "date_batch_size": args.date_batch_size,
            "steady_seconds_per_optimizer_step": steady, "valid_seconds_per_date": valid_seconds,
            "estimated_epoch_seconds": estimate, "estimated_20_epochs_hours": estimate * 20 / 3600,
            "max_epochs_hours": estimate * args.epochs / 3600,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "warning": "short real-data estimate, not a completion guarantee",
        })
        write_json(args.output / "status.json", {"status": "benchmark_completed", "pid": os.getpid()})
        return

    for epoch in range(first_epoch, args.epochs):
        epoch_started = time.monotonic()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, _ = train_epoch(epoch)
        valid = evaluate("valid")
        improved = valid["nll_scaled_3leg"] < best
        if improved:
            best, stale = valid["nll_scaled_3leg"], 0
            payload = {"model": model.state_dict(), "configuration": configuration,
                       "epoch": epoch + 1, "selection_metric": "nll_scaled_3leg",
                       "selection_value": best}
            for filename in ("best_model.pt", "best_nll_model.pt"):
                temporary = args.output / (filename + ".tmp")
                torch.save(payload, temporary)
                replace_with_retry(temporary, args.output / filename)
        else:
            stale += 1
        for horizon in HORIZON_NAMES:
            key = horizon + "_rank_ic"
            value = valid[key]
            if np.isfinite(value) and value > best_rank_ic[horizon]:
                best_rank_ic[horizon] = value
                filename = f"best_{horizon}_rank_ic_model.pt"
                temporary = args.output / (filename + ".tmp")
                torch.save({"model": model.state_dict(), "configuration": configuration,
                            "epoch": epoch + 1, "selection_metric": key,
                            "selection_value": value}, temporary)
                replace_with_retry(temporary, args.output / filename)
        history.append({"epoch": epoch + 1, "train_nll": train_loss, **valid,
                        "learning_rate": epoch_learning_rate,
                        "epoch_seconds": time.monotonic() - epoch_started, "best_valid_nll": best})
        scheduler.step()
        pd.DataFrame(history).to_csv(args.output / "epoch_metrics.csv", index=False)
        snapshot = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_rank_ic": best_rank_ic, "stale": stale, "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        temporary = args.output / "last_checkpoint.pt.tmp"
        torch.save(snapshot, temporary)
        replace_with_retry(temporary, args.output / "last_checkpoint.pt")
        temporary = args.output / "last_model.pt.tmp"
        torch.save({"model": model.state_dict(), "configuration": configuration,
                    "epoch": epoch + 1, "selection_metric": "last_epoch",
                    "selection_value": epoch + 1}, temporary)
        replace_with_retry(temporary, args.output / "last_model.pt")
        print("EPOCH " + json.dumps(history[-1]), flush=True)
        if (args.output / "STOP_AFTER_EPOCH").exists():
            write_json(args.output / "status.json", {"status": "paused", "epoch": epoch + 1, "resumable": True})
            return
    selected_filename = ("best_nll_model.pt" if args.selection_metric == "nll_scaled_3leg"
                         else f"best_{args.selection_metric}_model.pt")
    best_state = torch.load(args.output / selected_filename, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    summary = []
    for split in ("valid", "selection_valid", "test"):
        write_json(args.output / "status.json", {"status": "final_evaluation", "split": split, "pid": os.getpid()})
        summary.append({"split": split, **evaluate(split, collect=True)})
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    write_json(args.output / "status.json", {
        "status": "completed", "best_epoch": best_state["epoch"],
        "selected_checkpoint": selected_filename,
        "selection_metric": best_state["selection_metric"],
        "selection_value": best_state["selection_value"],
        "elapsed_seconds": previous_epoch_seconds + time.monotonic() - started,
        "elapsed_basis": "completed epochs plus final attempt/evaluation; excludes failed partial epoch and downtime",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backtest": "not performed; predictions/metrics are not tradable PnL",
    })
    print("TRAINING COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "train", "pipeline"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--pool", default="csi1000")
    parser.add_argument("--segments-json", type=Path, help="Optional JSON file; never silently changes fixed boundaries")
    parser.add_argument("--export-days", type=int, default=120)
    parser.add_argument("--min-stocks-per-date", type=int, default=800,
                        help="Reject incomplete eval universes; log and exclude incomplete train dates")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--min-learning-rate", type=float, default=0.000001)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-factor", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--selection-metric",
        choices=["nll_scaled_3leg", *[h + "_rank_ic" for h in HORIZON_NAMES_FOR_CLI]],
        default="close1_close2_rank_ic",
    )
    parser.add_argument("--date-batch-size", type=int, default=4)
    parser.add_argument("--stock-embedding-width", type=int, default=64, help="0 disables identity for ablation")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-days", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-io-repair", action="store_true",
                        help="Explicitly resume the identified pre-repair version after Windows atomic-write repair")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.resume_io_repair and not args.resume:
        parser.error("--resume-io-repair requires --resume")
    if args.mode in ("train", "pipeline") and args.output is None:
        parser.error("train/pipeline require --output")
    if min(args.epochs, args.export_days, args.date_batch_size, args.benchmark_days, args.warmup_epochs) < 1:
        parser.error("counts must be positive")
    if args.warmup_epochs >= args.epochs:
        parser.error("warmup epochs must be smaller than total epochs")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error("min learning rate must be in (0, learning rate]")
    if not 0 < args.warmup_start_factor <= 1:
        parser.error("warmup start factor must be in (0, 1]")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.mode in ("export", "pipeline"):
            if args.mode != "pipeline" or not (args.data / "manifest.json").exists():
                export_data(args)
        if args.mode in ("train", "pipeline"):
            train(args)
    except Exception as error:
        failure_directory = args.output if args.output is not None else args.data
        # Do not overwrite a completed run on an accidental repeated invocation.
        write_json(args.data / "last_failure.json", {
            "error": repr(error), "output": str(failure_directory),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
