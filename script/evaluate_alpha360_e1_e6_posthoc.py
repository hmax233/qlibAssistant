#!/usr/bin/env python3
"""Post-Test diagnostic inference for every E1 or E6 probabilistic head.

This command is intentionally separate from the locked model-selection path.
Its outputs are diagnostic-only and must never be used as a new unbiased Test
selection result.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roll.alpha360_cross_stock import HORIZON_NAMES
from script.train_alpha360_cross_stock import file_hash, write_json
from script.train_alpha360_decoupled import (
    Trainer as E1Trainer,
    merge_prediction_columns as merge_e1_predictions,
)
from script.train_alpha360_cross_market import (
    Trainer as E6Trainer,
    merge_prediction_columns as merge_e6_predictions,
)


def arguments_from_configuration(
    family: str, data: Path, run: Path, configuration: dict, device: str, threads: int
) -> Namespace:
    common = dict(
        command="evaluate-test",
        data=data,
        output=run,
        device=device,
        threads=threads,
        seed=int(configuration["seed"]),
        learning_rate=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
        warmup_epochs=int(configuration["warmup_epochs"]),
        warmup_start_factor=float(configuration["warmup_start_factor"]),
        date_batch_size=int(configuration["date_batch_size"]),
        target_scale=float(configuration["target_scale"]),
        epochs=int(configuration["epochs"]),
        resume=False,
        benchmark_only=False,
        benchmark_days=16,
        selection_manifest=None,
        candidate_name=None,
        log_file=None,
    )
    if family == "E1":
        common.update(
            model_mode="shared_four_head",
            horizon=None,
            min_learning_rate=float(configuration["minimum_learning_rate"]),
        )
    else:
        common.update(
            bf16=True,
            minimum_learning_rate=float(configuration["minimum_learning_rate"]),
        )
    return Namespace(**common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("E1", "E6"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    data = args.data.expanduser().resolve()
    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    configuration_path = run / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8-sig"))
    trainer_args = arguments_from_configuration(
        args.family, data, run, configuration, args.device, args.threads
    )
    trainer = (
        E1Trainer(trainer_args)
        if args.family == "E1"
        else E6Trainer(trainer_args, command="evaluate-test")
    )
    trainer.store.verify_parts("test" if args.family == "E1" else ("test",))

    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    checkpoint_audit: dict[str, dict] = {}
    for horizon in HORIZON_NAMES:
        checkpoint_path = run / f"best_{horizon}_rank_ic_model.pt"
        checkpoint = trainer.torch.load(
            checkpoint_path, map_location=trainer.device, weights_only=False
        )
        if checkpoint.get("selection_metric") != f"{horizon}_rank_ic":
            raise RuntimeError(f"Checkpoint metric mismatch: {horizon}")
        trainer.model.load_state_dict(checkpoint["model"])
        metrics, daily, predictions = trainer.evaluate_current(
            "test", collect=True, only_horizon=horizon
        )
        daily.to_csv(output / f"test_{horizon}_daily_metrics.csv", index=False)
        frames.append(predictions)
        summaries.append(
            {"family": args.family, "horizon": horizon,
             "checkpoint_epoch": checkpoint["epoch"], **metrics}
        )
        checkpoint_audit[horizon] = {
            "path": str(checkpoint_path),
            "sha256": file_hash(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": float(checkpoint["selection_value"]),
        }

    merge = merge_e1_predictions if args.family == "E1" else merge_e6_predictions
    prediction_path = output / "test_predictions.csv"
    summary_path = output / "test_summary.csv"
    merge(frames).to_csv(prediction_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    write_json(output / "posthoc_audit.json", {
        "schema_version": 1,
        "status": "complete",
        "warning": (
            "Post-Test diagnostic only. Test has already been opened; these results must not "
            "be used as an unbiased model-selection result."
        ),
        "family": args.family,
        "configuration": {"path": str(configuration_path), "sha256": file_hash(configuration_path)},
        "data_manifest": {"path": str(data / "manifest.json"), "sha256": file_hash(data / "manifest.json")},
        "checkpoints": checkpoint_audit,
        "test_predictions_sha256": file_hash(prediction_path),
        "test_summary_sha256": file_hash(summary_path),
        "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"POSTHOC {args.family} FOUR-HEAD TEST COMPLETE: {output}")


if __name__ == "__main__":
    main()
