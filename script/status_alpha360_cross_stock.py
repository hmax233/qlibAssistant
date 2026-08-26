#!/usr/bin/env python3
"""Read-only compact status for a standalone Alpha360 GPU run."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def inspect(root):
    result = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    state = read_json(root / "run/status.json")
    result["state"] = state or read_json(root / "benchmark/status.json") or read_json(root / "data/export_status.json")
    manifest = read_json(root / "data/manifest.json")
    if manifest:
        result["data"] = {"stocks_historical": manifest["stock_count"], "export_seconds": manifest["export_seconds"],
                          "excluded_train_dates": len(manifest["excluded_train_dates"]), "segments": {}}
        for split in ("train", "valid", "selection_valid", "test"):
            parts = [part for part in manifest["parts"] if part["split"] == split]
            result["data"]["segments"][split] = {
                "days": sum(len(part["dates"]) for part in parts),
                "rows": sum(part["rows"] for part in parts),
                "usable_labels": sum(part["usable_labels"] for part in parts),
            }
    path = root / "run/epoch_metrics.csv"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if history:
            fields = ("epoch", "train_nll", "nll_scaled_3leg", "epoch_seconds", "best_valid_nll",
                      "close1_close2_rank_ic", "close1_close2_rank_icir", "close1_close2_brier", "close1_close2_coverage95")
            result["last_epochs"] = [{key: row[key] for key in fields} for row in history[-2:]]
            duration = statistics.median(float(row["epoch_seconds"]) for row in history[-3:])
            config = read_json(root / "run/configuration.json") or {}
            result["timing"] = {"recent_epoch_seconds": duration,
                                "maximum_remaining_hours": max(0, config.get("epochs", 50) - int(history[-1]["epoch"])) * duration / 3600,
                                "note": "max-epoch projection; early stopping may finish sooner"}
    failure = read_json(root / "data/last_failure.json")
    if failure:
        result["failure"] = failure
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        result["gpu"] = gpu.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["gpu"] = "unavailable"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), ensure_ascii=False))
