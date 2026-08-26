#!/usr/bin/env python3
"""One-shot, bounded finalizer for the user-authorized Alpha360 training run.

Waits without occupying GPU, then audits the completed checkpoint, verifies
signal-only inference, and packages portable results. Never restarts training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from train_alpha360_cross_stock import file_hash, write_json


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def finalize(root, timeout_hours):
    begun = time.monotonic()
    status_path = root / "finalization_status.json"
    previous = read_json(status_path)
    if previous and previous.get("status") == "completed":
        bundle = Path(previous["bundle"])
        if bundle.is_file() and file_hash(bundle) == previous["bundle_sha256"]:
            print("Already finalized; verified existing bundle, nothing restarted", flush=True)
            return
        raise RuntimeError("Completed marker exists but the result bundle is missing or changed")
    write_json(status_path, {"status": "waiting_for_training", "pid": os.getpid(),
                             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "poll_seconds": 30})
    while True:
        state = read_json(root / "run/status.json") or {}
        failure = read_json(root / "data/last_failure.json")
        if failure:
            raise RuntimeError(f"Training/export reported failure: {failure}")
        if state.get("status") == "completed":
            break
        if state.get("status") in ("failed", "paused"):
            raise RuntimeError(f"Training did not complete: {state}")
        if time.monotonic() - begun > timeout_hours * 3600:
            raise TimeoutError("Timed out waiting for training; no automatic restart attempted")
        time.sleep(30)

    audit_dir = root / "audit"
    if audit_dir.exists():
        audit_dir = root / ("audit_retry_" + time.strftime("%Y%m%d_%H%M%S"))
    write_json(status_path, {"status": "auditing", "pid": os.getpid(), "audit_dir": str(audit_dir)})
    subprocess.run([
        sys.executable, "-u", str(root / "script/audit_alpha360_cross_stock.py"),
        "--run", str(root / "run"), "--data", str(root / "data"),
        "--output", str(audit_dir), "--device", "cuda",
    ], check=True)
    audit = read_json(audit_dir / "audit.json")
    if not audit or audit.get("status") != "passed":
        raise AssertionError("Audit did not pass")
    configuration = read_json(root / "run/configuration.json")
    signal_date = configuration["segments"]["test"][1]
    prediction_dir = root / ("inference_check_" + time.strftime("%Y%m%d_%H%M%S"))
    write_json(status_path, {"status": "checking_signal_only_inference", "pid": os.getpid()})
    subprocess.run([
        sys.executable, "-u", str(root / "script/predict_alpha360_cross_stock.py"),
        "--run", str(root / "run"), "--provider", str(root / "provider"),
        "--date", signal_date, "--device", "cuda", "--output", str(prediction_dir),
    ], check=True)
    import numpy as np
    import pandas as pd

    test = pd.read_csv(root / "run/test_predictions.csv")
    test = test.loc[test["datetime"] == signal_date].set_index("instrument")
    replay = pd.read_csv(prediction_dir / "ranking_all.csv").set_index("instrument")
    if set(test.index) != set(replay.index):
        raise AssertionError("Signal-only inference universe differs from saved test predictions")
    for horizon in ("open1_close2", "close1_open2", "open1_open2", "close1_close2"):
        column = horizon + "_expected_return"
        np.testing.assert_allclose(replay.loc[test.index, column], test[column], rtol=0.01, atol=2e-6)
    write_json(audit_dir / "signal_only_inference_check.json", {
        "status": "passed", "signal_date": signal_date, "stocks": len(test),
        "uses_future_prices": False, "comparison": "independent raw-price inference vs saved test predictions",
    })
    bundle = root / "alpha360_completed_results.zip"
    temporary = bundle.with_suffix(".zip.tmp")
    write_json(status_path, {"status": "packaging", "pid": os.getpid()})
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in (root / "run", audit_dir, prediction_dir):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    relative = path.relative_to(root)
                    if directory == audit_dir:
                        relative = Path("audit") / path.relative_to(directory)
                    archive.write(path, str(relative))
        for path in (root / "data/manifest.json", root / "data/stock_ids.json", root / "data/normalizer.npz",
                     root / "benchmark/benchmark.json", root / "train.log"):
            archive.write(path, str(path.relative_to(root)))
        for directory in (root / "script", root / "roll"):
            for path in directory.glob("*.py"):
                archive.write(path, str(path.relative_to(root)))
    temporary.replace(bundle)
    write_json(status_path, {"status": "completed", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "best_epoch": state["best_epoch"], "audit_dir": str(audit_dir),
                             "bundle": str(bundle), "bundle_sha256": file_hash(bundle),
                             "bundle_bytes": bundle.stat().st_size,
                             "elapsed_seconds_including_wait": time.monotonic() - begun,
                             "scope": "training, statistical evaluation, checkpoint audit, independent inference; not trading backtest"})
    print("FINALIZATION COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=48)
    args = parser.parse_args()
    stream = (args.root / "finalize.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        finalize(args.root, args.timeout_hours)
    except Exception as error:
        write_json(args.root / "finalization_status.json", {"status": "failed", "error": repr(error),
                                                             "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
