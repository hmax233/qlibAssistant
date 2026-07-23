#!/usr/bin/env python3
"""为已有 recorder 的 valid_sig_analysis PKL 原地补充可读 CSV。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / ".qlibAssistant" / "mlruns"


def load(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def as_frame(value, name):
    if isinstance(value, pd.Series):
        return value.rename(name).to_frame()
    result = value.copy()
    if len(result.columns) == 1:
        result.columns = [name]
    return result


def export(folder: Path, overwrite: bool) -> bool:
    required = [folder / name for name in ("pred.pkl", "label.pkl", "ic.pkl", "ric.pkl", "metrics.pkl")]
    if not all(path.exists() for path in required):
        return False
    if (folder / "metrics.csv").exists() and not overwrite:
        return False

    as_frame(load(folder / "pred.pkl"), "score").reset_index().to_csv(folder / "pred.csv", index=False)
    as_frame(load(folder / "label.pkl"), "label").reset_index().to_csv(folder / "label.csv", index=False)
    ic = load(folder / "ic.pkl").rename("IC")
    ric = load(folder / "ric.pkl").rename("Rank IC")
    pd.concat([ic, ric], axis=1).rename_axis("datetime").reset_index().to_csv(
        folder / "daily_ic.csv", index=False
    )
    pd.DataFrame([load(folder / "metrics.pkl")]).to_csv(folder / "metrics.csv", index=False)
    segment_path = folder / "segment.pkl"
    if segment_path.exists():
        segment = load(segment_path)
        pd.DataFrame(
            [{"segment": segment.get("name"), "range": json.dumps(segment.get("range"))}]
        ).to_csv(folder / "segment.csv", index=False)
    return True


def export_test(artifacts: Path, overwrite: bool) -> bool:
    sig = artifacts / "sig_analysis"
    required = [
        artifacts / "pred.pkl",
        artifacts / "label.pkl",
        sig / "ic.pkl",
        sig / "ric.pkl",
    ]
    if not all(path.exists() for path in required):
        return False
    if (sig / "metrics.csv").exists() and not overwrite:
        return False

    as_frame(load(artifacts / "pred.pkl"), "score").reset_index().to_csv(
        artifacts / "pred.csv", index=False
    )
    as_frame(load(artifacts / "label.pkl"), "label").reset_index().to_csv(
        artifacts / "label.csv", index=False
    )
    ic = load(sig / "ic.pkl").rename("IC")
    ric = load(sig / "ric.pkl").rename("Rank IC")
    daily = pd.concat([ic, ric], axis=1)
    daily.rename_axis("datetime").reset_index().to_csv(sig / "daily_ic.csv", index=False)
    ic_std, ric_std = float(ic.std()), float(ric.std())
    metrics = {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean()) / ic_std if ic_std else float("nan"),
        "Rank IC": float(ric.mean()),
        "Rank ICIR": float(ric.mean()) / ric_std if ric_std else float("nan"),
        "IC Positive Ratio": float((ic > 0).mean()),
        "Rank IC Positive Ratio": float((ric > 0).mean()),
        "Date Count": int(daily.dropna().shape[0]),
    }
    pd.DataFrame([metrics]).to_csv(sig / "metrics.csv", index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    artifacts_folders = list(MLRUNS.glob("*/*/artifacts"))
    valid_exported = sum(
        export(folder / "valid_sig_analysis", args.overwrite)
        for folder in artifacts_folders
    )
    test_exported = sum(export_test(folder, args.overwrite) for folder in artifacts_folders)
    print(
        f"Exported readable CSV: validation={valid_exported}, test={test_exported} recorder(s)."
    )


if __name__ == "__main__":
    main()
