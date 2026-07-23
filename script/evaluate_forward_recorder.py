#!/usr/bin/env python3
"""Re-run a saved recorder on dates beyond its stored pred.pkl and evaluate available labels."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config

from evaluation_variants import evaluate_board_variants


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if type(value).__name__ == "TRAModel" and not hasattr(value, "_writer"):
        value._writer = None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--recorder-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    args = parser.parse_args()

    artifacts = ROOT / ".qlibAssistant" / "mlruns" / args.experiment_id / args.recorder_id / "artifacts"
    task = load(artifacts / "task")
    model = load(artifacts / "params.pkl")
    config = copy.deepcopy(task["dataset"])
    config["kwargs"]["segments"]["test"] = (args.start, args.end)
    handler = config["kwargs"]["handler"]["kwargs"]
    handler["end_time"] = max(str(handler.get("end_time", args.end)), args.end)
    pool = handler.get("instruments", "csi1000")

    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN)
    dataset = init_instance_by_config(config)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")
    label = D.features(
        D.instruments(pool),
        ["Ref($close,-2)/Ref($close,-1)-1"],
        start_time=args.start,
        end_time=args.end,
        freq="day",
    ).iloc[:, 0].rename("label")
    if set(pred.index.names) == {"datetime", "instrument"}:
        pred = pred.reorder_levels(["datetime", "instrument"]).sort_index()
    if set(label.index.names) == {"datetime", "instrument"}:
        label = label.reorder_levels(["datetime", "instrument"]).sort_index()
    print(
        f"prediction_rows={len(pred)} prediction_dates={list(map(str, pred.index.get_level_values('datetime').unique()))} "
        f"label_rows={label.notna().sum()} label_dates={list(map(str, label.dropna().index.get_level_values('datetime').unique()))}"
    )
    frame = pd.concat([pred, label], axis=1).dropna()
    if frame.empty:
        raise SystemExit("日期范围内没有完整标签；1日目标要求信号日之后两个交易日收盘价已入库")

    summary, daily = evaluate_board_variants(frame, cost_rate=args.cost_rate)
    output = ROOT / ".qlibAssistant" / "analysis" / f"forward_evaluation_{time.strftime('%Y%m%d_%H_%M_%S')}"
    output.mkdir(parents=True)
    frame.reset_index().to_csv(output / "predictions_with_labels.csv", index=False)
    summary.to_csv(output / "board_variant_summary.csv", index=False)
    daily.to_csv(output / "board_variant_daily.csv", index=False)
    metadata = {
        "experiment_id": args.experiment_id,
        "recorder_id": args.recorder_id,
        "requested_start": args.start,
        "requested_end": args.end,
        "effective_start": str(frame.index.get_level_values("datetime").min().date()),
        "effective_end": str(frame.index.get_level_values("datetime").max().date()),
        "pool": pool,
        "cost_rate": args.cost_rate,
    }
    (output / "evaluation_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"report_dir={output}")


if __name__ == "__main__":
    main()
