#!/usr/bin/env python3
"""将若干测试/实盘预测按模型家族标准化集成，输出最新信号候选。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.workflow import R


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def init_qlib():
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(
        (PROJECT_ROOT / ".qlibAssistant" / "mlruns").resolve()
    )
    qlib.init(
        provider_uri=str(Path("~/.qlib/qlib_data/cn_data").expanduser()),
        region=REG_CN,
        exp_manager=exp_manager,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True,
        metavar="FAMILY,CONFIG,VALID_RANK_ICIR,REPORT_DIR",
    )
    parser.add_argument("--signal-date")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()
    init_qlib()
    components = []
    metadata = []
    for spec in args.report:
        family, config, valid_weight, directory = spec.split(",", 3)
        recorder_table = pd.read_csv(Path(directory) / "recorders.csv")
        if "selected_for_ensemble" in recorder_table.columns:
            chosen = recorder_table[
                recorder_table["selected_for_ensemble"].astype(str).str.lower() == "true"
            ]
            if not chosen.empty:
                recorder_table = chosen
        identity = recorder_table.iloc[0]
        recorder = R.get_exp(experiment_id=str(identity["experiment_id"])).get_recorder(
            recorder_id=str(identity["recorder_id"])
        )
        raw_prediction = recorder.load_object("pred.pkl")
        if isinstance(raw_prediction, pd.DataFrame):
            raw_prediction = raw_prediction.iloc[:, 0]
        prediction = raw_prediction.rename("score").reset_index()
        prediction["datetime"] = pd.to_datetime(prediction["datetime"])
        latest = prediction["datetime"].max()
        signal_date = pd.Timestamp(args.signal_date) if args.signal_date else latest
        current = prediction[prediction["datetime"] == signal_date][["instrument", "score"]].dropna()
        mean, std = current["score"].mean(), current["score"].std()
        current[f"z_{config}"] = (current["score"] - mean) / std
        current[f"rank_{config}"] = current[f"z_{config}"].rank(ascending=False, method="min")
        components.append(current.drop(columns="score").set_index("instrument"))
        metadata.append({"family": family, "config": config, "valid_rank_icir": float(valid_weight), "report_dir": directory})

    merged = pd.concat(components, axis=1, join="inner")
    family_scores = {}
    family_weights = {}
    for family in sorted({item["family"] for item in metadata}):
        configs = [item for item in metadata if item["family"] == family]
        family_scores[family] = merged[[f"z_{item['config']}" for item in configs]].mean(axis=1)
        family_weights[family] = sum(item["valid_rank_icir"] for item in configs) / len(configs)
    weight_sum = sum(family_weights.values())
    for family, score in family_scores.items():
        merged[f"family_{family}"] = score
        merged[f"weight_{family}"] = family_weights[family] / weight_sum
    merged["ensemble_score"] = sum(
        merged[f"family_{family}"] * merged[f"weight_{family}"] for family in family_scores
    )
    rank_columns = [column for column in merged if column.startswith("rank_")]
    merged["top20_votes"] = sum(merged[column] <= 20 for column in rank_columns)
    merged["ensemble_rank"] = merged["ensemble_score"].rank(ascending=False, method="min").astype(int)
    merged = merged.sort_values("ensemble_score", ascending=False).reset_index()

    names_path = PROJECT_ROOT / ".qlibAssistant" / "cache" / "stock_basic.csv"
    if names_path.exists():
        names = pd.read_csv(names_path).drop_duplicates("code")
        merged = merged.merge(names, left_on="instrument", right_on="code", how="left")
        ordered = ["ensemble_rank", "instrument", "name", "ensemble_score", "top20_votes"]
        merged = merged[ordered + [column for column in merged if column not in ordered + ["code"]]]

    signal_date = pd.Timestamp(args.signal_date) if args.signal_date else latest
    output = PROJECT_ROOT / ".qlibAssistant" / "analysis" / f"live_ensemble_{pd.Timestamp(signal_date):%Y%m%d}_{time.strftime('%H%M%S')}"
    output.mkdir(parents=True)
    merged.to_csv(output / "ensemble_ranking.csv", index=False)
    (output / "ensemble_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    top = merged.head(args.topk)
    (output / "top_candidates.md").write_text(
        f"# {pd.Timestamp(signal_date):%Y-%m-%d} 实验集成候选\n\n"
        + top.to_markdown(index=False)
        + "\n\n> 研究信号，不构成保证收益或个性化投资建议；未模拟当日实时价格、涨跌停和成交失败。\n"
    )
    print(top.to_string(index=False))
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
