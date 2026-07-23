#!/usr/bin/env python3
"""Predict a date with the frozen XGB60/120 + LGB84 + Cat120 family ensemble."""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

from predict_recorder_date import (
    ROOT,
    estimated_execution_dates,
    load,
    market_board,
    name_map,
    write_prediction_metadata,
)


COMPONENTS = [
    ("XGBoost", "XGB60", "858177734427251719", "5267a756271340f28c0c390bb31f1202", 0.266971),
    ("XGBoost", "XGB120", "515608396499357652", "ab7a8c9a05c64ec59f3b46099ab8c628", 0.329498),
    ("LightGBM", "LGB84", "285578722069352086", "3baed6997dff4755be9c86039765d6c9", 0.363679),
    ("CatBoost", "Cat120", "210252862375826404", "871dba9c1c854fc4a250846c2da35942", 0.340983),
]


def predict_component(experiment_id, recorder_id, signal_date):
    artifacts = ROOT / ".qlibAssistant" / "mlruns" / experiment_id / recorder_id / "artifacts"
    task = load(artifacts / "task")
    model = load(artifacts / "params.pkl")
    config = copy.deepcopy(task["dataset"])
    config["kwargs"]["segments"]["test"] = (signal_date, signal_date)
    handler = config["kwargs"]["handler"]["kwargs"]
    if str(handler.get("end_time", "")) < signal_date:
        handler["end_time"] = signal_date
    dataset = init_instance_by_config(config)
    prediction = model.predict(dataset, segment="test")
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.iloc[:, 0]
    return prediction.droplevel("datetime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--exclude-star-market",
        action="store_true",
        help="仅控制终端显示和兼容文件 ensemble_ranking.csv；两种版本始终都会生成",
    )
    args = parser.parse_args()
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN)
    frames, metadata = [], []
    for family, config, experiment_id, recorder_id, valid_rank_icir in COMPONENTS:
        score = predict_component(experiment_id, recorder_id, args.date).rename(config)
        zscore = ((score - score.mean()) / score.std()).rename(f"z_{config}")
        rank = zscore.rank(ascending=False, method="min").rename(f"rank_{config}")
        frames.extend([zscore, rank])
        metadata.append((family, config, valid_rank_icir))
    output = pd.concat(frames, axis=1, join="inner")
    family_scores, family_weights = {}, {}
    for family in sorted({item[0] for item in metadata}):
        members = [item for item in metadata if item[0] == family]
        family_scores[family] = output[[f"z_{item[1]}" for item in members]].mean(axis=1)
        family_weights[family] = sum(item[2] for item in members) / len(members)
    total_weight = sum(family_weights.values())
    output["ensemble_score"] = sum(
        family_scores[family] * family_weights[family] / total_weight
        for family in family_scores
    )
    output["top20_votes"] = sum(output[f"rank_{item[1]}"] <= 20 for item in metadata)
    output = output.sort_values("ensemble_score", ascending=False).reset_index()
    output.rename(columns={output.columns[0]: "instrument"}, inplace=True)
    output.insert(0, "model_rank", range(1, len(output) + 1))
    output = output.merge(name_map(), on="instrument", how="left")
    output["所属板块"] = output.instrument.map(market_board)
    output["是否科创板"] = output["所属板块"].eq("科创板")
    output.insert(0, "rank", range(1, len(output) + 1))
    output.insert(output.columns.get_loc("instrument") + 1, "datetime", args.date)
    output_all = output
    output_ex_star = output_all[~output_all["是否科创板"]].copy()
    output_ex_star["rank"] = range(1, len(output_ex_star) + 1)
    displayed_output = output_ex_star if args.exclude_star_market else output_all
    folder = ROOT / ".qlibAssistant" / "analysis" / f"fixed_ensemble_{args.date.replace('-', '')}_{time.strftime('%H%M%S')}"
    folder.mkdir(parents=True)
    output_all.to_csv(folder / "ensemble_ranking_all.csv", index=False)
    output_ex_star.to_csv(folder / "ensemble_ranking_ex_star.csv", index=False)
    displayed_output.to_csv(folder / "ensemble_ranking.csv", index=False)
    buy_date, sell_date = estimated_execution_dates(args.date)
    components = [
        {
            "family": family,
            "name": name,
            "experiment_id": experiment_id,
            "recorder_id": recorder_id,
            "validation_rank_icir": valid_rank_icir,
        }
        for family, name, experiment_id, recorder_id, valid_rank_icir in COMPONENTS
    ]
    metadata = {
        "prediction_type": "fixed_ensemble",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_date": args.date,
        "estimated_buy_date": buy_date,
        "estimated_sell_date": sell_date,
        "execution_semantics": "signal T close -> buy T+1 close -> sell T+2 close",
        "stock_pool": "csi1000",
        "weighting": "model-family mean, then family validation Rank ICIR weighted",
        "components": components,
        "terminal_excludes_star_market": args.exclude_star_market,
        "topk_displayed_in_terminal": args.topk,
        "all_output_rows": len(output_all),
        "ex_star_output_rows": len(output_ex_star),
        "ranking_alias": "ensemble_ranking_ex_star.csv" if args.exclude_star_market else "ensemble_ranking_all.csv",
        "all_ranking_file": "ensemble_ranking_all.csv",
        "ex_star_ranking_file": "ensemble_ranking_ex_star.csv",
        "alias_ranking_file": "ensemble_ranking.csv",
        "ranking_files": [
            "ensemble_ranking_all.csv",
            "ensemble_ranking_ex_star.csv",
            "ensemble_ranking.csv",
        ],
    }
    write_prediction_metadata(folder, metadata)
    print(displayed_output.head(args.topk).to_string(index=False))
    print(f"output_dir={folder}")


if __name__ == "__main__":
    main()
