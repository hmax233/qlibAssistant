#!/usr/bin/env python3
"""Run one saved MLflow/Qlib recorder on a requested signal date."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def name_map():
    candidates = sorted(
        (ROOT / ".qlibAssistant" / "analysis").glob("selection*/20*_ret.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    mappings = []
    for path in candidates:
        frame = pd.read_csv(path)
        if {"instrument", "name"}.issubset(frame.columns) and frame.name.notna().any():
            mappings.append(frame[["instrument", "name"]].dropna())
    if mappings:
        return pd.concat(mappings, ignore_index=True).drop_duplicates("instrument")
    return pd.DataFrame(columns=["instrument", "name"])


def market_board(instrument: str) -> str:
    instrument = str(instrument).upper()
    if instrument.startswith(("SH688", "SH689")):
        return "科创板"
    if instrument.startswith(("SZ300", "SZ301")):
        return "创业板"
    if instrument.startswith("BJ"):
        return "北交所"
    if instrument.startswith("SH"):
        return "沪市主板"
    if instrument.startswith("SZ"):
        return "深市主板"
    return "其他"


def estimated_execution_dates(signal_date: str) -> tuple[str, str]:
    """Return business-day estimates; exchange holidays may move these dates."""
    signal = pd.Timestamp(signal_date)
    buy = signal + pd.offsets.BDay(1)
    sell = signal + pd.offsets.BDay(2)
    return buy.strftime("%Y-%m-%d"), sell.strftime("%Y-%m-%d")


def write_prediction_metadata(folder: Path, metadata: dict) -> None:
    (folder / "prediction_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    components = metadata.get("components", [])
    component_lines = "\n".join(
        f"- {item.get('name', item.get('model', 'model'))}: "
        f"experiment `{item['experiment_id']}`, recorder `{item['recorder_id']}`"
        for item in components
    )
    if not component_lines:
        component_lines = (
            f"- {metadata.get('model', 'unknown')}: experiment "
            f"`{metadata.get('experiment_id', '')}`, recorder "
            f"`{metadata.get('recorder_id', '')}`"
        )
    readme = f"""# 预测结果说明

- 预测类型：{metadata['prediction_type']}
- 信号日（特征截面）：{metadata['signal_date']}
- 理论买入：T+1 收盘附近（日期估算：{metadata['estimated_buy_date']}）
- 理论卖出：T+2 收盘附近（日期估算：{metadata['estimated_sell_date']}）
- 运行时间：{metadata['generated_at']}
- 股票池：{metadata.get('stock_pool', 'unknown')}
- `{metadata['all_ranking_file']}`：完整股票池，{metadata['all_output_rows']}只股票
- `{metadata['ex_star_ranking_file']}`：剔除科创板后，{metadata['ex_star_output_rows']}只股票
- `{metadata['alias_ranking_file']}`：兼容入口，当前内容对应 `{metadata['ranking_alias']}`

## 模型组件

{component_lines}

> 买卖日期按工作日估算，交易所节假日可能使日期顺延；模型分数主要用于横截面排序，不应直接解释为绝对收益率。
"""
    (folder / "README.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--recorder-id", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--exclude-star-market",
        action="store_true",
        help="仅控制终端显示和兼容文件 ranking.csv；两种版本始终都会生成",
    )
    args = parser.parse_args()
    artifacts = ROOT / ".qlibAssistant" / "mlruns" / args.experiment_id / args.recorder_id / "artifacts"
    task = load(artifacts / "task")
    model = load(artifacts / "params.pkl")
    dataset_config = copy.deepcopy(task["dataset"])
    dataset_config["kwargs"]["segments"]["test"] = (args.date, args.date)
    handler_kwargs = dataset_config["kwargs"]["handler"]["kwargs"]
    if str(handler_kwargs.get("end_time", "")) < args.date:
        handler_kwargs["end_time"] = args.date
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/cn_data"), region=REG_CN)
    dataset = init_instance_by_config(dataset_config)
    prediction = model.predict(dataset, segment="test")
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.iloc[:, 0]
    output = prediction.rename("score").reset_index().sort_values("score", ascending=False)
    output.insert(0, "model_rank", range(1, len(output) + 1))
    output = output.merge(name_map(), on="instrument", how="left")
    output["所属板块"] = output.instrument.map(market_board)
    output["是否科创板"] = output["所属板块"].eq("科创板")
    output.insert(0, "rank", range(1, len(output) + 1))
    output_all = output[[
        "rank", "model_rank", "instrument", "name", "所属板块", "是否科创板", "datetime", "score"
    ]]
    output_ex_star = output_all[~output_all["是否科创板"]].copy()
    output_ex_star["rank"] = range(1, len(output_ex_star) + 1)
    displayed_output = output_ex_star if args.exclude_star_market else output_all
    folder = ROOT / ".qlibAssistant" / "analysis" / f"recorder_prediction_{args.date.replace('-', '')}_{time.strftime('%H%M%S')}"
    folder.mkdir(parents=True)
    output_all.to_csv(folder / "ranking_all.csv", index=False)
    output_ex_star.to_csv(folder / "ranking_ex_star.csv", index=False)
    displayed_output.to_csv(folder / "ranking.csv", index=False)
    buy_date, sell_date = estimated_execution_dates(args.date)
    segments = task["dataset"]["kwargs"]["segments"]
    experiment_meta = ROOT / ".qlibAssistant" / "mlruns" / args.experiment_id / "meta.yaml"
    experiment_name = args.experiment_id
    if experiment_meta.exists():
        for line in experiment_meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                experiment_name = line.split(":", 1)[1].strip()
                break
    metadata = {
        "prediction_type": "single_recorder",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_date": args.date,
        "estimated_buy_date": buy_date,
        "estimated_sell_date": sell_date,
        "execution_semantics": "signal T close -> buy T+1 close -> sell T+2 close",
        "experiment_id": args.experiment_id,
        "experiment_name": experiment_name,
        "recorder_id": args.recorder_id,
        "model": task["model"].get("class", "unknown"),
        "stock_pool": task["dataset"]["kwargs"]["handler"]["kwargs"].get("instruments", "unknown"),
        "segments": segments,
        "terminal_excludes_star_market": args.exclude_star_market,
        "topk_displayed_in_terminal": args.topk,
        "all_output_rows": len(output_all),
        "ex_star_output_rows": len(output_ex_star),
        "ranking_alias": "ranking_ex_star.csv" if args.exclude_star_market else "ranking_all.csv",
        "all_ranking_file": "ranking_all.csv",
        "ex_star_ranking_file": "ranking_ex_star.csv",
        "alias_ranking_file": "ranking.csv",
        "ranking_files": ["ranking_all.csv", "ranking_ex_star.csv", "ranking.csv"],
    }
    write_prediction_metadata(folder, metadata)
    print(displayed_output.head(args.topk).to_string(index=False))
    print(f"output_dir={folder}")


if __name__ == "__main__":
    main()
