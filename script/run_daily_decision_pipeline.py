#!/usr/bin/env python3
"""Update data, run four daily signals, and build one decision checklist."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".qlibAssistant/analysis"
QLIB_CALENDAR = Path.home() / ".qlib/qlib_data/cn_data/calendars/day.txt"
BREADTH_CACHE = ROOT / ".qlibAssistant/cache/market_breadth_daily.csv"
PYTHON = Path("/Users/hmax/miniconda3/envs/qlibAssistant/bin/python")
XGB240_EXPERIMENT_ID = "473901139733640553"
XGB240_RECORDER_ID = "021853d841234c4fb397e0e919800b58"
MARKET_GATE_THRESHOLD = 0.40


def latest_qlib_date() -> str:
    if not QLIB_CALENDAR.exists():
        raise RuntimeError(f"Qlib calendar not found: {QLIB_CALENDAR}")
    dates = [
        line.strip()
        for line in QLIB_CALENDAR.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dates:
        raise RuntimeError(f"Qlib calendar is empty: {QLIB_CALENDAR}")
    return dates[-1]


def run_command(
    name: str,
    command: list[str],
    log_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> Path | None:
    log_path = log_dir / f"{name}.log"
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(command)}", flush=True)
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8") as stream:
        if process.stdout is None:
            raise RuntimeError(f"{name} did not expose stdout")
        for line in process.stdout:
            lines.append(line)
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"{name} failed with exit code {returncode}; see {log_path}"
        )
    output_dir = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("output_dir="):
            output_dir = Path(line.split("=", 1)[1].strip()).expanduser().resolve()
            break
    return output_dir


def newest_matching(pattern: str) -> Path:
    matches = sorted(ANALYSIS.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise RuntimeError(f"No existing output matches {ANALYSIS / pattern}")
    return matches[-1]


def newest_breadth_for_date(signal_date: str) -> Path:
    matches = sorted(
        ANALYSIS.glob("market_breadth_prediction_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for folder in matches:
        config_path = folder / "prediction_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("signal_date") == signal_date:
            return folder
    raise RuntimeError(f"No market-breadth output found for {signal_date}")


def existing_outputs(signal_date: str) -> dict[str, Path]:
    compact = signal_date.replace("-", "")
    return {
        "mainboard20": newest_matching(f"mainboard_top20_ensemble_{compact}_*"),
        "xgb240": newest_matching(f"recorder_prediction_{compact}_*"),
        "fixed": newest_matching(f"fixed_ensemble_{compact}_*"),
        "market_breadth": newest_breadth_for_date(signal_date),
    }


def breadth_cache_latest_date() -> str | None:
    if not BREADTH_CACHE.exists():
        return None
    dates = pd.read_csv(BREADTH_CACHE, usecols=["datetime"])["datetime"]
    if dates.empty:
        return None
    return str(pd.Timestamp(dates.iloc[-1]).date())


def run_predictions(signal_date: str, log_dir: Path) -> dict[str, Path]:
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    outputs = {}
    outputs["mainboard20"] = run_command(
        "predict_mainboard20",
        [
            str(PYTHON),
            "script/predict_mainboard_ensemble_date.py",
            "--date",
            signal_date,
            "--fold",
            "fold3",
            "--top-models",
            "20",
            "--topk",
            "10",
        ],
        log_dir,
        env=env,
    )
    outputs["xgb240"] = run_command(
        "predict_xgb240",
        [
            str(PYTHON),
            "script/predict_recorder_date.py",
            "--experiment-id",
            XGB240_EXPERIMENT_ID,
            "--recorder-id",
            XGB240_RECORDER_ID,
            "--date",
            signal_date,
            "--topk",
            "10",
        ],
        log_dir,
        env=env,
    )
    outputs["fixed"] = run_command(
        "predict_fixed",
        [
            str(PYTHON),
            "script/predict_fixed_ensemble_date.py",
            "--date",
            signal_date,
            "--topk",
            "10",
        ],
        log_dir,
        env=env,
    )
    outputs["market_breadth"] = run_command(
        "predict_market_breadth",
        [
            str(PYTHON),
            "script/predict_market_breadth_date.py",
            "--date",
            signal_date,
        ],
        log_dir,
        env=env,
    )
    missing = [name for name, path in outputs.items() if path is None]
    if missing:
        raise RuntimeError(f"Prediction scripts did not report output_dir: {missing}")
    return {name: path for name, path in outputs.items() if path is not None}


def read_ranking(path: Path, source: str) -> pd.DataFrame:
    if source == "xgb240":
        csv_path = path / "ranking_ex_star_chinext.csv"
    else:
        csv_path = path / "ensemble_ranking_ex_star_chinext.csv"
    frame = pd.read_csv(csv_path, dtype={"instrument": str})
    if "rank" not in frame:
        raise RuntimeError(f"Ranking file has no rank column: {csv_path}")
    columns = ["instrument", "name", "所属板块", "rank"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing}")
    return frame[columns].rename(columns={"rank": f"{source}_rank"})


def estimated_dates(folder: Path) -> tuple[str, str]:
    config = json.loads(
        (folder / "prediction_config.json").read_text(encoding="utf-8")
    )
    return (
        str(config.get("estimated_buy_date", "next trading day")),
        str(config.get("estimated_sell_date", "following trading day")),
    )


def build_reports(
    signal_date: str,
    outputs: dict[str, Path],
    output_dir: Path,
) -> None:
    mainboard = read_ranking(outputs["mainboard20"], "mainboard20")
    xgb = read_ranking(outputs["xgb240"], "xgb240")
    fixed = read_ranking(outputs["fixed"], "fixed")

    combined = mainboard.merge(
        xgb,
        on=["instrument", "name", "所属板块"],
        how="outer",
    ).merge(
        fixed,
        on=["instrument", "name", "所属板块"],
        how="outer",
    )
    rank_columns = ["mainboard20_rank", "xgb240_rank", "fixed_rank"]
    combined["top10_votes"] = sum(combined[column].le(10) for column in rank_columns)
    combined["top20_votes"] = sum(combined[column].le(20) for column in rank_columns)
    combined["rank_sum"] = combined[rank_columns].fillna(10000).sum(axis=1)
    combined = combined.sort_values(
        ["top10_votes", "rank_sum", "mainboard20_rank"],
        ascending=[False, True, True],
    )
    combined.insert(0, "consensus_rank", range(1, len(combined) + 1))
    combined.head(100).to_csv(output_dir / "consensus_top100.csv", index=False)

    breadth = pd.read_csv(
        outputs["market_breadth"] / "ensemble_predictions.csv"
    )
    top2 = breadth.loc[breadth["ensemble_size"].eq(2)]
    if len(top2) != 1:
        raise RuntimeError(
            f"Expected one market Top2 row, got {len(top2)} in "
            f"{outputs['market_breadth']}"
        )
    breadth_value = float(top2.iloc[0]["predicted_up_ratio"])
    market_gate_pass = breadth_value >= MARKET_GATE_THRESHOLD

    source_frames = {
        "Mainboard20": mainboard,
        "XGBoost240": xgb,
        "FixedArchitecture": fixed,
    }
    actions = []
    for display_name, frame in source_frames.items():
        rank_column = next(
            column for column in frame.columns if column.endswith("_rank")
        )
        candidate = frame.sort_values(rank_column).iloc[0]
        gate_required = display_name != "Mainboard20"
        if gate_required and not market_gate_pass:
            preliminary_action = "市场门槛未通过：留现金"
        else:
            preliminary_action = "候选：待买入日尾盘event_guard确认"
        actions.append(
            {
                "strategy": display_name,
                "instrument": candidate["instrument"],
                "name": candidate["name"],
                "所属板块": candidate["所属板块"],
                "source_rank": int(candidate[rank_column]),
                "market_gate_required": gate_required,
                "market_top2_predicted_up_ratio": breadth_value,
                "market_gate_threshold": MARKET_GATE_THRESHOLD,
                "market_gate_pass": (
                    market_gate_pass if gate_required else "未正式测试"
                ),
                "event_guard_status": "待买入日尾盘确认",
                "preliminary_action": preliminary_action,
            }
        )
    action_frame = pd.DataFrame(actions)
    action_frame.to_csv(output_dir / "source_actions.csv", index=False)

    buy_date, sell_date = estimated_dates(outputs["mainboard20"])
    top_consensus = combined.head(10)[
        [
            "instrument",
            "name",
            "所属板块",
            "mainboard20_rank",
            "xgb240_rank",
            "fixed_rank",
            "top10_votes",
        ]
    ]
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "estimated_buy_date": buy_date,
        "estimated_sell_date": sell_date,
        "market_top2_predicted_up_ratio": breadth_value,
        "market_gate_threshold": MARKET_GATE_THRESHOLD,
        "market_gate_pass_for_xgb_fixed": market_gate_pass,
        "event_guard": {
            "status": "pending_buy_day_close_confirmation",
            "max_limit_down_count_10d": 1,
            "min_drawdown_20d": -0.30,
            "buy_day_must_not_touch_limit_down": True,
            "fallback_to_lower_ranked": False,
        },
        "important_scope": {
            "mainboard20_market_gate": "not_backtested",
            "consensus_vote_rule": "monitoring_only_not_backtested",
            "fixed_ensemble": "no direct future-data leakage; post-hoc selection bias",
        },
        "source_outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "decision_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    action_markdown = action_frame.to_markdown(index=False)
    consensus_markdown = top_consensus.to_markdown(index=False)
    market_state = "通过" if market_gate_pass else "未通过"
    summary = f"""# {signal_date} 每日量化决策清单

## 时序

- 信号日：{signal_date}
- 预计买入：{buy_date}尾盘
- 预计卖出：{sell_date}尾盘
- 市场宽度Top2：{breadth_value:.2%}
- XGBoost/Fixed市场门槛：{MARKET_GATE_THRESHOLD:.0%}，当前{market_state}

## 三路Top1

{action_markdown}

## 三路Top10共识

{consensus_markdown}

## 买入日前必须再次确认

`event_guard`不能在信号日晚间提前判定。买入日临近收盘时检查：

1. 最近10个交易日跌停次数不超过1次；
2. 当前价格相对近20日最高收盘回撤高于-30%；
3. 买入日最低价没有触及跌停；
4. 当前未涨停、未停牌并可成交；
5. 不通过时留现金，不自动购买下一名。

主板20的市场40%门槛尚未回测；三路Top10投票也只作观察，不是已验证的硬规则。
Fixed Ensemble没有直接使用未来数据，但组件是在看过历史实验后冻结，存在事后
选择偏差。
"""
    (output_dir / "DECISION_SUMMARY.md").write_text(summary, encoding="utf-8")
    print("\n===== 每日决策摘要 =====")
    print(action_frame.to_string(index=False))
    print("\n===== 三路共识Top10 =====")
    print(top_consensus.to_string(index=False))
    print(f"\nmarket_top2={breadth_value:.4f} gate_pass={market_gate_pass}")
    print(f"output_dir={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update Tushare/Dolt/Qlib data before prediction.",
    )
    parser.add_argument(
        "--date",
        help="Signal date; default is the latest Qlib calendar date.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse latest four prediction folders for the signal date.",
    )
    parser.add_argument(
        "--force-breadth-rebuild",
        action="store_true",
        help="Rebuild market-breadth history even when its latest date matches.",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        raise RuntimeError(f"Python environment not found: {PYTHON}")

    bootstrap_dir = ANALYSIS / (
        "daily_decision_bootstrap_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    bootstrap_dir.mkdir(parents=True, exist_ok=False)
    if args.update:
        run_command(
            "update_data",
            [str(PYTHON), "update-predict.py", "--update-only"],
            bootstrap_dir,
        )

    signal_date = args.date or latest_qlib_date()
    compact = signal_date.replace("-", "")
    output_dir = ANALYSIS / (
        f"daily_decision_{compact}_{datetime.now().strftime('%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Refresh breadth features after every data update. In reuse mode, retain
    # the exact historical prediction artifacts instead of regenerating them.
    if not args.reuse_existing:
        cached_breadth_date = breadth_cache_latest_date()
        if args.force_breadth_rebuild or cached_breadth_date != signal_date:
            run_command(
                "build_market_breadth_dataset",
                [str(PYTHON), "script/build_market_breadth_dataset.py"],
                output_dir,
            )
        else:
            message = (
                f"breadth cache already covers {signal_date}; "
                "skip full-history rebuild\n"
            )
            (output_dir / "build_market_breadth_dataset.log").write_text(
                message,
                encoding="utf-8",
            )
            print(f"[{time.strftime('%H:%M:%S')}] {message}", end="", flush=True)
        outputs = run_predictions(signal_date, output_dir)
    else:
        outputs = existing_outputs(signal_date)

    build_reports(signal_date, outputs, output_dir)
    if bootstrap_dir.exists():
        for source in list(bootstrap_dir.iterdir()):
            destination = output_dir / source.name
            if destination.exists():
                destination = output_dir / f"bootstrap_{source.name}"
            source.replace(destination)
        bootstrap_dir.rmdir()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
