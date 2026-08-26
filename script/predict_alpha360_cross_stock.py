#!/usr/bin/env python3
"""Run a completed Alpha360 model using only signal-day and earlier prices.

This experimental predictor is separate from the production daily decision
pipeline. Its probability outputs are not a trading instruction or a calibrated
win-rate guarantee.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

from train_alpha360_cross_stock import construct_alpha360, file_hash, write_json


def signal_features(data_api, signal_date, pool="csi1000"):
    import numpy as np
    import pandas as pd

    calendar = pd.DatetimeIndex(data_api.calendar(future=False))
    signal = pd.Timestamp(signal_date)
    if signal not in calendar:
        raise ValueError(f"Signal date is absent from provider calendar: {signal_date}")
    position = calendar.get_loc(signal)
    if position < 59:
        raise ValueError("Less than 60 trading dates of history")
    history = calendar[position - 59:position + 1]
    members = data_api.list_instruments(data_api.instruments(pool), start_time=signal, end_time=signal, as_list=False)
    codes = sorted(members)
    if not codes:
        raise ValueError("Empty signal-day universe")
    fields = ["$close", "$open", "$high", "$low", "$vwap", "$volume"]
    raw = data_api.features(codes, fields, start_time=history[0], end_time=signal)
    features, quote_valid = [], []
    for code in codes:
        values = raw.xs(code, level="instrument").reindex(history).to_numpy(dtype="float32")
        features.append(construct_alpha360(values, [59])[0])
        quote_valid.append(bool(np.isfinite(values[-1, 0]) and values[-1, 0] > 0))
    return codes, np.stack(features), np.asarray(quote_valid)


def board(code):
    if code.startswith(("SH688", "SH689")):
        return "科创板"
    if code.startswith(("SZ300", "SZ301")):
        return "创业板"
    if code.startswith("SH60"):
        return "沪市主板"
    if code.startswith("SZ00"):
        return "深市主板"
    return "其他"


def main(args):
    import qlib
    import numpy as np
    import pandas as pd
    import torch
    from qlib.data import D
    from roll.alpha360_cross_stock import Alpha360CrossStockTransformer, Alpha360TransformerConfig, HORIZON_NAMES, distribution_report

    status = json.loads((args.run / "status.json").read_text(encoding="utf-8-sig"))
    if status.get("status") != "completed":
        raise RuntimeError("This entry point requires a completed, saved run")
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve older predictions")
    torch.set_num_threads(2)
    qlib.init(provider_uri=str(Path(args.provider).expanduser()), region="cn", kernels=2)
    calendar = pd.DatetimeIndex(D.calendar(future=False))
    date = str(calendar[-1].date()) if args.date == "latest" else args.date
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    requested = pd.Timestamp(date).normalize()
    if requested.date() > now.date() or (requested.date() == now.date() and (now.hour, now.minute) < (15, 30)):
        raise ValueError("Daily inference requires a past date or today's confirmed post-15:30 close data")
    codes, features, quote_valid = signal_features(D, date, args.pool)
    stock_ids = json.loads((args.run / "stock_ids.json").read_text())
    normalizer = np.load(args.run / "normalizer.npz")
    normalized = (features - normalizer["mean"]) / normalizer["std"]
    normalized[~np.isfinite(normalized)] = 0
    ids = np.asarray([stock_ids.get(code, 0) for code in codes], dtype="int64")
    checkpoint = torch.load(args.run / "best_model.pt", map_location="cpu", weights_only=False)
    config = Alpha360TransformerConfig(**checkpoint["configuration"]["model"])
    model = Alpha360CrossStockTransformer(len(stock_ids), config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    dtype = torch.bfloat16 if checkpoint["configuration"]["autocast_dtype"] == "torch.bfloat16" else torch.float16
    with torch.no_grad(), torch.autocast(device_type=args.device, dtype=dtype, enabled=args.device == "cuda"):
        result = model(torch.from_numpy(normalized)[None].to(args.device), torch.from_numpy(ids)[None].to(args.device))
    report = distribution_report(result["horizon_mean"][0] / config.target_scale,
                                 result["horizon_covariance"][0] / config.target_scale**2)
    frame = pd.DataFrame({"instrument": codes, "code": [code[2:] for code in codes],
                          "name": "", "board": [board(code) for code in codes], "datetime": date,
                          "signal_close_available": quote_valid, "known_stock_embedding": ids != 0})
    if args.names_csv:
        names = pd.read_csv(args.names_csv, dtype=str)
        if not {"instrument", "name"}.issubset(names):
            raise ValueError("names CSV requires instrument and name columns")
        mapping = names.drop_duplicates("instrument").set_index("instrument")["name"]
        frame["name"] = frame["instrument"].map(mapping).fillna("")
    for column, horizon in enumerate(HORIZON_NAMES):
        for key, values in report.items():
            frame[horizon + "_" + key] = values[:, column].float().cpu().numpy()
    frame = frame.sort_values(["signal_close_available", args.rank_horizon + "_expected_return"], ascending=False)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "ranking_all.csv", index=False)
    mainboard = frame.loc[frame["board"].isin(["沪市主板", "深市主板"]) & frame["signal_close_available"]]
    mainboard.to_csv(args.output / "ranking_mainboard.csv", index=False)
    write_json(args.output / "metadata.json", {
        "signal_date": date, "input_end_time": date, "input_fields": ["close", "open", "high", "low", "vwap", "volume"],
        "uses_future_prices": False, "pool": args.pool, "stocks": len(frame), "mainboard_stocks": len(mainboard),
        "unknown_stock_ids": [code for code, idx in zip(codes, ids) if idx == 0],
        "unknown_identity_policy": "reserved all-zero embedding; explicitly flagged",
        "checkpoint_sha256": file_hash(args.run / "best_model.pt"), "best_epoch": checkpoint["epoch"],
        "segments": checkpoint["configuration"]["segments"], "device": args.device,
        "ranking_horizon": args.rank_horizon, "rank_key": "expected ordinary return",
        "caveat": "experimental, no event_guard/market gate/execution/cost filters; not a live buy instruction",
    })
    print(json.dumps({"output": str(args.output), "signal_date": date, "stocks": len(frame), "mainboard_stocks": len(mainboard)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--date", default="latest")
    parser.add_argument("--pool", default="csi1000")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--names-csv", type=Path)
    parser.add_argument("--rank-horizon", choices=["open1_close2", "close1_open2", "open1_open2", "close1_close2"], default="close1_close2")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
