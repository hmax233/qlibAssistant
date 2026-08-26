#!/usr/bin/env python3
"""Verify a completed Alpha360 experiment and create a readable statistical report.

This reports prediction/distribution quality, not a trading backtest. It does not
select thresholds or change the trained model after looking at Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score

from train_alpha360_cross_stock import DateStore, file_hash, write_json

NAMES = ("open1_close2", "close1_open2", "open1_open2", "close1_close2")
MATRIX = np.asarray([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype="float64")


def mainboard_code(code):
    return str(code).startswith(("SH60", "SZ00"))


def training_baseline(store, mainboard_only=False):
    """Constant predictor fit on Train only; equal weight per usable date."""
    means, seconds, return_means, positive_rates = [], [], [], []
    for part in store.manifest["parts"]:
        if part["split"] != "train":
            continue
        labels = np.load(store.directory / f"{part['prefix']}_labels.npy", mmap_mode="r")
        offsets = np.load(store.directory / f"{part['prefix']}_offsets.npy")
        ids = np.load(store.directory / f"{part['prefix']}_stock_ids.npy", mmap_mode="r") if mainboard_only else None
        for begin, end in zip(offsets[:-1], offsets[1:]):
            z = np.asarray(labels[begin:end], dtype="float64")
            if mainboard_only:
                eligible = [mainboard_code(store.id_to_code[int(idx)]) for idx in ids[begin:end]]
                z = z[np.asarray(eligible)]
            z = z[np.isfinite(z).all(axis=1)]
            if not len(z):
                continue
            means.append(z.mean(axis=0))
            seconds.append(z.T @ z / len(z))
            returns = np.expm1(z @ MATRIX.T)
            return_means.append(returns.mean(axis=0))
            positive_rates.append((returns > 0).mean(axis=0))
    if not means:
        raise RuntimeError("No Train labels for the baseline")
    mean = np.mean(means, axis=0)
    covariance = np.mean(seconds, axis=0) - np.outer(mean, mean)
    covariance += np.eye(3) * 1e-10
    return {
        "fit_split": "train", "fit_dates": len(means), "universe": "mainboard" if mainboard_only else "all",
        "leg_mean": mean.tolist(),
        "leg_covariance": covariance.tolist(),
        "ordinary_return_mean": np.mean(return_means, axis=0).tolist(),
        "probability_positive": np.mean(positive_rates, axis=0).tolist(),
    }


def validate_probability_columns(frame):
    for name in NAMES:
        for field in ("log_mean", "log_variance", "expected_return", "return_std", "probability_positive"):
            values = frame[f"{name}_{field}"].to_numpy()
            if not np.isfinite(values).all():
                raise AssertionError(f"Nonfinite predictions: {name}_{field}")
        mean = frame[name + "_log_mean"].to_numpy()
        variance = frame[name + "_log_variance"].to_numpy()
        if not (variance > 0).all():
            raise AssertionError("Nonpositive variance")
        probability = frame[name + "_probability_positive"].to_numpy()
        if not ((probability >= 0) & (probability <= 1)).all():
            raise AssertionError("Probability outside [0,1]")
        np.testing.assert_allclose(probability, ndtr(mean / np.sqrt(variance)), atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(frame[name + "_expected_return"], np.expm1(mean + variance / 2), atol=5e-7, rtol=2e-5)
        if not (frame[name + "_return_std"] >= 0).all():
            raise AssertionError("Negative return standard deviation")
    means = np.column_stack([frame[name + "_log_mean"] for name in NAMES])
    np.testing.assert_allclose(means[:, 0] + means[:, 1], means[:, 2] + means[:, 3], atol=2e-7, rtol=2e-5)


def validate_rows_and_labels(store, split, predictions):
    if predictions.index.duplicated().any():
        raise AssertionError("Duplicated date/stock predictions")
    all_indices, daily_baseline_labels = [], []
    for part in store.manifest["parts"]:
        if part["split"] != split:
            continue
        labels = np.load(store.directory / f"{part['prefix']}_labels.npy", mmap_mode="r")
        ids = np.load(store.directory / f"{part['prefix']}_stock_ids.npy", mmap_mode="r")
        offsets = np.load(store.directory / f"{part['prefix']}_offsets.npy")
        for date, begin, end in zip(part["dates"], offsets[:-1], offsets[1:]):
            codes = [store.id_to_code[int(idx)] for idx in ids[begin:end]]
            index = pd.MultiIndex.from_arrays([[pd.Timestamp(date)] * len(codes), codes], names=["datetime", "instrument"])
            actual = predictions.loc[index]
            truth = np.expm1(np.asarray(labels[begin:end], dtype="float64") @ MATRIX.T)
            reported = np.column_stack([actual[name + "_actual_return"].values for name in NAMES])
            np.testing.assert_allclose(reported, truth, atol=5e-7, rtol=2e-5, equal_nan=True)
            if date in store.manifest["purged_signal_dates"].get(split, []):
                if np.isfinite(truth).any():
                    raise AssertionError(f"Nonpurged boundary labels: {date}")
            all_indices.append(index)
            z = np.asarray(labels[begin:end], dtype="float64")
            daily_baseline_labels.append(z[np.isfinite(z).all(axis=1)])
    expected = all_indices[0]
    for index in all_indices[1:]:
        expected = expected.append(index)
    if set(predictions.index) != set(expected) or len(predictions) != len(expected):
        raise AssertionError(f"Prediction coverage mismatch in {split}")
    return daily_baseline_labels


def gaussian_baseline_nll(daily_labels, baseline, scale=100.0):
    mean = np.asarray(baseline["leg_mean"]) * scale
    covariance = np.asarray(baseline["leg_covariance"]) * scale**2
    factor = np.linalg.cholesky(covariance)
    constant = 2 * np.log(np.diag(factor)).sum() + 3 * np.log(2 * np.pi)
    per_day = []
    for labels in daily_labels:
        if not len(labels):
            continue
        residual = labels * scale - mean
        whitened = np.linalg.solve(factor, residual.T)
        per_day.append(float(np.mean((whitened**2).sum(axis=0) + constant) / 2))
    return float(np.mean(per_day))


def statistical_metrics(predictions, split, baseline):
    daily_rows, calibration_rows, summary_rows = [], [], []
    for horizon_idx, name in enumerate(NAMES):
        groups = []
        for date, frame in predictions.groupby(level="datetime"):
            y = frame[name + "_actual_return"].to_numpy()
            valid = np.isfinite(y)
            if not valid.any():
                continue
            y = y[valid]
            mean = frame[name + "_expected_return"].to_numpy()[valid]
            log_mean = frame[name + "_log_mean"].to_numpy()[valid]
            log_variance = frame[name + "_log_variance"].to_numpy()[valid]
            probability = frame[name + "_probability_positive"].to_numpy()[valid]
            up = y > 0
            rank_ic = pd.Series(mean).corr(pd.Series(y), method="spearman") if len(y) > 1 else np.nan
            row = {
                "split": split, "horizon": name, "datetime": str(date.date()), "stocks": len(y),
                "rank_ic": rank_ic, "mae": float(np.abs(mean - y).mean()),
                "zero_mae": float(np.abs(y).mean()),
                "train_mean_mae": float(np.abs(y - baseline["ordinary_return_mean"][horizon_idx]).mean()),
                "brier": float(((probability - up)**2).mean()),
                "train_rate_brier": float(((baseline["probability_positive"][horizon_idx] - up)**2).mean()),
                "up_accuracy": float(((probability >= 0.5) == up).mean()),
                "auc": float(roc_auc_score(up, probability)) if len(np.unique(up)) == 2 else np.nan,
                "actual_up_fraction": float(up.mean()),
            }
            distance = np.abs(np.log1p(y) - log_mean)
            for percent, zscore in ((50, 0.67448975), (80, 1.28155157), (95, 1.95996398)):
                row[f"coverage{percent}"] = float((distance <= zscore * np.sqrt(log_variance)).mean())
            daily_rows.append(row)
            groups.append(pd.DataFrame({"probability": probability, "up": up}))
        daily = pd.DataFrame([row for row in daily_rows if row["horizon"] == name])
        if daily.empty:
            raise AssertionError(f"No realized labels for {split}/{name}")
        summary = {"split": split, "horizon": name, "signal_days": len(daily), "stock_days": int(daily["stocks"].sum())}
        for field in ("rank_ic", "mae", "zero_mae", "train_mean_mae", "brier", "train_rate_brier", "up_accuracy", "auc", "actual_up_fraction", "coverage50", "coverage80", "coverage95"):
            summary[field] = float(daily[field].mean())
        deviation = daily["rank_ic"].std(ddof=0)
        summary["rank_icir"] = float(daily["rank_ic"].mean() / deviation) if deviation > 0 else np.nan
        summary["brier_skill_vs_train_rate"] = 1 - summary["brier"] / summary["train_rate_brier"] if summary["train_rate_brier"] > 0 else np.nan
        summary_rows.append(summary)
        pooled = pd.concat(groups, ignore_index=True)
        bins = np.minimum((pooled["probability"].to_numpy() * 10).astype(int), 9)
        for bucket in range(10):
            selected = pooled.iloc[np.flatnonzero(bins == bucket)]
            if selected.empty:
                continue
            calibration_rows.append({"split": split, "horizon": name, "bin_low": bucket / 10,
                                     "bin_high": (bucket + 1) / 10, "count": len(selected),
                                     "predicted_probability": float(selected["probability"].mean()),
                                     "observed_up_fraction": float(selected["up"].mean())})
    return summary_rows, daily_rows, calibration_rows


def replay_checkpoints(run, store, predictions_by_split, device):
    import qlib  # noqa: F401  # Windows DLL initialization order
    import torch
    from roll.alpha360_cross_stock import Alpha360CrossStockTransformer, Alpha360TransformerConfig, distribution_report

    torch.set_num_threads(2)

    checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
    config = Alpha360TransformerConfig(**checkpoint["configuration"]["model"])
    model = Alpha360CrossStockTransformer(store.manifest["stock_count"], config)
    initial_identity = model.stock_identity.weight.detach().clone()
    model.load_state_dict(checkpoint["model"])
    torch.testing.assert_close(model.stock_identity.weight, initial_identity, rtol=0, atol=0)
    model.to(device).eval()
    dtype = torch.bfloat16 if checkpoint["configuration"]["autocast_dtype"] == "torch.bfloat16" else torch.float16
    results = []
    for split, predictions in predictions_by_split.items():
        batch = next(store.iterate(split))
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            output = model(torch.from_numpy(batch["features"])[None].to(device),
                           torch.from_numpy(batch["stock_ids"].astype("int64"))[None].to(device))
        report = distribution_report(output["horizon_mean"][0] / config.target_scale,
                                     output["horizon_covariance"][0] / config.target_scale**2)
        saved = predictions.xs(pd.Timestamp(batch["date"]), level="datetime")
        codes = [store.id_to_code[int(idx)] for idx in batch["stock_ids"]]
        error = 0.0
        for column, name in enumerate(NAMES):
            replayed = report["expected_return"][:, column].float().cpu().numpy()
            expected = saved.loc[codes, name + "_expected_return"].to_numpy()
            tolerance = 2e-6 if device == "cuda" else 2e-4
            np.testing.assert_allclose(replayed, expected, rtol=0.01, atol=tolerance)
            error = max(error, float(np.abs(replayed - expected).max()))
        results.append({"split": split, "date": batch["date"], "stocks": len(codes), "max_return_error": error})
    return {"best_epoch": checkpoint["epoch"], "frozen_identity_exact_match": True, "replayed_dates": results}


def run_audit(args):
    state = json.loads((args.run / "status.json").read_text(encoding="utf-8-sig"))
    if state.get("status") != "completed":
        raise RuntimeError("Training is not complete; refusing to evaluate or report Test prematurely")
    store = DateStore(args.data)
    configuration = json.loads((args.run / "configuration.json").read_text())
    if configuration["data_manifest_sha256"] != file_hash(args.data / "manifest.json"):
        raise AssertionError("Run refers to a different dataset")
    epochs = pd.read_csv(args.run / "epoch_metrics.csv")
    if not np.isfinite(epochs[["train_nll", "nll_scaled_3leg"]].values).all():
        raise AssertionError("Nonfinite epoch loss")
    best_epoch = int(epochs.loc[epochs["nll_scaled_3leg"].idxmin(), "epoch"])
    if best_epoch != state["best_epoch"]:
        raise AssertionError("Checkpoint is not the best early-stopping Valid epoch")
    baselines = {"all": training_baseline(store), "mainboard": training_baseline(store, mainboard_only=True)}
    summaries, daily, calibration, nll_rows = [], [], [], []
    original_summary = pd.read_csv(args.run / "summary.csv").set_index("split")
    prediction_sets = {}
    audit_rows = []
    for split in ("valid", "selection_valid", "test"):
        path = args.run / f"{split}_predictions.csv"
        frame = pd.read_csv(path, parse_dates=["datetime"]).set_index(["datetime", "instrument"]).sort_index()
        validate_probability_columns(frame)
        labels = validate_rows_and_labels(store, split, frame)
        rows, daily_rows, calibration_rows = statistical_metrics(frame, split, baselines["all"])
        for row in rows:
            name = row["horizon"]
            np.testing.assert_allclose(row["rank_ic"], original_summary.loc[split, name + "_rank_ic"], atol=1e-7, rtol=1e-5)
            np.testing.assert_allclose(row["brier"], original_summary.loc[split, name + "_brier"], atol=1e-7, rtol=1e-5)
        for group in (rows, daily_rows, calibration_rows):
            for row in group:
                row["universe"] = "all"
        summaries.extend(rows)
        daily.extend(daily_rows)
        calibration.extend(calibration_rows)
        eligible = frame.index.get_level_values("instrument").map(mainboard_code)
        board_rows, board_daily, board_calibration = statistical_metrics(frame.loc[eligible], split, baselines["mainboard"])
        for group in (board_rows, board_daily, board_calibration):
            for row in group:
                row["universe"] = "mainboard"
        summaries.extend(board_rows)
        daily.extend(board_daily)
        calibration.extend(board_calibration)
        nll_rows.append({"split": split,
                         "model_nll_scaled_3leg": float(original_summary.loc[split, "nll_scaled_3leg"]),
                         "train_gaussian_nll_scaled_3leg": gaussian_baseline_nll(labels, baselines["all"])})
        prediction_sets[split] = frame
        audit_rows.append({"split": split, "rows": len(frame), "dates": frame.index.get_level_values("datetime").nunique(),
                           "sha256": file_hash(path), "probabilities_and_labels_verified": True})
    replay = replay_checkpoints(args.run, store, prediction_sets, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(args.output / "metrics_compact.csv", index=False)
    pd.DataFrame(daily).to_csv(args.output / "daily_recomputed.csv", index=False)
    pd.DataFrame(calibration).to_csv(args.output / "probability_calibration.csv", index=False)
    pd.DataFrame(nll_rows).to_csv(args.output / "nll_baseline.csv", index=False)
    write_json(args.output / "training_baseline.json", baselines)
    write_json(args.output / "audit.json", {"status": "passed", "data_manifest_sha256": configuration["data_manifest_sha256"],
                                           "prediction_files": audit_rows, "checkpoint_replay": replay,
                                           "interpretation": "statistical evaluation only; not executable backtest PnL"})
    lines = ["# Alpha360 截面 Transformer：完成核验与预测评估", "", 
             "这是预测与概率分布评估，不是实际可成交的交易回测。不会用 Test 结果倒推修改模型。", "",
             "项目此前已反复查看过该历史 Test 区间，因此这是冻结规则的历史对照，不是整个研究从未接触过的新样本外证据。", "",
             f"- 最优 epoch：{best_epoch}；实际完成：{len(epochs)} epoch。",
             f"- 完整 epoch 平均耗时：{epochs['epoch_seconds'].mean():.1f} 秒；训练及最终评估：{state['elapsed_seconds']/3600:.2f} 小时。",
             f"- 检查点重放、冻结股票 embedding、四区间代数关系和真实标签一致性检查全部通过。", "",
             "## Test 指标", "", "| 股票范围 | 持仓区间 | Rank IC | Rank ICIR | MAE | Brier | 历史上涨率基线 Brier | 95%区间覆盖率 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["split"] == "test":
            lines.append(f"| {row['universe']} | {row['horizon']} | {row['rank_ic']:.4f} | {row['rank_icir']:.4f} | {row['mae']:.2%} | {row['brier']:.4f} | {row['train_rate_brier']:.4f} | {row['coverage95']:.2%} |")
    lines.extend(["", "## 指标怎么读", "",
                  "- Rank IC 看每天的横截面排序能力，Rank ICIR 是这些日度相关系数的均值/标准差，不是胜率。",
                  "- MAE 是绝对收益预测的平均误差；例如 0.02 表示平均相差约 2 个百分点。",
                  "- Brier 越小越好；只有比 Train 历史上涨比例常数基线更小，才体现该项预测增益。",
                  "- 95% 预测区间若只覆盖很少真实收益，说明不确定性被低估；输出高概率不能当作真实高胜率。",
                  "- NLL 比较在相同的 log-return×100 尺度上；不同尺度的 NLL 不能直接比数值。",
                  "- 基线仅使用 Train 拟合。Valid 用于早停，Selection-valid 和 Test 不参加训练更新。",
                  "- all 是历史 CSI1000 全池；mainboard 只评估其中沪深主板，排除科创板和创业板。两种范围分别拟合各自的 Train 常数基线。",
                  "- 本报告没有计入手续费、涨跌停、停牌、交易门槛或 event_guard，不能当作实盘收益。", "",
                  "## 文件", "", "- metrics_compact.csv：四种区间在三个分段上的完整统计指标。",
                  "- probability_calibration.csv：预测概率分箱与真实上涨比例。",
                  "- nll_baseline.csv：模型联合 NLL 与 Train 常数高斯基线。",
                  "- audit.json：文件覆盖、标签、概率、检查点与身份向量核验。", ""])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": "passed", "output": str(args.output), "best_epoch": best_epoch}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run_audit(parser.parse_args())
