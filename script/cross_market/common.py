#!/usr/bin/env python3
"""Shared paths, data normalization, and Alpha158-compatible factor generation."""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / ".qlibAssistant" / "cross_market"
DATA_STORE = Path(
    os.environ.get(
        "CROSS_MARKET_DATA_ROOT",
        str(ROOT.parent / "investment_data" / "cross_market_daily"),
    )
).expanduser()
RAW = DATA_STORE / "raw"
FACTORS = DATA_STORE / "factors"
REPORTS = STORE / "reports"
MODELS = STORE / "models"
LOGS = STORE / "logs"
UNIVERSES = DATA_STORE / "universes"
WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12


def ensure_dirs() -> None:
    for path in (RAW, FACTORS, REPORTS, MODELS, LOGS, UNIVERSES):
        path.mkdir(parents=True, exist_ok=True)


def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(symbol))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted OHLCV with consistent columns.

    Yahoo's ``adjclose`` is used to adjust prices and split-adjust volume.
    Existing Qlib A-share prices are already adjusted and therefore use factor 1.
    VWAP is deliberately approximated from OHLC for every market, avoiding a
    market-specific definition mismatch.
    """
    df = frame.copy()
    df.columns = [str(c).lower().lstrip("$") for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume", "adjclose"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    factor = (
        df["adjclose"].div(df["close"]).replace([np.inf, -np.inf], np.nan).ffill()
        if "adjclose" in df
        else pd.Series(1.0, index=df.index)
    )
    factor = factor.fillna(1.0)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    df["volume"] = df["volume"].div(factor.replace(0, np.nan))
    df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    invalid = (
        (df["close"] <= 0)
        | (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["volume"] <= 0)
    )
    df.loc[invalid, ["open", "high", "low", "close", "volume", "vwap"]] = np.nan
    return df[["date", "open", "high", "low", "close", "volume", "vwap"]].reset_index(drop=True)


def _rolling_linear(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorized rolling slope, R² and last-point residual."""
    y = series.astype(float)
    x = np.arange(window, dtype=float)
    sx = x.sum()
    sx2 = np.square(x).sum()
    denom = window * sx2 - sx * sx
    sy = y.rolling(window).sum()
    sy2 = y.pow(2).rolling(window).sum()
    xy = y.rolling(window).apply(lambda values: float(np.dot(values, x)), raw=True)
    slope = (window * xy - sx * sy) / denom
    intercept = (sy - slope * sx) / window
    fitted_last = intercept + slope * (window - 1)
    residual = y - fitted_last
    corr_denom = np.sqrt((window * sx2 - sx * sx) * (window * sy2 - sy * sy))
    corr = (window * xy - sx * sy) / corr_denom.replace(0, np.nan)
    return slope, corr.pow(2), residual


def _last_rank(values: np.ndarray) -> float:
    if not np.isfinite(values[-1]):
        return np.nan
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan
    return float((valid <= values[-1]).sum() / len(valid))


def _idxmax(values: np.ndarray) -> float:
    return float(np.nanargmax(values) + 1) if np.isfinite(values).any() else np.nan


def _idxmin(values: np.ndarray) -> float:
    return float(np.nanargmin(values) + 1) if np.isfinite(values).any() else np.nan


def alpha158_frame(frame: pd.DataFrame, symbol: str, market: str) -> pd.DataFrame:
    """Compute the official Alpha158 field layout from adjusted daily OHLCV.

    The formulas mirror Qlib Alpha158. The sole source adaptation is VWAP:
    Yahoo does not expose historical daily turnover amount consistently, so
    ``VWAP0`` uses OHLC4 for all three markets.
    """
    raw = normalize_ohlcv(frame)
    o, h, l, c, v, vw = (raw[k] for k in ("open", "high", "low", "close", "volume", "vwap"))
    out = pd.DataFrame({"date": raw["date"]})
    out["symbol"] = symbol
    out["market"] = market
    spread = h - l
    out["KMID"] = (c - o) / o
    out["KLEN"] = spread / o
    out["KMID2"] = (c - o) / (spread + EPS)
    out["KUP"] = (h - np.maximum(o, c)) / o
    out["KUP2"] = (h - np.maximum(o, c)) / (spread + EPS)
    out["KLOW"] = (np.minimum(o, c) - l) / o
    out["KLOW2"] = (np.minimum(o, c) - l) / (spread + EPS)
    out["KSFT"] = (2 * c - h - l) / o
    out["KSFT2"] = (2 * c - h - l) / (spread + EPS)
    out["OPEN0"] = o / c
    out["HIGH0"] = h / c
    out["LOW0"] = l / c
    out["VWAP0"] = vw / c

    logv = np.log(v + 1)
    price_ratio = c / c.shift(1)
    volume_ratio_log = np.log(v / v.shift(1) + 1)
    price_delta = c - c.shift(1)
    volume_delta = v - v.shift(1)
    price_abs = price_delta.abs()
    volume_abs = volume_delta.abs()
    weighted_move = (price_ratio - 1).abs() * v

    for w in WINDOWS:
        out[f"ROC{w}"] = c.shift(w) / c
        out[f"MA{w}"] = c.rolling(w).mean() / c
        out[f"STD{w}"] = c.rolling(w).std() / c
        slope, rsquare, residual = _rolling_linear(c, w)
        out[f"BETA{w}"] = slope / c
        out[f"RSQR{w}"] = rsquare
        out[f"RESI{w}"] = residual / c
        rolling_high = h.rolling(w).max()
        rolling_low = l.rolling(w).min()
        out[f"MAX{w}"] = rolling_high / c
        out[f"MIN{w}"] = rolling_low / c
        out[f"QTLU{w}"] = c.rolling(w).quantile(0.8) / c
        out[f"QTLD{w}"] = c.rolling(w).quantile(0.2) / c
        out[f"RANK{w}"] = c.rolling(w).apply(_last_rank, raw=True)
        out[f"RSV{w}"] = (c - rolling_low) / (rolling_high - rolling_low + EPS)
        imax = h.rolling(w).apply(_idxmax, raw=True)
        imin = l.rolling(w).apply(_idxmin, raw=True)
        out[f"IMAX{w}"] = imax / w
        out[f"IMIN{w}"] = imin / w
        out[f"IMXD{w}"] = (imax - imin) / w
        out[f"CORR{w}"] = c.rolling(w).corr(logv)
        out[f"CORD{w}"] = price_ratio.rolling(w).corr(volume_ratio_log)
        positive = (price_delta > 0).astype(float)
        negative = (price_delta < 0).astype(float)
        out[f"CNTP{w}"] = positive.rolling(w).mean()
        out[f"CNTN{w}"] = negative.rolling(w).mean()
        out[f"CNTD{w}"] = out[f"CNTP{w}"] - out[f"CNTN{w}"]
        pos_sum = price_delta.clip(lower=0).rolling(w).sum()
        neg_sum = (-price_delta).clip(lower=0).rolling(w).sum()
        abs_sum = price_abs.rolling(w).sum() + EPS
        out[f"SUMP{w}"] = pos_sum / abs_sum
        out[f"SUMN{w}"] = neg_sum / abs_sum
        out[f"SUMD{w}"] = (pos_sum - neg_sum) / abs_sum
        out[f"VMA{w}"] = v.rolling(w).mean() / (v + EPS)
        out[f"VSTD{w}"] = v.rolling(w).std() / (v + EPS)
        out[f"WVMA{w}"] = weighted_move.rolling(w).std() / (weighted_move.rolling(w).mean() + EPS)
        vpos = volume_delta.clip(lower=0).rolling(w).sum()
        vneg = (-volume_delta).clip(lower=0).rolling(w).sum()
        vabs = volume_abs.rolling(w).sum() + EPS
        out[f"VSUMP{w}"] = vpos / vabs
        out[f"VSUMN{w}"] = vneg / vabs
        out[f"VSUMD{w}"] = (vpos - vneg) / vabs

    # Keep both the project's close(T+1)->close(T+2) target and absolute prices
    # needed for transparent evaluation.
    out["label_abs"] = c.shift(-2) / c.shift(-1) - 1
    out["close"] = c
    out["next_buy_close"] = c.shift(-1)
    out["next_sell_close"] = c.shift(-2)
    out = out.replace([np.inf, -np.inf], np.nan).copy()
    return out


def factor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "symbol",
        "market",
        "label_abs",
        "label_z",
        "close",
        "next_buy_close",
        "next_sell_close",
    }
    return [
        c for c in frame.columns
        if c not in excluded and not c.startswith("MARKET_")
    ]
