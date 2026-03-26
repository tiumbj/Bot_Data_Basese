#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_base_feature_cache_v1_0_1.py
Version: v1.0.1

Create reusable feature cache from OHLC parquet files.
Production fixes:
- robust OHLC schema normalization
- robust time/index normalization
- add pullback_to_ema20_long / pullback_to_ema20_short as native upstream features
- keep downstream-compatible columns for pending logic
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "v1.0.1"
TIME_ALIASES = {"time", "datetime", "timestamp", "date", "dt"}
OPEN_ALIASES = {"open", "<open>", "open_price", "o"}
HIGH_ALIASES = {"high", "<high>", "high_price", "h"}
LOW_ALIASES = {"low", "<low>", "low_price", "l"}
CLOSE_ALIASES = {"close", "<close>", "close_price", "c"}
VOLUME_ALIASES = {"tick_volume", "volume", "<tick_volume>", "<volume>"}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def flatten_col_name(col: object) -> str:
    if isinstance(col, tuple):
        parts = [str(x) for x in col if str(x).strip() not in {"", "None"}]
        raw = "_".join(parts)
    else:
        raw = str(col)

    raw = raw.strip()
    raw = raw.replace("\n", "_").replace(" ", "_").replace("-", "_").replace("/", "_")
    raw = re.sub(r"[()]+", "", raw)
    raw = re.sub(r"__+", "_", raw)
    return raw.strip("_")


def normalize_name(name: object) -> str:
    return flatten_col_name(name).lower()


def bring_index_to_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    normalized_cols = [normalize_name(c) for c in out.columns]
    if any(c in TIME_ALIASES for c in normalized_cols):
        return out

    idx_name = normalize_name(out.index.name) if out.index.name is not None else ""
    if idx_name in TIME_ALIASES:
        return out.reset_index()

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "time"})
        return out

    return out


def canonicalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = bring_index_to_column_if_needed(df).copy()
    out.columns = [flatten_col_name(c) for c in out.columns]

    rename_map = {}
    for col in out.columns:
        norm = normalize_name(col)
        if norm in TIME_ALIASES and "time" not in out.columns:
            rename_map[col] = "time"
        elif norm in OPEN_ALIASES and "open" not in out.columns:
            rename_map[col] = "open"
        elif norm in HIGH_ALIASES and "high" not in out.columns:
            rename_map[col] = "high"
        elif norm in LOW_ALIASES and "low" not in out.columns:
            rename_map[col] = "low"
        elif norm in CLOSE_ALIASES and "close" not in out.columns:
            rename_map[col] = "close"
        elif norm in VOLUME_ALIASES and "tick_volume" not in out.columns:
            rename_map[col] = "tick_volume"

    out = out.rename(columns=rename_map)

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}; available={list(out.columns)}")

    keep = ["time", "open", "high", "low", "close"] + [c for c in ["tick_volume"] if c in out.columns]
    out = out[keep].copy()

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return canonicalize_price_columns(df)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr_components = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    tr_components = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    tr = tr_components.max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def label_volatility(atr_pct: pd.Series) -> pd.Series:
    q1 = atr_pct.quantile(0.33)
    q2 = atr_pct.quantile(0.66)
    labels = np.where(
        atr_pct <= q1, "LOW_VOL",
        np.where(atr_pct <= q2, "MID_VOL", "HIGH_VOL")
    )
    return pd.Series(labels, index=atr_pct.index)


def label_trend(adx: pd.Series, ema20: pd.Series, ema50: pd.Series) -> pd.Series:
    bullish = ema20 > ema50
    labels = np.where(
        adx >= 25,
        np.where(bullish, "BULL_TREND", "BEAR_TREND"),
        np.where(bullish, "WEAK_BULL", "WEAK_BEAR")
    )
    return pd.Series(labels, index=adx.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["time"] = df["time"]
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]

    out["ret_1"] = df["close"].pct_change()

    # เก็บทั้งชุดเดิมและชุดที่ downstream บางไฟล์อาจอ้างอิง
    out["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    out["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    out["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    out["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
    out["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    out["atr_14"] = compute_atr(df, 14)
    out["atr_pct_14"] = out["atr_14"] / df["close"].replace(0.0, np.nan)
    out["adx_14"] = compute_adx(df, 14)

    out["above_ema20"] = (df["close"] > out["ema_20"]).astype(np.int8)
    out["above_ema50"] = (df["close"] > out["ema_50"]).astype(np.int8)

    out["ema_stack_bull"] = ((out["ema_10"] > out["ema_20"]) & (out["ema_20"] > out["ema_50"])).astype(np.int8)
    out["ema_stack_bear"] = ((out["ema_10"] < out["ema_20"]) & (out["ema_20"] < out["ema_50"])).astype(np.int8)

    out["swing_high_3"] = ((df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))).astype(np.int8)
    out["swing_low_3"] = ((df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))).astype(np.int8)

    # แก้ต้นตอ: สร้างฟีเจอร์ที่ pending ใช้โดยตรงจาก upstream
    out["pullback_to_ema20_long"] = ((df["low"] <= out["ema_20"]) & (df["close"] >= out["ema_20"])).astype(np.int8)
    out["pullback_to_ema20_short"] = ((df["high"] >= out["ema_20"]) & (df["close"] <= out["ema_20"])).astype(np.int8)

    out["vol_bucket"] = label_volatility(out["atr_pct_14"].fillna(0.0))
    out["trend_bucket"] = label_trend(out["adx_14"].fillna(0.0), out["ema_20"], out["ema_50"])
    out["price_location_bucket"] = np.where(df["close"] > out["ema_50"], "ABOVE_EMA_STACK", "BELOW_EMA_STACK")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base feature cache")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache"))
    parser.add_argument("--timeframes", default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    start_all = time.time()

    for tf in timeframes:
        price_path = args.data_root / f"{args.symbol}_{tf}.parquet"
        feat_path = outdir / f"{args.symbol}_{tf}_base_features.parquet"
        state_path = outdir / "state" / f"{args.symbol}_{tf}_feature_state.json"

        if feat_path.exists() and args.resume and not args.force:
            state = {
                "version": VERSION,
                "timeframe": tf,
                "status": "DONE",
                "updated_at_utc": now_utc_iso(),
                "feature_path": str(feat_path),
                "rows": int(pd.read_parquet(feat_path, columns=["time"]).shape[0]),
            }
            save_json(state_path, state)
            summary_rows.append(state)
            print(f"[SKIP] timeframe={tf} feature_path={feat_path}")
            continue

        if not price_path.exists():
            state = {
                "version": VERSION,
                "timeframe": tf,
                "status": "MISSING_DATA",
                "updated_at_utc": now_utc_iso(),
                "price_path": str(price_path),
            }
            save_json(state_path, state)
            summary_rows.append(state)
            print(f"[MISS] timeframe={tf} price_path={price_path}")
            continue

        t0 = time.time()
        try:
            price_df = read_price_parquet(price_path)
            feat_df = build_features(price_df)
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            feat_df.to_parquet(feat_path, index=False)

            elapsed = time.time() - t0
            state = {
                "version": VERSION,
                "timeframe": tf,
                "status": "DONE",
                "updated_at_utc": now_utc_iso(),
                "price_path": str(price_path),
                "feature_path": str(feat_path),
                "rows": int(len(feat_df)),
                "elapsed_sec": round(elapsed, 4),
            }
            save_json(state_path, state)
            summary_rows.append(state)
            print(f"[DONE] timeframe={tf} rows={len(feat_df)} feature_path={feat_path} elapsed_sec={elapsed:.4f}")
        except Exception as exc:
            elapsed = time.time() - t0
            state = {
                "version": VERSION,
                "timeframe": tf,
                "status": "ERROR",
                "updated_at_utc": now_utc_iso(),
                "price_path": str(price_path),
                "feature_path": str(feat_path),
                "elapsed_sec": round(elapsed, 4),
                "error": str(exc),
            }
            save_json(state_path, state)
            summary_rows.append(state)
            print(f"[ERROR] timeframe={tf} error={exc}")

    summary = {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "outdir": str(outdir),
        "timeframes": timeframes,
        "elapsed_sec_total": round(time.time() - start_all, 4),
        "rows": summary_rows,
    }
    save_json(outdir / "summary.json", summary)
    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] summary_json={outdir / 'summary.json'}")
    print("=" * 120)


if __name__ == "__main__":
    main()