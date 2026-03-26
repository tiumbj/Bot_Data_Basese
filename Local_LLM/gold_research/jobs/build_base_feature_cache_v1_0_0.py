
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_base_feature_cache_v1_0_0.py
Version: v1.0.0

Create reusable feature cache from OHLC parquet files.
Features:
- returns
- EMA 10/20/50/100
- ATR 14
- ADX 14
- price location vs EMA20/50
- simple trend/volatility labels
- market structure helper columns
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "v1.0.0"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for target in ["time", "open", "high", "low", "close", "tick_volume", "volume"]:
        if target in cols:
            rename_map[cols[target]] = target
    df = df.rename(columns=rename_map)
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    df = df[["time", "open", "high", "low", "close"] + [c for c in ["tick_volume", "volume"] if c in df.columns]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df = df.sort_values("time").reset_index(drop=True)
    return df


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
    out["close"] = df["close"]
    out["ret_1"] = df["close"].pct_change()
    out["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    out["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    out["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
    out["atr_14"] = compute_atr(df, 14)
    out["atr_pct_14"] = out["atr_14"] / df["close"].replace(0.0, np.nan)
    out["adx_14"] = compute_adx(df, 14)
    out["above_ema20"] = (df["close"] > out["ema_20"]).astype(np.int8)
    out["above_ema50"] = (df["close"] > out["ema_50"]).astype(np.int8)
    out["ema_stack_bull"] = ((out["ema_10"] > out["ema_20"]) & (out["ema_20"] > out["ema_50"])).astype(np.int8)
    out["ema_stack_bear"] = ((out["ema_10"] < out["ema_20"]) & (out["ema_20"] < out["ema_50"])).astype(np.int8)
    out["swing_high_3"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    out["swing_low_3"] = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
    out["pullback_to_ema20_long"] = ((df["low"] <= out["ema_20"]) & (df["close"] >= out["ema_20"])).astype(np.int8)
    out["pullback_to_ema20_short"] = ((df["high"] >= out["ema_20"]) & (df["close"] <= out["ema_20"])).astype(np.int8)
    out["vol_bucket"] = label_volatility(out["atr_pct_14"].fillna(0.0))
    out["trend_bucket"] = label_trend(out["adx_14"].fillna(0.0), out["ema_20"], out["ema_50"])
    out["price_location_bucket"] = np.where(
        df["close"] > out["ema_50"], "ABOVE_EMA_STACK", "BELOW_EMA_STACK"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base feature cache")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--timeframes", default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--resume", action="store_true", default=True)
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
        if feat_path.exists():
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
