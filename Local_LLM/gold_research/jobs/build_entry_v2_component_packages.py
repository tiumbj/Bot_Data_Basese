# ============================================================
# ชื่อโค้ด: build_entry_v2_component_packages.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_entry_v2_component_packages.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_entry_v2_component_packages.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VERSION = "v1.0.0"
CANONICAL_SYMBOL = "XAUUSD"

ROOT = Path(r"C:\Data\Bot")
MARKET_DATA_ROOT = ROOT / "central_market_data" / "tf"
REGISTRY_ROOT = ROOT / "central_strategy_registry" / "candidates"

BASE_STRATEGY_ID = "locked_ms_bos_choch_pullback_atr_adx_ema"
BASE_VERSION = "v1.1.0"

RESEARCH_TIMEFRAMES = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "M10", "M15", "M30", "H1", "H4", "D1",
]

TF_EXIT_WINNER = {
    "M1": "time_stop",
    "M2": "time_stop",
    "M3": "fast_invalidation",
    "M4": "momentum_fade",
    "M5": "momentum_fade",
    "M6": "time_stop",
    "M10": "time_stop",
    "M15": "momentum_fade",
    "M30": "short_ema_weakness",
    "H1": "fast_invalidation",
    "H4": "fast_invalidation",
    "D1": "base",
}

ENTRY_VARIANTS = {
    "entry_v2_bos_strict": {
        "bos_fresh_bars": 3,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "normal",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
    },
    "entry_v2_choch_confirm": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 2,
        "swing_lookback": 2,
        "pullback_mode": "normal",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
    },
    "entry_v2_swing_lkb3": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 3,
        "pullback_mode": "normal",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
    },
    "entry_v2_pullback_deep": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "deep",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
        "block_short_below_ema_stack": False,
    },
    "entry_v2_pullback_deep_m30_block_short_below_ema": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "deep",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
        "block_short_below_ema_stack": True,
    },
    "entry_v2_adx25": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "normal",
        "adx_threshold": 25.0,
        "ema_gap_min": 0.0,
    },
    "entry_v2_ema_tight": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "normal",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.10,
    },
}


@dataclass
class PackagePaths:
    strategy_id: str
    version: str
    root: Path
    signals_file: Path
    strategy_spec_file: Path
    approved_parameters_file: Path
    validation_report_file: Path

    @classmethod
    def build(cls, strategy_id: str, version: str) -> "PackagePaths":
        root = REGISTRY_ROOT / strategy_id / version
        return cls(
            strategy_id=strategy_id,
            version=version,
            root=root,
            signals_file=root / "signals.csv",
            strategy_spec_file=root / "strategy_spec.json",
            approved_parameters_file=root / "approved_parameters.json",
            validation_report_file=root / "validation_report.json",
        )


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_price_data(tf: str) -> pd.DataFrame:
    file_path = MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{tf}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ราคา | tf={tf} file={file_path}")

    raw = pd.read_csv(file_path)
    cols = list(raw.columns)

    ts_col = find_col(cols, ["timestamp", "datetime", "date_time"])
    if ts_col is None:
        date_col = find_col(cols, ["date"])
        time_col = find_col(cols, ["time"])
        if date_col and time_col:
            ts = raw[date_col].astype(str).str.strip() + " " + raw[time_col].astype(str).str.strip()
            raw["timestamp"] = pd.to_datetime(ts, errors="coerce")
        else:
            raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ | tf={tf} missing=['timestamp']")
    else:
        raw["timestamp"] = pd.to_datetime(raw[ts_col], errors="coerce")

    mapping = {}
    for canonical, candidates in {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
    }.items():
        col = find_col(cols, candidates)
        if col is None:
            raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ | tf={tf} missing=['{canonical}']")
        mapping[canonical] = col

    out = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "open": pd.to_numeric(raw[mapping["open"]], errors="coerce"),
        "high": pd.to_numeric(raw[mapping["high"]], errors="coerce"),
        "low": pd.to_numeric(raw[mapping["low"]], errors="coerce"),
        "close": pd.to_numeric(raw[mapping["close"]], errors="coerce"),
    })
    out = out.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return out


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_rma = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_rma
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_rma

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    return pd.DataFrame({
        "adx14": adx_line.fillna(0.0),
        "di_plus": plus_di.fillna(0.0),
        "di_minus": minus_di.fillna(0.0),
    })


def compute_swings(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = np.nan
    out["swing_low"] = np.nan

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(lookback, len(df) - lookback):
        if highs[i] > highs[i - lookback:i].max() and highs[i] >= highs[i + 1:i + 1 + lookback].max():
            out.iloc[i, 0] = highs[i]
        if lows[i] < lows[i - lookback:i].min() and lows[i] <= lows[i + 1:i + 1 + lookback].min():
            out.iloc[i, 1] = lows[i]

    out["last_swing_high"] = out["swing_high"].ffill()
    out["last_swing_low"] = out["swing_low"].ffill()
    return out


def build_structure(df: pd.DataFrame, swing_lookback: int) -> pd.DataFrame:
    swings = compute_swings(df, swing_lookback)
    out = pd.DataFrame(index=df.index)
    out["last_swing_high"] = swings["last_swing_high"]
    out["last_swing_low"] = swings["last_swing_low"]
    out["bos_up"] = False
    out["bos_down"] = False
    out["choch_up"] = False
    out["choch_down"] = False
    out["bias"] = "NEUTRAL"
    out["bars_since_bos_up"] = 999999
    out["bars_since_bos_down"] = 999999

    current_bias = "NEUTRAL"
    last_bos_up_idx = -999999
    last_bos_down_idx = -999999

    for i in range(1, len(df)):
        close_i = float(df["close"].iloc[i])
        prev_high = out["last_swing_high"].iloc[i - 1]
        prev_low = out["last_swing_low"].iloc[i - 1]

        bos_up = pd.notna(prev_high) and close_i > float(prev_high)
        bos_down = pd.notna(prev_low) and close_i < float(prev_low)

        choch_up = bos_up and current_bias == "BEARISH"
        choch_down = bos_down and current_bias == "BULLISH"

        if bos_up:
            current_bias = "BULLISH"
            last_bos_up_idx = i
        elif bos_down:
            current_bias = "BEARISH"
            last_bos_down_idx = i

        out.at[i, "bos_up"] = bool(bos_up)
        out.at[i, "bos_down"] = bool(bos_down)
        out.at[i, "choch_up"] = bool(choch_up)
        out.at[i, "choch_down"] = bool(choch_down)
        out.at[i, "bias"] = current_bias
        out.at[i, "bars_since_bos_up"] = i - last_bos_up_idx
        out.at[i, "bars_since_bos_down"] = i - last_bos_down_idx

    out["bias"] = out["bias"].replace("", "NEUTRAL").ffill().fillna("NEUTRAL")
    return out


def apply_exit_overlay(work: pd.DataFrame, tf: str, exit_variant: str) -> pd.DataFrame:
    work["ema5"] = ema(work["close"], 5)
    work["ema10"] = ema(work["close"], 10)
    work["atr14"] = work["atr14"].ffill()
    adx_slope = work["adx14"] - work["adx14"].shift(1)
    body = (work["close"] - work["open"]).abs()
    range_ = (work["high"] - work["low"]).replace(0, np.nan)
    body_ratio = (body / range_).fillna(0.0)

    if exit_variant == "base":
        return work

    if exit_variant == "fast_invalidation":
        work["exit_long"] = work["exit_long"] | (work["close"] < work["low"].shift(1))
        work["exit_short"] = work["exit_short"] | (work["close"] > work["high"].shift(1))

    elif exit_variant == "momentum_fade":
        weak_long = (adx_slope < 0) & (body_ratio < 0.35) & (work["di_plus"] < work["di_plus"].shift(1))
        weak_short = (adx_slope < 0) & (body_ratio < 0.35) & (work["di_minus"] < work["di_minus"].shift(1))
        work["exit_long"] = work["exit_long"] | weak_long
        work["exit_short"] = work["exit_short"] | weak_short

    elif exit_variant == "short_ema_weakness":
        work["exit_long"] = work["exit_long"] | ((work["ema5"] < work["ema10"]) & (work["close"] < work["ema5"]))
        work["exit_short"] = work["exit_short"] | ((work["ema5"] > work["ema10"]) & (work["close"] > work["ema5"]))

    elif exit_variant == "time_stop":
        # runner ปัจจุบันยังไม่มี bars-in-trade state จึง approximate โดยออกเร็วเมื่อ momentum ไม่ไปต่อภายในโครงสั้น
        work["exit_long"] = work["exit_long"] | (
            (work["bias"] == "BULLISH") &
            (work["close"] <= work["close"].rolling(8, min_periods=1).max().shift(1)) &
            (work["close"] < work["ema20"])
        )
        work["exit_short"] = work["exit_short"] | (
            (work["bias"] == "BEARISH") &
            (work["close"] >= work["close"].rolling(8, min_periods=1).min().shift(1)) &
            (work["close"] > work["ema20"])
        )

    return work


def build_signals_for_variant(df: pd.DataFrame, tf: str, params: dict) -> pd.DataFrame:
    work = df.copy()
    work["ema20"] = ema(work["close"], 20)
    work["ema50"] = ema(work["close"], 50)
    work["atr14"] = atr(work, 14)

    adx_df = adx(work, 14)
    work["adx14"] = adx_df["adx14"]
    work["di_plus"] = adx_df["di_plus"]
    work["di_minus"] = adx_df["di_minus"]

    ms = build_structure(work, params["swing_lookback"])
    for col in ms.columns:
        work[col] = ms[col]

    if params["pullback_mode"] == "deep":
        bullish_pullback = (
            (work["bias"] == "BULLISH") &
            (work["low"] <= work["ema20"]) &
            (work["close"] >= ((work["ema20"] + work["ema50"]) / 2.0)) &
            (work["ema20"] > work["ema50"])
        )
        bearish_pullback = (
            (work["bias"] == "BEARISH") &
            (work["high"] >= work["ema20"]) &
            (work["close"] >= work["ema20"]) &
            (work["close"] <= ((work["ema20"] + work["ema50"]) / 2.0)) &
            (work["ema20"] < work["ema50"])
        )
    else:
        bullish_pullback = (
            (work["bias"] == "BULLISH") &
            (work["low"] <= work["ema20"]) &
            (work["close"] >= work["ema20"]) &
            (work["ema20"] > work["ema50"])
        )
        bearish_pullback = (
            (work["bias"] == "BEARISH") &
            (work["high"] >= work["ema20"]) &
            (work["close"] <= work["ema20"]) &
            (work["ema20"] < work["ema50"])
        )

    work["price_location_bucket"] = "INSIDE_EMA_ZONE"
    work.loc[
        (work["close"] > work["ema20"]) & (work["close"] > work["ema50"]),
        "price_location_bucket"
    ] = "ABOVE_EMA_STACK"
    work.loc[
        (work["close"] < work["ema20"]) & (work["close"] < work["ema50"]),
        "price_location_bucket"
    ] = "BELOW_EMA_STACK"

    ema_gap = (work["ema20"] - work["ema50"]).abs()

    bull_filter_ok = (
        (work["atr14"] > 0) &
        (work["adx14"] >= params["adx_threshold"]) &
        (work["ema20"] > work["ema50"]) &
        (work["di_plus"] > work["di_minus"]) &
        (ema_gap >= params["ema_gap_min"])
    )

    bear_filter_ok = (
        (work["atr14"] > 0) &
        (work["adx14"] >= params["adx_threshold"]) &
        (work["ema20"] < work["ema50"]) &
        (work["di_minus"] > work["di_plus"]) &
        (ema_gap >= params["ema_gap_min"])
    )

    choch_long_ok = (
        work["choch_up"].rolling(params["choch_confirm_bars"], min_periods=1).max().astype(bool)
        if params["choch_confirm_bars"] > 1 else True
    )
    choch_short_ok = (
        work["choch_down"].rolling(params["choch_confirm_bars"], min_periods=1).max().astype(bool)
        if params["choch_confirm_bars"] > 1 else True
    )

    fresh_bos_long = work["bars_since_bos_up"] <= params["bos_fresh_bars"]
    fresh_bos_short = work["bars_since_bos_down"] <= params["bos_fresh_bars"]

    work["long_entry"] = bullish_pullback & bull_filter_ok & fresh_bos_long & choch_long_ok
    work["short_entry"] = bearish_pullback & bear_filter_ok & fresh_bos_short & choch_short_ok

    if bool(params.get("block_short_below_ema_stack", False)):
        work["short_entry"] = work["short_entry"] & (work["price_location_bucket"] != "BELOW_EMA_STACK")

    work["exit_long"] = work["choch_down"] | (work["close"] < work["ema50"])
    work["exit_short"] = work["choch_up"] | (work["close"] > work["ema50"])

    work["sl_price"] = np.nan
    work["tp_price"] = np.nan

    long_mask = work["long_entry"] & work["last_swing_low"].notna() & work["atr14"].notna()
    short_mask = work["short_entry"] & work["last_swing_high"].notna() & work["atr14"].notna()

    work.loc[long_mask, "sl_price"] = work.loc[long_mask, "last_swing_low"] - work.loc[long_mask, "atr14"] * 0.25
    long_risk = work["close"] - work["sl_price"]
    work.loc[long_mask, "tp_price"] = work.loc[long_mask, "close"] + (long_risk.loc[long_mask] * 2.0)

    work.loc[short_mask, "sl_price"] = work.loc[short_mask, "last_swing_high"] + work.loc[short_mask, "atr14"] * 0.25
    short_risk = work["sl_price"] - work["close"]
    work.loc[short_mask, "tp_price"] = work.loc[short_mask, "close"] - (short_risk.loc[short_mask] * 2.0)

    work = apply_exit_overlay(work, tf, TF_EXIT_WINNER[tf])

    ready = (
        work["ema20"].notna() &
        work["ema50"].notna() &
        work["atr14"].notna() &
        work["adx14"].notna()
    )

    out = pd.DataFrame({
        "timestamp": work.loc[ready, "timestamp"],
        "symbol": CANONICAL_SYMBOL,
        "timeframe": tf,
        "long_entry": work.loc[ready, "long_entry"].fillna(False).astype(bool),
        "short_entry": work.loc[ready, "short_entry"].fillna(False).astype(bool),
        "price_location_bucket": work.loc[ready, "price_location_bucket"].fillna("UNKNOWN").astype(str),
        "exit_long": work.loc[ready, "exit_long"].fillna(False).astype(bool),
        "exit_short": work.loc[ready, "exit_short"].fillna(False).astype(bool),
        "sl_price": pd.to_numeric(work.loc[ready, "sl_price"], errors="coerce"),
        "tp_price": pd.to_numeric(work.loc[ready, "tp_price"], errors="coerce"),
    }).reset_index(drop=True)

    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_strategy_spec(strategy_id: str, variant_name: str, params: dict) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": VERSION,
        "status": "candidate",
        "canonical_symbol": CANONICAL_SYMBOL,
        "research_backtest_timeframes": RESEARCH_TIMEFRAMES,
        "deployment_entry_timeframes": ["M1", "M5", "M10", "M15", "H1", "H4"],
        "parent_strategy_id": BASE_STRATEGY_ID,
        "parent_version": BASE_VERSION,
        "entry_variant": variant_name,
        "tf_exit_winner_mapping": TF_EXIT_WINNER,
        "entry_parameters": params,
        "concept": "Entry v2 component research paired with TF-specific exit winner",
    }


def build_validation_report(strategy_id: str, variant_name: str, summaries: List[dict]) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "candidate_built_not_backtested_yet",
        "entry_variant": variant_name,
        "tf_exit_winner_mapping": TF_EXIT_WINNER,
        "signal_summary": summaries,
    }


def main() -> None:
    for variant_name, params in ENTRY_VARIANTS.items():
        strategy_id = f"{BASE_STRATEGY_ID}_{variant_name}"
        pkg = PackagePaths.build(strategy_id, VERSION)
        pkg.root.mkdir(parents=True, exist_ok=True)

        all_signals = []
        summaries = []

        for tf in RESEARCH_TIMEFRAMES:
            price_df = load_price_data(tf)
            signals_df = build_signals_for_variant(price_df, tf, params)
            all_signals.append(signals_df)

            summaries.append({
                "timeframe": tf,
                "rows": int(len(signals_df)),
                "long_entries": int(signals_df["long_entry"].sum()),
                "short_entries": int(signals_df["short_entry"].sum()),
                "exit_winner": TF_EXIT_WINNER[tf],
                "first_timestamp": signals_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if not signals_df.empty else "",
                "last_timestamp": signals_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if not signals_df.empty else "",
            })

            print(
                f"[DONE] variant={variant_name} tf={tf} rows={len(signals_df)} "
                f"long_entries={int(signals_df['long_entry'].sum())} "
                f"short_entries={int(signals_df['short_entry'].sum())} "
                f"exit={TF_EXIT_WINNER[tf]}"
            )

        combined = pd.concat(all_signals, axis=0, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp"]).sort_values(["timeframe", "timestamp"]).reset_index(drop=True)
        combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        combined.to_csv(pkg.signals_file, index=False)

        write_json(pkg.strategy_spec_file, build_strategy_spec(strategy_id, variant_name, params))
        write_json(pkg.approved_parameters_file, {
            "strategy_id": strategy_id,
            "version": VERSION,
            "entry_variant": variant_name,
            "parameters": params,
            "tf_exit_winner_mapping": TF_EXIT_WINNER,
        })
        write_json(pkg.validation_report_file, build_validation_report(strategy_id, variant_name, summaries))

        print(f"[PACKAGE] {pkg.root}")

    print("[DONE] entry v2 component packages built")


if __name__ == "__main__":
    main()
