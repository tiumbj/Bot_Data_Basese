# ============================================================
# ชื่อโค้ด: run_test_c_top3_m30.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\run_test_c_top3_m30.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\run_test_c_top3_m30.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.0"
CANONICAL_SYMBOL = "XAUUSD"

ROOT = Path(r"C:\Data\Bot")
CENTRAL_MARKET_DATA_ROOT = ROOT / "central_market_data" / "tf"
CENTRAL_BACKTEST_RESULTS_ROOT = ROOT / "central_backtest_results"
CENTRAL_STRATEGY_REGISTRY_ROOT = ROOT / "central_strategy_registry"
SHORTLIST_CSV = CENTRAL_BACKTEST_RESULTS_ROOT / "index" / "final_top3_m30_shortlist.csv"

TARGET_TIMEFRAME = "M30"
VALIDATION_MODES = ["insample", "outsample", "walkforward"]
EXPERIMENT_NAME = "C_regime_filtered_top3_m30"

ALLOW_TREND_BUCKETS = {"BULL_TREND"}
ALLOW_VOLATILITY_BUCKETS = {"HIGH_VOL"}
ALLOW_PRICE_LOCATION_BUCKETS = {"ABOVE_EMA_STACK"}

BLOCK_TREND_BUCKETS = {"BEAR_TREND"}
BLOCK_PRICE_LOCATION_BUCKETS = {"BELOW_EMA_STACK"}

PRICE_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "Time", "DateTime", "DATE_TIME"],
    "open": ["open", "Open", "o", "O"],
    "high": ["high", "High", "h", "H"],
    "low": ["low", "Low", "l", "L"],
    "close": ["close", "Close", "c", "C"],
    "tick_volume": ["tick_volume", "TickVolume", "tickvol", "volume", "Volume"],
    "spread": ["spread", "Spread"],
    "real_volume": ["real_volume", "RealVolume", "realvol"],
}

SIGNAL_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "Time", "DateTime"],
    "symbol": ["symbol", "Symbol"],
    "timeframe": ["timeframe", "tf", "Timeframe", "TF"],
    "long_entry": ["long_entry", "buy_entry", "entry_long"],
    "short_entry": ["short_entry", "sell_entry", "entry_short"],
    "exit_long": ["exit_long", "close_long", "long_exit"],
    "exit_short": ["exit_short", "close_short", "short_exit"],
    "sl_price": ["sl_price", "sl", "stop_loss"],
    "tp_price": ["tp_price", "tp", "take_profit"],
}


@dataclass
class StrategyPackage:
    strategy_id: str
    version: str
    status: str
    package_root: Path
    strategy_spec_file: Path
    signals_file: Path


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    pnl: float
    exit_reason: str
    session_bucket: str
    trend_bucket: str
    volatility_bucket: str
    price_location_bucket: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    for col in ["timestamp", "entry_time", "exit_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def find_first_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_price_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw_df.columns)
    out = pd.DataFrame()

    ts_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["timestamp"])
    open_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["open"])
    high_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["high"])
    low_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["low"])
    close_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["close"])

    if ts_col is None:
        date_col = find_first_existing_column(columns, ["date", "Date", "DATE"])
        time_col = find_first_existing_column(columns, ["time", "Time", "TIME"])
        if date_col is None or time_col is None:
            raise RuntimeError("ไฟล์ราคาไม่มี timestamp หรือ date+time")
        out["timestamp"] = pd.to_datetime(
            raw_df[date_col].astype(str).str.strip() + " " + raw_df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        out["timestamp"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    missing = []
    for name, col in [("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"ไฟล์ราคา M30 คอลัมน์ไม่ครบ | missing={missing}")

    out["open"] = pd.to_numeric(raw_df[open_col], errors="coerce")
    out["high"] = pd.to_numeric(raw_df[high_col], errors="coerce")
    out["low"] = pd.to_numeric(raw_df[low_col], errors="coerce")
    out["close"] = pd.to_numeric(raw_df[close_col], errors="coerce")

    tick_volume_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["tick_volume"])
    spread_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["spread"])
    real_volume_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["real_volume"])

    out["tick_volume"] = pd.to_numeric(raw_df[tick_volume_col], errors="coerce").fillna(0.0) if tick_volume_col else 0.0
    out["spread"] = pd.to_numeric(raw_df[spread_col], errors="coerce").fillna(0.0) if spread_col else 0.0
    out["real_volume"] = pd.to_numeric(raw_df[real_volume_col], errors="coerce").fillna(0.0) if real_volume_col else 0.0

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def load_price_context() -> pd.DataFrame:
    file_path = CENTRAL_MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{TARGET_TIMEFRAME}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ราคา | file={file_path}")

    price_df = normalize_price_dataframe(pd.read_csv(file_path))

    price_df["ema20"] = price_df["close"].ewm(span=20, adjust=False).mean()
    price_df["ema50"] = price_df["close"].ewm(span=50, adjust=False).mean()

    prev_close = price_df["close"].shift(1)
    tr = pd.concat(
        [
            (price_df["high"] - price_df["low"]).abs(),
            (price_df["high"] - prev_close).abs(),
            (price_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price_df["atr14"] = tr.rolling(14, min_periods=14).mean()

    price_df["ema20_slope"] = price_df["ema20"] - price_df["ema20"].shift(3)
    price_df["ema50_slope"] = price_df["ema50"] - price_df["ema50"].shift(3)

    valid_atr = price_df["atr14"].dropna()
    low_q = float(valid_atr.quantile(0.33)) if not valid_atr.empty else 0.0
    high_q = float(valid_atr.quantile(0.66)) if not valid_atr.empty else 0.0

    price_df["session_bucket"] = price_df["timestamp"].apply(map_session_bucket)
    price_df["trend_bucket"] = price_df.apply(map_trend_bucket, axis=1)
    price_df["volatility_bucket"] = price_df["atr14"].apply(lambda x: map_volatility_bucket(x, low_q, high_q))
    price_df["price_location_bucket"] = price_df.apply(map_price_location_bucket, axis=1)

    return price_df


def map_session_bucket(ts: pd.Timestamp) -> str:
    hour = int(pd.Timestamp(ts).hour)
    if 0 <= hour <= 6:
        return "ASIA"
    if 7 <= hour <= 12:
        return "LONDON"
    if 13 <= hour <= 14:
        return "NY_PREOPEN"
    if 15 <= hour <= 19:
        return "NEW_YORK"
    return "LATE_US"


def map_trend_bucket(row: pd.Series) -> str:
    close = float(row["close"]) if pd.notna(row["close"]) else np.nan
    ema20 = float(row["ema20"]) if pd.notna(row["ema20"]) else np.nan
    ema50 = float(row["ema50"]) if pd.notna(row["ema50"]) else np.nan
    slope20 = float(row["ema20_slope"]) if pd.notna(row["ema20_slope"]) else 0.0
    slope50 = float(row["ema50_slope"]) if pd.notna(row["ema50_slope"]) else 0.0

    if np.isnan(close) or np.isnan(ema20) or np.isnan(ema50):
        return "UNKNOWN"
    if close > ema20 > ema50 and slope20 > 0 and slope50 > 0:
        return "BULL_TREND"
    if close < ema20 < ema50 and slope20 < 0 and slope50 < 0:
        return "BEAR_TREND"
    if close >= ema50:
        return "WEAK_BULL"
    return "WEAK_BEAR"


def map_volatility_bucket(atr_value: float, low_q: float, high_q: float) -> str:
    if pd.isna(atr_value):
        return "UNKNOWN"
    x = float(atr_value)
    if x <= low_q:
        return "LOW_VOL"
    if x <= high_q:
        return "MID_VOL"
    return "HIGH_VOL"


def map_price_location_bucket(row: pd.Series) -> str:
    close = float(row["close"]) if pd.notna(row["close"]) else np.nan
    ema20 = float(row["ema20"]) if pd.notna(row["ema20"]) else np.nan
    ema50 = float(row["ema50"]) if pd.notna(row["ema50"]) else np.nan

    if np.isnan(close) or np.isnan(ema20) or np.isnan(ema50):
        return "UNKNOWN"
    if close > ema20 and close > ema50:
        return "ABOVE_EMA_STACK"
    if close < ema20 and close < ema50:
        return "BELOW_EMA_STACK"
    return "INSIDE_EMA_STACK"


def normalize_signal_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw_df.columns)
    out = pd.DataFrame()

    ts_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["timestamp"])
    if ts_col is None:
        raise RuntimeError("signals ไม่มีคอลัมน์ timestamp/datetime")

    out["timestamp"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    symbol_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["symbol"])
    timeframe_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["timeframe"])
    long_entry_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["long_entry"])
    short_entry_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["short_entry"])
    exit_long_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["exit_long"])
    exit_short_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["exit_short"])
    sl_price_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["sl_price"])
    tp_price_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["tp_price"])

    required_missing = []
    for name, col in [
        ("long_entry", long_entry_col),
        ("short_entry", short_entry_col),
        ("exit_long", exit_long_col),
        ("exit_short", exit_short_col),
        ("sl_price", sl_price_col),
        ("tp_price", tp_price_col),
    ]:
        if col is None:
            required_missing.append(name)
    if required_missing:
        raise RuntimeError(f"signals คอลัมน์ไม่ครบ | missing={required_missing}")

    out["symbol"] = raw_df[symbol_col].astype(str) if symbol_col else CANONICAL_SYMBOL
    out["timeframe"] = raw_df[timeframe_col].astype(str) if timeframe_col else ""

    for canonical, source_col in [
        ("long_entry", long_entry_col),
        ("short_entry", short_entry_col),
        ("exit_long", exit_long_col),
        ("exit_short", exit_short_col),
    ]:
        out[canonical] = raw_df[source_col].fillna(False).astype(bool)

    out["sl_price"] = pd.to_numeric(raw_df[sl_price_col], errors="coerce")
    out["tp_price"] = pd.to_numeric(raw_df[tp_price_col], errors="coerce")

    out = out.dropna(subset=["timestamp"]).copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def discover_strategy_packages(target_strategy_ids: set[str]) -> List[StrategyPackage]:
    packages: List[StrategyPackage] = []

    for status_dir, status_name in [("candidates", "candidate"), ("approved", "approved")]:
        base = CENTRAL_STRATEGY_REGISTRY_ROOT / status_dir
        if not base.exists():
            continue

        for strategy_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            if strategy_dir.name not in target_strategy_ids:
                continue

            version_dirs = sorted([p for p in strategy_dir.iterdir() if p.is_dir()])
            if not version_dirs:
                continue

            latest = version_dirs[-1]
            strategy_spec_file = latest / "strategy_spec.json"
            signals_csv = latest / "signals.csv"
            signals_parquet = latest / "signals.parquet"
            signals_file = signals_csv if signals_csv.exists() else signals_parquet

            if not strategy_spec_file.exists() or not signals_file.exists():
                continue

            packages.append(
                StrategyPackage(
                    strategy_id=strategy_dir.name,
                    version=latest.name,
                    status=status_name,
                    package_root=latest,
                    strategy_spec_file=strategy_spec_file,
                    signals_file=signals_file,
                )
            )

    return packages


def load_signals(package: StrategyPackage) -> pd.DataFrame:
    if package.signals_file.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(package.signals_file)
    else:
        raw_df = pd.read_csv(package.signals_file)
    return normalize_signal_dataframe(raw_df)


def load_shortlist_strategy_ids() -> set[str]:
    if not SHORTLIST_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ shortlist | file={SHORTLIST_CSV}")

    df = pd.read_csv(SHORTLIST_CSV)
    if "strategy_id" not in df.columns or "timeframe" not in df.columns:
        raise RuntimeError("shortlist คอลัมน์ไม่ครบ")

    work = df.copy()
    work["strategy_id"] = work["strategy_id"].astype(str).str.strip()
    work["timeframe"] = work["timeframe"].astype(str).str.strip()
    target_ids = set(work.loc[work["timeframe"] == TARGET_TIMEFRAME, "strategy_id"].unique().tolist())
    if not target_ids:
        raise RuntimeError("shortlist ไม่มี strategy M30")
    return target_ids


def build_merged_frame(price_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    tf_signals = signals_df.loc[signals_df["timeframe"].astype(str) == TARGET_TIMEFRAME].copy()
    if tf_signals.empty:
        raise RuntimeError(f"signals ไม่มีข้อมูล TF={TARGET_TIMEFRAME}")

    merged = price_df.merge(
        tf_signals[["timestamp", "long_entry", "short_entry", "exit_long", "exit_short", "sl_price", "tp_price"]],
        on="timestamp",
        how="left",
    )

    for col in ["long_entry", "short_entry", "exit_long", "exit_short"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    merged["sl_price"] = pd.to_numeric(merged["sl_price"], errors="coerce")
    merged["tp_price"] = pd.to_numeric(merged["tp_price"], errors="coerce")
    return merged


def apply_test_c_filters(merged_df: pd.DataFrame) -> pd.DataFrame:
    work = merged_df.copy()

    allowed_entry = (
        work["trend_bucket"].isin(ALLOW_TREND_BUCKETS)
        & work["volatility_bucket"].isin(ALLOW_VOLATILITY_BUCKETS)
        & work["price_location_bucket"].isin(ALLOW_PRICE_LOCATION_BUCKETS)
        & ~work["trend_bucket"].isin(BLOCK_TREND_BUCKETS)
        & ~work["price_location_bucket"].isin(BLOCK_PRICE_LOCATION_BUCKETS)
    )

    work["long_entry_raw"] = work["long_entry"].astype(bool)
    work["short_entry_raw"] = work["short_entry"].astype(bool)

    work["long_entry"] = work["long_entry"] & allowed_entry
    work["short_entry"] = work["short_entry"] & allowed_entry

    return work


def split_validation_windows(df: pd.DataFrame, mode: str) -> List[Tuple[str, pd.DataFrame]]:
    n = len(df)
    if n < 300:
        return []

    if mode == "insample":
        end = int(n * 0.7)
        return [("window_01", df.iloc[:end].reset_index(drop=True))]
    if mode == "outsample":
        start = int(n * 0.7)
        return [("window_01", df.iloc[start:].reset_index(drop=True))]
    if mode == "walkforward":
        train_size = int(n * 0.5)
        test_size = int(n * 0.1)
        step = test_size
        windows: List[Tuple[str, pd.DataFrame]] = []
        idx = 0
        counter = 1
        while (idx + train_size + test_size) <= n:
            window_df = df.iloc[idx + train_size : idx + train_size + test_size].reset_index(drop=True)
            windows.append((f"window_{counter:02d}", window_df))
            idx += step
            counter += 1
        return windows

    raise RuntimeError(f"validation mode ไม่รองรับ: {mode}")


def simulate_trades(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    trades: List[Trade] = []
    equity_rows: List[dict] = []
    equity = 0.0
    position: Optional[dict] = None

    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        if position is None:
            if bool(row["long_entry"]) and pd.notna(row["sl_price"]) and pd.notna(row["tp_price"]):
                position = {
                    "side": "LONG",
                    "entry_time": ts,
                    "entry_price": c,
                    "sl_price": float(row["sl_price"]),
                    "tp_price": float(row["tp_price"]),
                    "session_bucket": str(row["session_bucket"]),
                    "trend_bucket": str(row["trend_bucket"]),
                    "volatility_bucket": str(row["volatility_bucket"]),
                    "price_location_bucket": str(row["price_location_bucket"]),
                }
            elif bool(row["short_entry"]) and pd.notna(row["sl_price"]) and pd.notna(row["tp_price"]):
                position = {
                    "side": "SHORT",
                    "entry_time": ts,
                    "entry_price": c,
                    "sl_price": float(row["sl_price"]),
                    "tp_price": float(row["tp_price"]),
                    "session_bucket": str(row["session_bucket"]),
                    "trend_bucket": str(row["trend_bucket"]),
                    "volatility_bucket": str(row["volatility_bucket"]),
                    "price_location_bucket": str(row["price_location_bucket"]),
                }
        else:
            exit_reason = None
            exit_price = None

            if position["side"] == "LONG":
                if l <= position["sl_price"]:
                    exit_reason = "stop_loss"
                    exit_price = position["sl_price"]
                elif h >= position["tp_price"]:
                    exit_reason = "take_profit"
                    exit_price = position["tp_price"]
            else:
                if h >= position["sl_price"]:
                    exit_reason = "stop_loss"
                    exit_price = position["sl_price"]
                elif l <= position["tp_price"]:
                    exit_reason = "take_profit"
                    exit_price = position["tp_price"]

            if exit_reason is not None and exit_price is not None:
                pnl = (
                    float(exit_price) - float(position["entry_price"])
                    if position["side"] == "LONG"
                    else float(position["entry_price"]) - float(exit_price)
                )
                equity += pnl
                trades.append(
                    Trade(
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        side=position["side"],
                        entry_price=float(position["entry_price"]),
                        exit_price=float(exit_price),
                        sl_price=float(position["sl_price"]),
                        tp_price=float(position["tp_price"]),
                        pnl=float(pnl),
                        exit_reason=exit_reason,
                        session_bucket=position["session_bucket"],
                        trend_bucket=position["trend_bucket"],
                        volatility_bucket=position["volatility_bucket"],
                        price_location_bucket=position["price_location_bucket"],
                    )
                )
                position = None

        equity_rows.append({"timestamp": ts, "equity": equity, "close": c})

    if position is not None and len(df) > 0:
        last = df.iloc[-1]
        ts = pd.Timestamp(last["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        c = float(last["close"])
        pnl = c - float(position["entry_price"]) if position["side"] == "LONG" else float(position["entry_price"]) - c
        equity += pnl
        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=ts,
                side=position["side"],
                entry_price=float(position["entry_price"]),
                exit_price=float(c),
                sl_price=float(position["sl_price"]),
                tp_price=float(position["tp_price"]),
                pnl=float(pnl),
                exit_reason="end_of_data",
                session_bucket=position["session_bucket"],
                trend_bucket=position["trend_bucket"],
                volatility_bucket=position["volatility_bucket"],
                price_location_bucket=position["price_location_bucket"],
            )
        )
        if equity_rows:
            equity_rows[-1]["equity"] = equity

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(trades_df, equity_df)
    return trades_df, equity_df, metrics


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnl = trades_df["pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    if equity_df.empty:
        max_drawdown = 0.0
    else:
        curve = equity_df["equity"].astype(float)
        rolling_max = curve.cummax()
        drawdown = curve - rolling_max
        max_drawdown = float(drawdown.min())

    return {
        "net_profit": float(pnl.sum()),
        "profit_factor": 999999.0 if math.isinf(profit_factor) else float(profit_factor),
        "expectancy": float(pnl.mean()),
        "max_drawdown": max_drawdown,
        "total_trades": int(len(trades_df)),
        "win_rate": float((pnl > 0).mean() * 100.0),
        "avg_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "avg_loss": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0,
    }


def build_bucket_summary(trades_df: pd.DataFrame, bucket_col: str, bucket_type: str) -> pd.DataFrame:
    if trades_df.empty or bucket_col not in trades_df.columns:
        return pd.DataFrame()

    rows = []
    for bucket_name, g in trades_df.groupby(bucket_col, dropna=False):
        pnl = pd.to_numeric(g["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket_name": str(bucket_name) if pd.notna(bucket_name) else "UNKNOWN",
                "trades": int(len(g)),
                "wins": wins,
                "losses": losses,
                "win_rate": round((wins / max(len(g), 1)) * 100.0, 6),
                "pnl_sum": round(float(pnl.sum()), 6),
                "avg_pnl": round(float(pnl.mean()) if len(g) > 0 else 0.0, 6),
                "profit_factor": round(999999.0 if np.isinf(pf) else float(pf), 6),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["bucket_type", "pnl_sum"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_regime_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        build_bucket_summary(trades_df, "session_bucket", "session_bucket"),
        build_bucket_summary(trades_df, "trend_bucket", "trend_bucket"),
        build_bucket_summary(trades_df, "volatility_bucket", "volatility_bucket"),
        build_bucket_summary(trades_df, "price_location_bucket", "price_location_bucket"),
        build_bucket_summary(trades_df, "exit_reason", "exit_reason"),
    ]
    non_empty = [x for x in pieces if not x.empty]
    return pd.concat(non_empty, axis=0, ignore_index=True) if non_empty else pd.DataFrame()


def get_run_week() -> str:
    now = datetime.now()
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def build_result_dir(run_week: str, timeframe: str, mode: str, strategy_id: str, version: str, window_name: str) -> Path:
    return (
        CENTRAL_BACKTEST_RESULTS_ROOT
        / "weekly"
        / run_week
        / EXPERIMENT_NAME
        / timeframe
        / mode
        / strategy_id
        / version
        / window_name
    )


def build_latest_dir(timeframe: str, mode: str, strategy_id: str, version: str) -> Path:
    return CENTRAL_BACKTEST_RESULTS_ROOT / "latest" / EXPERIMENT_NAME / timeframe / mode / strategy_id / version


def copy_to_latest(src_dir: Path, latest_dir: Path) -> None:
    ensure_dir(latest_dir.parent)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(src_dir, latest_dir)


def append_index_rows(rows: List[dict]) -> None:
    index_dir = CENTRAL_BACKTEST_RESULTS_ROOT / "index"
    ensure_dir(index_dir)

    csv_path = index_dir / "test_c_top3_m30_index.csv"
    json_path = index_dir / "test_c_top3_m30_index.json"

    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
            df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
        except Exception:
            df_all = df_new.copy()
    else:
        df_all = df_new.copy()

    if not df_all.empty:
        dedup_keys = ["run_week", "experiment_name", "strategy_id", "version", "timeframe", "validation_mode", "window_name"]
        dedup_keys = [c for c in dedup_keys if c in df_all.columns]
        df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")

    df_all.to_csv(csv_path, index=False)
    save_json(
        json_path,
        {
            "version": VERSION,
            "updated_at_utc": utc_now_iso(),
            "rows": df_all.to_dict(orient="records"),
        },
    )


def main() -> None:
    print("=" * 120)
    print(f"[INFO] run_test_c_top3_m30.py version={VERSION}")
    print(f"[INFO] experiment_name={EXPERIMENT_NAME}")
    print(f"[INFO] shortlist={SHORTLIST_CSV}")
    print(f"[INFO] timeframe={TARGET_TIMEFRAME}")
    print(f"[INFO] allow_trend={sorted(ALLOW_TREND_BUCKETS)}")
    print(f"[INFO] allow_volatility={sorted(ALLOW_VOLATILITY_BUCKETS)}")
    print(f"[INFO] allow_price_location={sorted(ALLOW_PRICE_LOCATION_BUCKETS)}")
    print("=" * 120)

    run_week = get_run_week()
    target_strategy_ids = load_shortlist_strategy_ids()
    packages = discover_strategy_packages(target_strategy_ids)
    if not packages:
        raise RuntimeError("ไม่พบ top3 packages ใน strategy registry")

    price_df = load_price_context()
    index_rows: List[dict] = []

    for package in packages:
        print(f"[PACKAGE] strategy={package.strategy_id} version={package.version} status={package.status}")
        signals_df = load_signals(package)
        merged_df = build_merged_frame(price_df, signals_df)
        filtered_df = apply_test_c_filters(merged_df)

        raw_long_entries = int(filtered_df["long_entry_raw"].sum())
        raw_short_entries = int(filtered_df["short_entry_raw"].sum())
        kept_long_entries = int(filtered_df["long_entry"].sum())
        kept_short_entries = int(filtered_df["short_entry"].sum())

        filter_report = {
            "version": VERSION,
            "generated_at_utc": utc_now_iso(),
            "experiment_name": EXPERIMENT_NAME,
            "strategy_id": package.strategy_id,
            "timeframe": TARGET_TIMEFRAME,
            "raw_long_entries": raw_long_entries,
            "raw_short_entries": raw_short_entries,
            "kept_long_entries": kept_long_entries,
            "kept_short_entries": kept_short_entries,
            "filtered_long_entries": raw_long_entries - kept_long_entries,
            "filtered_short_entries": raw_short_entries - kept_short_entries,
            "allow_trend_buckets": sorted(ALLOW_TREND_BUCKETS),
            "allow_volatility_buckets": sorted(ALLOW_VOLATILITY_BUCKETS),
            "allow_price_location_buckets": sorted(ALLOW_PRICE_LOCATION_BUCKETS),
            "block_trend_buckets": sorted(BLOCK_TREND_BUCKETS),
            "block_price_location_buckets": sorted(BLOCK_PRICE_LOCATION_BUCKETS),
        }

        for mode in VALIDATION_MODES:
            windows = split_validation_windows(filtered_df, mode)
            validation_window_rows = []

            for window_name, window_df in windows:
                trades_df, equity_df, metrics = simulate_trades(window_df)
                regime_summary_df = build_regime_summary(trades_df)

                result_dir = build_result_dir(
                    run_week=run_week,
                    timeframe=TARGET_TIMEFRAME,
                    mode=mode,
                    strategy_id=package.strategy_id,
                    version=package.version,
                    window_name=window_name,
                )
                ensure_dir(result_dir)

                metrics_payload = {
                    "strategy_id": package.strategy_id,
                    "version": package.version,
                    "status": package.status,
                    "experiment_name": EXPERIMENT_NAME,
                    "timeframe": TARGET_TIMEFRAME,
                    "validation_mode": mode,
                    "window_name": window_name,
                    "run_week": run_week,
                    **metrics,
                }

                save_json(result_dir / "metrics.json", metrics_payload)
                save_dataframe_csv(result_dir / "trades.csv", trades_df)
                save_dataframe_csv(result_dir / "equity_curve.csv", equity_df)
                save_dataframe_csv(result_dir / "regime_summary.csv", regime_summary_df)
                save_json(result_dir / "filter_report.json", filter_report)

                validation_window_rows.append(
                    {
                        "window_name": window_name,
                        "rows": int(len(window_df)),
                        "metrics": metrics_payload,
                    }
                )

                index_rows.append(
                    {
                        "run_week": run_week,
                        "experiment_name": EXPERIMENT_NAME,
                        "strategy_id": package.strategy_id,
                        "version": package.version,
                        "status": package.status,
                        "timeframe": TARGET_TIMEFRAME,
                        "validation_mode": mode,
                        "window_name": window_name,
                        "net_profit": metrics["net_profit"],
                        "profit_factor": metrics["profit_factor"],
                        "expectancy": metrics["expectancy"],
                        "max_drawdown": metrics["max_drawdown"],
                        "total_trades": metrics["total_trades"],
                        "win_rate": metrics["win_rate"],
                        "avg_win": metrics["avg_win"],
                        "avg_loss": metrics["avg_loss"],
                        "raw_long_entries": raw_long_entries,
                        "raw_short_entries": raw_short_entries,
                        "kept_long_entries": kept_long_entries,
                        "kept_short_entries": kept_short_entries,
                        "result_path": str(result_dir),
                    }
                )

                print(
                    f"[DONE] strategy={package.strategy_id} mode={mode} window={window_name} "
                    f"trades={metrics['total_trades']} net_profit={metrics['net_profit']:.5f} "
                    f"pf={metrics['profit_factor']:.5f}"
                )

            summary_dir = (
                CENTRAL_BACKTEST_RESULTS_ROOT
                / "weekly"
                / run_week
                / EXPERIMENT_NAME
                / TARGET_TIMEFRAME
                / mode
                / package.strategy_id
                / package.version
            )
            save_json(
                summary_dir / "validation_windows.json",
                {
                    "version": VERSION,
                    "generated_at_utc": utc_now_iso(),
                    "experiment_name": EXPERIMENT_NAME,
                    "strategy_id": package.strategy_id,
                    "version_tag": package.version,
                    "status": package.status,
                    "timeframe": TARGET_TIMEFRAME,
                    "validation_mode": mode,
                    "run_week": run_week,
                    "filter_report": filter_report,
                    "windows": validation_window_rows,
                },
            )

            latest_dir = build_latest_dir(TARGET_TIMEFRAME, mode, package.strategy_id, package.version)
            copy_to_latest(summary_dir, latest_dir)

    append_index_rows(index_rows)
    print("=" * 120)
    print(f"[DONE] experiment={EXPERIMENT_NAME} rows={len(index_rows)}")
    print(f"[INDEX] {CENTRAL_BACKTEST_RESULTS_ROOT / 'index' / 'test_c_top3_m30_index.csv'}")
    print("=" * 120)


if __name__ == "__main__":
    main()