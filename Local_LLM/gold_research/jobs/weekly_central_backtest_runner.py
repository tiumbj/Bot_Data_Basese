# ============================================================
# ชื่อโค้ด: weekly_central_backtest_runner.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\weekly_central_backtest_runner.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\weekly_central_backtest_runner.py
# เวอร์ชัน: v1.3.0
# ============================================================

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

VERSION = "v1.3.0"
CANONICAL_SYMBOL = "XAUUSD"

ROOT = Path(r"C:\Data\Bot")
CENTRAL_MARKET_DATA_ROOT = ROOT / "central_market_data" / "tf"
CENTRAL_BACKTEST_RESULTS_ROOT = ROOT / "central_backtest_results"
CENTRAL_STRATEGY_REGISTRY_ROOT = ROOT / "central_strategy_registry"

DEPLOYMENT_ENTRY_TIMEFRAMES = ["M1", "M5", "M10", "M15", "H1", "H4"]
RESEARCH_BACKTEST_TIMEFRAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]
VALIDATION_MODES = ["insample", "outsample", "walkforward"]

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
    approved_parameters_file: Path
    validation_report_file: Path
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


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    enable_signal_exit: bool


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(name="A_baseline", enable_signal_exit=True),
    ExperimentConfig(name="B_no_signal_exit", enable_signal_exit=False),
]


def find_first_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_price_dataframe(raw_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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
        if date_col is not None and time_col is not None:
            out["timestamp"] = pd.to_datetime(
                raw_df[date_col].astype(str).str.strip() + " " + raw_df[time_col].astype(str).str.strip(),
                errors="coerce",
            )
        else:
            raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ TF={timeframe} | missing=['timestamp']")
    else:
        out["timestamp"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    missing = []
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")

    if missing:
        raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ TF={timeframe} | missing={missing}")

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


def normalize_signal_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw_df.columns)
    out = pd.DataFrame()

    ts_col = find_first_existing_column(columns, SIGNAL_COLUMN_CANDIDATES["timestamp"])
    if ts_col is None:
        raise RuntimeError("signals.csv ไม่มีคอลัมน์ timestamp/datetime")

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
        raise RuntimeError(f"signals.csv คอลัมน์ไม่ครบ | missing={required_missing}")

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


def discover_strategy_packages() -> List[StrategyPackage]:
    packages: List[StrategyPackage] = []

    for status in ["candidates", "approved"]:
        base = CENTRAL_STRATEGY_REGISTRY_ROOT / status
        if not base.exists():
            continue

        for strategy_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            for version_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir()]):
                strategy_spec_file = version_dir / "strategy_spec.json"
                approved_parameters_file = version_dir / "approved_parameters.json"
                validation_report_file = version_dir / "validation_report.json"
                signals_csv = version_dir / "signals.csv"
                signals_parquet = version_dir / "signals.parquet"

                if not strategy_spec_file.exists():
                    continue

                signals_file = signals_csv if signals_csv.exists() else signals_parquet
                if not signals_file.exists():
                    continue

                packages.append(
                    StrategyPackage(
                        strategy_id=strategy_dir.name,
                        version=version_dir.name,
                        status="candidate" if status == "candidates" else "approved",
                        package_root=version_dir,
                        strategy_spec_file=strategy_spec_file,
                        approved_parameters_file=approved_parameters_file,
                        validation_report_file=validation_report_file,
                        signals_file=signals_file,
                    )
                )

    return packages


def load_price_data(timeframe: str) -> pd.DataFrame:
    file_path = CENTRAL_MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ราคา TF={timeframe} | file={file_path}")

    raw_df = pd.read_csv(file_path)
    return normalize_price_dataframe(raw_df, timeframe)


def load_signals(package: StrategyPackage) -> pd.DataFrame:
    if package.signals_file.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(package.signals_file)
    else:
        raw_df = pd.read_csv(package.signals_file)

    return normalize_signal_dataframe(raw_df)


def build_merged_frame(price_df: pd.DataFrame, signals_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf_signals = signals_df.loc[signals_df["timeframe"].astype(str) == timeframe].copy()

    if tf_signals.empty:
        raise RuntimeError(f"signals ไม่มีข้อมูลสำหรับ TF={timeframe}")

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
            window_df = df.iloc[idx + train_size: idx + train_size + test_size].reset_index(drop=True)
            windows.append((f"window_{counter:02d}", window_df))
            idx += step
            counter += 1
        return windows
    raise RuntimeError(f"validation mode ไม่รองรับ: {mode}")


def simulate_trades(
    df: pd.DataFrame,
    enable_signal_exit: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
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
                }
            elif bool(row["short_entry"]) and pd.notna(row["sl_price"]) and pd.notna(row["tp_price"]):
                position = {
                    "side": "SHORT",
                    "entry_time": ts,
                    "entry_price": c,
                    "sl_price": float(row["sl_price"]),
                    "tp_price": float(row["tp_price"]),
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
                elif enable_signal_exit and bool(row["exit_long"]):
                    exit_reason = "signal_exit"
                    exit_price = c

            elif position["side"] == "SHORT":
                if h >= position["sl_price"]:
                    exit_reason = "stop_loss"
                    exit_price = position["sl_price"]
                elif l <= position["tp_price"]:
                    exit_reason = "take_profit"
                    exit_price = position["tp_price"]
                elif enable_signal_exit and bool(row["exit_short"]):
                    exit_reason = "signal_exit"
                    exit_price = c

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
    expectancy = float(pnl.mean())
    total_trades = int(len(trades_df))
    win_rate = float((pnl > 0).mean() * 100.0)
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
    net_profit = float(pnl.sum())

    if equity_df.empty:
        max_drawdown = 0.0
    else:
        curve = equity_df["equity"].astype(float)
        rolling_max = curve.cummax()
        drawdown = curve - rolling_max
        max_drawdown = float(drawdown.min())

    return {
        "net_profit": net_profit,
        "profit_factor": 999999.0 if math.isinf(profit_factor) else float(profit_factor),
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def get_run_week() -> str:
    now = datetime.now()
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def build_result_dir(
    run_week: str,
    experiment_name: str,
    timeframe: str,
    mode: str,
    strategy_id: str,
    version: str,
    window_name: str,
) -> Path:
    return (
        CENTRAL_BACKTEST_RESULTS_ROOT
        / "weekly"
        / run_week
        / experiment_name
        / timeframe
        / mode
        / strategy_id
        / version
        / window_name
    )


def build_latest_dir(
    experiment_name: str,
    timeframe: str,
    mode: str,
    strategy_id: str,
    version: str,
) -> Path:
    return CENTRAL_BACKTEST_RESULTS_ROOT / "latest" / experiment_name / timeframe / mode / strategy_id / version


def copy_to_latest(src_dir: Path, latest_dir: Path) -> None:
    ensure_dir(latest_dir.parent)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(src_dir, latest_dir)


def update_scope_manifest(run_week: str) -> None:
    payload = {
        "version": VERSION,
        "run_week": run_week,
        "canonical_symbol": CANONICAL_SYMBOL,
        "deployment_entry_timeframes": DEPLOYMENT_ENTRY_TIMEFRAMES,
        "research_backtest_timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "validation_modes": VALIDATION_MODES,
        "market_data_root": str(CENTRAL_MARKET_DATA_ROOT),
        "backtest_results_root": str(CENTRAL_BACKTEST_RESULTS_ROOT),
        "strategy_registry_root": str(CENTRAL_STRATEGY_REGISTRY_ROOT),
        "experiments": [
            {
                "name": experiment.name,
                "enable_signal_exit": experiment.enable_signal_exit,
            }
            for experiment in EXPERIMENTS
        ],
    }
    save_json(CENTRAL_BACKTEST_RESULTS_ROOT / "index" / "scope_manifest.json", payload)


def append_index_rows(rows: List[dict]) -> None:
    index_dir = CENTRAL_BACKTEST_RESULTS_ROOT / "index"
    ensure_dir(index_dir)

    csv_path = index_dir / "backtest_index.csv"
    json_path = index_dir / "backtest_index.json"

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
        dedup_keys = [
            "run_week",
            "experiment_name",
            "strategy_id",
            "version",
            "timeframe",
            "validation_mode",
            "window_name",
        ]
        dedup_keys = [col for col in dedup_keys if col in df_all.columns]
        df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")

    df_all.to_csv(csv_path, index=False)

    payload = {
        "version": VERSION,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": df_all.to_dict(orient="records"),
    }
    save_json(json_path, payload)


def main() -> None:
    run_week = get_run_week()
    update_scope_manifest(run_week)

    packages = discover_strategy_packages()

    print(f"[INFO] version={VERSION}")
    print(f"[INFO] deployment_entry_timeframes={DEPLOYMENT_ENTRY_TIMEFRAMES}")
    print(f"[INFO] research_backtest_timeframes={RESEARCH_BACKTEST_TIMEFRAMES}")
    print(f"[INFO] validation_modes={VALIDATION_MODES}")
    print(f"[INFO] run_week={run_week}")
    print(f"[INFO] experiments={[e.name for e in EXPERIMENTS]}")
    print(f"[INFO] packages={len(packages)}")

    if not packages:
        print("[INFO] ไม่พบ strategy package ใน candidates/approved")
        return

    index_rows: List[dict] = []

    for package in packages:
        try:
            spec = json.loads(package.strategy_spec_file.read_text(encoding="utf-8"))
            strategy_tfs = spec.get("research_backtest_timeframes", RESEARCH_BACKTEST_TIMEFRAMES)
        except Exception:
            strategy_tfs = RESEARCH_BACKTEST_TIMEFRAMES

        strategy_tfs = [tf for tf in strategy_tfs if tf in RESEARCH_BACKTEST_TIMEFRAMES]

        print(
            f"[PACKAGE] strategy={package.strategy_id} "
            f"version={package.version} status={package.status} tfs={strategy_tfs}"
        )

        try:
            signals_df = load_signals(package)
        except Exception as exc:
            print(f"[ERROR] strategy={package.strategy_id} version={package.version} load_signals_error={exc}")
            continue

        for experiment in EXPERIMENTS:
            print(
                f"[EXPERIMENT] strategy={package.strategy_id} "
                f"version={package.version} name={experiment.name} "
                f"enable_signal_exit={experiment.enable_signal_exit}"
            )

            for tf in strategy_tfs:
                try:
                    price_df = load_price_data(tf)
                    merged_df = build_merged_frame(price_df, signals_df, tf)

                    for mode in VALIDATION_MODES:
                        windows = split_validation_windows(merged_df, mode)
                        if not windows:
                            print(
                                f"[WARN] strategy={package.strategy_id} version={package.version} "
                                f"experiment={experiment.name} tf={tf} mode={mode} no_windows"
                            )
                            continue

                        validation_window_rows = []
                        for window_name, window_df in windows:
                            trades_df, equity_df, metrics = simulate_trades(
                                window_df,
                                enable_signal_exit=experiment.enable_signal_exit,
                            )

                            result_dir = build_result_dir(
                                run_week=run_week,
                                experiment_name=experiment.name,
                                timeframe=tf,
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
                                "experiment_name": experiment.name,
                                "enable_signal_exit": experiment.enable_signal_exit,
                                "timeframe": tf,
                                "validation_mode": mode,
                                "window_name": window_name,
                                "run_week": run_week,
                                **metrics,
                            }

                            save_json(result_dir / "metrics.json", metrics_payload)
                            save_dataframe_csv(result_dir / "trades.csv", trades_df)
                            save_dataframe_csv(result_dir / "equity_curve.csv", equity_df)

                            validation_window_rows.append(
                                {
                                    "window_name": window_name,
                                    "rows": int(len(window_df)),
                                    "first_timestamp": pd.Timestamp(window_df["timestamp"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
                                    "last_timestamp": pd.Timestamp(window_df["timestamp"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                                    "metrics": metrics_payload,
                                }
                            )

                            index_rows.append(
                                {
                                    "run_week": run_week,
                                    "experiment_name": experiment.name,
                                    "enable_signal_exit": experiment.enable_signal_exit,
                                    "strategy_id": package.strategy_id,
                                    "version": package.version,
                                    "status": package.status,
                                    "timeframe": tf,
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
                                    "result_path": str(result_dir),
                                    "approved_for_registry": False,
                                }
                            )

                            print(
                                f"[DONE] strategy={package.strategy_id} version={package.version} "
                                f"experiment={experiment.name} tf={tf} mode={mode} window={window_name} "
                                f"trades={metrics['total_trades']} "
                                f"net_profit={metrics['net_profit']:.5f} "
                                f"pf={metrics['profit_factor']:.5f}"
                            )

                        summary_payload = {
                            "strategy_id": package.strategy_id,
                            "version": package.version,
                            "status": package.status,
                            "experiment_name": experiment.name,
                            "enable_signal_exit": experiment.enable_signal_exit,
                            "timeframe": tf,
                            "validation_mode": mode,
                            "run_week": run_week,
                            "windows": validation_window_rows,
                        }

                        summary_dir = (
                            CENTRAL_BACKTEST_RESULTS_ROOT
                            / "weekly"
                            / run_week
                            / experiment.name
                            / tf
                            / mode
                            / package.strategy_id
                            / package.version
                        )
                        save_json(summary_dir / "validation_windows.json", summary_payload)

                        latest_dir = build_latest_dir(
                            experiment_name=experiment.name,
                            timeframe=tf,
                            mode=mode,
                            strategy_id=package.strategy_id,
                            version=package.version,
                        )
                        copy_to_latest(summary_dir, latest_dir)

                except Exception as exc:
                    print(
                        f"[ERROR] strategy={package.strategy_id} version={package.version} "
                        f"experiment={experiment.name} tf={tf} error={exc}"
                    )

    append_index_rows(index_rows)

    print(f"[DONE] weekly central backtest completed | rows={len(index_rows)}")
    print(f"[INDEX] {CENTRAL_BACKTEST_RESULTS_ROOT / 'index' / 'backtest_index.csv'}")
    print(f"[SCOPE] {CENTRAL_BACKTEST_RESULTS_ROOT / 'index' / 'scope_manifest.json'}")


if __name__ == "__main__":
    main()