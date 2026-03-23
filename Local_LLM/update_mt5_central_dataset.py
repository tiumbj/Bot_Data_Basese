# ============================================================
# ชื่อโค้ด: update_mt5_central_dataset.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\update_mt5_central_dataset.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\update_mt5_central_dataset.py --root C:\Data\Bot\Local_LLM --source-symbol AUTO
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise SystemExit(
        "ไม่พบแพ็กเกจ MetaTrader5\n"
        "ติดตั้งก่อนด้วยคำสั่ง: pip install MetaTrader5 pandas"
    ) from exc


VERSION = "v1.0.0"
CANONICAL_SYMBOL = "XAUUSD"

# ระบบข้อมูลกลางที่ล็อกแล้ว
TIMEFRAME_MINUTES: Dict[str, int] = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


@dataclass
class DatasetPaths:
    root: Path
    dataset_root: Path
    tf_root: Path
    manifest_csv: Path
    manifest_json: Path
    m1_file: Path

    @classmethod
    def build(cls, root: Path) -> "DatasetPaths":
        dataset_root = root / "dataset"
        tf_root = dataset_root / "tf"
        return cls(
            root=root,
            dataset_root=dataset_root,
            tf_root=tf_root,
            manifest_csv=tf_root / "manifest.csv",
            manifest_json=tf_root / "manifest.json",
            m1_file=tf_root / f"{CANONICAL_SYMBOL}_M1.csv",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily incremental updater สำหรับระบบข้อมูลกลาง XAUUSD ของทุก bot"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Project root เช่น C:\\Data\\Bot\\Local_LLM",
    )
    parser.add_argument(
        "--source-symbol",
        type=str,
        default="AUTO",
        help="symbol ที่ใช้ดึงจาก MT5 เช่น AUTO, XAUUSD, GOLD, XAUUSDm",
    )
    parser.add_argument(
        "--start-buffer-minutes",
        type=int,
        default=10,
        help="ดึงย้อนทับจากเวลาสุดท้ายของ M1 กี่นาที เพื่อกันข้อมูลขาด/ซ้ำ",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="บังคับ rebuild จาก M1 เดิมใหม่ทั้งหมด แม้ไม่มีข้อมูลใหม่",
    )
    return parser.parse_args()


def ensure_directories(paths: DatasetPaths) -> None:
    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    paths.tf_root.mkdir(parents=True, exist_ok=True)


def initialize_mt5() -> None:
    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise RuntimeError(f"เชื่อมต่อ MT5 ไม่สำเร็จ | code={code} msg={msg}")


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def detect_source_symbol(requested: str) -> str:
    if requested and requested.upper() != "AUTO":
        return requested

    candidates = ["XAUUSD", "GOLD", "XAUUSDm"]
    available = mt5.symbols_get()
    if available is None:
        code, msg = mt5.last_error()
        raise RuntimeError(f"อ่าน symbols จาก MT5 ไม่สำเร็จ | code={code} msg={msg}")

    names = {s.name for s in available}
    for symbol in candidates:
        if symbol in names:
            return symbol

    raise RuntimeError(
        "ไม่พบ symbol ที่รองรับใน MT5 | tried=XAUUSD,GOLD,XAUUSDm"
    )


def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"ไม่พบ symbol ใน MT5: {symbol}")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            code, msg = mt5.last_error()
            raise RuntimeError(f"เลือก symbol ไม่สำเร็จ | symbol={symbol} code={code} msg={msg}")


def read_existing_m1(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.read_csv(file_path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"ไฟล์ M1 เดิมคอลัมน์ไม่ครบ | missing={sorted(missing)} | file={file_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    for col in CSV_COLUMNS:
        if col not in df.columns:
            if col == "spread":
                df[col] = 0
            else:
                df[col] = 0

    return df[CSV_COLUMNS].copy()


def determine_fetch_start(existing_m1: pd.DataFrame, start_buffer_minutes: int) -> datetime:
    if existing_m1.empty:
        return datetime(2009, 1, 1, tzinfo=timezone.utc)

    last_ts = pd.Timestamp(existing_m1["timestamp"].iloc[-1]).to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    return last_ts - timedelta(minutes=start_buffer_minutes)


def fetch_m1_from_mt5(symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_utc, end_utc)
    if rates is None:
        code, msg = mt5.last_error()
        raise RuntimeError(
            f"ดึงข้อมูล M1 จาก MT5 ไม่สำเร็จ | symbol={symbol} code={code} msg={msg}"
        )

    if len(rates) == 0:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df = df.rename(
        columns={
            "tick_volume": "tick_volume",
            "real_volume": "real_volume",
        }
    )

    for col in CSV_COLUMNS:
        if col not in df.columns:
            if col == "spread":
                df[col] = 0
            else:
                df[col] = 0

    return df[CSV_COLUMNS].copy()


def merge_m1(existing_m1: pd.DataFrame, new_m1: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([existing_m1, new_m1], axis=0, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=False, errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).copy()
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return merged[CSV_COLUMNS].copy()


def timeframe_to_pandas_rule(tf: str) -> str:
    mapping = {
        "M1": "1min",
        "M2": "2min",
        "M3": "3min",
        "M4": "4min",
        "M5": "5min",
        "M6": "6min",
        "M10": "10min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
    }
    if tf not in mapping:
        raise RuntimeError(f"TF ไม่รองรับ: {tf}")
    return mapping[tf]


def resample_from_m1(m1_df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if m1_df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    if tf == "M1":
        return m1_df.copy()

    df = m1_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").set_index("timestamp")

    rule = timeframe_to_pandas_rule(tf)

    agg = (
        df.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "max",
                "real_volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )

    return agg[CSV_COLUMNS].copy()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def build_manifest_rows(tf_root: Path) -> List[dict]:
    rows: List[dict] = []

    for tf in TIMEFRAME_MINUTES.keys():
        file_path = tf_root / f"{CANONICAL_SYMBOL}_{tf}.csv"
        if not file_path.exists():
            rows.append(
                {
                    "symbol": CANONICAL_SYMBOL,
                    "timeframe": tf,
                    "rows": 0,
                    "first_datetime": "",
                    "last_datetime": "",
                    "output_file": str(file_path),
                }
            )
            continue

        df = pd.read_csv(file_path)
        if df.empty or "timestamp" not in df.columns:
            rows.append(
                {
                    "symbol": CANONICAL_SYMBOL,
                    "timeframe": tf,
                    "rows": 0,
                    "first_datetime": "",
                    "last_datetime": "",
                    "output_file": str(file_path),
                }
            )
            continue

        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            first_dt = ""
            last_dt = ""
        else:
            first_dt = ts.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            last_dt = ts.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

        rows.append(
            {
                "symbol": CANONICAL_SYMBOL,
                "timeframe": tf,
                "rows": int(len(df)),
                "first_datetime": first_dt,
                "last_datetime": last_dt,
                "output_file": str(file_path),
            }
        )

    return rows


def save_manifest(paths: DatasetPaths) -> None:
    rows = build_manifest_rows(paths.tf_root)
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(paths.manifest_csv, index=False)

    payload = {
        "version": VERSION,
        "canonical_symbol": CANONICAL_SYMBOL,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_root": str(paths.tf_root),
        "items": rows,
    }
    paths.manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(paths: DatasetPaths) -> None:
    if not paths.manifest_csv.exists():
        print("manifest.csv ยังไม่ถูกสร้าง")
        return

    df = pd.read_csv(paths.manifest_csv)
    print(df.to_csv(index=False))


def update_central_dataset(
    root: Path,
    requested_symbol: str,
    start_buffer_minutes: int,
    full_refresh: bool,
) -> None:
    paths = DatasetPaths.build(root)
    ensure_directories(paths)

    initialize_mt5()
    try:
        source_symbol = detect_source_symbol(requested_symbol)
        ensure_symbol_selected(source_symbol)

        existing_m1 = read_existing_m1(paths.m1_file)
        start_utc = determine_fetch_start(existing_m1, start_buffer_minutes)
        end_utc = datetime.now(timezone.utc)

        print(f"[INFO] version={VERSION}")
        print(f"[INFO] root={root}")
        print(f"[INFO] source_symbol={source_symbol}")
        print(f"[INFO] canonical_symbol={CANONICAL_SYMBOL}")
        print(f"[INFO] tf_root={paths.tf_root}")
        print(f"[INFO] fetch_start_utc={start_utc.isoformat()}")
        print(f"[INFO] fetch_end_utc={end_utc.isoformat()}")

        new_m1 = fetch_m1_from_mt5(source_symbol, start_utc, end_utc)

        if new_m1.empty and existing_m1.empty:
            raise RuntimeError("ไม่มีข้อมูล M1 ทั้งในไฟล์เดิมและจาก MT5")

        merged_m1 = merge_m1(existing_m1, new_m1)

        # บันทึก M1 ก่อน แล้ว rebuild ทั้งหมดจาก M1 กลาง
        save_csv(merged_m1, paths.m1_file)

        if new_m1.empty and not full_refresh:
            print("[INFO] ไม่มีข้อมูลใหม่จาก MT5 แต่จะอัปเดต manifest จากไฟล์ที่มีอยู่")
            save_manifest(paths)
            print_summary(paths)
            return

        for tf in TIMEFRAME_MINUTES.keys():
            tf_df = resample_from_m1(merged_m1, tf)
            out_file = paths.tf_root / f"{CANONICAL_SYMBOL}_{tf}.csv"
            save_csv(tf_df, out_file)
            print(f"[DONE] tf={tf} rows={len(tf_df)} file={out_file}")

        save_manifest(paths)
        print("[DONE] manifest updated")
        print_summary(paths)

    finally:
        shutdown_mt5()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    update_central_dataset(
        root=root,
        requested_symbol=args.source_symbol,
        start_buffer_minutes=args.start_buffer_minutes,
        full_refresh=args.full_refresh,
    )


if __name__ == "__main__":
    main()