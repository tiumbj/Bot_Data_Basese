# ============================================================
# ชื่อโค้ด: Multi-Timeframe Dataset Builder
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\build_multitimeframe_dataset.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\build_multitimeframe_dataset.py --input C:\Data\Bot\Local_LLM\M1_XAUUSD\XAUUSD_M1_ALL.csv --symbol XAUUSD
# เวอร์ชัน: v1.0.0
# ============================================================

"""
build_multitimeframe_dataset.py
Version: v1.0.0

Purpose:
- อ่านไฟล์ HistData XAUUSD M1 แบบไม่มี header
- แปลงข้อมูล M1 เป็นหลาย Timeframe:
  M1, M2, M3, M4, M5, M6,M10, M15, M30, H1, H4, D1
- บันทึกออกเป็น canonical dataset สำหรับใช้ backtest ทุกระบบ
- สร้าง manifest สรุปจำนวนแท่ง ช่วงเวลา และ path ของแต่ละ TF

Input format (HistData no-header):
    date,time,open,high,low,close,volume
Example:
    2009.03.15,17:00,929.600000,929.600000,929.600000,929.600000,0

Output structure:
    dataset/
      raw/
        XAUUSD_M1_ALL_clean.csv
      tf/
        XAUUSD_M1.csv
        XAUUSD_M2.csv
        XAUUSD_M3.csv
        XAUUSD_M4.csv
        XAUUSD_M5.csv
        XAUUSD_M6.csv
        XAUUSD_M10.csv
        XAUUSD_M15.csv
        XAUUSD_M30.csv
        XAUUSD_H1.csv
        XAUUSD_H4.csv
        XAUUSD_D1.csv
      manifest/
        dataset_manifest.csv
        dataset_manifest.json

Canonical column format:
    datetime,open,high,low,close,volume

Timezone:
- ใช้ข้อมูลตาม timestamp ที่อยู่ในไฟล์ต้นฉบับ
- ไม่ทำ timezone conversion ในขั้นนี้
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


VERSION = "v1.0.0"

TF_RULES: Dict[str, str] = {
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


@dataclass
class ManifestRow:
    symbol: str
    timeframe: str
    rows: int
    first_datetime: str
    last_datetime: str
    output_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical multi-timeframe dataset from HistData M1 source")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to HistData M1 ALL CSV file",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Trading symbol name used in output filenames",
    )
    parser.add_argument(
        "--output-root",
        default="C:\\Data\\Bot\\Local_LLM\\dataset",
        help="Root output folder",
    )
    parser.add_argument(
        "--drop-duplicate-keep",
        choices=["first", "last"],
        default="first",
        help="How to keep duplicate datetime rows",
    )
    return parser.parse_args()


def load_histdata_m1(input_path: Path, duplicate_keep: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(
        input_path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date": str,
            "time": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        },
        encoding="utf-8-sig",
    )

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y.%m.%d %H:%M",
        errors="coerce",
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.drop_duplicates(subset=["datetime"], keep=duplicate_keep).reset_index(drop=True)

    if df.empty:
        raise ValueError("Loaded dataframe is empty after cleaning.")

    canonical = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    return canonical


def validate_m1_continuity(df: pd.DataFrame) -> None:
    if len(df) < 100:
        raise ValueError("Not enough data. Need more than 100 M1 rows.")

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Datetime is not strictly sorted ascending.")

    duplicated = int(df["datetime"].duplicated().sum())
    if duplicated > 0:
        raise ValueError(f"Duplicate datetime still exists after cleaning: {duplicated}")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    local_df = df.copy()
    local_df = local_df.set_index("datetime")

    resampled = local_df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    resampled["volume"] = resampled["volume"].fillna(0.0)

    return resampled


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_manifest_row(symbol: str, timeframe: str, df: pd.DataFrame, output_file: Path) -> ManifestRow:
    return ManifestRow(
        symbol=symbol,
        timeframe=timeframe,
        rows=int(len(df)),
        first_datetime=str(df["datetime"].iloc[0]) if not df.empty else "",
        last_datetime=str(df["datetime"].iloc[-1]) if not df.empty else "",
        output_file=str(output_file),
    )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    tf_dir = output_root / "tf"
    manifest_dir = output_root / "manifest"

    print("=" * 100)
    print(f"Multi-Timeframe Dataset Builder | version={VERSION}")
    print("=" * 100)
    print(f"Input file    : {input_path}")
    print(f"Symbol        : {args.symbol}")
    print(f"Output root   : {output_root}")
    print(f"Duplicate keep: {args.drop_duplicate_keep}")
    print("=" * 100)

    m1_df = load_histdata_m1(input_path, duplicate_keep=args.drop_duplicate_keep)
    validate_m1_continuity(m1_df)

    clean_m1_path = raw_dir / f"{args.symbol}_M1_ALL_clean.csv"
    save_csv(m1_df, clean_m1_path)

    manifest_rows: List[ManifestRow] = []

    m1_out_path = tf_dir / f"{args.symbol}_M1.csv"
    save_csv(m1_df, m1_out_path)
    manifest_rows.append(build_manifest_row(args.symbol, "M1", m1_df, m1_out_path))

    for timeframe, rule in TF_RULES.items():
        if timeframe == "M1":
            continue

        tf_df = resample_ohlcv(m1_df, rule)
        out_path = tf_dir / f"{args.symbol}_{timeframe}.csv"
        save_csv(tf_df, out_path)
        manifest_rows.append(build_manifest_row(args.symbol, timeframe, tf_df, out_path))

        print(
            f"[OK] {timeframe:<3} | rows={len(tf_df):>8} | "
            f"start={tf_df['datetime'].iloc[0]} | end={tf_df['datetime'].iloc[-1]} | "
            f"file={out_path}"
        )

    manifest_df = pd.DataFrame([asdict(row) for row in manifest_rows])
    manifest_csv_path = manifest_dir / "dataset_manifest.csv"
    manifest_json_path = manifest_dir / "dataset_manifest.json"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_csv_path, index=False, encoding="utf-8-sig")

    with manifest_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": VERSION,
                "symbol": args.symbol,
                "input_file": str(input_path),
                "output_root": str(output_root),
                "timeframes": [asdict(row) for row in manifest_rows],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("[DONE] Build completed")
    print(f"Clean M1 file : {clean_m1_path}")
    print(f"TF folder     : {tf_dir}")
    print(f"Manifest CSV  : {manifest_csv_path}")
    print(f"Manifest JSON : {manifest_json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()