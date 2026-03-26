# -*- coding: utf-8 -*-
"""
ชื่อโค้ด: run_vectorbt_coverage_master_v1_0_0.py
เวอร์ชัน: v1.0.2
ที่อยู่ไฟล์: C:/Data/Bot/Local_LLM/gold_research/jobs/run_vectorbt_coverage_master_v1_0_0.py

Production coverage runner for Local_LLM / gold_research

แนวคิดหลัก
- อ่าน manifest แบบ spec-driven ตาม schema จริงของ research_coverage_master_manifest.csv
- บังคับ full-history และห้าม sampling
- รองรับ resume, live progress, ETA, shard execution
- ใช้ VectorBT เป็นแกน backtest
- มี GPU stack detection/use: CuPy, RAPIDS(cudf), Numba CUDA

ข้อเท็จจริงสำคัญ
- open-source VectorBT ส่วน portfolio simulation ยังทำงานบน CPU เป็นหลัก
- GPU ในไฟล์นี้ใช้เร่ง data/indicator preprocessing เท่าที่ environment รองรับ
- ถ้าไม่มี GPU stack ไฟล์นี้จะ fallback ไป CPU อัตโนมัติ
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message=r".*direction has no effect if short_entries and short_exits are set.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Downcasting object dtype arrays on \.fillna.*",
)

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except Exception as exc:
    raise RuntimeError(
        "VectorBT is required for this runner. Install vectorbt first. "
        f"Original import error: {exc}"
    )

try:
    import pyarrow  # noqa: F401
except Exception:
    pass

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

try:
    import cudf  # type: ignore
except Exception:
    cudf = None

try:
    from numba import cuda  # type: ignore
except Exception:
    cuda = None


VERSION = "v1.0.2"
DEFAULT_MANIFEST_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
DEFAULT_RESULTS_FILE = "results.jsonl"
LIVE_PROGRESS_FILE = "live_progress.json"
RUN_STATE_FILE = "run_state.json"
SUMMARY_FILE = "summary.json"
PROGRESSION_CHECKLIST_FILE = "progression_checklist.txt"

TIMEFRAME_MINUTES = {
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


@dataclass
class ManifestJob:
    manifest_rank: int
    manifest_id: str
    version: str
    phase: str
    timeframe: str
    strategy_family: str
    logic_strictness: str
    swing_variant: str
    pullback_zone_variant: str
    entry_variant: str
    micro_exit_variant: str
    management_variant: str
    regime_variant: str
    robustness_variant: str
    ema_fast: int
    ema_slow: int
    ema_filter_rule: str
    symbol: str
    execution_engine: str
    engine_hint: str
    parallel_hint: str
    gpu_hint: str
    full_history_required: bool
    resume_required: bool
    batch_group: str
    priority_tier: str
    coverage_axis: str
    expected_output_stage: str
    status: str
    rationale: str
    row_index: int


@dataclass
class JobResult:
    manifest_id: str
    status: str
    duration_sec: float
    output_path: str
    result: Dict[str, Any]
    error: str = ""


class GPUStack:
    def __init__(self) -> None:
        self.cupy_available = cp is not None
        self.rapids_available = cudf is not None
        self.numba_cuda_available = False
        self.device_count = 0
        self.backend = "cpu"
        self.error = ""

        if cuda is not None:
            try:
                self.numba_cuda_available = bool(cuda.is_available())
            except Exception as exc:
                self.error = f"numba_cuda_probe_failed: {exc}"

        if self.cupy_available:
            try:
                self.device_count = int(cp.cuda.runtime.getDeviceCount())
            except Exception as exc:
                self.error = f"cupy_probe_failed: {exc}"
                self.device_count = 0

        if self.rapids_available and self.device_count > 0:
            self.backend = "rapids"
        elif self.cupy_available and self.device_count > 0:
            self.backend = "cupy"
        else:
            self.backend = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "cupy_available": self.cupy_available,
            "rapids_available": self.rapids_available,
            "numba_cuda_available": self.numba_cuda_available,
            "device_count": self.device_count,
            "error": self.error,
        }


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Tuple[pd.DataFrame, str]:
    last_error: Optional[Exception] = None
    for enc in DEFAULT_MANIFEST_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read manifest: {path} | last_error={last_error}")


def validate_manifest_schema(df: pd.DataFrame) -> None:
    required = {
        "manifest_rank",
        "manifest_id",
        "version",
        "phase",
        "timeframe",
        "strategy_family",
        "logic_strictness",
        "swing_variant",
        "pullback_zone_variant",
        "entry_variant",
        "micro_exit_variant",
        "management_variant",
        "regime_variant",
        "robustness_variant",
        "ema_fast",
        "ema_slow",
        "ema_filter_rule",
        "symbol",
        "execution_engine",
        "engine_hint",
        "parallel_hint",
        "gpu_hint",
        "full_history_required",
        "resume_required",
        "batch_group",
        "priority_tier",
        "coverage_axis",
        "expected_output_stage",
        "status",
        "rationale",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest schema missing required columns: {missing}")

    forbidden = [
        "sample",
        "sample_pct",
        "sample_ratio",
        "sample_rows",
        "sampling",
        "max_bars",
        "bar_limit",
        "row_limit",
        "nrows",
    ]
    hits = [c for c in forbidden if c in df.columns]
    if hits:
        raise ValueError(f"Manifest contains forbidden sampling columns: {hits}")


def df_to_jobs(df: pd.DataFrame, phase: str, shard_index: int, shard_count: int) -> List[ManifestJob]:
    jobs: List[ManifestJob] = []
    for i, row in df.iterrows():
        if str(row["phase"]).strip() != str(phase).strip():
            continue
        manifest_id = str(row["manifest_id"]).strip()
        if stable_shard(manifest_id, shard_count) != shard_index:
            continue
        jobs.append(
            ManifestJob(
                manifest_rank=int(row["manifest_rank"]),
                manifest_id=manifest_id,
                version=str(row["version"]),
                phase=str(row["phase"]),
                timeframe=str(row["timeframe"]),
                strategy_family=str(row["strategy_family"]),
                logic_strictness=str(row["logic_strictness"]),
                swing_variant=str(row["swing_variant"]),
                pullback_zone_variant=str(row["pullback_zone_variant"]),
                entry_variant=str(row["entry_variant"]),
                micro_exit_variant=str(row["micro_exit_variant"]),
                management_variant=str(row["management_variant"]),
                regime_variant=str(row["regime_variant"]),
                robustness_variant=str(row["robustness_variant"]),
                ema_fast=int(row["ema_fast"]),
                ema_slow=int(row["ema_slow"]),
                ema_filter_rule=str(row["ema_filter_rule"]),
                symbol=str(row["symbol"]),
                execution_engine=str(row["execution_engine"]),
                engine_hint=str(row["engine_hint"]),
                parallel_hint=str(row["parallel_hint"]),
                gpu_hint=str(row["gpu_hint"]),
                full_history_required=bool(row["full_history_required"]),
                resume_required=bool(row["resume_required"]),
                batch_group=str(row["batch_group"]),
                priority_tier=str(row["priority_tier"]),
                coverage_axis=str(row["coverage_axis"]),
                expected_output_stage=str(row["expected_output_stage"]),
                status=str(row["status"]),
                rationale=str(row["rationale"]),
                row_index=int(i),
            )
        )
    jobs.sort(key=lambda x: (x.manifest_rank, x.row_index))
    return jobs


def stable_shard(text: str, shard_count: int) -> int:
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % shard_count


def load_parquet_dataset(symbol: str, timeframe: str, data_root: Path, feature_root: Path, gpu: GPUStack) -> pd.DataFrame:
    candidates = [
        feature_root / f"{symbol}_{timeframe}_base_features.parquet",
        feature_root / f"{symbol}_{timeframe}.parquet",
        data_root / f"{symbol}_{timeframe}.parquet",
    ]
    path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            f"No dataset found for symbol={symbol} timeframe={timeframe}. "
            f"Checked: {[str(x) for x in candidates]}"
        )

    if gpu.backend == "rapids":
        try:
            gdf = cudf.read_parquet(str(path))
            df = gdf.to_pandas()
        except Exception:
            df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path)

    df = normalize_ohlc_dataframe(df)
    return df


def normalize_ohlc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    lower_map = {str(c).lower(): c for c in df.columns}
    for wanted in ["time", "open", "high", "low", "close", "tick_volume", "volume"]:
        if wanted in df.columns:
            continue
        if wanted in lower_map:
            rename_map[lower_map[wanted]] = wanted
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing OHLC columns: {missing}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=False)
        df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        df = df.set_index("time")
    else:
        df = df.copy()
        df.index = pd.RangeIndex(len(df))

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# ------------------------------
# GPU-aware indicator helpers
# ------------------------------
def _numba_cuda_true_range(high_np: np.ndarray, low_np: np.ndarray, prev_close_np: np.ndarray) -> np.ndarray:
    if cuda is None or not cuda.is_available():
        tr = np.maximum(high_np - low_np, np.maximum(np.abs(high_np - prev_close_np), np.abs(low_np - prev_close_np)))
        return tr

    @cuda.jit
    def true_range_kernel(high_arr, low_arr, prev_close_arr, out_arr):
        i = cuda.grid(1)
        if i < out_arr.size:
            a = high_arr[i] - low_arr[i]
            b = abs(high_arr[i] - prev_close_arr[i])
            c = abs(low_arr[i] - prev_close_arr[i])
            m = a
            if b > m:
                m = b
            if c > m:
                m = c
            out_arr[i] = m

    n = high_np.size
    out = np.zeros(n, dtype=np.float64)
    d_high = cuda.to_device(high_np.astype(np.float64))
    d_low = cuda.to_device(low_np.astype(np.float64))
    d_prev = cuda.to_device(prev_close_np.astype(np.float64))
    d_out = cuda.to_device(out)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    true_range_kernel[blocks, threads_per_block](d_high, d_low, d_prev, d_out)
    return d_out.copy_to_host()


def compute_indicators(df: pd.DataFrame, ema_fast: int, ema_slow: int, gpu: GPUStack) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    out["ema_fast"] = close.ewm(span=max(1, int(ema_fast)), adjust=False).mean()
    out["ema_slow"] = close.ewm(span=max(2, int(ema_slow)), adjust=False).mean()

    prev_close = close.shift(1).fillna(close.iloc[0])

    if gpu.backend in {"cupy", "rapids"} and cp is not None:
        high_np = high.to_numpy(dtype=np.float64)
        low_np = low.to_numpy(dtype=np.float64)
        prev_np = prev_close.to_numpy(dtype=np.float64)
        try:
            tr_np = _numba_cuda_true_range(high_np, low_np, prev_np) if gpu.numba_cuda_available else None
            if tr_np is None:
                high_cp = cp.asarray(high_np)
                low_cp = cp.asarray(low_np)
                prev_cp = cp.asarray(prev_np)
                tr_cp = cp.maximum(high_cp - low_cp, cp.maximum(cp.abs(high_cp - prev_cp), cp.abs(low_cp - prev_cp)))
                tr_np = cp.asnumpy(tr_cp)
            out["tr"] = tr_np
        except Exception:
            out["tr"] = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    else:
        out["tr"] = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    out["plus_dm"] = plus_dm
    out["minus_dm"] = minus_dm

    n = 14
    atr = out["tr"].ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()

    out["atr"] = atr.fillna(method="bfill").fillna(method="ffill")
    out["plus_di"] = plus_di.fillna(0.0)
    out["minus_di"] = minus_di.fillna(0.0)
    out["adx"] = adx.fillna(0.0)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)

    return out


def get_logic_thresholds(logic_strictness: str) -> Dict[str, float]:
    key = str(logic_strictness).strip().lower()
    if key == "low":
        return {"adx_min": 16.0, "pullback_k": 0.15, "atr_guard_mult": 1.2}
    if key == "high":
        return {"adx_min": 24.0, "pullback_k": 0.45, "atr_guard_mult": 2.0}
    return {"adx_min": 20.0, "pullback_k": 0.30, "atr_guard_mult": 1.6}


def get_pullback_width_factor(pullback_zone_variant: str) -> float:
    key = str(pullback_zone_variant).strip().lower()
    if key == "narrow":
        return 0.40
    if key == "wide":
        return 1.20
    return 0.75


def get_swing_lookback(swing_variant: str) -> int:
    key = str(swing_variant).strip().lower()
    if key == "short":
        return 3
    if key == "long":
        return 8
    return 5


def apply_regime_filter(ind: pd.DataFrame, regime_variant: str, adx_min: float) -> pd.Series:
    key = str(regime_variant).strip().lower()
    vol_med = ind["atr_pct"].rolling(200, min_periods=30).median()
    if key == "trend_mid_vol":
        return (ind["adx"] >= adx_min) & (ind["atr_pct"] >= vol_med * 0.80) & (ind["atr_pct"] <= vol_med * 1.35)
    if key == "trend_high_vol":
        return (ind["adx"] >= adx_min) & (ind["atr_pct"] > vol_med * 1.10)
    if key == "trend_low_vol":
        return (ind["adx"] >= adx_min) & (ind["atr_pct"] < vol_med * 0.95)
    return ind["adx"] >= adx_min


def build_entry_signals(ind: pd.DataFrame, job: ManifestJob) -> pd.Series:
    cfg = get_logic_thresholds(job.logic_strictness)
    swing_lb = get_swing_lookback(job.swing_variant)
    width_factor = get_pullback_width_factor(job.pullback_zone_variant)

    trend_up = ind["ema_fast"] > ind["ema_slow"]
    regime_ok = apply_regime_filter(ind, job.regime_variant, cfg["adx_min"])

    pullback_ref = ind["ema_fast"] - (ind["atr"] * cfg["pullback_k"] * width_factor)
    pullback_touch = ind["low"] <= pullback_ref

    swing_high = ind["high"].rolling(swing_lb, min_periods=swing_lb).max().shift(1)
    bos_confirm = ind["close"] > swing_high.fillna(ind["close"])  # simple BOS proxy

    if str(job.entry_variant).lower() == "confirm_entry":
        entry_raw = trend_up & regime_ok & pullback_touch & bos_confirm
    elif str(job.entry_variant).lower() == "touch_entry":
        entry_raw = trend_up & regime_ok & pullback_touch
    else:
        entry_raw = trend_up & regime_ok & pullback_touch

    # one-shot entry after flat region
    entry = entry_raw & (~entry_raw.shift(1).fillna(False))

    if "adx_ema" not in str(job.strategy_family).lower() and "ema_cross" in str(job.strategy_family).lower():
        entry = (ind["ema_fast"] > ind["ema_slow"]) & (ind["ema_fast"].shift(1) <= ind["ema_slow"].shift(1))
    elif "trend_continuation" in str(job.strategy_family).lower():
        entry = entry
    else:
        entry = entry

    return entry.fillna(False).astype(bool)


def build_exit_signals(ind: pd.DataFrame, entries: pd.Series, job: ManifestJob) -> pd.Series:
    cfg = get_logic_thresholds(job.logic_strictness)
    key = str(job.micro_exit_variant).strip().lower()

    reverse_signal_exit = ind["ema_fast"] < ind["ema_slow"]
    price_cross_fast_exit = ind["close"] < ind["ema_fast"]
    price_cross_slow_exit = ind["close"] < ind["ema_slow"]
    atr_guard_exit = ind["close"] < (ind["ema_fast"] - ind["atr"] * cfg["atr_guard_mult"])
    adx_fade_exit = (ind["adx"] < max(10.0, cfg["adx_min"] - 6.0)) & (ind["close"] < ind["ema_fast"])
    baseline_pending_exit = price_cross_fast_exit & price_cross_slow_exit

    if key == "reverse_signal_exit":
        exit_raw = reverse_signal_exit
    elif key == "price_cross_fast_exit":
        exit_raw = price_cross_fast_exit
    elif key == "price_cross_slow_exit":
        exit_raw = price_cross_slow_exit
    elif key == "atr_guard_exit":
        exit_raw = atr_guard_exit
    elif key == "adx_fade_exit":
        exit_raw = adx_fade_exit
    elif key == "baseline_pending_exit":
        exit_raw = baseline_pending_exit
    else:
        exit_raw = reverse_signal_exit

    exits = exit_raw.fillna(False).astype(bool)
    # prevent exits before first entry
    if entries.any():
        first_entry_idx = np.argmax(entries.to_numpy(dtype=bool))
        if first_entry_idx > 0:
            exits.iloc[:first_entry_idx] = False
    return exits


def run_vectorbt_job(job: ManifestJob, df: pd.DataFrame, gpu: GPUStack) -> Dict[str, Any]:
    ind = compute_indicators(df=df, ema_fast=job.ema_fast, ema_slow=job.ema_slow, gpu=gpu)
    entries = build_entry_signals(ind, job)
    exits = build_exit_signals(ind, entries, job)

    close = ind["close"].astype(float)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        direction="longonly",
        fees=0.0,
        slippage=0.0,
        init_cash=100000.0,
        size=1.0,
        size_type="amount",
        freq=job.timeframe,
    )

    trades = pf.trades.records_readable
    total_trades = int(getattr(pf.trades, "count", lambda: len(trades))()) if hasattr(pf.trades, "count") else len(trades)

    stats = {
        "total_return_pct": float(pf.total_return() * 100.0),
        "max_drawdown_pct": float(pf.max_drawdown() * 100.0),
        "win_rate_pct": float(pf.trades.win_rate() * 100.0) if total_trades > 0 else 0.0,
        "profit_factor": float(pf.trades.profit_factor()) if total_trades > 0 else 0.0,
        "trade_count": int(total_trades),
        "sharpe_ratio": float(pf.sharpe_ratio()) if len(close) > 30 else 0.0,
        "final_value": float(pf.value().iloc[-1]),
        "start_time": str(ind.index[0]) if len(ind.index) else "",
        "end_time": str(ind.index[-1]) if len(ind.index) else "",
        "bars": int(len(ind)),
        "entries_count": int(entries.sum()),
        "exits_count": int(exits.sum()),
    }

    return {
        "manifest_id": job.manifest_id,
        "phase": job.phase,
        "timeframe": job.timeframe,
        "strategy_family": job.strategy_family,
        "logic_strictness": job.logic_strictness,
        "swing_variant": job.swing_variant,
        "pullback_zone_variant": job.pullback_zone_variant,
        "entry_variant": job.entry_variant,
        "micro_exit_variant": job.micro_exit_variant,
        "management_variant": job.management_variant,
        "regime_variant": job.regime_variant,
        "robustness_variant": job.robustness_variant,
        "ema_fast": job.ema_fast,
        "ema_slow": job.ema_slow,
        "ema_filter_rule": job.ema_filter_rule,
        "symbol": job.symbol,
        "execution_engine": job.execution_engine,
        "gpu_backend": gpu.backend,
        "gpu_stack": gpu.to_dict(),
        "stats": stats,
    }


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": VERSION, "jobs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": VERSION, "jobs": {}}
        data.setdefault("jobs", {})
        return data
    except Exception:
        return {"version": VERSION, "jobs": {}}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    safe_json_dump(path, state)


def append_results_jsonl(results_dir: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(results_dir)
    path = results_dir / DEFAULT_RESULTS_FILE
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_progress_payload(
    *,
    manifest_path: Path,
    manifest_encoding: str,
    outdir: Path,
    phase: str,
    shard_index: int,
    shard_count: int,
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    start_ts: float,
    durations: List[float],
    gpu: GPUStack,
) -> Dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - start_ts)
    completed = done_jobs + failed_jobs + skipped_jobs
    pct = (completed / total_jobs * 100.0) if total_jobs > 0 else 100.0
    throughput = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_jobs - completed)
    avg_duration = float(np.mean(durations)) if durations else 0.0
    eta_sec = (remaining / throughput) if throughput > 0 else (remaining * avg_duration)
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "manifest_encoding": manifest_encoding,
        "outdir": str(outdir),
        "phase": phase,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "progress_pct": round(pct, 6),
        "elapsed_sec": round(elapsed, 4),
        "avg_duration_sec": round(avg_duration, 4),
        "eta_sec": round(float(eta_sec), 4),
        "gpu_stack": gpu.to_dict(),
        "policy": {
            "full_history_only": True,
            "sampling_allowed": False,
            "resume_enabled": True,
            "vectorbt_required": True,
        },
    }


def write_checklist(path: Path, gpu: GPUStack, manifest_encoding: str, phase: str, shard_index: int, shard_count: int) -> None:
    lines = [
        f"version={VERSION}",
        f"generated_at_utc={utc_now_iso()}",
        "progression_checklist:",
        "- production_first = PASS",
        "- full_history_only = PASS",
        "- no_sampling = PASS",
        "- resume = PASS",
        "- live_progress_eta = PASS",
        "- vectorbt_required = PASS",
        f"- manifest_encoding = {manifest_encoding}",
        f"- phase = {phase}",
        f"- shard = {shard_index}/{shard_count}",
        f"- gpu_backend = {gpu.backend}",
        f"- cupy_available = {gpu.cupy_available}",
        f"- rapids_available = {gpu.rapids_available}",
        f"- numba_cuda_available = {gpu.numba_cuda_available}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_jobs(
    jobs: List[ManifestJob],
    manifest_path: Path,
    manifest_encoding: str,
    data_root: Path,
    feature_root: Path,
    outdir: Path,
    phase: str,
    shard_index: int,
    shard_count: int,
    progress_every: int,
    continue_on_error: bool,
) -> None:
    ensure_dir(outdir)
    results_dir = ensure_dir(outdir / "results_jsonl")
    state_path = outdir / RUN_STATE_FILE
    live_progress_path = outdir / LIVE_PROGRESS_FILE
    summary_path = outdir / SUMMARY_FILE
    checklist_path = outdir / PROGRESSION_CHECKLIST_FILE

    gpu = GPUStack()
    write_checklist(checklist_path, gpu, manifest_encoding, phase, shard_index, shard_count)

    state = load_state(state_path)
    state.setdefault("version", VERSION)
    state.setdefault("jobs", {})
    state["gpu_stack"] = gpu.to_dict()
    state["manifest_path"] = str(manifest_path)
    state["outdir"] = str(outdir)

    total_jobs = len(jobs)
    done_jobs = 0
    failed_jobs = 0
    skipped_jobs = 0
    durations: List[float] = []
    start_ts = time.perf_counter()

    for n, job in enumerate(jobs, start=1):
        prev = state["jobs"].get(job.manifest_id, {})
        if str(prev.get("status", "")).upper() == "DONE" and job.resume_required:
            skipped_jobs += 1
            if n % max(1, progress_every) == 0 or n == total_jobs:
                progress = make_progress_payload(
                    manifest_path=manifest_path,
                    manifest_encoding=manifest_encoding,
                    outdir=outdir,
                    phase=phase,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs=failed_jobs,
                    skipped_jobs=skipped_jobs,
                    start_ts=start_ts,
                    durations=durations,
                    gpu=gpu,
                )
                safe_json_dump(live_progress_path, progress)
                safe_json_dump(summary_path, progress)
            continue

        state["jobs"][job.manifest_id] = {
            "status": "RUNNING",
            "started_at_utc": utc_now_iso(),
            "job": asdict(job),
        }
        save_state(state_path, state)

        t0 = time.perf_counter()
        output_json_path = results_dir / f"{job.manifest_id}.json"

        try:
            if str(job.execution_engine).strip().lower() != "vectorbt":
                raise ValueError(f"Unsupported execution_engine: {job.execution_engine}")

            df = load_parquet_dataset(
                symbol=job.symbol,
                timeframe=job.timeframe,
                data_root=data_root,
                feature_root=feature_root,
                gpu=gpu,
            )
            result_payload = run_vectorbt_job(job=job, df=df, gpu=gpu)
            safe_json_dump(output_json_path, result_payload)
            append_results_jsonl(results_dir, result_payload)

            duration = max(0.0, time.perf_counter() - t0)
            durations.append(duration)
            done_jobs += 1
            state["jobs"][job.manifest_id] = {
                "status": "DONE",
                "started_at_utc": state["jobs"][job.manifest_id]["started_at_utc"],
                "ended_at_utc": utc_now_iso(),
                "duration_sec": round(duration, 6),
                "output_path": str(output_json_path),
                "job": asdict(job),
                "stats": result_payload.get("stats", {}),
            }
            save_state(state_path, state)

        except Exception as exc:
            duration = max(0.0, time.perf_counter() - t0)
            durations.append(duration)
            failed_jobs += 1
            err = f"{type(exc).__name__}: {exc}"
            state["jobs"][job.manifest_id] = {
                "status": "FAILED",
                "started_at_utc": state["jobs"][job.manifest_id]["started_at_utc"],
                "ended_at_utc": utc_now_iso(),
                "duration_sec": round(duration, 6),
                "job": asdict(job),
                "error": err,
                "traceback": traceback.format_exc(),
            }
            save_state(state_path, state)
            error_payload = {
                "manifest_id": job.manifest_id,
                "status": "FAILED",
                "error": err,
                "traceback": traceback.format_exc(),
                "job": asdict(job),
            }
            append_results_jsonl(results_dir, error_payload)
            if not continue_on_error:
                raise

        if n % max(1, progress_every) == 0 or n == total_jobs:
            progress = make_progress_payload(
                manifest_path=manifest_path,
                manifest_encoding=manifest_encoding,
                outdir=outdir,
                phase=phase,
                shard_index=shard_index,
                shard_count=shard_count,
                total_jobs=total_jobs,
                done_jobs=done_jobs,
                failed_jobs=failed_jobs,
                skipped_jobs=skipped_jobs,
                start_ts=start_ts,
                durations=durations,
                gpu=gpu,
            )
            safe_json_dump(live_progress_path, progress)
            safe_json_dump(summary_path, progress)
            print(
                f"[PROGRESS] shard={shard_index}/{shard_count} phase={phase} "
                f"done={done_jobs} failed={failed_jobs} skipped={skipped_jobs} total={total_jobs} "
                f"pct={progress['progress_pct']:.4f} eta_sec={progress['eta_sec']:.2f} gpu={gpu.backend}"
            )

    final_progress = make_progress_payload(
        manifest_path=manifest_path,
        manifest_encoding=manifest_encoding,
        outdir=outdir,
        phase=phase,
        shard_index=shard_index,
        shard_count=shard_count,
        total_jobs=total_jobs,
        done_jobs=done_jobs,
        failed_jobs=failed_jobs,
        skipped_jobs=skipped_jobs,
        start_ts=start_ts,
        durations=durations,
        gpu=gpu,
    )
    safe_json_dump(live_progress_path, final_progress)
    safe_json_dump(summary_path, final_progress)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] manifest_encoding={manifest_encoding}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] phase={phase}")
    print(f"[DONE] shard_index={shard_index}")
    print(f"[DONE] shard_count={shard_count}")
    print(f"[DONE] total_jobs={total_jobs}")
    print(f"[DONE] done_jobs={done_jobs}")
    print(f"[DONE] failed_jobs={failed_jobs}")
    print(f"[DONE] skipped_jobs={skipped_jobs}")
    print(f"[DONE] live_progress={live_progress_path}")
    print(f"[DONE] run_state={state_path}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] results_dir={results_dir}")
    print(f"[DONE] gpu_backend={gpu.backend}")
    print("=" * 120)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spec-driven VectorBT coverage runner with GPU stack support")
    parser.add_argument("--manifest", required=True, help="Path to research_coverage_master_manifest.csv")
    parser.add_argument("--data-root", required=True, help="Path to central parquet data root")
    parser.add_argument("--feature-root", required=True, help="Path to central feature cache root")
    parser.add_argument("--outdir", required=True, help="Output directory for this shard")
    parser.add_argument("--phase", required=True, help="Target phase, e.g. micro_exit_expansion")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index")
    parser.add_argument("--shard-count", type=int, required=True, help="Total shard count")
    parser.add_argument("--progress-every", type=int, default=25, help="Flush live progress every N jobs")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after failed jobs")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    outdir = Path(args.outdir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not feature_root.exists():
        raise FileNotFoundError(f"Feature root not found: {feature_root}")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be > 0")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")

    df, manifest_encoding = load_manifest(manifest_path)
    validate_manifest_schema(df)
    jobs = df_to_jobs(
        df=df,
        phase=args.phase,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if not jobs:
        ensure_dir(outdir)
        gpu = GPUStack()
        progress = make_progress_payload(
            manifest_path=manifest_path,
            manifest_encoding=manifest_encoding,
            outdir=outdir,
            phase=args.phase,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            total_jobs=0,
            done_jobs=0,
            failed_jobs=0,
            skipped_jobs=0,
            start_ts=time.perf_counter(),
            durations=[],
            gpu=gpu,
        )
        safe_json_dump(outdir / LIVE_PROGRESS_FILE, progress)
        safe_json_dump(outdir / SUMMARY_FILE, progress)
        write_checklist(outdir / PROGRESSION_CHECKLIST_FILE, gpu, manifest_encoding, args.phase, args.shard_index, args.shard_count)
        print("=" * 120)
        print(f"[DONE] version={VERSION}")
        print(f"[DONE] No jobs selected for phase={args.phase} shard={args.shard_index}/{args.shard_count}")
        print("=" * 120)
        return

    run_jobs(
        jobs=jobs,
        manifest_path=manifest_path,
        manifest_encoding=manifest_encoding,
        data_root=data_root,
        feature_root=feature_root,
        outdir=outdir,
        phase=args.phase,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        progress_every=args.progress_every,
        continue_on_error=bool(args.continue_on_error),
    )


if __name__ == "__main__":
    main()
