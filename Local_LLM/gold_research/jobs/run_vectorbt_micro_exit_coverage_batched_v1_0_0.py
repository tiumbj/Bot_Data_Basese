"""
Code Name : run_vectorbt_micro_exit_coverage_batched_v1_0_0.py
Version   : v1.0.0
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_0.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_0.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outdir C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_0\run_0 --phase micro_exit_expansion --portfolio-chunk-size 64 --progress-every-groups 1 --continue-on-error

Purpose
-------
Run the micro_exit_expansion coverage manifest directly in one persistent batched process.
This avoids the old subprocess-per-job pattern and keeps the external run structure stable:
- bootstrap.log
- state.jsonl
- completed_ids.txt
- live_progress.json
- summary.json
- per_timeframe/*.csv

Changelog v1.0.0
----------------
1) New add-on runner only. No change to old coverage runner or existing worker files.
2) Read the existing coverage manifest directly and filter by phase.
3) Group jobs by timeframe + logic_variant + ema_fast + ema_slow.
4) Reuse price/features/EMAs/entries per group.
5) Evaluate many jobs at once via vectorbt multi-column Portfolio.from_signals.
6) Keep resume via completed_ids.txt and append-only state.jsonl.
7) Fall back to single-job evaluation if a batch group fails, so long runs do not stop unnecessarily.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


VERSION = "v1.0.0"
DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
STATE_RUNNING = "RUNNING"
STATE_DONE = "DONE"
STATE_FAILED = "FAILED"
FINAL_DONE = "DONE"
FINAL_DONE_WITH_ERRORS = "DONE_WITH_ERRORS"
FINAL_FAILED = "FAILED"

PHASE_KEYS = ("phase", "research_phase", "family", "category")
TIMEFRAME_KEYS = ("timeframe", "tf", "bar_tf")
SYMBOL_KEYS = ("symbol", "ticker", "asset")
JOB_ID_KEYS = ("job_id", "strategy_id", "id", "run_id", "name", "strategy_name")
MICRO_EXIT_KEYS = ("micro_exit_variant", "micro_exit", "exit_family", "exit_name")
LOGIC_KEYS = ("logic_variant", "logic", "entry_name", "entry_family", "strategy_name")
FAST_KEYS = ("ema_fast", "fast", "fast_ema", "ema_fast_window")
SLOW_KEYS = ("ema_slow", "slow", "slow_ema", "ema_slow_window")
STRATEGY_FAMILY_KEYS = ("strategy_family", "family_name", "strategy_group")
REGIME_KEYS = ("regime_summary", "regime", "regime_name")

SUPPORTED_MICRO_EXITS = {
    "reverse_signal_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "adx_fade_exit",
    "atr_guard_exit",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_text_line(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")


def write_bootstrap_log(path: Path, message: str) -> None:
    append_text_line(path, f"{utc_now_iso()} {message}")


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def sniff_manifest(manifest_path: Path) -> Tuple[str, str, List[str]]:
    errors: List[str] = []
    for enc in DEFAULT_ENCODINGS:
        try:
            with manifest_path.open("r", encoding=enc, newline="") as fh:
                header = fh.readline()
                if not header:
                    raise ValueError("manifest is empty")
                counts = {
                    ",": header.count(","),
                    ";": header.count(";"),
                    "\t": header.count("\t"),
                    "|": header.count("|"),
                }
                delim = max(counts, key=counts.get)
                cols = next(csv.reader([header], delimiter=delim))
                if len(cols) < 2:
                    raise ValueError("unable to parse manifest header")
                return enc, delim, cols
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{enc}:{type(exc).__name__}:{exc}")
    raise RuntimeError("Unable to read manifest header: " + " | ".join(errors))


def iter_manifest_rows(manifest_path: Path, encoding: str, delimiter: str):
    with manifest_path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            clean = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
            yield row_index, clean


def pick_first_nonempty(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def row_matches_phase(row: Dict[str, str], phase: Optional[str]) -> bool:
    if not phase:
        return True
    target = phase.strip().lower()
    for key in PHASE_KEYS:
        if (row.get(key) or "").strip().lower() == target:
            return True
    return False


def normalize_job_row(row_index: int, row: Dict[str, str], default_symbol: str) -> Dict:
    timeframe = pick_first_nonempty(row, TIMEFRAME_KEYS)
    logic_variant = pick_first_nonempty(row, LOGIC_KEYS)
    micro_exit_variant = pick_first_nonempty(row, MICRO_EXIT_KEYS)
    symbol = pick_first_nonempty(row, SYMBOL_KEYS) or default_symbol
    strategy_family = pick_first_nonempty(row, STRATEGY_FAMILY_KEYS) or "micro_exit_expansion"
    regime_summary = pick_first_nonempty(row, REGIME_KEYS)
    fast_text = pick_first_nonempty(row, FAST_KEYS)
    slow_text = pick_first_nonempty(row, SLOW_KEYS)

    if not timeframe:
        raise ValueError("missing timeframe")
    if not logic_variant:
        raise ValueError("missing logic_variant")
    if not micro_exit_variant:
        raise ValueError("missing micro_exit_variant")
    if micro_exit_variant not in SUPPORTED_MICRO_EXITS:
        raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")
    if not fast_text or not slow_text:
        raise ValueError("missing ema_fast/ema_slow")

    fast = int(float(fast_text))
    slow = int(float(slow_text))
    seed = pick_first_nonempty(row, JOB_ID_KEYS)
    if not seed:
        seed = json.dumps(
            {
                "row_index": row_index,
                "symbol": symbol,
                "timeframe": timeframe,
                "logic_variant": logic_variant,
                "micro_exit_variant": micro_exit_variant,
                "ema_fast": fast,
                "ema_slow": slow,
            },
            sort_keys=True,
        )
    job_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
    return {
        "row_index": row_index,
        "job_id": job_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "logic_variant": logic_variant,
        "micro_exit_variant": micro_exit_variant,
        "ema_fast": fast,
        "ema_slow": slow,
        "strategy_family": strategy_family,
        "regime_summary": regime_summary,
        "_raw_row": row,
    }


def load_completed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_completed_ids(path: Path, job_ids: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        for job_id in job_ids:
            fh.write(str(job_id) + "\n")


def append_state_rows(path: Path, rows: Sequence[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for target in ("time", "open", "high", "low", "close"):
        if target in cols:
            rename_map[cols[target]] = target
    df = df.rename(columns=rename_map)
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def read_feature_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" not in df.columns:
        raise ValueError(f"missing time column in {path}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def make_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rebuild_entries(
    feat_df: pd.DataFrame,
    logic_variant: str,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    adx = feat_df["adx_14"].fillna(0.0)
    pb_long = feat_df["pullback_to_ema20_long"].fillna(0).astype(bool)
    pb_short = feat_df["pullback_to_ema20_short"].fillna(0).astype(bool)

    if logic_variant == "ema_cross_long_only":
        signal = (ema_fast > ema_slow).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic_variant == "ema_cross_short_only":
        signal = (ema_fast < ema_slow).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    if logic_variant == "ema_pullback_long_only":
        signal = ((ema_fast > ema_slow) & pb_long).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic_variant == "ema_pullback_short_only":
        signal = ((ema_fast < ema_slow) & pb_short).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    if logic_variant == "adx_ema_cross_long_only":
        signal = ((ema_fast > ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic_variant == "adx_ema_cross_short_only":
        signal = ((ema_fast < ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    raise ValueError(f"unsupported logic_variant={logic_variant}")


def build_exit_series(
    micro_exit_variant: str,
    close: pd.Series,
    adx: pd.Series,
    atr_pct: pd.Series,
    atr_pct_roll20_mean: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    is_long: bool,
) -> pd.Series:
    if micro_exit_variant == "reverse_signal_exit":
        return ((ema_fast <= ema_slow) if is_long else (ema_fast >= ema_slow)).astype(bool)
    if micro_exit_variant == "price_cross_fast_exit":
        return ((close < ema_fast) if is_long else (close > ema_fast)).astype(bool)
    if micro_exit_variant == "price_cross_slow_exit":
        return ((close < ema_slow) if is_long else (close > ema_slow)).astype(bool)
    if micro_exit_variant == "adx_fade_exit":
        return (adx < 20.0).astype(bool)
    if micro_exit_variant == "atr_guard_exit":
        return (atr_pct > (atr_pct_roll20_mean * 1.25)).astype(bool)
    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")


def chunked(items: Sequence[Dict], size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def stat_map(obj, columns: Sequence[str], default=np.nan) -> Dict[str, float]:
    if isinstance(obj, pd.Series):
        return {str(col): float(obj.get(col, default)) for col in columns}
    try:
        scalar = float(obj)
        return {str(col): scalar for col in columns}
    except Exception:
        return {str(col): default for col in columns}


def extract_metrics(pf: vbt.Portfolio, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    total_return_map = stat_map(pf.total_return(), columns)
    max_dd_map = stat_map(pf.max_drawdown(), columns)
    trade_count_map = stat_map(pf.trades.count(), columns, default=0.0)
    win_rate_map = stat_map(pf.trades.win_rate(), columns)
    profit_factor_map = stat_map(pf.trades.profit_factor(), columns)
    expectancy_map = stat_map(pf.trades.expectancy(), columns)
    avg_win_map = stat_map(pf.trades.winning.pnl.mean(), columns)
    avg_loss_map = stat_map(pf.trades.losing.pnl.mean(), columns)

    out: Dict[str, Dict[str, float]] = {}
    for col in columns:
        win_rate_value = float(win_rate_map.get(col, np.nan))
        out[str(col)] = {
            "trade_count": int(round(float(trade_count_map.get(col, 0.0)))),
            "win_rate_pct": win_rate_value * 100.0 if win_rate_value == win_rate_value else np.nan,
            "profit_factor": float(profit_factor_map.get(col, np.nan)),
            "expectancy": float(expectancy_map.get(col, np.nan)),
            "pnl_sum": float(total_return_map.get(col, np.nan)),
            "max_drawdown": float(max_dd_map.get(col, np.nan)),
            "avg_win": float(avg_win_map.get(col, np.nan)),
            "avg_loss": float(avg_loss_map.get(col, np.nan)),
        }
    return out


def evaluate_batch(
    close: pd.Series,
    adx: pd.Series,
    atr_pct: pd.Series,
    atr_pct_roll20_mean: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    entries: pd.Series,
    jobs: Sequence[Dict],
    side: str,
) -> List[Dict]:
    columns = [str(job["job_id"]) for job in jobs]
    entries_bool = entries.fillna(False).astype(bool)
    entries_df = pd.concat([entries_bool.rename(col) for col in columns], axis=1)
    exits_df = pd.concat(
        [
            build_exit_series(
                micro_exit_variant=str(job["micro_exit_variant"]),
                close=close,
                adx=adx,
                atr_pct=atr_pct,
                atr_pct_roll20_mean=atr_pct_roll20_mean,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                is_long=(side == "LONG"),
            ).fillna(False).astype(bool).rename(str(job["job_id"]))
            for job in jobs
        ],
        axis=1,
    )

    if side == "LONG":
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_df,
            exits=exits_df,
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )
    else:
        false_df = pd.DataFrame(False, index=close.index, columns=columns)
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=false_df,
            exits=false_df,
            short_entries=entries_df,
            short_exits=exits_df,
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )

    metric_map = extract_metrics(pf, columns)
    created = utc_now_iso()
    rows: List[Dict] = []
    for job in jobs:
        jid = str(job["job_id"])
        rows.append(
            {
                "job_id": jid,
                "row_index": int(job["row_index"]),
                "symbol": str(job["symbol"]),
                "timeframe": str(job["timeframe"]),
                "strategy_family": str(job["strategy_family"]),
                "logic_variant": str(job["logic_variant"]),
                "side": side,
                "ema_fast": int(job["ema_fast"]),
                "ema_slow": int(job["ema_slow"]),
                "micro_exit_variant": str(job["micro_exit_variant"]),
                "regime_summary": str(job.get("regime_summary", "")),
                **metric_map[jid],
                "status": STATE_DONE,
                "created_at_utc": created,
            }
        )
    return rows


def append_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    header = not path.exists()
    frame.to_csv(path, mode="a", header=header, index=False)


def build_live_progress(
    path: Path,
    manifest_path: Path,
    outdir: Path,
    phase: str,
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    current_job_id: str,
    current_row_index: int,
    started_at_utc: str,
    started_perf: float,
) -> None:
    elapsed_sec = max(time.perf_counter() - started_perf, 0.0)
    remaining = max(total_jobs - done_jobs - failed_jobs - skipped_jobs, 0)
    rate = done_jobs / elapsed_sec if elapsed_sec > 0 and done_jobs > 0 else 0.0
    eta_min = (remaining / rate / 60.0) if rate > 0 else None
    payload = {
        "version": VERSION,
        "status": STATE_RUNNING if remaining > 0 else FINAL_DONE,
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "phase": phase,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "remaining_jobs": remaining,
        "progress_pct": round((done_jobs / total_jobs) * 100.0, 4) if total_jobs else 100.0,
        "current_job_id": current_job_id,
        "current_row_index": current_row_index,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
        "observed_elapsed_min": round(elapsed_sec / 60.0, 4),
        "eta_remaining_min": round(eta_min, 4) if eta_min is not None else None,
    }
    atomic_write_json(path, payload)


def build_summary(
    path: Path,
    manifest_path: Path,
    outdir: Path,
    phase: str,
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    started_at_utc: str,
    started_perf: float,
    final_status: str,
) -> None:
    elapsed_sec = max(time.perf_counter() - started_perf, 0.0)
    payload = {
        "version": VERSION,
        "final_status": final_status,
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "phase": phase,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "remaining_jobs": max(total_jobs - done_jobs - failed_jobs - skipped_jobs, 0),
        "progress_pct": round((done_jobs / total_jobs) * 100.0, 4) if total_jobs else 100.0,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
        "elapsed_sec": round(elapsed_sec, 4),
    }
    atomic_write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched coverage runner for micro_exit_expansion")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--phase", default="micro_exit_expansion")
    parser.add_argument("--symbol-default", default="XAUUSD")
    parser.add_argument("--portfolio-chunk-size", type=int, default=64)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--encoding-order", default=",".join(DEFAULT_ENCODINGS))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at_utc = utc_now_iso()
    started_perf = time.perf_counter()

    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "per_timeframe")
    ensure_dir(outdir / "job_logs")

    bootstrap_log_path = outdir / "bootstrap.log"
    state_path = outdir / "state.jsonl"
    completed_ids_path = outdir / "completed_ids.txt"
    live_progress_path = outdir / "live_progress.json"
    summary_path = outdir / "summary.json"

    write_bootstrap_log(bootstrap_log_path, f"START version={VERSION}")
    write_bootstrap_log(
        bootstrap_log_path,
        f"ARGS manifest={manifest_path} data_root={data_root} feature_root={feature_root} outdir={outdir} phase={args.phase} portfolio_chunk_size={args.portfolio_chunk_size}",
    )

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    encoding, delimiter, _ = sniff_manifest(manifest_path)
    completed_ids = set() if args.no_resume else load_completed_ids(completed_ids_path)

    jobs: List[Dict] = []
    bad_rows = 0
    for row_index, row in iter_manifest_rows(manifest_path, encoding, delimiter):
        if not row_matches_phase(row, args.phase):
            continue
        try:
            job = normalize_job_row(row_index=row_index, row=row, default_symbol=args.symbol_default)
            if job["job_id"] in completed_ids:
                continue
            jobs.append(job)
        except Exception as exc:  # noqa: BLE001
            bad_rows += 1
            write_bootstrap_log(bootstrap_log_path, f"ROW_SKIP row_index={row_index} reason={type(exc).__name__}:{exc}")

    total_jobs = len(jobs)
    done_jobs = 0
    failed_jobs = 0
    skipped_jobs = bad_rows

    build_live_progress(
        live_progress_path,
        manifest_path,
        outdir,
        args.phase,
        total_jobs,
        done_jobs,
        failed_jobs,
        skipped_jobs,
        current_job_id="",
        current_row_index=-1,
        started_at_utc=started_at_utc,
        started_perf=started_perf,
    )
    build_summary(
        summary_path,
        manifest_path,
        outdir,
        args.phase,
        total_jobs,
        done_jobs,
        failed_jobs,
        skipped_jobs,
        started_at_utc=started_at_utc,
        started_perf=started_perf,
        final_status=STATE_RUNNING,
    )

    if total_jobs == 0:
        build_summary(
            summary_path,
            manifest_path,
            outdir,
            args.phase,
            total_jobs,
            done_jobs,
            failed_jobs,
            skipped_jobs,
            started_at_utc=started_at_utc,
            started_perf=started_perf,
            final_status=FINAL_DONE,
        )
        write_bootstrap_log(bootstrap_log_path, "FINISH final_status=DONE total_jobs=0")
        return

    grouped: Dict[Tuple[str, str, int, int], List[Dict]] = {}
    for job in jobs:
        key = (
            str(job["timeframe"]),
            str(job["logic_variant"]),
            int(job["ema_fast"]),
            int(job["ema_slow"]),
        )
        grouped.setdefault(key, []).append(job)

    write_bootstrap_log(bootstrap_log_path, f"LOAD_DONE total_jobs={total_jobs} groups={len(grouped)} skipped_bad_rows={bad_rows}")

    timeframe_cache: Dict[str, Dict] = {}
    processed_groups = 0
    encountered_error = False

    for (timeframe, logic_variant, fast, slow), group_jobs in grouped.items():
        processed_groups += 1
        try:
            if timeframe not in timeframe_cache:
                symbol = str(group_jobs[0]["symbol"])
                price_path = data_root / f"{symbol}_{timeframe}.parquet"
                feature_path = feature_root / f"{symbol}_{timeframe}_base_features.parquet"
                if not price_path.exists():
                    raise FileNotFoundError(f"missing price file: {price_path}")
                if not feature_path.exists():
                    raise FileNotFoundError(f"missing feature file: {feature_path}")
                price_df = read_price_parquet(price_path)
                feat_df = read_feature_parquet(feature_path)
                close = price_df["close"].astype(float)
                atr_pct = feat_df["atr_pct_14"].fillna(0.0).astype(float)
                timeframe_cache[timeframe] = {
                    "symbol": symbol,
                    "price_df": price_df,
                    "feat_df": feat_df,
                    "close": close,
                    "adx": feat_df["adx_14"].fillna(0.0).astype(float),
                    "atr_pct": atr_pct,
                    "atr_pct_roll20_mean": atr_pct.rolling(20).mean().bfill().astype(float),
                    "ema_cache": {},
                }
                write_bootstrap_log(
                    bootstrap_log_path,
                    f"TIMEFRAME_LOAD timeframe={timeframe} price_path={price_path} feature_path={feature_path} rows={len(price_df)}",
                )

            tfc = timeframe_cache[timeframe]
            ema_cache = tfc["ema_cache"]
            if fast not in ema_cache:
                ema_cache[fast] = make_ema(tfc["close"], fast)
            if slow not in ema_cache:
                ema_cache[slow] = make_ema(tfc["close"], slow)
            ema_fast = ema_cache[fast]
            ema_slow = ema_cache[slow]

            long_entries, short_entries = rebuild_entries(tfc["feat_df"], logic_variant, ema_fast, ema_slow)
            if long_entries is not None:
                side = "LONG"
                entry_series = long_entries
            elif short_entries is not None:
                side = "SHORT"
                entry_series = short_entries
            else:
                raise RuntimeError(f"Unable to determine side for logic_variant={logic_variant}")

            result_csv = outdir / "per_timeframe" / f"{tfc['symbol']}_{timeframe}_micro_exit_coverage_results.csv"
            for chunk in chunked(group_jobs, max(1, int(args.portfolio_chunk_size))):
                rows = evaluate_batch(
                    close=tfc["close"],
                    adx=tfc["adx"],
                    atr_pct=tfc["atr_pct"],
                    atr_pct_roll20_mean=tfc["atr_pct_roll20_mean"],
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    entries=entry_series,
                    jobs=chunk,
                    side=side,
                )
                append_frame_csv(result_csv, pd.DataFrame(rows))
                append_completed_ids(completed_ids_path, [row["job_id"] for row in rows])
                state_rows = []
                for row in rows:
                    state_rows.append(
                        {
                            "updated_at_utc": utc_now_iso(),
                            "status": STATE_DONE,
                            "job_id": row["job_id"],
                            "row_index": row["row_index"],
                            "timeframe": row["timeframe"],
                            "logic_variant": row["logic_variant"],
                            "micro_exit_variant": row["micro_exit_variant"],
                            "ema_fast": row["ema_fast"],
                            "ema_slow": row["ema_slow"],
                            "result_csv": str(result_csv),
                        }
                    )
                append_state_rows(state_path, state_rows)
                done_jobs += len(rows)
                completed_ids.update(row["job_id"] for row in rows)

            write_bootstrap_log(
                bootstrap_log_path,
                f"GROUP_DONE groups_completed={processed_groups}/{len(grouped)} timeframe={timeframe} logic={logic_variant} ema_fast={fast} ema_slow={slow} jobs={len(group_jobs)} done_jobs={done_jobs}/{total_jobs}",
            )
        except Exception as exc:  # noqa: BLE001
            encountered_error = True
            write_bootstrap_log(
                bootstrap_log_path,
                f"GROUP_FAIL groups_completed={processed_groups}/{len(grouped)} timeframe={timeframe} logic={logic_variant} ema_fast={fast} ema_slow={slow} jobs={len(group_jobs)} error={type(exc).__name__}:{exc}",
            )
            write_bootstrap_log(bootstrap_log_path, traceback.format_exc())
            # Fallback to single-job evaluation so long runs keep moving
            fallback_success = 0
            for job in group_jobs:
                try:
                    if timeframe not in timeframe_cache:
                        raise RuntimeError("timeframe cache unavailable after group failure")
                    tfc = timeframe_cache[timeframe]
                    ema_cache = tfc["ema_cache"]
                    ema_fast = ema_cache.get(fast) or make_ema(tfc["close"], fast)
                    ema_slow = ema_cache.get(slow) or make_ema(tfc["close"], slow)
                    long_entries, short_entries = rebuild_entries(tfc["feat_df"], logic_variant, ema_fast, ema_slow)
                    if long_entries is not None:
                        side = "LONG"
                        entry_series = long_entries
                    else:
                        side = "SHORT"
                        entry_series = short_entries
                    result_csv = outdir / "per_timeframe" / f"{tfc['symbol']}_{timeframe}_micro_exit_coverage_results.csv"
                    rows = evaluate_batch(
                        close=tfc["close"],
                        adx=tfc["adx"],
                        atr_pct=tfc["atr_pct"],
                        atr_pct_roll20_mean=tfc["atr_pct_roll20_mean"],
                        ema_fast=ema_fast,
                        ema_slow=ema_slow,
                        entries=entry_series,
                        jobs=[job],
                        side=side,
                    )
                    append_frame_csv(result_csv, pd.DataFrame(rows))
                    append_completed_ids(completed_ids_path, [rows[0]["job_id"]])
                    append_state_rows(
                        state_path,
                        [{
                            "updated_at_utc": utc_now_iso(),
                            "status": STATE_DONE,
                            "job_id": rows[0]["job_id"],
                            "row_index": rows[0]["row_index"],
                            "timeframe": rows[0]["timeframe"],
                            "logic_variant": rows[0]["logic_variant"],
                            "micro_exit_variant": rows[0]["micro_exit_variant"],
                            "ema_fast": rows[0]["ema_fast"],
                            "ema_slow": rows[0]["ema_slow"],
                            "result_csv": str(result_csv),
                            "mode": "fallback_single",
                        }],
                    )
                    done_jobs += 1
                    fallback_success += 1
                except Exception as inner_exc:  # noqa: BLE001
                    failed_jobs += 1
                    append_state_rows(
                        state_path,
                        [{
                            "updated_at_utc": utc_now_iso(),
                            "status": STATE_FAILED,
                            "job_id": job["job_id"],
                            "row_index": job["row_index"],
                            "timeframe": job["timeframe"],
                            "logic_variant": job["logic_variant"],
                            "micro_exit_variant": job["micro_exit_variant"],
                            "ema_fast": job["ema_fast"],
                            "ema_slow": job["ema_slow"],
                            "error_message": f"{type(inner_exc).__name__}: {inner_exc}",
                            "mode": "fallback_single",
                        }],
                    )
                    write_bootstrap_log(
                        bootstrap_log_path,
                        f"JOB_FAIL job_id={job['job_id']} row_index={job['row_index']} error={type(inner_exc).__name__}:{inner_exc}",
                    )
                    if not args.continue_on_error:
                        build_live_progress(
                            live_progress_path,
                            manifest_path,
                            outdir,
                            args.phase,
                            total_jobs,
                            done_jobs,
                            failed_jobs,
                            skipped_jobs,
                            current_job_id=job["job_id"],
                            current_row_index=int(job["row_index"]),
                            started_at_utc=started_at_utc,
                            started_perf=started_perf,
                        )
                        build_summary(
                            summary_path,
                            manifest_path,
                            outdir,
                            args.phase,
                            total_jobs,
                            done_jobs,
                            failed_jobs,
                            skipped_jobs,
                            started_at_utc=started_at_utc,
                            started_perf=started_perf,
                            final_status=FINAL_FAILED,
                        )
                        raise SystemExit(1)
            write_bootstrap_log(
                bootstrap_log_path,
                f"GROUP_FALLBACK_DONE timeframe={timeframe} logic={logic_variant} ema_fast={fast} ema_slow={slow} fallback_success={fallback_success} group_jobs={len(group_jobs)}",
            )

        if processed_groups % max(args.progress_every_groups, 1) == 0 or processed_groups == len(grouped):
            current_job = group_jobs[-1]
            build_live_progress(
                live_progress_path,
                manifest_path,
                outdir,
                args.phase,
                total_jobs,
                done_jobs,
                failed_jobs,
                skipped_jobs,
                current_job_id=str(current_job["job_id"]),
                current_row_index=int(current_job["row_index"]),
                started_at_utc=started_at_utc,
                started_perf=started_perf,
            )
            build_summary(
                summary_path,
                manifest_path,
                outdir,
                args.phase,
                total_jobs,
                done_jobs,
                failed_jobs,
                skipped_jobs,
                started_at_utc=started_at_utc,
                started_perf=started_perf,
                final_status=STATE_RUNNING,
            )

    final_status = FINAL_DONE_WITH_ERRORS if encountered_error or failed_jobs > 0 else FINAL_DONE
    build_live_progress(
        live_progress_path,
        manifest_path,
        outdir,
        args.phase,
        total_jobs,
        done_jobs,
        failed_jobs,
        skipped_jobs,
        current_job_id="",
        current_row_index=-1,
        started_at_utc=started_at_utc,
        started_perf=started_perf,
    )
    build_summary(
        summary_path,
        manifest_path,
        outdir,
        args.phase,
        total_jobs,
        done_jobs,
        failed_jobs,
        skipped_jobs,
        started_at_utc=started_at_utc,
        started_perf=started_perf,
        final_status=final_status,
    )
    write_bootstrap_log(
        bootstrap_log_path,
        f"FINISH final_status={final_status} total_jobs={total_jobs} done_jobs={done_jobs} failed_jobs={failed_jobs} skipped_jobs={skipped_jobs}",
    )


if __name__ == "__main__":
    main()
