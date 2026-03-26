#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Code Name : run_vectorbt_micro_exit_coverage_batched_v1_0_3.py
Version   : v1.0.6
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_3.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_3.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outdir C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_3\run_0 --phase micro_exit_expansion --portfolio-chunk-size 64 --progress-every-groups 1 --continue-on-error

Changelog
---------
[v1.0.4]
1) Expand micro-exit support set for phase micro_exit_expansion without changing strategy-family semantics.
2) Keep medium/relaxed ambiguity policy unchanged (still classified as unsupported).
3) Add deterministic exit mappings for newly supported micro-exit variants.

[v1.0.6]
1) Enable bos_continuation in safe rollout defaults for coverage expansion.
2) Keep swing_pullback and market_structure_continuation disabled by default.
3) Keep ambiguous schema policy unchanged (no relaxed and no medium guessing).

[v1.0.5]
1) Add preflight-ready strategy family model for ema_trend_continuation, bos_continuation, swing_pullback, market_structure_continuation.
2) Add family dispatch layer and dedicated family entry builders.
3) Add per-family required feature contract validation before execution.
4) Add safe rollout control for enabled strategy families.
5) Add test mode controls via max groups and max jobs.

[v1.0.3]
1) Rebuild the runner around a 2-stage architecture: compile/validate first, execute second.
2) Stop guessing ambiguous component semantics such as swing_variant=medium; classify them as UNSUPPORTED_CONFIG.
3) Add manifest preflight outputs so invalid/unsupported rows never enter heavy compute.
4) Make job_id collision-safe by hashing the full normalized component spec.
5) Keep only one timeframe bundle in memory at a time and release it explicitly on timeframe switches.
6) Replace the old progress math with separate overall/execution progress fields to avoid misleading ETA from fast failures.
7) Remove the pandas fillna/shift warning path by forcing boolean arrays before shifting.

Purpose
-------
Production-safe compatibility runner for the componentized coverage manifest.
The goal of this version is correctness first:
- classify every manifest row deterministically,
- execute only rows that are valid and fully supported,
- reduce wasted RAM/time caused by invalid jobs entering the execution path.

Design Notes
------------
- This file is intentionally conservative. Unsupported semantics are logged and classified, not guessed.
- GPU is not the active path here. The optimization target is RAM discipline and deterministic classification.
- M1 is very large, so only required columns are loaded and only one timeframe is kept resident.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.6"

STATUS_DONE = "DONE"
STATUS_RESUMED = "RESUMED"
STATUS_INVALID_PARAM = "INVALID_PARAM"
STATUS_UNSUPPORTED_CONFIG = "UNSUPPORTED_CONFIG"
STATUS_MISSING_FEATURE_COLUMN = "MISSING_FEATURE_COLUMN"
STATUS_EXECUTION_ERROR = "EXECUTION_ERROR"
STATUS_SKIPPED_ROW = "SKIPPED_ROW"

FINAL_DONE = "DONE"
FINAL_DONE_WITH_WARNINGS = "DONE_WITH_WARNINGS"
FINAL_FAILED = "FAILED"

TIMEFRAME_ORDER = {
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

STRICTNESS_THRESHOLD = {
    "loose": 18.0,
    "light": 18.0,
    "medium": 25.0,
    "mid": 25.0,
    "strict": 30.0,
    "hard": 30.0,
}

PULLBACK_RATIO_BANDS = {
    "narrow": (0.00, 0.35),
    "medium": (0.15, 0.70),
    "wide": (0.35, 1.25),
}

LONG_SWING_ALIASES = {"long", "bull", "bullish", "buy", "up"}
SHORT_SWING_ALIASES = {"short", "bear", "bearish", "sell", "down"}
AMBIGUOUS_SWING_ALIASES = {"medium", "mid", "neutral", "both", "balanced"}
KNOWN_STRATEGY_FAMILIES = {
    "adx_ema_trend_continuation",
    "ema_trend_continuation",
    "bos_continuation",
    "swing_pullback",
    "market_structure_continuation",
}
DEFAULT_ENABLED_STRATEGY_FAMILIES = {
    "adx_ema_trend_continuation",
    "ema_trend_continuation",
    "bos_continuation",
}
SUPPORTED_ENTRY_VARIANTS = {"confirm_entry"}
SUPPORTED_MICRO_EXIT_VARIANTS = {
    "",
    "none",
    "no_micro_exit",
    "hold_until_end",
    "adx_fade_exit",
    "fast_invalidation",
    "momentum_fade",
    "atr_guard_exit",
    "atr_trailing_exit",
    "break_even_ladder_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "reverse_signal_exit",
    "time_decay_exit",
    "structure_fail_exit",
    "partial_tp_runner_exit",
    "volatility_crush_exit",
    "baseline_pending_exit",
}

DEFAULT_RESULT_COLUMNS = [
    "job_id",
    "manifest_id",
    "manifest_rank",
    "row_index",
    "version",
    "phase",
    "symbol",
    "timeframe",
    "strategy_family",
    "logic_variant",
    "logic_strictness",
    "swing_variant",
    "pullback_zone_variant",
    "entry_variant",
    "micro_exit_variant",
    "management_variant",
    "regime_variant",
    "robustness_variant",
    "side",
    "ema_fast",
    "ema_slow",
    "entry_count",
    "trade_count",
    "wins",
    "losses",
    "win_rate_pct",
    "pnl_sum",
    "avg_pnl",
    "profit_factor",
    "payoff_ratio",
    "max_consecutive_losses",
    "avg_bars_held",
    "median_bars_held",
    "first_trade_time",
    "last_trade_time",
    "status",
    "status_reason",
    "error_reason",
]

FAMILY_REQUIRED_FEATURES: Dict[str, set[str]] = {
    "adx_ema_trend_continuation": {"adx_14"},
    "ema_trend_continuation": {"adx_14"},
    "bos_continuation": {"adx_14"},
    "swing_pullback": {"adx_14"},
    "market_structure_continuation": {"adx_14"},
}


@dataclass(frozen=True)
class Job:
    row_index: int
    manifest_rank: int
    manifest_id: str
    symbol: str
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
    phase: str
    logic_variant: str
    job_id: str
    source_row: Dict[str, str]


@dataclass(frozen=True)
class PreflightResult:
    job: Optional[Job]
    status: str
    reason: str


@dataclass
class GroupEvaluation:
    side: str
    entries: pd.Series


@dataclass
class TimeframeBundle:
    symbol: str
    timeframe: str
    price_df: pd.DataFrame
    close: pd.Series
    adx: pd.Series
    ema_cache: Dict[int, pd.Series]


@dataclass
class CounterState:
    manifest_total: int = 0
    preflight_valid: int = 0
    preflight_invalid: int = 0
    preflight_unsupported: int = 0
    preflight_skipped: int = 0
    execution_total: int = 0
    execution_done: int = 0
    execution_resumed: int = 0
    execution_error: int = 0
    execution_missing_feature: int = 0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_text(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def write_bootstrap_log(path: Path, message: str) -> None:
    append_text(path, f"{now_utc_iso()} {message}")


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def sanitize_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def to_int(value: object, field_name: str) -> int:
    text = sanitize_text(value)
    if not text:
        raise ValueError(f"missing {field_name}")
    try:
        return int(float(text))
    except Exception as exc:
        raise ValueError(f"invalid {field_name}={text}") from exc


def canonical_logic_variant(strategy_family: str, logic_strictness: str, swing_variant: str, pullback_zone_variant: str, entry_variant: str) -> str:
    return "__".join([
        sanitize_text(strategy_family, "unknown_family"),
        sanitize_text(logic_strictness, "unknown_strictness"),
        sanitize_text(swing_variant, "unknown_swing"),
        sanitize_text(pullback_zone_variant, "unknown_pullback"),
        sanitize_text(entry_variant, "unknown_entry"),
    ])


def stable_job_id(*parts: object) -> str:
    raw = "__".join(sanitize_text(x) for x in parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def empty_stats() -> Dict[str, object]:
    return {
        "entry_count": 0,
        "trade_count": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "pnl_sum": 0.0,
        "avg_pnl": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "max_consecutive_losses": 0,
        "avg_bars_held": 0.0,
        "median_bars_held": 0.0,
        "first_trade_time": "",
        "last_trade_time": "",
    }


def build_result_row(job: Job, status: str, reason: str = "", side: str = "", stats: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    payload = {
        "job_id": job.job_id,
        "manifest_id": job.manifest_id,
        "manifest_rank": job.manifest_rank,
        "row_index": job.row_index,
        "version": VERSION,
        "phase": job.phase,
        "symbol": job.symbol,
        "timeframe": job.timeframe,
        "strategy_family": job.strategy_family,
        "logic_variant": job.logic_variant,
        "logic_strictness": job.logic_strictness,
        "swing_variant": job.swing_variant,
        "pullback_zone_variant": job.pullback_zone_variant,
        "entry_variant": job.entry_variant,
        "micro_exit_variant": job.micro_exit_variant,
        "management_variant": job.management_variant,
        "regime_variant": job.regime_variant,
        "robustness_variant": job.robustness_variant,
        "side": side,
        "ema_fast": job.ema_fast,
        "ema_slow": job.ema_slow,
        "status": status,
        "status_reason": reason,
        "error_reason": reason,
    }
    payload.update(empty_stats())
    if stats:
        payload.update(stats)
    return payload


def append_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8")


def iter_manifest_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def normalize_job_row(row_index: int, row: Dict[str, str], default_symbol: str, default_phase: str) -> Job:
    timeframe = sanitize_text(row.get("timeframe")).upper()
    if not timeframe:
        raise ValueError("missing timeframe")

    strategy_family = sanitize_text(row.get("strategy_family"), "").lower()
    logic_strictness = sanitize_text(row.get("logic_strictness"), "").lower()
    swing_variant = sanitize_text(row.get("swing_variant"), "").lower()
    pullback_zone_variant = sanitize_text(row.get("pullback_zone_variant"), "").lower()
    entry_variant = sanitize_text(row.get("entry_variant"), "").lower()
    micro_exit_variant = sanitize_text(row.get("micro_exit_variant"), "no_micro_exit").lower()
    management_variant = sanitize_text(row.get("management_variant"), "").lower()
    regime_variant = sanitize_text(row.get("regime_variant"), "").lower()
    robustness_variant = sanitize_text(row.get("robustness_variant"), "").lower()
    symbol = sanitize_text(row.get("symbol"), default_symbol)
    phase = sanitize_text(row.get("phase"), default_phase)
    manifest_id = sanitize_text(row.get("manifest_id"), f"row_{row_index}")
    manifest_rank = to_int(row.get("manifest_rank", row_index + 1), "manifest_rank")
    ema_fast = to_int(row.get("ema_fast"), "ema_fast")
    ema_slow = to_int(row.get("ema_slow"), "ema_slow")

    if not strategy_family:
        raise ValueError("missing strategy_family")
    if not logic_strictness:
        raise ValueError("missing logic_strictness")
    if not swing_variant:
        raise ValueError("missing swing_variant")
    if not pullback_zone_variant:
        raise ValueError("missing pullback_zone_variant")
    if not entry_variant:
        raise ValueError("missing entry_variant")
    if ema_fast <= 0 or ema_slow <= 0:
        raise ValueError("ema periods must be positive")
    if ema_fast >= ema_slow:
        raise ValueError(f"invalid ema pair fast={ema_fast} slow={ema_slow}")
    if timeframe not in TIMEFRAME_ORDER:
        raise ValueError(f"unsupported timeframe={timeframe}")

    logic_variant = canonical_logic_variant(strategy_family, logic_strictness, swing_variant, pullback_zone_variant, entry_variant)
    job_id = stable_job_id(
        manifest_id,
        timeframe,
        strategy_family,
        logic_strictness,
        swing_variant,
        pullback_zone_variant,
        entry_variant,
        micro_exit_variant,
        management_variant,
        regime_variant,
        robustness_variant,
        ema_fast,
        ema_slow,
    )
    return Job(
        row_index=row_index,
        manifest_rank=manifest_rank,
        manifest_id=manifest_id,
        symbol=symbol,
        timeframe=timeframe,
        strategy_family=strategy_family,
        logic_strictness=logic_strictness,
        swing_variant=swing_variant,
        pullback_zone_variant=pullback_zone_variant,
        entry_variant=entry_variant,
        micro_exit_variant=micro_exit_variant,
        management_variant=management_variant,
        regime_variant=regime_variant,
        robustness_variant=robustness_variant,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        phase=phase,
        logic_variant=logic_variant,
        job_id=job_id,
        source_row={},
    )


def parse_enabled_strategy_families(raw: str) -> set[str]:
    items = [sanitize_text(x).lower() for x in sanitize_text(raw).split(",") if sanitize_text(x)]
    return set(items) if items else set(DEFAULT_ENABLED_STRATEGY_FAMILIES)


def preflight_validate(job: Job, enabled_strategy_families: set[str]) -> PreflightResult:
    if job.strategy_family not in KNOWN_STRATEGY_FAMILIES:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported strategy_family={job.strategy_family}")
    if job.strategy_family not in enabled_strategy_families:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"strategy_family_disabled={job.strategy_family}")
    if job.logic_strictness not in STRICTNESS_THRESHOLD:
        if job.logic_strictness in {"relaxed", "relax"}:
            return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"ambiguous logic_strictness={job.logic_strictness}")
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported logic_strictness={job.logic_strictness}")
    if job.swing_variant in AMBIGUOUS_SWING_ALIASES:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"ambiguous swing_variant={job.swing_variant}")
    if job.swing_variant not in LONG_SWING_ALIASES and job.swing_variant not in SHORT_SWING_ALIASES:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported swing_variant={job.swing_variant}")
    if job.pullback_zone_variant not in PULLBACK_RATIO_BANDS:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported pullback_zone_variant={job.pullback_zone_variant}")
    if job.entry_variant not in SUPPORTED_ENTRY_VARIANTS:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported entry_variant={job.entry_variant}")
    if job.micro_exit_variant not in SUPPORTED_MICRO_EXIT_VARIANTS:
        return PreflightResult(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=f"unsupported micro_exit_variant={job.micro_exit_variant}")
    return PreflightResult(job=job, status="VALID", reason="")


def load_completed_ids(result_csv: Path) -> set[str]:
    if not result_csv.exists():
        return set()
    try:
        df = pd.read_csv(result_csv, usecols=["job_id", "status"])
        keep = df["status"].isin([STATUS_DONE, STATUS_RESUMED])
        return set(df.loc[keep, "job_id"].dropna().astype(str).tolist())
    except Exception:
        return set()


def read_price_parquet(path: Path) -> pd.DataFrame:
    columns = ["time", "close"]
    df = pd.read_parquet(path, columns=columns).copy()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"price parquet missing columns={missing} path={path}")
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def read_feature_adx(path: Path, target_times: pd.Series) -> pd.Series:
    df = pd.read_parquet(path, columns=["time", "adx_14"]).copy()
    missing = [c for c in ["time", "adx_14"] if c not in df.columns]
    if missing:
        raise KeyError(",".join(missing))
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df["adx_14"] = pd.to_numeric(df["adx_14"], errors="coerce").astype("float32")
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
    adx_by_time = df.set_index("time")["adx_14"]
    aligned = adx_by_time.reindex(pd.to_datetime(target_times), copy=False)
    out = pd.to_numeric(aligned, errors="coerce").fillna(0.0).astype("float32")
    return pd.Series(out.to_numpy(dtype=np.float32), index=range(len(out)), dtype="float32")


def make_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().astype("float32")


def bars_since(condition: pd.Series) -> pd.Series:
    values = pd.Series(condition, copy=False).fillna(False).astype(bool).to_numpy(dtype=bool)
    out = np.full(len(values), np.nan, dtype=np.float32)
    last_true = -1
    for i, flag in enumerate(values):
        if flag:
            last_true = i
            out[i] = 0.0
        elif last_true >= 0:
            out[i] = float(i - last_true)
    return pd.Series(out, index=condition.index, dtype="float32")


def prior_bool(series: pd.Series) -> pd.Series:
    values = pd.Series(series, copy=False).fillna(False).astype(bool).to_numpy(dtype=bool)
    shifted = np.empty(len(values), dtype=bool)
    if len(values) == 0:
        return pd.Series(dtype=bool)
    shifted[0] = False
    shifted[1:] = values[:-1]
    return pd.Series(shifted, index=series.index, dtype=bool)


def build_directional_signal(
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    adx: pd.Series,
    adx_threshold: float,
    pullback_zone_variant: str,
    side: str,
) -> pd.Series:
    ema_spread = (ema_slow - ema_fast).abs().replace(0.0, np.nan)
    ratio_lo, ratio_hi = PULLBACK_RATIO_BANDS[pullback_zone_variant]

    if side == "LONG":
        bars_below_fast = bars_since(close < ema_fast)
        trend_ok = ((ema_fast > ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
        pullback_depth = ((ema_fast - close) / ema_spread).astype("float32")
        pullback_ok = (
            (close <= ema_fast)
            & (close >= ema_slow)
            & (pullback_depth >= ratio_lo)
            & (pullback_depth <= ratio_hi)
        ).fillna(False).astype(bool)
        reclaim = ((close > ema_fast) & (prior_bool(close <= ema_fast))).fillna(False).astype(bool)
        recent_pullback = (bars_below_fast <= 6).fillna(False).astype(bool)
        signal = trend_ok & reclaim & recent_pullback & prior_bool(pullback_ok)
        return signal & (~prior_bool(signal))

    bars_above_fast = bars_since(close > ema_fast)
    trend_ok = ((ema_fast < ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
    pullback_depth = ((close - ema_fast) / ema_spread).astype("float32")
    pullback_ok = (
        (close >= ema_fast)
        & (close <= ema_slow)
        & (pullback_depth >= ratio_lo)
        & (pullback_depth <= ratio_hi)
    ).fillna(False).astype(bool)
    reclaim = ((close < ema_fast) & (prior_bool(close >= ema_fast))).fillna(False).astype(bool)
    recent_pullback = (bars_above_fast <= 6).fillna(False).astype(bool)
    signal = trend_ok & reclaim & recent_pullback & prior_bool(pullback_ok)
    return signal & (~prior_bool(signal))


def build_component_entries(
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    adx: pd.Series,
    strategy_family: str,
    logic_strictness: str,
    swing_variant: str,
    pullback_zone_variant: str,
    entry_variant: str,
) -> GroupEvaluation:
    if strategy_family == "adx_ema_trend_continuation":
        if entry_variant != "confirm_entry":
            raise ValueError(f"unsupported entry_variant={entry_variant}")
        if pullback_zone_variant not in PULLBACK_RATIO_BANDS:
            raise ValueError(f"unsupported pullback_zone_variant={pullback_zone_variant}")
        adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
        if adx_threshold is None:
            raise ValueError(f"unsupported logic_strictness={logic_strictness}")
        if swing_variant in LONG_SWING_ALIASES:
            entries = build_directional_signal(close, ema_fast, ema_slow, adx, adx_threshold, pullback_zone_variant, "LONG")
            return GroupEvaluation(side="LONG", entries=entries)
        if swing_variant in SHORT_SWING_ALIASES:
            entries = build_directional_signal(close, ema_fast, ema_slow, adx, adx_threshold, pullback_zone_variant, "SHORT")
            return GroupEvaluation(side="SHORT", entries=entries)
        raise ValueError(f"unsupported swing_variant={swing_variant}")

    if strategy_family == "ema_trend_continuation":
        adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
        if adx_threshold is None:
            raise ValueError(f"unsupported logic_strictness={logic_strictness}")
        if swing_variant in LONG_SWING_ALIASES:
            trend = ((ema_fast > ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            entries = trend & (~prior_bool(trend))
            return GroupEvaluation(side="LONG", entries=entries)
        if swing_variant in SHORT_SWING_ALIASES:
            trend = ((ema_fast < ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            entries = trend & (~prior_bool(trend))
            return GroupEvaluation(side="SHORT", entries=entries)
        raise ValueError(f"unsupported swing_variant={swing_variant}")

    if strategy_family == "bos_continuation":
        adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
        if adx_threshold is None:
            raise ValueError(f"unsupported logic_strictness={logic_strictness}")
        if swing_variant in LONG_SWING_ALIASES:
            breakout = (close > close.rolling(5, min_periods=5).max().shift(1)).fillna(False).astype(bool)
            trend = ((ema_fast > ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            signal = breakout & trend
            return GroupEvaluation(side="LONG", entries=signal & (~prior_bool(signal)))
        if swing_variant in SHORT_SWING_ALIASES:
            breakdown = (close < close.rolling(5, min_periods=5).min().shift(1)).fillna(False).astype(bool)
            trend = ((ema_fast < ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            signal = breakdown & trend
            return GroupEvaluation(side="SHORT", entries=signal & (~prior_bool(signal)))
        raise ValueError(f"unsupported swing_variant={swing_variant}")

    if strategy_family == "swing_pullback":
        adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
        if adx_threshold is None:
            raise ValueError(f"unsupported logic_strictness={logic_strictness}")
        if swing_variant in LONG_SWING_ALIASES:
            pullback = ((close <= ema_fast) & (close >= ema_slow)).fillna(False).astype(bool)
            trend = ((ema_fast > ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            reclaim = ((close > ema_fast) & prior_bool(close <= ema_fast)).fillna(False).astype(bool)
            signal = trend & pullback & reclaim
            return GroupEvaluation(side="LONG", entries=signal & (~prior_bool(signal)))
        if swing_variant in SHORT_SWING_ALIASES:
            pullback = ((close >= ema_fast) & (close <= ema_slow)).fillna(False).astype(bool)
            trend = ((ema_fast < ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            reclaim = ((close < ema_fast) & prior_bool(close >= ema_fast)).fillna(False).astype(bool)
            signal = trend & pullback & reclaim
            return GroupEvaluation(side="SHORT", entries=signal & (~prior_bool(signal)))
        raise ValueError(f"unsupported swing_variant={swing_variant}")

    if strategy_family == "market_structure_continuation":
        adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
        if adx_threshold is None:
            raise ValueError(f"unsupported logic_strictness={logic_strictness}")
        if swing_variant in LONG_SWING_ALIASES:
            hh = (close > close.shift(3)).fillna(False).astype(bool)
            trend = ((ema_fast > ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            signal = hh & trend
            return GroupEvaluation(side="LONG", entries=signal & (~prior_bool(signal)))
        if swing_variant in SHORT_SWING_ALIASES:
            ll = (close < close.shift(3)).fillna(False).astype(bool)
            trend = ((ema_fast < ema_slow) & (adx >= adx_threshold)).fillna(False).astype(bool)
            signal = ll & trend
            return GroupEvaluation(side="SHORT", entries=signal & (~prior_bool(signal)))
        raise ValueError(f"unsupported swing_variant={swing_variant}")

    raise ValueError(f"unsupported strategy_family={strategy_family}")


def build_exit_series(
    micro_exit_variant: str,
    side: str,
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    adx: pd.Series,
) -> pd.Series:
    variant = micro_exit_variant.lower()
    side = side.upper()

    if variant in {"", "none", "no_micro_exit", "hold_until_end"}:
        return pd.Series(np.zeros(len(close), dtype=bool), index=close.index, dtype=bool)
    if variant == "baseline_pending_exit":
        return pd.Series(np.zeros(len(close), dtype=bool), index=close.index, dtype=bool)
    if variant == "adx_fade_exit":
        if side == "LONG":
            return ((adx < 20.0) | (close < ema_fast)).fillna(False).astype(bool)
        return ((adx < 20.0) | (close > ema_fast)).fillna(False).astype(bool)
    if variant in {"fast_invalidation", "price_cross_fast_exit", "break_even_ladder_exit"}:
        if side == "LONG":
            return (close < ema_fast).fillna(False).astype(bool)
        return (close > ema_fast).fillna(False).astype(bool)
    if variant == "price_cross_slow_exit":
        if side == "LONG":
            return (close < ema_slow).fillna(False).astype(bool)
        return (close > ema_slow).fillna(False).astype(bool)
    if variant in {"reverse_signal_exit", "structure_fail_exit"}:
        if side == "LONG":
            return ((ema_fast < ema_slow) | (close < ema_fast)).fillna(False).astype(bool)
        return ((ema_fast > ema_slow) | (close > ema_fast)).fillna(False).astype(bool)
    if variant == "volatility_crush_exit":
        return (adx < 18.0).fillna(False).astype(bool)
    if variant == "atr_guard_exit":
        if side == "LONG":
            return ((adx < 20.0) | (close < ema_fast)).fillna(False).astype(bool)
        return ((adx < 20.0) | (close > ema_fast)).fillna(False).astype(bool)
    if variant in {"atr_trailing_exit", "partial_tp_runner_exit", "time_decay_exit", "momentum_fade"}:
        fast_slope = ema_fast.diff().fillna(0.0)
        slow_slope = ema_slow.diff().fillna(0.0)
        if side == "LONG":
            return ((fast_slope <= 0.0) | (close < ema_fast) | (slow_slope <= 0.0)).fillna(False).astype(bool)
        return ((fast_slope >= 0.0) | (close > ema_fast) | (slow_slope >= 0.0)).fillna(False).astype(bool)
    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")


def simulate_trades(
    time_index: pd.Series,
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    side: str,
    *,
    entry_idx: Optional[np.ndarray] = None,
    close_np: Optional[np.ndarray] = None,
    time_np: Optional[np.ndarray] = None,
    entry_count_hint: Optional[int] = None,
) -> Dict[str, object]:
    if entry_idx is None:
        entry_idx = np.flatnonzero(pd.Series(entries, copy=False).fillna(False).astype(bool).to_numpy(dtype=bool))
    exit_flags = pd.Series(exits, copy=False).fillna(False).astype(bool).to_numpy(dtype=bool)
    if close_np is None:
        close_np = close.to_numpy(dtype=np.float32)
    if time_np is None:
        time_np = pd.to_datetime(time_index).to_numpy()

    trade_pnls: List[float] = []
    bars_held: List[int] = []
    wins = 0
    losses = 0
    max_consecutive_losses = 0
    current_loss_streak = 0
    first_trade_time: Optional[str] = None
    last_trade_time: Optional[str] = None
    last_exit_i = -1

    for i in entry_idx:
        if i >= len(close_np) - 1:
            continue
        start = i + 1
        if start <= last_exit_i:
            continue
        j = start
        while j < len(close_np) - 1 and not exit_flags[j]:
            j += 1
        exit_i = j if j < len(close_np) else len(close_np) - 1
        if exit_i <= i:
            continue

        entry_price = float(close_np[start])
        exit_price = float(close_np[exit_i])
        pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        trade_pnls.append(float(pnl))
        bars_held.append(int(max(exit_i - start, 0)))
        last_exit_i = exit_i

        entry_time = pd.Timestamp(time_np[start]).isoformat() if start < len(time_np) else None
        exit_time = pd.Timestamp(time_np[exit_i]).isoformat() if exit_i < len(time_np) else None
        if first_trade_time is None:
            first_trade_time = entry_time
        last_trade_time = exit_time

        if pnl > 0:
            wins += 1
            current_loss_streak = 0
        elif pnl < 0:
            losses += 1
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)

    trade_count = len(trade_pnls)
    pnl_sum = float(np.sum(trade_pnls)) if trade_pnls else 0.0
    avg_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    gross_profit = float(np.sum([x for x in trade_pnls if x > 0]))
    gross_loss_abs = abs(float(np.sum([x for x in trade_pnls if x < 0])))
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_win = float(np.mean([x for x in trade_pnls if x > 0])) if wins > 0 else 0.0
    avg_loss_abs = abs(float(np.mean([x for x in trade_pnls if x < 0]))) if losses > 0 else 0.0
    payoff_ratio = avg_win / avg_loss_abs if avg_loss_abs > 0 else (999.0 if avg_win > 0 else 0.0)
    win_rate_pct = (wins / trade_count * 100.0) if trade_count else 0.0
    avg_bars_held = float(np.mean(bars_held)) if bars_held else 0.0
    median_bars_held = float(np.median(bars_held)) if bars_held else 0.0

    return {
        "entry_count": int(entry_count_hint if entry_count_hint is not None else pd.Series(entries, copy=False).fillna(False).astype(bool).sum()),
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate_pct, 4),
        "pnl_sum": round(pnl_sum, 8),
        "avg_pnl": round(avg_pnl, 8),
        "profit_factor": round(profit_factor, 8),
        "payoff_ratio": round(payoff_ratio, 8),
        "max_consecutive_losses": max_consecutive_losses,
        "avg_bars_held": round(avg_bars_held, 4),
        "median_bars_held": round(median_bars_held, 4),
        "first_trade_time": first_trade_time or "",
        "last_trade_time": last_trade_time or "",
    }


def build_live_progress(
    outdir: Path,
    phase: str,
    current_timeframe: str,
    counters: CounterState,
    groups_total: int,
    groups_completed: int,
    start_ts: float,
) -> Dict[str, object]:
    elapsed_min = (time.perf_counter() - start_ts) / 60.0
    overall_processed = (
        counters.preflight_invalid
        + counters.preflight_unsupported
        + counters.preflight_skipped
        + counters.execution_done
        + counters.execution_resumed
        + counters.execution_error
        + counters.execution_missing_feature
    )
    overall_total = counters.manifest_total
    overall_progress_pct = (overall_processed / overall_total * 100.0) if overall_total else 100.0

    execution_processed = counters.execution_done + counters.execution_resumed + counters.execution_error + counters.execution_missing_feature
    execution_total = counters.execution_total
    execution_progress_pct = (execution_processed / execution_total * 100.0) if execution_total else 100.0
    execution_rate_per_min = (execution_processed / elapsed_min) if elapsed_min > 0 else 0.0
    execution_remaining = max(execution_total - execution_processed, 0)
    execution_eta_min = (execution_remaining / execution_rate_per_min) if execution_rate_per_min > 0 else None

    payload = {
        "version": VERSION,
        "phase": phase,
        "current_timeframe": current_timeframe,
        "manifest_total": overall_total,
        "preflight_valid": counters.preflight_valid,
        "preflight_invalid": counters.preflight_invalid,
        "preflight_unsupported": counters.preflight_unsupported,
        "preflight_skipped": counters.preflight_skipped,
        "execution_total": execution_total,
        "execution_done": counters.execution_done,
        "execution_resumed": counters.execution_resumed,
        "execution_error": counters.execution_error,
        "execution_missing_feature": counters.execution_missing_feature,
        "overall_processed": overall_processed,
        "overall_remaining": max(overall_total - overall_processed, 0),
        "overall_progress_pct": round(overall_progress_pct, 4),
        "execution_processed": execution_processed,
        "execution_remaining": execution_remaining,
        "execution_progress_pct": round(execution_progress_pct, 4),
        "observed_elapsed_min": round(elapsed_min, 4),
        "observed_execution_rate_jobs_per_min": round(execution_rate_per_min, 4),
        "execution_eta_remaining_min": round(execution_eta_min, 4) if execution_eta_min is not None else None,
        "groups_total": groups_total,
        "groups_completed": groups_completed,
        "updated_at_utc": now_utc_iso(),
        "outdir": str(outdir),
    }
    save_json(outdir / "live_progress.json", payload)
    return payload


def load_timeframe_bundle(symbol: str, timeframe: str, data_root: Path, feature_root: Path) -> TimeframeBundle:
    price_path = data_root / f"{symbol}_{timeframe}.parquet"
    feature_path = feature_root / f"{symbol}_{timeframe}_base_features.parquet"
    if not price_path.exists():
        raise FileNotFoundError(f"missing price file: {price_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature file: {feature_path}")

    price_df = read_price_parquet(price_path)
    adx = read_feature_adx(feature_path, price_df["time"])
    close = price_df["close"].astype("float32")
    return TimeframeBundle(
        symbol=symbol,
        timeframe=timeframe,
        price_df=price_df,
        close=close,
        adx=adx,
        ema_cache={},
    )


def safe_release_bundle(bundle: Optional[TimeframeBundle]) -> None:
    if bundle is None:
        return
    bundle.price_df = pd.DataFrame()
    bundle.close = pd.Series(dtype="float32")
    bundle.adx = pd.Series(dtype="float32")
    bundle.ema_cache.clear()
    gc.collect()


def execute_job_chunk(bundle: TimeframeBundle, jobs: Sequence[Job], side: str, entries: pd.Series) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    entries_bool = pd.Series(entries, copy=False).fillna(False).astype(bool)
    entry_idx = np.flatnonzero(entries_bool.to_numpy(dtype=bool))
    entry_count_hint = int(entries_bool.sum())
    close_np = bundle.close.to_numpy(dtype=np.float32)
    time_np = pd.to_datetime(bundle.price_df["time"]).to_numpy()
    for job in jobs:
        try:
            ema_fast = bundle.ema_cache[job.ema_fast]
            ema_slow = bundle.ema_cache[job.ema_slow]
            exits = build_exit_series(job.micro_exit_variant, side, bundle.close, ema_fast, ema_slow, bundle.adx)
            stats = simulate_trades(
                bundle.price_df["time"],
                bundle.close,
                entries,
                exits,
                side,
                entry_idx=entry_idx,
                close_np=close_np,
                time_np=time_np,
                entry_count_hint=entry_count_hint,
            )
            results.append(build_result_row(job=job, status=STATUS_DONE, reason="", side=side, stats=stats))
        except Exception as exc:
            results.append(build_result_row(job=job, status=STATUS_EXECUTION_ERROR, reason=f"{type(exc).__name__}:{exc}", side=side))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component-schema batched micro-exit coverage runner")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--phase", default="micro_exit_expansion")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--enabled-strategy-families", default="adx_ema_trend_continuation,ema_trend_continuation,bos_continuation")
    parser.add_argument("--portfolio-chunk-size", type=int, default=64)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    parser.add_argument("--preflight-flush-size", type=int, default=20000)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    ensure_dir(outdir)
    ensure_dir(outdir / "per_timeframe")

    bootstrap_log_path = outdir / "bootstrap.log"
    result_csv = outdir / "coverage_results_all.csv"
    preflight_csv = outdir / "preflight_results.csv"
    summary_json = outdir / "summary.json"
    state_json = outdir / "run_state.json"
    group_summary_csv = outdir / "summary_by_group.csv"
    status_summary_csv = outdir / "status_summary.csv"

    start_ts = time.perf_counter()
    started_at_utc = now_utc_iso()
    write_bootstrap_log(bootstrap_log_path, f"START version={VERSION} manifest={args.manifest} phase={args.phase}")
    enabled_strategy_families = parse_enabled_strategy_families(args.enabled_strategy_families)
    write_bootstrap_log(bootstrap_log_path, f"ENABLED_STRATEGY_FAMILIES {','.join(sorted(enabled_strategy_families))}")

    counters = CounterState(manifest_total=0)
    completed_ids = load_completed_ids(result_csv)

    valid_jobs: List[Job] = []
    preflight_rows_buf: List[Dict[str, object]] = []

    def flush_preflight_rows() -> None:
        nonlocal preflight_rows_buf
        if not preflight_rows_buf:
            return
        preflight_frame = pd.DataFrame(preflight_rows_buf).reindex(columns=DEFAULT_RESULT_COLUMNS)
        append_frame_csv(preflight_csv, preflight_frame)
        append_frame_csv(result_csv, preflight_frame)
        preflight_rows_buf = []
        gc.collect()

    last_row_index = -1
    for row_index, row in enumerate(iter_manifest_rows(args.manifest)):
        last_row_index = row_index
        try:
            job = normalize_job_row(row_index=row_index, row=row, default_symbol=args.symbol, default_phase=args.phase)
            if sanitize_text(job.phase).lower() != sanitize_text(args.phase).lower():
                counters.preflight_skipped += 1
                preflight_rows_buf.append(build_result_row(job=job, status=STATUS_SKIPPED_ROW, reason=f"phase_mismatch job_phase={job.phase} requested_phase={args.phase}"))
                continue
            check = preflight_validate(job, enabled_strategy_families=enabled_strategy_families)
            if check.status == "VALID":
                valid_jobs.append(job)
                counters.preflight_valid += 1
            elif check.status == STATUS_UNSUPPORTED_CONFIG:
                counters.preflight_unsupported += 1
                preflight_rows_buf.append(build_result_row(job=job, status=STATUS_UNSUPPORTED_CONFIG, reason=check.reason))
            else:
                counters.preflight_invalid += 1
                preflight_rows_buf.append(build_result_row(job=job, status=STATUS_INVALID_PARAM, reason=check.reason))
        except Exception as exc:
            counters.preflight_invalid += 1
            synthetic_job = Job(
                row_index=row_index,
                manifest_rank=row_index + 1,
                manifest_id=sanitize_text(row.get("manifest_id"), f"row_{row_index}"),
                symbol=sanitize_text(row.get("symbol"), args.symbol),
                timeframe=sanitize_text(row.get("timeframe"), ""),
                strategy_family=sanitize_text(row.get("strategy_family"), "").lower(),
                logic_strictness=sanitize_text(row.get("logic_strictness"), "").lower(),
                swing_variant=sanitize_text(row.get("swing_variant"), "").lower(),
                pullback_zone_variant=sanitize_text(row.get("pullback_zone_variant"), "").lower(),
                entry_variant=sanitize_text(row.get("entry_variant"), "").lower(),
                micro_exit_variant=sanitize_text(row.get("micro_exit_variant"), "").lower(),
                management_variant=sanitize_text(row.get("management_variant"), "").lower(),
                regime_variant=sanitize_text(row.get("regime_variant"), "").lower(),
                robustness_variant=sanitize_text(row.get("robustness_variant"), "").lower(),
                ema_fast=0,
                ema_slow=0,
                phase=sanitize_text(row.get("phase"), args.phase),
                logic_variant="",
                job_id=stable_job_id(row_index, sanitize_text(row.get("manifest_id"), f"row_{row_index}"), "invalid"),
                source_row={},
            )
            preflight_rows_buf.append(build_result_row(job=synthetic_job, status=STATUS_INVALID_PARAM, reason=f"{type(exc).__name__}:{exc}"))
            write_bootstrap_log(bootstrap_log_path, f"ROW_INVALID row_index={row_index} reason={type(exc).__name__}:{exc}")
        if len(preflight_rows_buf) >= max(1000, int(args.preflight_flush_size)):
            flush_preflight_rows()

    counters.manifest_total = last_row_index + 1 if last_row_index >= 0 else 0
    flush_preflight_rows()

    counters.execution_total = len(valid_jobs)
    write_bootstrap_log(
        bootstrap_log_path,
        f"PREFLIGHT_DONE manifest_total={counters.manifest_total} valid={counters.preflight_valid} invalid={counters.preflight_invalid} unsupported={counters.preflight_unsupported} skipped={counters.preflight_skipped}",
    )

    if not valid_jobs:
        status_counts = {
            STATUS_INVALID_PARAM: counters.preflight_invalid,
            STATUS_UNSUPPORTED_CONFIG: counters.preflight_unsupported,
            STATUS_SKIPPED_ROW: counters.preflight_skipped,
        }
        status_frame = pd.DataFrame([{"status": k, "count": v} for k, v in status_counts.items() if v > 0])
        if not status_frame.empty:
            status_frame.to_csv(status_summary_csv, index=False, encoding="utf-8")
        payload = {
            "version": VERSION,
            "manifest": str(args.manifest),
            "phase": args.phase,
            "started_at_utc": started_at_utc,
            "finished_at_utc": now_utc_iso(),
            "status": FINAL_DONE_WITH_WARNINGS,
            "manifest_total": counters.manifest_total,
            "preflight_valid": counters.preflight_valid,
            "preflight_invalid": counters.preflight_invalid,
            "preflight_unsupported": counters.preflight_unsupported,
            "preflight_skipped": counters.preflight_skipped,
            "execution_total": counters.execution_total,
            "execution_done": counters.execution_done,
            "execution_resumed": counters.execution_resumed,
            "execution_error": counters.execution_error,
            "execution_missing_feature": counters.execution_missing_feature,
            "coverage_results_all_csv": str(result_csv),
            "preflight_results_csv": str(preflight_csv),
        }
        save_json(summary_json, payload)
        save_json(state_json, payload)
        build_live_progress(outdir, args.phase, "", counters, groups_total=0, groups_completed=0, start_ts=start_ts)
        write_bootstrap_log(bootstrap_log_path, "FINISH final_status=DONE_WITH_WARNINGS no_valid_jobs=true")
        return

    valid_jobs = sorted(
        valid_jobs,
        key=lambda j: (
            TIMEFRAME_ORDER.get(j.timeframe, 99999),
            j.symbol,
            j.strategy_family,
            j.logic_variant,
            j.ema_fast,
            j.ema_slow,
            j.micro_exit_variant,
            j.manifest_rank,
        ),
    )

    grouped: Dict[Tuple[str, str, str, str, str, str, str, int, int], List[Job]] = {}
    for job in valid_jobs:
        key = (
            job.symbol,
            job.timeframe,
            job.strategy_family,
            job.logic_strictness,
            job.swing_variant,
            job.pullback_zone_variant,
            job.entry_variant,
            job.ema_fast,
            job.ema_slow,
        )
        grouped.setdefault(key, []).append(job)

    grouped_items = sorted(grouped.items(), key=lambda kv: kv[0])
    if int(args.max_groups) > 0:
        grouped_items = grouped_items[: int(args.max_groups)]
    groups_total = len(grouped_items)
    groups_completed = 0
    group_rows: List[Dict[str, object]] = []
    active_bundle: Optional[TimeframeBundle] = None
    active_bundle_key: Optional[Tuple[str, str]] = None
    remaining_job_budget = int(args.max_jobs) if int(args.max_jobs) > 0 else 0

    try:
        for group_key, group_jobs in grouped_items:
            groups_completed += 1
            symbol, timeframe, strategy_family, logic_strictness, swing_variant, pullback_zone_variant, entry_variant, ema_fast_n, ema_slow_n = group_key
            pending_jobs = [job for job in group_jobs if job.job_id not in completed_ids]
            if remaining_job_budget > 0:
                if remaining_job_budget <= 0:
                    pending_jobs = []
                else:
                    pending_jobs = pending_jobs[:remaining_job_budget]
                    remaining_job_budget -= len(pending_jobs)
            resumed_jobs = len(group_jobs) - len(pending_jobs)
            counters.execution_resumed += resumed_jobs
            required_feature_cols = FAMILY_REQUIRED_FEATURES.get(strategy_family, set())

            if active_bundle_key != (symbol, timeframe):
                safe_release_bundle(active_bundle)
                active_bundle = None
                active_bundle_key = (symbol, timeframe)
                try:
                    active_bundle = load_timeframe_bundle(symbol=symbol, timeframe=timeframe, data_root=args.data_root, feature_root=args.feature_root)
                    if "adx_14" in required_feature_cols and len(active_bundle.adx) == 0:
                        raise KeyError("adx_14")
                    write_bootstrap_log(
                        bootstrap_log_path,
                        f"TIMEFRAME_LOAD symbol={symbol} timeframe={timeframe} rows={len(active_bundle.price_df)} mode=single_timeframe_resident",
                    )
                except KeyError as exc:
                    missing_name = str(exc).strip("'\"")
                    reason = f"missing feature columns={missing_name}"
                    missing_rows = [build_result_row(job=j, status=STATUS_MISSING_FEATURE_COLUMN, reason=reason) for j in pending_jobs]
                    if missing_rows:
                        frame = pd.DataFrame(missing_rows).reindex(columns=DEFAULT_RESULT_COLUMNS)
                        append_frame_csv(result_csv, frame)
                        tf_csv = outdir / "per_timeframe" / f"{symbol}_{timeframe}_micro_exit_coverage_results.csv"
                        append_frame_csv(tf_csv, frame)
                        counters.execution_missing_feature += len(missing_rows)
                    group_rows.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "group_key": "|".join(map(str, group_key)),
                        "group_jobs": len(group_jobs),
                        "pending_jobs": len(pending_jobs),
                        "resumed_jobs": resumed_jobs,
                        "entry_count": 0,
                        "side": "",
                        "status": STATUS_MISSING_FEATURE_COLUMN,
                        "status_reason": reason,
                    })
                    write_bootstrap_log(bootstrap_log_path, f"GROUP_MISSING_FEATURE group={group_key} reason={reason}")
                    live = build_live_progress(outdir, args.phase, timeframe, counters, groups_total, groups_completed, start_ts)
                    if groups_completed % max(1, int(args.progress_every_groups)) == 0 or groups_completed == groups_total:
                        print(
                            f"[PROGRESS] groups={groups_completed}/{groups_total} overall={live['overall_progress_pct']:.2f}% execution={live['execution_progress_pct']:.2f}% done={counters.execution_done} resumed={counters.execution_resumed} exec_error={counters.execution_error} unsupported={counters.preflight_unsupported} invalid={counters.preflight_invalid}"
                        )
                    continue
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    reason = f"{type(exc).__name__}:{exc}"
                    error_rows = [build_result_row(job=j, status=STATUS_EXECUTION_ERROR, reason=reason) for j in pending_jobs]
                    if error_rows:
                        frame = pd.DataFrame(error_rows).reindex(columns=DEFAULT_RESULT_COLUMNS)
                        append_frame_csv(result_csv, frame)
                        tf_csv = outdir / "per_timeframe" / f"{symbol}_{timeframe}_micro_exit_coverage_results.csv"
                        append_frame_csv(tf_csv, frame)
                        counters.execution_error += len(error_rows)
                    group_rows.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "group_key": "|".join(map(str, group_key)),
                        "group_jobs": len(group_jobs),
                        "pending_jobs": len(pending_jobs),
                        "resumed_jobs": resumed_jobs,
                        "entry_count": 0,
                        "side": "",
                        "status": STATUS_EXECUTION_ERROR,
                        "status_reason": reason,
                    })
                    write_bootstrap_log(bootstrap_log_path, f"TIMEFRAME_LOAD_ERROR group={group_key} reason={reason}")
                    live = build_live_progress(outdir, args.phase, timeframe, counters, groups_total, groups_completed, start_ts)
                    if groups_completed % max(1, int(args.progress_every_groups)) == 0 or groups_completed == groups_total:
                        print(
                            f"[PROGRESS] groups={groups_completed}/{groups_total} overall={live['overall_progress_pct']:.2f}% execution={live['execution_progress_pct']:.2f}% done={counters.execution_done} resumed={counters.execution_resumed} exec_error={counters.execution_error} unsupported={counters.preflight_unsupported} invalid={counters.preflight_invalid}"
                        )
                    continue

            if active_bundle is None:
                raise RuntimeError("timeframe bundle is not available")

            if ema_fast_n not in active_bundle.ema_cache:
                active_bundle.ema_cache[ema_fast_n] = make_ema(active_bundle.close, ema_fast_n)
            if ema_slow_n not in active_bundle.ema_cache:
                active_bundle.ema_cache[ema_slow_n] = make_ema(active_bundle.close, ema_slow_n)

            ema_fast = active_bundle.ema_cache[ema_fast_n]
            ema_slow = active_bundle.ema_cache[ema_slow_n]

            try:
                group_eval = build_component_entries(
                    close=active_bundle.close,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    adx=active_bundle.adx,
                    strategy_family=strategy_family,
                    logic_strictness=logic_strictness,
                    swing_variant=swing_variant,
                    pullback_zone_variant=pullback_zone_variant,
                    entry_variant=entry_variant,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                error_rows = [build_result_row(job=j, status=STATUS_EXECUTION_ERROR, reason=reason) for j in pending_jobs]
                if error_rows:
                    frame = pd.DataFrame(error_rows).reindex(columns=DEFAULT_RESULT_COLUMNS)
                    append_frame_csv(result_csv, frame)
                    tf_csv = outdir / "per_timeframe" / f"{symbol}_{timeframe}_micro_exit_coverage_results.csv"
                    append_frame_csv(tf_csv, frame)
                    counters.execution_error += len(error_rows)
                group_rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "group_key": "|".join(map(str, group_key)),
                    "group_jobs": len(group_jobs),
                    "pending_jobs": len(pending_jobs),
                    "resumed_jobs": resumed_jobs,
                    "entry_count": 0,
                    "side": "",
                    "status": STATUS_EXECUTION_ERROR,
                    "status_reason": reason,
                })
                write_bootstrap_log(bootstrap_log_path, f"GROUP_BUILD_ERROR group={group_key} reason={reason}")
                live = build_live_progress(outdir, args.phase, timeframe, counters, groups_total, groups_completed, start_ts)
                if groups_completed % max(1, int(args.progress_every_groups)) == 0 or groups_completed == groups_total:
                    print(
                        f"[PROGRESS] groups={groups_completed}/{groups_total} overall={live['overall_progress_pct']:.2f}% execution={live['execution_progress_pct']:.2f}% done={counters.execution_done} resumed={counters.execution_resumed} exec_error={counters.execution_error} unsupported={counters.preflight_unsupported} invalid={counters.preflight_invalid}"
                    )
                continue

            for resumed_job in group_jobs:
                if resumed_job.job_id in completed_ids:
                    continue
            if pending_jobs:
                for job_chunk in chunked(pending_jobs, max(1, int(args.portfolio_chunk_size))):
                    result_rows = execute_job_chunk(bundle=active_bundle, jobs=job_chunk, side=group_eval.side, entries=group_eval.entries)
                    frame = pd.DataFrame(result_rows).reindex(columns=DEFAULT_RESULT_COLUMNS)
                    append_frame_csv(result_csv, frame)
                    tf_csv = outdir / "per_timeframe" / f"{symbol}_{timeframe}_micro_exit_coverage_results.csv"
                    append_frame_csv(tf_csv, frame)
                    counters.execution_done += int((frame["status"] == STATUS_DONE).sum())
                    counters.execution_error += int((frame["status"] == STATUS_EXECUTION_ERROR).sum())
                    for jid in frame["job_id"].astype(str).tolist():
                        completed_ids.add(jid)
                    del result_rows
                    del frame
                    gc.collect()
            group_rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "group_key": "|".join(map(str, group_key)),
                "group_jobs": len(group_jobs),
                "pending_jobs": len(pending_jobs),
                "resumed_jobs": resumed_jobs,
                "entry_count": int(group_eval.entries.sum()),
                "side": group_eval.side,
                "status": STATUS_DONE,
                "status_reason": "",
            })

            live = build_live_progress(outdir, args.phase, timeframe, counters, groups_total, groups_completed, start_ts)
            if groups_completed % max(1, int(args.progress_every_groups)) == 0 or groups_completed == groups_total:
                print(
                    f"[PROGRESS] groups={groups_completed}/{groups_total} overall={live['overall_progress_pct']:.2f}% execution={live['execution_progress_pct']:.2f}% done={counters.execution_done} resumed={counters.execution_resumed} exec_error={counters.execution_error} unsupported={counters.preflight_unsupported} invalid={counters.preflight_invalid}"
                )
    finally:
        safe_release_bundle(active_bundle)

    group_frame = pd.DataFrame(group_rows)
    if not group_frame.empty:
        group_frame.to_csv(group_summary_csv, index=False, encoding="utf-8")

    status_counts = {
        STATUS_INVALID_PARAM: counters.preflight_invalid,
        STATUS_UNSUPPORTED_CONFIG: counters.preflight_unsupported,
        STATUS_SKIPPED_ROW: counters.preflight_skipped,
        STATUS_DONE: counters.execution_done,
        STATUS_RESUMED: counters.execution_resumed,
        STATUS_EXECUTION_ERROR: counters.execution_error,
        STATUS_MISSING_FEATURE_COLUMN: counters.execution_missing_feature,
    }
    status_frame = pd.DataFrame([{"status": k, "count": v} for k, v in status_counts.items() if v > 0])
    if not status_frame.empty:
        status_frame.to_csv(status_summary_csv, index=False, encoding="utf-8")

    finished_at_utc = now_utc_iso()
    warnings_present = any(v > 0 for k, v in status_counts.items() if k not in {STATUS_DONE, STATUS_RESUMED})
    final_status = FINAL_DONE_WITH_WARNINGS if warnings_present else FINAL_DONE
    summary_payload = {
        "version": VERSION,
        "manifest": str(args.manifest),
        "phase": args.phase,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": final_status,
        "manifest_total": counters.manifest_total,
        "preflight_valid": counters.preflight_valid,
        "preflight_invalid": counters.preflight_invalid,
        "preflight_unsupported": counters.preflight_unsupported,
        "preflight_skipped": counters.preflight_skipped,
        "execution_total": counters.execution_total,
        "execution_done": counters.execution_done,
        "execution_resumed": counters.execution_resumed,
        "execution_error": counters.execution_error,
        "execution_missing_feature": counters.execution_missing_feature,
        "groups_total": groups_total,
        "groups_completed": groups_completed,
        "coverage_results_all_csv": str(result_csv),
        "preflight_results_csv": str(preflight_csv),
        "summary_by_group_csv": str(group_summary_csv),
        "status_summary_csv": str(status_summary_csv),
    }
    save_json(summary_json, summary_payload)
    save_json(state_json, summary_payload)
    build_live_progress(outdir, args.phase, active_bundle_key[1] if active_bundle_key else "", counters, groups_total, groups_completed, start_ts)
    write_bootstrap_log(
        bootstrap_log_path,
        f"FINISH final_status={final_status} manifest_total={counters.manifest_total} valid={counters.preflight_valid} invalid={counters.preflight_invalid} unsupported={counters.preflight_unsupported} done={counters.execution_done} resumed={counters.execution_resumed} execution_error={counters.execution_error} missing_feature={counters.execution_missing_feature}",
    )


if __name__ == "__main__":
    main()
