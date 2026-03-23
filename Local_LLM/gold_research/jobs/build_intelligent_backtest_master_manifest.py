# ==================================================================================================
# FILE: build_intelligent_backtest_master_manifest.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_intelligent_backtest_master_manifest.py
# VERSION: v1.0.2
#
# CHANGELOG:
# - v1.0.2
#   1) Rewrite as streaming manifest writer
#   2) Remove full in-memory job accumulation
#   3) Add progress log every N jobs
#   4) Add duplicate detection during streaming
#   5) Write summary + README + duplicate audit after stream completes
#   6) Keep manifest contract stable for downstream shard runner
#
# DESIGN:
# - This file does NOT run backtests
# - This file generates the full discovery job universe as JSONL
# - Output is written incrementally to avoid 9+ GB RAM spikes
# ==================================================================================================

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

VERSION = "v1.0.2"
SYSTEM_NAME = "intelligent_backtest_system"
VALIDATION_MODE = "DISCOVERY"
SYMBOL = "XAUUSD"

# --------------------------------------------------------------------------------------------------
# LOCKED RESEARCH TIMEFRAMES
# --------------------------------------------------------------------------------------------------
RESEARCH_TIMEFRAMES: List[str] = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M10",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
]

# --------------------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------------------
FEATURE_CACHE_DIR = Path(r"C:\Data\Bot\central_feature_cache")
OUTDIR = Path(r"C:\Data\Bot\central_backtest_results\research_jobs_full_discovery")
MANIFEST_PATH = OUTDIR / "research_job_manifest_full_discovery.jsonl"
SUMMARY_PATH = OUTDIR / "research_job_manifest_full_discovery_summary.json"
DUPLICATE_AUDIT_PATH = OUTDIR / "research_job_duplicate_audit.json"
README_PATH = OUTDIR / "research_job_manifest_full_discovery_README.txt"

# --------------------------------------------------------------------------------------------------
# SEARCH SPACE
# NOTE:
# - Keep regime_filters=4, cooldowns=4, side_policies=3, volatility_filters=3, trend_strength_filters=3
# - This preserves the same total job count shape as the run that yielded 518,400 jobs
# --------------------------------------------------------------------------------------------------
STRATEGY_FAMILIES: List[str] = [
    "pullback_deep",
    "pullback_shallow",
    "trend_continuation",
    "range_reversal",
    "breakout_expansion",
]

ENTRY_LOGIC_VARIANTS: List[str] = [
    "bos_choch_atr_adx_ema",
    "bos_choch_ema_reclaim",
    "pullback_to_ema_stack",
    "liquidity_sweep_reclaim",
    "breakout_retest",
]

MICRO_EXIT_VARIANTS: List[str] = [
    "micro_exit_v2_fast_invalidation",
    "micro_exit_v2_momentum_fade",
    "micro_exit_v2_structure_trail",
    "micro_exit_v2_range_fail",
]

REGIME_FILTERS: List[str] = [
    "trend_only",
    "trend_or_neutral",
    "range_only",
    "all_regimes",
]

COOLDOWN_BARS: List[int] = [0, 1, 3, 6]

SIDE_POLICIES: List[str] = [
    "long_only",
    "short_only",
    "both",
]

VOLATILITY_FILTERS: List[str] = [
    "any_vol",
    "mid_high_vol",
    "high_vol_only",
]

TREND_STRENGTH_FILTERS: List[str] = [
    "any_trend_strength",
    "mid_trend_plus",
    "strong_trend_only",
]

# --------------------------------------------------------------------------------------------------
# OPTIONAL CONTEXT MAP
# NOTE:
# - Reserved for later shard runner; included now as metadata only
# --------------------------------------------------------------------------------------------------
HTF_CONTEXT_MAP: Dict[str, str] = {
    "M1": "M5",
    "M2": "M10",
    "M3": "M15",
    "M4": "M15",
    "M5": "M15",
    "M6": "M30",
    "M10": "H1",
    "M15": "H1",
    "M30": "H4",
    "H1": "H4",
    "H4": "D1",
    "D1": "D1",
}

PROGRESS_EVERY = 10_000


@dataclass
class ResearchJobRecord:
    version: str
    system_name: str
    validation_mode: str
    job_id: str
    symbol: str
    timeframe: str
    feature_cache_path: str
    htf_context_timeframe: str
    htf_feature_cache_path: str
    strategy_family: str
    entry_logic: str
    micro_exit: str
    regime_filter: str
    cooldown_bars: int
    side_policy: str
    volatility_filter: str
    trend_strength_filter: str
    parameter_fingerprint: str
    one_axis_group_key: str
    rank_priority: int
    status: str
    generated_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def feature_cache_path_for_tf(timeframe: str) -> Path:
    return FEATURE_CACHE_DIR / f"{SYMBOL}_{timeframe}_base_features.parquet"


def stable_hash(payload: Dict[str, Any], length: int = 24) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def compute_rank_priority(timeframe: str, strategy_family: str) -> int:
    tf_priority = {
        "M1": 120,
        "M2": 115,
        "M3": 110,
        "M4": 105,
        "M5": 100,
        "M6": 95,
        "M10": 90,
        "M15": 85,
        "M30": 80,
        "H1": 75,
        "H4": 70,
        "D1": 65,
    }
    strategy_bonus = {
        "pullback_deep": 30,
        "pullback_shallow": 24,
        "trend_continuation": 20,
        "range_reversal": 14,
        "breakout_expansion": 10,
    }
    return tf_priority[timeframe] + strategy_bonus[strategy_family]


def validate_static_configuration() -> List[str]:
    missing: List[str] = []
    for tf in RESEARCH_TIMEFRAMES:
        p = feature_cache_path_for_tf(tf)
        if not p.exists():
            missing.append(tf)

    for tf in RESEARCH_TIMEFRAMES:
        if tf not in HTF_CONTEXT_MAP:
            raise RuntimeError(f"Missing HTF context mapping for timeframe={tf}")

    return missing


def build_readme_text() -> str:
    lines = [
        f"FILE: {MANIFEST_PATH.name}",
        f"VERSION: {VERSION}",
        f"SYSTEM_NAME: {SYSTEM_NAME}",
        f"VALIDATION_MODE: {VALIDATION_MODE}",
        "",
        "PURPOSE:",
        "- Streaming JSONL manifest for full discovery research campaign",
        "- Uses central feature cache files as dataset source",
        "- One line = one research job",
        "",
        "KEY FIELDS:",
        "- job_id",
        "- timeframe",
        "- feature_cache_path",
        "- strategy_family",
        "- entry_logic",
        "- micro_exit",
        "- regime_filter",
        "- cooldown_bars",
        "- side_policy",
        "- volatility_filter",
        "- trend_strength_filter",
        "",
        "EXPECTED JOB COUNT FORMULA:",
        "timeframes * strategy_families * entry_logics * micro_exits * regime_filters * cooldowns * side_policies * volatility_filters * trend_strength_filters",
        "",
        "DOWNSTREAM USAGE:",
        "- shard runner should read JSONL incrementally",
        "- shard runner should support start/end offsets",
        "- shard runner should resume safely using result files/state files",
    ]
    return "\n".join(lines) + "\n"


def iter_parameter_space() -> Iterable[Tuple[str, str, str, str, int, str, str, str]]:
    return itertools.product(
        STRATEGY_FAMILIES,
        ENTRY_LOGIC_VARIANTS,
        MICRO_EXIT_VARIANTS,
        REGIME_FILTERS,
        COOLDOWN_BARS,
        SIDE_POLICIES,
        VOLATILITY_FILTERS,
        TREND_STRENGTH_FILTERS,
    )


def build_job_record(
    timeframe: str,
    strategy_family: str,
    entry_logic: str,
    micro_exit: str,
    regime_filter: str,
    cooldown_bars: int,
    side_policy: str,
    volatility_filter: str,
    trend_strength_filter: str,
) -> ResearchJobRecord:
    feature_path = feature_cache_path_for_tf(timeframe)
    htf_tf = HTF_CONTEXT_MAP[timeframe]
    htf_feature_path = feature_cache_path_for_tf(htf_tf)

    fingerprint_payload = {
        "timeframe": timeframe,
        "strategy_family": strategy_family,
        "entry_logic": entry_logic,
        "micro_exit": micro_exit,
        "regime_filter": regime_filter,
        "cooldown_bars": cooldown_bars,
        "side_policy": side_policy,
        "volatility_filter": volatility_filter,
        "trend_strength_filter": trend_strength_filter,
        "htf_context_timeframe": htf_tf,
    }
    parameter_fingerprint = stable_hash(fingerprint_payload)
    job_id = f"job_{parameter_fingerprint}"

    one_axis_group_key = (
        f"{strategy_family}|{entry_logic}|{micro_exit}|{regime_filter}|"
        f"{cooldown_bars}|{side_policy}|{volatility_filter}|{trend_strength_filter}"
    )

    return ResearchJobRecord(
        version=VERSION,
        system_name=SYSTEM_NAME,
        validation_mode=VALIDATION_MODE,
        job_id=job_id,
        symbol=SYMBOL,
        timeframe=timeframe,
        feature_cache_path=str(feature_path),
        htf_context_timeframe=htf_tf,
        htf_feature_cache_path=str(htf_feature_path),
        strategy_family=strategy_family,
        entry_logic=entry_logic,
        micro_exit=micro_exit,
        regime_filter=regime_filter,
        cooldown_bars=int(cooldown_bars),
        side_policy=side_policy,
        volatility_filter=volatility_filter,
        trend_strength_filter=trend_strength_filter,
        parameter_fingerprint=parameter_fingerprint,
        one_axis_group_key=one_axis_group_key,
        rank_priority=compute_rank_priority(timeframe, strategy_family),
        status="PENDING",
        generated_at_utc=utc_now_iso(),
    )


def expected_total_jobs() -> int:
    return (
        len(RESEARCH_TIMEFRAMES)
        * len(STRATEGY_FAMILIES)
        * len(ENTRY_LOGIC_VARIANTS)
        * len(MICRO_EXIT_VARIANTS)
        * len(REGIME_FILTERS)
        * len(COOLDOWN_BARS)
        * len(SIDE_POLICIES)
        * len(VOLATILITY_FILTERS)
        * len(TREND_STRENGTH_FILTERS)
    )


def stream_manifest() -> Dict[str, Any]:
    duplicate_job_ids: Dict[str, int] = {}
    duplicate_fingerprints: Dict[str, int] = {}

    seen_job_ids = set()
    seen_fingerprints = set()

    timeframe_counts: Dict[str, int] = {tf: 0 for tf in RESEARCH_TIMEFRAMES}
    strategy_counts: Dict[str, int] = {x: 0 for x in STRATEGY_FAMILIES}
    entry_logic_counts: Dict[str, int] = {x: 0 for x in ENTRY_LOGIC_VARIANTS}
    micro_exit_counts: Dict[str, int] = {x: 0 for x in MICRO_EXIT_VARIANTS}
    regime_counts: Dict[str, int] = {x: 0 for x in REGIME_FILTERS}
    cooldown_counts: Dict[str, int] = {str(x): 0 for x in COOLDOWN_BARS}
    side_policy_counts: Dict[str, int] = {x: 0 for x in SIDE_POLICIES}
    volatility_counts: Dict[str, int] = {x: 0 for x in VOLATILITY_FILTERS}
    trend_strength_counts: Dict[str, int] = {x: 0 for x in TREND_STRENGTH_FILTERS}

    total_jobs = 0
    started = time.perf_counter()

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for tf_index, timeframe in enumerate(RESEARCH_TIMEFRAMES, start=1):
            tf_started = time.perf_counter()
            print(f"[TF-START] {tf_index}/{len(RESEARCH_TIMEFRAMES)} timeframe={timeframe}")

            for (
                strategy_family,
                entry_logic,
                micro_exit,
                regime_filter,
                cooldown_bars,
                side_policy,
                volatility_filter,
                trend_strength_filter,
            ) in iter_parameter_space():
                job = build_job_record(
                    timeframe=timeframe,
                    strategy_family=strategy_family,
                    entry_logic=entry_logic,
                    micro_exit=micro_exit,
                    regime_filter=regime_filter,
                    cooldown_bars=cooldown_bars,
                    side_policy=side_policy,
                    volatility_filter=volatility_filter,
                    trend_strength_filter=trend_strength_filter,
                )

                if job.job_id in seen_job_ids:
                    duplicate_job_ids[job.job_id] = duplicate_job_ids.get(job.job_id, 1) + 1
                else:
                    seen_job_ids.add(job.job_id)

                if job.parameter_fingerprint in seen_fingerprints:
                    duplicate_fingerprints[job.parameter_fingerprint] = (
                        duplicate_fingerprints.get(job.parameter_fingerprint, 1) + 1
                    )
                else:
                    seen_fingerprints.add(job.parameter_fingerprint)

                f.write(json.dumps(asdict(job), ensure_ascii=False))
                f.write("\n")

                total_jobs += 1
                timeframe_counts[timeframe] += 1
                strategy_counts[strategy_family] += 1
                entry_logic_counts[entry_logic] += 1
                micro_exit_counts[micro_exit] += 1
                regime_counts[regime_filter] += 1
                cooldown_counts[str(cooldown_bars)] += 1
                side_policy_counts[side_policy] += 1
                volatility_counts[volatility_filter] += 1
                trend_strength_counts[trend_strength_filter] += 1

                if total_jobs % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - started
                    rate = total_jobs / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[PROGRESS] jobs={total_jobs} "
                        f"elapsed_sec={elapsed:.2f} "
                        f"jobs_per_sec={rate:.2f} "
                        f"current_tf={timeframe}"
                    )

            tf_elapsed = time.perf_counter() - tf_started
            print(
                f"[TF-DONE] timeframe={timeframe} "
                f"jobs={timeframe_counts[timeframe]} "
                f"elapsed_sec={tf_elapsed:.2f}"
            )

    total_elapsed = time.perf_counter() - started

    return {
        "total_jobs": total_jobs,
        "total_elapsed_sec": round(total_elapsed, 4),
        "timeframe_counts": timeframe_counts,
        "strategy_counts": strategy_counts,
        "entry_logic_counts": entry_logic_counts,
        "micro_exit_counts": micro_exit_counts,
        "regime_counts": regime_counts,
        "cooldown_counts": cooldown_counts,
        "side_policy_counts": side_policy_counts,
        "volatility_counts": volatility_counts,
        "trend_strength_counts": trend_strength_counts,
        "duplicate_job_ids": duplicate_job_ids,
        "duplicate_fingerprints": duplicate_fingerprints,
        "duplicate_job_id_count": len(duplicate_job_ids),
        "duplicate_fingerprint_count": len(duplicate_fingerprints),
    }


def build_summary(stats: Dict[str, Any], missing_timeframes: List[str]) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "system_name": SYSTEM_NAME,
        "validation_mode": VALIDATION_MODE,
        "generated_at_utc": utc_now_iso(),
        "symbol": SYMBOL,
        "feature_cache_dir": str(FEATURE_CACHE_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "existing_dataset_timeframes": [tf for tf in RESEARCH_TIMEFRAMES if tf not in missing_timeframes],
        "missing_dataset_timeframes": missing_timeframes,
        "timeframes": RESEARCH_TIMEFRAMES,
        "strategy_families": STRATEGY_FAMILIES,
        "entry_logic_variants": ENTRY_LOGIC_VARIANTS,
        "micro_exit_variants": MICRO_EXIT_VARIANTS,
        "regime_filters": REGIME_FILTERS,
        "cooldown_bars": COOLDOWN_BARS,
        "side_policies": SIDE_POLICIES,
        "volatility_filters": VOLATILITY_FILTERS,
        "trend_strength_filters": TREND_STRENGTH_FILTERS,
        "total_jobs": stats["total_jobs"],
        "expected_total_jobs": expected_total_jobs(),
        "total_elapsed_sec": stats["total_elapsed_sec"],
        "timeframe_counts": stats["timeframe_counts"],
        "strategy_counts": stats["strategy_counts"],
        "entry_logic_counts": stats["entry_logic_counts"],
        "micro_exit_counts": stats["micro_exit_counts"],
        "regime_counts": stats["regime_counts"],
        "cooldown_counts": stats["cooldown_counts"],
        "side_policy_counts": stats["side_policy_counts"],
        "volatility_counts": stats["volatility_counts"],
        "trend_strength_counts": stats["trend_strength_counts"],
        "feature_cache_complete": len(missing_timeframes) == 0,
        "streaming_mode": True,
        "progress_every_jobs": PROGRESS_EVERY,
    }


def build_duplicate_audit(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "duplicate_job_id_count": stats["duplicate_job_id_count"],
        "duplicate_fingerprint_count": stats["duplicate_fingerprint_count"],
        "duplicate_job_ids": stats["duplicate_job_ids"],
        "duplicate_fingerprints": stats["duplicate_fingerprints"],
    }


def main() -> None:
    ensure_dir(OUTDIR)

    reset_file(MANIFEST_PATH)
    reset_file(SUMMARY_PATH)
    reset_file(DUPLICATE_AUDIT_PATH)
    reset_file(README_PATH)

    missing_timeframes = validate_static_configuration()
    if missing_timeframes:
        raise RuntimeError(
            "Missing feature cache files. Build base feature cache first for: "
            + ", ".join(missing_timeframes)
        )

    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] system_name={SYSTEM_NAME}")
    print(f"[START] validation_mode={VALIDATION_MODE}")
    print(f"[START] symbol={SYMBOL}")
    print(f"[START] feature_cache_dir={FEATURE_CACHE_DIR}")
    print(f"[START] outdir={OUTDIR}")
    print(f"[START] expected_total_jobs={expected_total_jobs()}")
    print(f"[START] progress_every={PROGRESS_EVERY}")
    print("=" * 120)

    stats = stream_manifest()

    if stats["duplicate_job_id_count"] != 0:
        raise RuntimeError(f"Duplicate job_id detected: {stats['duplicate_job_id_count']}")

    if stats["duplicate_fingerprint_count"] != 0:
        raise RuntimeError(
            f"Duplicate parameter_fingerprint detected: {stats['duplicate_fingerprint_count']}"
        )

    summary = build_summary(stats, missing_timeframes)
    duplicate_audit = build_duplicate_audit(stats)

    write_json(SUMMARY_PATH, summary)
    write_json(DUPLICATE_AUDIT_PATH, duplicate_audit)
    write_text(README_PATH, build_readme_text())

    print("=" * 120)
    print(f"[DONE] manifest={MANIFEST_PATH}")
    print(f"[DONE] summary={SUMMARY_PATH}")
    print(f"[DONE] duplicate_audit={DUPLICATE_AUDIT_PATH}")
    print(f"[DONE] readme={README_PATH}")
    print(f"[DONE] total_jobs={stats['total_jobs']}")
    print(f"[DONE] total_elapsed_sec={stats['total_elapsed_sec']:.2f}")
    print("=" * 120)


if __name__ == "__main__":
    main()