#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================================================
build_research_coverage_master_manifest_v1_0_0.py
Version: v1.0.0

Purpose
-------
Build the master research coverage manifest for the Local_LLM / gold_research project.

This file fixes the root problem:
- We have large bar datasets across M1..D1
- But research coverage is still incomplete across strategy, logic, entry, exit, management, regime
- We need a manifest-first coverage system to prove what is covered and what is still missing

This script creates a bounded exhaustive research contract that:
1) defines the exact research dimensions to cover
2) emits one row per research job
3) groups jobs into execution phases
4) includes VectorBT / speed / batching / GPU hints for the downstream runner
5) creates audit-friendly outputs for project control

Design Principles
-----------------
- Production-first
- Full-history evaluation expected in downstream runner
- VectorBT is the main research engine
- GPU should be used by downstream runners where beneficial
- This manifest builder itself is lightweight and CPU-fast
- Accuracy > vanity speed
- Human-usable intelligent database is the end goal

Locked Project Coverage Dimensions
----------------------------------
Timeframes:
- M1, M2, M3, M4, M5, M6, M10, M15, M30, H1, H4, D1

Core logic families:
- market_structure_continuation
- bos_continuation
- choch_reversal
- swing_pullback
- breakout_retest
- ema_trend_continuation
- adx_ema_trend_continuation
- failed_breakout_reversal
- session_conditioned_trend
- volatility_expansion_contraction

Logic strictness / structure variants:
- strict
- medium
- relaxed

Swing variants:
- short
- medium
- long

Pullback zone variants:
- narrow
- medium
- wide

Entry variants:
- confirm_entry
- delayed_confirm_entry
- pullback_entry
- stop_entry
- retrace_bucket_entry
- minor_break_entry

Micro exits:
- reverse_signal_exit
- baseline_pending_exit
- adx_fade_exit
- atr_guard_exit
- price_cross_fast_exit
- price_cross_slow_exit
- structure_fail_exit
- break_even_ladder_exit
- atr_trailing_exit
- time_decay_exit
- partial_tp_runner_exit
- volatility_crush_exit

Management variants:
- no_extra_management
- break_even_promotion
- partial_tp_management
- runner_management
- cooldown_after_loss_cluster
- reentry_control
- session_close_protection

Research phases:
1. micro_exit_expansion
2. entry_timing_architecture
3. trade_management
4. regime_switching
5. robustness_stability

Outputs
-------
- research_coverage_master_manifest.csv
- research_coverage_master_manifest.jsonl
- research_coverage_summary.json
- research_coverage_checklist.txt
- research_coverage_phase_counts.csv
- research_coverage_batch_plan.csv

Run
---
python C:\\Data\\Bot\\Local_LLM\\gold_research\\jobs\\build_research_coverage_master_manifest_v1_0_0.py ^
  --outdir C:\\Data\\Bot\\central_backtest_results\\research_coverage_master_v1_0_0

Notes on Speed / GPU
--------------------
- This script only builds the manifest, so GPU has little value here.
- The downstream runner should use:
    * VectorBT
    * batch partitioning
    * resume state
    * multiprocessing / shard execution
    * optional GPU-accelerated indicator computation where applicable
- This manifest includes engine/gpu/batch hints for that runner.
====================================================================================================
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


VERSION = "v1.0.0"

TIMEFRAMES: List[str] = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"
]

TIMEFRAME_SORT_KEY: Dict[str, int] = {
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

STRATEGY_FAMILIES: List[str] = [
    "market_structure_continuation",
    "bos_continuation",
    "choch_reversal",
    "swing_pullback",
    "breakout_retest",
    "ema_trend_continuation",
    "adx_ema_trend_continuation",
    "failed_breakout_reversal",
    "session_conditioned_trend",
    "volatility_expansion_contraction",
]

LOGIC_STRICTNESS: List[str] = ["strict", "medium", "relaxed"]
SWING_VARIANTS: List[str] = ["short", "medium", "long"]
PULLBACK_ZONE_VARIANTS: List[str] = ["narrow", "medium", "wide"]

ENTRY_VARIANTS: List[str] = [
    "confirm_entry",
    "delayed_confirm_entry",
    "pullback_entry",
    "stop_entry",
    "retrace_bucket_entry",
    "minor_break_entry",
]

MICRO_EXIT_VARIANTS: List[str] = [
    "reverse_signal_exit",
    "baseline_pending_exit",
    "adx_fade_exit",
    "atr_guard_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "structure_fail_exit",
    "break_even_ladder_exit",
    "atr_trailing_exit",
    "time_decay_exit",
    "partial_tp_runner_exit",
    "volatility_crush_exit",
]

MANAGEMENT_VARIANTS: List[str] = [
    "no_extra_management",
    "break_even_promotion",
    "partial_tp_management",
    "runner_management",
    "cooldown_after_loss_cluster",
    "reentry_control",
    "session_close_protection",
]

REGIME_VARIANTS: List[str] = [
    "trend_high_vol",
    "trend_mid_vol",
    "trend_low_vol",
    "weak_trend_high_vol",
    "weak_trend_mid_vol",
    "weak_trend_low_vol",
    "range_high_vol",
    "range_mid_vol",
    "range_low_vol",
    "chop_high_vol",
    "chop_mid_vol",
    "chop_low_vol",
]

ROBUSTNESS_VARIANTS: List[str] = [
    "full_history_baseline",
    "rolling_window_walkforward",
    "parameter_neighborhood_stability",
    "regime_drift_sensitivity",
    "time_segment_persistence",
]

EMA_FILTER_VARIANTS: List[Tuple[int, int]] = [
    (5, 20),
    (8, 21),
    (10, 30),
    (12, 36),
    (20, 50),
]

BATCH_GROUP_TIMEFRAME_MAP: Dict[str, str] = {
    "M1": "ultra_heavy",
    "M2": "very_heavy",
    "M3": "very_heavy",
    "M4": "heavy",
    "M5": "heavy",
    "M6": "heavy",
    "M10": "medium",
    "M15": "medium",
    "M30": "light",
    "H1": "light",
    "H4": "very_light",
    "D1": "very_light",
}


@dataclass(frozen=True)
class ManifestRow:
    manifest_id: str
    version: str
    phase: str
    phase_order: int
    timeframe: str
    timeframe_sort_key: int
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build master research coverage manifest for bounded exhaustive coverage"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for the master manifest and summaries",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Research symbol",
    )
    return parser.parse_args()


def make_row(
    *,
    phase: str,
    phase_order: int,
    timeframe: str,
    strategy_family: str,
    logic_strictness: str,
    swing_variant: str,
    pullback_zone_variant: str,
    entry_variant: str,
    micro_exit_variant: str,
    management_variant: str,
    regime_variant: str,
    robustness_variant: str,
    ema_fast: int,
    ema_slow: int,
    symbol: str,
    priority_tier: str,
    coverage_axis: str,
    expected_output_stage: str,
    rationale: str,
) -> ManifestRow:
    timeframe_sort_key = TIMEFRAME_SORT_KEY[timeframe]
    batch_group = BATCH_GROUP_TIMEFRAME_MAP[timeframe]

    return ManifestRow(
        manifest_id="__".join(
            [
                phase,
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
                f"ema{ema_fast}_{ema_slow}",
            ]
        ),
        version=VERSION,
        phase=phase,
        phase_order=phase_order,
        timeframe=timeframe,
        timeframe_sort_key=timeframe_sort_key,
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
        ema_filter_rule="fast_1_50_slow_20_100_fast_lt_slow_filter_only",
        symbol=symbol,
        execution_engine="vectorbt",
        engine_hint="use_vectorbt_with_resume_and_batching",
        parallel_hint="parallel_by_manifest_shard_and_timeframe",
        gpu_hint="use_gpu_if_supported_for_indicator_and_array_ops_in_runner",
        full_history_required=True,
        resume_required=True,
        batch_group=batch_group,
        priority_tier=priority_tier,
        coverage_axis=coverage_axis,
        expected_output_stage=expected_output_stage,
        status="PLANNED",
        rationale=rationale,
    )


def build_phase_micro_exit(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    supported_families = [
        "ema_trend_continuation",
        "adx_ema_trend_continuation",
        "market_structure_continuation",
        "bos_continuation",
        "swing_pullback",
    ]

    for tf in TIMEFRAMES:
        for family in supported_families:
            for strictness in LOGIC_STRICTNESS:
                for swing_variant in SWING_VARIANTS:
                    for zone in PULLBACK_ZONE_VARIANTS:
                        for ema_fast, ema_slow in EMA_FILTER_VARIANTS:
                            for micro_exit in MICRO_EXIT_VARIANTS:
                                rows.append(
                                    make_row(
                                        phase="micro_exit_expansion",
                                        phase_order=1,
                                        timeframe=tf,
                                        strategy_family=family,
                                        logic_strictness=strictness,
                                        swing_variant=swing_variant,
                                        pullback_zone_variant=zone,
                                        entry_variant="confirm_entry",
                                        micro_exit_variant=micro_exit,
                                        management_variant="no_extra_management",
                                        regime_variant="trend_mid_vol",
                                        robustness_variant="full_history_baseline",
                                        ema_fast=ema_fast,
                                        ema_slow=ema_slow,
                                        symbol=symbol,
                                        priority_tier="P0",
                                        coverage_axis="timeframe_x_family_x_logic_x_micro_exit",
                                        expected_output_stage="micro_exit_research",
                                        rationale="Expand exit coverage first because current micro-exit evidence is biased and incomplete",
                                    )
                                )
    return rows


def build_phase_entry_timing(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    families = [
        "market_structure_continuation",
        "bos_continuation",
        "choch_reversal",
        "swing_pullback",
        "breakout_retest",
        "ema_trend_continuation",
        "adx_ema_trend_continuation",
    ]
    locked_exits = [
        "reverse_signal_exit",
        "baseline_pending_exit",
        "adx_fade_exit",
        "structure_fail_exit",
    ]

    for tf in TIMEFRAMES:
        for family in families:
            for strictness in LOGIC_STRICTNESS:
                for swing_variant in SWING_VARIANTS:
                    for zone in PULLBACK_ZONE_VARIANTS:
                        for entry_variant in ENTRY_VARIANTS:
                            for ema_fast, ema_slow in EMA_FILTER_VARIANTS:
                                for micro_exit in locked_exits:
                                    rows.append(
                                        make_row(
                                            phase="entry_timing_architecture",
                                            phase_order=2,
                                            timeframe=tf,
                                            strategy_family=family,
                                            logic_strictness=strictness,
                                            swing_variant=swing_variant,
                                            pullback_zone_variant=zone,
                                            entry_variant=entry_variant,
                                            micro_exit_variant=micro_exit,
                                            management_variant="no_extra_management",
                                            regime_variant="trend_mid_vol",
                                            robustness_variant="full_history_baseline",
                                            ema_fast=ema_fast,
                                            ema_slow=ema_slow,
                                            symbol=symbol,
                                            priority_tier="P0",
                                            coverage_axis="timeframe_x_family_x_logic_x_entry",
                                            expected_output_stage="entry_timing_research",
                                            rationale="Entry timing is the next major intelligence gap after micro-exit coverage",
                                        )
                                    )
    return rows


def build_phase_trade_management(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    families = [
        "market_structure_continuation",
        "bos_continuation",
        "swing_pullback",
        "ema_trend_continuation",
        "adx_ema_trend_continuation",
    ]
    locked_entries = [
        "confirm_entry",
        "pullback_entry",
        "delayed_confirm_entry",
    ]
    locked_exits = [
        "reverse_signal_exit",
        "baseline_pending_exit",
        "adx_fade_exit",
        "break_even_ladder_exit",
        "partial_tp_runner_exit",
    ]

    for tf in TIMEFRAMES:
        for family in families:
            for strictness in LOGIC_STRICTNESS:
                for swing_variant in SWING_VARIANTS:
                    for zone in PULLBACK_ZONE_VARIANTS:
                        for entry_variant in locked_entries:
                            for micro_exit in locked_exits:
                                for management_variant in MANAGEMENT_VARIANTS:
                                    for ema_fast, ema_slow in EMA_FILTER_VARIANTS:
                                        rows.append(
                                            make_row(
                                                phase="trade_management",
                                                phase_order=3,
                                                timeframe=tf,
                                                strategy_family=family,
                                                logic_strictness=strictness,
                                                swing_variant=swing_variant,
                                                pullback_zone_variant=zone,
                                                entry_variant=entry_variant,
                                                micro_exit_variant=micro_exit,
                                                management_variant=management_variant,
                                                regime_variant="trend_mid_vol",
                                                robustness_variant="full_history_baseline",
                                                ema_fast=ema_fast,
                                                ema_slow=ema_slow,
                                                symbol=symbol,
                                                priority_tier="P1",
                                                coverage_axis="timeframe_x_family_x_entry_x_management",
                                                expected_output_stage="management_research",
                                                rationale="Trade management beyond micro-exit is required for survivability and human usability",
                                            )
                                        )
    return rows


def build_phase_regime_switching(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    families = [
        "market_structure_continuation",
        "bos_continuation",
        "choch_reversal",
        "swing_pullback",
        "breakout_retest",
        "ema_trend_continuation",
        "adx_ema_trend_continuation",
        "failed_breakout_reversal",
        "session_conditioned_trend",
        "volatility_expansion_contraction",
    ]
    locked_entries = ["confirm_entry", "pullback_entry"]
    locked_exits = ["reverse_signal_exit", "baseline_pending_exit", "adx_fade_exit", "structure_fail_exit"]
    locked_managements = ["no_extra_management", "break_even_promotion", "partial_tp_management"]

    for tf in TIMEFRAMES:
        for family in families:
            for strictness in LOGIC_STRICTNESS:
                for swing_variant in SWING_VARIANTS:
                    for zone in PULLBACK_ZONE_VARIANTS:
                        for entry_variant in locked_entries:
                            for micro_exit in locked_exits:
                                for management_variant in locked_managements:
                                    for regime_variant in REGIME_VARIANTS:
                                        for ema_fast, ema_slow in EMA_FILTER_VARIANTS:
                                            rows.append(
                                                make_row(
                                                    phase="regime_switching",
                                                    phase_order=4,
                                                    timeframe=tf,
                                                    strategy_family=family,
                                                    logic_strictness=strictness,
                                                    swing_variant=swing_variant,
                                                    pullback_zone_variant=zone,
                                                    entry_variant=entry_variant,
                                                    micro_exit_variant=micro_exit,
                                                    management_variant=management_variant,
                                                    regime_variant=regime_variant,
                                                    robustness_variant="full_history_baseline",
                                                    ema_fast=ema_fast,
                                                    ema_slow=ema_slow,
                                                    symbol=symbol,
                                                    priority_tier="P1",
                                                    coverage_axis="timeframe_x_family_x_regime",
                                                    expected_output_stage="regime_switch_research",
                                                    rationale="Intelligent database must map context to strategy, not just global winners",
                                                )
                                            )
    return rows


def build_phase_robustness(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    families = [
        "market_structure_continuation",
        "bos_continuation",
        "swing_pullback",
        "ema_trend_continuation",
        "adx_ema_trend_continuation",
    ]
    locked_entries = ["confirm_entry", "pullback_entry"]
    locked_exits = ["reverse_signal_exit", "baseline_pending_exit", "adx_fade_exit", "structure_fail_exit"]
    locked_managements = ["no_extra_management", "break_even_promotion"]
    locked_regimes = ["trend_mid_vol", "weak_trend_mid_vol", "range_mid_vol", "trend_high_vol"]

    for tf in TIMEFRAMES:
        for family in families:
            for strictness in LOGIC_STRICTNESS:
                for swing_variant in SWING_VARIANTS:
                    for zone in PULLBACK_ZONE_VARIANTS:
                        for entry_variant in locked_entries:
                            for micro_exit in locked_exits:
                                for management_variant in locked_managements:
                                    for regime_variant in locked_regimes:
                                        for robustness_variant in ROBUSTNESS_VARIANTS:
                                            for ema_fast, ema_slow in EMA_FILTER_VARIANTS:
                                                rows.append(
                                                    make_row(
                                                        phase="robustness_stability",
                                                        phase_order=5,
                                                        timeframe=tf,
                                                        strategy_family=family,
                                                        logic_strictness=strictness,
                                                        swing_variant=swing_variant,
                                                        pullback_zone_variant=zone,
                                                        entry_variant=entry_variant,
                                                        micro_exit_variant=micro_exit,
                                                        management_variant=management_variant,
                                                        regime_variant=regime_variant,
                                                        robustness_variant=robustness_variant,
                                                        ema_fast=ema_fast,
                                                        ema_slow=ema_slow,
                                                        symbol=symbol,
                                                        priority_tier="P2",
                                                        coverage_axis="timeframe_x_family_x_robustness",
                                                        expected_output_stage="robustness_research",
                                                        rationale="Winners must survive regime drift and stability checks before intelligent promotion",
                                                    )
                                                )
    return rows


def build_master_rows(symbol: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    rows.extend(build_phase_micro_exit(symbol))
    rows.extend(build_phase_entry_timing(symbol))
    rows.extend(build_phase_trade_management(symbol))
    rows.extend(build_phase_regime_switching(symbol))
    rows.extend(build_phase_robustness(symbol))
    return rows


def rows_to_dataframe(rows: Sequence[ManifestRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_phase_counts(df: pd.DataFrame) -> pd.DataFrame:
    phase_counts = (
        df.groupby(["phase_order", "phase", "priority_tier", "batch_group"], dropna=False)
        .size()
        .reset_index(name="job_count")
        .sort_values(["phase_order", "priority_tier", "batch_group", "phase"])
        .reset_index(drop=True)
    )
    return phase_counts


def build_batch_plan(df: pd.DataFrame) -> pd.DataFrame:
    batch_plan = (
        df.groupby(["phase_order", "phase", "timeframe", "batch_group"], dropna=False)
        .size()
        .reset_index(name="job_count")
        .sort_values(["phase_order", "timeframe_sort_order"], na_position="last")
        if False else None
    )
    grouped = (
        df.groupby(["phase_order", "phase", "timeframe", "batch_group"], dropna=False)
        .size()
        .reset_index(name="job_count")
    )
    grouped["timeframe_sort_key"] = grouped["timeframe"].map(TIMEFRAME_SORT_KEY)
    grouped = grouped.sort_values(
        ["phase_order", "timeframe_sort_key", "batch_group"]
    ).reset_index(drop=True)
    return grouped


def build_summary(df: pd.DataFrame, outdir: Path) -> Dict[str, object]:
    phase_counts = (
        df.groupby("phase", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    timeframe_counts = (
        df.groupby("timeframe", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    strategy_family_counts = (
        df.groupby("strategy_family", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    micro_exit_counts = (
        df.groupby("micro_exit_variant", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    management_counts = (
        df.groupby("management_variant", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    entry_counts = (
        df.groupby("entry_variant", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    regime_counts = (
        df.groupby("regime_variant", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    robustness_counts = (
        df.groupby("robustness_variant", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "version": VERSION,
        "outdir": str(outdir),
        "total_jobs": int(len(df)),
        "phases": phase_counts,
        "timeframes": timeframe_counts,
        "strategy_families": strategy_family_counts,
        "micro_exit_variants": micro_exit_counts,
        "management_variants": management_counts,
        "entry_variants": entry_counts,
        "regime_variants": regime_counts,
        "robustness_variants": robustness_counts,
        "execution_contract": {
            "engine": "vectorbt",
            "full_history_required": True,
            "resume_required": True,
            "gpu_policy": "downstream_runner_should_use_gpu_if_supported_for_indicator_and_array_ops",
            "parallel_policy": "parallel_by_shard_and_timeframe",
            "goal": "bounded_exhaustive_coverage_for_intelligent_database",
        },
    }


def write_checklist(df: pd.DataFrame, path: Path) -> None:
    lines: List[str] = []
    lines.append("RESEARCH COVERAGE CHECKLIST")
    lines.append("=" * 100)
    lines.append(f"version={VERSION}")
    lines.append("")
    lines.append("Locked Direction:")
    lines.append("- Use manifest-first coverage system")
    lines.append("- Use VectorBT as research engine")
    lines.append("- Full-history evaluation for every job")
    lines.append("- Resume / state / progress / ETA required in downstream runner")
    lines.append("- GPU should be used in downstream runner where supported and beneficial")
    lines.append("")
    lines.append("Coverage Dimensions:")
    lines.append(f"- Timeframes: {', '.join(TIMEFRAMES)}")
    lines.append(f"- Strategy Families: {', '.join(STRATEGY_FAMILIES)}")
    lines.append(f"- Logic Strictness: {', '.join(LOGIC_STRICTNESS)}")
    lines.append(f"- Swing Variants: {', '.join(SWING_VARIANTS)}")
    lines.append(f"- Pullback Zones: {', '.join(PULLBACK_ZONE_VARIANTS)}")
    lines.append(f"- Entry Variants: {', '.join(ENTRY_VARIANTS)}")
    lines.append(f"- Micro Exits: {', '.join(MICRO_EXIT_VARIANTS)}")
    lines.append(f"- Management Variants: {', '.join(MANAGEMENT_VARIANTS)}")
    lines.append(f"- Regime Variants: {', '.join(REGIME_VARIANTS)}")
    lines.append(f"- Robustness Variants: {', '.join(ROBUSTNESS_VARIANTS)}")
    lines.append("")
    lines.append("Phases:")
    for phase_order, phase_name in sorted(df[["phase_order", "phase"]].drop_duplicates().values.tolist()):
        count = int((df["phase"] == phase_name).sum())
        lines.append(f"- Phase {phase_order}: {phase_name} | jobs={count}")
    lines.append("")
    lines.append("Checklist Rules:")
    lines.append("- A dimension is not considered covered unless it exists in the master manifest")
    lines.append("- A row is not considered covered unless downstream runner marks it done")
    lines.append("- Winners alone do not prove coverage")
    lines.append("- Negative evidence must also be preserved in intelligent database outputs")
    lines.append("- Micro exit remains mandatory base layer")
    lines.append("- Practical profitability and survivability remain primary objective")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_master_rows(symbol=args.symbol)
    df = rows_to_dataframe(rows)

    df = df.sort_values(
        ["phase_order", "timeframe_sort_key", "strategy_family", "logic_strictness", "entry_variant", "micro_exit_variant", "management_variant", "regime_variant", "robustness_variant", "ema_fast", "ema_slow"]
    ).reset_index(drop=True)

    df.insert(0, "manifest_rank", range(1, len(df) + 1))

    manifest_csv = outdir / "research_coverage_master_manifest.csv"
    manifest_jsonl = outdir / "research_coverage_master_manifest.jsonl"
    summary_json = outdir / "research_coverage_summary.json"
    checklist_txt = outdir / "research_coverage_checklist.txt"
    phase_counts_csv = outdir / "research_coverage_phase_counts.csv"
    batch_plan_csv = outdir / "research_coverage_batch_plan.csv"

    df.to_csv(manifest_csv, index=False, encoding="utf-8")
    write_jsonl(df, manifest_jsonl)

    phase_counts = build_phase_counts(df)
    phase_counts.to_csv(phase_counts_csv, index=False, encoding="utf-8")

    batch_plan = build_batch_plan(df)
    batch_plan.to_csv(batch_plan_csv, index=False, encoding="utf-8")

    summary = build_summary(df, outdir)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_checklist(df, checklist_txt)

    print("=" * 108)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] total_jobs={len(df)}")
    print(f"[DONE] manifest_csv={manifest_csv}")
    print(f"[DONE] manifest_jsonl={manifest_jsonl}")
    print(f"[DONE] summary_json={summary_json}")
    print(f"[DONE] checklist_txt={checklist_txt}")
    print(f"[DONE] phase_counts_csv={phase_counts_csv}")
    print(f"[DONE] batch_plan_csv={batch_plan_csv}")
    print("=" * 108)


if __name__ == "__main__":
    main()