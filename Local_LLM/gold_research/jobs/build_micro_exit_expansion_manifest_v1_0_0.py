#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================================================
build_micro_exit_expansion_manifest_v1_0_0.py
Version: v1.0.0

Purpose
-------
Build a production-grade micro-exit expansion manifest from the latest winner registry.
This script audits current micro-exit imbalance, ranks under-covered TF x strategy-family x exit
combinations, and emits:
    1) supported expansion manifest   -> for exits already present in research outputs
    2) design expansion manifest      -> for new exit families that should be implemented next
    3) coverage summaries             -> CSV / JSON / TXT outputs for decision-grade review

Locked Project Direction
------------------------
- Production-first, not demo
- Optimize for practical profitability + survivability + real human usability
- Micro exit is mandatory base layer
- Use evidence from registry, not guesswork
- Build intelligence for decision-making, not just more backtests

Evidence Snapshot Used To Design This File
------------------------------------------
Registry evidence from latest confirmed run:
- candidate_rows = 44112
- active_winners = 240
- top_per_timeframe = 20

Observed micro-exit imbalance:
- reverse_signal_exit   = 42768 candidates / 165 active winners
- baseline_pending_exit =   864 candidates /  50 active winners
- adx_fade_exit         =   120 candidates /  25 active winners
- atr_guard_exit        =   120 candidates /   0 active winners
- price_cross_fast_exit =   120 candidates /   0 active winners
- price_cross_slow_exit =   120 candidates /   0 active winners

Observed active winner strategy-family concentration:
- ema_cross_long_only       dominates
- adx_ema_cross_long_only   secondary
=> This confirms exit research is still under-covered and biased.

Changelog
---------
v1.0.0
- Initial production version
- Reads registry leaderboard + active winners
- Builds supported expansion manifest for currently supported exits
- Builds design manifest for new exit families requiring implementation
- Exports audit summary files and human-readable recommendation text

Run Example
-----------
python C:\\Data\\Bot\\Local_LLM\\gold_research\\jobs\\build_micro_exit_expansion_manifest_v1_0_0.py ^
  --registry-root C:\\Data\\Bot\\central_backtest_results\\winner_registry ^
  --outdir C:\\Data\\Bot\\central_backtest_results\\micro_exit_expansion_v1_0_0

Outputs
-------
- micro_exit_candidate_variant_stats.csv
- micro_exit_active_variant_stats.csv
- micro_exit_tf_family_exit_stats.csv
- micro_exit_supported_expansion_manifest.csv
- micro_exit_supported_expansion_manifest.jsonl
- micro_exit_design_manifest.csv
- micro_exit_design_manifest.jsonl
- micro_exit_expansion_summary.json
- micro_exit_expansion_recommendation.txt
====================================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


VERSION = "v1.0.0"

# Existing exit families already visible in registry outputs
SUPPORTED_EXITS: List[str] = [
    "reverse_signal_exit",
    "baseline_pending_exit",
    "adx_fade_exit",
    "atr_guard_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
]

# New exit families that should exist in a truly intelligent database roadmap.
# These are emitted into a "design manifest" so the next implementation step is explicit.
PROPOSED_NEW_EXITS: List[str] = [
    "structure_fail_exit",
    "break_even_ladder_exit",
    "atr_trailing_exit",
    "time_decay_exit",
    "partial_tp_runner_exit",
    "volatility_crush_exit",
]

NUMERIC_COLUMNS_CANDIDATE = [
    "score",
    "profit_factor",
    "expectancy",
    "trade_count",
    "pnl_sum",
    "max_drawdown",
    "avg_win",
    "avg_loss",
    "ema_fast",
    "ema_slow",
    "timeframe_rank",
]

NUMERIC_COLUMNS_ACTIVE = [
    "score",
    "profit_factor",
    "expectancy",
    "trade_count",
    "pnl_sum",
    "max_drawdown",
    "avg_win",
    "avg_loss",
    "ema_fast",
    "ema_slow",
    "timeframe_rank",
]

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


@dataclass(frozen=True)
class Paths:
    registry_root: Path
    outdir: Path
    timeframe_leaderboard_csv: Path
    active_winner_registry_csv: Path
    database_ready_strategy_packages_csv: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced micro-exit expansion manifest from latest winner registry"
    )
    parser.add_argument(
        "--registry-root",
        required=True,
        help="Winner registry root directory",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for expansion manifests and summaries",
    )
    parser.add_argument(
        "--min-target-per-exit",
        type=int,
        default=8,
        help="Minimum active-winner target per TF x strategy-family x exit combo for supported exits",
    )
    parser.add_argument(
        "--max-target-per-exit",
        type=int,
        default=20,
        help="Upper cap for active-winner target per TF x strategy-family x exit combo",
    )
    parser.add_argument(
        "--min-candidate-rows-floor",
        type=int,
        default=40,
        help="Minimum candidate-row target per TF x strategy-family x exit combo",
    )
    parser.add_argument(
        "--top-combos",
        type=int,
        default=250,
        help="Maximum number of supported expansion rows to emit",
    )
    parser.add_argument(
        "--top-design-rows",
        type=int,
        default=250,
        help="Maximum number of design-manifest rows to emit",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_paths(registry_root: Path, outdir: Path) -> Paths:
    if not registry_root.exists():
        fail(f"registry_root not found: {registry_root}")

    leaderboard = registry_root / "timeframe_leaderboard.csv"
    active = registry_root / "active_winner_registry.csv"
    packages = registry_root / "database_ready_strategy_packages.csv"

    if not leaderboard.exists():
        fail(f"required file not found: {leaderboard}")
    if not active.exists():
        fail(f"required file not found: {active}")

    outdir.mkdir(parents=True, exist_ok=True)

    return Paths(
        registry_root=registry_root,
        outdir=outdir,
        timeframe_leaderboard_csv=leaderboard,
        active_winner_registry_csv=active,
        database_ready_strategy_packages_csv=packages if packages.exists() else None,
    )


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail(f"failed to read CSV: {path} | reason={exc}")
        raise
    return df


def normalize_df(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()

    for col in ["timeframe", "strategy_family", "logic_variant", "micro_exit_variant", "winner_status", "is_active"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).str.strip()

    for col in numeric_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "timeframe" in out.columns:
        out["timeframe_sort_key"] = out["timeframe"].map(TIMEFRAME_ORDER).fillna(999999)
    else:
        out["timeframe_sort_key"] = 999999

    return out


def round_safe(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), ndigits)


def avg_or_none(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    value = pd.to_numeric(series, errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.mean())


def first_non_empty(series: pd.Series) -> str:
    for value in series.astype(str):
        v = value.strip()
        if v:
            return v
    return ""


def build_variant_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    grouped = (
        df.groupby("micro_exit_variant", dropna=False)
        .agg(
            rows=("micro_exit_variant", "size"),
            avg_score=("score", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_expectancy=("expectancy", "mean"),
            avg_trade_count=("trade_count", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "dataset", label)
    grouped["avg_score"] = grouped["avg_score"].round(4)
    grouped["avg_profit_factor"] = grouped["avg_profit_factor"].round(4)
    grouped["avg_expectancy"] = grouped["avg_expectancy"].round(4)
    grouped["avg_trade_count"] = grouped["avg_trade_count"].round(2)
    grouped = grouped.sort_values(["rows", "micro_exit_variant"], ascending=[False, True]).reset_index(drop=True)
    return grouped


def build_tf_family_exit_stats(
    candidate_df: pd.DataFrame,
    active_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_stats = (
        candidate_df.groupby(["timeframe", "strategy_family", "micro_exit_variant"], dropna=False)
        .agg(
            candidate_rows=("micro_exit_variant", "size"),
            candidate_avg_score=("score", "mean"),
            candidate_avg_profit_factor=("profit_factor", "mean"),
            candidate_avg_expectancy=("expectancy", "mean"),
            candidate_avg_trade_count=("trade_count", "mean"),
            logic_variant_sample=("logic_variant", first_non_empty),
        )
        .reset_index()
    )

    active_stats = (
        active_df.groupby(["timeframe", "strategy_family", "micro_exit_variant"], dropna=False)
        .agg(
            active_winners=("micro_exit_variant", "size"),
            active_avg_score=("score", "mean"),
            active_avg_profit_factor=("profit_factor", "mean"),
            active_avg_expectancy=("expectancy", "mean"),
            active_avg_trade_count=("trade_count", "mean"),
        )
        .reset_index()
    )

    merged = candidate_stats.merge(
        active_stats,
        how="outer",
        on=["timeframe", "strategy_family", "micro_exit_variant"],
    )

    for col in [
        "candidate_rows",
        "candidate_avg_score",
        "candidate_avg_profit_factor",
        "candidate_avg_expectancy",
        "candidate_avg_trade_count",
        "active_winners",
        "active_avg_score",
        "active_avg_profit_factor",
        "active_avg_expectancy",
        "active_avg_trade_count",
    ]:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged["candidate_rows"] = pd.to_numeric(merged["candidate_rows"], errors="coerce").fillna(0).astype(int)
    merged["active_winners"] = pd.to_numeric(merged["active_winners"], errors="coerce").fillna(0).astype(int)

    for col in [
        "candidate_avg_score",
        "candidate_avg_profit_factor",
        "candidate_avg_expectancy",
        "candidate_avg_trade_count",
        "active_avg_score",
        "active_avg_profit_factor",
        "active_avg_expectancy",
        "active_avg_trade_count",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["timeframe_sort_key"] = merged["timeframe"].map(TIMEFRAME_ORDER).fillna(999999)
    merged = merged.sort_values(
        ["timeframe_sort_key", "strategy_family", "micro_exit_variant"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return merged


def combo_strength_score(active_total: int, avg_score: float, avg_pf: float, avg_expectancy: float) -> float:
    # Conservative score:
    # - reward existence of active winners
    # - reward avg score
    # - reward PF modestly
    # - reward expectancy on log scale to avoid runaway dominance
    exp_component = 0.0
    if avg_expectancy is not None and not pd.isna(avg_expectancy):
        exp_component = math.log1p(max(float(avg_expectancy), 0.0))
    pf_component = max(float(avg_pf), 0.0) if avg_pf is not None and not pd.isna(avg_pf) else 0.0
    score_component = max(float(avg_score), 0.0) if avg_score is not None and not pd.isna(avg_score) else 0.0
    return round((active_total * 1.25) + (score_component * 4.0) + (pf_component * 2.5) + exp_component, 4)


def build_supported_expansion_manifest(
    stats_df: pd.DataFrame,
    min_target_per_exit: int,
    max_target_per_exit: int,
    min_candidate_rows_floor: int,
    top_combos: int,
) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()

    combos: List[Dict[str, object]] = []

    tf_family_groups = stats_df.groupby(["timeframe", "strategy_family"], dropna=False)
    for (timeframe, strategy_family), grp in tf_family_groups:
        grp_supported = grp[grp["micro_exit_variant"].isin(SUPPORTED_EXITS)].copy()

        if grp_supported.empty:
            continue

        # Current strength of this TF x family block
        active_total = int(grp_supported["active_winners"].fillna(0).sum())
        avg_score = avg_or_none(grp_supported["active_avg_score"])
        avg_pf = avg_or_none(grp_supported["active_avg_profit_factor"])
        avg_expectancy = avg_or_none(grp_supported["active_avg_expectancy"])
        family_strength = combo_strength_score(active_total, avg_score, avg_pf, avg_expectancy)

        max_active_supported = int(grp_supported["active_winners"].max()) if not grp_supported.empty else 0
        target_active = max(min_target_per_exit, max_active_supported)
        target_active = min(target_active, max_target_per_exit)

        max_candidate_supported = int(grp_supported["candidate_rows"].max()) if not grp_supported.empty else 0
        target_candidate = max(min_candidate_rows_floor, max_candidate_supported)

        logic_variant_hint = first_non_empty(grp_supported["logic_variant_sample"])

        existing_exits = set(grp_supported["micro_exit_variant"].astype(str).tolist())
        for exit_name in SUPPORTED_EXITS:
            row_match = grp_supported[grp_supported["micro_exit_variant"] == exit_name]
            if row_match.empty:
                current_candidate_rows = 0
                current_active_winners = 0
                active_avg_score = None
                active_avg_pf = None
                active_avg_expectancy = None
                candidate_avg_score = None
                candidate_avg_pf = None
                candidate_avg_expectancy = None
            else:
                record = row_match.iloc[0]
                current_candidate_rows = int(record["candidate_rows"])
                current_active_winners = int(record["active_winners"])
                active_avg_score = round_safe(record.get("active_avg_score"))
                active_avg_pf = round_safe(record.get("active_avg_profit_factor"))
                active_avg_expectancy = round_safe(record.get("active_avg_expectancy"))
                candidate_avg_score = round_safe(record.get("candidate_avg_score"))
                candidate_avg_pf = round_safe(record.get("candidate_avg_profit_factor"))
                candidate_avg_expectancy = round_safe(record.get("candidate_avg_expectancy"))

            active_gap = max(target_active - current_active_winners, 0)
            candidate_gap = max(target_candidate - current_candidate_rows, 0)

            if active_gap <= 0 and candidate_gap <= 0:
                continue

            missing_penalty_bonus = 1.75 if current_active_winners == 0 else 1.0
            zero_candidate_bonus = 1.50 if current_candidate_rows == 0 else 1.0
            priority_score = round(
                (
                    family_strength
                    + (active_gap * 3.0)
                    + (candidate_gap * 0.05)
                )
                * missing_penalty_bonus
                * zero_candidate_bonus,
                4,
            )

            reason_parts = []
            if current_active_winners == 0:
                reason_parts.append("no_active_winners")
            elif active_gap > 0:
                reason_parts.append("under_target_active_winners")

            if current_candidate_rows == 0:
                reason_parts.append("no_candidate_rows")
            elif candidate_gap > 0:
                reason_parts.append("under_target_candidate_rows")

            if exit_name not in existing_exits:
                reason_parts.append("exit_not_present_in_tf_family_combo")

            if not reason_parts:
                reason_parts.append("rebalance_needed")

            combos.append(
                {
                    "job_id": f"{timeframe}__{strategy_family}__{exit_name}",
                    "registry_version": VERSION,
                    "timeframe": timeframe,
                    "strategy_family": strategy_family,
                    "logic_variant_hint": logic_variant_hint,
                    "micro_exit_variant": exit_name,
                    "support_status": "supported",
                    "current_candidate_rows": current_candidate_rows,
                    "current_active_winners": current_active_winners,
                    "target_candidate_rows": target_candidate,
                    "target_active_winners": target_active,
                    "candidate_gap": candidate_gap,
                    "active_gap": active_gap,
                    "family_strength_score": family_strength,
                    "priority_score": priority_score,
                    "candidate_avg_score": candidate_avg_score,
                    "candidate_avg_profit_factor": candidate_avg_pf,
                    "candidate_avg_expectancy": candidate_avg_expectancy,
                    "active_avg_score": active_avg_score,
                    "active_avg_profit_factor": active_avg_pf,
                    "active_avg_expectancy": active_avg_expectancy,
                    "reason": "|".join(reason_parts),
                    "recommended_action": "expand_supported_exit_coverage",
                }
            )

    manifest = pd.DataFrame(combos)
    if manifest.empty:
        return manifest

    manifest["timeframe_sort_key"] = manifest["timeframe"].map(TIMEFRAME_ORDER).fillna(999999)
    manifest = manifest.sort_values(
        ["priority_score", "timeframe_sort_key", "strategy_family", "micro_exit_variant"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    if top_combos > 0:
        manifest = manifest.head(top_combos).copy()

    manifest.insert(0, "manifest_rank", range(1, len(manifest) + 1))
    return manifest


def build_design_manifest(
    active_df: pd.DataFrame,
    top_design_rows: int,
) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()

    family_tf_stats = (
        active_df.groupby(["timeframe", "strategy_family"], dropna=False)
        .agg(
            active_winners_total=("strategy_family", "size"),
            avg_score=("score", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_expectancy=("expectancy", "mean"),
            dominant_logic_variant=("logic_variant", first_non_empty),
        )
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for _, row in family_tf_stats.iterrows():
        timeframe = str(row["timeframe"])
        strategy_family = str(row["strategy_family"])
        active_winners_total = int(row["active_winners_total"])
        avg_score = round_safe(row["avg_score"])
        avg_pf = round_safe(row["avg_profit_factor"])
        avg_expectancy = round_safe(row["avg_expectancy"])
        dominant_logic_variant = str(row["dominant_logic_variant"]).strip()

        strength = combo_strength_score(active_winners_total, row["avg_score"], row["avg_profit_factor"], row["avg_expectancy"])

        for exit_name in PROPOSED_NEW_EXITS:
            priority = round(
                strength
                + (active_winners_total * 0.75)
                + (0.5 if avg_pf and avg_pf > 1.2 else 0.0)
                + (0.5 if avg_score and avg_score > 3.0 else 0.0),
                4,
            )
            rows.append(
                {
                    "job_id": f"DESIGN__{timeframe}__{strategy_family}__{exit_name}",
                    "registry_version": VERSION,
                    "timeframe": timeframe,
                    "strategy_family": strategy_family,
                    "logic_variant_hint": dominant_logic_variant,
                    "micro_exit_variant": exit_name,
                    "support_status": "requires_implementation",
                    "current_active_winners_total_tf_family": active_winners_total,
                    "tf_family_strength_score": strength,
                    "priority_score": priority,
                    "active_avg_score": avg_score,
                    "active_avg_profit_factor": avg_pf,
                    "active_avg_expectancy": avg_expectancy,
                    "reason": "new_exit_family_needed_for_true_intelligence",
                    "recommended_action": "design_and_implement_new_exit_family",
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest

    manifest["timeframe_sort_key"] = manifest["timeframe"].map(TIMEFRAME_ORDER).fillna(999999)
    manifest = manifest.sort_values(
        ["priority_score", "timeframe_sort_key", "strategy_family", "micro_exit_variant"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    if top_design_rows > 0:
        manifest = manifest.head(top_design_rows).copy()

    manifest.insert(0, "manifest_rank", range(1, len(manifest) + 1))
    return manifest


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_value_counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if col not in df.columns:
        return {}
    vc = df[col].fillna("").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def build_summary(
    candidate_df: pd.DataFrame,
    active_df: pd.DataFrame,
    supported_manifest: pd.DataFrame,
    design_manifest: pd.DataFrame,
) -> Dict[str, object]:
    candidate_exit_counts = safe_value_counts(candidate_df, "micro_exit_variant")
    active_exit_counts = safe_value_counts(active_df, "micro_exit_variant")
    strategy_family_counts = safe_value_counts(active_df, "strategy_family")
    timeframe_counts = safe_value_counts(active_df, "timeframe")

    top_supported_rows = supported_manifest.head(20).to_dict(orient="records") if not supported_manifest.empty else []
    top_design_rows = design_manifest.head(20).to_dict(orient="records") if not design_manifest.empty else []

    return {
        "version": VERSION,
        "candidate_rows": int(len(candidate_df)),
        "active_winners": int(len(active_df)),
        "candidate_micro_exit_counts": candidate_exit_counts,
        "active_micro_exit_counts": active_exit_counts,
        "active_strategy_family_counts": strategy_family_counts,
        "active_timeframe_counts": timeframe_counts,
        "supported_manifest_rows": int(len(supported_manifest)),
        "design_manifest_rows": int(len(design_manifest)),
        "top_supported_manifest_rows": top_supported_rows,
        "top_design_manifest_rows": top_design_rows,
    }


def write_summary_json(summary: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def build_recommendation_text(
    candidate_variant_stats: pd.DataFrame,
    active_variant_stats: pd.DataFrame,
    supported_manifest: pd.DataFrame,
    design_manifest: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("MICRO EXIT EXPANSION RECOMMENDATION")
    lines.append("=" * 100)
    lines.append(f"version={VERSION}")
    lines.append("")
    lines.append("Decision:")
    lines.append("- Current micro-exit coverage is imbalanced and not research-complete.")
    lines.append("- reverse_signal_exit dominates candidate generation too heavily.")
    lines.append("- The next mandatory move is balanced micro-exit expansion, not random sweep.")
    lines.append("")

    lines.append("Observed Candidate Micro Exit Coverage:")
    for _, row in candidate_variant_stats.iterrows():
        lines.append(
            f"- {row['micro_exit_variant']}: rows={int(row['rows'])} "
            f"avg_score={row['avg_score']} avg_pf={row['avg_profit_factor']} "
            f"avg_expectancy={row['avg_expectancy']}"
        )
    lines.append("")

    lines.append("Observed Active Winner Micro Exit Coverage:")
    for _, row in active_variant_stats.iterrows():
        lines.append(
            f"- {row['micro_exit_variant']}: winners={int(row['rows'])} "
            f"avg_score={row['avg_score']} avg_pf={row['avg_profit_factor']} "
            f"avg_expectancy={row['avg_expectancy']}"
        )
    lines.append("")

    lines.append("Top Supported Expansion Priorities:")
    if supported_manifest.empty:
        lines.append("- No supported expansion rows generated.")
    else:
        top_rows = supported_manifest.head(20)
        for _, row in top_rows.iterrows():
            lines.append(
                f"- rank={int(row['manifest_rank'])} tf={row['timeframe']} family={row['strategy_family']} "
                f"exit={row['micro_exit_variant']} active_gap={int(row['active_gap'])} "
                f"candidate_gap={int(row['candidate_gap'])} priority={row['priority_score']} "
                f"reason={row['reason']}"
            )
    lines.append("")

    lines.append("Top New Exit Design Priorities:")
    if design_manifest.empty:
        lines.append("- No design rows generated.")
    else:
        top_rows = design_manifest.head(20)
        for _, row in top_rows.iterrows():
            lines.append(
                f"- rank={int(row['manifest_rank'])} tf={row['timeframe']} family={row['strategy_family']} "
                f"exit={row['micro_exit_variant']} priority={row['priority_score']} "
                f"reason={row['reason']}"
            )
    lines.append("")
    lines.append("Operational Recommendation:")
    lines.append("- First run supported-expansion research for current exits to fix coverage imbalance.")
    lines.append("- Then implement new exit families from the design manifest.")
    lines.append("- After expansion, refresh registry and re-audit micro-exit distribution before moving to entry research.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    registry_root = Path(args.registry_root).resolve()
    outdir = Path(args.outdir).resolve()
    paths = ensure_paths(registry_root=registry_root, outdir=outdir)

    candidate_df = read_csv_safe(paths.timeframe_leaderboard_csv)
    active_df = read_csv_safe(paths.active_winner_registry_csv)

    candidate_df = normalize_df(candidate_df, NUMERIC_COLUMNS_CANDIDATE)
    active_df = normalize_df(active_df, NUMERIC_COLUMNS_ACTIVE)

    candidate_variant_stats = build_variant_stats(candidate_df, label="candidate")
    active_variant_stats = build_variant_stats(active_df, label="active_winner")

    tf_family_exit_stats = build_tf_family_exit_stats(candidate_df=candidate_df, active_df=active_df)

    supported_manifest = build_supported_expansion_manifest(
        stats_df=tf_family_exit_stats,
        min_target_per_exit=args.min_target_per_exit,
        max_target_per_exit=args.max_target_per_exit,
        min_candidate_rows_floor=args.min_candidate_rows_floor,
        top_combos=args.top_combos,
    )

    design_manifest = build_design_manifest(
        active_df=active_df,
        top_design_rows=args.top_design_rows,
    )

    summary = build_summary(
        candidate_df=candidate_df,
        active_df=active_df,
        supported_manifest=supported_manifest,
        design_manifest=design_manifest,
    )

    write_csv(candidate_variant_stats, outdir / "micro_exit_candidate_variant_stats.csv")
    write_csv(active_variant_stats, outdir / "micro_exit_active_variant_stats.csv")
    write_csv(tf_family_exit_stats, outdir / "micro_exit_tf_family_exit_stats.csv")

    write_csv(supported_manifest, outdir / "micro_exit_supported_expansion_manifest.csv")
    write_jsonl(supported_manifest, outdir / "micro_exit_supported_expansion_manifest.jsonl")

    write_csv(design_manifest, outdir / "micro_exit_design_manifest.csv")
    write_jsonl(design_manifest, outdir / "micro_exit_design_manifest.jsonl")

    write_summary_json(summary, outdir / "micro_exit_expansion_summary.json")

    recommendation_text = build_recommendation_text(
        candidate_variant_stats=candidate_variant_stats,
        active_variant_stats=active_variant_stats,
        supported_manifest=supported_manifest,
        design_manifest=design_manifest,
    )
    (outdir / "micro_exit_expansion_recommendation.txt").write_text(
        recommendation_text,
        encoding="utf-8",
    )

    print("=" * 108)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] registry_root={paths.registry_root}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] candidate_rows={len(candidate_df)}")
    print(f"[DONE] active_winners={len(active_df)}")
    print(f"[DONE] supported_manifest_rows={len(supported_manifest)}")
    print(f"[DONE] design_manifest_rows={len(design_manifest)}")
    print(f"[DONE] summary_json={outdir / 'micro_exit_expansion_summary.json'}")
    print(f"[DONE] supported_manifest_csv={outdir / 'micro_exit_supported_expansion_manifest.csv'}")
    print(f"[DONE] design_manifest_csv={outdir / 'micro_exit_design_manifest.csv'}")
    print("=" * 108)


if __name__ == "__main__":
    main()