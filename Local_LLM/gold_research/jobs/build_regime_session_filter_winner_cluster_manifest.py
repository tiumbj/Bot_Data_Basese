# ==================================================================================================
# FILE: build_regime_session_filter_winner_cluster_manifest.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_regime_session_filter_winner_cluster_manifest.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) New production-grade manifest builder for winner-cluster focused validation
#   2) Read promoted_jobs.csv + top_promoted_parameter_combinations.csv + analysis summary json
#   3) Build narrow-search phase-2 manifest only around promoted clusters
#   4) Output:
#      - winner_cluster_manifest.jsonl
#      - winner_cluster_manifest.csv
#      - winner_cluster_summary.json
#      - winner_cluster_recommendation.txt
#   5) Best-practice direction:
#      - stop broad search
#      - keep only parameters already proven by promoted jobs
#      - generate deterministic focused jobs for next validation phase
# ==================================================================================================

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import polars as pl

VERSION = "v1.0.0"

OUT_MANIFEST_JSONL = "winner_cluster_manifest.jsonl"
OUT_MANIFEST_CSV = "winner_cluster_manifest.csv"
OUT_SUMMARY_JSON = "winner_cluster_summary.json"
OUT_RECOMMENDATION_TXT = "winner_cluster_recommendation.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise RuntimeError(f"CSV file not found: {path}")
    return pl.read_csv(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def require_columns(df: pl.DataFrame, required_cols: List[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def top_values(df: pl.DataFrame, col: str, top_n: int) -> List[Any]:
    if df.height == 0:
        return []
    return (
        df.group_by(col)
        .agg(pl.len().alias("n"))
        .sort(["n", col], descending=[True, False])
        .head(top_n)
        .get_column(col)
        .to_list()
    )


def top_numeric_values(df: pl.DataFrame, col: str, top_n: int) -> List[float]:
    vals = top_values(df, col, top_n)
    return [safe_float(v) for v in vals]


def top_int_values(df: pl.DataFrame, col: str, top_n: int) -> List[int]:
    vals = top_values(df, col, top_n)
    return [safe_int(v) for v in vals]


def row_signature(row: Dict[str, Any], keys: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in keys)


def dedupe_rows(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        sig = row_signature(row, keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def build_job_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def build_combo_seed_jobs(combo_df: pl.DataFrame, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if combo_df.height == 0:
        return rows

    selected = combo_df.head(limit).to_dicts()
    for idx, combo in enumerate(selected, start=1):
        row = {
            "job_id": build_job_id("winner_combo", idx),
            "timeframe": "M30",
            "strategy_family": "REGIME_SESSION_FILTER_FAMILY_V1_0_0_FOCUSED",
            "entry_logic": "BASE_BREAKOUT_PLUS_FILTER",
            "regime_filter": "ALL",
            "side_policy": normalize_text(combo.get("side_policy"), "BOTH"),
            "session_filter": normalize_text(combo.get("session_filter"), "ALL"),
            "volatility_filter": normalize_text(combo.get("volatility_filter"), "ALL"),
            "trend_strength_filter": normalize_text(combo.get("trend_strength_filter"), "ALL"),
            "price_location_filter": normalize_text(combo.get("price_location_filter"), "ALL"),
            "selected_side_hint": normalize_text(combo.get("selected_side"), ""),
            "micro_exit": normalize_text(combo.get("micro_exit"), "LET_WINNERS_RUN"),
            "adx_min": safe_float(combo.get("adx_min"), 23.0),
            "breakout_atr_min": safe_float(combo.get("breakout_atr_min"), 0.05),
            "body_ratio_min": safe_float(combo.get("body_ratio_min"), 0.30),
            "max_extension_atr": safe_float(combo.get("max_extension_atr"), 1.00),
            "cooldown_bars": safe_int(combo.get("cooldown_bars"), 0),
            "focus_source": "top_promoted_parameter_combinations",
            "focus_rank": idx,
        }
        rows.append(row)

    return rows


def build_cluster_cross_jobs(
    promoted_df: pl.DataFrame,
    sessions: List[str],
    vols: List[str],
    trends: List[str],
    price_locations: List[str],
    side_policies: List[str],
    micro_exits: List[str],
    adx_vals: List[float],
    breakout_atr_vals: List[float],
    body_ratio_vals: List[float],
    max_extension_vals: List[float],
    cooldown_vals: List[int],
    per_side_limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 1

    for side_policy in side_policies:
        for session_filter in sessions:
            for volatility_filter in vols:
                for trend_strength_filter in trends:
                    for price_location_filter in price_locations:
                        for micro_exit in micro_exits:
                            for adx_min in adx_vals:
                                for breakout_atr_min in breakout_atr_vals:
                                    for body_ratio_min in body_ratio_vals:
                                        for max_extension_atr in max_extension_vals:
                                            for cooldown_bars in cooldown_vals:
                                                row = {
                                                    "job_id": build_job_id("winner_cluster", idx),
                                                    "timeframe": "M30",
                                                    "strategy_family": "REGIME_SESSION_FILTER_FAMILY_V1_0_0_FOCUSED",
                                                    "entry_logic": "BASE_BREAKOUT_PLUS_FILTER",
                                                    "regime_filter": "ALL",
                                                    "side_policy": side_policy,
                                                    "session_filter": session_filter,
                                                    "volatility_filter": volatility_filter,
                                                    "trend_strength_filter": trend_strength_filter,
                                                    "price_location_filter": price_location_filter,
                                                    "micro_exit": micro_exit,
                                                    "adx_min": round(safe_float(adx_min), 6),
                                                    "breakout_atr_min": round(safe_float(breakout_atr_min), 6),
                                                    "body_ratio_min": round(safe_float(body_ratio_min), 6),
                                                    "max_extension_atr": round(safe_float(max_extension_atr), 6),
                                                    "cooldown_bars": safe_int(cooldown_bars),
                                                    "focus_source": "winner_cluster_cross",
                                                    "focus_rank": idx,
                                                }
                                                rows.append(row)
                                                idx += 1
                                                if len(rows) >= per_side_limit:
                                                    return rows
    return rows


def build_local_perturbation_jobs(combo_df: pl.DataFrame, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 1

    if combo_df.height == 0:
        return rows

    base_rows = combo_df.head(min(10, combo_df.height)).to_dicts()

    adx_offsets = [-3.0, 0.0, 3.0]
    breakout_offsets = [-0.02, 0.0, 0.02]
    body_offsets = [-0.05, 0.0, 0.05]
    extension_offsets = [-0.20, 0.0, 0.20]
    cooldown_offsets = [-1, 0, 1]

    for base in base_rows:
        for adx_off in adx_offsets:
            for breakout_off in breakout_offsets:
                for body_off in body_offsets:
                    for ext_off in extension_offsets:
                        for cooldown_off in cooldown_offsets:
                            adx_min = max(15.0, safe_float(base.get("adx_min"), 23.0) + adx_off)
                            breakout_atr_min = max(0.01, safe_float(base.get("breakout_atr_min"), 0.05) + breakout_off)
                            body_ratio_min = min(0.80, max(0.10, safe_float(base.get("body_ratio_min"), 0.30) + body_off))
                            max_extension_atr = min(2.00, max(0.40, safe_float(base.get("max_extension_atr"), 1.00) + ext_off))
                            cooldown_bars = max(0, safe_int(base.get("cooldown_bars"), 0) + cooldown_off)

                            row = {
                                "job_id": build_job_id("winner_perturb", idx),
                                "timeframe": "M30",
                                "strategy_family": "REGIME_SESSION_FILTER_FAMILY_V1_0_0_FOCUSED",
                                "entry_logic": "BASE_BREAKOUT_PLUS_FILTER",
                                "regime_filter": "ALL",
                                "side_policy": normalize_text(base.get("side_policy"), "BOTH"),
                                "session_filter": normalize_text(base.get("session_filter"), "ALL"),
                                "volatility_filter": normalize_text(base.get("volatility_filter"), "ALL"),
                                "trend_strength_filter": normalize_text(base.get("trend_strength_filter"), "ALL"),
                                "price_location_filter": normalize_text(base.get("price_location_filter"), "ALL"),
                                "selected_side_hint": normalize_text(base.get("selected_side"), ""),
                                "micro_exit": normalize_text(base.get("micro_exit"), "LET_WINNERS_RUN"),
                                "adx_min": round(adx_min, 6),
                                "breakout_atr_min": round(breakout_atr_min, 6),
                                "body_ratio_min": round(body_ratio_min, 6),
                                "max_extension_atr": round(max_extension_atr, 6),
                                "cooldown_bars": cooldown_bars,
                                "focus_source": "top_combo_local_perturbation",
                                "focus_rank": idx,
                            }
                            rows.append(row)
                            idx += 1
                            if len(rows) >= limit:
                                return rows

    return rows


def build_recommendation(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"VERSION: {VERSION}")
    lines.append("PURPOSE: Build focused phase-2 manifest from promoted winner clusters")
    lines.append("")
    lines.append(f"INPUT_PROMOTED_JOBS: {summary['input_promoted_jobs']}")
    lines.append(f"INPUT_TOP_COMBOS_USED: {summary['top_combo_rows_used']}")
    lines.append(f"OUTPUT_MANIFEST_JOBS: {summary['output_manifest_jobs']}")
    lines.append("")
    lines.append("TOP WINNER CLUSTER VALUES:")
    lines.append(f"- session_filters = {', '.join(summary['top_session_filters'])}")
    lines.append(f"- volatility_filters = {', '.join(summary['top_volatility_filters'])}")
    lines.append(f"- trend_strength_filters = {', '.join(summary['top_trend_strength_filters'])}")
    lines.append(f"- price_location_filters = {', '.join(summary['top_price_location_filters'])}")
    lines.append(f"- side_policies = {', '.join(summary['top_side_policies'])}")
    lines.append(f"- micro_exits = {', '.join(summary['top_micro_exits'])}")
    lines.append(f"- adx_min_values = {', '.join(str(x) for x in summary['top_adx_min_values'])}")
    lines.append(f"- breakout_atr_min_values = {', '.join(str(x) for x in summary['top_breakout_atr_min_values'])}")
    lines.append(f"- body_ratio_min_values = {', '.join(str(x) for x in summary['top_body_ratio_min_values'])}")
    lines.append(f"- max_extension_atr_values = {', '.join(str(x) for x in summary['top_max_extension_atr_values'])}")
    lines.append(f"- cooldown_bars_values = {', '.join(str(x) for x in summary['top_cooldown_bars_values'])}")
    lines.append("")
    lines.append("DECISION:")
    lines.append("- Focus only on winner clusters proven by promoted jobs.")
    lines.append("- Use this manifest for phase-2 focused validation, not for broad discovery.")
    lines.append("- Best practice is to tighten around winner clusters before any new family expansion.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build winner-cluster manifest for focused phase-2 validation")
    parser.add_argument("--promoted-csv", required=True, dest="promoted_csv", help="Path to promoted_jobs.csv")
    parser.add_argument("--combo-csv", required=True, dest="combo_csv", help="Path to top_promoted_parameter_combinations.csv")
    parser.add_argument(
        "--analysis-summary-json",
        required=True,
        dest="analysis_summary_json",
        help="Path to regime_session_filter_family_analysis_summary.json",
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--combo-seed-limit", type=int, default=25, help="Top combo seed jobs to keep")
    parser.add_argument("--cluster-cross-limit", type=int, default=500, help="Cross-cluster jobs limit")
    parser.add_argument("--perturb-limit", type=int, default=500, help="Local perturbation jobs limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    promoted_csv_path = Path(args.promoted_csv)
    combo_csv_path = Path(args.combo_csv)
    analysis_summary_json_path = Path(args.analysis_summary_json)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    promoted_df = load_csv(promoted_csv_path)
    combo_df = load_csv(combo_csv_path)
    analysis_summary = load_json(analysis_summary_json_path)

    require_columns(
        promoted_df,
        [
            "job_id",
            "session_filter",
            "volatility_filter",
            "trend_strength_filter",
            "price_location_filter",
            "side_policy",
            "selected_side",
            "micro_exit",
            "adx_min",
            "breakout_atr_min",
            "body_ratio_min",
            "max_extension_atr",
            "cooldown_bars",
            "profit_factor",
            "total_return_pct",
            "max_drawdown_pct",
            "score",
        ],
        "promoted_csv",
    )
    require_columns(
        combo_df,
        [
            "session_filter",
            "volatility_filter",
            "trend_strength_filter",
            "price_location_filter",
            "side_policy",
            "selected_side",
            "micro_exit",
            "adx_min",
            "breakout_atr_min",
            "body_ratio_min",
            "max_extension_atr",
            "cooldown_bars",
            "combo_jobs",
            "best_job_id",
        ],
        "combo_csv",
    )

    if promoted_df.height <= 0:
        raise RuntimeError("promoted_csv contains zero promoted jobs. Cannot build focused manifest.")

    top_session_filters = [normalize_text(x) for x in top_values(promoted_df, "session_filter", 3)]
    top_volatility_filters = [normalize_text(x) for x in top_values(promoted_df, "volatility_filter", 3)]
    top_trend_strength_filters = [normalize_text(x) for x in top_values(promoted_df, "trend_strength_filter", 3)]
    top_price_location_filters = [normalize_text(x) for x in top_values(promoted_df, "price_location_filter", 3)]
    top_side_policies = [normalize_text(x) for x in top_values(promoted_df, "side_policy", 3)]
    top_micro_exits = [normalize_text(x) for x in top_values(promoted_df, "micro_exit", 3)]

    top_adx_min_values = top_numeric_values(promoted_df, "adx_min", 3)
    top_breakout_atr_min_values = top_numeric_values(promoted_df, "breakout_atr_min", 3)
    top_body_ratio_min_values = top_numeric_values(promoted_df, "body_ratio_min", 3)
    top_max_extension_atr_values = top_numeric_values(promoted_df, "max_extension_atr", 3)
    top_cooldown_bars_values = top_int_values(promoted_df, "cooldown_bars", 3)

    combo_seed_jobs = build_combo_seed_jobs(combo_df, limit=max(1, args.combo_seed_limit))
    cluster_cross_jobs = build_cluster_cross_jobs(
        promoted_df=promoted_df,
        sessions=top_session_filters or ["ALL"],
        vols=top_volatility_filters or ["ALL"],
        trends=top_trend_strength_filters or ["ALL"],
        price_locations=top_price_location_filters or ["ALL"],
        side_policies=top_side_policies or ["BOTH"],
        micro_exits=top_micro_exits or ["LET_WINNERS_RUN"],
        adx_vals=top_adx_min_values or [23.0],
        breakout_atr_vals=top_breakout_atr_min_values or [0.05],
        body_ratio_vals=top_body_ratio_min_values or [0.30],
        max_extension_vals=top_max_extension_atr_values or [1.00],
        cooldown_vals=top_cooldown_bars_values or [0],
        per_side_limit=max(1, args.cluster_cross_limit),
    )
    perturb_jobs = build_local_perturbation_jobs(combo_df, limit=max(1, args.perturb_limit))

    manifest_rows = combo_seed_jobs + cluster_cross_jobs + perturb_jobs
    manifest_rows = dedupe_rows(
        manifest_rows,
        keys=[
            "timeframe",
            "strategy_family",
            "entry_logic",
            "regime_filter",
            "side_policy",
            "session_filter",
            "volatility_filter",
            "trend_strength_filter",
            "price_location_filter",
            "micro_exit",
            "adx_min",
            "breakout_atr_min",
            "body_ratio_min",
            "max_extension_atr",
            "cooldown_bars",
        ],
    )

    for idx, row in enumerate(manifest_rows, start=1):
        row["job_id"] = build_job_id("focused_phase2", idx)

    manifest_df = pl.DataFrame(manifest_rows)
    append_jsonl(outdir / OUT_MANIFEST_JSONL, manifest_rows)
    manifest_df.write_csv(outdir / OUT_MANIFEST_CSV)

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "input_promoted_csv_path": str(promoted_csv_path),
        "input_combo_csv_path": str(combo_csv_path),
        "input_analysis_summary_json_path": str(analysis_summary_json_path),
        "input_promoted_jobs": int(promoted_df.height),
        "input_analysis_top_promoted_job_id": normalize_text(analysis_summary.get("top_promoted_job_id"), ""),
        "top_combo_rows_used": int(min(combo_df.height, max(1, args.combo_seed_limit))),
        "combo_seed_jobs": int(len(combo_seed_jobs)),
        "cluster_cross_jobs": int(len(cluster_cross_jobs)),
        "perturb_jobs": int(len(perturb_jobs)),
        "output_manifest_jobs": int(manifest_df.height),
        "top_session_filters": top_session_filters,
        "top_volatility_filters": top_volatility_filters,
        "top_trend_strength_filters": top_trend_strength_filters,
        "top_price_location_filters": top_price_location_filters,
        "top_side_policies": top_side_policies,
        "top_micro_exits": top_micro_exits,
        "top_adx_min_values": top_adx_min_values,
        "top_breakout_atr_min_values": top_breakout_atr_min_values,
        "top_body_ratio_min_values": top_body_ratio_min_values,
        "top_max_extension_atr_values": top_max_extension_atr_values,
        "top_cooldown_bars_values": top_cooldown_bars_values,
    }
    write_json(outdir / OUT_SUMMARY_JSON, summary)
    write_text(outdir / OUT_RECOMMENDATION_TXT, build_recommendation(summary))

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest_jsonl={outdir / OUT_MANIFEST_JSONL}")
    print(f"[DONE] manifest_csv={outdir / OUT_MANIFEST_CSV}")
    print(f"[DONE] summary_json={outdir / OUT_SUMMARY_JSON}")
    print(f"[DONE] recommendation_txt={outdir / OUT_RECOMMENDATION_TXT}")
    print(f"[DONE] input_promoted_jobs={summary['input_promoted_jobs']}")
    print(f"[DONE] output_manifest_jobs={summary['output_manifest_jobs']}")
    print("=" * 120)


if __name__ == "__main__":
    main()