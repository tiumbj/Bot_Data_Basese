# ============================================================
# ชื่อโค้ด: build_truth_gate_seed_manifest_v1_0_0.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_truth_gate_seed_manifest_v1_0_0.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_truth_gate_seed_manifest_v1_0_0.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

VERSION = "v1.0.0"
ROOT = Path(r"C:\Data\Bot")
CENTRAL_BACKTEST_RESULTS = ROOT / "central_backtest_results"
TRUTH_GATE_DIR = CENTRAL_BACKTEST_RESULTS / "research_evidence_truth_gate_v1_0_0"
DEFAULT_OUTDIR = CENTRAL_BACKTEST_RESULTS / "truth_gate_seed_manifest_v1_0_0"
MANIFEST_CSV = CENTRAL_BACKTEST_RESULTS / "research_coverage_master_v1_0_0" / "research_coverage_master_manifest.csv"
MANIFEST_JSONL = CENTRAL_BACKTEST_RESULTS / "research_coverage_master_v1_0_0" / "research_coverage_master_manifest.jsonl"

LOCKED_RESEARCH_TIMEFRAMES = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"
]

SEED_FAMILY_HINTS = [
    "micro_exit",
    "pullback",
    "trend",
    "breakout",
    "continuation",
    "bos",
    "choch",
    "ema",
    "adx",
    "regime",
]

REQUIRED_TRUTH_GATE_FILES = {
    "coverage": "coverage_by_timeframe.csv",
    "micro_exit": "micro_exit_coverage_matrix.csv",
    "kill_stage": "kill_stage_summary.csv",
    "metrics": "standardized_metrics_pack.csv",
    "summary": "truth_gate_summary.json",
    "recommendation": "truth_gate_recommendation.txt",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def normalize_tf(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in LOCKED_RESEARCH_TIMEFRAMES else None


@dataclass
class TruthGateContext:
    coverage_df: pd.DataFrame
    micro_exit_df: pd.DataFrame
    kill_stage_df: pd.DataFrame
    metrics_df: pd.DataFrame
    summary_payload: Dict[str, Any]
    recommendation_text: str


@dataclass
class ManifestSource:
    path: Path
    kind: str
    exists: bool



def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build seed manifest from truth gate outputs.")
    parser.add_argument("--truth-gate-dir", type=Path, default=TRUTH_GATE_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--manifest-jsonl", type=Path, default=MANIFEST_JSONL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed-per-timeframe", type=int, default=250)
    parser.add_argument("--max-manifest-rows", type=int, default=150000)
    return parser



def validate_truth_gate_dir(truth_gate_dir: Path) -> None:
    missing: List[str] = []
    for file_name in REQUIRED_TRUTH_GATE_FILES.values():
        if not (truth_gate_dir / file_name).exists():
            missing.append(file_name)
    if missing:
        raise FileNotFoundError(
            f"truth gate directory is incomplete: missing={missing} at {truth_gate_dir}"
        )



def load_truth_gate_context(truth_gate_dir: Path) -> TruthGateContext:
    validate_truth_gate_dir(truth_gate_dir)
    coverage_df = load_csv(truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["coverage"])
    micro_exit_df = load_csv(truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["micro_exit"])
    kill_stage_df = load_csv(truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["kill_stage"])
    metrics_df = load_csv(truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["metrics"])
    summary_payload = read_json(truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["summary"])
    recommendation_text = (truth_gate_dir / REQUIRED_TRUTH_GATE_FILES["recommendation"]).read_text(encoding="utf-8")
    return TruthGateContext(
        coverage_df=coverage_df,
        micro_exit_df=micro_exit_df,
        kill_stage_df=kill_stage_df,
        metrics_df=metrics_df,
        summary_payload=summary_payload,
        recommendation_text=recommendation_text,
    )



def choose_manifest_source(manifest_csv: Path, manifest_jsonl: Path) -> ManifestSource:
    if manifest_csv.exists():
        return ManifestSource(path=manifest_csv, kind="csv", exists=True)
    if manifest_jsonl.exists():
        return ManifestSource(path=manifest_jsonl, kind="jsonl", exists=True)
    return ManifestSource(path=manifest_csv, kind="missing", exists=False)



def read_manifest(source: ManifestSource, max_rows: int) -> pd.DataFrame:
    if not source.exists:
        return pd.DataFrame()

    if source.kind == "csv":
        try:
            return pd.read_csv(source.path, nrows=max_rows)
        except UnicodeDecodeError:
            return pd.read_csv(source.path, nrows=max_rows, encoding="latin1")

    rows: List[Dict[str, Any]] = []
    with source.path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception:
                continue
    return pd.DataFrame(rows)



def infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lowered_map = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in lowered_map:
            return lowered_map[col.lower()]
    return None



def prepare_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    timeframe_col = infer_column(df, ["timeframe", "tf", "current_timeframe", "deployment_timeframe", "research_timeframe"])
    family_col = infer_column(df, ["strategy_family", "family", "strategy_type", "strategy_group"])
    strictness_col = infer_column(df, ["logic_strictness", "strictness", "filter_strictness"])
    micro_exit_col = infer_column(df, ["micro_exit_name", "micro_exit_variant", "exit_variant", "exit_name"])
    strategy_id_col = infer_column(df, ["strategy_id", "job_id", "id", "package_id"])
    side_col = infer_column(df, ["side", "trade_side", "direction", "side_policy"])

    working = df.copy()
    working["seed_timeframe"] = working[timeframe_col].map(normalize_tf) if timeframe_col else None
    working["seed_family"] = working[family_col].astype(str) if family_col else "unknown_family"
    working["seed_strictness"] = working[strictness_col].astype(str) if strictness_col else "UNKNOWN"
    working["seed_micro_exit"] = working[micro_exit_col].astype(str) if micro_exit_col else "UNKNOWN"
    working["seed_strategy_id"] = working[strategy_id_col].astype(str) if strategy_id_col else working.index.astype(str)
    working["seed_side"] = working[side_col].astype(str) if side_col else "BOTH"

    def score_family(text: Any) -> int:
        lowered = str(text).lower()
        score = 0
        for idx, hint in enumerate(SEED_FAMILY_HINTS):
            if hint in lowered:
                score += max(1, len(SEED_FAMILY_HINTS) - idx)
        return score

    def score_micro_exit(text: Any) -> int:
        lowered = str(text).lower()
        if lowered == "unknown":
            return 0
        if "micro_exit" in lowered:
            return 8
        if "fast_invalidation" in lowered:
            return 7
        if "time_stop" in lowered:
            return 6
        return 3

    working["seed_family_score"] = working["seed_family"].map(score_family)
    working["seed_micro_exit_score"] = working["seed_micro_exit"].map(score_micro_exit)
    working["seed_total_score"] = working["seed_family_score"] + working["seed_micro_exit_score"]
    return working



def build_seed_set(
    manifest_df: pd.DataFrame,
    truth_gate: TruthGateContext,
    seed_per_timeframe: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if manifest_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {
            "decision": "NO_MANIFEST_DATA",
            "seed_rows": 0,
            "timeframes_selected": [],
            "timeframes_missing": LOCKED_RESEARCH_TIMEFRAMES,
        }

    coverage_df = truth_gate.coverage_df.copy()
    micro_exit_df = truth_gate.micro_exit_df.copy()

    coverage_df["timeframe"] = coverage_df["timeframe"].map(normalize_tf)
    micro_exit_df["timeframe"] = micro_exit_df["timeframe"].map(normalize_tf)

    merged_tf = coverage_df.merge(
        micro_exit_df[["timeframe", "gate_pass", "micro_exit_status"]],
        how="left",
        on="timeframe",
        suffixes=("", "_micro"),
    )

    eligible_tfs = merged_tf.loc[
        (merged_tf["coverage_status"].isin(["PRESENT", "STRONG"]))
        & (merged_tf["gate_pass"] == True),
        "timeframe",
    ].dropna().tolist()

    manifest_df = manifest_df.loc[manifest_df["seed_timeframe"].isin(eligible_tfs)].copy()
    if manifest_df.empty:
        empty = pd.DataFrame()
        return empty, merged_tf, {
            "decision": "NO_ELIGIBLE_ROWS_AFTER_TF_GATE",
            "seed_rows": 0,
            "timeframes_selected": eligible_tfs,
            "timeframes_missing": [tf for tf in LOCKED_RESEARCH_TIMEFRAMES if tf not in eligible_tfs],
        }

    kill_stage = "UNSPECIFIED"
    if not truth_gate.kill_stage_df.empty and "top_kill_stage" in truth_gate.kill_stage_df.columns:
        kill_stage = str(truth_gate.kill_stage_df.iloc[0]["top_kill_stage"])

    strictness_penalty = {
        "FILTER_INTERSECTION": ["VERY_STRICT", "STRICT", "ULTRA_STRICT"],
        "REGIME_FILTER": ["REGIME_HARD"],
        "TIMEFRAME_FILTER": ["TIMEFRAME_HARD"],
    }
    blocked_strictness = strictness_penalty.get(kill_stage, [])
    if blocked_strictness:
        manifest_df = manifest_df.loc[~manifest_df["seed_strictness"].str.upper().isin(blocked_strictness)].copy()

    if manifest_df.empty:
        empty = pd.DataFrame()
        return empty, merged_tf, {
            "decision": "NO_ROWS_AFTER_KILL_STAGE_FILTER",
            "seed_rows": 0,
            "timeframes_selected": eligible_tfs,
            "timeframes_missing": [tf for tf in LOCKED_RESEARCH_TIMEFRAMES if tf not in eligible_tfs],
        }

    manifest_df = manifest_df.sort_values(
        by=["seed_timeframe", "seed_total_score", "seed_family_score", "seed_micro_exit_score", "seed_strategy_id"],
        ascending=[True, False, False, False, True],
    ).copy()

    grouped = []
    for tf in eligible_tfs:
        tf_df = manifest_df.loc[manifest_df["seed_timeframe"] == tf].copy()
        if tf_df.empty:
            continue
        top_df = tf_df.head(seed_per_timeframe).copy()
        grouped.append(top_df)

    if not grouped:
        empty = pd.DataFrame()
        return empty, merged_tf, {
            "decision": "NO_GROUPED_SEEDS",
            "seed_rows": 0,
            "timeframes_selected": eligible_tfs,
            "timeframes_missing": [tf for tf in LOCKED_RESEARCH_TIMEFRAMES if tf not in eligible_tfs],
        }

    seed_df = pd.concat(grouped, axis=0, ignore_index=True)
    seed_df.insert(0, "seed_rank", range(1, len(seed_df) + 1))

    manifest_jsonl_rows: List[Dict[str, Any]] = []
    for _, row in seed_df.iterrows():
        payload = row.to_dict()
        payload["truth_gate_version"] = truth_gate.summary_payload.get("version")
        payload["seed_manifest_version"] = VERSION
        payload["seed_generated_at_utc"] = utc_now_iso()
        manifest_jsonl_rows.append(payload)

    summary = {
        "decision": "ALLOW_VT_FULL_RUN_ON_SEED_SET",
        "seed_rows": len(seed_df),
        "eligible_timeframes": eligible_tfs,
        "timeframes_missing": [tf for tf in LOCKED_RESEARCH_TIMEFRAMES if tf not in eligible_tfs],
        "kill_stage_applied": kill_stage,
        "seed_per_timeframe": seed_per_timeframe,
        "family_distribution": seed_df["seed_family"].value_counts().head(50).to_dict(),
        "timeframe_distribution": seed_df["seed_timeframe"].value_counts().to_dict(),
    }
    return seed_df, pd.DataFrame(manifest_jsonl_rows), summary



def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")



def build_recommendation_text(summary: Dict[str, Any], truth_gate: TruthGateContext) -> str:
    lines: List[str] = []
    lines.append("=" * 120)
    lines.append(f"[SEED MANIFEST] version={VERSION}")
    lines.append(f"[SEED MANIFEST] generated_at_utc={utc_now_iso()}")
    lines.append(f"[SEED MANIFEST] decision={summary.get('decision')}")
    lines.append(f"[SEED MANIFEST] seed_rows={summary.get('seed_rows')}")
    lines.append(f"[SEED MANIFEST] kill_stage_applied={summary.get('kill_stage_applied')}")
    lines.append(f"[SEED MANIFEST] eligible_timeframes={summary.get('eligible_timeframes')}")
    lines.append(f"[SEED MANIFEST] timeframes_missing={summary.get('timeframes_missing')}")
    lines.append("=" * 120)
    lines.append("หลักการตัดสิน:")
    lines.append("1) ผ่าน Truth Gate ก่อน")
    lines.append("2) เลือกเฉพาะ TF ที่มี coverage + micro exit evidence")
    lines.append("3) ตัดแถวที่เสี่ยงชน killer layer หลัก")
    lines.append("4) เก็บเฉพาะ seed set สำหรับ VT full-run รอบถัดไป")
    lines.append("")
    lines.append("Truth Gate Recommendation เดิม:")
    lines.append(truth_gate.recommendation_text.strip())
    lines.append("")
    if summary.get("decision") != "ALLOW_VT_FULL_RUN_ON_SEED_SET":
        lines.append("คำตัดสิน: ห้ามเปิด full-run ใหม่จนกว่าจะมี seed manifest ที่ใช้ได้")
    else:
        lines.append("คำตัดสิน: อนุญาตให้ใช้ manifest นี้สำหรับ VT full-run เฉพาะ seed set")
    return "\n".join(lines) + "\n"



def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    truth_gate = load_truth_gate_context(args.truth_gate_dir)
    ensure_dir(args.outdir)

    source = choose_manifest_source(args.manifest_csv, args.manifest_jsonl)
    manifest_df = read_manifest(source, max_rows=args.max_manifest_rows)
    manifest_df = prepare_manifest_df(manifest_df)

    seed_df, seed_jsonl_df, summary = build_seed_set(
        manifest_df=manifest_df,
        truth_gate=truth_gate,
        seed_per_timeframe=args.seed_per_timeframe,
    )

    out_csv = args.outdir / "truth_gate_seed_manifest.csv"
    out_jsonl = args.outdir / "truth_gate_seed_manifest.jsonl"
    out_summary = args.outdir / "truth_gate_seed_summary.json"
    out_reco = args.outdir / "truth_gate_seed_recommendation.txt"

    if not seed_df.empty:
        seed_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        save_jsonl(seed_jsonl_df, out_jsonl)
    else:
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        out_jsonl.write_text("", encoding="utf-8")

    summary_payload = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "truth_gate_dir": str(args.truth_gate_dir),
        "manifest_source": str(source.path),
        "manifest_kind": source.kind,
        "manifest_exists": source.exists,
        "max_manifest_rows": args.max_manifest_rows,
        "seed_per_timeframe": args.seed_per_timeframe,
        "recommendation": summary,
        "output_csv": str(out_csv),
        "output_jsonl": str(out_jsonl),
        "output_recommendation": str(out_reco),
        "output_summary": str(out_summary),
    }
    write_json(out_summary, summary_payload)
    write_text(out_reco, build_recommendation_text(summary, truth_gate))

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] truth_gate_dir={args.truth_gate_dir}")
    print(f"[DONE] manifest_source={source.path}")
    print(f"[DONE] manifest_kind={source.kind}")
    print(f"[DONE] outdir={args.outdir}")
    print(f"[DONE] truth_gate_seed_manifest_csv={out_csv}")
    print(f"[DONE] truth_gate_seed_manifest_jsonl={out_jsonl}")
    print(f"[DONE] truth_gate_seed_summary_json={out_summary}")
    print(f"[DONE] truth_gate_seed_recommendation_txt={out_reco}")
    print(f"[DONE] decision={summary.get('decision')}")
    print(f"[DONE] seed_rows={summary.get('seed_rows')}")
    print(f"[DONE] eligible_timeframes={summary.get('eligible_timeframes')}")
    print("=" * 120)


if __name__ == "__main__":
    main()
