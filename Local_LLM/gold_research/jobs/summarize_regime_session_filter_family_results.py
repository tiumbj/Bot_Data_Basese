# ==================================================================================================
# FILE: summarize_regime_session_filter_family_results.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\summarize_regime_session_filter_family_results.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) New production-grade summarizer for regime/session filter family research results
#   2) Read results.csv and summary.json from vectorbt_regime_session_filter_family_v1_0_0
#   3) Summarize promoted-job patterns to find shared winning conditions
#   4) Output ranking files for:
#      - session_filter
#      - volatility_filter
#      - trend_strength_filter
#      - price_location_filter
#      - side_policy
#      - selected_side
#      - micro_exit
#      - adx_min
#      - breakout_atr_min
#      - body_ratio_min
#      - max_extension_atr
#      - cooldown_bars
#      - RSI bands
#   5) Output top promoted jobs and recommendation text
# ==================================================================================================

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import polars as pl

VERSION = "v1.0.0"

OUT_PROMOTED_CSV = "promoted_jobs.csv"
OUT_TOP_PROMOTED_CSV = "top_promoted_jobs.csv"
OUT_SESSION_CSV = "promoted_by_session_filter.csv"
OUT_VOL_CSV = "promoted_by_volatility_filter.csv"
OUT_TREND_CSV = "promoted_by_trend_strength_filter.csv"
OUT_PRICE_LOC_CSV = "promoted_by_price_location_filter.csv"
OUT_SIDE_POLICY_CSV = "promoted_by_side_policy.csv"
OUT_SELECTED_SIDE_CSV = "promoted_by_selected_side.csv"
OUT_MICRO_EXIT_CSV = "promoted_by_micro_exit.csv"
OUT_ADX_CSV = "promoted_by_adx_min.csv"
OUT_BREAKOUT_ATR_CSV = "promoted_by_breakout_atr_min.csv"
OUT_BODY_RATIO_CSV = "promoted_by_body_ratio_min.csv"
OUT_MAX_EXTENSION_CSV = "promoted_by_max_extension_atr.csv"
OUT_COOLDOWN_CSV = "promoted_by_cooldown_bars.csv"
OUT_RSI_LONG_BAND_CSV = "promoted_by_rsi_long_band.csv"
OUT_RSI_SHORT_BAND_CSV = "promoted_by_rsi_short_band.csv"
OUT_COMBO_CSV = "top_promoted_parameter_combinations.csv"
OUT_SUMMARY_JSON = "regime_session_filter_family_analysis_summary.json"
OUT_RECOMMENDATION_TXT = "regime_session_filter_family_recommendation.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_summary_json(path: Path) -> Dict:
    if not path.exists():
        raise RuntimeError(f"Summary JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_results_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Results CSV not found: {path}")

    df = pl.read_csv(path)

    required_cols = [
        "job_id",
        "selected_side",
        "trade_count",
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor",
        "expectancy",
        "score",
        "session_filter",
        "volatility_filter",
        "trend_strength_filter",
        "price_location_filter",
        "side_policy",
        "micro_exit",
        "adx_min",
        "breakout_atr_min",
        "body_ratio_min",
        "max_extension_atr",
        "rsi_long_min",
        "rsi_long_max",
        "rsi_short_min",
        "rsi_short_max",
        "cooldown_bars",
        "promoted",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Results CSV missing columns: {missing}")

    return df


def add_band_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.col("rsi_long_min").cast(pl.Float64).round(3).cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("rsi_long_max").cast(pl.Float64).round(3).cast(pl.Utf8)
            ).alias("rsi_long_band"),
            (
                pl.col("rsi_short_min").cast(pl.Float64).round(3).cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("rsi_short_max").cast(pl.Float64).round(3).cast(pl.Utf8)
            ).alias("rsi_short_band"),
        ]
    )


def summarize_dimension(df_all: pl.DataFrame, group_col: str) -> pl.DataFrame:
    promoted = df_all.filter(pl.col("promoted") == True)

    if promoted.height == 0:
        return pl.DataFrame(
            {
                group_col: [],
                "total_jobs": [],
                "promoted_jobs": [],
                "promoted_rate_pct": [],
                "mean_trade_count": [],
                "mean_profit_factor": [],
                "mean_total_return_pct": [],
                "mean_max_drawdown_pct": [],
                "mean_expectancy": [],
                "best_job_id": [],
                "best_score": [],
            }
        )

    total_by_group = (
        df_all.group_by(group_col)
        .agg(pl.len().alias("total_jobs"))
    )

    promoted_stats = (
        promoted.group_by(group_col)
        .agg(
            [
                pl.len().alias("promoted_jobs"),
                pl.col("trade_count").mean().round(6).alias("mean_trade_count"),
                pl.col("profit_factor").mean().round(6).alias("mean_profit_factor"),
                pl.col("total_return_pct").mean().round(6).alias("mean_total_return_pct"),
                pl.col("max_drawdown_pct").mean().round(6).alias("mean_max_drawdown_pct"),
                pl.col("expectancy").mean().round(6).alias("mean_expectancy"),
            ]
        )
    )

    best_rows = (
        promoted.sort(["score", "profit_factor", "total_return_pct"], descending=[True, True, True])
        .group_by(group_col)
        .agg(
            [
                pl.col("job_id").first().alias("best_job_id"),
                pl.col("score").first().round(6).alias("best_score"),
            ]
        )
    )

    out = (
        total_by_group
        .join(promoted_stats, on=group_col, how="left")
        .join(best_rows, on=group_col, how="left")
        .with_columns(
            [
                pl.col("promoted_jobs").fill_null(0).cast(pl.Int64),
                pl.col("mean_trade_count").fill_null(0.0),
                pl.col("mean_profit_factor").fill_null(0.0),
                pl.col("mean_total_return_pct").fill_null(0.0),
                pl.col("mean_max_drawdown_pct").fill_null(0.0),
                pl.col("mean_expectancy").fill_null(0.0),
                pl.col("best_job_id").fill_null(""),
                pl.col("best_score").fill_null(0.0),
            ]
        )
        .with_columns(
            (
                (pl.col("promoted_jobs") / pl.col("total_jobs")) * 100.0
            ).round(6).alias("promoted_rate_pct")
        )
        .sort(
            ["promoted_jobs", "promoted_rate_pct", "mean_profit_factor", "mean_total_return_pct"],
            descending=[True, True, True, True],
        )
    )
    return out


def summarize_combinations(promoted: pl.DataFrame) -> pl.DataFrame:
    if promoted.height == 0:
        return pl.DataFrame(
            {
                "session_filter": [],
                "volatility_filter": [],
                "trend_strength_filter": [],
                "price_location_filter": [],
                "side_policy": [],
                "micro_exit": [],
                "adx_min": [],
                "breakout_atr_min": [],
                "body_ratio_min": [],
                "max_extension_atr": [],
                "cooldown_bars": [],
                "selected_side": [],
                "combo_jobs": [],
                "mean_profit_factor": [],
                "mean_total_return_pct": [],
                "mean_max_drawdown_pct": [],
                "best_job_id": [],
                "best_score": [],
            }
        )

    key_cols = [
        "session_filter",
        "volatility_filter",
        "trend_strength_filter",
        "price_location_filter",
        "side_policy",
        "micro_exit",
        "adx_min",
        "breakout_atr_min",
        "body_ratio_min",
        "max_extension_atr",
        "cooldown_bars",
        "selected_side",
    ]

    combo_stats = (
        promoted.group_by(key_cols)
        .agg(
            [
                pl.len().alias("combo_jobs"),
                pl.col("profit_factor").mean().round(6).alias("mean_profit_factor"),
                pl.col("total_return_pct").mean().round(6).alias("mean_total_return_pct"),
                pl.col("max_drawdown_pct").mean().round(6).alias("mean_max_drawdown_pct"),
            ]
        )
    )

    combo_best = (
        promoted.sort(["score", "profit_factor", "total_return_pct"], descending=[True, True, True])
        .group_by(key_cols)
        .agg(
            [
                pl.col("job_id").first().alias("best_job_id"),
                pl.col("score").first().round(6).alias("best_score"),
            ]
        )
    )

    return (
        combo_stats
        .join(combo_best, on=key_cols, how="left")
        .sort(
            ["combo_jobs", "mean_profit_factor", "mean_total_return_pct", "best_score"],
            descending=[True, True, True, True],
        )
        .head(100)
    )


def build_recommendation(
    raw_summary: Dict,
    df_all: pl.DataFrame,
    promoted: pl.DataFrame,
    session_df: pl.DataFrame,
    vol_df: pl.DataFrame,
    trend_df: pl.DataFrame,
    side_df: pl.DataFrame,
    micro_exit_df: pl.DataFrame,
    adx_df: pl.DataFrame,
    combo_df: pl.DataFrame,
) -> str:
    lines: List[str] = []

    lines.append(f"VERSION: {VERSION}")
    lines.append("PURPOSE: Summarize promoted-job patterns from regime/session filter family")
    lines.append("")
    lines.append(f"INPUT_TOTAL_JOBS: {raw_summary.get('total_jobs', 0)}")
    lines.append(f"INPUT_PROMOTED_JOBS: {raw_summary.get('promoted_jobs', 0)}")
    lines.append(f"INPUT_PASS_JOBS: {raw_summary.get('pass_jobs', 0)}")
    lines.append(f"INPUT_NO_TRADE_JOBS: {raw_summary.get('no_trade_jobs', 0)}")
    lines.append(f"INPUT_BEST_JOB_ID: {raw_summary.get('best_job_id', '')}")
    lines.append("")

    if promoted.height == 0:
        lines.append("DECISION:")
        lines.append("- No promoted jobs found.")
        lines.append("- Current regime/session filter family did not produce a promotable edge.")
        lines.append("")
        return "\n".join(lines)

    top_promoted = (
        promoted.sort(["score", "profit_factor", "total_return_pct"], descending=[True, True, True])
        .select(
            [
                "job_id",
                "selected_side",
                "trade_count",
                "profit_factor",
                "total_return_pct",
                "max_drawdown_pct",
                "session_filter",
                "volatility_filter",
                "trend_strength_filter",
                "price_location_filter",
                "micro_exit",
                "adx_min",
                "cooldown_bars",
            ]
        )
        .head(10)
        .to_dicts()
    )

    lines.append("TOP 10 PROMOTED JOBS:")
    for idx, row in enumerate(top_promoted, start=1):
        lines.append(
            f"- #{idx} job_id={row['job_id']} side={row['selected_side']} "
            f"trades={row['trade_count']} pf={row['profit_factor']} ret%={row['total_return_pct']} "
            f"dd%={row['max_drawdown_pct']} session={row['session_filter']} vol={row['volatility_filter']} "
            f"trend={row['trend_strength_filter']} price_loc={row['price_location_filter']} "
            f"exit={row['micro_exit']} adx={row['adx_min']} cooldown={row['cooldown_bars']}"
        )

    def first_value(df: pl.DataFrame, col: str) -> str:
        if df.height == 0:
            return ""
        return str(df.select(col).item(0, 0))

    lines.append("")
    lines.append("TOP PATTERNS:")
    lines.append(
        f"- Best session_filter = {first_value(session_df, 'session_filter')}"
    )
    lines.append(
        f"- Best volatility_filter = {first_value(vol_df, 'volatility_filter')}"
    )
    lines.append(
        f"- Best trend_strength_filter = {first_value(trend_df, 'trend_strength_filter')}"
    )
    lines.append(
        f"- Best selected_side = {first_value(side_df, 'selected_side')}"
    )
    lines.append(
        f"- Best micro_exit = {first_value(micro_exit_df, 'micro_exit')}"
    )
    lines.append(
        f"- Best adx_min = {first_value(adx_df, 'adx_min')}"
    )

    if combo_df.height > 0:
        best_combo = combo_df.row(0, named=True)
        lines.append("")
        lines.append("BEST PARAMETER COMBINATION:")
        lines.append(
            f"- session={best_combo['session_filter']}, vol={best_combo['volatility_filter']}, "
            f"trend={best_combo['trend_strength_filter']}, price_loc={best_combo['price_location_filter']}, "
            f"side_policy={best_combo['side_policy']}, selected_side={best_combo['selected_side']}, "
            f"exit={best_combo['micro_exit']}, adx={best_combo['adx_min']}, "
            f"breakout_atr={best_combo['breakout_atr_min']}, body_ratio={best_combo['body_ratio_min']}, "
            f"max_ext={best_combo['max_extension_atr']}, cooldown={best_combo['cooldown_bars']}, "
            f"combo_jobs={best_combo['combo_jobs']}, best_job_id={best_combo['best_job_id']}"
        )

    lines.append("")
    lines.append("DECISION:")
    lines.append("- Regime/session filtering created real promotable jobs.")
    lines.append("- Next production research step should narrow the search space around the winning parameter clusters only.")
    lines.append("- Do not go back to broad entry-family branching at this stage.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize regime/session filter family results")
    parser.add_argument("--results-csv", required=True, dest="results_csv", help="Results CSV path")
    parser.add_argument("--summary-json", required=True, dest="summary_json", help="Summary JSON path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_csv_path = Path(args.results_csv)
    summary_json_path = Path(args.summary_json)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    raw_summary = read_summary_json(summary_json_path)
    df_all = add_band_columns(load_results_csv(results_csv_path))

    promoted = (
        df_all.filter(pl.col("promoted") == True)
        .sort(["score", "profit_factor", "total_return_pct"], descending=[True, True, True])
    )

    promoted.write_csv(outdir / OUT_PROMOTED_CSV)
    promoted.head(50).write_csv(outdir / OUT_TOP_PROMOTED_CSV)

    session_df = summarize_dimension(df_all, "session_filter")
    vol_df = summarize_dimension(df_all, "volatility_filter")
    trend_df = summarize_dimension(df_all, "trend_strength_filter")
    price_loc_df = summarize_dimension(df_all, "price_location_filter")
    side_policy_df = summarize_dimension(df_all, "side_policy")
    selected_side_df = summarize_dimension(df_all, "selected_side")
    micro_exit_df = summarize_dimension(df_all, "micro_exit")
    adx_df = summarize_dimension(df_all, "adx_min")
    breakout_atr_df = summarize_dimension(df_all, "breakout_atr_min")
    body_ratio_df = summarize_dimension(df_all, "body_ratio_min")
    max_extension_df = summarize_dimension(df_all, "max_extension_atr")
    cooldown_df = summarize_dimension(df_all, "cooldown_bars")
    rsi_long_band_df = summarize_dimension(df_all, "rsi_long_band")
    rsi_short_band_df = summarize_dimension(df_all, "rsi_short_band")
    combo_df = summarize_combinations(promoted)

    session_df.write_csv(outdir / OUT_SESSION_CSV)
    vol_df.write_csv(outdir / OUT_VOL_CSV)
    trend_df.write_csv(outdir / OUT_TREND_CSV)
    price_loc_df.write_csv(outdir / OUT_PRICE_LOC_CSV)
    side_policy_df.write_csv(outdir / OUT_SIDE_POLICY_CSV)
    selected_side_df.write_csv(outdir / OUT_SELECTED_SIDE_CSV)
    micro_exit_df.write_csv(outdir / OUT_MICRO_EXIT_CSV)
    adx_df.write_csv(outdir / OUT_ADX_CSV)
    breakout_atr_df.write_csv(outdir / OUT_BREAKOUT_ATR_CSV)
    body_ratio_df.write_csv(outdir / OUT_BODY_RATIO_CSV)
    max_extension_df.write_csv(outdir / OUT_MAX_EXTENSION_CSV)
    cooldown_df.write_csv(outdir / OUT_COOLDOWN_CSV)
    rsi_long_band_df.write_csv(outdir / OUT_RSI_LONG_BAND_CSV)
    rsi_short_band_df.write_csv(outdir / OUT_RSI_SHORT_BAND_CSV)
    combo_df.write_csv(outdir / OUT_COMBO_CSV)

    analysis_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "results_csv_path": str(results_csv_path),
        "summary_json_path": str(summary_json_path),
        "input_total_jobs": int(df_all.height),
        "input_promoted_jobs": int(promoted.height),
        "input_pass_jobs": int(df_all.filter(pl.col("trade_count") > 0).height),
        "input_no_trade_jobs": int(df_all.filter(pl.col("trade_count") <= 0).height),
        "top_promoted_job_id": str(promoted.select("job_id").item(0, 0)) if promoted.height > 0 else "",
        "top_session_filter": str(session_df.select("session_filter").item(0, 0)) if session_df.height > 0 else "",
        "top_volatility_filter": str(vol_df.select("volatility_filter").item(0, 0)) if vol_df.height > 0 else "",
        "top_trend_strength_filter": str(trend_df.select("trend_strength_filter").item(0, 0)) if trend_df.height > 0 else "",
        "top_selected_side": str(selected_side_df.select("selected_side").item(0, 0)) if selected_side_df.height > 0 else "",
        "top_micro_exit": str(micro_exit_df.select("micro_exit").item(0, 0)) if micro_exit_df.height > 0 else "",
        "top_adx_min": str(adx_df.select("adx_min").item(0, 0)) if adx_df.height > 0 else "",
        "top_combo_job_id": str(combo_df.select("best_job_id").item(0, 0)) if combo_df.height > 0 else "",
    }
    (outdir / OUT_SUMMARY_JSON).write_text(json.dumps(analysis_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    recommendation_text = build_recommendation(
        raw_summary=raw_summary,
        df_all=df_all,
        promoted=promoted,
        session_df=session_df,
        vol_df=vol_df,
        trend_df=trend_df,
        side_df=selected_side_df,
        micro_exit_df=micro_exit_df,
        adx_df=adx_df,
        combo_df=combo_df,
    )
    (outdir / OUT_RECOMMENDATION_TXT).write_text(recommendation_text, encoding="utf-8")

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] promoted_csv={outdir / OUT_PROMOTED_CSV}")
    print(f"[DONE] top_promoted_csv={outdir / OUT_TOP_PROMOTED_CSV}")
    print(f"[DONE] session_csv={outdir / OUT_SESSION_CSV}")
    print(f"[DONE] vol_csv={outdir / OUT_VOL_CSV}")
    print(f"[DONE] trend_csv={outdir / OUT_TREND_CSV}")
    print(f"[DONE] combo_csv={outdir / OUT_COMBO_CSV}")
    print(f"[DONE] analysis_summary_json={outdir / OUT_SUMMARY_JSON}")
    print(f"[DONE] recommendation_txt={outdir / OUT_RECOMMENDATION_TXT}")
    print(f"[DONE] input_total_jobs={analysis_summary['input_total_jobs']}")
    print(f"[DONE] input_promoted_jobs={analysis_summary['input_promoted_jobs']}")
    print(f"[DONE] top_promoted_job_id={analysis_summary['top_promoted_job_id']}")
    print("=" * 120)


if __name__ == "__main__":
    main()