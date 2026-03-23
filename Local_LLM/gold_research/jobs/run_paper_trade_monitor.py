# ============================================================
# ชื่อโค้ด: run_paper_trade_monitor.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\run_paper_trade_monitor.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\run_paper_trade_monitor.py --spec C:\Data\Bot\central_backtest_results\index\paper_trade_monitor_spec.json --signals C:\Data\Bot\central_backtest_results\paper_trade\signals.jsonl --trades C:\Data\Bot\central_backtest_results\paper_trade\trades.jsonl --out C:\Data\Bot\central_backtest_results\index\paper_trade_monitor_report.json
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0.1"


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ JSON | file={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input data | file={path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL parse error | file={path} | line={line_no} | error={exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be object | file={path} | line={line_no}")
                rows.append(row)
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]

    raise ValueError(f"รองรับเฉพาะ .jsonl หรือ .csv | file={path}")


def iso_year_week_from_dt(dt: datetime | None) -> tuple[int | None, int | None]:
    if dt is None:
        return (None, None)
    iso = dt.isocalendar()
    return (iso.year, iso.week)


def safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def normalize_trade_date(value: Any, fallback_dt: datetime | None = None) -> str:
    if value:
        return str(value)
    if fallback_dt is None:
        return ""
    return fallback_dt.date().isoformat()


# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------
def validate_required_fields(
    records: list[dict[str, Any]],
    required_fields: list[str],
    dataset_name: str,
) -> list[str]:
    errors: list[str] = []
    if not records:
        errors.append(f"{dataset_name}: ไม่มีข้อมูล")
        return errors

    first_missing_counter: Counter[str] = Counter()
    missing_examples: dict[str, int] = {}

    for idx, row in enumerate(records):
        missing = [field for field in required_fields if field not in row]
        for field in missing:
            first_missing_counter[field] += 1
            missing_examples.setdefault(field, idx)

    for field, count in sorted(first_missing_counter.items()):
        errors.append(
            f"{dataset_name}: ขาด field={field} ใน {count} rows | first_row_index={missing_examples[field]}"
        )

    return errors


# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
def filter_signals_by_candidate_identity(
    signals: list[dict[str, Any]],
    strategy_id: str,
    timeframe: str,
    side_policy: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in signals:
        if str(row.get("strategy_id", "")) != strategy_id:
            continue
        if str(row.get("timeframe", "")) != timeframe:
            continue

        side = str(row.get("side", "")).upper()
        if side_policy == "LONG_ONLY" and side not in {"BUY", "LONG"}:
            continue
        if side_policy == "SHORT_ONLY" and side not in {"SELL", "SHORT"}:
            continue

        filtered.append(row)
    return filtered


def filter_trades_by_signal_ids(
    trades: list[dict[str, Any]],
    signal_ids: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in trades:
        signal_id = str(row.get("signal_id", ""))
        if signal_id in signal_ids:
            filtered.append(row)
    return filtered


# ------------------------------------------------------------
# Regime logic
# ------------------------------------------------------------
@dataclass
class RegimeProfile:
    allow_trend_buckets: set[str]
    allow_volatility_buckets: set[str]
    allow_price_location_buckets: set[str]
    block_trend_buckets: set[str]
    block_price_location_buckets: set[str]


def build_regime_profile(spec: dict[str, Any]) -> RegimeProfile:
    profile = spec["allowed_regime_profile"]
    return RegimeProfile(
        allow_trend_buckets={str(x) for x in profile.get("allow_trend_buckets", [])},
        allow_volatility_buckets={str(x) for x in profile.get("allow_volatility_buckets", [])},
        allow_price_location_buckets={str(x) for x in profile.get("allow_price_location_buckets", [])},
        block_trend_buckets={str(x) for x in profile.get("block_trend_buckets", [])},
        block_price_location_buckets={str(x) for x in profile.get("block_price_location_buckets", [])},
    )


def is_allowed_regime(row: dict[str, Any], profile: RegimeProfile) -> bool:
    trend = str(row.get("trend_bucket", ""))
    vol = str(row.get("volatility_bucket", ""))
    price_loc = str(row.get("price_location_bucket", ""))

    if trend in profile.block_trend_buckets:
        return False
    if price_loc in profile.block_price_location_buckets:
        return False

    if profile.allow_trend_buckets and trend not in profile.allow_trend_buckets:
        return False
    if profile.allow_volatility_buckets and vol not in profile.allow_volatility_buckets:
        return False
    if profile.allow_price_location_buckets and price_loc not in profile.allow_price_location_buckets:
        return False

    return True


# ------------------------------------------------------------
# Check 1: signal_count_per_week
# ------------------------------------------------------------
def compute_signal_count_per_week(signals: list[dict[str, Any]]) -> dict[str, Any]:
    weekly_counts: Counter[tuple[int, int]] = Counter()

    for row in signals:
        if not parse_bool(row.get("approved")):
            continue
        dt = parse_dt(row.get("ts_utc"))
        year, week = iso_year_week_from_dt(dt)
        if year is None or week is None:
            continue
        weekly_counts[(year, week)] += 1

    weeks = []
    for (year, week), count in sorted(weekly_counts.items()):
        weeks.append(
            {
                "iso_year": year,
                "iso_week": week,
                "signal_count": count,
            }
        )

    alerts: list[str] = []
    prev_count: int | None = None
    for item in weeks:
        count = item["signal_count"]
        if count == 0:
            alerts.append(f"WARN: zero_signal_week | year={item['iso_year']} | week={item['iso_week']}")
        if prev_count is not None and prev_count > 0:
            drop_pct = safe_pct(prev_count - count, prev_count)
            spike_pct = safe_pct(count - prev_count, prev_count)
            if drop_pct >= 60.0:
                alerts.append(
                    f"WARN: sudden_drop_vs_previous_week | year={item['iso_year']} | week={item['iso_week']} | drop_pct={drop_pct:.2f}"
                )
            if spike_pct >= 150.0:
                alerts.append(
                    f"WARN: sudden_spike_vs_previous_week | year={item['iso_year']} | week={item['iso_week']} | spike_pct={spike_pct:.2f}"
                )
        prev_count = count

    return {
        "summary": {
            "weekly_rows": len(weeks),
            "total_approved_signals": sum(x["signal_count"] for x in weeks),
            "avg_signal_count_per_week": (
                sum(x["signal_count"] for x in weeks) / len(weeks) if weeks else 0.0
            ),
        },
        "weekly_breakdown": weeks,
        "alerts": alerts,
        "passed": not any(a.startswith("WARN: zero_signal_week") for a in alerts),
    }


# ------------------------------------------------------------
# Check 2: stop_loss_clustering
# ------------------------------------------------------------
def compute_stop_loss_clustering(trades: list[dict[str, Any]]) -> dict[str, Any]:
    stop_loss_trades = [row for row in trades if str(row.get("exit_reason", "")).lower() == "stop_loss"]

    total = len(stop_loss_trades)
    by_day: Counter[str] = Counter()
    by_hour: Counter[str] = Counter()
    by_regime: Counter[str] = Counter()

    for row in stop_loss_trades:
        exit_dt = parse_dt(row.get("exit_time_utc"))
        trade_date = normalize_trade_date(row.get("trade_date"), exit_dt)
        hour_bucket = str(row.get("hour_bucket", ""))
        trend = str(row.get("trend_bucket", ""))
        vol = str(row.get("volatility_bucket", ""))
        price_loc = str(row.get("price_location_bucket", ""))
        regime_key = f"{trend}|{vol}|{price_loc}"

        by_day[trade_date] += 1
        by_hour[hour_bucket] += 1
        by_regime[regime_key] += 1

    max_day_key, max_day_count = ("", 0)
    if by_day:
        max_day_key, max_day_count = by_day.most_common(1)[0]

    max_hour_key, max_hour_count = ("", 0)
    if by_hour:
        max_hour_key, max_hour_count = by_hour.most_common(1)[0]

    max_regime_key, max_regime_count = ("", 0)
    if by_regime:
        max_regime_key, max_regime_count = by_regime.most_common(1)[0]

    max_day_share = safe_pct(max_day_count, total)
    max_hour_share = safe_pct(max_hour_count, total)
    max_regime_share = safe_pct(max_regime_count, total)

    alerts: list[str] = []
    if max_day_share > 40.0:
        alerts.append(f"WARN: max_stop_loss_share_same_day_pct={max_day_share:.2f} > 40.0 | day={max_day_key}")
    if max_hour_share > 25.0:
        alerts.append(f"WARN: max_stop_loss_share_same_hour_pct={max_hour_share:.2f} > 25.0 | hour_bucket={max_hour_key}")
    if max_regime_share > 60.0:
        alerts.append(f"WARN: max_stop_loss_share_same_regime_pct={max_regime_share:.2f} > 60.0 | regime={max_regime_key}")

    return {
        "summary": {
            "total_stop_loss_trades": total,
            "max_stop_loss_share_same_day_pct": round(max_day_share, 6),
            "max_stop_loss_share_same_hour_pct": round(max_hour_share, 6),
            "max_stop_loss_share_same_regime_pct": round(max_regime_share, 6),
            "max_day_key": max_day_key,
            "max_hour_key": max_hour_key,
            "max_regime_key": max_regime_key,
        },
        "alerts": alerts,
        "passed": len(alerts) == 0,
    }


# ------------------------------------------------------------
# Check 3: win_loss_distribution
# ------------------------------------------------------------
def compute_consecutive_losses(trades_sorted: list[dict[str, Any]]) -> int:
    max_streak = 0
    current = 0

    for row in trades_sorted:
        pnl = parse_float(row.get("pnl"))
        if pnl < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak


def compute_win_loss_distribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    weekly_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    all_trades_sorted = sorted(
        trades,
        key=lambda row: parse_dt(row.get("exit_time_utc")) or datetime.min.replace(tzinfo=timezone.utc),
    )

    for row in trades:
        exit_dt = parse_dt(row.get("exit_time_utc")) or parse_dt(row.get("entry_time_utc"))
        year, week = iso_year_week_from_dt(exit_dt)
        if year is None or week is None:
            continue
        weekly_groups[(year, week)].append(row)

    weekly_stats = []
    alerts: list[str] = []

    for (year, week), rows in sorted(weekly_groups.items()):
        wins = [parse_float(x.get("pnl")) for x in rows if parse_float(x.get("pnl")) > 0]
        losses = [parse_float(x.get("pnl")) for x in rows if parse_float(x.get("pnl")) < 0]
        total = len(rows)
        win_rate = safe_pct(len(wins), total)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss_abs = abs(sum(losses) / len(losses)) if losses else 0.0
        payoff_ratio = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else (math.inf if avg_win > 0 else 0.0)

        weekly_stats.append(
            {
                "iso_year": year,
                "iso_week": week,
                "trades": total,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": round(win_rate, 6),
                "avg_win": round(avg_win, 6),
                "avg_loss_abs": round(avg_loss_abs, 6),
                "payoff_ratio": round(payoff_ratio, 6) if math.isfinite(payoff_ratio) else "inf",
            }
        )

        if win_rate < 30.0:
            alerts.append(
                f"WARN: min_weekly_win_rate_pct breach | year={year} | week={week} | win_rate_pct={win_rate:.2f}"
            )
        if math.isfinite(payoff_ratio) and payoff_ratio < 1.20:
            alerts.append(
                f"WARN: min_payoff_ratio breach | year={year} | week={week} | payoff_ratio={payoff_ratio:.4f}"
            )

    max_consecutive_losses = compute_consecutive_losses(all_trades_sorted)
    if max_consecutive_losses > 8:
        alerts.append(f"WARN: max_consecutive_losses breach | max_consecutive_losses={max_consecutive_losses}")

    total_trades = len(trades)
    total_wins = sum(1 for row in trades if parse_float(row.get("pnl")) > 0)
    total_losses = sum(1 for row in trades if parse_float(row.get("pnl")) < 0)
    overall_win_rate = safe_pct(total_wins, total_trades)

    return {
        "summary": {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_win_rate_pct": round(overall_win_rate, 6),
            "max_consecutive_losses": max_consecutive_losses,
        },
        "weekly_breakdown": weekly_stats,
        "alerts": alerts,
        "passed": len(alerts) == 0,
    }


# ------------------------------------------------------------
# Check 4: pnl_concentration_by_day
# ------------------------------------------------------------
def compute_pnl_concentration_by_day(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_by_day: dict[str, float] = defaultdict(float)
    count_by_day: Counter[str] = Counter()

    for row in trades:
        exit_dt = parse_dt(row.get("exit_time_utc"))
        trade_date = normalize_trade_date(row.get("trade_date"), exit_dt)
        pnl = parse_float(row.get("pnl"))
        pnl_by_day[trade_date] += pnl
        count_by_day[trade_date] += 1

    daily_rows = []
    for trade_date in sorted(pnl_by_day.keys()):
        daily_rows.append(
            {
                "trade_date": trade_date,
                "daily_pnl_sum": round(pnl_by_day[trade_date], 6),
                "daily_trade_count": count_by_day[trade_date],
            }
        )

    positive_pnl_days = sum(1 for x in daily_rows if x["daily_pnl_sum"] > 0)
    total_days = len(daily_rows)
    positive_day_share = safe_pct(positive_pnl_days, total_days)

    positive_daily_pnls = [max(0.0, x["daily_pnl_sum"]) for x in daily_rows]
    total_positive_pnl = sum(positive_daily_pnls)

    sorted_positive = sorted(positive_daily_pnls, reverse=True)
    top_1 = sorted_positive[0] if len(sorted_positive) >= 1 else 0.0
    top_3 = sum(sorted_positive[:3]) if sorted_positive else 0.0

    max_single_day_share = safe_pct(top_1, total_positive_pnl)
    max_top_3_days_share = safe_pct(top_3, total_positive_pnl)

    alerts: list[str] = []
    if max_single_day_share > 50.0:
        alerts.append(f"WARN: max_single_day_pnl_share_pct={max_single_day_share:.2f} > 50.0")
    if max_top_3_days_share > 80.0:
        alerts.append(f"WARN: max_top_3_days_pnl_share_pct={max_top_3_days_share:.2f} > 80.0")
    if positive_day_share < 35.0:
        alerts.append(f"WARN: min_positive_day_share_pct breach | positive_day_share_pct={positive_day_share:.2f}")

    return {
        "summary": {
            "trading_days": total_days,
            "positive_pnl_days": positive_pnl_days,
            "positive_day_share_pct": round(positive_day_share, 6),
            "max_single_day_pnl_share_pct": round(max_single_day_share, 6),
            "max_top_3_days_pnl_share_pct": round(max_top_3_days_share, 6),
        },
        "daily_breakdown": daily_rows,
        "alerts": alerts,
        "passed": len(alerts) == 0,
    }


# ------------------------------------------------------------
# Check 5: regime_match_rate
# ------------------------------------------------------------
def compute_regime_match_rate(signals: list[dict[str, Any]], profile: RegimeProfile) -> dict[str, Any]:
    approved_signals = [row for row in signals if parse_bool(row.get("approved"))]

    allowed_hits = 0
    blocked_hits = 0
    mismatches = []

    for row in approved_signals:
        signal_id = str(row.get("signal_id", ""))
        allowed = is_allowed_regime(row, profile)
        if allowed:
            allowed_hits += 1
        else:
            blocked_hits += 1
            mismatches.append(
                {
                    "signal_id": signal_id,
                    "trend_bucket": row.get("trend_bucket"),
                    "volatility_bucket": row.get("volatility_bucket"),
                    "price_location_bucket": row.get("price_location_bucket"),
                }
            )

    total = len(approved_signals)
    regime_match_rate_pct = safe_pct(allowed_hits, total)

    alerts: list[str] = []
    if regime_match_rate_pct < 95.0:
        alerts.append(f"WARN: min_regime_match_rate_pct breach | regime_match_rate_pct={regime_match_rate_pct:.2f}")
    if blocked_hits > 0:
        alerts.append(f"WARN: max_blocked_regime_hits breach | blocked_regime_hits={blocked_hits}")

    return {
        "summary": {
            "approved_signals": total,
            "allowed_regime_hits": allowed_hits,
            "blocked_regime_hits": blocked_hits,
            "regime_match_rate_pct": round(regime_match_rate_pct, 6),
        },
        "mismatch_examples": mismatches[:20],
        "alerts": alerts,
        "passed": len(alerts) == 0,
    }


# ------------------------------------------------------------
# Observation window
# ------------------------------------------------------------
def compute_observation_window(signals: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps: list[datetime] = []

    for row in signals:
        dt = parse_dt(row.get("ts_utc"))
        if dt is not None:
            timestamps.append(dt)

    for row in trades:
        dt = parse_dt(row.get("entry_time_utc")) or parse_dt(row.get("exit_time_utc"))
        if dt is not None:
            timestamps.append(dt)

    if not timestamps:
        return {
            "start_utc": None,
            "end_utc": None,
            "observed_days": 0.0,
            "completed_weeks": 0,
        }

    start_dt = min(timestamps)
    end_dt = max(timestamps)
    observed_days = (end_dt - start_dt).total_seconds() / 86400.0
    completed_weeks = int(observed_days // 7)

    return {
        "start_utc": start_dt,
        "end_utc": end_dt,
        "observed_days": round(observed_days, 6),
        "completed_weeks": completed_weeks,
    }


# ------------------------------------------------------------
# Final gate
# ------------------------------------------------------------
def compute_overall_status(
    spec: dict[str, Any],
    observation_window: dict[str, Any],
    check_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate = spec["promotion_gate"]
    minimum_expectations = gate.get("minimum_expectations", {})

    observed_days = float(observation_window.get("observed_days") or 0.0)
    completed_weeks = int(observation_window.get("completed_weeks") or 0)

    regime_blocked = int(check_results["regime_match_rate"]["summary"]["blocked_regime_hits"])
    regime_match_rate_pct = float(check_results["regime_match_rate"]["summary"]["regime_match_rate_pct"])

    weekly_breakdown = check_results["signal_count_per_week"].get("weekly_breakdown", [])
    zero_signal_weeks = sum(1 for x in weekly_breakdown if int(x["signal_count"]) == 0)

    pnl_daily = check_results["pnl_concentration_by_day"].get("daily_breakdown", [])
    positive_days = sum(1 for x in pnl_daily if float(x["daily_pnl_sum"]) > 0)

    min_live_days_required = float(minimum_expectations.get("min_live_observation_days", 10))
    max_blocked_hits_allowed = int(minimum_expectations.get("max_blocked_regime_hits", 0))
    min_regime_match_rate_pct = float(minimum_expectations.get("min_regime_match_rate_pct", 95.0))
    min_positive_weeks = int(minimum_expectations.get("min_positive_weeks", 1))
    max_zero_signal_weeks = int(minimum_expectations.get("max_zero_signal_weeks", 1))

    positive_weeks = 0
    win_loss_weekly = check_results["win_loss_distribution"].get("weekly_breakdown", [])
    for item in win_loss_weekly:
        wins = parse_int(item.get("wins"))
        losses = parse_int(item.get("losses"))
        avg_win = parse_float(item.get("avg_win"))
        avg_loss_abs = parse_float(item.get("avg_loss_abs"))
        if wins > losses or (wins > 0 and avg_win > avg_loss_abs):
            positive_weeks += 1

    check_passes = {name: bool(result.get("passed")) for name, result in check_results.items()}

    gate_conditions = {
        "min_live_observation_days_passed": observed_days >= min_live_days_required,
        "min_completed_weeks_observed": completed_weeks >= 2,
        "max_blocked_regime_hits_passed": regime_blocked <= max_blocked_hits_allowed,
        "min_regime_match_rate_pct_passed": regime_match_rate_pct >= min_regime_match_rate_pct,
        "min_positive_weeks_passed": positive_weeks >= min_positive_weeks,
        "max_zero_signal_weeks_passed": zero_signal_weeks <= max_zero_signal_weeks,
        "all_checks_passed": all(check_passes.values()),
    }

    overall_pass = all(gate_conditions.values())

    return {
        "check_passes": check_passes,
        "gate_conditions": gate_conditions,
        "metrics_for_gate": {
            "observed_days": observed_days,
            "completed_weeks": completed_weeks,
            "blocked_regime_hits": regime_blocked,
            "regime_match_rate_pct": regime_match_rate_pct,
            "positive_weeks": positive_weeks,
            "zero_signal_weeks": zero_signal_weeks,
            "positive_days": positive_days,
        },
        "final_decision": "PASS" if overall_pass else "HOLD",
    }


# ------------------------------------------------------------
# Main report builder
# ------------------------------------------------------------
def build_report(
    spec: dict[str, Any],
    signals: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_identity = spec["candidate_identity"]
    strategy_id = str(candidate_identity["strategy_id"])
    timeframe = str(candidate_identity["timeframe"])
    side_policy = str(candidate_identity["side_policy"])

    regime_profile = build_regime_profile(spec)

    filtered_signals = filter_signals_by_candidate_identity(
        signals=signals,
        strategy_id=strategy_id,
        timeframe=timeframe,
        side_policy=side_policy,
    )

    signal_ids = {
        str(row.get("signal_id", ""))
        for row in filtered_signals
        if row.get("signal_id") is not None
    }
    filtered_trades = filter_trades_by_signal_ids(trades=trades, signal_ids=signal_ids)

    observation_window = compute_observation_window(filtered_signals, filtered_trades)

    check_results = {
        "signal_count_per_week": compute_signal_count_per_week(filtered_signals),
        "stop_loss_clustering": compute_stop_loss_clustering(filtered_trades),
        "win_loss_distribution": compute_win_loss_distribution(filtered_trades),
        "pnl_concentration_by_day": compute_pnl_concentration_by_day(filtered_trades),
        "regime_match_rate": compute_regime_match_rate(filtered_signals, regime_profile),
    }

    overall_status = compute_overall_status(
        spec=spec,
        observation_window=observation_window,
        check_results=check_results,
    )

    report = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_spec": spec.get("source_manifest"),
        "monitor_spec_version": spec.get("version"),
        "candidate_identity": candidate_identity,
        "input_summary": {
            "raw_signal_rows": len(signals),
            "raw_trade_rows": len(trades),
            "filtered_signal_rows": len(filtered_signals),
            "filtered_trade_rows": len(filtered_trades),
        },
        "observation_window": observation_window,
        "check_results": check_results,
        "overall_status": overall_status,
        "notes": [
            "ผลนี้ใช้ตัดสิน paper-trade validation เท่านั้น",
            "ถ้า final_decision = HOLD ห้าม promote ขึ้น production auto-trade",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-trade monitor from monitor spec.")
    parser.add_argument("--spec", required=True, help="Path to paper_trade_monitor_spec.json")
    parser.add_argument("--signals", required=True, help="Path to signals data (.jsonl or .csv)")
    parser.add_argument("--trades", required=True, help="Path to trades data (.jsonl or .csv)")
    parser.add_argument("--out", required=True, help="Path to output report json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec_path = Path(args.spec)
    signals_path = Path(args.signals)
    trades_path = Path(args.trades)
    out_path = Path(args.out)

    print("=" * 120)
    print(f"[INFO] run_paper_trade_monitor.py version={VERSION}")
    print(f"[INFO] spec={spec_path}")
    print(f"[INFO] signals={signals_path}")
    print(f"[INFO] trades={trades_path}")
    print(f"[INFO] out={out_path}")
    print("=" * 120)

    spec = load_json(spec_path)
    signals = load_records(signals_path)
    trades = load_records(trades_path)

    required_signal_fields = spec["runtime_data_contract"]["required_signal_fields"]
    required_trade_fields = spec["runtime_data_contract"]["required_trade_fields"]

    validation_errors = []
    validation_errors.extend(validate_required_fields(signals, required_signal_fields, "signals"))
    validation_errors.extend(validate_required_fields(trades, required_trade_fields, "trades"))

    if validation_errors:
        for error in validation_errors:
            print(f"[ERROR] {error}")
        raise ValueError("input data contract validation failed")

    report = build_report(spec=spec, signals=signals, trades=trades)
    save_json(out_path, report)

    final_decision = report["overall_status"]["final_decision"]
    filtered_signal_rows = report["input_summary"]["filtered_signal_rows"]
    filtered_trade_rows = report["input_summary"]["filtered_trade_rows"]

    print(f"[DONE] final_decision={final_decision}")
    print(f"[DONE] filtered_signal_rows={filtered_signal_rows}")
    print(f"[DONE] filtered_trade_rows={filtered_trade_rows}")
    print(f"[DONE] report={out_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()