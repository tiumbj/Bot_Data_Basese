from __future__ import annotations

import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "v1.0.6"

TRADES_JSONL = Path(r"C:\Data\Bot\central_backtest_results\paper_trade_historical\trades.jsonl")


@dataclass(frozen=True)
class TradeRow:
    trade_id: str
    signal_id: str
    entry_time_utc: str
    exit_time_utc: str
    entry_price: float
    exit_price: float
    side: str
    exit_reason: str
    trend_bucket: str
    volatility_bucket: str
    price_location_bucket: str
    pnl: float


@dataclass
class OhlcBar:
    ts_utc: str
    open: float
    high: float
    low: float
    close: float
    body_ratio: float = 0.0
    bullish_close: bool = False
    bearish_close: bool = False
    lower_low: bool = False
    higher_high: bool = False
    atr14: float = 0.0
    adx14: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    ema5: float = 0.0
    ema10: float = 0.0


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty UTC timestamp")
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def first_present(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_side(raw_side: Any, entry_price: float, exit_price: float, exit_reason: str) -> str:
    text = str(raw_side or "").strip().upper()
    if text in {"BUY", "LONG"}:
        return "BUY"
    if text in {"SELL", "SHORT"}:
        return "SELL"

    reason = (exit_reason or "").lower()
    if "sell" in reason or "short" in reason:
        return "SELL"
    if "buy" in reason or "long" in reason:
        return "BUY"

    if exit_price >= entry_price:
        return "BUY"
    return "SELL"


def load_trades() -> list[TradeRow]:
    if not TRADES_JSONL.exists():
        raise FileNotFoundError(f"Missing trades file: {TRADES_JSONL}")

    rows: list[TradeRow] = []
    with TRADES_JSONL.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            entry_price = to_float(first_present(raw, ["entry_price", "open_price", "price_open"]))
            exit_price = to_float(first_present(raw, ["exit_price", "close_price", "price_close"]))
            exit_reason = str(first_present(raw, ["exit_reason", "close_reason"], ""))

            trade = TradeRow(
                trade_id=str(first_present(raw, ["trade_id", "ticket", "id"], f"trade_{line_no}")),
                signal_id=str(first_present(raw, ["signal_id", "setup_id", "candidate_id"], f"signal_{line_no}")),
                entry_time_utc=str(first_present(raw, ["entry_time_utc", "entry_time", "ts_utc"])),
                exit_time_utc=str(first_present(raw, ["exit_time_utc", "exit_time", "close_time_utc"])),
                entry_price=entry_price,
                exit_price=exit_price,
                side=normalize_side(
                    first_present(raw, ["side", "direction", "trade_side", "position_side"], ""),
                    entry_price,
                    exit_price,
                    exit_reason,
                ),
                exit_reason=exit_reason,
                trend_bucket=str(first_present(raw, ["trend_bucket"], "")),
                volatility_bucket=str(first_present(raw, ["volatility_bucket"], "")),
                price_location_bucket=str(first_present(raw, ["price_location_bucket"], "")),
                pnl=to_float(first_present(raw, ["pnl", "pnl_sum", "profit"], 0.0)),
            )
            rows.append(trade)

    if not rows:
        raise ValueError(f"No rows loaded from trades file: {TRADES_JSONL}")
    return rows


def ema(values: list[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("EMA period must be > 0")
    out: list[float] = []
    alpha = 2.0 / (period + 1.0)
    prev: float | None = None
    for value in values:
        if prev is None:
            prev = value
        else:
            prev = (alpha * value) + ((1.0 - alpha) * prev)
        out.append(prev)
    return out


def rma(values: list[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("RMA period must be > 0")
    out: list[float] = []
    prev: float | None = None
    alpha = 1.0 / period
    for value in values:
        if prev is None:
            prev = value
        else:
            prev = (alpha * value) + ((1.0 - alpha) * prev)
        out.append(prev)
    return out


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: list[float] = []
    running_sum = 0.0
    buffer: list[float] = []
    for value in values:
        buffer.append(value)
        running_sum += value
        if len(buffer) > window:
            running_sum -= buffer.pop(0)
        out.append(running_sum / len(buffer))
    return out


def load_ohlc_from_csv(ohlc_csv: Path) -> tuple[list[datetime], list[OhlcBar]]:
    if not ohlc_csv.exists():
        raise FileNotFoundError(f"Missing OHLC file: {ohlc_csv}")

    ts_index: list[datetime] = []
    bars: list[OhlcBar] = []

    with ohlc_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ts_raw = first_present(row, ["ts_utc", "timestamp", "time", "datetime", "Date", "date"], "")
            if not ts_raw:
                continue

            ts_text = str(ts_raw).strip().replace(" ", "T")
            if "+" not in ts_text and not ts_text.endswith("Z"):
                ts_text = ts_text + "+00:00"
            dt = parse_utc(ts_text)

            bar = OhlcBar(
                ts_utc=dt.isoformat().replace("+00:00", "Z"),
                open=to_float(first_present(row, ["open", "Open", "o"], 0.0)),
                high=to_float(first_present(row, ["high", "High", "h"], 0.0)),
                low=to_float(first_present(row, ["low", "Low", "l"], 0.0)),
                close=to_float(first_present(row, ["close", "Close", "c"], 0.0)),
            )
            ts_index.append(dt)
            bars.append(bar)

    if not bars:
        raise ValueError(f"No OHLC rows loaded from: {ohlc_csv}")

    enrich_price_features(bars)
    return ts_index, bars


def enrich_price_features(bars: list[OhlcBar]) -> None:
    closes = [bar.close for bar in bars]
    ema5_values = ema(closes, 5)
    ema10_values = ema(closes, 10)

    tr_values: list[float] = []
    plus_dm_values: list[float] = []
    minus_dm_values: list[float] = []

    prev_bar: OhlcBar | None = None
    for bar in bars:
        candle_range = max(bar.high - bar.low, 1e-9)
        body = abs(bar.close - bar.open)
        bar.body_ratio = body / candle_range
        bar.bullish_close = bar.close > bar.open
        bar.bearish_close = bar.close < bar.open

        if prev_bar is None:
            bar.lower_low = False
            bar.higher_high = False
            tr_values.append(bar.high - bar.low)
            plus_dm_values.append(0.0)
            minus_dm_values.append(0.0)
        else:
            bar.lower_low = bar.low < prev_bar.low
            bar.higher_high = bar.high > prev_bar.high

            true_range = max(
                bar.high - bar.low,
                abs(bar.high - prev_bar.close),
                abs(bar.low - prev_bar.close),
            )
            up_move = bar.high - prev_bar.high
            down_move = prev_bar.low - bar.low

            plus_dm = up_move if (up_move > down_move and up_move > 0.0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0.0) else 0.0

            tr_values.append(true_range)
            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)

        prev_bar = bar

    atr14_values = rma(tr_values, 14)
    plus_dm_rma = rma(plus_dm_values, 14)
    minus_dm_rma = rma(minus_dm_values, 14)

    plus_di_values: list[float] = []
    minus_di_values: list[float] = []
    dx_values: list[float] = []

    for i in range(len(bars)):
        atr = atr14_values[i]
        if atr <= 0.0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100.0 * (plus_dm_rma[i] / atr)
            minus_di = 100.0 * (minus_dm_rma[i] / atr)

        denom = plus_di + minus_di
        dx = 0.0 if denom <= 0.0 else 100.0 * abs(plus_di - minus_di) / denom

        plus_di_values.append(plus_di)
        minus_di_values.append(minus_di)
        dx_values.append(dx)

    adx14_values = rma(dx_values, 14)

    for i, bar in enumerate(bars):
        bar.ema5 = ema5_values[i]
        bar.ema10 = ema10_values[i]
        bar.atr14 = atr14_values[i]
        bar.di_plus = plus_di_values[i]
        bar.di_minus = minus_di_values[i]
        bar.adx14 = adx14_values[i]


def find_first_bar_at_or_after(target_ts: datetime, ts_index: list[datetime], bars: list[OhlcBar]) -> int | None:
    pos = bisect_left(ts_index, target_ts)
    if pos >= len(bars):
        return None
    return pos


def normalize_regime_filter_id(regime_filter_id: str) -> str:
    mapping = {
        "none": "none",
        "regime_filter_off": "none",
        "light": "light",
        "regime_filter_trend_or_neutral": "light",
        "strict": "strict",
        "regime_filter_trend_only": "strict",
    }
    normalized = mapping.get(str(regime_filter_id).strip())
    if normalized is None:
        raise ValueError(f"Unknown regime_filter_id: {regime_filter_id}")
    return normalized


def normalize_micro_exit_id(micro_exit_id: str) -> str:
    mapping = {
        "none": "none",
        "conditional_max_hold_4h": "conditional_max_hold_4h",
        "conditional_max_hold_6h": "conditional_max_hold_6h",
        "micro_exit_v2_fast_invalidation": "micro_exit_v2_fast_invalidation",
        "micro_exit_v2_momentum_fade": "micro_exit_v2_momentum_fade",
        "micro_exit_v2_structure_trail": "conditional_max_hold_6h",
    }
    normalized = mapping.get(str(micro_exit_id).strip())
    if normalized is None:
        raise NotImplementedError(f"Micro exit not implemented in v1.0.6: {micro_exit_id}")
    return normalized


def normalize_cooldown_id(cooldown_id: Any) -> str:
    if isinstance(cooldown_id, int):
        mapping = {
            0: "none",
            2: "cooldown_2L_skip1",
            3: "cooldown_3L_skip1",
            6: "cooldown_3L_skip1",
        }
        normalized = mapping.get(cooldown_id)
        if normalized is None:
            raise NotImplementedError(f"Cooldown not implemented in v1.0.6: {cooldown_id}")
        return normalized

    text = str(cooldown_id).strip()
    mapping = {
        "none": "none",
        "0": "none",
        "cooldown_0": "none",
        "cooldown_2L_skip1": "cooldown_2L_skip1",
        "2": "cooldown_2L_skip1",
        "cooldown_2": "cooldown_2L_skip1",
        "cooldown_3L_skip1": "cooldown_3L_skip1",
        "3": "cooldown_3L_skip1",
        "cooldown_3": "cooldown_3L_skip1",
        "6": "cooldown_3L_skip1",
        "cooldown_6": "cooldown_3L_skip1",
    }
    normalized = mapping.get(text)
    if normalized is None:
        raise NotImplementedError(f"Cooldown not implemented in v1.0.6: {cooldown_id}")
    return normalized


def normalize_side_policy_id(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "": "both",
        "both": "both",
        "both_sides": "both",
        "long_only": "long_only",
        "buy_only": "long_only",
        "short_only": "short_only",
        "sell_only": "short_only",
    }
    normalized = mapping.get(text)
    if normalized is None:
        raise NotImplementedError(f"side_policy not implemented in v1.0.6: {value}")
    return normalized


def normalize_volatility_filter_id(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "": "any_vol",
        "any_vol": "any_vol",
        "off": "any_vol",
        "none": "any_vol",
        "low_vol_only": "low_vol_only",
        "mid_vol_only": "mid_vol_only",
        "high_vol_only": "high_vol_only",
        "mid_high_vol": "mid_high_vol",
    }
    normalized = mapping.get(text)
    if normalized is None:
        raise NotImplementedError(f"volatility_filter not implemented in v1.0.6: {value}")
    return normalized


def normalize_trend_strength_filter_id(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "": "any_trend",
        "off": "any_trend",
        "none": "any_trend",
        "any_trend": "any_trend",
        "any_trend_strength": "any_trend",
        "mid_trend_plus": "mid_trend_plus",
        "strong_trend_only": "strong_trend_only",
    }
    normalized = mapping.get(text)
    if normalized is None:
        raise NotImplementedError(f"trend_strength_filter not implemented in v1.0.6: {value}")
    return normalized


def read_job_config_id(job: dict[str, Any], field_name: str, nested_key: str, default: str = "") -> str:
    raw = job.get(field_name, default)

    if isinstance(raw, dict):
        value = raw.get(nested_key, default)
        return str(value).strip()

    if raw is None:
        return str(default).strip()

    return str(raw).strip()


def apply_regime_filter(row: TradeRow, regime_filter_id: str) -> tuple[bool, str]:
    normalized = normalize_regime_filter_id(regime_filter_id)

    if normalized == "none":
        return False, ""

    if normalized == "light":
        if row.price_location_bucket == "INSIDE_EMA_STACK":
            return True, "INSIDE_EMA_STACK"
        return False, ""

    if normalized == "strict":
        if row.price_location_bucket == "INSIDE_EMA_STACK":
            return True, "INSIDE_EMA_STACK"
        if (
            row.trend_bucket == "BULL_TREND"
            and row.volatility_bucket == "LOW_VOL"
            and row.price_location_bucket == "ABOVE_EMA_STACK"
        ):
            return True, "BULL_TREND_LOW_VOL_ABOVE_EMA_STACK"
        return False, ""

    raise ValueError(f"Unknown regime_filter_id: {regime_filter_id}")


def apply_side_policy_filter(row: TradeRow, side_policy_id: str) -> tuple[bool, str]:
    normalized = normalize_side_policy_id(side_policy_id)

    if normalized == "both":
        return False, ""
    if normalized == "long_only":
        return (row.side != "BUY"), "SIDE_POLICY_LONG_ONLY"
    if normalized == "short_only":
        return (row.side != "SELL"), "SIDE_POLICY_SHORT_ONLY"

    raise ValueError(f"Unknown side_policy_id: {side_policy_id}")


def apply_volatility_filter(row: TradeRow, volatility_filter_id: str) -> tuple[bool, str]:
    normalized = normalize_volatility_filter_id(volatility_filter_id)
    bucket = str(row.volatility_bucket or "").strip().upper()

    if normalized == "any_vol":
        return False, ""

    if normalized == "low_vol_only":
        return (bucket != "LOW_VOL"), "VOLATILITY_FILTER_LOW_ONLY"

    if normalized == "mid_vol_only":
        return (bucket != "MID_VOL"), "VOLATILITY_FILTER_MID_ONLY"

    if normalized == "high_vol_only":
        return (bucket != "HIGH_VOL"), "VOLATILITY_FILTER_HIGH_ONLY"

    if normalized == "mid_high_vol":
        return (bucket not in {"MID_VOL", "HIGH_VOL"}), "VOLATILITY_FILTER_MID_HIGH_ONLY"

    raise ValueError(f"Unknown volatility_filter_id: {volatility_filter_id}")


def apply_trend_strength_filter(row: TradeRow, trend_strength_filter_id: str) -> tuple[bool, str]:
    normalized = normalize_trend_strength_filter_id(trend_strength_filter_id)
    bucket = str(row.trend_bucket or "").strip().upper()

    if normalized == "any_trend":
        return False, ""

    if normalized == "mid_trend_plus":
        allowed = {
            "MID_TREND",
            "MID_TREND_PLUS",
            "STRONG_TREND",
            "BULL_TREND",
            "BEAR_TREND",
        }
        return (bucket not in allowed), "TREND_STRENGTH_FILTER_MID_PLUS"

    if normalized == "strong_trend_only":
        allowed = {
            "STRONG_TREND",
            "BULL_TREND",
            "BEAR_TREND",
        }
        return (bucket not in allowed), "TREND_STRENGTH_FILTER_STRONG_ONLY"

    raise ValueError(f"Unknown trend_strength_filter_id: {trend_strength_filter_id}")


def is_fast_invalidation_exit(side: str, prev_bar: OhlcBar, cur_bar: OhlcBar) -> bool:
    if side == "BUY":
        return (cur_bar.close < prev_bar.low) or (cur_bar.bearish_close and cur_bar.lower_low)
    if side == "SELL":
        return (cur_bar.close > prev_bar.high) or (cur_bar.bullish_close and cur_bar.higher_high)
    raise ValueError(f"Unknown side: {side}")


def is_momentum_fade_exit(
    side: str,
    prev_bar: OhlcBar,
    cur_bar: OhlcBar,
    atr_roll5: list[float],
    index: int,
) -> bool:
    adx_falling = cur_bar.adx14 < prev_bar.adx14
    stall = cur_bar.atr14 < atr_roll5[index]
    weak_body = cur_bar.body_ratio < 0.25

    if side == "BUY":
        weakening = cur_bar.di_plus < prev_bar.di_plus
    elif side == "SELL":
        weakening = cur_bar.di_minus < prev_bar.di_minus
    else:
        raise ValueError(f"Unknown side: {side}")

    return adx_falling and stall and weak_body and weakening


def vectorized_fast_invalidation_exit_slice(
    side: str,
    closes: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    bullish_closes: np.ndarray,
    lower_lows: np.ndarray,
    bullish_mask: np.ndarray,
) -> np.ndarray:
    if side == "BUY":
        invalidation_a = closes < np.roll(lows, 1)
        invalidation_a[0] = False
        invalidation_b = bullish_closes & lower_lows
        return invalidation_a | invalidation_b
    return np.zeros(len(closes), dtype=bool)


def vectorized_momentum_fade_exit_slice(
    side: str,
    adx14: np.ndarray,
    atr14: np.ndarray,
    atr_roll5: np.ndarray,
    body_ratio: np.ndarray,
    di_plus: np.ndarray,
    di_minus: np.ndarray,
) -> np.ndarray:
    adx_falling = adx14 < np.roll(adx14, 1)
    adx_falling[0] = False
    stall = atr14 < atr_roll5
    weak_body = body_ratio < 0.25
    if side == "BUY":
        weakening = di_plus < np.roll(di_plus, 1)
    else:
        weakening = di_minus < np.roll(di_minus, 1)
    weakening[0] = False
    return adx_falling & stall & weak_body & weakening


def build_effective_trade(
    row: TradeRow,
    normalized_micro_exit_id: str,
    ts_index: list[datetime],
    bars: list[OhlcBar],
    atr_roll5: list[float] | None = None,
) -> dict[str, Any]:
    out = {
        "trade_id": row.trade_id,
        "signal_id": row.signal_id,
        "entry_time_utc": row.entry_time_utc,
        "effective_exit_time_utc": row.exit_time_utc,
        "entry_price": row.entry_price,
        "effective_exit_price": row.exit_price,
        "side": row.side,
        "original_exit_reason": row.exit_reason,
        "effective_exit_reason": row.exit_reason,
        "trend_bucket": row.trend_bucket,
        "volatility_bucket": row.volatility_bucket,
        "price_location_bucket": row.price_location_bucket,
        "original_pnl": row.pnl,
        "effective_pnl": row.pnl,
        "conditional_time_stop_applied": False,
        "threshold_bar_time_utc": None,
        "threshold_bar_close": None,
        "threshold_bar_pnl": None,
        "included_in_variant": True,
        "cooldown_skipped_signal": False,
        "cooldown_rule_name": "",
        "regime_blocked": False,
        "blocked_reason": "",
        "side_policy_blocked": False,
        "volatility_filter_blocked": False,
        "trend_strength_filter_blocked": False,
        "native_micro_exit_logic": normalized_micro_exit_id,
    }

    entry_dt = parse_utc(row.entry_time_utc)
    exit_dt = parse_utc(row.exit_time_utc)

    entry_idx = find_first_bar_at_or_after(entry_dt, ts_index, bars)
    exit_idx = find_first_bar_at_or_after(exit_dt, ts_index, bars)

    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        return out

    if normalized_micro_exit_id == "none":
        return out

    if normalized_micro_exit_id in {"conditional_max_hold_4h", "conditional_max_hold_6h"}:
        threshold_hours = 4 if normalized_micro_exit_id == "conditional_max_hold_4h" else 6
        threshold_dt = datetime.fromtimestamp(entry_dt.timestamp() + (threshold_hours * 3600.0), tz=timezone.utc)
        threshold_idx = find_first_bar_at_or_after(threshold_dt, ts_index, bars)
        if threshold_idx is None:
            return out
        if threshold_idx <= exit_idx:
            threshold_bar = bars[threshold_idx]
            if row.side == "BUY":
                threshold_pnl = threshold_bar.close - row.entry_price
            else:
                threshold_pnl = row.entry_price - threshold_bar.close

            out["effective_exit_time_utc"] = threshold_bar.ts_utc
            out["effective_exit_price"] = threshold_bar.close
            out["effective_pnl"] = threshold_pnl
            out["effective_exit_reason"] = "conditional_max_hold_exit"
            out["conditional_time_stop_applied"] = True
            out["threshold_bar_time_utc"] = threshold_bar.ts_utc
            out["threshold_bar_close"] = threshold_bar.close
            out["threshold_bar_pnl"] = threshold_pnl
        return out

    if atr_roll5 is None:
        raise ValueError("atr_roll5 precomputed list is required for native micro exits")

    bar_slice = bars[entry_idx + 1:exit_idx + 1]
    n = len(bar_slice)
    if n == 0:
        return out

    closes = np.array([b.close for b in bar_slice], dtype=np.float64)
    highs = np.array([b.high for b in bar_slice], dtype=np.float64)
    lows = np.array([b.low for b in bar_slice], dtype=np.float64)
    bullish_closes = np.array([b.bullish_close for b in bar_slice], dtype=np.bool_)
    lower_lows = np.array([b.lower_low for b in bar_slice], dtype=np.bool_)
    adx_vals = np.array([b.adx14 for b in bar_slice], dtype=np.float64)
    atr_vals = np.array([b.atr14 for b in bar_slice], dtype=np.float64)
    body_ratios = np.array([b.body_ratio for b in bar_slice], dtype=np.float64)
    di_plus_vals = np.array([b.di_plus for b in bar_slice], dtype=np.float64)
    di_minus_vals = np.array([b.di_minus for b in bar_slice], dtype=np.float64)
    atr_roll5_slice = np.array(atr_roll5[entry_idx + 1:exit_idx + 1], dtype=np.float64)

    if normalized_micro_exit_id == "micro_exit_v2_fast_invalidation":
        trigger_flags = vectorized_fast_invalidation_exit_slice(
            row.side, closes, lows, highs, bullish_closes, lower_lows, None
        )
    elif normalized_micro_exit_id == "micro_exit_v2_momentum_fade":
        trigger_flags = vectorized_momentum_fade_exit_slice(
            row.side, adx_vals, atr_vals, atr_roll5_slice, body_ratios, di_plus_vals, di_minus_vals
        )
    else:
        raise NotImplementedError(f"Micro exit not implemented in v1.0.6: {normalized_micro_exit_id}")

    if trigger_flags.any():
        first_trigger_idx = int(np.argmax(trigger_flags))
        trigger_bar = bar_slice[first_trigger_idx]
        if row.side == "BUY":
            effective_pnl = trigger_bar.close - row.entry_price
        else:
            effective_pnl = row.entry_price - trigger_bar.close
        out["effective_exit_time_utc"] = trigger_bar.ts_utc
        out["effective_exit_price"] = trigger_bar.close
        out["effective_pnl"] = effective_pnl
        out["effective_exit_reason"] = normalized_micro_exit_id
        return out

    return out


def apply_cooldown(rows: list[dict[str, Any]], cooldown_id: Any) -> list[dict[str, Any]]:
    normalized = normalize_cooldown_id(cooldown_id)

    if normalized == "none":
        return rows

    if normalized == "cooldown_2L_skip1":
        loss_threshold = 2
        rule_name = "cooldown_after_2_losses_skip_1"
    elif normalized == "cooldown_3L_skip1":
        loss_threshold = 3
        rule_name = "cooldown_after_3_losses_skip_1"
    else:
        raise NotImplementedError(f"Cooldown not implemented in v1.0.6: {cooldown_id}")

    out: list[dict[str, Any]] = []
    consecutive_losses = 0
    skip_next = False

    for row in rows:
        cloned = dict(row)

        if skip_next:
            cloned["included_in_variant"] = False
            cloned["cooldown_skipped_signal"] = True
            cloned["cooldown_rule_name"] = rule_name
            out.append(cloned)
            skip_next = False
            continue

        cloned["cooldown_skipped_signal"] = False
        cloned["cooldown_rule_name"] = rule_name
        out.append(cloned)

        if not cloned.get("included_in_variant", True):
            continue

        pnl_value = float(cloned.get("effective_pnl", 0.0))
        if pnl_value <= 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        if consecutive_losses >= loss_threshold:
            skip_next = True
            consecutive_losses = 0

    return out


def max_consecutive_losses(rows: list[dict[str, Any]]) -> int:
    max_loss_streak = 0
    current = 0

    for row in rows:
        if not row.get("included_in_variant", True):
            continue
        pnl_value = float(row.get("effective_pnl", 0.0))
        if pnl_value <= 0:
            current += 1
            if current > max_loss_streak:
                max_loss_streak = current
        else:
            current = 0

    return max_loss_streak


def summarize(
    job: dict[str, Any],
    rows: list[dict[str, Any]],
    side_policy_id: str,
    volatility_filter_id: str,
    trend_strength_filter_id: str,
) -> dict[str, Any]:
    active_rows = [r for r in rows if r.get("included_in_variant", True)]
    trades = len(active_rows)
    wins = sum(1 for r in active_rows if float(r.get("effective_pnl", 0.0)) > 0)
    losses = trades - wins
    pnl_sum = round(sum(float(r.get("effective_pnl", 0.0)) for r in active_rows), 4)

    gross_profit = sum(float(r.get("effective_pnl", 0.0)) for r in active_rows if float(r.get("effective_pnl", 0.0)) > 0)
    gross_loss_abs = abs(sum(float(r.get("effective_pnl", 0.0)) for r in active_rows if float(r.get("effective_pnl", 0.0)) <= 0))
    payoff_ratio = round(gross_profit / gross_loss_abs, 4) if gross_loss_abs > 0 else 0.0

    return {
        "runner_version": VERSION,
        "job_id": job["job_id"],
        "status": "done",
        "mode": "real_deep_pullback_family_execution",
        "stage": job["stage"],
        "symbol": job["symbol"],
        "timeframe": job["timeframe"],
        "dataset_ohlc_csv": job["dataset"]["ohlc_csv"],
        "family_id": job["family_id"],
        "entry_style": job["entry_style"],
        "micro_exit": job["micro_exit"],
        "cooldown": job["cooldown"],
        "regime_filter": job["regime_filter"],
        "side_policy": {"side_policy_id": side_policy_id},
        "volatility_filter": {"volatility_filter_id": volatility_filter_id},
        "trend_strength_filter": {"trend_strength_filter_id": trend_strength_filter_id},
        "metrics": {
            "pnl_sum": pnl_sum,
            "payoff_ratio": payoff_ratio,
            "max_consecutive_losses": max_consecutive_losses(rows),
            "trade_count": trades,
            "win_rate_pct": round((wins * 100.0 / trades), 2) if trades > 0 else 0.0,
            "wins": wins,
            "losses": losses,
            "native_micro_exit_count": sum(
                1 for r in active_rows
                if r.get("effective_exit_reason") in {
                    "micro_exit_v2_fast_invalidation",
                    "micro_exit_v2_momentum_fade",
                }
            ),
            "conditional_max_hold_exit_count": sum(
                1 for r in active_rows if r.get("effective_exit_reason") == "conditional_max_hold_exit"
            ),
            "cooldown_skipped_count": sum(1 for r in rows if r.get("cooldown_skipped_signal", False)),
            "regime_blocked_count": sum(1 for r in rows if r.get("regime_blocked", False)),
            "side_policy_blocked_count": sum(1 for r in rows if r.get("side_policy_blocked", False)),
            "volatility_filter_blocked_count": sum(1 for r in rows if r.get("volatility_filter_blocked", False)),
            "trend_strength_filter_blocked_count": sum(1 for r in rows if r.get("trend_strength_filter_blocked", False)),
        },
        "notes": [
            "v1.0.6 preserves v1.0.4 atr_roll5 precompute optimization",
            "v1.0.6 accepts side_policy/volatility_filter/trend_strength_filter as either string or dict",
            "v1.0.6 now applies side_policy from job manifest",
            "v1.0.6 now applies volatility_filter from job manifest using trades.jsonl volatility_bucket",
            "v1.0.6 now applies trend_strength_filter from job manifest using trades.jsonl trend_bucket",
            "v1.0.6 keeps regime_filter, cooldown, and micro_exit output schema compatible",
            "micro_exit_v2_structure_trail is still temporarily normalized to conditional_max_hold_6h",
            "cooldown_bars=6 is still temporarily normalized to cooldown_3L_skip1",
        ],
    }


def execute_deep_pullback_m30_family_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("family_id") != "deep_pullback_continuation":
        raise ValueError("family_id mismatch")
    if job.get("entry_style") != "deep":
        raise ValueError("entry_style mismatch for real executor")

    dataset = job.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset mapping is required")
    ohlc_csv_raw = dataset.get("ohlc_csv")
    if not ohlc_csv_raw:
        raise ValueError("dataset.ohlc_csv is required")

    trades = load_trades()
    ts_index, bars = load_ohlc_from_csv(Path(str(ohlc_csv_raw)))
    atr_roll5 = rolling_mean([bar.atr14 for bar in bars], 5)

    regime_filter_id = read_job_config_id(job, "regime_filter", "regime_filter_id", "regime_filter_off")
    micro_exit_id = read_job_config_id(job, "micro_exit", "exit_id", "none")
    cooldown_id = read_job_config_id(job, "cooldown", "cooldown_id", "none")
    side_policy_id = read_job_config_id(job, "side_policy", "side_policy_id", "both")
    volatility_filter_id = read_job_config_id(job, "volatility_filter", "volatility_filter_id", "any_vol")
    trend_strength_filter_id = read_job_config_id(job, "trend_strength_filter", "trend_strength_filter_id", "any_trend")

    normalized_micro_exit_id = normalize_micro_exit_id(micro_exit_id)

    processed: list[dict[str, Any]] = []
    for row in trades:
        side_blocked, side_reason = apply_side_policy_filter(row, side_policy_id)
        if side_blocked:
            item = build_effective_trade(
                row=row,
                normalized_micro_exit_id="none",
                ts_index=ts_index,
                bars=bars,
                atr_roll5=None,
            )
            item["included_in_variant"] = False
            item["side_policy_blocked"] = True
            item["blocked_reason"] = side_reason
            processed.append(item)
            continue

        vol_blocked, vol_reason = apply_volatility_filter(row, volatility_filter_id)
        if vol_blocked:
            item = build_effective_trade(
                row=row,
                normalized_micro_exit_id="none",
                ts_index=ts_index,
                bars=bars,
                atr_roll5=None,
            )
            item["included_in_variant"] = False
            item["volatility_filter_blocked"] = True
            item["blocked_reason"] = vol_reason
            processed.append(item)
            continue

        trend_blocked, trend_reason = apply_trend_strength_filter(row, trend_strength_filter_id)
        if trend_blocked:
            item = build_effective_trade(
                row=row,
                normalized_micro_exit_id="none",
                ts_index=ts_index,
                bars=bars,
                atr_roll5=None,
            )
            item["included_in_variant"] = False
            item["trend_strength_filter_blocked"] = True
            item["blocked_reason"] = trend_reason
            processed.append(item)
            continue

        regime_blocked, regime_reason = apply_regime_filter(row, regime_filter_id)
        if regime_blocked:
            item = build_effective_trade(
                row=row,
                normalized_micro_exit_id="none",
                ts_index=ts_index,
                bars=bars,
                atr_roll5=None,
            )
            item["included_in_variant"] = False
            item["regime_blocked"] = True
            item["blocked_reason"] = regime_reason
            processed.append(item)
            continue

        item = build_effective_trade(
            row=row,
            normalized_micro_exit_id=normalized_micro_exit_id,
            ts_index=ts_index,
            bars=bars,
            atr_roll5=atr_roll5,
        )
        processed.append(item)

    final_rows = apply_cooldown(processed, cooldown_id)
    summary = summarize(
        job=job,
        rows=final_rows,
        side_policy_id=side_policy_id,
        volatility_filter_id=volatility_filter_id,
        trend_strength_filter_id=trend_strength_filter_id,
    )

    result_dir = Path(job["artifact_paths"]["job_result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (result_dir / "audit_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in final_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary
