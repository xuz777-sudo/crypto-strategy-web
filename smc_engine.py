# -*- coding: utf-8 -*-
"""
SMC Engine V0.1
===============
Bybit / Crypto 5m short-term trading Smart Money Concepts engine.

設計目標
--------
1. 5 分 K：主要進出場訊號
2. 15 分 K / 1H：後續可作為趨勢濾網
3. 核心 SMC：
   - Swing High / Swing Low
   - BOS (Break of Structure)
   - CHoCH (Change of Character)
   - Fair Value Gap (FVG)
   - Order Block (OB)
   - Liquidity Sweep
   - Premium / Discount Zone
4. 輸出多頭/空頭分數與原因，方便串接 Streamlit、掃描器與回測引擎。

依賴：
    pandas
    numpy

注意：
- 本模組不直接呼叫 Bybit API。
- 輸入 DataFrame 至少需要：open, high, low, close
- volume 可選。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class SMCConfig:
    # Swing pivot confirmation:
    # a pivot is valid only after `swing_right` candles have closed.
    swing_left: int = 3
    swing_right: int = 3

    # ATR
    atr_period: int = 14

    # Structure break
    break_atr_buffer: float = 0.03

    # FVG
    fvg_min_atr: float = 0.08
    fvg_max_age: int = 80

    # Order block
    ob_lookback: int = 12
    ob_max_age: int = 100

    # Liquidity sweep
    sweep_atr_buffer: float = 0.02
    sweep_max_age: int = 30

    # Premium / Discount dealing range
    dealing_range_lookback: int = 80

    # Score freshness
    signal_fresh_bars: int = 8

    # Risk reference
    stop_atr_buffer: float = 0.15

    def validate(self) -> None:
        ints = {
            "swing_left": self.swing_left,
            "swing_right": self.swing_right,
            "atr_period": self.atr_period,
            "fvg_max_age": self.fvg_max_age,
            "ob_lookback": self.ob_lookback,
            "ob_max_age": self.ob_max_age,
            "sweep_max_age": self.sweep_max_age,
            "dealing_range_lookback": self.dealing_range_lookback,
            "signal_fresh_bars": self.signal_fresh_bars,
        }
        for name, value in ints.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _to_python(value: Any) -> Any:
    """Convert numpy/pandas scalar to JSON-friendly Python types."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common OHLCV column names to lowercase.
    Preserves index and additional columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    if df.empty:
        raise ValueError("OHLCV DataFrame is empty")

    out = df.copy()

    rename_map: Dict[str, str] = {}
    for c in out.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        aliases = {
            "open": "open",
            "o": "open",
            "high": "high",
            "h": "high",
            "low": "low",
            "l": "low",
            "close": "close",
            "c": "close",
            "volume": "volume",
            "vol": "volume",
            "v": "volume",
            "turnover": "turnover",
            "timestamp": "timestamp",
            "time": "timestamp",
            "datetime": "timestamp",
            "starttime": "timestamp",
            "opentime": "timestamp",
        }
        if key in aliases and aliases[key] not in out.columns:
            rename_map[c] = aliases[key]

    if rename_map:
        out = out.rename(columns=rename_map)

    missing = [c for c in _REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=list(_REQUIRED_COLUMNS)).copy()
    if out.empty:
        raise ValueError("No valid OHLC rows after numeric conversion")

    # Sort by timestamp column when available.
    if "timestamp" in out.columns:
        try:
            ts_num = pd.to_numeric(out["timestamp"], errors="coerce")
            if ts_num.notna().mean() > 0.8:
                # Bybit commonly returns milliseconds.
                unit = "ms" if ts_num.abs().median() > 1e11 else "s"
                out["timestamp"] = pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")
            else:
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.sort_values("timestamp")
        except Exception:
            pass

    out = out.reset_index(drop=True)
    return out


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    tr = _true_range(out)
    out["atr"] = tr.ewm(alpha=1.0 / max(period, 1), adjust=False).mean()
    return out


def _latest_non_null(series: pd.Series, before_or_at: Optional[int] = None) -> Tuple[Optional[int], Optional[float]]:
    if before_or_at is None:
        s = series
    else:
        s = series.iloc[: before_or_at + 1]
    s = s.dropna()
    if s.empty:
        return None, None
    idx = int(s.index[-1])
    return idx, float(s.iloc[-1])


def _event_age(last_index: int, event_index: Optional[int]) -> Optional[int]:
    if event_index is None:
        return None
    return int(last_index - event_index)


# ---------------------------------------------------------------------
# Swing High / Swing Low
# ---------------------------------------------------------------------

def detect_swings(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> pd.DataFrame:
    """
    Detect confirmed swing highs/lows.

    Columns:
      swing_high        price at pivot candle
      swing_low         price at pivot candle
      swing_high_confirmed_at
      swing_low_confirmed_at

    The pivot price is written on the pivot candle itself. Confirmation
    happens `right` bars later, and confirmed_at records that bar index.
    """
    out = df.copy()
    n = len(out)

    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    sh_confirm = np.full(n, np.nan)
    sl_confirm = np.full(n, np.nan)

    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)

    for i in range(left, n - right):
        h_window = highs[i - left : i + right + 1]
        l_window = lows[i - left : i + right + 1]

        # Unique-extreme requirement reduces duplicate flat pivots.
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(h_window):
            if np.sum(np.isclose(h_window, highs[i], equal_nan=False)) == 1:
                swing_high[i] = highs[i]
                sh_confirm[i] = i + right

        if np.isfinite(lows[i]) and lows[i] == np.nanmin(l_window):
            if np.sum(np.isclose(l_window, lows[i], equal_nan=False)) == 1:
                swing_low[i] = lows[i]
                sl_confirm[i] = i + right

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["swing_high_confirmed_at"] = sh_confirm
    out["swing_low_confirmed_at"] = sl_confirm
    return out


# ---------------------------------------------------------------------
# BOS / CHoCH
# ---------------------------------------------------------------------

def detect_structure(
    df: pd.DataFrame,
    atr_buffer: float = 0.03,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Detect BOS / CHoCH using confirmed swing levels and candle close breaks.

    state:
      1  = bullish structure
     -1  = bearish structure
      0  = unknown

    A break in current direction => BOS.
    A break against current direction => CHoCH.
    """
    out = df.copy()
    n = len(out)

    structure_event = np.array([""] * n, dtype=object)
    structure_side = np.array([""] * n, dtype=object)
    broken_level = np.full(n, np.nan)
    structure_state = np.zeros(n, dtype=int)

    events: List[Dict[str, Any]] = []

    last_sh_price: Optional[float] = None
    last_sh_pivot: Optional[int] = None
    last_sl_price: Optional[float] = None
    last_sl_pivot: Optional[int] = None

    broken_sh_pivots = set()
    broken_sl_pivots = set()

    state = 0

    # Build confirmation schedules to avoid using future pivots.
    sh_confirms: Dict[int, List[Tuple[int, float]]] = {}
    sl_confirms: Dict[int, List[Tuple[int, float]]] = {}

    for pivot_i, row in out.iterrows():
        if pd.notna(row.get("swing_high")) and pd.notna(row.get("swing_high_confirmed_at")):
            c = int(row["swing_high_confirmed_at"])
            sh_confirms.setdefault(c, []).append((int(pivot_i), float(row["swing_high"])))

        if pd.notna(row.get("swing_low")) and pd.notna(row.get("swing_low_confirmed_at")):
            c = int(row["swing_low_confirmed_at"])
            sl_confirms.setdefault(c, []).append((int(pivot_i), float(row["swing_low"])))

    for i in range(n):
        # Make newly confirmed swings available from this candle onward.
        for pivot_i, price in sh_confirms.get(i, []):
            last_sh_pivot = pivot_i
            last_sh_price = price

        for pivot_i, price in sl_confirms.get(i, []):
            last_sl_pivot = pivot_i
            last_sl_price = price

        atr = float(out["atr"].iloc[i]) if pd.notna(out["atr"].iloc[i]) else 0.0
        close = float(out["close"].iloc[i])
        buffer = atr * float(atr_buffer)

        bull_break = (
            last_sh_price is not None
            and last_sh_pivot is not None
            and last_sh_pivot not in broken_sh_pivots
            and close > last_sh_price + buffer
        )

        bear_break = (
            last_sl_price is not None
            and last_sl_pivot is not None
            and last_sl_pivot not in broken_sl_pivots
            and close < last_sl_price - buffer
        )

        # Extremely large candles can theoretically break both old boundaries.
        # Use the close direction relative to candle open to choose one.
        if bull_break and bear_break:
            if close >= float(out["open"].iloc[i]):
                bear_break = False
            else:
                bull_break = False

        if bull_break:
            event_type = "BOS" if state >= 0 else "CHoCH"
            state = 1
            broken_sh_pivots.add(last_sh_pivot)

            structure_event[i] = event_type
            structure_side[i] = "bullish"
            broken_level[i] = float(last_sh_price)

            events.append({
                "index": i,
                "type": event_type,
                "side": "bullish",
                "level": float(last_sh_price),
                "pivot_index": int(last_sh_pivot),
                "close": close,
            })

        elif bear_break:
            event_type = "BOS" if state <= 0 else "CHoCH"
            state = -1
            broken_sl_pivots.add(last_sl_pivot)

            structure_event[i] = event_type
            structure_side[i] = "bearish"
            broken_level[i] = float(last_sl_price)

            events.append({
                "index": i,
                "type": event_type,
                "side": "bearish",
                "level": float(last_sl_price),
                "pivot_index": int(last_sl_pivot),
                "close": close,
            })

        structure_state[i] = state

    out["structure_event"] = structure_event
    out["structure_side"] = structure_side
    out["broken_level"] = broken_level
    out["structure_state"] = structure_state

    return out, events


# ---------------------------------------------------------------------
# Fair Value Gap
# ---------------------------------------------------------------------

def detect_fvg(
    df: pd.DataFrame,
    min_atr: float = 0.08,
    max_age: int = 80,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    ICT-style 3-candle FVG:
      bullish: high[i-2] < low[i]
      bearish: low[i-2] > high[i]

    Active FVG remains active until price fully trades through the far edge.
    """
    out = df.copy()
    n = len(out)
    events: List[Dict[str, Any]] = []

    fvg_side = np.array([""] * n, dtype=object)
    fvg_low = np.full(n, np.nan)
    fvg_high = np.full(n, np.nan)
    fvg_size = np.full(n, np.nan)

    for i in range(2, n):
        atr = float(out["atr"].iloc[i]) if pd.notna(out["atr"].iloc[i]) else 0.0
        min_gap = atr * float(min_atr)

        left_high = float(out["high"].iloc[i - 2])
        left_low = float(out["low"].iloc[i - 2])
        cur_high = float(out["high"].iloc[i])
        cur_low = float(out["low"].iloc[i])

        # Bullish imbalance
        gap = cur_low - left_high
        if gap > max(min_gap, 0.0):
            lo, hi = left_high, cur_low
            fvg_side[i] = "bullish"
            fvg_low[i] = lo
            fvg_high[i] = hi
            fvg_size[i] = gap

            filled_at = None
            for j in range(i + 1, min(n, i + max_age + 1)):
                if float(out["low"].iloc[j]) <= lo:
                    filled_at = j
                    break

            events.append({
                "index": i,
                "side": "bullish",
                "low": lo,
                "high": hi,
                "size": gap,
                "filled_at": filled_at,
                "active": filled_at is None,
            })

        # Bearish imbalance
        gap = left_low - cur_high
        if gap > max(min_gap, 0.0):
            lo, hi = cur_high, left_low
            fvg_side[i] = "bearish"
            fvg_low[i] = lo
            fvg_high[i] = hi
            fvg_size[i] = gap

            filled_at = None
            for j in range(i + 1, min(n, i + max_age + 1)):
                if float(out["high"].iloc[j]) >= hi:
                    filled_at = j
                    break

            events.append({
                "index": i,
                "side": "bearish",
                "low": lo,
                "high": hi,
                "size": gap,
                "filled_at": filled_at,
                "active": filled_at is None,
            })

    out["fvg_side"] = fvg_side
    out["fvg_low"] = fvg_low
    out["fvg_high"] = fvg_high
    out["fvg_size"] = fvg_size
    return out, events


# ---------------------------------------------------------------------
# Order Block
# ---------------------------------------------------------------------

def detect_order_blocks(
    df: pd.DataFrame,
    structure_events: List[Dict[str, Any]],
    lookback: int = 12,
    max_age: int = 100,
) -> List[Dict[str, Any]]:
    """
    Simplified SMC Order Block:
      bullish BOS/CHoCH -> last bearish candle before break
      bearish BOS/CHoCH -> last bullish candle before break

    Zone uses candle low/high.
    """
    n = len(df)
    obs: List[Dict[str, Any]] = []

    for ev in structure_events:
        i = int(ev["index"])
        side = str(ev["side"])

        start = max(0, i - int(lookback))
        candidate: Optional[int] = None

        if side == "bullish":
            for j in range(i - 1, start - 1, -1):
                if float(df["close"].iloc[j]) < float(df["open"].iloc[j]):
                    candidate = j
                    break
        else:
            for j in range(i - 1, start - 1, -1):
                if float(df["close"].iloc[j]) > float(df["open"].iloc[j]):
                    candidate = j
                    break

        if candidate is None:
            continue

        lo = float(df["low"].iloc[candidate])
        hi = float(df["high"].iloc[candidate])

        invalidated_at = None
        for j in range(i + 1, min(n, i + max_age + 1)):
            close = float(df["close"].iloc[j])
            if side == "bullish" and close < lo:
                invalidated_at = j
                break
            if side == "bearish" and close > hi:
                invalidated_at = j
                break

        obs.append({
            "index": candidate,
            "created_at": i,
            "structure_type": ev["type"],
            "side": side,
            "low": lo,
            "high": hi,
            "mid": (lo + hi) / 2.0,
            "invalidated_at": invalidated_at,
            "active": invalidated_at is None,
        })

    # Remove exact duplicate blocks caused by repeated structure events.
    dedup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for ob in obs:
        dedup[(int(ob["index"]), str(ob["side"]))] = ob

    return list(dedup.values())


# ---------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    atr_buffer: float = 0.02,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Buy-side sweep:
        wick above previous confirmed swing high, close back below level.
        This is bearish rejection evidence.

    Sell-side sweep:
        wick below previous confirmed swing low, close back above level.
        This is bullish rejection evidence.
    """
    out = df.copy()
    n = len(out)

    sweep_side = np.array([""] * n, dtype=object)
    sweep_level = np.full(n, np.nan)
    events: List[Dict[str, Any]] = []

    last_sh_price: Optional[float] = None
    last_sl_price: Optional[float] = None

    sh_confirms: Dict[int, List[float]] = {}
    sl_confirms: Dict[int, List[float]] = {}

    for _, row in out.iterrows():
        if pd.notna(row.get("swing_high")) and pd.notna(row.get("swing_high_confirmed_at")):
            sh_confirms.setdefault(int(row["swing_high_confirmed_at"]), []).append(float(row["swing_high"]))
        if pd.notna(row.get("swing_low")) and pd.notna(row.get("swing_low_confirmed_at")):
            sl_confirms.setdefault(int(row["swing_low_confirmed_at"]), []).append(float(row["swing_low"]))

    for i in range(n):
        for p in sh_confirms.get(i, []):
            last_sh_price = p
        for p in sl_confirms.get(i, []):
            last_sl_price = p

        atr = float(out["atr"].iloc[i]) if pd.notna(out["atr"].iloc[i]) else 0.0
        buffer = atr * float(atr_buffer)

        high = float(out["high"].iloc[i])
        low = float(out["low"].iloc[i])
        close = float(out["close"].iloc[i])

        if last_sh_price is not None:
            if high > last_sh_price + buffer and close < last_sh_price:
                sweep_side[i] = "buy_side"
                sweep_level[i] = last_sh_price
                events.append({
                    "index": i,
                    "side": "buy_side",
                    "bias": "bearish",
                    "level": last_sh_price,
                    "high": high,
                    "close": close,
                })

        if last_sl_price is not None:
            if low < last_sl_price - buffer and close > last_sl_price:
                sweep_side[i] = "sell_side"
                sweep_level[i] = last_sl_price
                events.append({
                    "index": i,
                    "side": "sell_side",
                    "bias": "bullish",
                    "level": last_sl_price,
                    "low": low,
                    "close": close,
                })

    out["liquidity_sweep"] = sweep_side
    out["liquidity_level"] = sweep_level
    return out, events


# ---------------------------------------------------------------------
# Premium / Discount
# ---------------------------------------------------------------------

def _dealing_range(df: pd.DataFrame, lookback: int) -> Dict[str, Any]:
    """
    Determine recent dealing range using recent confirmed pivots when possible.
    Falls back to rolling high/low.
    """
    last = len(df) - 1
    start = max(0, last - int(lookback) + 1)
    window = df.iloc[start : last + 1]

    highs = window["swing_high"].dropna()
    lows = window["swing_low"].dropna()

    range_high = float(highs.max()) if not highs.empty else float(window["high"].max())
    range_low = float(lows.min()) if not lows.empty else float(window["low"].min())

    eq = (range_high + range_low) / 2.0
    price = float(df["close"].iloc[-1])

    if price < eq:
        zone = "discount"
    elif price > eq:
        zone = "premium"
    else:
        zone = "equilibrium"

    pos = None
    if range_high > range_low:
        pos = (price - range_low) / (range_high - range_low)

    return {
        "high": range_high,
        "low": range_low,
        "equilibrium": eq,
        "price": price,
        "zone": zone,
        "position": pos,
    }


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def _latest_event(
    events: List[Dict[str, Any]],
    side: Optional[str] = None,
    bias: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    filtered = events
    if side is not None:
        filtered = [e for e in filtered if e.get("side") == side]
    if bias is not None:
        filtered = [e for e in filtered if e.get("bias") == bias]
    if not filtered:
        return None
    return max(filtered, key=lambda e: int(e.get("index", -1)))


def _recent(index_now: int, event: Optional[Dict[str, Any]], bars: int) -> bool:
    return event is not None and (index_now - int(event.get("index", -10**9))) <= bars


def _price_in_zone(price: float, low: float, high: float, tolerance: float = 0.0) -> bool:
    return (low - tolerance) <= price <= (high + tolerance)


def score_smc(
    df: pd.DataFrame,
    structure_events: List[Dict[str, Any]],
    fvgs: List[Dict[str, Any]],
    order_blocks: List[Dict[str, Any]],
    sweeps: List[Dict[str, Any]],
    dealing_range: Dict[str, Any],
    config: SMCConfig,
) -> Dict[str, Any]:
    """
    Returns independent long/short scores from 0 to 100.
    """
    last_i = len(df) - 1
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0

    long_score = 0
    short_score = 0
    long_reasons: List[str] = []
    short_reasons: List[str] = []

    structure_state = int(df["structure_state"].iloc[-1])

    # 1) Structure state: 25
    if structure_state > 0:
        long_score += 25
        long_reasons.append("市場結構偏多")
    elif structure_state < 0:
        short_score += 25
        short_reasons.append("市場結構偏空")

    # 2) Recent CHoCH/BOS: up to 25
    latest_bull_structure = _latest_event(structure_events, side="bullish")
    latest_bear_structure = _latest_event(structure_events, side="bearish")

    if _recent(last_i, latest_bull_structure, config.signal_fresh_bars):
        pts = 25 if latest_bull_structure["type"] == "CHoCH" else 18
        long_score += pts
        long_reasons.append(f"近期 bullish {latest_bull_structure['type']}")

    if _recent(last_i, latest_bear_structure, config.signal_fresh_bars):
        pts = 25 if latest_bear_structure["type"] == "CHoCH" else 18
        short_score += pts
        short_reasons.append(f"近期 bearish {latest_bear_structure['type']}")

    # 3) Liquidity sweep: 20
    bull_sweep = _latest_event(sweeps, bias="bullish")
    bear_sweep = _latest_event(sweeps, bias="bearish")

    if _recent(last_i, bull_sweep, config.sweep_max_age):
        long_score += 20
        long_reasons.append("近期 sell-side liquidity sweep")

    if _recent(last_i, bear_sweep, config.sweep_max_age):
        short_score += 20
        short_reasons.append("近期 buy-side liquidity sweep")

    # 4) Active FVG proximity: 12
    active_bull_fvgs = [f for f in fvgs if f.get("active") and f.get("side") == "bullish"]
    active_bear_fvgs = [f for f in fvgs if f.get("active") and f.get("side") == "bearish"]

    fvg_tol = atr * 0.20
    near_bull_fvg = [
        f for f in active_bull_fvgs
        if _price_in_zone(price, float(f["low"]), float(f["high"]), fvg_tol)
    ]
    near_bear_fvg = [
        f for f in active_bear_fvgs
        if _price_in_zone(price, float(f["low"]), float(f["high"]), fvg_tol)
    ]

    if near_bull_fvg:
        long_score += 12
        long_reasons.append("價格位於/接近 bullish FVG")

    if near_bear_fvg:
        short_score += 12
        short_reasons.append("價格位於/接近 bearish FVG")

    # 5) Active OB proximity: 12
    active_bull_obs = [o for o in order_blocks if o.get("active") and o.get("side") == "bullish"]
    active_bear_obs = [o for o in order_blocks if o.get("active") and o.get("side") == "bearish"]

    ob_tol = atr * 0.15
    near_bull_ob = [
        o for o in active_bull_obs
        if _price_in_zone(price, float(o["low"]), float(o["high"]), ob_tol)
    ]
    near_bear_ob = [
        o for o in active_bear_obs
        if _price_in_zone(price, float(o["low"]), float(o["high"]), ob_tol)
    ]

    if near_bull_ob:
        long_score += 12
        long_reasons.append("價格位於/接近 bullish Order Block")

    if near_bear_ob:
        short_score += 12
        short_reasons.append("價格位於/接近 bearish Order Block")

    # 6) Premium / Discount: 10
    zone = dealing_range.get("zone")
    if zone == "discount":
        long_score += 10
        long_reasons.append("價格位於 Discount Zone")
    elif zone == "premium":
        short_score += 10
        short_reasons.append("價格位於 Premium Zone")

    # Clamp
    long_score = int(max(0, min(100, long_score)))
    short_score = int(max(0, min(100, short_score)))

    if long_score >= 70 and long_score >= short_score + 10:
        bias = "LONG"
    elif short_score >= 70 and short_score >= long_score + 10:
        bias = "SHORT"
    elif long_score >= 55 and long_score > short_score:
        bias = "LEAN_LONG"
    elif short_score >= 55 and short_score > long_score:
        bias = "LEAN_SHORT"
    else:
        bias = "NEUTRAL"

    confidence = max(long_score, short_score)

    return {
        "bias": bias,
        "confidence": confidence,
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
    }


# ---------------------------------------------------------------------
# Entry / Stop references
# ---------------------------------------------------------------------

def _trade_levels(
    df: pd.DataFrame,
    score: Dict[str, Any],
    fvgs: List[Dict[str, Any]],
    order_blocks: List[Dict[str, Any]],
    dealing_range: Dict[str, Any],
    config: SMCConfig,
) -> Dict[str, Any]:
    """
    Produces reference entry/SL/TP levels for later strategy_engine/backtest.
    These are deterministic references, not exchange orders.
    """
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
    bias = score["bias"]

    result = {
        "entry": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "rr_tp1": None,
        "rr_tp2": None,
    }

    if bias not in {"LONG", "SHORT", "LEAN_LONG", "LEAN_SHORT"}:
        return result

    if bias in {"LONG", "LEAN_LONG"}:
        active_obs = [o for o in order_blocks if o.get("active") and o.get("side") == "bullish"]
        active_fvg = [f for f in fvgs if f.get("active") and f.get("side") == "bullish"]

        candidates: List[Tuple[float, float, float]] = []
        for o in active_obs:
            candidates.append((abs(price - float(o["mid"])), float(o["low"]), float(o["high"])))
        for f in active_fvg:
            mid = (float(f["low"]) + float(f["high"])) / 2.0
            candidates.append((abs(price - mid), float(f["low"]), float(f["high"])))

        if candidates:
            _, zone_low, zone_high = min(candidates, key=lambda x: x[0])
            entry = min(max(price, zone_low), zone_high)
            stop = zone_low - atr * config.stop_atr_buffer
        else:
            entry = price
            _, swing_low = _latest_non_null(df["swing_low"])
            stop = (swing_low if swing_low is not None else price - atr) - atr * config.stop_atr_buffer

        risk = max(entry - stop, atr * 0.10, 1e-12)
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5

    else:
        active_obs = [o for o in order_blocks if o.get("active") and o.get("side") == "bearish"]
        active_fvg = [f for f in fvgs if f.get("active") and f.get("side") == "bearish"]

        candidates = []
        for o in active_obs:
            candidates.append((abs(price - float(o["mid"])), float(o["low"]), float(o["high"])))
        for f in active_fvg:
            mid = (float(f["low"]) + float(f["high"])) / 2.0
            candidates.append((abs(price - mid), float(f["low"]), float(f["high"])))

        if candidates:
            _, zone_low, zone_high = min(candidates, key=lambda x: x[0])
            entry = min(max(price, zone_low), zone_high)
            stop = zone_high + atr * config.stop_atr_buffer
        else:
            entry = price
            _, swing_high = _latest_non_null(df["swing_high"])
            stop = (swing_high if swing_high is not None else price + atr) + atr * config.stop_atr_buffer

        risk = max(stop - entry, atr * 0.10, 1e-12)
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5

    result.update({
        "entry": float(entry),
        "stop_loss": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr_tp1": 1.5,
        "rr_tp2": 2.5,
    })
    return result


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

class SMCEngine:
    def __init__(self, config: Optional[SMCConfig] = None):
        self.config = config or SMCConfig()
        self.config.validate()

    def analyze(
        self,
        df: pd.DataFrame,
        include_dataframe: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Returns JSON-friendly summary. When include_dataframe=True,
        enriched_df is also returned for charts/debugging.
        """
        cfg = self.config
        x = _normalize_ohlcv(df)

        min_rows = cfg.swing_left + cfg.swing_right + 5
        if len(x) < min_rows:
            raise ValueError(
                f"Not enough candles for SMC analysis: got {len(x)}, need at least {min_rows}"
            )

        x = add_atr(x, cfg.atr_period)
        x = detect_swings(x, cfg.swing_left, cfg.swing_right)
        x, structure_events = detect_structure(x, cfg.break_atr_buffer)
        x, fvgs = detect_fvg(x, cfg.fvg_min_atr, cfg.fvg_max_age)
        x, sweeps = detect_liquidity_sweeps(x, cfg.sweep_atr_buffer)

        order_blocks = detect_order_blocks(
            x,
            structure_events=structure_events,
            lookback=cfg.ob_lookback,
            max_age=cfg.ob_max_age,
        )

        dr = _dealing_range(x, cfg.dealing_range_lookback)

        score = score_smc(
            x,
            structure_events=structure_events,
            fvgs=fvgs,
            order_blocks=order_blocks,
            sweeps=sweeps,
            dealing_range=dr,
            config=cfg,
        )

        levels = _trade_levels(
            x,
            score=score,
            fvgs=fvgs,
            order_blocks=order_blocks,
            dealing_range=dr,
            config=cfg,
        )

        last_i = len(x) - 1

        latest_structure = max(structure_events, key=lambda e: e["index"]) if structure_events else None
        latest_sweep = max(sweeps, key=lambda e: e["index"]) if sweeps else None

        active_fvgs = [
            f for f in fvgs
            if f.get("active") and (last_i - int(f["index"])) <= cfg.fvg_max_age
        ]
        active_obs = [
            o for o in order_blocks
            if o.get("active") and (last_i - int(o["created_at"])) <= cfg.ob_max_age
        ]

        # Last confirmed pivots as of current candle.
        confirmed_sh = []
        confirmed_sl = []
        for i, row in x.iterrows():
            if pd.notna(row["swing_high"]) and pd.notna(row["swing_high_confirmed_at"]):
                if int(row["swing_high_confirmed_at"]) <= last_i:
                    confirmed_sh.append((int(i), float(row["swing_high"])))
            if pd.notna(row["swing_low"]) and pd.notna(row["swing_low_confirmed_at"]):
                if int(row["swing_low_confirmed_at"]) <= last_i:
                    confirmed_sl.append((int(i), float(row["swing_low"])))

        last_sh = confirmed_sh[-1] if confirmed_sh else None
        last_sl = confirmed_sl[-1] if confirmed_sl else None

        result: Dict[str, Any] = {
            "engine": "SMC Engine V0.1",
            "bars": int(len(x)),
            "last_price": _to_python(x["close"].iloc[-1]),
            "atr": _to_python(x["atr"].iloc[-1]),
            "bias": score["bias"],
            "confidence": score["confidence"],
            "long_score": score["long_score"],
            "short_score": score["short_score"],
            "long_reasons": score["long_reasons"],
            "short_reasons": score["short_reasons"],
            "structure_state": (
                "bullish" if int(x["structure_state"].iloc[-1]) > 0
                else "bearish" if int(x["structure_state"].iloc[-1]) < 0
                else "neutral"
            ),
            "latest_structure": latest_structure,
            "latest_liquidity_sweep": latest_sweep,
            "last_swing_high": (
                {"index": last_sh[0], "price": last_sh[1]} if last_sh else None
            ),
            "last_swing_low": (
                {"index": last_sl[0], "price": last_sl[1]} if last_sl else None
            ),
            "dealing_range": {k: _to_python(v) for k, v in dr.items()},
            "active_fvgs": active_fvgs[-8:],
            "active_order_blocks": active_obs[-8:],
            "trade_levels": levels,
            "counts": {
                "structure_events": len(structure_events),
                "fvgs": len(fvgs),
                "active_fvgs": len(active_fvgs),
                "order_blocks": len(order_blocks),
                "active_order_blocks": len(active_obs),
                "liquidity_sweeps": len(sweeps),
            },
            "config": asdict(cfg),
        }

        # Optional time info.
        if "timestamp" in x.columns and pd.notna(x["timestamp"].iloc[-1]):
            result["last_timestamp"] = _to_python(x["timestamp"].iloc[-1])

        if include_dataframe:
            result["enriched_df"] = x

        return result


# ---------------------------------------------------------------------
# Functional API / compatibility aliases
# ---------------------------------------------------------------------

def analyze_smc(
    df: pd.DataFrame,
    config: Optional[SMCConfig] = None,
    include_dataframe: bool = False,
) -> Dict[str, Any]:
    """Functional wrapper around SMCEngine.analyze()."""
    return SMCEngine(config=config).analyze(df, include_dataframe=include_dataframe)


def calculate_smc(
    df: pd.DataFrame,
    config: Optional[SMCConfig] = None,
) -> Dict[str, Any]:
    """Compatibility alias."""
    return analyze_smc(df, config=config, include_dataframe=False)


def get_smc_score(
    df: pd.DataFrame,
    config: Optional[SMCConfig] = None,
) -> Dict[str, Any]:
    """
    Return only the most commonly needed scoring fields.
    """
    r = analyze_smc(df, config=config, include_dataframe=False)
    return {
        "bias": r["bias"],
        "confidence": r["confidence"],
        "long_score": r["long_score"],
        "short_score": r["short_score"],
        "long_reasons": r["long_reasons"],
        "short_reasons": r["short_reasons"],
        "trade_levels": r["trade_levels"],
    }


__all__ = [
    "SMCConfig",
    "SMCEngine",
    "add_atr",
    "detect_swings",
    "detect_structure",
    "detect_fvg",
    "detect_order_blocks",
    "detect_liquidity_sweeps",
    "score_smc",
    "analyze_smc",
    "calculate_smc",
    "get_smc_score",
]
