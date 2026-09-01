# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from bybit_client import BybitClient
from cloud_state import CloudStateStore
from indicators import prepare_indicators
from smc_engine import analyze_smc
from strategy_engine import build_trade_plan


# =============================================================================
# Page
# =============================================================================

st.set_page_config(page_title="5分K即時進場監控 V0.5.4.1", layout="wide")
st.title("5分K 即時進出場監控 V0.5.4.1｜真實化虛擬績效")
st.caption(
    "讀取第二階段精查保存在 GitHub Private Repo 的候選，只更新候選的最新 5分K，"
    "不重新掃描全市場。所有交易皆為策略虛擬訊號，不代表 Bybit 真實下單。"
)

cloud = CloudStateStore()
client = BybitClient()

PRECISION_KEY = "precision/latest"
MONITOR_KEY = "entry_monitor/latest"
HISTORY_KEY = "entry_monitor/history"

TW_TZ = timezone(timedelta(hours=8))

OPEN_TRADE_STATUSES = {
    "多頭訊號成立",
    "空頭訊號成立",
    "虛擬持倉監控",
    "TP1 已達・保本中",
}

WAIT_STATUSES = {
    "等待多頭觸發",
    "等待空頭觸發",
    "等待新結構",
}

CLOSED_TRADE_STATUSES = {
    "TP2 完成",
    "停損出場",
    "保本出場",
    "結構反轉出場",
}

NON_TRADE_TERMINAL = {
    "禁止追價",
    "訊號失效",
}


# =============================================================================
# Basic helpers
# =============================================================================

def safe_float(v, default=float("nan")):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def fmt_price(v):
    v = safe_float(v)
    if not math.isfinite(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    return f"{v:.8f}".rstrip("0").rstrip(".")


def now_utc_dt():
    return datetime.now(timezone.utc)


def now_utc_text():
    return now_utc_dt().strftime("%Y-%m-%d %H:%M:%S")


def now_tw_text():
    return now_utc_dt().astimezone(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def parse_utc_text(value):
    if not value:
        return None
    text = str(value).strip()
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None


def elapsed_minutes(start_text, end_text=None):
    start = parse_utc_text(start_text)
    if start is None:
        return None

    end = parse_utc_text(end_text) if end_text else now_utc_dt()
    if end is None:
        end = now_utc_dt()

    return max(0, int((end - start).total_seconds() // 60))


def direction_code(row):
    text = str(row.get("精查方向", row.get("方向", "")))
    return "LONG" if text == "多頭" else ("SHORT" if text == "空頭" else "NEUTRAL")


def smc_bias_zh(value):
    return {
        "LONG": "多頭",
        "LEAN_LONG": "偏多",
        "SHORT": "空頭",
        "LEAN_SHORT": "偏空",
        "NEUTRAL": "中性",
        "bullish": "多頭",
        "bearish": "空頭",
        "neutral": "中性",
    }.get(str(value), str(value))


def structure_signature(smc):
    """
    同一訊號鎖定用。
    停損 / 出場後，不允許同一個 BOS / CHoCH 結構立刻再次進場。
    直到 latest_structure 改變，才視為新訊號。
    """
    ev = smc.get("latest_structure") or {}
    side = str(ev.get("side", "-"))
    typ = str(ev.get("type", "-"))
    idx = str(ev.get("index", "-"))
    level = safe_float(ev.get("level"))
    level_text = f"{level:.10g}" if math.isfinite(level) else "-"
    return f"{side}|{typ}|{idx}|{level_text}"


def current_structure_text(smc):
    ev = smc.get("latest_structure") or {}
    if not ev:
        return "-"
    return (
        f"{smc_bias_zh(ev.get('side'))} "
        f"{ev.get('type', '-')} @ {fmt_price(ev.get('level'))}"
    )


# =============================================================================
# Cloud state
# =============================================================================

def load_precision_rows():
    body = cloud.load(PRECISION_KEY)
    if not body:
        return [], None

    payload = body.get("payload", {}) or {}
    return payload.get("precision_rows", []) or [], body.get("saved_at_utc")


def load_monitor_payload():
    """
    Backward compatible with V0.5:
    - V0.5 only had payload.rows
    - V0.5.3 adds rows + locks
    """
    try:
        body = cloud.load(MONITOR_KEY)
        if not body:
            return {}, {}

        payload = body.get("payload", {}) or {}
        rows = payload.get("rows", []) or []
        locks = payload.get("locks", {}) or {}

        row_map = {
            str(r.get("交易對", "")).upper(): r
            for r in rows
            if r.get("交易對")
        }
        return row_map, locks
    except Exception:
        return {}, {}


def load_history_rows():
    try:
        body = cloud.load(HISTORY_KEY)
        if not body:
            return []

        payload = body.get("payload", {}) or {}
        return payload.get("history", []) or []
    except Exception:
        return []


def save_monitor_payload(rows, locks):
    if not cloud.available:
        return

    cloud.save(
        MONITOR_KEY,
        {
            "version": "V0.5.4.1",
            "rows": rows,
            "locks": locks,
            "updated_at_utc": now_utc_dt().isoformat(),
        },
    )


def save_history_rows(history):
    if not cloud.available:
        return

    cloud.save(
        HISTORY_KEY,
        {
            "version": "V0.5.4.1",
            "history": history,
            "updated_at_utc": now_utc_dt().isoformat(),
        },
    )


# =============================================================================
# Candidate selection
# =============================================================================

def candidate_rows(rows, mode, min_score, top_n):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "最終精查分數" in df.columns:
        df["最終精查分數"] = pd.to_numeric(
            df["最終精查分數"],
            errors="coerce",
        )

    if mode == "只監控可進場／等待觸發":
        allowed = [
            "多頭進場條件成立",
            "空頭進場條件成立",
            "等待多頭進場觸發",
            "等待空頭進場觸發",
        ]
        df = df[df["最終判定"].isin(allowed)]

    elif mode == "包含保留觀察":
        allowed = [
            "多頭進場條件成立",
            "空頭進場條件成立",
            "等待多頭進場觸發",
            "等待空頭進場觸發",
            "保留觀察",
        ]
        df = df[df["最終判定"].isin(allowed)]

    df = df[df["最終精查分數"].fillna(0) >= int(min_score)]
    df = df.sort_values(
        "最終精查分數",
        ascending=False,
    ).head(int(top_n))

    return df.reset_index(drop=True)


# =============================================================================
# Trade-plan / R helpers
# =============================================================================

def make_plan(symbol, direction, direction_score, m5, trigger_score):
    if direction == "LONG":
        score = {
            "long_score": max(float(direction_score), float(trigger_score)),
            "short_score": 0,
            "bias": "LONG",
        }
    else:
        score = {
            "long_score": 0,
            "short_score": max(float(direction_score), float(trigger_score)),
            "bias": "SHORT",
        }

    return build_trade_plan(
        symbol,
        m5,
        score,
        min_score=int(trigger_score),
    )


def risk_per_unit(row):
    entry = safe_float(row.get("進場價"))
    stop = safe_float(row.get("原始停損", row.get("停損")))

    if not math.isfinite(entry) or not math.isfinite(stop):
        return float("nan")

    risk = abs(entry - stop)
    return risk if risk > 0 else float("nan")


def calc_r(direction, entry, original_stop, exit_price):
    entry = safe_float(entry)
    original_stop = safe_float(original_stop)
    exit_price = safe_float(exit_price)

    risk = abs(entry - original_stop)

    if (
        not math.isfinite(entry)
        or not math.isfinite(original_stop)
        or not math.isfinite(exit_price)
        or risk <= 0
    ):
        return float("nan")

    if direction == "多頭":
        return (exit_price - entry) / risk

    if direction == "空頭":
        return (entry - exit_price) / risk

    return float("nan")


def closed_result_r(status, row, exit_price):
    """
    固定策略終點：
    - 停損 = -1R
    - 保本 = 0R
    - TP2 = 依實際 TP2 / 原始風險計算，通常約 +2.5R
    - 結構反轉 = 按實際出場價計算
    """
    if status == "停損出場":
        return -1.0
    if status == "保本出場":
        return 0.0

    return calc_r(
        row.get("方向"),
        row.get("進場價"),
        row.get("原始停損", row.get("停損")),
        exit_price,
    )



# =============================================================================
# V0.5.4 realistic position sizing / trading costs
# =============================================================================

def apply_entry_slippage(direction, raw_entry, slippage_bps):
    raw_entry = safe_float(raw_entry)
    if not math.isfinite(raw_entry):
        return raw_entry
    slip = float(slippage_bps) / 10000.0
    if direction == "多頭":
        return raw_entry * (1.0 + slip)
    if direction == "空頭":
        return raw_entry * (1.0 - slip)
    return raw_entry


def apply_exit_slippage(direction, raw_exit, slippage_bps):
    raw_exit = safe_float(raw_exit)
    if not math.isfinite(raw_exit):
        return raw_exit
    slip = float(slippage_bps) / 10000.0
    if direction == "多頭":
        return raw_exit * (1.0 - slip)
    if direction == "空頭":
        return raw_exit * (1.0 + slip)
    return raw_exit


def position_metrics(
    virtual_capital,
    risk_pct,
    entry,
    stop,
    fee_rate,
    leverage,
):
    entry = safe_float(entry)
    stop = safe_float(stop)

    if (
        not math.isfinite(entry)
        or not math.isfinite(stop)
        or entry <= 0
    ):
        return {}

    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return {}

    capital = max(0.0, float(virtual_capital))
    risk_budget = capital * float(risk_pct) / 100.0

    qty_by_risk = risk_budget / risk_per_unit
    max_notional = capital * max(1.0, float(leverage))
    qty_by_leverage = max_notional / entry

    qty = min(qty_by_risk, qty_by_leverage)
    notional = qty * entry
    effective_risk = qty * risk_per_unit
    est_entry_fee = notional * float(fee_rate)

    return {
        "虛擬本金": capital,
        "每筆風險%": float(risk_pct),
        "風險預算USDT": risk_budget,
        "槓桿倍數": float(leverage),
        "虛擬數量": qty,
        "名目部位USDT": notional,
        "原始1R_USDT": effective_risk,
        "預估進場手續費USDT": est_entry_fee,
    }


def calc_trade_pnl(
    row,
    raw_exit_price,
    fee_rate,
    slippage_bps,
):
    direction = str(row.get("方向", ""))
    entry_exec = safe_float(
        row.get("成交進場價", row.get("進場價"))
    )
    qty = safe_float(row.get("虛擬數量"), 0.0)
    original_r = safe_float(row.get("原始1R_USDT"))

    exit_exec = apply_exit_slippage(
        direction,
        raw_exit_price,
        slippage_bps,
    )

    if not all(
        math.isfinite(x)
        for x in [entry_exec, exit_exec, qty]
    ):
        return {}

    if direction == "多頭":
        gross = (exit_exec - entry_exec) * qty
    elif direction == "空頭":
        gross = (entry_exec - exit_exec) * qty
    else:
        gross = 0.0

    entry_notional = abs(entry_exec * qty)
    exit_notional = abs(exit_exec * qty)
    entry_fee = entry_notional * float(fee_rate)
    exit_fee = exit_notional * float(fee_rate)
    total_fee = entry_fee + exit_fee
    net = gross - total_fee

    net_r = (
        net / original_r
        if math.isfinite(original_r) and original_r > 0
        else float("nan")
    )

    return {
        "成交出場價": exit_exec,
        "毛損益USDT": gross,
        "進場手續費USDT": entry_fee,
        "出場手續費USDT": exit_fee,
        "總手續費USDT": total_fee,
        "淨損益USDT": net,
        "淨R": net_r,
    }


def history_equity_stats(history, starting_capital):
    capital = float(starting_capital)
    equity = capital

    closed = []
    for row in history:
        pnl = safe_float(row.get("淨損益USDT"))
        if math.isfinite(pnl):
            closed.append((row, pnl))

    closed.sort(
        key=lambda x: str(x[0].get("出場時間UTC", ""))
    )

    peak = equity
    max_dd = 0.0

    for _, pnl in closed:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)

    return {
        "期初本金": capital,
        "目前權益": equity,
        "累積淨損益": equity - capital,
        "報酬率%": (
            (equity / capital - 1.0) * 100.0
            if capital > 0
            else 0.0
        ),
        "最大回撤%": max_dd,
    }



def backfill_legacy_position_costs(
    row,
    virtual_capital,
    risk_pct,
    fee_rate,
    slippage_bps,
    leverage,
):
    """
    V0.5.4.1：
    將 V0.5.3 以前已存在的虛擬持倉補上真實化績效欄位。
    不改動原 Entry / Stop / TP，只補算成交價、部位大小與費用基礎。
    """
    if not row:
        return row

    out = dict(row)

    direction = str(out.get("方向", ""))
    raw_entry = safe_float(out.get("進場價"))
    stop = safe_float(out.get("原始停損", out.get("停損")))

    if not math.isfinite(raw_entry) or not math.isfinite(stop):
        return out

    # 已有成交進場價就沿用，避免每次刷新因設定改變而重算。
    exec_entry = safe_float(out.get("成交進場價"))
    if not math.isfinite(exec_entry):
        exec_entry = apply_entry_slippage(
            direction,
            raw_entry,
            slippage_bps,
        )
        out["成交進場價"] = exec_entry

    # 已有部位大小就不回溯改變。
    qty = safe_float(out.get("虛擬數量"))
    if not math.isfinite(qty) or qty <= 0:
        sizing = position_metrics(
            virtual_capital=virtual_capital,
            risk_pct=risk_pct,
            entry=exec_entry,
            stop=stop,
            fee_rate=fee_rate,
            leverage=leverage,
        )
        out.update(sizing)

    out.setdefault("虛擬本金", float(virtual_capital))
    out.setdefault("每筆風險%", float(risk_pct))
    out.setdefault("槓桿倍數", float(leverage))
    out.setdefault("費率%", float(fee_rate) * 100.0)
    out.setdefault("滑價bps", float(slippage_bps))
    out.setdefault("原始停損", stop)

    return out


# =============================================================================
# Existing virtual position lifecycle
# =============================================================================

def evaluate_open_trade(prev, last_bar, smc, fee_rate, slippage_bps):
    """
    Returns:
      None -> no active virtual position
      dict -> updated active/closed state
    """
    status = str(prev.get("監控狀態", ""))

    if status not in OPEN_TRADE_STATUSES:
        return None

    direction = str(prev.get("方向", ""))
    entry = safe_float(prev.get("進場價"))
    original_stop = safe_float(prev.get("原始停損", prev.get("停損")))
    current_stop = safe_float(prev.get("停損"))
    tp1 = safe_float(prev.get("TP1"))
    tp2 = safe_float(prev.get("TP2"))

    if not all(
        math.isfinite(x)
        for x in [entry, original_stop, current_stop, tp1, tp2]
    ):
        return None

    high = safe_float(last_bar.get("high"))
    low = safe_float(last_bar.get("low"))
    close = safe_float(last_bar.get("close"))

    if not all(math.isfinite(x) for x in [high, low, close]):
        return None

    out = dict(prev)

    out["目前價"] = close
    out["5分K方向"] = smc_bias_zh(smc.get("bias"))
    out["5分K最新結構"] = current_structure_text(smc)
    out["最後更新UTC"] = now_utc_text()
    out["最後更新台灣"] = now_tw_text()

    tp1_done = bool(prev.get("TP1已達", False))

    # TP1 後停損移到 Entry。
    effective_stop = entry if tp1_done else original_stop
    out["停損"] = effective_stop

    # 保守處理：同一根 K 同時碰停損與 TP，先視為停損 / 保本。
    if direction == "多頭":
        if low <= effective_stop:
            final_status = "保本出場" if tp1_done else "停損出場"
            exit_price = effective_stop

        elif high >= tp2:
            final_status = "TP2 完成"
            exit_price = tp2

        elif high >= tp1:
            out["監控狀態"] = "TP1 已達・保本中"
            out["TP1已達"] = True
            out["TP1時間UTC"] = (
                out.get("TP1時間UTC")
                or now_utc_text()
            )
            out["停損"] = entry
            out["提示"] = "TP1 已達，虛擬停損已移至進場價保本。"
            out["持倉分鐘"] = elapsed_minutes(out.get("進場時間UTC"))
            return out

        else:
            # 已進場後，若 5m SMC 出現明確強反向，執行結構反轉出場。
            if smc.get("bias") == "SHORT":
                final_status = "結構反轉出場"
                exit_price = close
            else:
                out["監控狀態"] = "虛擬持倉監控"
                out["提示"] = "虛擬持倉中，持續監控 5分K。"
                out["持倉分鐘"] = elapsed_minutes(out.get("進場時間UTC"))
                return out

    elif direction == "空頭":
        if high >= effective_stop:
            final_status = "保本出場" if tp1_done else "停損出場"
            exit_price = effective_stop

        elif low <= tp2:
            final_status = "TP2 完成"
            exit_price = tp2

        elif low <= tp1:
            out["監控狀態"] = "TP1 已達・保本中"
            out["TP1已達"] = True
            out["TP1時間UTC"] = (
                out.get("TP1時間UTC")
                or now_utc_text()
            )
            out["停損"] = entry
            out["提示"] = "TP1 已達，虛擬停損已移至進場價保本。"
            out["持倉分鐘"] = elapsed_minutes(out.get("進場時間UTC"))
            return out

        else:
            if smc.get("bias") == "LONG":
                final_status = "結構反轉出場"
                exit_price = close
            else:
                out["監控狀態"] = "虛擬持倉監控"
                out["提示"] = "虛擬持倉中，持續監控 5分K。"
                out["持倉分鐘"] = elapsed_minutes(out.get("進場時間UTC"))
                return out

    else:
        return None

    # Closed trade
    out["監控狀態"] = final_status
    out["出場價"] = exit_price
    out["出場時間UTC"] = now_utc_text()
    out["出場時間台灣"] = now_tw_text()
    out["持倉分鐘"] = elapsed_minutes(
        out.get("進場時間UTC"),
        out.get("出場時間UTC"),
    )

    result_r = closed_result_r(final_status, out, exit_price)
    out["結果R"] = round(result_r, 3) if math.isfinite(result_r) else None

    pnl = calc_trade_pnl(
        out,
        raw_exit_price=exit_price,
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
    )
    out.update(pnl)

    net_r = safe_float(out.get("淨R"))
    if math.isfinite(net_r):
        out["淨R"] = round(net_r, 3)

    for key in [
        "毛損益USDT",
        "進場手續費USDT",
        "出場手續費USDT",
        "總手續費USDT",
        "淨損益USDT",
        "成交出場價",
    ]:
        value = safe_float(out.get(key))
        if math.isfinite(value):
            out[key] = round(value, 6 if key == "成交出場價" else 4)

    if final_status == "TP2 完成":
        out["提示"] = f"TP2 完成，本筆虛擬交易結果 {out.get('結果R')}R。"
    elif final_status == "停損出場":
        out["提示"] = "停損出場，本筆虛擬交易 -1R。"
    elif final_status == "保本出場":
        out["提示"] = "TP1 後保本出場，本筆虛擬交易 0R。"
    else:
        out["提示"] = (
            f"5分K 結構反轉出場，本筆虛擬交易結果 "
            f"{out.get('結果R')}R。"
        )

    return out


# =============================================================================
# New / waiting signal evaluation
# =============================================================================

def analyze_candidate(
    precision_row,
    prev,
    locked_structure,
    trigger_score,
    chase_atr,
    virtual_capital,
    risk_pct,
    fee_rate,
    slippage_bps,
    leverage,
):
    symbol = str(precision_row["交易對"]).upper()
    direction = direction_code(precision_row)

    if direction == "NEUTRAL":
        raise RuntimeError("精查方向中性，略過。")

    m5 = client.kline(symbol, "5", 180)
    if len(m5) < 50:
        raise RuntimeError("5分K資料不足")

    x = prepare_indicators(m5)
    smc = analyze_smc(m5)

    last = x.iloc[-1]
    current = safe_float(last.get("close"))
    atr = safe_float(last.get("ATR14"))
    signature = structure_signature(smc)

    # Existing open virtual trade has priority.
    # V0.5.4.1 先把 V0.5.3 舊持倉補上部位 / 成本欄位。
    if prev:
        prev = backfill_legacy_position_costs(
            prev,
            virtual_capital=virtual_capital,
            risk_pct=risk_pct,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            leverage=leverage,
        )

        lifecycle = evaluate_open_trade(
            prev,
            last,
            smc,
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
        )
        if lifecycle:
            return lifecycle, signature

    base_score = safe_float(
        precision_row.get("原策略方向分數"),
        0.0,
    )

    plan = make_plan(
        symbol,
        direction,
        base_score,
        m5,
        trigger_score,
    )

    expected = "LONG_TRIGGER" if direction == "LONG" else "SHORT_TRIGGER"

    # Same structure was already traded/closed.
    if locked_structure and signature == locked_structure:
        status = "等待新結構"

    # Structure invalidated before entry.
    elif direction == "LONG" and smc.get("bias") == "SHORT":
        status = "訊號失效"

    elif direction == "SHORT" and smc.get("bias") == "LONG":
        status = "訊號失效"

    elif str(plan.get("status")) == expected:
        entry = safe_float(plan.get("entry"))
        too_far = False

        if math.isfinite(entry) and math.isfinite(atr):
            if direction == "LONG":
                too_far = current > entry + atr * float(chase_atr)
            else:
                too_far = current < entry - atr * float(chase_atr)

        if too_far:
            status = "禁止追價"
        else:
            status = (
                "多頭訊號成立"
                if direction == "LONG"
                else "空頭訊號成立"
            )

    else:
        status = (
            "等待多頭觸發"
            if direction == "LONG"
            else "等待空頭觸發"
        )

    entry = safe_float(plan.get("entry"))
    stop = safe_float(plan.get("stop"))
    tp1 = safe_float(plan.get("tp1"))
    tp2 = safe_float(plan.get("tp2"))

    direction_zh = "多頭" if direction == "LONG" else "空頭"
    exec_entry = apply_entry_slippage(
        direction_zh,
        entry,
        slippage_bps,
    )
    sizing = position_metrics(
        virtual_capital=virtual_capital,
        risk_pct=risk_pct,
        entry=exec_entry,
        stop=stop,
        fee_rate=fee_rate,
        leverage=leverage,
    )

    is_new_entry = status in {
        "多頭訊號成立",
        "空頭訊號成立",
    }

    old = prev or {}

    out = {
        "交易對": symbol,
        "方向": direction_zh,
        "精查分數": int(
            safe_float(
                precision_row.get("最終精查分數"),
                0,
            )
        ),
        "目前價": current,
        "5分K方向": smc_bias_zh(smc.get("bias")),
        "5分K最新結構": current_structure_text(smc),
        "結構指紋": signature,
        "監控狀態": status,
        "進場價": entry,
        "成交進場價": exec_entry,
        "原始停損": stop,
        "停損": stop,
        "TP1": tp1,
        "TP2": tp2,
        "TP1已達": False,
        "訊號成立時間UTC": (
            now_utc_text()
            if is_new_entry
            else old.get("訊號成立時間UTC")
        ),
        "進場時間UTC": (
            now_utc_text()
            if is_new_entry
            else old.get("進場時間UTC")
        ),
        "進場時間台灣": (
            now_tw_text()
            if is_new_entry
            else old.get("進場時間台灣")
        ),
        "最後更新UTC": now_utc_text(),
        "最後更新台灣": now_tw_text(),
        "持倉分鐘": (
            0
            if is_new_entry
            else old.get("持倉分鐘")
        ),
        "結果R": None,
        "提示": (
            "策略虛擬進場訊號成立；這不是 Bybit 真實下單。"
            if is_new_entry
            else "同一 BOS / CHoCH 已交易過，等待新的市場結構。"
            if status == "等待新結構"
            else "價格已離觸發點過遠，不追價。"
            if status == "禁止追價"
            else "SMC 方向反轉，候選暫時失效。"
            if status == "訊號失效"
            else "等待下一根 5分K 確認。"
        ),
    }

    if is_new_entry:
        out.update(sizing)
        out["費率%"] = float(fee_rate) * 100.0
        out["滑價bps"] = float(slippage_bps)
    elif prev:
        for key in [
            "虛擬本金",
            "每筆風險%",
            "風險預算USDT",
            "槓桿倍數",
            "虛擬數量",
            "名目部位USDT",
            "原始1R_USDT",
            "預估進場手續費USDT",
            "費率%",
            "滑價bps",
            "成交進場價",
        ]:
            if key in prev:
                out[key] = prev.get(key)

    return out, signature


# =============================================================================
# History helpers
# =============================================================================

def history_identity(row):
    return (
        str(row.get("交易對", "")),
        str(row.get("進場時間UTC", "")),
        str(row.get("出場時間UTC", "")),
        str(row.get("監控狀態", "")),
    )


def append_history_unique(history, closed_row):
    identity = history_identity(closed_row)
    known = {history_identity(r) for r in history}

    if identity not in known:
        history.append(dict(closed_row))

    return history


def performance_stats(history):
    trades = []
    for row in history:
        r = safe_float(row.get("結果R"))
        if math.isfinite(r):
            trades.append(r)

    if not trades:
        return {
            "交易數": 0,
            "勝率": 0.0,
            "總R": 0.0,
            "平均R": 0.0,
            "獲利筆": 0,
            "虧損筆": 0,
            "保本筆": 0,
            "ProfitFactor": None,
        }

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    breakeven = [x for x in trades if x == 0]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    pf = (
        gross_win / gross_loss
        if gross_loss > 0
        else None
    )

    return {
        "交易數": len(trades),
        "勝率": len(wins) / len(trades) * 100,
        "總R": sum(trades),
        "平均R": sum(trades) / len(trades),
        "獲利筆": len(wins),
        "虧損筆": len(losses),
        "保本筆": len(breakeven),
        "ProfitFactor": pf,
    }


# =============================================================================
# Connection / source data
# =============================================================================

status = cloud.status()

if status.get("ok"):
    st.success(
        f"☁️ {status['message']}｜"
        "虛擬持倉與歷史績效會自動保存"
    )
else:
    st.error(
        f"☁️ {status['message']}。"
        "請先完成 GitHub 私有儲存設定。"
    )
    st.stop()

try:
    precision_rows, source_saved_at = load_precision_rows()
except Exception as exc:
    st.error(f"讀取第二階段結果失敗：{exc}")
    st.stop()

if not precision_rows:
    st.info(
        "GitHub 尚無第二階段精查結果。"
        "請先到「多空掃描」完成第二階段精查。"
    )
    st.stop()

st.caption(
    f"第二階段資料時間：{source_saved_at or '-'} ｜ "
    "本頁所有持倉皆為虛擬策略紀錄"
)


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.markdown("### 監控設定")

mode = st.sidebar.radio(
    "候選範圍",
    ["只監控可進場／等待觸發", "包含保留觀察"],
    index=0,
)

min_score = st.sidebar.slider(
    "最低精查分數",
    50,
    95,
    65,
    5,
)

top_n = st.sidebar.number_input(
    "最多監控檔數",
    5,
    50,
    20,
    5,
)

trigger_score = st.sidebar.slider(
    "5分K觸發最低分",
    55,
    100,
    70,
    5,
)

chase_atr = st.sidebar.slider(
    "禁止追價距離（ATR倍數）",
    0.25,
    2.00,
    0.75,
    0.25,
)

auto_refresh = st.sidebar.toggle(
    "每 5 分鐘自動更新",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 真實化績效設定")

virtual_capital = st.sidebar.number_input(
    "虛擬本金（USDT）",
    min_value=100.0,
    max_value=10000000.0,
    value=10000.0,
    step=1000.0,
)

risk_pct = st.sidebar.slider(
    "每筆風險（%本金）",
    min_value=0.25,
    max_value=5.00,
    value=1.00,
    step=0.25,
)

leverage = st.sidebar.number_input(
    "部位槓桿上限",
    min_value=1.0,
    max_value=20.0,
    value=3.0,
    step=1.0,
)

fee_pct = st.sidebar.number_input(
    "單邊手續費（%）",
    min_value=0.0,
    max_value=0.20,
    value=0.055,
    step=0.005,
    format="%.3f",
    help="預設 0.055% 作保守模擬，可依你的 Bybit 實際費率調整。",
)

slippage_bps = st.sidebar.number_input(
    "單邊滑價（bps）",
    min_value=0.0,
    max_value=30.0,
    value=2.0,
    step=0.5,
    format="%.1f",
    help="1 bps = 0.01%，預設 2 bps = 0.02%。",
)

fee_rate = float(fee_pct) / 100.0

manual_refresh = st.sidebar.button(
    "立即重新檢查",
    type="primary",
    use_container_width=True,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "虛擬交易規則：TP1 後停損移至 Entry；"
    "停損後同一 BOS/CHoCH 不重複進場，必須等待新結構。"
)


# =============================================================================
# Candidate list
# =============================================================================

candidates = candidate_rows(
    precision_rows,
    mode,
    min_score,
    top_n,
)

if candidates.empty:
    st.warning("目前沒有符合監控條件的第二階段候選。")
    st.stop()

st.metric("目前候選來源", len(candidates))


# =============================================================================
# Main monitoring execution
# =============================================================================

def render_monitor():
    previous_map, locks = load_monitor_payload()
    history = load_history_rows()

    current_rows = []
    invalid_rows = []
    errors = []

    progress = st.progress(0.0)
    live = st.empty()

    for i, (_, precision_row) in enumerate(candidates.iterrows()):
        symbol = str(precision_row["交易對"]).upper()
        live.info(
            f"正在更新 {i + 1}/{len(candidates)}｜{symbol}"
        )

        try:
            result, latest_signature = analyze_candidate(
                precision_row.to_dict(),
                previous_map.get(symbol),
                locks.get(symbol),
                trigger_score,
                chase_atr,
                virtual_capital,
                risk_pct,
                fee_rate,
                slippage_bps,
                leverage,
            )

            status_text = str(result.get("監控狀態", ""))

            if status_text in CLOSED_TRADE_STATUSES:
                # Closed trade goes to history, not current monitor table.
                history = append_history_unique(
                    history,
                    result,
                )

                # Lock the structure that produced this trade.
                locks[symbol] = str(
                    result.get("結構指紋")
                    or latest_signature
                )

                # Keep a waiting-new-structure row for the next cycle,
                # but do not mix the closed trade into active positions.
                follow = dict(result)
                follow["監控狀態"] = "等待新結構"
                follow["結果R"] = None
                follow["出場價"] = None
                follow["出場時間UTC"] = None
                follow["出場時間台灣"] = None
                follow["提示"] = (
                    "上一筆虛擬交易已結束；"
                    "等待新的 BOS / CHoCH 後才允許再次進場。"
                )
                current_rows.append(follow)

            elif status_text in NON_TRADE_TERMINAL:
                invalid_rows.append(result)
                current_rows.append(result)

            else:
                current_rows.append(result)

        except Exception as exc:
            errors.append({
                "交易對": symbol,
                "錯誤": str(exc),
            })

        progress.progress((i + 1) / len(candidates))

    live.empty()

    save_monitor_payload(current_rows, locks)
    save_history_rows(history)

    current_df = pd.DataFrame(current_rows)
    history_df = pd.DataFrame(history)

    if current_df.empty:
        st.error("本輪沒有成功取得監控結果。")
        return

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    signal_count = int(
        current_df["監控狀態"].isin(
            ["多頭訊號成立", "空頭訊號成立"]
        ).sum()
    )

    waiting_count = int(
        current_df["監控狀態"].isin(WAIT_STATUSES).sum()
    )

    virtual_position_count = int(
        current_df["監控狀態"].isin(
            ["虛擬持倉監控", "TP1 已達・保本中"]
        ).sum()
    )

    invalid_count = int(
        current_df["監控狀態"].isin(NON_TRADE_TERMINAL).sum()
    )

    a, b, c, d = st.columns(4)
    a.metric("新虛擬進場訊號", signal_count)
    b.metric("等待觸發／新結構", waiting_count)
    c.metric("虛擬持倉／TP1", virtual_position_count)
    d.metric("禁止追價／失效", invalid_count)

    # -----------------------------------------------------------------
    # Performance
    # -----------------------------------------------------------------

    stats = performance_stats(history)
    equity_stats = history_equity_stats(
        history,
        virtual_capital,
    )

    active_positions = current_df[
        current_df["監控狀態"].isin(
            ["多頭訊號成立", "空頭訊號成立", "虛擬持倉監控", "TP1 已達・保本中"]
        )
    ].copy()

    if not active_positions.empty:
        active_notional = pd.to_numeric(
            active_positions.get("名目部位USDT"),
            errors="coerce",
        ).fillna(0).sum()
        active_risk = pd.to_numeric(
            active_positions.get("原始1R_USDT"),
            errors="coerce",
        ).fillna(0).sum()

        ps1, ps2, ps3 = st.columns(3)
        ps1.metric("目前虛擬持倉數", len(active_positions))
        ps2.metric("目前名目部位", f"{active_notional:,.2f} USDT")
        ps3.metric("目前總風險", f"{active_risk:,.2f} USDT")

    st.markdown("### 虛擬策略績效")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("已平倉交易", stats["交易數"])
    s2.metric("勝率", f'{stats["勝率"]:.1f}%')
    s3.metric("總 R", f'{stats["總R"]:+.2f}R')
    s4.metric("平均 R", f'{stats["平均R"]:+.2f}R')
    s5.metric(
        "Profit Factor",
        (
            f'{stats["ProfitFactor"]:.2f}'
            if stats["ProfitFactor"] is not None
            else "-"
        ),
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric(
        "目前模擬權益",
        f'{equity_stats["目前權益"]:,.2f} USDT',
    )
    e2.metric(
        "累積淨損益",
        f'{equity_stats["累積淨損益"]:+,.2f} USDT',
    )
    e3.metric(
        "淨報酬率",
        f'{equity_stats["報酬率%"]:+.2f}%',
    )
    e4.metric(
        "最大回撤",
        f'{equity_stats["最大回撤%"]:.2f}%',
    )

    st.caption(
        f'獲利 {stats["獲利筆"]} 筆 ｜ '
        f'虧損 {stats["虧損筆"]} 筆 ｜ '
        f'保本 {stats["保本筆"]} 筆。'
        "毛 R 以原始 Entry→Stop 風險距離為 1R；"
        "V0.5.4 另計手續費、滑價、淨R與淨損益 USDT。"
    )

    # -----------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------

    tab_current, tab_history, tab_invalid, tab_errors = st.tabs(
        ["目前監控", "已平倉紀錄", "失效／禁止追價", "錯誤紀錄"]
    )

    with tab_current:
        priority = {
            "多頭訊號成立": 0,
            "空頭訊號成立": 0,
            "TP1 已達・保本中": 1,
            "虛擬持倉監控": 2,
            "等待多頭觸發": 3,
            "等待空頭觸發": 3,
            "等待新結構": 4,
            "禁止追價": 5,
            "訊號失效": 6,
        }

        current_df["_p"] = (
            current_df["監控狀態"]
            .map(priority)
            .fillna(9)
        )

        current_df = current_df.sort_values(
            ["_p", "精查分數"],
            ascending=[True, False],
        ).drop(columns=["_p"])

        current_cols = [
            "交易對",
            "方向",
            "精查分數",
            "目前價",
            "5分K方向",
            "5分K最新結構",
            "監控狀態",
            "進場價",
            "成交進場價",
            "停損",
            "TP1",
            "TP2",
            "虛擬數量",
            "名目部位USDT",
            "風險預算USDT",
            "原始1R_USDT",
            "進場時間台灣",
            "持倉分鐘",
            "最後更新台灣",
            "提示",
        ]
        current_cols = [
            c for c in current_cols
            if c in current_df.columns
        ]

        st.dataframe(
            current_df[current_cols],
            use_container_width=True,
            hide_index=True,
        )

    with tab_history:
        if history_df.empty:
            st.info("尚無已平倉虛擬交易。")
        else:
            # Newest closed trade first.
            if "出場時間UTC" in history_df.columns:
                history_df = history_df.sort_values(
                    "出場時間UTC",
                    ascending=False,
                    na_position="last",
                )

            history_cols = [
                "交易對",
                "方向",
                "精查分數",
                "監控狀態",
                "進場價",
                "成交進場價",
                "原始停損",
                "TP1",
                "TP2",
                "出場價",
                "成交出場價",
                "虛擬數量",
                "名目部位USDT",
                "原始1R_USDT",
                "結果R",
                "淨R",
                "毛損益USDT",
                "總手續費USDT",
                "淨損益USDT",
                "進場時間台灣",
                "出場時間台灣",
                "持倉分鐘",
                "5分K最新結構",
            ]
            history_cols = [
                c for c in history_cols
                if c in history_df.columns
            ]

            st.dataframe(
                history_df[history_cols],
                use_container_width=True,
                hide_index=True,
            )

            history_csv = history_df.to_csv(
                index=False,
            ).encode("utf-8-sig")

            st.download_button(
                "下載虛擬交易歷史 CSV",
                data=history_csv,
                file_name="virtual_trade_history_v053.csv",
                mime="text/csv",
            )

    with tab_invalid:
        invalid_df = current_df[
            current_df["監控狀態"].isin(
                NON_TRADE_TERMINAL
            )
        ].copy()

        if invalid_df.empty:
            st.success("目前沒有禁止追價或失效候選。")
        else:
            invalid_cols = [
                "交易對",
                "方向",
                "精查分數",
                "目前價",
                "5分K方向",
                "5分K最新結構",
                "監控狀態",
                "提示",
                "最後更新台灣",
            ]
            invalid_cols = [
                c for c in invalid_cols
                if c in invalid_df.columns
            ]

            st.dataframe(
                invalid_df[invalid_cols],
                use_container_width=True,
                hide_index=True,
            )

    with tab_errors:
        if not errors:
            st.success("本輪沒有監控錯誤。")
        else:
            st.dataframe(
                pd.DataFrame(errors),
                use_container_width=True,
                hide_index=True,
            )

    st.success(
        f"✅ 本輪 5分K 虛擬持倉監控完成："
        f"{len(current_rows)} 檔｜"
        f"歷史平倉 {len(history)} 筆。"
        "目前監控與歷史績效都已保存至 GitHub Private Repo。"
    )


# =============================================================================
# Auto refresh
# =============================================================================

@st.fragment(run_every="5m")
def auto_monitor_fragment():
    if auto_refresh:
        render_monitor()

    elif manual_refresh:
        render_monitor()

    else:
        previous_map, _ = load_monitor_payload()
        history = load_history_rows()

        st.info(
            "自動更新已關閉，目前顯示 GitHub 上一次保存的結果。"
        )

        if previous_map:
            st.dataframe(
                pd.DataFrame(previous_map.values()),
                use_container_width=True,
                hide_index=True,
            )

        stats = performance_stats(history)
        st.caption(
            f'歷史平倉 {stats["交易數"]} 筆 ｜ '
            f'勝率 {stats["勝率"]:.1f}% ｜ '
            f'總R {stats["總R"]:+.2f}R'
        )


auto_monitor_fragment()
