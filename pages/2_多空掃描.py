# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from bybit_client import BybitClient
from smc_engine import analyze_smc
from score_engine import score_symbol
from strategy_engine import build_trade_plan


# =============================================================================
# Page
# =============================================================================

st.set_page_config(page_title="多空掃描 V0.4", layout="wide")
st.title("全市場多空掃描 V0.4｜SMC 二階段精查")
st.caption(
    "預設掃描全部 Bybit USDT 永續，不綁成交量排名。"
    "建立清單後按一次「開始／繼續自動掃描」，系統會自動分批往下掃到全部完成；"
    "可隨時暫停，之後從原進度繼續。第一階段完成後，再針對多空就緒與高分觀察標的進行 OI、Funding、Long/Short Ratio、成交量與 5分K 進場精查。"
)


# =============================================================================
# Display translations
# =============================================================================

LONG_SET = {"LONG", "LEAN_LONG"}
SHORT_SET = {"SHORT", "LEAN_SHORT"}

STATUS_ZH = {
    "LONG_READY": "多頭就緒",
    "SHORT_READY": "空頭就緒",
    "WATCH_LONG": "多頭觀察",
    "WATCH_SHORT": "空頭觀察",
    "NEUTRAL": "中性觀望",
}

BIAS_ZH = {
    "LONG": "多頭",
    "LEAN_LONG": "偏多",
    "SHORT": "空頭",
    "LEAN_SHORT": "偏空",
    "NEUTRAL": "中性",
    "bullish": "多頭",
    "bearish": "空頭",
    "neutral": "中性",
}

ZONE_ZH = {
    "PREMIUM": "溢價區",
    "DISCOUNT": "折價區",
    "EQUILIBRIUM": "均衡區",
    "premium": "溢價區",
    "discount": "折價區",
    "equilibrium": "均衡區",
}


def status_zh(value):
    return STATUS_ZH.get(str(value), str(value))


def bias_zh(value):
    return BIAS_ZH.get(str(value), str(value))


def zone_zh(value):
    return ZONE_ZH.get(str(value), str(value))


# =============================================================================
# Helpers
# =============================================================================

def fmt_price(value):
    try:
        value = float(value)
        if not math.isfinite(value):
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:,.4f}"
        return f"{value:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


def fmt_duration(seconds):
    try:
        seconds = max(0, int(seconds))
    except Exception:
        return "-"

    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)

    if hours:
        return f"{hours}時 {minutes}分 {secs}秒"
    if minutes:
        return f"{minutes}分 {secs}秒"
    return f"{secs}秒"


def mtf_score(s5, s15, s60):
    """
    與單幣分析相同的 MTF 權重：
      5分K = 55%
      15分K = 30%
      1H = 15%
    """
    long_score = round(
        float(s5["long_score"]) * 0.55
        + float(s15["long_score"]) * 0.30
        + float(s60["long_score"]) * 0.15
    )
    short_score = round(
        float(s5["short_score"]) * 0.55
        + float(s15["short_score"]) * 0.30
        + float(s60["short_score"]) * 0.15
    )

    diff = long_score - short_score

    if long_score >= 70 and diff >= 10:
        bias = "LONG"
    elif short_score >= 70 and diff <= -10:
        bias = "SHORT"
    elif long_score >= 55 and diff > 5:
        bias = "LEAN_LONG"
    elif short_score >= 55 and diff < -5:
        bias = "LEAN_SHORT"
    else:
        bias = "NEUTRAL"

    return int(long_score), int(short_score), bias


def readiness(mtf_bias, s5, s15, s60):
    """
    多頭就緒：
      MTF 已偏多
      5分K 同向
      15分K 不反向
      1H 不是強空

    空頭就緒：
      MTF 已偏空
      5分K 同向
      15分K 不反向
      1H 不是強多

    其餘方向性訊號進入「觀察」。
    """
    b5 = s5["bias"]
    b15 = s15["bias"]
    b60 = s60["bias"]

    if mtf_bias in LONG_SET:
        if b5 in LONG_SET and b15 not in SHORT_SET and b60 != "SHORT":
            return "LONG_READY"
        return "WATCH_LONG"

    if mtf_bias in SHORT_SET:
        if b5 in SHORT_SET and b15 not in LONG_SET and b60 != "LONG":
            return "SHORT_READY"
        return "WATCH_SHORT"

    return "NEUTRAL"


def structure_text_zh(smc):
    event = smc.get("latest_structure")
    if not event:
        return "-"

    side = bias_zh(event.get("side", "-"))
    event_type = str(event.get("type", "-"))
    level = fmt_price(event.get("level"))
    return f"{side} {event_type} @{level}"


def zone_text_zh(smc):
    dealing_range = smc.get("dealing_range", {}) or {}
    return zone_zh(dealing_range.get("zone", "-"))


def build_universe(client: BybitClient, mode: str, quick_n: int) -> pd.DataFrame:
    """
    全市場：
      - 全部 USDT 永續
      - 不綁成交量排名
      - 不設總候選上限
      - 依 Symbol 固定排序，方便辨識進度

    快速測試：
      - 僅在使用者明確選擇時，依 24H Turnover 取前 N
    """
    instruments = client.instruments_linear_usdt()
    tickers = client.tickers_linear()

    if instruments.empty:
        raise RuntimeError("USDT 永續商品清單為空。")

    universe = (
        instruments[["symbol"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    universe["symbol"] = universe["symbol"].astype(str).str.upper()

    if not tickers.empty and "symbol" in tickers.columns:
        keep_cols = [
            col for col in [
                "symbol",
                "lastPrice",
                "turnover24h",
                "volume24h",
                "price24hPcnt",
                "fundingRate",
                "openInterest",
            ]
            if col in tickers.columns
        ]

        metadata = tickers[keep_cols].copy()
        metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
        universe = universe.merge(metadata, on="symbol", how="left")

    if mode == "快速測試｜流動性前 N":
        if "turnover24h" in universe.columns:
            universe["turnover24h"] = pd.to_numeric(
                universe["turnover24h"], errors="coerce"
            )
            universe = universe.sort_values(
                "turnover24h",
                ascending=False,
                na_position="last",
            ).head(int(quick_n))
        else:
            universe = universe.head(int(quick_n))
    else:
        universe = universe.sort_values("symbol")

    return universe.reset_index(drop=True)


def analyze_symbol(client: BybitClient, symbol: str, metadata: dict) -> dict:
    """
    第一階段全市場掃描只抓：
      1H / 15分K / 5分K

    先用 SMC 做全市場初篩。
    OI / Funding / Long-Short Ratio 留給高分候選精查，
    避免 700+ 檔第一輪產生過多 API 請求。
    """
    h1 = client.kline(symbol, "60", 240)
    m15 = client.kline(symbol, "15", 240)
    m5 = client.kline(symbol, "5", 240)

    if min(len(h1), len(m15), len(m5)) < 50:
        raise RuntimeError(
            f"K線不足：1H={len(h1)}、15分K={len(m15)}、5分K={len(m5)}"
        )

    s60 = analyze_smc(h1)
    s15 = analyze_smc(m15)
    s5 = analyze_smc(m5)

    mtf_long, mtf_short, mtf_bias = mtf_score(s5, s15, s60)
    internal_status = readiness(mtf_bias, s5, s15, s60)

    last_price = s5.get("last_price")
    if last_price is None:
        last_price = metadata.get("lastPrice")

    return {
        "Symbol": symbol,

        # 內部代碼：策略判斷用，不直接顯示給使用者
        "狀態代碼": internal_status,
        "方向代碼": mtf_bias,
        "1H代碼": s60.get("bias", "NEUTRAL"),
        "15m代碼": s15.get("bias", "NEUTRAL"),
        "5m代碼": s5.get("bias", "NEUTRAL"),

        # 中文顯示
        "狀態": status_zh(internal_status),
        "方向": bias_zh(mtf_bias),
        "MTF_LONG": mtf_long,
        "MTF_SHORT": mtf_short,
        "差值": mtf_long - mtf_short,
        "目前價": last_price,
        "1H": bias_zh(s60.get("bias", "NEUTRAL")),
        "15m": bias_zh(s15.get("bias", "NEUTRAL")),
        "5m": bias_zh(s5.get("bias", "NEUTRAL")),
        "1H結構": bias_zh(s60.get("structure_state", "-")),
        "15m結構": bias_zh(s15.get("structure_state", "-")),
        "5m結構": bias_zh(s5.get("structure_state", "-")),
        "5m最新結構": structure_text_zh(s5),
        "5m區域": zone_text_zh(s5),
        "5m信心": s5.get("confidence", 0),
        "Turnover24h": metadata.get("turnover24h"),
        "24h%": metadata.get("price24hPcnt"),
        "掃描時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def analyze_symbol_with_retry(
    client: BybitClient,
    symbol: str,
    metadata: dict,
    max_retries: int,
    retry_delay: float,
):
    """
    暫時性 API / 網路錯誤自動重試。
    max_retries=2 代表最多共嘗試 3 次。
    """
    attempts_total = max(1, int(max_retries) + 1)
    last_error = None

    for attempt in range(1, attempts_total + 1):
        try:
            row = analyze_symbol(client, symbol, metadata)
            return row, attempt, None
        except Exception as exc:
            last_error = exc

            if attempt < attempts_total:
                # 線性退避：0.8s -> 1.6s -> ...
                sleep_seconds = float(retry_delay) * attempt
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    return None, attempts_total, last_error



STATUS_CODE_FROM_ZH = {
    "多頭就緒": "LONG_READY",
    "空頭就緒": "SHORT_READY",
    "多頭觀察": "WATCH_LONG",
    "空頭觀察": "WATCH_SHORT",
    "中性觀望": "NEUTRAL",
}

BIAS_CODE_FROM_ZH = {
    "多頭": "LONG",
    "偏多": "LEAN_LONG",
    "空頭": "SHORT",
    "偏空": "LEAN_SHORT",
    "中性": "NEUTRAL",
}


def safe_float(value, default=float("nan")):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def normalize_stage1_import(df: pd.DataFrame) -> pd.DataFrame:
    """
    支援匯入 V0.3/V0.4 第一階段 CSV。
    這可以在更新 Cloud Run 後直接復用已掃完的 735 檔結果，
    不必因為部署新版而重新花約 10 分鐘做第一階段。
    """
    if df is None or df.empty:
        raise ValueError("CSV 沒有資料。")

    out = df.copy()

    reverse_rename = {
        "交易對": "Symbol",
        "多頭分數": "MTF_LONG",
        "空頭分數": "MTF_SHORT",
        "多空差值": "差值",
        "1H方向": "1H",
        "15分K方向": "15m",
        "5分K方向": "5m",
        "5分K最新結構": "5m最新結構",
        "5分K區域": "5m區域",
        "5分K信心": "5m信心",
        "24H成交額": "Turnover24h",
        "24H漲跌": "24h%",
    }
    out = out.rename(columns={k: v for k, v in reverse_rename.items() if k in out.columns})

    if "Symbol" not in out.columns:
        raise ValueError("CSV 找不到「交易對 / Symbol」欄位。")

    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()

    for col in ["MTF_LONG", "MTF_SHORT", "差值", "目前價", "5m信心", "Turnover24h", "24h%"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "狀態代碼" not in out.columns:
        if "狀態" in out.columns:
            out["狀態代碼"] = out["狀態"].map(STATUS_CODE_FROM_ZH).fillna("NEUTRAL")
        else:
            out["狀態代碼"] = "NEUTRAL"

    if "方向代碼" not in out.columns:
        if "方向" in out.columns:
            out["方向代碼"] = out["方向"].map(BIAS_CODE_FROM_ZH).fillna("NEUTRAL")
        else:
            out["方向代碼"] = "NEUTRAL"

    for display_col, code_col in [("1H", "1H代碼"), ("15m", "15m代碼"), ("5m", "5m代碼")]:
        if code_col not in out.columns:
            if display_col in out.columns:
                out[code_col] = out[display_col].map(BIAS_CODE_FROM_ZH).fillna("NEUTRAL")
            else:
                out[code_col] = "NEUTRAL"

    # 若 CSV 內只有代碼，補回中文顯示。
    if "狀態" not in out.columns:
        out["狀態"] = out["狀態代碼"].map(status_zh)
    if "方向" not in out.columns:
        out["方向"] = out["方向代碼"].map(bias_zh)

    for display_col, code_col in [("1H", "1H代碼"), ("15m", "15m代碼"), ("5m", "5m代碼")]:
        if display_col not in out.columns:
            out[display_col] = out[code_col].map(bias_zh)

    if "差值" not in out.columns and {"MTF_LONG", "MTF_SHORT"}.issubset(out.columns):
        out["差值"] = out["MTF_LONG"] - out["MTF_SHORT"]

    return out.reset_index(drop=True)


def reset_precision():
    keys = [
        "precision_queue",
        "precision_pos",
        "precision_rows",
        "precision_errors",
        "precision_auto",
        "precision_started",
        "precision_finished",
        "precision_seconds_total",
        "precision_processed_total",
        "precision_retry_total",
        "precision_last_batch_count",
        "precision_last_batch_seconds",
        "precision_signature",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def build_precision_queue(
    stage1: pd.DataFrame,
    include_mode: str,
    watch_threshold: int,
) -> pd.DataFrame:
    """
    第二階段候選規則：
    - 多頭就緒 / 空頭就緒：預設全部納入精查。
    - 多頭觀察 / 空頭觀察：方向分數達門檻才納入。
    - 不設硬性候選檔數上限，符合條件者全部精查。
    """
    if stage1.empty:
        return pd.DataFrame()

    x = stage1.copy()

    def directional_score(row):
        code = str(row.get("狀態代碼", "NEUTRAL"))
        if code in {"LONG_READY", "WATCH_LONG"}:
            return safe_float(row.get("MTF_LONG"), 0.0)
        if code in {"SHORT_READY", "WATCH_SHORT"}:
            return safe_float(row.get("MTF_SHORT"), 0.0)
        return max(
            safe_float(row.get("MTF_LONG"), 0.0),
            safe_float(row.get("MTF_SHORT"), 0.0),
        )

    x["精查方向分數"] = x.apply(directional_score, axis=1)

    ready_mask = x["狀態代碼"].isin(["LONG_READY", "SHORT_READY"])
    watch_mask = (
        x["狀態代碼"].isin(["WATCH_LONG", "WATCH_SHORT"])
        & (x["精查方向分數"] >= int(watch_threshold))
    )

    if include_mode == "僅多空就緒":
        selected = ready_mask
    elif include_mode == "僅高分觀察":
        selected = watch_mask
    else:
        selected = ready_mask | watch_mask

    q = x.loc[selected].copy()

    priority = {
        "LONG_READY": 0,
        "SHORT_READY": 0,
        "WATCH_LONG": 1,
        "WATCH_SHORT": 1,
    }
    q["_p"] = q["狀態代碼"].map(priority).fillna(9)
    q = q.sort_values(
        ["_p", "精查方向分數"],
        ascending=[True, False],
    ).drop(columns=["_p"])

    return q.reset_index(drop=True)


def precision_direction(stage1_row: dict) -> str:
    status_code = str(stage1_row.get("狀態代碼", ""))
    if status_code in {"LONG_READY", "WATCH_LONG"}:
        return "LONG"
    if status_code in {"SHORT_READY", "WATCH_SHORT"}:
        return "SHORT"

    bias = str(stage1_row.get("方向代碼", "NEUTRAL"))
    if bias in LONG_SET:
        return "LONG"
    if bias in SHORT_SET:
        return "SHORT"
    return "NEUTRAL"


def derivatives_points(side, oi_change, price_change, funding, buy_ratio, sell_ratio, rel_vol):
    """
    衍生資料滿分 25：
      OI + Price       10
      Account Ratio     5
      Funding           5
      Relative Volume   5
    """
    points = 0
    reasons = []

    # OI + Price
    if side == "LONG":
        if oi_change > 0.002 and price_change > 0:
            points += 10
            reasons.append("價格上漲＋OI增加")
        elif oi_change >= 0:
            points += 5
            reasons.append("OI未明顯流失")
        else:
            reasons.append("OI下降，動能較弱")
    elif side == "SHORT":
        if oi_change > 0.002 and price_change < 0:
            points += 10
            reasons.append("價格下跌＋OI增加")
        elif oi_change >= 0:
            points += 5
            reasons.append("OI未明顯流失")
        else:
            reasons.append("OI下降，動能較弱")

    # Long/Short account ratio
    if math.isfinite(buy_ratio) and math.isfinite(sell_ratio):
        if side == "LONG":
            if 0.50 <= buy_ratio <= 0.62:
                points += 5
                reasons.append("多空帳戶比支持多方")
            elif buy_ratio > 0.70:
                reasons.append("多方帳戶過度擁擠")
            else:
                points += 2
        elif side == "SHORT":
            if 0.38 <= buy_ratio < 0.50:
                points += 5
                reasons.append("多空帳戶比支持空方")
            elif sell_ratio > 0.70:
                reasons.append("空方帳戶過度擁擠")
            else:
                points += 2

    # Funding
    if math.isfinite(funding):
        if -0.001 <= funding <= 0.001:
            points += 5
            reasons.append("Funding 正常")
        elif side == "LONG" and funding < -0.001:
            points += 3
            reasons.append("Funding 偏負，具反向擠空條件")
        elif side == "SHORT" and funding > 0.001:
            points += 3
            reasons.append("Funding 偏正，具反向擠多條件")
        else:
            reasons.append("Funding 同向過度擁擠")

    # Relative volume
    if math.isfinite(rel_vol):
        if rel_vol >= 1.20:
            points += 5
            reasons.append("5分K 相對量放大")
        elif rel_vol >= 0.80:
            points += 2
        else:
            reasons.append("5分K 量能不足")

    return max(0, min(25, int(round(points)))), reasons


def analyze_precision_symbol(
    client: BybitClient,
    stage1_row: dict,
    trigger_score: int,
) -> dict:
    """
    V0.4 第二階段精查：
    1. 重抓最新 1H / 15m / 5m
    2. Open Interest
    3. Funding
    4. Long/Short Account Ratio
    5. score_engine 技術/籌碼分數
    6. 5分K build_trade_plan 進場 / 停損 / TP1 / TP2
    """
    symbol = str(stage1_row.get("Symbol", "")).upper()
    side = precision_direction(stage1_row)

    if side == "NEUTRAL":
        raise RuntimeError("第一階段方向為中性，無法進行方向性精查。")

    h1 = client.kline(symbol, "60", 300)
    m15 = client.kline(symbol, "15", 300)
    m5 = client.kline(symbol, "5", 300)

    oi = client.open_interest(symbol, "5min", 20)
    ratio = client.long_short_ratio(symbol, "5min", 20)
    funding = client.funding_history(symbol, 10)

    base = score_symbol(h1, m15, m5, oi, ratio, funding)
    plan = build_trade_plan(
        symbol,
        m5,
        base,
        min_score=int(trigger_score),
    )

    mtf_direction_score = (
        safe_float(stage1_row.get("MTF_LONG"), 0.0)
        if side == "LONG"
        else safe_float(stage1_row.get("MTF_SHORT"), 0.0)
    )
    base_direction_score = (
        safe_float(base.get("long_score"), 0.0)
        if side == "LONG"
        else safe_float(base.get("short_score"), 0.0)
    )

    oi_change = safe_float(base.get("oi_change"))
    funding_rate = safe_float(base.get("funding_rate"))
    buy_ratio = safe_float(base.get("buy_ratio"))
    sell_ratio = safe_float(base.get("sell_ratio"))
    rel_vol = safe_float(base.get("rel_vol"))
    rsi = safe_float(base.get("rsi14"))
    adx = safe_float(base.get("adx14"))

    price_change = float("nan")
    if len(m5) >= 3:
        old_price = safe_float(m5["close"].iloc[-3])
        new_price = safe_float(m5["close"].iloc[-1])
        if math.isfinite(old_price) and old_price != 0 and math.isfinite(new_price):
            price_change = new_price / old_price - 1.0

    derivative_score, derivative_reasons = derivatives_points(
        side,
        oi_change,
        price_change,
        funding_rate,
        buy_ratio,
        sell_ratio,
        rel_vol,
    )

    # 最終精查分數 0~100：
    # MTF SMC 45% + 原策略方向分數 30% + 衍生資料 25%
    final_score = round(
        0.45 * min(100.0, max(0.0, mtf_direction_score))
        + 0.30 * min(100.0, max(0.0, base_direction_score))
        + derivative_score
    )
    final_score = int(max(0, min(100, final_score)))

    base_bias = str(base.get("bias", "NEUTRAL"))
    plan_status = str(plan.get("status", ""))

    expected_trigger = "LONG_TRIGGER" if side == "LONG" else "SHORT_TRIGGER"
    base_opposite = (
        side == "LONG" and base_bias == "SHORT"
    ) or (
        side == "SHORT" and base_bias == "LONG"
    )

    stage1_status = str(stage1_row.get("狀態代碼", ""))

    if base_opposite:
        decision = "方向衝突"
    elif plan_status == expected_trigger and final_score >= 70 and stage1_status in {"LONG_READY", "SHORT_READY"}:
        decision = "多頭進場條件成立" if side == "LONG" else "空頭進場條件成立"
    elif final_score >= 70:
        decision = "等待多頭進場觸發" if side == "LONG" else "等待空頭進場觸發"
    elif final_score >= 60:
        decision = "保留觀察"
    else:
        decision = "第二階段淘汰"

    ls_ratio = float("nan")
    if math.isfinite(buy_ratio) and math.isfinite(sell_ratio) and sell_ratio != 0:
        ls_ratio = buy_ratio / sell_ratio

    return {
        "交易對": symbol,
        "第一階段狀態": status_zh(stage1_status),
        "精查方向": "多頭" if side == "LONG" else "空頭",
        "最終精查分數": final_score,
        "第一階段SMC分數": int(round(mtf_direction_score)),
        "原策略方向分數": int(round(base_direction_score)),
        "衍生資料分數": derivative_score,
        "最終判定": decision,
        "目前價": safe_float(base.get("last_price")),
        "OI變化%": oi_change * 100 if math.isfinite(oi_change) else float("nan"),
        "近10分價格變化%": price_change * 100 if math.isfinite(price_change) else float("nan"),
        "Funding%": funding_rate * 100 if math.isfinite(funding_rate) else float("nan"),
        "BuyRatio%": buy_ratio * 100 if math.isfinite(buy_ratio) else float("nan"),
        "SellRatio%": sell_ratio * 100 if math.isfinite(sell_ratio) else float("nan"),
        "LongShortRatio": ls_ratio,
        "相對量": rel_vol,
        "RSI14": rsi,
        "ADX14": adx,
        "進場價": safe_float(plan.get("entry")),
        "停損": safe_float(plan.get("stop")),
        "TP1": safe_float(plan.get("tp1")),
        "TP2": safe_float(plan.get("tp2")),
        "進場引擎狀態": plan_status,
        "精查依據": "、".join(derivative_reasons),
        "精查時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def analyze_precision_with_retry(
    client,
    stage1_row,
    trigger_score,
    max_retries,
    retry_delay,
):
    attempts_total = max(1, int(max_retries) + 1)
    last_error = None

    for attempt in range(1, attempts_total + 1):
        try:
            row = analyze_precision_symbol(
                client,
                stage1_row,
                trigger_score=int(trigger_score),
            )
            return row, attempt, None
        except Exception as exc:
            last_error = exc
            if attempt < attempts_total and retry_delay > 0:
                time.sleep(float(retry_delay) * attempt)

    return None, attempts_total, last_error

def reset_scan():
    keys = [
        "scan_universe",
        "scan_pos",
        "scan_rows",
        "scan_errors",
        "scan_mode",
        "scan_started",
        "scan_finished",
        "scan_signature",
        "scan_seconds_total",
        "scan_processed_total",
        "scan_retry_total",
        "auto_scan",
        "last_batch_count",
        "last_batch_seconds",
    ]

    for key in keys:
        st.session_state.pop(key, None)

    reset_precision()


def result_df() -> pd.DataFrame:
    rows = st.session_state.get("scan_rows", [])
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    if "MTF_LONG" in out.columns and "MTF_SHORT" in out.columns:
        out["MAX"] = out[["MTF_LONG", "MTF_SHORT"]].max(axis=1)

        priority = {
            "LONG_READY": 0,
            "SHORT_READY": 0,
            "WATCH_LONG": 1,
            "WATCH_SHORT": 1,
            "NEUTRAL": 2,
        }

        if "狀態代碼" in out.columns:
            out["_priority"] = out["狀態代碼"].map(priority).fillna(9)
        else:
            out["_priority"] = 9

        out = out.sort_values(
            ["_priority", "MAX", "差值"],
            ascending=[True, False, False],
        ).drop(columns=["_priority"])

    return out.reset_index(drop=True)


def estimated_eta(total: int, done: int) -> str:
    processed = int(st.session_state.get("scan_processed_total", 0))
    seconds_total = float(st.session_state.get("scan_seconds_total", 0.0))

    if processed <= 0 or seconds_total <= 0:
        return "計算中"

    avg_seconds = seconds_total / processed
    remaining = max(total - done, 0)
    return fmt_duration(avg_seconds * remaining)


# =============================================================================
# Sidebar settings
# =============================================================================

client = BybitClient()

st.sidebar.markdown("### 掃描設定")

mode = st.sidebar.radio(
    "掃描範圍",
    ["全市場｜全部 USDT 永續", "快速測試｜流動性前 N"],
    index=0,
)

quick_n = st.sidebar.number_input(
    "快速測試 N",
    min_value=10,
    max_value=200,
    value=30,
    step=10,
    disabled=(mode != "快速測試｜流動性前 N"),
)

batch_size = st.sidebar.number_input(
    "每批最多掃描檔數",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="這是每批大小，不是全市場候選上限。",
)

batch_time_limit = st.sidebar.number_input(
    "單批時間保護（秒）",
    min_value=60,
    max_value=260,
    value=210,
    step=10,
    help="若本批已接近此時間，會提早結束並自動進下一批。",
)

per_symbol_cooldown = st.sidebar.number_input(
    "每檔冷卻秒數",
    min_value=0.00,
    max_value=2.00,
    value=0.08,
    step=0.02,
    format="%.2f",
)

between_batch_delay = st.sidebar.number_input(
    "批次間冷卻秒數",
    min_value=0.0,
    max_value=30.0,
    value=1.5,
    step=0.5,
    format="%.1f",
)

max_retries = st.sidebar.number_input(
    "單幣失敗重試次數",
    min_value=0,
    max_value=5,
    value=2,
    step=1,
)

retry_delay = st.sidebar.number_input(
    "重試等待秒數",
    min_value=0.0,
    max_value=5.0,
    value=0.8,
    step=0.2,
    format="%.1f",
)

min_show_score = st.sidebar.slider(
    "訊號顯示門檻",
    min_value=40,
    max_value=90,
    value=55,
    step=5,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### V0.4 第二階段精查")

precision_include_mode = st.sidebar.radio(
    "精查候選",
    ["多空就緒＋高分觀察", "僅多空就緒", "僅高分觀察"],
    index=0,
)

precision_watch_threshold = st.sidebar.slider(
    "觀察標的精查門檻",
    min_value=55,
    max_value=90,
    value=70,
    step=5,
)

precision_trigger_score = st.sidebar.slider(
    "5分K進場引擎最低分",
    min_value=55,
    max_value=100,
    value=70,
    step=5,
)

precision_batch_size = st.sidebar.number_input(
    "精查每批檔數",
    min_value=5,
    max_value=50,
    value=15,
    step=5,
)

precision_batch_time_limit = st.sidebar.number_input(
    "精查單批時間保護（秒）",
    min_value=60,
    max_value=260,
    value=210,
    step=10,
)

precision_cooldown = st.sidebar.number_input(
    "精查每檔冷卻秒數",
    min_value=0.00,
    max_value=2.00,
    value=0.12,
    step=0.02,
    format="%.2f",
)

precision_between_batch_delay = st.sidebar.number_input(
    "精查批次間冷卻秒數",
    min_value=0.0,
    max_value=30.0,
    value=1.5,
    step=0.5,
    format="%.1f",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 復用第一階段結果")
stage1_upload = st.sidebar.file_uploader(
    "匯入 V0.3/V0.4 掃描結果 CSV",
    type=["csv"],
    help="升級部署後可匯入剛才下載的 735 檔 CSV，避免重新掃第一階段約 10 分鐘。",
)
load_stage1_csv_clicked = st.sidebar.button(
    "載入 CSV 作為第一階段完成結果",
    use_container_width=True,
)

signature = f"{mode}|{int(quick_n)}"



if load_stage1_csv_clicked:
    if stage1_upload is None:
        st.sidebar.error("請先選擇第一階段 CSV。")
    else:
        try:
            imported = pd.read_csv(stage1_upload)
            imported = normalize_stage1_import(imported)

            reset_scan()

            rows = imported.to_dict("records")
            symbols = imported["Symbol"].astype(str).tolist()

            st.session_state["scan_rows"] = rows
            st.session_state["scan_universe"] = [{"symbol": s} for s in symbols]
            st.session_state["scan_pos"] = len(rows)
            st.session_state["scan_errors"] = []
            st.session_state["scan_mode"] = "CSV匯入"
            st.session_state["scan_started"] = datetime.now(timezone.utc).isoformat()
            st.session_state["scan_finished"] = datetime.now(timezone.utc).isoformat()
            st.session_state["scan_signature"] = signature
            st.session_state["scan_seconds_total"] = 0.0
            st.session_state["scan_processed_total"] = len(rows)
            st.session_state["scan_retry_total"] = 0
            st.session_state["auto_scan"] = False
            st.session_state["last_batch_count"] = 0
            st.session_state["last_batch_seconds"] = 0.0

            st.sidebar.success(f"已載入第一階段結果：{len(rows)} 檔")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"CSV 載入失敗：{exc}")


# =============================================================================
# Build / reset controls
# =============================================================================

build_col, clear_col = st.sidebar.columns(2)

build_clicked = build_col.button(
    "建立／重設",
    type="primary",
    use_container_width=True,
)

clear_clicked = clear_col.button(
    "清除",
    use_container_width=True,
)

if clear_clicked:
    reset_scan()
    st.rerun()

if build_clicked:
    reset_scan()

    try:
        with st.spinner("正在建立全市場掃描清單..."):
            universe_df = build_universe(client, mode, int(quick_n))

        st.session_state["scan_universe"] = universe_df.to_dict("records")
        st.session_state["scan_pos"] = 0
        st.session_state["scan_rows"] = []
        st.session_state["scan_errors"] = []
        st.session_state["scan_mode"] = mode
        st.session_state["scan_started"] = datetime.now(timezone.utc).isoformat()
        st.session_state["scan_signature"] = signature
        st.session_state["scan_seconds_total"] = 0.0
        st.session_state["scan_processed_total"] = 0
        st.session_state["scan_retry_total"] = 0
        st.session_state["auto_scan"] = False
        st.session_state["last_batch_count"] = 0
        st.session_state["last_batch_seconds"] = 0.0

        st.success(f"掃描清單建立完成：{len(universe_df)} 檔")
    except Exception as exc:
        st.error(f"建立掃描清單失敗：{exc}")


# =============================================================================
# Current state
# =============================================================================

universe = st.session_state.get("scan_universe", [])
position = int(st.session_state.get("scan_pos", 0))

total = len(universe)
done = min(position, total)
remaining = max(total - done, 0)

if total == 0:
    st.info(
        "請先按左側「建立／重設」。"
        "全市場模式會建立目前 Bybit 全部 USDT 線性永續合約清單。"
    )
    st.stop()

if st.session_state.get("scan_signature") != signature:
    st.warning("掃描範圍設定已變更。請按「建立／重設」重新建立清單。")

auto_scan = bool(st.session_state.get("auto_scan", False))


# =============================================================================
# Main control buttons
# =============================================================================

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.0, 1.0, 2.8])

start_auto_clicked = ctrl1.button(
    "▶ 開始／繼續自動掃描",
    type="primary",
    disabled=(remaining == 0 or auto_scan),
    use_container_width=True,
)

pause_clicked = ctrl2.button(
    "⏸ 暫停",
    disabled=(not auto_scan),
    use_container_width=True,
)

one_batch_clicked = ctrl3.button(
    "掃描一批",
    disabled=(remaining == 0 or auto_scan),
    use_container_width=True,
)

if start_auto_clicked:
    st.session_state["auto_scan"] = True
    auto_scan = True
    st.rerun()

if pause_clicked:
    st.session_state["auto_scan"] = False
    auto_scan = False
    st.info("已暫停。再次按「開始／繼續自動掃描」會從目前進度繼續。")

if remaining == 0:
    ctrl4.success("✅ 本輪掃描已全部完成")
elif auto_scan:
    ctrl4.success("🟢 自動掃描中｜每批完成後會自動繼續")
else:
    ctrl4.info("目前已暫停／待命，可自動續掃或只掃一批")


# =============================================================================
# Progress summary
# =============================================================================

p1, p2, p3, p4, p5 = st.columns(5)

p1.metric("全市場候選", total)
p2.metric("已完成", done)
p3.metric("剩餘", remaining)
p4.metric("錯誤", len(st.session_state.get("scan_errors", [])))
p5.metric("預估剩餘時間", estimated_eta(total, done))

progress_value = 1.0 if total == 0 else done / total
st.progress(progress_value)

last_batch_count = int(st.session_state.get("last_batch_count", 0))
last_batch_seconds = float(st.session_state.get("last_batch_seconds", 0.0))
retry_total = int(st.session_state.get("scan_retry_total", 0))

st.caption(
    f"最近一批：{last_batch_count} 檔／{last_batch_seconds:.1f} 秒 ｜ "
    f"累計額外重試：{retry_total} 次 ｜ "
    f"目前進度：{done}/{total}"
)


# =============================================================================
# One batch executor
# =============================================================================

should_run_batch = (auto_scan and remaining > 0) or (one_batch_clicked and remaining > 0)

if should_run_batch:
    batch_start_time = time.monotonic()
    batch_start_position = int(st.session_state.get("scan_pos", 0))
    batch_end_position = min(batch_start_position + int(batch_size), total)

    live_status = st.empty()
    live_progress = st.progress(batch_start_position / total)

    processed_this_batch = 0

    for index in range(batch_start_position, batch_end_position):
        # 單批時間保護：至少先完成一檔才檢查
        elapsed = time.monotonic() - batch_start_time
        if processed_this_batch > 0 and elapsed >= float(batch_time_limit):
            live_status.warning(
                f"本批已達時間保護 {batch_time_limit} 秒，提前結束本批。"
            )
            break

        item = universe[index]
        symbol = str(item.get("symbol", "")).upper()

        live_status.info(
            f"正在掃描 {index + 1}/{total}｜{symbol}｜"
            f"本批第 {processed_this_batch + 1} 檔"
        )

        row, attempts, error = analyze_symbol_with_retry(
            client=client,
            symbol=symbol,
            metadata=item,
            max_retries=int(max_retries),
            retry_delay=float(retry_delay),
        )

        extra_retries = max(0, attempts - 1)
        st.session_state["scan_retry_total"] = (
            int(st.session_state.get("scan_retry_total", 0))
            + extra_retries
        )

        if row is not None:
            st.session_state["scan_rows"].append(row)
        else:
            st.session_state["scan_errors"].append({
                "交易對": symbol,
                "錯誤": str(error),
                "嘗試次數": attempts,
                "時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            })

        # 不論成功/失敗，這一檔都視為已處理，避免永遠卡在同一檔。
        st.session_state["scan_pos"] = index + 1
        processed_this_batch += 1

        live_progress.progress((index + 1) / total)

        if per_symbol_cooldown > 0:
            time.sleep(float(per_symbol_cooldown))

    batch_seconds = time.monotonic() - batch_start_time

    st.session_state["last_batch_count"] = processed_this_batch
    st.session_state["last_batch_seconds"] = batch_seconds
    st.session_state["scan_processed_total"] = (
        int(st.session_state.get("scan_processed_total", 0))
        + processed_this_batch
    )
    st.session_state["scan_seconds_total"] = (
        float(st.session_state.get("scan_seconds_total", 0.0))
        + batch_seconds
    )

    new_position = int(st.session_state.get("scan_pos", 0))
    new_remaining = max(total - new_position, 0)

    if new_remaining == 0:
        st.session_state["auto_scan"] = False
        st.session_state["scan_finished"] = datetime.now(timezone.utc).isoformat()
        live_status.success(
            f"✅ 全市場掃描完成：{total} 檔全部處理完畢。"
        )
        time.sleep(0.5)
        st.rerun()

    elif auto_scan:
        live_status.success(
            f"本批完成 {processed_this_batch} 檔；"
            f"目前 {new_position}/{total}，即將自動進入下一批。"
        )

        if between_batch_delay > 0:
            time.sleep(float(between_batch_delay))

        st.rerun()

    else:
        live_status.success(
            f"本批完成 {processed_this_batch} 檔；"
            f"目前 {new_position}/{total}。"
        )
        st.rerun()


# =============================================================================
# Results
# =============================================================================

out = result_df()

if out.empty:
    st.caption("尚未產生掃描結果。")
    st.stop()

st.markdown("## 掃描結果")

long_ready = out[out["狀態代碼"] == "LONG_READY"].copy()
short_ready = out[out["狀態代碼"] == "SHORT_READY"].copy()
watch = out[out["狀態代碼"].isin(["WATCH_LONG", "WATCH_SHORT"])].copy()

signal_mask = (
    (out["MTF_LONG"] >= int(min_show_score))
    | (out["MTF_SHORT"] >= int(min_show_score))
)
signal_rows = out.loc[signal_mask].copy()

s1, s2, s3, s4 = st.columns(4)
s1.metric("多頭就緒", len(long_ready))
s2.metric("空頭就緒", len(short_ready))
s3.metric("觀察中", len(watch))
s4.metric(f"≥ {min_show_score} 分", len(signal_rows))


display_cols = [
    "Symbol",
    "狀態",
    "方向",
    "MTF_LONG",
    "MTF_SHORT",
    "差值",
    "目前價",
    "1H",
    "15m",
    "5m",
    "5m最新結構",
    "5m區域",
    "5m信心",
    "Turnover24h",
    "24h%",
]
display_cols = [col for col in display_cols if col in out.columns]

display_rename = {
    "Symbol": "交易對",
    "MTF_LONG": "多頭分數",
    "MTF_SHORT": "空頭分數",
    "差值": "多空差值",
    "1H": "1H方向",
    "15m": "15分K方向",
    "5m": "5分K方向",
    "5m最新結構": "5分K最新結構",
    "5m區域": "5分K區域",
    "5m信心": "5分K信心",
    "Turnover24h": "24H成交額",
    "24h%": "24H漲跌",
}


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["優先訊號", "多頭", "空頭", "全部結果", "錯誤紀錄"]
)

with tab1:
    priority = out[
        out["狀態代碼"].isin(
            ["LONG_READY", "SHORT_READY", "WATCH_LONG", "WATCH_SHORT"]
        )
    ].copy()

    priority = priority[
        (priority["MTF_LONG"] >= int(min_show_score))
        | (priority["MTF_SHORT"] >= int(min_show_score))
    ]

    if priority.empty:
        st.info("目前已掃描區段尚無達門檻訊號。")
    else:
        st.dataframe(
            priority[display_cols].rename(columns=display_rename),
            use_container_width=True,
            hide_index=True,
        )


with tab2:
    long_rows = out[
        out["狀態代碼"].isin(["LONG_READY", "WATCH_LONG"])
    ].copy()

    if long_rows.empty:
        st.info("目前沒有多頭候選。")
    else:
        long_rows = long_rows.sort_values(
            ["MTF_LONG", "差值"],
            ascending=[False, False],
        )

        st.dataframe(
            long_rows[display_cols].rename(columns=display_rename),
            use_container_width=True,
            hide_index=True,
        )


with tab3:
    short_rows = out[
        out["狀態代碼"].isin(["SHORT_READY", "WATCH_SHORT"])
    ].copy()

    if short_rows.empty:
        st.info("目前沒有空頭候選。")
    else:
        short_rows = short_rows.sort_values(
            ["MTF_SHORT", "差值"],
            ascending=[False, True],
        )

        st.dataframe(
            short_rows[display_cols].rename(columns=display_rename),
            use_container_width=True,
            hide_index=True,
        )


with tab4:
    st.dataframe(
        out[display_cols].rename(columns=display_rename),
        use_container_width=True,
        hide_index=True,
    )

    export_df = out.copy()

    internal_cols = [
        "狀態代碼",
        "方向代碼",
        "1H代碼",
        "15m代碼",
        "5m代碼",
        "MAX",
    ]
    export_df = export_df.drop(
        columns=[col for col in internal_cols if col in export_df.columns],
        errors="ignore",
    )

    export_df = export_df.rename(columns=display_rename)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "下載目前掃描結果 CSV",
        data=csv_bytes,
        file_name="bybit_smc_stage1_v04.csv",
        mime="text/csv",
    )


with tab5:
    error_df = pd.DataFrame(st.session_state.get("scan_errors", []))

    if error_df.empty:
        st.success("目前沒有錯誤。")
    else:
        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Completion summary
# =============================================================================

current_position = int(st.session_state.get("scan_pos", 0))

if current_position >= total:
    success_count = len(st.session_state.get("scan_rows", []))
    error_count = len(st.session_state.get("scan_errors", []))
    total_seconds = float(st.session_state.get("scan_seconds_total", 0.0))

    st.success(
        f"✅ 本輪全市場掃描完成：共 {total} 檔｜"
        f"成功 {success_count} 檔｜錯誤 {error_count} 檔｜"
        f"掃描運算時間約 {fmt_duration(total_seconds)}"
    )


# =============================================================================
# V0.4 第二階段精查執行區
# =============================================================================

if current_position >= total:
    st.markdown("---")
    st.markdown("## V0.4 第二階段精查")
    st.caption(
        "第一階段只用多週期 SMC 快速篩選；第二階段只針對多空就緒與高分觀察標的，"
        "重新取得最新 K 線，再加入 Open Interest、Funding、Long/Short Ratio、"
        "相對成交量與 5分K 進場／停損／TP1／TP2。"
    )

    precision_signature = (
        f"{precision_include_mode}|{int(precision_watch_threshold)}|"
        f"{int(precision_trigger_score)}"
    )

    precision_queue = st.session_state.get("precision_queue", [])
    precision_pos = int(st.session_state.get("precision_pos", 0))
    precision_total = len(precision_queue)
    precision_remaining = max(precision_total - precision_pos, 0)

    q1, q2, q3 = st.columns([1.3, 1.3, 3.4])

    build_precision_clicked = q1.button(
        "建立第二階段精查清單",
        type="primary",
        use_container_width=True,
    )

    clear_precision_clicked = q2.button(
        "清除精查結果",
        use_container_width=True,
    )

    if clear_precision_clicked:
        reset_precision()
        st.rerun()

    if build_precision_clicked:
        reset_precision()

        queue_df = build_precision_queue(
            out,
            include_mode=precision_include_mode,
            watch_threshold=int(precision_watch_threshold),
        )

        st.session_state["precision_queue"] = queue_df.to_dict("records")
        st.session_state["precision_pos"] = 0
        st.session_state["precision_rows"] = []
        st.session_state["precision_errors"] = []
        st.session_state["precision_auto"] = False
        st.session_state["precision_started"] = datetime.now(timezone.utc).isoformat()
        st.session_state["precision_seconds_total"] = 0.0
        st.session_state["precision_processed_total"] = 0
        st.session_state["precision_retry_total"] = 0
        st.session_state["precision_last_batch_count"] = 0
        st.session_state["precision_last_batch_seconds"] = 0.0
        st.session_state["precision_signature"] = precision_signature

        st.success(
            f"第二階段精查清單建立完成：{len(queue_df)} 檔。"
            "多空就緒標的全部納入；觀察標的依精查門檻篩選。"
        )
        st.rerun()

    precision_queue = st.session_state.get("precision_queue", [])
    precision_pos = int(st.session_state.get("precision_pos", 0))
    precision_total = len(precision_queue)
    precision_remaining = max(precision_total - precision_pos, 0)

    if precision_total == 0:
        q3.info("請先建立第二階段精查清單。")
    else:
        if st.session_state.get("precision_signature") != precision_signature:
            q3.warning("精查設定已變更，請重新建立第二階段精查清單。")

        precision_auto = bool(st.session_state.get("precision_auto", False))

        a1, a2, a3, a4 = st.columns([1.3, 1.0, 1.0, 2.7])

        start_precision_clicked = a1.button(
            "▶ 開始／繼續自動精查",
            type="primary",
            disabled=(precision_remaining == 0 or precision_auto),
            use_container_width=True,
        )

        pause_precision_clicked = a2.button(
            "⏸ 暫停精查",
            disabled=(not precision_auto),
            use_container_width=True,
        )

        one_precision_batch_clicked = a3.button(
            "精查一批",
            disabled=(precision_remaining == 0 or precision_auto),
            use_container_width=True,
        )

        if start_precision_clicked:
            st.session_state["precision_auto"] = True
            precision_auto = True
            st.rerun()

        if pause_precision_clicked:
            st.session_state["precision_auto"] = False
            precision_auto = False
            st.info("第二階段精查已暫停，可從目前進度繼續。")

        if precision_remaining == 0:
            a4.success("✅ 第二階段精查已完成")
        elif precision_auto:
            a4.success("🟢 第二階段自動精查中")
        else:
            a4.info("第二階段目前待命／已暫停")

        pp1, pp2, pp3, pp4 = st.columns(4)
        pp1.metric("精查候選", precision_total)
        pp2.metric("已精查", precision_pos)
        pp3.metric("剩餘", precision_remaining)
        pp4.metric("精查錯誤", len(st.session_state.get("precision_errors", [])))

        st.progress(
            1.0 if precision_total == 0
            else precision_pos / precision_total
        )

        run_precision_batch = (
            (precision_auto and precision_remaining > 0)
            or (one_precision_batch_clicked and precision_remaining > 0)
        )

        if run_precision_batch:
            batch_started = time.monotonic()
            start_pos = int(st.session_state.get("precision_pos", 0))
            end_pos = min(
                start_pos + int(precision_batch_size),
                precision_total,
            )

            precision_status = st.empty()
            precision_progress = st.progress(start_pos / precision_total)
            processed = 0

            for idx in range(start_pos, end_pos):
                elapsed = time.monotonic() - batch_started
                if processed > 0 and elapsed >= float(precision_batch_time_limit):
                    precision_status.warning("精查本批已達時間保護，提前切換下一批。")
                    break

                item = precision_queue[idx]
                symbol = str(item.get("Symbol", "")).upper()

                precision_status.info(
                    f"第二階段精查 {idx + 1}/{precision_total}｜{symbol}｜"
                    f"本批第 {processed + 1} 檔"
                )

                row, attempts, error = analyze_precision_with_retry(
                    client=client,
                    stage1_row=item,
                    trigger_score=int(precision_trigger_score),
                    max_retries=int(max_retries),
                    retry_delay=float(retry_delay),
                )

                st.session_state["precision_retry_total"] = (
                    int(st.session_state.get("precision_retry_total", 0))
                    + max(0, attempts - 1)
                )

                if row is not None:
                    st.session_state["precision_rows"].append(row)
                else:
                    st.session_state["precision_errors"].append({
                        "交易對": symbol,
                        "錯誤": str(error),
                        "嘗試次數": attempts,
                        "時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    })

                st.session_state["precision_pos"] = idx + 1
                processed += 1
                precision_progress.progress((idx + 1) / precision_total)

                if precision_cooldown > 0:
                    time.sleep(float(precision_cooldown))

            batch_seconds = time.monotonic() - batch_started

            st.session_state["precision_last_batch_count"] = processed
            st.session_state["precision_last_batch_seconds"] = batch_seconds
            st.session_state["precision_processed_total"] = (
                int(st.session_state.get("precision_processed_total", 0))
                + processed
            )
            st.session_state["precision_seconds_total"] = (
                float(st.session_state.get("precision_seconds_total", 0.0))
                + batch_seconds
            )

            new_pos = int(st.session_state.get("precision_pos", 0))
            new_remaining = max(precision_total - new_pos, 0)

            if new_remaining == 0:
                st.session_state["precision_auto"] = False
                st.session_state["precision_finished"] = datetime.now(timezone.utc).isoformat()
                precision_status.success("✅ 第二階段精查全部完成。")
                time.sleep(0.5)
                st.rerun()

            elif precision_auto:
                precision_status.success(
                    f"精查本批完成 {processed} 檔；目前 {new_pos}/{precision_total}，"
                    "即將自動進下一批。"
                )
                if precision_between_batch_delay > 0:
                    time.sleep(float(precision_between_batch_delay))
                st.rerun()

            else:
                precision_status.success(
                    f"精查本批完成 {processed} 檔；目前 {new_pos}/{precision_total}。"
                )
                st.rerun()

        precision_results = pd.DataFrame(
            st.session_state.get("precision_rows", [])
        )

        if not precision_results.empty:
            precision_results = precision_results.sort_values(
                ["最終精查分數", "第一階段SMC分數"],
                ascending=[False, False],
            ).reset_index(drop=True)

            st.markdown("### 第二階段精查結果")

            trade_long = precision_results[
                precision_results["最終判定"] == "多頭進場條件成立"
            ]
            trade_short = precision_results[
                precision_results["最終判定"] == "空頭進場條件成立"
            ]
            waiting = precision_results[
                precision_results["最終判定"].isin(
                    ["等待多頭進場觸發", "等待空頭進場觸發"]
                )
            ]
            eliminated = precision_results[
                precision_results["最終判定"] == "第二階段淘汰"
            ]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("多頭進場條件成立", len(trade_long))
            r2.metric("空頭進場條件成立", len(trade_short))
            r3.metric("等待進場觸發", len(waiting))
            r4.metric("第二階段淘汰", len(eliminated))

            precision_tabs = st.tabs(
                ["最終優先清單", "多頭", "空頭", "全部精查", "精查錯誤"]
            )

            important_cols = [
                "交易對",
                "第一階段狀態",
                "精查方向",
                "最終精查分數",
                "第一階段SMC分數",
                "原策略方向分數",
                "衍生資料分數",
                "最終判定",
                "目前價",
                "OI變化%",
                "近10分價格變化%",
                "Funding%",
                "BuyRatio%",
                "SellRatio%",
                "LongShortRatio",
                "相對量",
                "RSI14",
                "ADX14",
                "進場價",
                "停損",
                "TP1",
                "TP2",
                "精查依據",
            ]
            important_cols = [
                c for c in important_cols
                if c in precision_results.columns
            ]

            with precision_tabs[0]:
                final_priority = precision_results[
                    precision_results["最終判定"].isin(
                        [
                            "多頭進場條件成立",
                            "空頭進場條件成立",
                            "等待多頭進場觸發",
                            "等待空頭進場觸發",
                            "保留觀察",
                        ]
                    )
                ]
                st.dataframe(
                    final_priority[important_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            with precision_tabs[1]:
                x = precision_results[
                    precision_results["精查方向"] == "多頭"
                ]
                st.dataframe(
                    x[important_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            with precision_tabs[2]:
                x = precision_results[
                    precision_results["精查方向"] == "空頭"
                ]
                st.dataframe(
                    x[important_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            with precision_tabs[3]:
                st.dataframe(
                    precision_results[important_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                precision_csv = precision_results.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "下載 V0.4 第二階段精查結果 CSV",
                    data=precision_csv,
                    file_name="bybit_precision_scan_v04.csv",
                    mime="text/csv",
                )

            with precision_tabs[4]:
                precision_error_df = pd.DataFrame(
                    st.session_state.get("precision_errors", [])
                )
                if precision_error_df.empty:
                    st.success("目前沒有第二階段精查錯誤。")
                else:
                    st.dataframe(
                        precision_error_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        current_precision_pos = int(st.session_state.get("precision_pos", 0))
        if current_precision_pos >= precision_total and precision_total > 0:
            seconds = float(
                st.session_state.get("precision_seconds_total", 0.0)
            )
            st.success(
                f"✅ 第二階段精查完成：{precision_total} 檔｜"
                f"錯誤 {len(st.session_state.get('precision_errors', []))} 檔｜"
                f"精查運算時間約 {fmt_duration(seconds)}"
            )

