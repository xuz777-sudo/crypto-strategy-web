# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from bybit_client import BybitClient
from cloud_state import CloudStateStore
from indicators import prepare_indicators
from smc_engine import analyze_smc
from strategy_engine import build_trade_plan


st.set_page_config(page_title="5分K即時進場監控 V0.5", layout="wide")
st.title("5分K 即時進出場監控 V0.5")
st.caption(
    "直接讀取 V0.5 第二階段精查保存在 GitHub Private Repo 的候選，只重新確認候選幣的最新 5分K；"
    "不會重新掃描 700 多檔全市場。頁面開啟時預設每 5 分鐘自動更新。"
)


cloud = CloudStateStore()
client = BybitClient()

PRECISION_KEY = "precision/latest"
MONITOR_KEY = "entry_monitor/latest"


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


def load_precision_rows():
    body = cloud.load(PRECISION_KEY)
    if not body:
        return [], None
    payload = body.get("payload", {}) or {}
    return payload.get("precision_rows", []) or [], body.get("saved_at_utc")


def load_monitor_state():
    try:
        body = cloud.load(MONITOR_KEY)
        if not body:
            return {}
        payload = body.get("payload", {}) or {}
        rows = payload.get("rows", []) or []
        return {
            str(r.get("交易對", "")).upper(): r
            for r in rows
            if r.get("交易對")
        }
    except Exception:
        return {}


def save_monitor_rows(rows):
    if not cloud.available:
        return
    cloud.save(
        MONITOR_KEY,
        {
            "version": "V0.5",
            "rows": rows,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def direction_code(row):
    text = str(row.get("精查方向", ""))
    return "LONG" if text == "多頭" else ("SHORT" if text == "空頭" else "NEUTRAL")


def candidate_rows(rows, mode, min_score, top_n):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "最終精查分數" in df.columns:
        df["最終精查分數"] = pd.to_numeric(df["最終精查分數"], errors="coerce")

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
    df = df.sort_values("最終精查分數", ascending=False).head(int(top_n))
    return df.reset_index(drop=True)


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


def evaluate_existing(prev, last_bar):
    status = str(prev.get("監控狀態", ""))
    direction = str(prev.get("方向", ""))
    entry = safe_float(prev.get("進場價"))
    stop = safe_float(prev.get("停損"))
    tp1 = safe_float(prev.get("TP1"))
    tp2 = safe_float(prev.get("TP2"))

    if not all(math.isfinite(x) for x in [entry, stop, tp1, tp2]):
        return None

    high = safe_float(last_bar.get("high"))
    low = safe_float(last_bar.get("low"))

    if not math.isfinite(high) or not math.isfinite(low):
        return None

    tp1_done = bool(prev.get("TP1已達", False))

    # 保守處理：同一根 K 同時碰停損與停利時，先視為停損。
    effective_stop = entry if tp1_done else stop

    if direction == "多頭":
        if low <= effective_stop:
            return "保本出場" if tp1_done else "停損出場"
        if high >= tp2:
            return "TP2 完成"
        if high >= tp1:
            return "TP1 已達"
    elif direction == "空頭":
        if high >= effective_stop:
            return "保本出場" if tp1_done else "停損出場"
        if low <= tp2:
            return "TP2 完成"
        if low <= tp1:
            return "TP1 已達"

    return "持倉監控" if status in {
        "多頭進場成立", "空頭進場成立", "持倉監控", "TP1 已達"
    } else None


def analyze_candidate(row, prev, trigger_score, chase_atr):
    symbol = str(row["交易對"]).upper()
    direction = direction_code(row)

    m5 = client.kline(symbol, "5", 180)
    if len(m5) < 50:
        raise RuntimeError("5分K資料不足")

    x = prepare_indicators(m5)
    smc = analyze_smc(m5)

    last = x.iloc[-1]
    current = safe_float(last.get("close"))
    atr = safe_float(last.get("ATR14"))

    # 已經有固定交易計畫時，不再每 5 分鐘重算 Entry。
    existing_status = evaluate_existing(prev or {}, last)
    if existing_status:
        out = dict(prev)
        out.update({
            "目前價": current,
            "監控狀態": existing_status,
            "最後更新UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        if existing_status == "TP1 已達":
            out["TP1已達"] = True
            out["提示"] = "TP1 已到，停損建議移至進場價保本。"
        elif existing_status == "TP2 完成":
            out["提示"] = "TP2 已到，交易計畫完成。"
        elif existing_status in {"停損出場", "保本出場"}:
            out["提示"] = "此筆監控交易已結束。"
        else:
            out["提示"] = "持續監控 5分K。"
        return out

    # 未進場：重新判斷最新 5m 觸發。
    base_score = safe_float(row.get("原策略方向分數"), 0.0)
    plan = make_plan(
        symbol,
        direction,
        base_score,
        m5,
        trigger_score,
    )

    expected = "LONG_TRIGGER" if direction == "LONG" else "SHORT_TRIGGER"

    # 結構反向時先判斷失效
    if direction == "LONG" and smc.get("bias") == "SHORT":
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

        status = "禁止追價" if too_far else (
            "多頭進場成立" if direction == "LONG" else "空頭進場成立"
        )
    else:
        status = "等待多頭觸發" if direction == "LONG" else "等待空頭觸發"

    out = {
        "交易對": symbol,
        "方向": "多頭" if direction == "LONG" else "空頭",
        "精查分數": int(safe_float(row.get("最終精查分數"), 0)),
        "目前價": current,
        "5分K方向": {
            "LONG": "多頭",
            "LEAN_LONG": "偏多",
            "SHORT": "空頭",
            "LEAN_SHORT": "偏空",
            "NEUTRAL": "中性",
        }.get(str(smc.get("bias")), str(smc.get("bias"))),
        "監控狀態": status,
        "進場價": safe_float(plan.get("entry")),
        "停損": safe_float(plan.get("stop")),
        "TP1": safe_float(plan.get("tp1")),
        "TP2": safe_float(plan.get("tp2")),
        "TP1已達": False,
        "訊號時間UTC": (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if status in {"多頭進場成立", "空頭進場成立"}
            else (prev or {}).get("訊號時間UTC")
        ),
        "最後更新UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "提示": (
            "進場條件成立。"
            if status in {"多頭進場成立", "空頭進場成立"}
            else "價格已離觸發點過遠，不追價。"
            if status == "禁止追價"
            else "SMC 方向反轉，候選失效。"
            if status == "訊號失效"
            else "等待下一根 5分K 確認。"
        ),
    }
    return out


status = cloud.status()
if status.get("ok"):
    st.success(f"☁️ {status['message']}｜監控結果也會自動保存")
else:
    st.error(
        f"☁️ {status['message']}。請先完成 V0.5 GitHub 私有儲存 設定，"
        "否則無法讀取第二階段候選。"
    )
    st.stop()

try:
    precision_rows, source_saved_at = load_precision_rows()
except Exception as exc:
    st.error(f"讀取第二階段雲端結果失敗：{exc}")
    st.stop()

if not precision_rows:
    st.info(
        "雲端尚無 V0.5 第二階段精查結果。"
        "請先到「多空掃描」完成第二階段精查。"
    )
    st.stop()

st.caption(f"第二階段雲端資料時間：{source_saved_at or '-'}")

st.sidebar.markdown("### 監控設定")
mode = st.sidebar.radio(
    "候選範圍",
    ["只監控可進場／等待觸發", "包含保留觀察"],
    index=0,
)
min_score = st.sidebar.slider(
    "最低精查分數",
    50, 95, 65, 5
)
top_n = st.sidebar.number_input(
    "最多監控檔數",
    5, 50, 20, 5
)
trigger_score = st.sidebar.slider(
    "5分K觸發最低分",
    55, 100, 70, 5
)
chase_atr = st.sidebar.slider(
    "禁止追價距離（ATR倍數）",
    0.25, 2.00, 0.75, 0.25
)
auto_refresh = st.sidebar.toggle(
    "每 5 分鐘自動更新",
    value=True,
)

manual_refresh = st.sidebar.button(
    "立即重新檢查",
    type="primary",
    use_container_width=True,
)

candidates = candidate_rows(
    precision_rows,
    mode,
    min_score,
    top_n,
)

if candidates.empty:
    st.warning("目前沒有符合監控設定的第二階段候選。")
    st.stop()

st.metric("目前監控候選", len(candidates))


def render_monitor():
    previous = load_monitor_state()
    results = []
    errors = []

    progress = st.progress(0.0)
    live = st.empty()

    for i, (_, row) in enumerate(candidates.iterrows()):
        symbol = str(row["交易對"]).upper()
        live.info(f"正在更新 {i + 1}/{len(candidates)}｜{symbol}")

        try:
            result = analyze_candidate(
                row.to_dict(),
                previous.get(symbol),
                trigger_score,
                chase_atr,
            )
            results.append(result)
        except Exception as exc:
            errors.append({
                "交易對": symbol,
                "錯誤": str(exc),
            })

        progress.progress((i + 1) / len(candidates))

    live.empty()
    save_monitor_rows(results)

    df = pd.DataFrame(results)

    if df.empty:
        st.error("本輪沒有成功取得監控結果。")
        return

    priority = {
        "多頭進場成立": 0,
        "空頭進場成立": 0,
        "TP1 已達": 1,
        "持倉監控": 2,
        "等待多頭觸發": 3,
        "等待空頭觸發": 3,
        "禁止追價": 4,
        "訊號失效": 5,
        "TP2 完成": 6,
        "停損出場": 6,
        "保本出場": 6,
    }
    df["_p"] = df["監控狀態"].map(priority).fillna(9)
    df = df.sort_values(
        ["_p", "精查分數"],
        ascending=[True, False],
    ).drop(columns=["_p"])

    enter_count = int(
        df["監控狀態"].isin(["多頭進場成立", "空頭進場成立"]).sum()
    )
    waiting_count = int(
        df["監控狀態"].isin(["等待多頭觸發", "等待空頭觸發"]).sum()
    )
    active_count = int(
        df["監控狀態"].isin(["持倉監控", "TP1 已達"]).sum()
    )
    invalid_count = int(
        df["監控狀態"].isin(["禁止追價", "訊號失效"]).sum()
    )

    a, b, c, d = st.columns(4)
    a.metric("新進場成立", enter_count)
    b.metric("等待觸發", waiting_count)
    c.metric("持倉／TP1", active_count)
    d.metric("禁止追價／失效", invalid_count)

    show_cols = [
        "交易對", "方向", "精查分數", "目前價", "5分K方向", "監控狀態",
        "進場價", "停損", "TP1", "TP2", "訊號時間UTC", "最後更新UTC", "提示"
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(
        df[show_cols],
        use_container_width=True,
        hide_index=True,
    )

    if errors:
        with st.expander(f"本輪錯誤 {len(errors)} 檔"):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    st.success(
        f"✅ 本輪 5分K 監控完成：{len(results)} 檔。"
        "結果已保存到 GitHub Private Repo，不會因重新整理而消失。"
    )


# Streamlit >=1.45 支援 fragment run_every。
# auto_refresh=False 時 fragment 不執行 API；按「立即重新檢查」仍可手動更新。
@st.fragment(run_every="5m")
def auto_monitor_fragment():
    if auto_refresh:
        render_monitor()
    elif manual_refresh:
        render_monitor()
    else:
        previous = load_monitor_state()
        if previous:
            df = pd.DataFrame(previous.values())
            st.info("自動更新已關閉，目前顯示上一次雲端監控結果。")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("自動更新已關閉。按左側「立即重新檢查」開始第一輪監控。")


auto_monitor_fragment()
