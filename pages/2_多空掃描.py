# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from bybit_client import BybitClient
from smc_engine import analyze_smc


# =============================================================================
# Page
# =============================================================================

st.set_page_config(page_title="多空掃描 V0.3", layout="wide")
st.title("全市場多空掃描 V0.3｜SMC 自動連續掃描")
st.caption(
    "預設掃描全部 Bybit USDT 永續，不綁成交量排名。"
    "建立清單後按一次「開始／繼續自動掃描」，系統會自動分批往下掃到全部完成；"
    "可隨時暫停，之後從原進度繼續。"
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

signature = f"{mode}|{int(quick_n)}"


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
        file_name="bybit_smc_scan_v03.csv",
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
