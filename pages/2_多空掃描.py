# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import math
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from bybit_client import BybitClient
from smc_engine import analyze_smc


st.set_page_config(page_title="多空掃描 V0.2", layout="wide")
st.title("全市場多空掃描 V0.2｜SMC 分批掃描")
st.caption(
    "預設掃描全部 Bybit USDT 永續，不綁成交量排名。"
    "每次只處理一批，避免 Cloud Run 單次執行逾時；掃描進度會保留在目前瀏覽器工作階段。"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

LONG_SET = {"LONG", "LEAN_LONG"}
SHORT_SET = {"SHORT", "LEAN_SHORT"}


def fmt_price(v):
    try:
        v = float(v)
        if not math.isfinite(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:,.4f}"
        return f"{v:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


def mtf_score(s5, s15, s60):
    """
    與單幣分析 V0.2.1 相同：
      5m  55%
      15m 30%
      1H  15%
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
    READY = 5m 已同向，15m 不反向，1H 沒有強烈反向。
    WATCH = 有方向分數，但週期尚未同步。
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


def structure_text(smc):
    ev = smc.get("latest_structure")
    if not ev:
        return "-"
    return f'{ev.get("side","-")} {ev.get("type","-")} @{fmt_price(ev.get("level"))}'


def zone_text(smc):
    dr = smc.get("dealing_range", {}) or {}
    return str(dr.get("zone", "-")).upper()


def build_universe(client: BybitClient, mode: str, quick_n: int) -> pd.DataFrame:
    """
    Full-market mode does NOT rank/filter by volume.
    Ticker data is used only to attach current price / turnover information.
    """
    inst = client.instruments_linear_usdt()
    tick = client.tickers_linear()

    if inst.empty:
        raise RuntimeError("USDT 永續商品清單為空。")

    symbols = (
        inst[["symbol"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    symbols["symbol"] = symbols["symbol"].astype(str).str.upper()

    if not tick.empty and "symbol" in tick.columns:
        keep_cols = [
            c for c in [
                "symbol", "lastPrice", "turnover24h", "volume24h",
                "price24hPcnt", "fundingRate", "openInterest"
            ]
            if c in tick.columns
        ]
        meta = tick[keep_cols].copy()
        meta["symbol"] = meta["symbol"].astype(str).str.upper()
        symbols = symbols.merge(meta, on="symbol", how="left")

    if mode == "快速測試｜流動性前 N":
        # 只在使用者明確選快速測試時才用 turnover 排序。
        if "turnover24h" in symbols.columns:
            symbols = symbols.sort_values(
                "turnover24h", ascending=False, na_position="last"
            ).head(int(quick_n))
        else:
            symbols = symbols.head(int(quick_n))
    else:
        # 全市場模式固定按 symbol 排序，完全不綁成交量排名。
        symbols = symbols.sort_values("symbol")

    return symbols.reset_index(drop=True)


def analyze_symbol(client: BybitClient, symbol: str, meta: dict) -> dict:
    """
    全市場初掃只抓 1H / 15m / 5m 三組 K 線，先跑 SMC。
    OI / Funding / Account Ratio 留給單幣分析或下一階段精查，
    可大幅減少全市場 API 請求量。
    """
    h1 = client.kline(symbol, "60", 240)
    m15 = client.kline(symbol, "15", 240)
    m5 = client.kline(symbol, "5", 240)

    if min(len(h1), len(m15), len(m5)) < 50:
        raise RuntimeError(
            f"K線不足：1H={len(h1)} 15m={len(m15)} 5m={len(m5)}"
        )

    s60 = analyze_smc(h1)
    s15 = analyze_smc(m15)
    s5 = analyze_smc(m5)

    ml, ms, mb = mtf_score(s5, s15, s60)
    status = readiness(mb, s5, s15, s60)

    last_price = s5.get("last_price")
    if last_price is None:
        last_price = meta.get("lastPrice")

    return {
        "Symbol": symbol,
        "狀態": status,
        "方向": mb,
        "MTF_LONG": ml,
        "MTF_SHORT": ms,
        "差值": ml - ms,
        "目前價": last_price,
        "1H": s60.get("bias", "NEUTRAL"),
        "15m": s15.get("bias", "NEUTRAL"),
        "5m": s5.get("bias", "NEUTRAL"),
        "1H結構": s60.get("structure_state", "-"),
        "15m結構": s15.get("structure_state", "-"),
        "5m結構": s5.get("structure_state", "-"),
        "5m最新結構": structure_text(s5),
        "5m區域": zone_text(s5),
        "5m信心": s5.get("confidence", 0),
        "Turnover24h": meta.get("turnover24h"),
        "24h%": meta.get("price24hPcnt"),
        "掃描時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def reset_scan():
    for key in [
        "scan_universe",
        "scan_pos",
        "scan_rows",
        "scan_errors",
        "scan_mode",
        "scan_started",
        "scan_signature",
    ]:
        st.session_state.pop(key, None)


def result_df():
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
        out["_priority"] = out["狀態"].map(priority).fillna(9)
        out = out.sort_values(
            ["_priority", "MAX", "差值"],
            ascending=[True, False, False],
        ).drop(columns=["_priority"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

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
    "本次最多掃描檔數",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="這只是每次執行批次大小，不是全市場候選上限。",
)

cooldown = st.sidebar.number_input(
    "每檔冷卻秒數",
    min_value=0.00,
    max_value=2.00,
    value=0.08,
    step=0.02,
    format="%.2f",
)

min_show_score = st.sidebar.slider(
    "訊號顯示門檻",
    min_value=40,
    max_value=90,
    value=55,
    step=5,
)

signature = f"{mode}|{int(quick_n)}"

c_start, c_clear = st.sidebar.columns(2)
start_clicked = c_start.button("開始/重設", type="primary", use_container_width=True)
clear_clicked = c_clear.button("清除", use_container_width=True)

if clear_clicked:
    reset_scan()
    st.rerun()

if start_clicked:
    reset_scan()
    try:
        with st.spinner("建立掃描清單..."):
            universe = build_universe(client, mode, int(quick_n))
        st.session_state["scan_universe"] = universe.to_dict("records")
        st.session_state["scan_pos"] = 0
        st.session_state["scan_rows"] = []
        st.session_state["scan_errors"] = []
        st.session_state["scan_mode"] = mode
        st.session_state["scan_started"] = datetime.now(timezone.utc).isoformat()
        st.session_state["scan_signature"] = signature
        st.success(f"掃描清單建立完成：{len(universe)} 檔")
    except Exception as e:
        st.error(f"建立掃描清單失敗：{e}")


# ---------------------------------------------------------------------
# Scan state
# ---------------------------------------------------------------------

universe = st.session_state.get("scan_universe", [])
pos = int(st.session_state.get("scan_pos", 0))
total = len(universe)
done = min(pos, total)
remaining = max(total - done, 0)

if total == 0:
    st.info(
        "按左側「開始/重設」建立掃描清單。"
        "全市場模式會納入全部目前交易中的 Bybit USDT 線性永續合約。"
    )
    st.stop()

if st.session_state.get("scan_signature") != signature:
    st.warning("你已變更掃描範圍設定。請按「開始/重設」重新建立清單。")

p1, p2, p3, p4 = st.columns(4)
p1.metric("全市場候選", total)
p2.metric("已完成", done)
p3.metric("剩餘", remaining)
p4.metric("錯誤", len(st.session_state.get("scan_errors", [])))

progress_value = 1.0 if total == 0 else done / total
st.progress(progress_value)

scan_col, note_col = st.columns([1, 3])
run_batch = scan_col.button(
    "掃描下一批" if remaining else "掃描完成",
    type="primary",
    disabled=(remaining == 0),
    use_container_width=True,
)

note_col.caption(
    f"本次最多 {int(batch_size)} 檔；完成後可再次按「掃描下一批」。"
    "這個設計避免 Cloud Run 300 秒請求逾時，同時不限制總候選檔數。"
)

if run_batch and remaining > 0:
    end = min(pos + int(batch_size), total)

    prog = st.progress(done / total)
    status_box = st.empty()

    for idx in range(pos, end):
        item = universe[idx]
        symbol = str(item.get("symbol", "")).upper()
        status_box.info(f"正在掃描 {idx + 1}/{total}｜{symbol}")

        try:
            row = analyze_symbol(client, symbol, item)
            st.session_state["scan_rows"].append(row)
        except Exception as e:
            st.session_state["scan_errors"].append({
                "Symbol": symbol,
                "錯誤": str(e),
                "時間UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            })

        st.session_state["scan_pos"] = idx + 1
        prog.progress((idx + 1) / total)

        if cooldown > 0:
            time.sleep(float(cooldown))

    status_box.success(
        f"本批完成：目前 {st.session_state['scan_pos']}/{total} 檔。"
    )
    st.rerun()


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

out = result_df()

if out.empty:
    st.caption("尚未有掃描結果。")
else:
    st.markdown("## 掃描結果")

    signal_mask = (
        (out["MTF_LONG"] >= int(min_show_score))
        | (out["MTF_SHORT"] >= int(min_show_score))
    )
    sig = out.loc[signal_mask].copy()

    long_ready = out[out["狀態"] == "LONG_READY"].copy()
    short_ready = out[out["狀態"] == "SHORT_READY"].copy()
    watch = out[out["狀態"].isin(["WATCH_LONG", "WATCH_SHORT"])].copy()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("LONG READY", len(long_ready))
    s2.metric("SHORT READY", len(short_ready))
    s3.metric("WATCH", len(watch))
    s4.metric(f"≥ {min_show_score} 分", len(sig))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["優先訊號", "LONG", "SHORT", "全部結果", "錯誤紀錄"]
    )

    display_cols = [
        "Symbol", "狀態", "方向", "MTF_LONG", "MTF_SHORT", "差值",
        "目前價", "1H", "15m", "5m", "5m最新結構", "5m區域",
        "5m信心", "Turnover24h", "24h%"
    ]
    display_cols = [c for c in display_cols if c in out.columns]

    with tab1:
        priority = out[
            out["狀態"].isin(
                ["LONG_READY", "SHORT_READY", "WATCH_LONG", "WATCH_SHORT"]
            )
        ]
        priority = priority[
            (priority["MTF_LONG"] >= int(min_show_score))
            | (priority["MTF_SHORT"] >= int(min_show_score))
        ]
        if priority.empty:
            st.info("目前已掃描區段尚無達門檻訊號。")
        else:
            st.dataframe(
                priority[display_cols],
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        x = out[
            (out["方向"].isin(LONG_SET))
            | (out["狀態"].isin(["LONG_READY", "WATCH_LONG"]))
        ].copy()
        if not x.empty:
            x = x.sort_values(["MTF_LONG", "差值"], ascending=[False, False])
            st.dataframe(x[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("目前沒有 LONG 候選。")

    with tab3:
        x = out[
            (out["方向"].isin(SHORT_SET))
            | (out["狀態"].isin(["SHORT_READY", "WATCH_SHORT"]))
        ].copy()
        if not x.empty:
            x = x.sort_values(["MTF_SHORT", "差值"], ascending=[False, True])
            st.dataframe(x[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("目前沒有 SHORT 候選。")

    with tab4:
        st.dataframe(out[display_cols], use_container_width=True, hide_index=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載目前掃描結果 CSV",
            data=csv_bytes,
            file_name="bybit_smc_scan_v02.csv",
            mime="text/csv",
        )

    with tab5:
        err = pd.DataFrame(st.session_state.get("scan_errors", []))
        if err.empty:
            st.success("目前沒有錯誤。")
        else:
            st.dataframe(err, use_container_width=True, hide_index=True)

    if remaining == 0:
        st.success(
            f"✅ 本輪全市場掃描完成：{total} 檔；"
            f"成功 {len(out)} 檔、錯誤 {len(st.session_state.get('scan_errors', []))} 檔。"
        )
