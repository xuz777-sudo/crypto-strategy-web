# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bybit_client import BybitClient
from score_engine import score_symbol
from smc_engine import analyze_smc


st.set_page_config(page_title="單幣分析", layout="wide")
st.title("單幣完整分析 V0.2.1｜SMC 多週期")
st.caption("1H / 15分K 判斷方向，5分K負責短線進出場；整合 BOS、CHoCH、FVG、Order Block、Liquidity Sweep、Premium / Discount。")


def fmt_price(v):
    if v is None:
        return "-"
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
        return str(v)


def find_column(df, candidates):
    """Case-insensitive column finder."""
    if df is None or df.empty:
        return None
    normalized = {str(c).replace("_", "").lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).replace("_", "").lower()
        if key in normalized:
            return normalized[key]
    return None


def bias_zh(v):
    mp = {
        "LONG": "偏多",
        "LEAN_LONG": "略偏多",
        "SHORT": "偏空",
        "LEAN_SHORT": "略偏空",
        "NEUTRAL": "中性",
        "bullish": "多頭",
        "bearish": "空頭",
        "neutral": "中性",
    }
    return mp.get(str(v), str(v))


def multi_tf_score(s5, s15, s60):
    # 5m = 進場週期；15m = 主要方向濾網；1H = 大方向。
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

    return long_score, short_score, bias


def trade_permission(final_bias, s5, s15, s60):
    b5 = s5["bias"]
    b15 = s15["bias"]
    b60 = s60["bias"]

    long_set = {"LONG", "LEAN_LONG"}
    short_set = {"SHORT", "LEAN_SHORT"}

    if final_bias in long_set:
        if b5 in long_set and b15 not in short_set and b60 != "SHORT":
            return "LONG 可等待 5分K 觸發", "✅"
        return "LONG 條件未完全同步", "⚠️"

    if final_bias in short_set:
        if b5 in short_set and b15 not in long_set and b60 != "LONG":
            return "SHORT 可等待 5分K 觸發", "✅"
        return "SHORT 條件未完全同步", "⚠️"

    return "觀望，不追單", "⏸️"


def structure_text(smc):
    ev = smc.get("latest_structure")
    if not ev:
        return "尚無"
    return f'{ev.get("side","-")} {ev.get("type","-")} @ {fmt_price(ev.get("level"))}'


def sweep_text(smc):
    ev = smc.get("latest_liquidity_sweep")
    if not ev:
        return "尚無"
    side = ev.get("side", "-")
    bias = ev.get("bias", "-")
    return f"{side} → {bias} @ {fmt_price(ev.get('level'))}"


symbol = st.sidebar.text_input("交易對", "BTCUSDT").upper().strip()
st.sidebar.caption("目前版本先使用 Bybit USDT 永續合約。")

refresh = st.sidebar.button("重新分析", use_container_width=True)

if not symbol:
    st.warning("請輸入交易對，例如 BTCUSDT")
    st.stop()

c = BybitClient()

try:
    with st.spinner(f"正在分析 {symbol} ..."):
        h1 = c.kline(symbol, "60", 350)
        m15 = c.kline(symbol, "15", 350)
        m5 = c.kline(symbol, "5", 350)

        if h1.empty or m15.empty or m5.empty:
            raise RuntimeError("K 線資料不足，請稍後再試。")

        oi = c.open_interest(symbol, "5min", 20)
        ratio = c.long_short_ratio(symbol, "5min", 20)
        funding = c.funding_history(symbol, 20)

        # 原本系統分數保留，方便後續與 SMC 綜合。
        base_score = score_symbol(h1, m15, m5, oi, ratio, funding)

        # SMC 多週期
        smc_1h = analyze_smc(h1)
        smc_15 = analyze_smc(m15)
        smc_5 = analyze_smc(m5)

        mtf_long, mtf_short, mtf_bias = multi_tf_score(smc_5, smc_15, smc_1h)
        permission, permission_icon = trade_permission(mtf_bias, smc_5, smc_15, smc_1h)

    # ------------------------------------------------------------------
    # Top summary
    # ------------------------------------------------------------------
    st.subheader(f"{symbol}｜多週期 SMC 結論")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("目前價", fmt_price(smc_5["last_price"]))
    c2.metric("MTF LONG", mtf_long)
    c3.metric("MTF SHORT", mtf_short)
    c4.metric("方向", bias_zh(mtf_bias))
    c5.metric("5分K信心", smc_5["confidence"])

    if permission_icon == "✅":
        st.success(f"{permission_icon} 進場狀態：{permission}")
    elif permission_icon == "⚠️":
        st.warning(f"{permission_icon} 進場狀態：{permission}")
    else:
        st.info(f"{permission_icon} 進場狀態：{permission}")

    # ------------------------------------------------------------------
    # Timeframe cards
    # ------------------------------------------------------------------
    st.markdown("### 多週期方向")
    tf_rows = pd.DataFrame(
        [
            {
                "週期": "1H 大方向",
                "Bias": bias_zh(smc_1h["bias"]),
                "結構": bias_zh(smc_1h["structure_state"]),
                "LONG": smc_1h["long_score"],
                "SHORT": smc_1h["short_score"],
                "最新 BOS / CHoCH": structure_text(smc_1h),
            },
            {
                "週期": "15分K 濾網",
                "Bias": bias_zh(smc_15["bias"]),
                "結構": bias_zh(smc_15["structure_state"]),
                "LONG": smc_15["long_score"],
                "SHORT": smc_15["short_score"],
                "最新 BOS / CHoCH": structure_text(smc_15),
            },
            {
                "週期": "5分K 進出場",
                "Bias": bias_zh(smc_5["bias"]),
                "結構": bias_zh(smc_5["structure_state"]),
                "LONG": smc_5["long_score"],
                "SHORT": smc_5["short_score"],
                "最新 BOS / CHoCH": structure_text(smc_5),
            },
        ]
    )
    st.dataframe(tf_rows, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 5m entry levels
    # ------------------------------------------------------------------
    st.markdown("### 5分K 進出場參考")
    levels = smc_5.get("trade_levels", {}) or {}

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("建議進場參考", fmt_price(levels.get("entry")))
    e2.metric("停損參考", fmt_price(levels.get("stop_loss")))
    e3.metric("TP1｜1.5R", fmt_price(levels.get("tp1")))
    e4.metric("TP2｜2.5R", fmt_price(levels.get("tp2")))

    if mtf_bias == "NEUTRAL":
        st.caption("目前多週期未形成足夠一致性，以上價位只作結構參考，不視為有效進場訊號。")
    else:
        st.caption("正式策略仍需等 5 分 K 觸發條件成立；停損與停利會在後續 strategy_engine / 回測模組統一管理。")

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["5分K SMC 圖表", "SMC 訊號明細", "原策略評分", "市場資料"]
    )

    with tab1:
        p = m5.tail(120).copy()

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=p["startTime"],
                    open=p["open"],
                    high=p["high"],
                    low=p["low"],
                    close=p["close"],
                    name="5m",
                )
            ]
        )

        line_defs = [
            ("Entry", levels.get("entry"), "dash"),
            ("Stop", levels.get("stop_loss"), "dot"),
            ("TP1", levels.get("tp1"), "dash"),
            ("TP2", levels.get("tp2"), "dash"),
        ]
        for label, value, dash in line_defs:
            if value is not None:
                try:
                    fig.add_hline(
                        y=float(value),
                        line_dash=dash,
                        annotation_text=label,
                        annotation_position="top left",
                    )
                except Exception:
                    pass

        # Dealing range equilibrium
        dr = smc_5.get("dealing_range", {}) or {}
        if dr.get("equilibrium") is not None:
            try:
                fig.add_hline(
                    y=float(dr["equilibrium"]),
                    line_dash="dot",
                    annotation_text="EQ 50%",
                    annotation_position="bottom left",
                )
            except Exception:
                pass

        fig.update_layout(
            height=620,
            xaxis_rangeslider_visible=False,
            title=f"{symbol} 5分K｜SMC 執行週期",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"5分K 最新結構：{structure_text(smc_5)} ｜ "
            f"流動性掃描：{sweep_text(smc_5)} ｜ "
            f"區域：{str((smc_5.get('dealing_range') or {}).get('zone','-')).upper()}"
        )

    with tab2:
        left, right = st.columns(2)

        with left:
            st.markdown("#### LONG SMC 依據")
            if smc_5["long_reasons"]:
                for item in smc_5["long_reasons"]:
                    st.write(f"• {item}")
            else:
                st.caption("目前沒有明確 LONG SMC 加分條件。")

            st.markdown("#### Active Bullish FVG")
            bull_fvg = [
                x for x in smc_5.get("active_fvgs", [])
                if x.get("side") == "bullish"
            ]
            if bull_fvg:
                st.dataframe(
                    pd.DataFrame(bull_fvg)[["index", "low", "high", "size"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("目前沒有有效 Bullish FVG。")

            st.markdown("#### Active Bullish Order Block")
            bull_ob = [
                x for x in smc_5.get("active_order_blocks", [])
                if x.get("side") == "bullish"
            ]
            if bull_ob:
                st.dataframe(
                    pd.DataFrame(bull_ob)[["index", "created_at", "low", "high", "mid"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("目前沒有有效 Bullish Order Block。")

        with right:
            st.markdown("#### SHORT SMC 依據")
            if smc_5["short_reasons"]:
                for item in smc_5["short_reasons"]:
                    st.write(f"• {item}")
            else:
                st.caption("目前沒有明確 SHORT SMC 加分條件。")

            st.markdown("#### Active Bearish FVG")
            bear_fvg = [
                x for x in smc_5.get("active_fvgs", [])
                if x.get("side") == "bearish"
            ]
            if bear_fvg:
                st.dataframe(
                    pd.DataFrame(bear_fvg)[["index", "low", "high", "size"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("目前沒有有效 Bearish FVG。")

            st.markdown("#### Active Bearish Order Block")
            bear_ob = [
                x for x in smc_5.get("active_order_blocks", [])
                if x.get("side") == "bearish"
            ]
            if bear_ob:
                st.dataframe(
                    pd.DataFrame(bear_ob)[["index", "created_at", "low", "high", "mid"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("目前沒有有效 Bearish Order Block。")

        st.markdown("#### 5分K SMC 計數")
        counts = smc_5.get("counts", {}) or {}
        st.dataframe(
            pd.DataFrame(
                [{"項目": k, "數量": v} for k, v in counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.markdown("#### 原策略 score_engine 輔助評分")
        st.info("此區是原本技術指標/籌碼輔助分數；若與頁首多週期 SMC 結論不同，實際進場判斷以 MTF SMC 為主。")
        a, b, d, e = st.columns(4)
        a.metric("目前價", fmt_price(base_score["last_price"]))
        b.metric("LONG", base_score["long_score"])
        d.metric("SHORT", base_score["short_score"])
        e.metric("Bias", base_score["bias"])

        l, r = st.columns(2)
        with l:
            st.dataframe(
                pd.DataFrame(base_score["long_details"], columns=["LONG條件", "分數"]),
                use_container_width=True,
                hide_index=True,
            )
        with r:
            st.dataframe(
                pd.DataFrame(base_score["short_details"], columns=["SHORT條件", "分數"]),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.markdown("#### 目前市場衍生資料")
        d1, d2, d3 = st.columns(3)

        if not oi.empty and "openInterest" in oi.columns:
            d1.metric("Open Interest", fmt_price(oi["openInterest"].iloc[0]))
        else:
            d1.metric("Open Interest", "-")

        # Bybit Long/Short Ratio：
        # 不可以直接抓第一個非 timestamp 欄位，因為第一欄常常是 symbol，
        # 會造成畫面錯誤顯示 BTCUSDT。這裡明確抓 buyRatio / sellRatio。
        if not ratio.empty:
            buy_col = find_column(ratio, ["buyRatio", "buy_ratio", "buy"])
            sell_col = find_column(ratio, ["sellRatio", "sell_ratio", "sell"])
            ratio_col = find_column(ratio, ["longShortRatio", "long_short_ratio", "ratio"])

            buy_value = None
            sell_value = None
            ls_value = None

            try:
                if buy_col is not None:
                    buy_value = float(ratio.iloc[0][buy_col])
                if sell_col is not None:
                    sell_value = float(ratio.iloc[0][sell_col])
                if buy_value is not None and sell_value is not None and sell_value != 0:
                    ls_value = buy_value / sell_value
                elif ratio_col is not None:
                    ls_value = float(ratio.iloc[0][ratio_col])
            except Exception:
                ls_value = None

            if ls_value is not None and math.isfinite(ls_value):
                d2.metric("Long/Short Ratio", f"{ls_value:.4f}")
                if buy_value is not None and sell_value is not None:
                    d2.caption(
                        f"Buy {buy_value * 100:.2f}% ｜ Sell {sell_value * 100:.2f}%"
                    )
            elif buy_value is not None and sell_value is not None:
                d2.metric("Long/Short Ratio", "-")
                d2.caption(
                    f"Buy {buy_value * 100:.2f}% ｜ Sell {sell_value * 100:.2f}%"
                )
            else:
                d2.metric("Long/Short Ratio", "-")
        else:
            d2.metric("Long/Short Ratio", "-")

        # Funding 以百分比顯示，0.0001 -> 0.0100%
        if not funding.empty:
            funding_col = find_column(funding, ["fundingRate", "funding_rate"])
            if funding_col is not None:
                try:
                    funding_value = float(funding.iloc[0][funding_col])
                    if math.isfinite(funding_value):
                        d3.metric("Funding", f"{funding_value * 100:.4f}%")
                        d3.caption(f"Raw: {funding_value:.8f}")
                    else:
                        d3.metric("Funding", "-")
                except Exception:
                    d3.metric("Funding", "-")
            else:
                d3.metric("Funding", "-")
        else:
            d3.metric("Funding", "-")

        with st.expander("查看 5分K SMC 完整摘要"):
            raw = {k: v for k, v in smc_5.items() if k != "enriched_df"}
            st.json(raw)

except Exception as e:
    st.error(f"{symbol} 分析失敗：{e}")
    st.exception(e)
