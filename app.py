import streamlit as st

st.set_page_config(
    page_title="Crypto Strategy Web V0.1",
    page_icon="₿",
    layout="wide",
)

st.title("Crypto Strategy Web V0.1")
st.caption(
    "Bybit USDT 永續｜SMC + 趨勢 + 動能 + 成交量 + OI/Funding/多空比｜5分K短線｜基礎回測"
)

st.info(
    "V0.1 先完成資料核心與策略骨架。使用 Bybit 公開 Market API，不需要 API Key，也不會下真實訂單。"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("市場", "USDT Perpetual")
c2.metric("主要週期", "1H / 15m / 5m")
c3.metric("執行週期", "5m")
c4.metric("交易", "模擬 / 回測")

st.subheader("第一版頁面")
st.markdown("""
- **市場總覽**：交易對、24h成交額、漲跌、Funding、OI。
- **多空掃描**：LONG / SHORT 雙向分數。
- **單幣分析**：1H / 15m / 5m + SMC + 衍生品籌碼。
- **5分K短線**：Entry / SL / TP1 / TP2。
- **策略回測**：5分K非重繪 Swing/BOS 技術核心回測。
""")
