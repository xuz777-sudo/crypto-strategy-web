import streamlit as st
from bybit_client import BybitClient

st.set_page_config(page_title="市場總覽",layout="wide")
st.title("幣圈市場總覽 V0.1")
client=BybitClient()

@st.cache_data(ttl=60,show_spinner=False)
def load():
    return client.instruments_linear_usdt(),client.tickers_linear()

try:
    inst,t=load()
    allowed=set(inst["symbol"]) if not inst.empty else set()
    if allowed: t=t[t["symbol"].isin(allowed)].copy()
    t=t.sort_values("turnover24h",ascending=False)
    c1,c2=st.columns(2); c1.metric("USDT永續",len(inst)); c2.metric("Ticker",len(t))
    show=t[["symbol","lastPrice","price24hPcnt","turnover24h","openInterestValue","fundingRate"]].copy()
    show["price24hPcnt"]=show["price24hPcnt"]*100
    show["fundingRate"]=show["fundingRate"]*100
    st.dataframe(show.head(100),use_container_width=True,hide_index=True)
except Exception as e:
    st.error(f"Bybit資料讀取失敗：{e}")
