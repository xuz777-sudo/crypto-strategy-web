import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from bybit_client import BybitClient
from score_engine import score_symbol

st.set_page_config(page_title="單幣分析",layout="wide")
st.title("單幣完整分析 V0.1")
symbol=st.sidebar.text_input("交易對","BTCUSDT").upper().strip()
c=BybitClient()
try:
    h1=c.kline(symbol,"60",350); m15=c.kline(symbol,"15",350); m5=c.kline(symbol,"5",350)
    oi=c.open_interest(symbol,"5min",20); ratio=c.long_short_ratio(symbol,"5min",20); f=c.funding_history(symbol,20)
    s=score_symbol(h1,m15,m5,oi,ratio,f)
    a,b,d,e=st.columns(4)
    a.metric("目前價",f"{s['last_price']:.8g}"); b.metric("LONG",s["long_score"]); d.metric("SHORT",s["short_score"]); e.metric("Bias",s["bias"])
    p=m5.tail(120)
    fig=go.Figure(data=[go.Candlestick(x=p["startTime"],open=p["open"],high=p["high"],low=p["low"],close=p["close"])])
    fig.update_layout(height=500,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    l,r=st.columns(2)
    with l: st.dataframe(pd.DataFrame(s["long_details"],columns=["LONG條件","分數"]),use_container_width=True,hide_index=True)
    with r: st.dataframe(pd.DataFrame(s["short_details"],columns=["SHORT條件","分數"]),use_container_width=True,hide_index=True)
except Exception as e:
    st.error(f"{symbol} 分析失敗：{e}")
