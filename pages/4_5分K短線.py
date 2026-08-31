import streamlit as st
from bybit_client import BybitClient
from score_engine import score_symbol
from strategy_engine import build_trade_plan

st.set_page_config(page_title="5分K短線",layout="wide")
st.title("5分K短線進出場 V0.1")
symbol=st.sidebar.text_input("交易對","BTCUSDT").upper().strip()
min_score=st.sidebar.number_input("最低觸發分數",60,150,90,5)
c=BybitClient()
try:
    h1=c.kline(symbol,"60",350); m15=c.kline(symbol,"15",350); m5=c.kline(symbol,"5",350)
    oi=c.open_interest(symbol,"5min",20); ratio=c.long_short_ratio(symbol,"5min",20); f=c.funding_history(symbol,20)
    s=score_symbol(h1,m15,m5,oi,ratio,f); p=build_trade_plan(symbol,m5,s,min_score)
    a,b,d,e=st.columns(4); a.metric("LONG",s["long_score"]); b.metric("SHORT",s["short_score"]); d.metric("Bias",s["bias"]); e.metric("狀態",p["status"])
    st.metric("目前/進場參考",f"{p.get('entry',0):.8g}")
    if p["status"] in {"LONG_TRIGGER","SHORT_TRIGGER"}:
        x,y,z=st.columns(3); x.metric("Stop",f"{p['stop']:.8g}"); y.metric("TP1",f"{p['tp1']:.8g}"); z.metric("TP2",f"{p['tp2']:.8g}")
    else: st.info("尚未觸發正式進場，等待 SMC + 分數共振。")
except Exception as e:
    st.error(f"短線分析失敗：{e}")
