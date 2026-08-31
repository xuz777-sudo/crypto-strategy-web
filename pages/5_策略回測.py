from datetime import datetime,timedelta,timezone
import streamlit as st
import pandas as pd
from bybit_client import BybitClient
from backtest_engine import BacktestConfig,run_backtest

st.set_page_config(page_title="策略回測",layout="wide")
st.title("Crypto 5分K策略回測 V0.1")
symbol=st.sidebar.text_input("交易對","BTCUSDT").upper().strip()
days=st.sidebar.selectbox("回測期間",[7,14,30,60],index=0,format_func=lambda x:f"近{x}天")
use_long=st.sidebar.checkbox("LONG",True); use_short=st.sidebar.checkbox("SHORT",True)
fee=st.sidebar.number_input("單邊手續費 %",0.0,1.0,0.055,0.005)/100
slip=st.sidebar.number_input("單邊滑價 %",0.0,1.0,0.020,0.005)/100
if st.button("開始回測",type="primary"):
    try:
        c=BybitClient(); end=datetime.now(timezone.utc); start=end-timedelta(days=int(days))
        df=c.kline_range(symbol,"5",int(start.timestamp()*1000),int(end.timestamp()*1000))
        t,s=run_backtest(df,BacktestConfig(fee_pct=fee,slippage_pct=slip,use_long=use_long,use_short=use_short))
        a,b,d,e=st.columns(4); a.metric("交易",s["trades"])
        b.metric("勝率","N/A" if pd.isna(s["win_rate"]) else f"{s['win_rate']:.1f}%")
        d.metric("平均淨報酬","N/A" if pd.isna(s["avg_return"]) else f"{s['avg_return']:+.2f}%")
        e.metric("Profit Factor","N/A" if pd.isna(s["profit_factor"]) else ("∞" if s["profit_factor"]==float("inf") else f"{s['profit_factor']:.2f}"))
        if t.empty: st.warning("期間內沒有符合條件的交易。")
        else: st.dataframe(t,use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"回測失敗：{e}")
