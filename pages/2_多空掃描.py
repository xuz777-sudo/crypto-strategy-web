import time
import streamlit as st
import pandas as pd
from bybit_client import BybitClient
from score_engine import score_symbol

st.set_page_config(page_title="多空掃描",layout="wide")
st.title("全市場多空掃描 V0.1")
client=BybitClient()
n=st.sidebar.number_input("候選檔數",5,30,10,5)
if st.button("開始掃描",type="primary"):
    try:
        t=client.tickers_linear().sort_values("turnover24h",ascending=False).head(int(n))
        rows=[]; bar=st.progress(0.0)
        for i,r in enumerate(t.itertuples(index=False),1):
            s=r.symbol
            try:
                h1=client.kline(s,"60",300); m15=client.kline(s,"15",300); m5=client.kline(s,"5",300)
                oi=client.open_interest(s,"5min",10); ratio=client.long_short_ratio(s,"5min",10); f=client.funding_history(s,5)
                z=score_symbol(h1,m15,m5,oi,ratio,f)
                rows.append({"Symbol":s,"LONG":z["long_score"],"SHORT":z["short_score"],"Bias":z["bias"],"Last":z["last_price"]})
            except Exception as e:
                rows.append({"Symbol":s,"Bias":"ERROR","錯誤":str(e)})
            bar.progress(i/len(t)); time.sleep(0.05)
        out=pd.DataFrame(rows)
        if "LONG" in out.columns:
            out["MAX"]=out[["LONG","SHORT"]].max(axis=1)
            out=out.sort_values("MAX",ascending=False)
        st.dataframe(out,use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"掃描失敗：{e}")
