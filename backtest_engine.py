from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from indicators import prepare_indicators
from smc_engine import confirmed_swings

@dataclass
class BacktestConfig:
    fee_pct: float=0.00055
    slippage_pct: float=0.00020
    stop_atr: float=1.2
    target_r: float=2.0
    use_long: bool=True
    use_short: bool=True

def _frame(df):
    x=confirmed_swings(prepare_indicators(df))
    ch=pd.Series(np.nan,index=x.index); cl=pd.Series(np.nan,index=x.index)
    for _,r in x.iterrows():
        if pd.notna(r["swing_high_confirmed_at"]):
            idx=x.index[x["startTime"]==r["swing_high_confirmed_at"]]
            if len(idx): ch.loc[idx[0]]=r["swing_high"]
        if pd.notna(r["swing_low_confirmed_at"]):
            idx=x.index[x["startTime"]==r["swing_low_confirmed_at"]]
            if len(idx): cl.loc[idx[0]]=r["swing_low"]
    x["CONF_HIGH"]=ch.ffill(); x["CONF_LOW"]=cl.ffill()
    x["LONG_SIGNAL"]=(x["close"]>x["CONF_HIGH"])&(x["close"].shift(1)<=x["CONF_HIGH"].shift(1))&(x["EMA20"]>x["EMA60"])&(x["MACD_HIST"]>0)&(x["ADX14"]>=20)
    x["SHORT_SIGNAL"]=(x["close"]<x["CONF_LOW"])&(x["close"].shift(1)>=x["CONF_LOW"].shift(1))&(x["EMA20"]<x["EMA60"])&(x["MACD_HIST"]<0)&(x["ADX14"]>=20)
    return x

def run_backtest(df_5m,cfg):
    x=_frame(df_5m); trades=[]; i=1
    while i<len(x)-1:
        sig=x.iloc[i]; side=None
        if cfg.use_long and bool(sig.get("LONG_SIGNAL",False)): side="LONG"
        elif cfg.use_short and bool(sig.get("SHORT_SIGNAL",False)): side="SHORT"
        if side is None: i+=1; continue
        ei=i+1; er=x.iloc[ei]; raw=float(er["open"])
        entry=raw*(1+cfg.slippage_pct if side=="LONG" else 1-cfg.slippage_pct)
        a=float(er.get("ATR14",np.nan))
        if not np.isfinite(a) or a<=0: i+=1; continue
        risk=cfg.stop_atr*a
        stop=entry-risk if side=="LONG" else entry+risk
        target=entry+cfg.target_r*risk if side=="LONG" else entry-cfg.target_r*risk
        xi=None; xp=None; reason=None
        for j in range(ei,len(x)):
            r=x.iloc[j]
            if side=="LONG":
                sh=float(r["low"])<=stop; th=float(r["high"])>=target
            else:
                sh=float(r["high"])>=stop; th=float(r["low"])<=target
            if sh and th: xi=j; xp=stop; reason="同K雙觸發→保守停損"; break
            if sh: xi=j; xp=stop; reason="停損"; break
            if th: xi=j; xp=target; reason="停利"; break
        if xi is None: xi=len(x)-1; xp=float(x.iloc[xi]["close"]); reason="回測期末"
        gross=(xp/entry-1) if side=="LONG" else (entry/xp-1)
        net=gross-2*cfg.fee_pct
        trades.append({"side":side,"signal_time":sig["startTime"],"entry_time":er["startTime"],
                       "exit_time":x.iloc[xi]["startTime"],"entry":entry,"stop":stop,"target":target,
                       "exit":xp,"net_return_pct":net*100,"reason":reason})
        i=max(xi+1,i+1)
    t=pd.DataFrame(trades)
    if t.empty: return t,{"trades":0,"win_rate":np.nan,"avg_return":np.nan,"profit_factor":np.nan}
    wins=t.loc[t["net_return_pct"]>0,"net_return_pct"].sum()
    losses=-t.loc[t["net_return_pct"]<0,"net_return_pct"].sum()
    return t,{"trades":len(t),"win_rate":(t["net_return_pct"]>0).mean()*100,
              "avg_return":t["net_return_pct"].mean(),
              "profit_factor":wins/losses if losses>0 else np.inf}
