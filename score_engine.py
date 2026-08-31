from __future__ import annotations
import numpy as np
import pandas as pd
from indicators import prepare_indicators
from smc_engine import latest_smc

def score_symbol(h1, m15, m5, oi=None, ratio=None, funding=None):
    h1,m15,m5 = map(prepare_indicators,[h1,m15,m5])
    if min(len(h1),len(m15),len(m5))<40:
        return {"long_score":0,"short_score":0,"bias":"NEUTRAL","error":"K線不足"}
    s1,s15,s5 = latest_smc(h1), latest_smc(m15), latest_smc(m5)
    r1,r5 = h1.iloc[-1],m5.iloc[-1]
    L=S=0; ld=[]; sd=[]
    def al(msg,pts): 
        nonlocal L; L+=pts; ld.append((msg,pts))
    def as_(msg,pts):
        nonlocal S; S+=pts; sd.append((msg,pts))

    if s1["structure"]=="BULL": al("1H Bull Structure",10)
    if s1["structure"]=="BEAR": as_("1H Bear Structure",10)
    if s15["structure"]=="BULL": al("15m Bull Structure",8)
    if s15["structure"]=="BEAR": as_("15m Bear Structure",8)
    if r1["EMA20"]>r1["EMA60"]>r1["EMA200"]: al("EMA20>60>200",6)
    if r1["EMA20"]<r1["EMA60"]<r1["EMA200"]: as_("EMA20<60<200",6)
    if float(r1.get("EMA200_SLOPE",0) or 0)>0: al("EMA200 Slope",3)
    if float(r1.get("EMA200_SLOPE",0) or 0)<0: as_("EMA200 Slope",3)
    if pd.notna(r1.get("VWAP")) and r1["close"]>r1["VWAP"]: al("Price>VWAP",3)
    if pd.notna(r1.get("VWAP")) and r1["close"]<r1["VWAP"]: as_("Price<VWAP",3)

    if s5["bull_bos"]: al("5m Bull BOS",10)
    if s5["bear_bos"]: as_("5m Bear BOS",10)
    if s5["bull_sweep"]: al("Sell-side Sweep",8)
    if s5["bear_sweep"]: as_("Buy-side Sweep",8)
    if s5["bull_fvg"]: al("Bull FVG",8)
    if s5["bear_fvg"]: as_("Bear FVG",8)
    if s5["bull_displacement"]: al("Bull Displacement",6)
    if s5["bear_displacement"]: as_("Bear Displacement",6)

    rsi=float(r5.get("RSI14",np.nan))
    if np.isfinite(rsi) and 52<=rsi<=72: al("RSI Bull Regime",5)
    if np.isfinite(rsi) and 28<=rsi<=48: as_("RSI Bear Regime",5)
    hist=float(r5.get("MACD_HIST",np.nan))
    if np.isfinite(hist) and hist>0: al("MACD Hist > 0",4)
    if np.isfinite(hist) and hist<0: as_("MACD Hist < 0",4)
    adx=float(r5.get("ADX14",np.nan))
    pdi=float(r5.get("PLUS_DI",np.nan)); mdi=float(r5.get("MINUS_DI",np.nan))
    if np.isfinite(adx) and adx>=20 and pdi>mdi: al("ADX/+DI",5)
    if np.isfinite(adx) and adx>=20 and mdi>pdi: as_("ADX/-DI",5)
    rel=float(r5.get("REL_VOL",np.nan))
    if np.isfinite(rel) and rel>=1.2:
        (al if r5["close"]>=r5["open"] else as_)("Relative Volume",5)

    oi_change=0.0
    if oi is not None and len(oi)>=3:
        a=float(oi["openInterest"].iloc[-1]); b=float(oi["openInterest"].iloc[-3])
        oi_change=(a/b-1) if b else 0.0
        pc=float(m5["close"].iloc[-1]/m5["close"].iloc[-3]-1)
        if oi_change>0.002 and pc>0: al("Price↑ + OI↑",10)
        if oi_change>0.002 and pc<0: as_("Price↓ + OI↑",10)

    fr=np.nan
    if funding is not None and not funding.empty:
        fr=float(funding.iloc[-1]["fundingRate"])
        if fr>0.001: L-=5; as_("Long Crowded",3)
        elif fr<-0.001: S-=5; al("Short Crowded",3)
        else:
            if fr>=0: al("Funding Normal",4)
            else: as_("Funding Normal",4)

    br=sr=np.nan
    if ratio is not None and not ratio.empty:
        br=float(ratio.iloc[-1]["buyRatio"]); sr=float(ratio.iloc[-1]["sellRatio"])
        if 0.50<=br<=0.62: al("Account Ratio Bull",6)
        if 0.38<=br<0.50: as_("Account Ratio Bear",6)
        if br>0.70: L-=4
        if sr>0.70: S-=4

    L=int(max(0,min(150,round(L))))
    S=int(max(0,min(150,round(S))))
    bias="LONG" if L>=S+15 else ("SHORT" if S>=L+15 else "NEUTRAL")
    return {
        "long_score":L,"short_score":S,"bias":bias,
        "long_details":ld,"short_details":sd,
        "smc_1h":s1,"smc_15m":s15,"smc_5m":s5,
        "last_price":float(r5["close"]),"rsi14":rsi,"adx14":adx,"rel_vol":rel,
        "funding_rate":fr,"buy_ratio":br,"sell_ratio":sr,"oi_change":oi_change,
    }
