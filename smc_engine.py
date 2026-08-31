from __future__ import annotations
import numpy as np
import pandas as pd

def confirmed_swings(df, left=2, right=2):
    x = df.copy()
    x["swing_high"] = np.nan
    x["swing_low"] = np.nan
    x["swing_high_confirmed_at"] = pd.NaT
    x["swing_low_confirmed_at"] = pd.NaT
    for i in range(left, len(x)-right):
        h, l = x["high"].iloc[i], x["low"].iloc[i]
        if h > x["high"].iloc[i-left:i].max() and h >= x["high"].iloc[i+1:i+right+1].max():
            x.at[x.index[i],"swing_high"] = h
            x.at[x.index[i],"swing_high_confirmed_at"] = x["startTime"].iloc[i+right]
        if l < x["low"].iloc[i-left:i].min() and l <= x["low"].iloc[i+1:i+right+1].min():
            x.at[x.index[i],"swing_low"] = l
            x.at[x.index[i],"swing_low_confirmed_at"] = x["startTime"].iloc[i+right]
    return x

def structure_state(df):
    x = confirmed_swings(df)
    now = x["startTime"].iloc[-1]
    hs = x[x["swing_high"].notna() & (x["swing_high_confirmed_at"]<=now)]
    ls = x[x["swing_low"].notna() & (x["swing_low_confirmed_at"]<=now)]
    if len(hs)<2 or len(ls)<2:
        return "RANGE"
    h1,h2 = hs.iloc[-2]["swing_high"], hs.iloc[-1]["swing_high"]
    l1,l2 = ls.iloc[-2]["swing_low"], ls.iloc[-1]["swing_low"]
    if h2>h1 and l2>l1: return "BULL"
    if h2<h1 and l2<l1: return "BEAR"
    return "RANGE"

def latest_smc(df):
    if df is None or len(df)<20:
        return {"structure":"RANGE","bull_bos":False,"bear_bos":False,
                "bull_sweep":False,"bear_sweep":False,"bull_fvg":False,
                "bear_fvg":False,"bull_displacement":False,"bear_displacement":False,
                "last_swing_high":np.nan,"last_swing_low":np.nan}
    x = confirmed_swings(df)
    row, prev = x.iloc[-1], x.iloc[-2]
    now = row["startTime"]
    hs = x.iloc[:-1]
    hs = hs[hs["swing_high"].notna() & (hs["swing_high_confirmed_at"]<=now)]
    ls = x.iloc[:-1]
    ls = ls[ls["swing_low"].notna() & (ls["swing_low_confirmed_at"]<=now)]
    lh = float(hs.iloc[-1]["swing_high"]) if not hs.empty else np.nan
    ll = float(ls.iloc[-1]["swing_low"]) if not ls.empty else np.nan
    c, pc = float(row["close"]), float(prev["close"])
    bull_bos = np.isfinite(lh) and pc<=lh and c>lh
    bear_bos = np.isfinite(ll) and pc>=ll and c<ll
    bull_sweep = np.isfinite(ll) and float(row["low"])<ll and c>ll
    bear_sweep = np.isfinite(lh) and float(row["high"])>lh and c<lh
    a = x.iloc[-3] if len(x)>=3 else row
    bull_fvg = float(row["low"])>float(a["high"])
    bear_fvg = float(row["high"])<float(a["low"])
    atr = float(row.get("ATR14",np.nan))
    body = abs(float(row["close"])-float(row["open"]))
    bull_disp = np.isfinite(atr) and atr>0 and body>=1.2*atr and c>float(row["open"])
    bear_disp = np.isfinite(atr) and atr>0 and body>=1.2*atr and c<float(row["open"])
    return {
        "structure":structure_state(x),"bull_bos":bool(bull_bos),"bear_bos":bool(bear_bos),
        "bull_sweep":bool(bull_sweep),"bear_sweep":bool(bear_sweep),
        "bull_fvg":bool(bull_fvg),"bear_fvg":bool(bear_fvg),
        "bull_displacement":bool(bull_disp),"bear_displacement":bool(bear_disp),
        "last_swing_high":lh,"last_swing_low":ll,
    }
