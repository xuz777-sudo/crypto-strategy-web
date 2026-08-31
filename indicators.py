from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ad = dn.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, period=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def prepare_indicators(df):
    x = df.copy()
    for n in [20,60,200]:
        x[f"EMA{n}"] = ema(x["close"], n)
    x["RSI14"] = rsi(x["close"], 14)
    x["ATR14"] = atr(x, 14)
    fast, slow = ema(x["close"],12), ema(x["close"],26)
    x["MACD"] = fast-slow
    x["MACD_SIGNAL"] = ema(x["MACD"],9)
    x["MACD_HIST"] = x["MACD"]-x["MACD_SIGNAL"]
    x["VOL_MA20"] = x["volume"].rolling(20, min_periods=20).mean()
    x["REL_VOL"] = x["volume"] / x["VOL_MA20"].replace(0,np.nan)
    x["EMA200_SLOPE"] = x["EMA200"].pct_change(5)

    up = x["high"].diff()
    down = -x["low"].diff()
    plus_dm = pd.Series(np.where((up>down)&(up>0),up,0.0), index=x.index)
    minus_dm = pd.Series(np.where((down>up)&(down>0),down,0.0), index=x.index)
    a = x["ATR14"]
    x["PLUS_DI"] = 100*plus_dm.ewm(alpha=1/14,adjust=False).mean()/a.replace(0,np.nan)
    x["MINUS_DI"] = 100*minus_dm.ewm(alpha=1/14,adjust=False).mean()/a.replace(0,np.nan)
    dx = 100*(x["PLUS_DI"]-x["MINUS_DI"]).abs()/(x["PLUS_DI"]+x["MINUS_DI"]).replace(0,np.nan)
    x["ADX14"] = dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

    day = x["startTime"].dt.floor("D")
    typical = (x["high"]+x["low"]+x["close"])/3
    x["VWAP"] = (typical*x["volume"]).groupby(day).cumsum()/x["volume"].groupby(day).cumsum().replace(0,np.nan)
    return x
