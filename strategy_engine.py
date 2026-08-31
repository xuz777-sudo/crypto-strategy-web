from __future__ import annotations
import numpy as np
from indicators import prepare_indicators
from smc_engine import latest_smc

def build_trade_plan(symbol, df_5m, score, min_score=90, atr_buffer=0.25):
    x=prepare_indicators(df_5m)
    if x.empty or len(x)<30: return {"status":"NO_DATA"}
    r=x.iloc[-1]; smc=latest_smc(x); price=float(r["close"]); a=float(r.get("ATR14",np.nan))
    long_ok=score.get("long_score",0)>=min_score and (smc["bull_bos"] or smc["bull_sweep"] or smc["bull_fvg"])
    short_ok=score.get("short_score",0)>=min_score and (smc["bear_bos"] or smc["bear_sweep"] or smc["bear_fvg"])
    if long_ok:
        sl=smc.get("last_swing_low",np.nan)
        if not np.isfinite(sl) or sl>=price: sl=price-(1.2*a if np.isfinite(a) else price*0.01)
        else: sl=sl-(a*atr_buffer if np.isfinite(a) else 0)
        risk=max(price-sl,price*0.001)
        return {"symbol":symbol,"status":"LONG_TRIGGER","entry":price,"stop":sl,
                "tp1":price+1.5*risk,"tp2":price+2.5*risk}
    if short_ok:
        sl=smc.get("last_swing_high",np.nan)
        if not np.isfinite(sl) or sl<=price: sl=price+(1.2*a if np.isfinite(a) else price*0.01)
        else: sl=sl+(a*atr_buffer if np.isfinite(a) else 0)
        risk=max(sl-price,price*0.001)
        return {"symbol":symbol,"status":"SHORT_TRIGGER","entry":price,"stop":sl,
                "tp1":price-1.5*risk,"tp2":price-2.5*risk}
    return {"symbol":symbol,"status":f"{score.get('bias','NEUTRAL')}_WATCH","entry":price}
