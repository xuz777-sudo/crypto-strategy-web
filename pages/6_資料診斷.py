import streamlit as st
import pandas as pd

from bybit_client import BybitClient

st.set_page_config(page_title="資料診斷", layout="wide")
st.title("Bybit 資料診斷 V0.1")
st.caption("Cloud Run 部署後先跑這一頁；8 項全部 PASS 再繼續開發 SMC。")

symbol = st.sidebar.text_input("測試交易對", "BTCUSDT").upper().strip()
client = BybitClient()

tests = []

def run_test(name, fn, detail_fn=None):
    try:
        value = fn()
        ok = value is not None
        if isinstance(value, pd.DataFrame):
            ok = not value.empty
        detail = detail_fn(value) if detail_fn else ""
        tests.append({
            "項目": name,
            "狀態": "PASS" if ok else "FAIL",
            "詳細": detail if ok else "無資料",
        })
        return value
    except Exception as e:
        tests.append({
            "項目": name,
            "狀態": "FAIL",
            "詳細": str(e),
        })
        return None

if st.button("執行完整診斷", type="primary"):
    server_time = run_test(
        "Bybit Server Time",
        client.server_time,
        lambda x: str(x),
    )

    inst = run_test(
        "USDT 永續商品清單",
        client.instruments_linear_usdt,
        lambda x: f"{len(x)} 檔",
    )

    tickers = run_test(
        f"{symbol} Ticker",
        client.tickers_linear,
        lambda x: (
            f"總Ticker {len(x)} 檔；"
            f"{symbol}={'有' if symbol in set(x['symbol']) else '無'}"
        ),
    )

    k5 = run_test(
        f"{symbol} 5分K",
        lambda: client.kline(symbol, "5", 120),
        lambda x: f"{len(x)} 根｜最新 {x['startTime'].iloc[-1]}",
    )

    k15 = run_test(
        f"{symbol} 15分K",
        lambda: client.kline(symbol, "15", 120),
        lambda x: f"{len(x)} 根｜最新 {x['startTime'].iloc[-1]}",
    )

    k60 = run_test(
        f"{symbol} 1H",
        lambda: client.kline(symbol, "60", 120),
        lambda x: f"{len(x)} 根｜最新 {x['startTime'].iloc[-1]}",
    )

    oi = run_test(
        f"{symbol} Open Interest",
        lambda: client.open_interest(symbol, "5min", 20),
        lambda x: f"{len(x)} 筆｜最新 OI={x['openInterest'].iloc[-1]}",
    )

    funding = run_test(
        f"{symbol} Funding",
        lambda: client.funding_history(symbol, 10),
        lambda x: f"{len(x)} 筆｜最新={x['fundingRate'].iloc[-1]:.8f}",
    )

    ratio = run_test(
        f"{symbol} Long/Short Ratio",
        lambda: client.long_short_ratio(symbol, "5min", 20),
        lambda x: (
            f"{len(x)} 筆｜Buy={x['buyRatio'].iloc[-1]:.4f} "
            f"Sell={x['sellRatio'].iloc[-1]:.4f}"
        ),
    )

    df = pd.DataFrame(tests)
    st.dataframe(df, use_container_width=True, hide_index=True)

    passes = int((df["狀態"] == "PASS").sum())
    total = len(df)

    if passes == total:
        st.success(f"資料診斷全部 PASS：{passes}/{total}")
    else:
        st.error(f"資料診斷未完成：PASS {passes}/{total}")

    st.info(
        "如果這裡仍出現 403，先不要修改策略程式；"
        "代表目前 Cloud Run 出站位置仍被 Bybit 拒絕。"
    )
