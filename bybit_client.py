from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import requests
import pandas as pd

BASE_URL = "https://api.bybit.com"

class BybitAPIError(RuntimeError):
    pass

@dataclass
class BybitClient:
    base_url: str = BASE_URL
    timeout: int = 15

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self.base_url.rstrip("/") + path
        r = requests.get(url, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        obj = r.json()
        if int(obj.get("retCode", -1)) != 0:
            raise BybitAPIError(
                f"{path}: retCode={obj.get('retCode')} retMsg={obj.get('retMsg')}"
            )
        return obj

    def server_time(self) -> datetime:
        obj = self._get("/v5/market/time")
        return datetime.fromtimestamp(int(obj["time"]) / 1000, tz=timezone.utc)

    def instruments_linear_usdt(self) -> pd.DataFrame:
        rows, cursor = [], ""
        while True:
            params = {"category": "linear", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            obj = self._get("/v5/market/instruments-info", params)
            result = obj.get("result", {})
            rows.extend(result.get("list", []))
            cursor = result.get("nextPageCursor", "") or ""
            if not cursor:
                break

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        keep = (
            (df.get("quoteCoin") == "USDT")
            & (df.get("contractType") == "LinearPerpetual")
            & (df.get("status") == "Trading")
        )
        return df.loc[keep].reset_index(drop=True)

    def tickers_linear(self) -> pd.DataFrame:
        obj = self._get("/v5/market/tickers", {"category": "linear"})
        df = pd.DataFrame(obj.get("result", {}).get("list", []))
        if df.empty:
            return df
        for c in [
            "lastPrice","price24hPcnt","turnover24h","volume24h",
            "openInterest","openInterestValue","fundingRate"
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def kline(self, symbol: str, interval: str = "5", limit: int = 300,
              start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        params = {
            "category": "linear",
            "symbol": symbol.upper(),
            "interval": str(interval),
            "limit": min(max(int(limit), 1), 1000),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        obj = self._get("/v5/market/kline", params)
        rows = obj.get("result", {}).get("list", [])
        cols = ["startTime","open","high","low","close","volume","turnover"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        df["startTime"] = pd.to_datetime(
            pd.to_numeric(df["startTime"], errors="coerce"), unit="ms", utc=True
        )
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("startTime").drop_duplicates("startTime").reset_index(drop=True)

    def kline_range(self, symbol: str, interval: str, start_ms: int, end_ms: int,
                    pause: float = 0.06) -> pd.DataFrame:
        parts, cursor_end = [], int(end_ms)
        while cursor_end >= int(start_ms):
            part = self.kline(symbol, interval, 1000, start_ms, cursor_end)
            if part.empty:
                break
            parts.append(part)
            earliest = int(part["startTime"].min().timestamp() * 1000)
            if earliest <= int(start_ms):
                break
            new_end = earliest - 1
            if new_end >= cursor_end:
                break
            cursor_end = new_end
            time.sleep(pause)

        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates("startTime").sort_values("startTime").reset_index(drop=True)
        ms = out["startTime"].astype("int64") // 10**6
        return out.loc[(ms >= int(start_ms)) & (ms <= int(end_ms))].reset_index(drop=True)

    def open_interest(self, symbol: str, interval_time: str = "5min",
                      limit: int = 20) -> pd.DataFrame:
        obj = self._get("/v5/market/open-interest", {
            "category": "linear", "symbol": symbol.upper(),
            "intervalTime": interval_time, "limit": min(max(int(limit),1),200)
        })
        df = pd.DataFrame(obj.get("result", {}).get("list", []))
        if not df.empty:
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
            df["timestamp"] = pd.to_datetime(
                pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True
            )
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def long_short_ratio(self, symbol: str, period: str = "5min",
                         limit: int = 20) -> pd.DataFrame:
        obj = self._get("/v5/market/account-ratio", {
            "category": "linear", "symbol": symbol.upper(),
            "period": period, "limit": min(max(int(limit),1),500)
        })
        df = pd.DataFrame(obj.get("result", {}).get("list", []))
        if not df.empty:
            for c in ["buyRatio","sellRatio"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["timestamp"] = pd.to_datetime(
                pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True
            )
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def funding_history(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        obj = self._get("/v5/market/funding/history", {
            "category": "linear", "symbol": symbol.upper(),
            "limit": min(max(int(limit),1),200)
        })
        df = pd.DataFrame(obj.get("result", {}).get("list", []))
        if not df.empty:
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df["fundingRateTimestamp"] = pd.to_datetime(
                pd.to_numeric(df["fundingRateTimestamp"], errors="coerce"),
                unit="ms", utc=True
            )
            df = df.sort_values("fundingRateTimestamp").reset_index(drop=True)
        return df
