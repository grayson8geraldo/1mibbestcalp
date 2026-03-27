"""
Market data feed module.
Supports multiple data providers: Twelve Data (primary), Polygon.io, Yahoo Finance.
Handles gaps, weekend breaks, and provider-specific ticker mapping.
"""

import logging
import time as _time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config.settings import (
    DATA_PROVIDER,
    POLYGON_API_KEY,
    SYMBOL,
    SYMBOL_MAP,
    TWELVEDATA_API_KEY,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]


class DataFetchError(Exception):
    """Raised when data cannot be fetched."""
    pass


def resolve_symbol(internal_symbol: str, provider: str) -> str:
    """Map internal symbol to provider-specific ticker."""
    if internal_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[internal_symbol].get(provider, internal_symbol)
    return internal_symbol


# ─── Provider Implementations ──────────────────────────────────────


class BaseProvider(ABC):
    """Abstract base for data providers."""

    @abstractmethod
    def fetch_1m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_5m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class TwelveDataProvider(BaseProvider):
    """
    Twelve Data API provider.
    Free tier: 800 requests/day, 8 requests/minute.
    Provides real-time data with minimal delay.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "TwelveData"

    def _fetch(self, symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
        """Fetch time series from Twelve Data API."""
        url = f"{self.BASE_URL}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "timezone": "America/New_York",
            "order": "ASC",
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "code" in data and data["code"] != 200:
            raise DataFetchError(f"TwelveData error: {data.get('message', data.get('code'))}")

        if "values" not in data or not data["values"]:
            raise DataFetchError("TwelveData returned no values")

        rows = data["values"]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.sort_index()

        # Convert columns to float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        else:
            df["volume"] = 0

        # Localize to US/Eastern
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        return df

    def fetch_1m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        # 1m data: ~390 bars per day, 5 days = ~1950
        return self._fetch(symbol, "1min", outputsize=min(days * 400, 5000))

    def fetch_5m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        return self._fetch(symbol, "5min", outputsize=min(days * 80, 5000))


class PolygonProvider(BaseProvider):
    """
    Polygon.io API provider.
    Free tier: 5 requests/minute. Good for end-of-day analysis.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "Polygon.io"

    def _fetch(self, symbol: str, multiplier: int, timespan: str, days: int) -> pd.DataFrame:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}"
            f"/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            raise DataFetchError("Polygon returned no results")

        results = data["results"]
        df = pd.DataFrame(results)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.sort_index()

        # Convert to US/Eastern
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")

        return df

    def fetch_1m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        return self._fetch(symbol, 1, "minute", days)

    def fetch_5m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        return self._fetch(symbol, 5, "minute", days)


class YahooProvider(BaseProvider):
    """Yahoo Finance provider (fallback, no API key needed)."""

    def __init__(self):
        import yfinance as yf
        self._yf = yf

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def fetch_1m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        ticker = self._yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1m")
        return self._clean(df)

    def fetch_5m(self, symbol: str, days: int = 5) -> pd.DataFrame:
        ticker = self._yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="5m")
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataFetchError("Yahoo Finance returned empty data")
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_localize("US/Eastern")
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols]


# ─── Gap Handler ───────────────────────────────────────────────────


def handle_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and mark price gaps (overnight/weekend breaks).
    Adds 'gap' column: positive = gap up, negative = gap down, 0 = no gap.
    Adds 'is_session_start' boolean column.
    """
    if df.empty or len(df) < 2:
        return df

    result = df.copy()
    result["gap"] = 0.0
    result["is_session_start"] = False

    for i in range(1, len(result)):
        time_diff = (result.index[i] - result.index[i - 1]).total_seconds()

        # Gap if more than 10 minutes between bars (session break)
        if time_diff > 600:
            gap = result.iloc[i]["open"] - result.iloc[i - 1]["close"]
            result.iloc[i, result.columns.get_loc("gap")] = gap
            result.iloc[i, result.columns.get_loc("is_session_start")] = True

    gap_count = (result["gap"] != 0).sum()
    if gap_count > 0:
        logger.info("Detected %d gaps in data", gap_count)

    return result


# ─── Main DataFeed ─────────────────────────────────────────────────


class DataFeed:
    """
    Multi-provider market data feed.
    Tries providers in order: configured primary -> fallbacks.
    Handles gaps and provider-specific quirks.
    """

    def __init__(self, symbol: str = SYMBOL, provider: str = DATA_PROVIDER):
        self.internal_symbol = symbol
        self._provider_name = provider
        self._providers = self._build_provider_chain(provider)
        self._data_1m: Optional[pd.DataFrame] = None
        self._data_5m: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
        self._consecutive_failures = 0
        self._active_provider: Optional[str] = None

    def _build_provider_chain(self, primary: str) -> list:
        """Build ordered list of providers to try."""
        providers = []

        if primary == "twelvedata" and TWELVEDATA_API_KEY:
            providers.append(("twelvedata", TwelveDataProvider(TWELVEDATA_API_KEY)))
        if primary == "polygon" and POLYGON_API_KEY:
            providers.append(("polygon", PolygonProvider(POLYGON_API_KEY)))

        # Always add Yahoo Finance as last fallback
        try:
            providers.append(("yahoo", YahooProvider()))
        except ImportError:
            pass

        # If primary wasn't added (no API key), add it with warning
        if not providers:
            logger.warning(
                "No API key configured for %s. Set %s_API_KEY environment variable.",
                primary,
                primary.upper(),
            )
            try:
                providers.append(("yahoo", YahooProvider()))
            except ImportError:
                pass

        if providers:
            logger.info(
                "Data providers: %s",
                " -> ".join(name for name, _ in providers),
            )

        return providers

    def fetch_data(self) -> bool:
        """
        Fetch latest 1m and 5m data.
        Tries each provider in the chain with retries.
        Returns True if successful.
        """
        for provider_name, provider in self._providers:
            symbol = resolve_symbol(self.internal_symbol, provider_name)
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    data_1m = provider.fetch_1m(symbol)
                    data_5m = provider.fetch_5m(symbol)

                    if data_1m.empty or data_5m.empty:
                        raise DataFetchError(f"Empty data from {provider.name}")

                    # Handle gaps
                    data_1m = handle_gaps(data_1m)
                    data_5m = handle_gaps(data_5m)

                    self._data_1m = data_1m
                    self._data_5m = data_5m
                    self._last_fetch = datetime.now()
                    self._consecutive_failures = 0
                    self._active_provider = provider.name

                    logger.info(
                        "Data fetched (%s): %d 1m bars, %d 5m bars | %s",
                        provider.name,
                        len(self._data_1m),
                        len(self._data_5m),
                        symbol,
                    )
                    return True

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF[attempt]
                        logger.warning(
                            "%s attempt %d/%d failed (%s), retrying in %ds...",
                            provider.name, attempt + 1, MAX_RETRIES, e, wait,
                        )
                        _time.sleep(wait)

            logger.warning(
                "%s failed after %d retries: %s. Trying next provider...",
                provider.name, MAX_RETRIES, last_error,
            )

        # All providers failed
        self._consecutive_failures += 1
        logger.error(
            "All providers failed (consecutive failures: %d)",
            self._consecutive_failures,
        )

        # Use stale data if available
        if self._data_1m is not None and not self._data_1m.empty:
            logger.warning(
                "Using previously fetched data (age: %s)",
                datetime.now() - self._last_fetch if self._last_fetch else "unknown",
            )
            return True

        return False

    @property
    def is_data_stale(self) -> bool:
        """Check if data hasn't been refreshed in over 5 minutes."""
        if self._last_fetch is None:
            return True
        return (datetime.now() - self._last_fetch).total_seconds() > 300

    @property
    def provider_name(self) -> str:
        return self._active_provider or "none"

    @property
    def data_1m(self) -> pd.DataFrame:
        if self._data_1m is None:
            self.fetch_data()
        return self._data_1m

    @property
    def data_5m(self) -> pd.DataFrame:
        if self._data_5m is None:
            self.fetch_data()
        return self._data_5m

    def get_latest_bar_1m(self) -> Optional[pd.Series]:
        """Get the most recent completed 1-minute bar."""
        if self._data_1m is None or self._data_1m.empty:
            return None
        return self._data_1m.iloc[-1]

    def get_latest_bar_5m(self) -> Optional[pd.Series]:
        """Get the most recent completed 5-minute bar."""
        if self._data_5m is None or self._data_5m.empty:
            return None
        return self._data_5m.iloc[-1]

    def get_recent_1m(self, n_bars: int = 60) -> pd.DataFrame:
        """Get the last N 1-minute bars."""
        if self._data_1m is None or self._data_1m.empty:
            return pd.DataFrame()
        return self._data_1m.tail(n_bars)

    def get_recent_5m(self, n_bars: int = 50) -> pd.DataFrame:
        """Get the last N 5-minute bars."""
        if self._data_5m is None or self._data_5m.empty:
            return pd.DataFrame()
        return self._data_5m.tail(n_bars)

    def get_today_data_1m(self) -> pd.DataFrame:
        """Get today's 1-minute bars only."""
        if self._data_1m is None or self._data_1m.empty:
            return pd.DataFrame()
        today = pd.Timestamp.now(tz="US/Eastern").normalize()
        return self._data_1m[self._data_1m.index >= today]

    def get_previous_day_data_5m(self) -> pd.DataFrame:
        """
        Get previous trading day's 5-minute bars for Value Area calculation.
        Uses index-based navigation (not timestamps) to handle weekend gaps.
        """
        if self._data_5m is None or self._data_5m.empty:
            return pd.DataFrame()
        today = pd.Timestamp.now(tz="US/Eastern").normalize()
        prev_data = self._data_5m[self._data_5m.index < today]
        if prev_data.empty:
            return pd.DataFrame()
        # Find the last trading day by looking at the date of the last bar
        last_day = prev_data.index[-1].normalize()
        return prev_data[prev_data.index >= last_day]

    def estimate_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate order flow from candle data.
        Uses close position within bar range to approximate buy/sell aggression.
        """
        result = df.copy()
        hl_range = result["high"] - result["low"]
        hl_range = hl_range.replace(0, np.nan)

        # Estimate buy/sell volume split
        result["buy_volume"] = result["volume"] * (
            (result["close"] - result["low"]) / hl_range
        )
        result["sell_volume"] = result["volume"] * (
            (result["high"] - result["close"]) / hl_range
        )
        result["buy_volume"] = result["buy_volume"].fillna(result["volume"] / 2)
        result["sell_volume"] = result["sell_volume"].fillna(result["volume"] / 2)

        # Volume delta per bar
        result["delta"] = result["buy_volume"] - result["sell_volume"]

        # Estimate aggressive order size
        vol_mean = result["volume"].rolling(20, min_periods=1).mean()
        vol_std = result["volume"].rolling(20, min_periods=1).std().fillna(1)
        result["volume_zscore"] = (result["volume"] - vol_mean) / vol_std

        # Contract estimate (direct volume for futures)
        result["est_contracts"] = result["volume"]

        return result
