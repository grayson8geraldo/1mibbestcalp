"""
Market data feed module.
Fetches real NQ futures data from Yahoo Finance.
Provides 1m and 5m candle data with volume.
"""

import logging
import time as _time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import SYMBOL, TIMEFRAME_CONTEXT, TIMEFRAME_ENTRY

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]


class DataFetchError(Exception):
    """Raised when data cannot be fetched from Yahoo Finance."""
    pass


class DataFeed:
    """Fetches and manages real-time market data for NQ futures."""

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self._data_1m: Optional[pd.DataFrame] = None
        self._data_5m: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
        self._consecutive_failures = 0

    def _create_ticker(self) -> yf.Ticker:
        """Create a fresh Ticker instance (avoids stale cached state)."""
        return yf.Ticker(self.symbol)

    def fetch_data(self) -> bool:
        """
        Fetch latest 1m and 5m data from Yahoo Finance.
        Retries up to 4 times with exponential backoff on failure.
        Returns True if successful, False otherwise.
        Never falls back to simulated data.
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                ticker = self._create_ticker()

                data_1m = ticker.history(period="5d", interval="1m")
                data_5m = ticker.history(period="5d", interval="5m")

                if data_1m is None or data_5m is None:
                    raise DataFetchError("Received None from Yahoo Finance")

                if data_1m.empty or data_5m.empty:
                    raise DataFetchError("Received empty data from Yahoo Finance")

                # Clean column names
                for df in [data_1m, data_5m]:
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

                # Ensure timezone-aware timestamps in US/Eastern
                for df in [data_1m, data_5m]:
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert("US/Eastern")
                    else:
                        df.index = df.index.tz_localize("US/Eastern")

                # Success — update stored data
                self._data_1m = data_1m
                self._data_5m = data_5m
                self._last_fetch = datetime.now()
                self._consecutive_failures = 0

                logger.info(
                    "Data fetched (LIVE): %d 1m bars, %d 5m bars",
                    len(self._data_1m),
                    len(self._data_5m),
                )
                return True

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Yahoo Finance attempt %d/%d failed (%s), retrying in %ds...",
                        attempt + 1,
                        MAX_RETRIES,
                        e,
                        wait,
                    )
                    _time.sleep(wait)

        # All retries exhausted
        self._consecutive_failures += 1
        logger.error(
            "Yahoo Finance failed after %d retries: %s (consecutive failures: %d)",
            MAX_RETRIES,
            last_error,
            self._consecutive_failures,
        )

        # If we have previous data, keep using it (stale but real)
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
        """Get previous trading day's 5-minute bars for Value Area calculation."""
        if self._data_5m is None or self._data_5m.empty:
            return pd.DataFrame()
        today = pd.Timestamp.now(tz="US/Eastern").normalize()
        prev_data = self._data_5m[self._data_5m.index < today]
        if prev_data.empty:
            return pd.DataFrame()
        last_day = prev_data.index[-1].normalize()
        return prev_data[prev_data.index >= last_day]

    def estimate_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate order flow from candle data.
        Since we don't have Level 2 data, we approximate:
        - Buy aggression: volume * (close - low) / (high - low)
        - Sell aggression: volume * (high - close) / (high - low)
        - Big trades estimated from volume spikes
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

        # Simulated "big trade" contracts: scale volume to contract-like units
        result["est_contracts"] = result["volume"]

        return result
