"""
Market data feed module.
Fetches real NQ futures data from Yahoo Finance.
Falls back to simulated realistic data if Yahoo Finance is unavailable.
Provides 1m and 5m candle data with volume.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from config.settings import SYMBOL, TIMEFRAME_CONTEXT, TIMEFRAME_ENTRY

logger = logging.getLogger(__name__)


class DataFeed:
    """Fetches and manages real-time market data for NQ futures."""

    def __init__(self, symbol: str = SYMBOL, use_simulation: bool = False):
        self.symbol = symbol
        self._use_simulation = use_simulation
        if HAS_YFINANCE and not use_simulation:
            self.ticker = yf.Ticker(symbol)
        else:
            self.ticker = None
        self._data_1m: Optional[pd.DataFrame] = None
        self._data_5m: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
        self._sim_base_price = 21000.0  # Base price for NQ simulation

    def fetch_data(self) -> bool:
        """Fetch latest 1m and 5m data. Returns True if successful."""
        if self._use_simulation or self.ticker is None:
            return self._generate_simulated_data()

        try:
            self._data_1m = self.ticker.history(period="5d", interval="1m")
            self._data_5m = self.ticker.history(period="5d", interval="5m")

            if self._data_1m.empty or self._data_5m.empty:
                logger.warning("Empty data from Yahoo Finance, falling back to simulation")
                return self._generate_simulated_data()

            # Clean column names
            for df in [self._data_1m, self._data_5m]:
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Ensure timezone-aware timestamps in US/Eastern
            for df in [self._data_1m, self._data_5m]:
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("US/Eastern")
                else:
                    df.index = df.index.tz_localize("US/Eastern")

            self._last_fetch = datetime.now()
            logger.info(
                "Data fetched (LIVE): %d 1m bars, %d 5m bars",
                len(self._data_1m),
                len(self._data_5m),
            )
            return True

        except Exception as e:
            logger.warning("Yahoo Finance failed (%s), falling back to simulation", e)
            return self._generate_simulated_data()

    def _generate_simulated_data(self) -> bool:
        """
        Generate realistic simulated NQ futures data.
        Includes trends, ranges, volume spikes, and session-like patterns.
        """
        logger.info("Generating simulated NQ data...")
        now = pd.Timestamp.now(tz="US/Eastern")

        # Generate 3 days of 1-minute data
        trading_minutes = []
        for day_offset in range(3, 0, -1):
            day = now - pd.Timedelta(days=day_offset)
            day_start = day.replace(hour=3, minute=0, second=0, microsecond=0)
            # Generate from 03:00 (London) to 16:00 (NY close)
            for minute in range(780):  # 13 hours * 60 minutes
                trading_minutes.append(day_start + pd.Timedelta(minutes=minute))

        # Also add today's bars up to current time
        today_start = now.replace(hour=3, minute=0, second=0, microsecond=0)
        minutes_today = int((now - today_start).total_seconds() / 60)
        for minute in range(max(0, minutes_today)):
            trading_minutes.append(today_start + pd.Timedelta(minutes=minute))

        if not trading_minutes:
            # If no trading minutes (e.g., before 3am), generate last trading day
            yesterday = now - pd.Timedelta(days=1)
            day_start = yesterday.replace(hour=3, minute=0, second=0, microsecond=0)
            for minute in range(780):
                trading_minutes.append(day_start + pd.Timedelta(minutes=minute))

        n = len(trading_minutes)
        np.random.seed(int(now.timestamp()) % 100000)

        # Generate price walk with trend/range regimes
        returns = np.random.randn(n) * 2.0  # NQ ~2 point per minute std

        # Add regime changes: trend periods and range periods
        regime = np.zeros(n)
        i = 0
        while i < n:
            regime_length = np.random.randint(60, 200)
            regime_type = np.random.choice([-1, 0, 1], p=[0.25, 0.35, 0.4])
            regime[i:i + regime_length] = regime_type
            i += regime_length

        # Trend bias
        returns += regime * 0.8

        # Add volume spike effects (large moves with high volume)
        spike_indices = np.random.choice(n, size=n // 50, replace=False)
        returns[spike_indices] *= 3.0

        prices = self._sim_base_price + np.cumsum(returns)

        # Generate OHLCV
        opens = prices
        noise = np.abs(np.random.randn(n))
        highs = prices + noise * 4.0
        lows = prices - noise * 4.0
        closes = prices + np.random.randn(n) * 1.5

        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Volume: higher during NY session (09:30-16:00), lower during London
        base_volume = np.random.randint(200, 1500, n).astype(float)
        for idx, ts in enumerate(trading_minutes):
            hour = ts.hour
            if 9 <= hour < 16:
                base_volume[idx] *= 2.5  # NY session higher volume
            if hour == 9 and ts.minute < 50:
                base_volume[idx] *= 3.0  # Open volatility
        # Volume spikes at regime changes
        base_volume[spike_indices] *= 5.0

        index = pd.DatetimeIndex(trading_minutes, tz="US/Eastern")
        self._data_1m = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": base_volume.astype(int),
        }, index=index)

        # Generate 5m data by resampling
        self._data_5m = self._data_1m.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        self._last_fetch = datetime.now()
        self._use_simulation = True

        logger.info(
            "Data generated (SIMULATED): %d 1m bars, %d 5m bars, Price ~%.0f",
            len(self._data_1m),
            len(self._data_5m),
            closes[-1],
        )
        return True

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
