"""
Cumulative Volume Delta (CVD) indicator.
Tracks the running sum of (buy volume - sell volume).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import CVD_LOOKBACK

logger = logging.getLogger(__name__)


@dataclass
class CVDResult:
    """CVD analysis result."""

    current_cvd: float
    cvd_trend: str  # "RISING", "FALLING", "FLAT"
    cvd_divergence: str  # "BULLISH_DIV", "BEARISH_DIV", "NONE"
    cvd_series: pd.Series


class CVDIndicator:
    """Computes and analyzes Cumulative Volume Delta."""

    def __init__(self, lookback: int = CVD_LOOKBACK):
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> CVDResult:
        """
        Calculate CVD from dataframe with estimated buy/sell volume.
        Expects columns: buy_volume, sell_volume (or computes from OHLCV).
        """
        if df.empty:
            return CVDResult(0, "FLAT", "NONE", pd.Series(dtype=float))

        if "delta" in df.columns:
            delta = df["delta"]
        elif "buy_volume" in df.columns and "sell_volume" in df.columns:
            delta = df["buy_volume"] - df["sell_volume"]
        else:
            # Estimate from OHLCV
            hl_range = (df["high"] - df["low"]).replace(0, np.nan)
            buy_pct = (df["close"] - df["low"]) / hl_range
            buy_pct = buy_pct.fillna(0.5)
            delta = df["volume"] * (2 * buy_pct - 1)

        cvd = delta.cumsum()
        current_cvd = cvd.iloc[-1] if len(cvd) > 0 else 0

        # Determine CVD trend over lookback period
        recent_cvd = cvd.tail(self.lookback)
        if len(recent_cvd) >= 3:
            slope = np.polyfit(range(len(recent_cvd)), recent_cvd.values, 1)[0]
            if slope > recent_cvd.std() * 0.1:
                cvd_trend = "RISING"
            elif slope < -recent_cvd.std() * 0.1:
                cvd_trend = "FALLING"
            else:
                cvd_trend = "FLAT"
        else:
            cvd_trend = "FLAT"

        # Detect divergence with price
        divergence = "NONE"
        if len(df) >= self.lookback:
            recent_price = df["close"].tail(self.lookback)
            price_higher = recent_price.iloc[-1] > recent_price.iloc[0]
            cvd_higher = recent_cvd.iloc[-1] > recent_cvd.iloc[0]

            if price_higher and not cvd_higher:
                divergence = "BEARISH_DIV"
            elif not price_higher and cvd_higher:
                divergence = "BULLISH_DIV"

        return CVDResult(
            current_cvd=current_cvd,
            cvd_trend=cvd_trend,
            cvd_divergence=divergence,
            cvd_series=cvd,
        )
