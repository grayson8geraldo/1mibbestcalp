"""
Order Flow / Footprint analysis module.
Estimates aggressive market orders and detects "big bubbles"
from available volume data.

Note: True order flow requires Level 2 / tick data.
This module approximates it from 1-minute OHLCV data.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import LONDON_ORDER_FILTER, NY_ORDER_FILTER

logger = logging.getLogger(__name__)


@dataclass
class AggressiveOrder:
    """Represents a detected aggressive market order."""

    timestamp: pd.Timestamp
    side: str  # "BUY" or "SELL"
    estimated_contracts: float
    price: float
    is_big_trade: bool


@dataclass
class OrderFlowSignal:
    """Aggregated order flow signal for a bar or zone."""

    net_aggression: str  # "BULLISH", "BEARISH", or "NEUTRAL"
    buy_aggression: float  # Total buy-side aggressive volume
    sell_aggression: float  # Total sell-side aggressive volume
    big_trades: List[AggressiveOrder]  # Detected big trades
    has_absorption: bool  # Large opposing orders didn't move price
    has_no_follow_through: bool  # Breakout attempt with no continuation


class OrderFlowAnalyzer:
    """Analyzes order flow from estimated volume data."""

    def __init__(self):
        self.ny_filter = NY_ORDER_FILTER
        self.london_filter = LONDON_ORDER_FILTER

    def get_filter_threshold(self, session: str) -> int:
        """Get the minimum contract filter for the current session."""
        if session == "LONDON":
            return self.london_filter
        return self.ny_filter

    def analyze_bar(
        self, bar: pd.Series, session: str = "NY"
    ) -> Optional[AggressiveOrder]:
        """
        Analyze a single bar for aggressive order detection.
        Returns an AggressiveOrder if volume exceeds filter threshold.
        """
        threshold = self.get_filter_threshold(session)
        contracts = bar.get("est_contracts", bar.get("volume", 0))

        if contracts < threshold:
            return None

        # Determine aggression side from candle characteristics
        body = bar["close"] - bar["open"]
        hl_range = bar["high"] - bar["low"]

        if hl_range == 0:
            return None

        body_ratio = abs(body) / hl_range

        # Strong directional candle = aggressive order
        if body > 0 and body_ratio > 0.4:
            side = "BUY"
        elif body < 0 and body_ratio > 0.4:
            side = "SELL"
        else:
            # Indecisive - use delta if available
            delta = bar.get("delta", 0)
            side = "BUY" if delta > 0 else "SELL"

        return AggressiveOrder(
            timestamp=bar.name if hasattr(bar, "name") else pd.Timestamp.now(),
            side=side,
            estimated_contracts=contracts,
            price=bar["close"],
            is_big_trade=contracts >= threshold * 1.5,
        )

    def scan_zone(
        self,
        df: pd.DataFrame,
        zone_low: float,
        zone_high: float,
        session: str = "NY",
        direction: str = "LONG",
    ) -> OrderFlowSignal:
        """
        Scan bars within a price zone for aggressive order flow.
        Used when price reaches an LVN or VA boundary.
        """
        threshold = self.get_filter_threshold(session)

        # Filter bars that traded within the zone
        zone_bars = df[
            (df["low"] <= zone_high) & (df["high"] >= zone_low)
        ]

        big_trades: List[AggressiveOrder] = []
        total_buy = 0.0
        total_sell = 0.0

        for idx, bar in zone_bars.iterrows():
            order = self.analyze_bar(bar, session)
            if order:
                big_trades.append(order)
                if order.side == "BUY":
                    total_buy += order.estimated_contracts
                else:
                    total_sell += order.estimated_contracts

        # Determine net aggression
        if total_buy > total_sell * 1.5:
            net_aggression = "BULLISH"
        elif total_sell > total_buy * 1.5:
            net_aggression = "BEARISH"
        else:
            net_aggression = "NEUTRAL"

        # Detect absorption: large opposing volume but price didn't move
        has_absorption = False
        if len(zone_bars) >= 2:
            price_change = abs(
                zone_bars.iloc[-1]["close"] - zone_bars.iloc[0]["open"]
            )
            total_volume = zone_bars["volume"].sum()
            avg_volume = df["volume"].mean() if len(df) > 0 else 1

            if total_volume > avg_volume * 2 and price_change < (zone_high - zone_low) * 0.3:
                has_absorption = True

        # Detect no follow-through: breakout attempt dies
        has_no_follow_through = False
        if len(zone_bars) >= 3:
            # Price went beyond zone but came back
            went_above = zone_bars["high"].max() > zone_high
            went_below = zone_bars["low"].min() < zone_low
            ended_inside = (
                zone_low <= zone_bars.iloc[-1]["close"] <= zone_high
            )
            if (went_above or went_below) and ended_inside:
                has_no_follow_through = True

        return OrderFlowSignal(
            net_aggression=net_aggression,
            buy_aggression=total_buy,
            sell_aggression=total_sell,
            big_trades=big_trades,
            has_absorption=has_absorption,
            has_no_follow_through=has_no_follow_through,
        )

    def detect_trend_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        session: str = "NY",
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Detect trend entry trigger from order flow.

        Returns: (signal_found, entry_price, stop_price)
        - signal_found: True if aggressive order in trend direction detected
        - entry_price: Close of confirming candle
        - stop_price: Low/high of the aggression cluster
        """
        if len(df) < 2:
            return False, None, None

        threshold = self.get_filter_threshold(session)
        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        contracts = last_bar.get("est_contracts", last_bar.get("volume", 0))

        if contracts < threshold:
            return False, None, None

        body = last_bar["close"] - last_bar["open"]

        if direction == "LONG":
            # Need bullish candle with significant body
            if body > 0 and abs(body) / max(last_bar["high"] - last_bar["low"], 0.01) > 0.5:
                entry_price = last_bar["close"]
                # Stop below the aggression cluster
                stop_price = min(last_bar["low"], prev_bar["low"])
                return True, entry_price, stop_price

        elif direction == "SHORT":
            if body < 0 and abs(body) / max(last_bar["high"] - last_bar["low"], 0.01) > 0.5:
                entry_price = last_bar["close"]
                stop_price = max(last_bar["high"], prev_bar["high"])
                return True, entry_price, stop_price

        return False, None, None

    def detect_reversal_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        session: str = "NY",
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Detect mean-reversion entry from order flow.
        Looks for opposing large orders pushing price back into range.

        Returns: (signal_found, entry_price, stop_price)
        """
        if len(df) < 3:
            return False, None, None

        threshold = self.get_filter_threshold(session)
        last_bar = df.iloc[-1]

        contracts = last_bar.get("est_contracts", last_bar.get("volume", 0))
        if contracts < threshold:
            return False, None, None

        body = last_bar["close"] - last_bar["open"]

        # For reversal LONG: price was below, now large buy pushing back up
        if direction == "LONG":
            if body > 0 and abs(body) / max(last_bar["high"] - last_bar["low"], 0.01) > 0.4:
                entry_price = last_bar["close"]
                # Stop below recent low (false breakout point)
                recent_low = df.tail(5)["low"].min()
                stop_price = recent_low
                return True, entry_price, stop_price

        # For reversal SHORT: price was above, now large sell pushing back down
        elif direction == "SHORT":
            if body < 0 and abs(body) / max(last_bar["high"] - last_bar["low"], 0.01) > 0.4:
                entry_price = last_bar["close"]
                recent_high = df.tail(5)["high"].max()
                stop_price = recent_high
                return True, entry_price, stop_price

        return False, None, None
