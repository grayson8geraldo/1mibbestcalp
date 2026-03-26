"""
Volume Profile indicator.
Computes Value Area (VAH, VAL), Point of Control (POC),
and Low Volume Nodes (LVN) from OHLCV data.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import VP_LVN_THRESHOLD, VP_NUM_BINS, VP_VALUE_AREA_PCT

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """Result of volume profile calculation."""

    poc: float = 0.0  # Point of Control (price with max volume)
    vah: float = 0.0  # Value Area High
    val: float = 0.0  # Value Area Low
    lvn_levels: List[float] = field(default_factory=list)  # Low Volume Nodes
    profile: Optional[pd.Series] = None  # Full profile (price bins -> volume)
    total_volume: float = 0.0


class VolumeProfile:
    """Calculates Volume Profile from OHLCV data."""

    def __init__(
        self,
        num_bins: int = VP_NUM_BINS,
        value_area_pct: float = VP_VALUE_AREA_PCT,
        lvn_threshold: float = VP_LVN_THRESHOLD,
    ):
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.lvn_threshold = lvn_threshold

    def calculate(self, df: pd.DataFrame) -> VolumeProfileResult:
        """
        Calculate volume profile from OHLCV dataframe.
        Distributes each bar's volume across its price range.
        """
        if df.empty or len(df) < 2:
            return VolumeProfileResult()

        price_low = df["low"].min()
        price_high = df["high"].max()

        if price_high == price_low:
            return VolumeProfileResult(
                poc=price_low, vah=price_high, val=price_low
            )

        # Create price bins
        bin_edges = np.linspace(price_low, price_high, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_volumes = np.zeros(self.num_bins)

        # Distribute each bar's volume across the price bins it spans
        for _, row in df.iterrows():
            bar_low = row["low"]
            bar_high = row["high"]
            bar_volume = row["volume"]

            if bar_volume == 0 or bar_high == bar_low:
                continue

            # Find bins that overlap with this bar's range
            for i in range(self.num_bins):
                bin_lo = bin_edges[i]
                bin_hi = bin_edges[i + 1]
                overlap_lo = max(bar_low, bin_lo)
                overlap_hi = min(bar_high, bin_hi)

                if overlap_hi > overlap_lo:
                    overlap_ratio = (overlap_hi - overlap_lo) / (
                        bar_high - bar_low
                    )
                    bin_volumes[i] += bar_volume * overlap_ratio

        # Create profile series
        profile = pd.Series(bin_volumes, index=bin_centers)
        total_volume = bin_volumes.sum()

        if total_volume == 0:
            return VolumeProfileResult()

        # POC: price level with maximum volume
        poc_idx = np.argmax(bin_volumes)
        poc = bin_centers[poc_idx]

        # Value Area: expand from POC until value_area_pct of total volume
        va_volume = bin_volumes[poc_idx]
        target_volume = total_volume * self.value_area_pct

        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_volume < target_volume and (lo_idx > 0 or hi_idx < self.num_bins - 1):
            # Look at volume on each side and expand toward the larger
            lo_vol = bin_volumes[lo_idx - 1] if lo_idx > 0 else 0
            hi_vol = bin_volumes[hi_idx + 1] if hi_idx < self.num_bins - 1 else 0

            if lo_vol >= hi_vol and lo_idx > 0:
                lo_idx -= 1
                va_volume += bin_volumes[lo_idx]
            elif hi_idx < self.num_bins - 1:
                hi_idx += 1
                va_volume += bin_volumes[hi_idx]
            else:
                lo_idx -= 1
                va_volume += bin_volumes[lo_idx]

        val = bin_centers[lo_idx]
        vah = bin_centers[hi_idx]

        # LVN: bins with volume significantly below max
        max_vol = bin_volumes.max()
        lvn_levels = []
        if max_vol > 0:
            for i in range(self.num_bins):
                if (
                    bin_volumes[i] < max_vol * self.lvn_threshold
                    and bin_volumes[i] > 0
                ):
                    lvn_levels.append(bin_centers[i])

        result = VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            lvn_levels=lvn_levels,
            profile=profile,
            total_volume=total_volume,
        )

        logger.debug(
            "Volume Profile: POC=%.2f, VAH=%.2f, VAL=%.2f, LVNs=%d",
            poc,
            vah,
            val,
            len(lvn_levels),
        )
        return result

    def calculate_local(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> VolumeProfileResult:
        """Calculate volume profile for a specific range of bars (impulse move)."""
        subset = df.iloc[start_idx:end_idx]
        return self.calculate(subset)
