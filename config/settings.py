"""
Trading bot configuration settings.
"""

import os

# ─── Data Provider ───────────────────────────────────────────────
# Supported: "twelvedata", "polygon", "yahoo"
# Set your API key as environment variable or paste here
DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "twelvedata")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# ─── Paper Trading ───────────────────────────────────────────────
INITIAL_BALANCE = 200.0  # USD
SYMBOL = "NQ=F"  # Default symbol (auto-mapped per provider)
CONTRACT_MULTIPLIER = 20.0  # NQ futures: $20 per point
# For paper trading with $200, we use micro contracts (MNQ) sizing
# MNQ multiplier is $2 per point; we simulate fractional exposure
MICRO_MULTIPLIER = 2.0
POSITION_SIZE = 1  # Number of micro contracts per trade

# ─── Symbol Mapping ─────────────────────────────────────────────
# Maps our internal symbol to provider-specific tickers
SYMBOL_MAP = {
    "NQ=F": {
        "twelvedata": "NQ1!",      # NASDAQ 100 E-mini futures
        "polygon": "NQ",           # Polygon futures
        "yahoo": "NQ=F",           # Yahoo Finance
        "name": "NASDAQ 100 E-mini Futures",
    },
    "ES=F": {
        "twelvedata": "ES1!",      # S&P 500 E-mini futures
        "polygon": "ES",
        "yahoo": "ES=F",
        "name": "S&P 500 E-mini Futures",
    },
    "QQQ": {
        "twelvedata": "QQQ",       # NASDAQ 100 ETF
        "polygon": "QQQ",
        "yahoo": "QQQ",
        "name": "Invesco QQQ Trust",
    },
    "SPY": {
        "twelvedata": "SPY",       # S&P 500 ETF
        "polygon": "SPY",
        "yahoo": "SPY",
        "name": "SPDR S&P 500 ETF",
    },
    "XAU/USD": {
        "twelvedata": "XAU/USD",   # Gold
        "polygon": "C:XAUUSD",
        "yahoo": "GC=F",
        "name": "Gold (XAU/USD)",
    },
}

# ─── Timeframes ──────────────────────────────────────────────────
TIMEFRAME_CONTEXT = "5m"  # For market phase detection
TIMEFRAME_ENTRY = "1m"  # For entry triggers

# ─── Session Times (EST/ET) ──────────────────────────────────────
# London session: 03:00 - 09:30 ET
LONDON_SESSION_START = "03:00"
LONDON_SESSION_END = "09:30"

# New York session: 09:30 - 16:00 ET
NY_SESSION_START = "09:30"
NY_SESSION_END = "16:00"

# Skip first 20 minutes after NY open (initial balance formation)
NY_SKIP_MINUTES = 20  # Don't trade until 09:50 ET

# Close all positions before session end
FORCE_CLOSE_TIME = "15:50"  # Close 10 min before market close

# ─── Order Flow Filters ─────────────────────────────────────────
# Minimum aggressive order size (in contracts) to be considered significant
NY_ORDER_FILTER = 30  # New York session filter
LONDON_ORDER_FILTER = 20  # London session filter

# ─── Volume Profile Settings ────────────────────────────────────
VP_LOOKBACK_BARS = 50  # Bars to compute volume profile
VP_VALUE_AREA_PCT = 0.70  # 70% of volume defines Value Area
VP_LVN_THRESHOLD = 0.3  # Bins with volume < 30% of max are LVN
VP_NUM_BINS = 50  # Number of price bins for volume profile

# ─── Strategy Parameters ────────────────────────────────────────
# Model 1: Trend Following
TREND_LVN_PROXIMITY_TICKS = 40  # How close price must be to LVN (in ticks) = 10 NQ points
TREND_BREAKEVEN_TICKS = 20  # Move to breakeven after N ticks profit = 5 NQ points
TICK_SIZE = 0.25  # NQ tick size

# Model 2: Mean Reverting
RANGE_VA_BUFFER_TICKS = 20  # Buffer around VA boundaries = 5 NQ points
RANGE_STOP_OFFSET_TICKS = 8  # Extra ticks beyond false breakout for stop = 2 NQ points
RANGE_FIRST_DRIVE_BARS = 3  # Ignore first N bars of breakout

# ─── Risk Management ────────────────────────────────────────────
MAX_DAILY_LOSS = 50.0  # Maximum daily loss in USD
MAX_POSITIONS = 1  # Only one position at a time
MAX_TRADES_PER_DAY = 6  # Maximum trades per day

# ─── CVD Settings ────────────────────────────────────────────────
CVD_LOOKBACK = 20  # Bars for CVD trend detection

# ─── Logging ─────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"
TRADE_LOG_FILE = "trades.csv"

# ─── Data Fetch Interval ────────────────────────────────────────
FETCH_INTERVAL_SECONDS = 60  # How often to fetch new data
