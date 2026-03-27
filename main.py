#!/usr/bin/env python3
"""
NQ Futures Paper Trading Bot
=============================
Paper trading bot for NASDAQ 100 E-mini Futures (NQ) with two models:
  - Model 1: Trend Following (breakout + LVN pullback + order flow)
  - Model 2: Mean Reverting (false breakout of Value Area + reversal flow)

Uses real market data with a virtual $200 balance.

Data Providers (in order of priority):
  - Twelve Data (set TWELVEDATA_API_KEY env var) — recommended
  - Polygon.io  (set POLYGON_API_KEY env var)
  - Yahoo Finance (no key needed, fallback)

Usage:
    python main.py                          # Run live paper trading
    python main.py --backtest               # Backtest on historical data
    python main.py --status                 # Show market status
    python main.py --symbol QQQ             # Trade QQQ instead of NQ
    python main.py --provider twelvedata    # Force specific provider

Environment variables:
    TWELVEDATA_API_KEY=your_key_here
    POLYGON_API_KEY=your_key_here
    DATA_PROVIDER=twelvedata|polygon|yahoo
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import SYMBOL, DATA_PROVIDER, SYMBOL_MAP


def run_status(symbol: str, provider: str):
    """Show current market status without trading."""
    from src.data.feed import DataFeed
    from src.engine.session_manager import SessionManager
    from src.indicators.volume_profile import VolumeProfile

    print("=" * 55)
    print("  Market Status")
    print("=" * 55)

    session_mgr = SessionManager()
    print(f"\n  {session_mgr.get_status()}")

    # Show symbol info
    sym_info = SYMBOL_MAP.get(symbol, {})
    print(f"  Symbol: {symbol} ({sym_info.get('name', 'Unknown')})")
    print(f"  Provider: {provider}")

    feed = DataFeed(symbol=symbol, provider=provider)
    if feed.fetch_data():
        print(f"  Data source: {feed.provider_name}")

        bar = feed.get_latest_bar_1m()
        if bar is not None:
            print(f"\n  Latest 1m bar:")
            print(f"    Open:   {bar['open']:.2f}")
            print(f"    High:   {bar['high']:.2f}")
            print(f"    Low:    {bar['low']:.2f}")
            print(f"    Close:  {bar['close']:.2f}")
            print(f"    Volume: {bar['volume']:.0f}")

        vp = VolumeProfile()
        prev_day = feed.get_previous_day_data_5m()
        if not prev_day.empty:
            result = vp.calculate(prev_day)
            print(f"\n  Previous Day Volume Profile:")
            print(f"    POC: {result.poc:.2f}")
            print(f"    VAH: {result.vah:.2f}")
            print(f"    VAL: {result.val:.2f}")
            print(f"    LVN levels: {len(result.lvn_levels)}")

        today = feed.get_today_data_1m()
        if not today.empty:
            result = vp.calculate(today)
            print(f"\n  Today's Volume Profile:")
            print(f"    POC: {result.poc:.2f}")
            print(f"    VAH: {result.vah:.2f}")
            print(f"    VAL: {result.val:.2f}")

        # Show gap info
        data_1m = feed.data_1m
        if "gap" in data_1m.columns:
            gaps = data_1m[data_1m["gap"] != 0]
            if not gaps.empty:
                print(f"\n  Detected gaps: {len(gaps)}")
                for idx, row in gaps.tail(3).iterrows():
                    print(f"    {idx}: {row['gap']:+.2f} points")
    else:
        print("\n  Failed to fetch data. Check API key and connection.")

    print(f"\n{'='*55}")


def run_backtest(symbol: str, provider: str):
    """Run a simple backtest on recent historical data."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data.feed import DataFeed
    from src.engine.paper_trader import OrderSide, PaperTrader
    from src.indicators.volume_profile import VolumeProfile
    from src.strategies.trend_model import TrendModel
    from src.strategies.range_model import RangeModel

    print("=" * 55)
    print("  Backtest")
    print("=" * 55)

    feed = DataFeed(symbol=symbol, provider=provider)
    if not feed.fetch_data():
        print("Failed to fetch data. Check API key and connection.")
        return

    print(f"  Data source: {feed.provider_name}")

    trader = PaperTrader()
    vp = VolumeProfile()
    trend = TrendModel()
    range_model = RangeModel()

    data_1m = feed.data_1m
    data_5m = feed.data_5m

    if data_1m.empty:
        print("No data available for backtest.")
        return

    # Compute VP from first portion of data (simulate "previous day")
    split_point = len(data_5m) // 2
    prev_data = data_5m.iloc[:split_point]
    prev_vp = vp.calculate(prev_data)

    print(f"\n  Data range: {data_1m.index[0]} to {data_1m.index[-1]}")
    print(f"  Total 1m bars: {len(data_1m)}")
    print(f"  Reference VP: POC={prev_vp.poc:.2f}, VAH={prev_vp.vah:.2f}, VAL={prev_vp.val:.2f}")
    print(f"\n  Running backtest...")

    # Walk through data bar by bar
    window = 60
    for i in range(window, len(data_1m)):
        d1m = data_1m.iloc[max(0, i - window):i + 1]

        # Skip gap bars (session opens) — don't trade on the gap itself
        if "is_session_start" in d1m.columns and d1m.iloc[-1].get("is_session_start", False):
            continue

        d5m_idx = data_5m.index.searchsorted(d1m.index[0])
        d5m_end = data_5m.index.searchsorted(d1m.index[-1])
        d5m = data_5m.iloc[max(0, d5m_idx - 20):d5m_end + 1]

        current_price = d1m.iloc[-1]["close"]

        if trader.has_open_position:
            trader.update_positions(current_price)
            continue

        if not trader.can_trade:
            continue

        d1m_flow = feed.estimate_order_flow(d1m)

        trend_signal = trend.evaluate(d5m, d1m, d1m_flow, prev_vp, session="NY")
        if trend_signal.active:
            side = OrderSide.LONG if trend_signal.direction == "LONG" else OrderSide.SHORT
            trader.open_position(
                side=side,
                entry_price=trend_signal.entry_price,
                stop_loss=trend_signal.stop_loss,
                take_profit=trend_signal.take_profit,
                model="TREND",
            )
            continue

        range_signal = range_model.evaluate(d5m, d1m, d1m_flow, prev_vp, session="LONDON")
        if range_signal.active:
            side = OrderSide.LONG if range_signal.direction == "LONG" else OrderSide.SHORT
            trader.open_position(
                side=side,
                entry_price=range_signal.entry_price,
                stop_loss=range_signal.stop_loss,
                take_profit=range_signal.take_profit,
                model="RANGE",
            )

    if trader.has_open_position:
        last_price = data_1m.iloc[-1]["close"]
        trader.close_all_positions(last_price, "BACKTEST_END")

    print(trader.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NQ Futures Paper Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported symbols:
  NQ=F      NASDAQ 100 E-mini Futures (default)
  ES=F      S&P 500 E-mini Futures
  QQQ       NASDAQ 100 ETF
  SPY       S&P 500 ETF
  XAU/USD   Gold

Examples:
  TWELVEDATA_API_KEY=xxx python main.py
  TWELVEDATA_API_KEY=xxx python main.py --status --symbol QQQ
  POLYGON_API_KEY=xxx python main.py --provider polygon --backtest
        """,
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest on historical data"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current market status"
    )
    parser.add_argument(
        "--symbol", type=str, default=SYMBOL, help=f"Trading symbol (default: {SYMBOL})"
    )
    parser.add_argument(
        "--provider", type=str, default=DATA_PROVIDER,
        choices=["twelvedata", "polygon", "yahoo"],
        help=f"Data provider (default: {DATA_PROVIDER})",
    )
    args = parser.parse_args()

    if args.status:
        run_status(args.symbol, args.provider)
    elif args.backtest:
        run_backtest(args.symbol, args.provider)
    else:
        from src.bot import TradingBot, main as bot_main
        bot_main()
