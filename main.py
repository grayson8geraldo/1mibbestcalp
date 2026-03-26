#!/usr/bin/env python3
"""
NQ Futures Paper Trading Bot
=============================
Paper trading bot for NASDAQ 100 E-mini Futures (NQ) with two models:
  - Model 1: Trend Following (breakout + LVN pullback + order flow)
  - Model 2: Mean Reverting (false breakout of Value Area + reversal flow)

Uses real market data from Yahoo Finance with a virtual $200 balance.

Usage:
    python main.py              # Run live paper trading
    python main.py --backtest   # Run backtest on historical data
    python main.py --status     # Show current market status
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bot import TradingBot, main as bot_main


def run_status():
    """Show current market status without trading."""
    from src.data.feed import DataFeed
    from src.engine.session_manager import SessionManager
    from src.indicators.volume_profile import VolumeProfile

    print("=" * 50)
    print("  NQ Futures Market Status")
    print("=" * 50)

    session_mgr = SessionManager()
    print(f"\n  {session_mgr.get_status()}")

    feed = DataFeed()
    if feed.fetch_data():
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
    else:
        print("\n  Failed to fetch data. Market may be closed.")

    print(f"\n{'='*50}")


def run_backtest():
    """Run a simple backtest on recent historical data."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data.feed import DataFeed
    from src.engine.paper_trader import OrderSide, PaperTrader
    from src.indicators.volume_profile import VolumeProfile
    from src.strategies.trend_model import TrendModel
    from src.strategies.range_model import RangeModel

    print("=" * 50)
    print("  NQ Futures Backtest")
    print("=" * 50)

    feed = DataFeed()
    if not feed.fetch_data():
        print("Failed to fetch data.")
        return

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
    window = 60  # Look-back window
    for i in range(window, len(data_1m)):
        # Get windowed data
        d1m = data_1m.iloc[max(0, i - window):i + 1]
        d5m_idx = data_5m.index.searchsorted(d1m.index[0])
        d5m_end = data_5m.index.searchsorted(d1m.index[-1])
        d5m = data_5m.iloc[max(0, d5m_idx - 20):d5m_end + 1]

        current_price = d1m.iloc[-1]["close"]

        # Update positions
        if trader.has_open_position:
            trader.update_positions(current_price)
            continue

        if not trader.can_trade:
            continue

        # Estimate order flow
        d1m_flow = feed.estimate_order_flow(d1m)

        # Try trend model
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

        # Try range model
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

    # Close any remaining positions
    if trader.has_open_position:
        last_price = data_1m.iloc[-1]["close"]
        trader.close_all_positions(last_price, "BACKTEST_END")

    print(trader.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NQ Futures Paper Trading Bot")
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest on historical data"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current market status"
    )
    args = parser.parse_args()

    if args.status:
        run_status()
    elif args.backtest:
        run_backtest()
    else:
        bot_main()
