"""
Main Trading Bot.
Orchestrates data feed, indicators, strategies, and paper trading engine.
Runs continuously during market hours.
"""

import logging
import signal
import sys
import time
from datetime import datetime

import pandas as pd

from config.settings import (
    FETCH_INTERVAL_SECONDS,
    LOG_FILE,
    LOG_LEVEL,
    TICK_SIZE,
)
from src.data.feed import DataFeed
from src.engine.paper_trader import OrderSide, PaperTrader
from src.engine.session_manager import SessionManager
from src.indicators.volume_profile import VolumeProfile, VolumeProfileResult
from src.strategies.range_model import RangeModel
from src.strategies.trend_model import TrendModel

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrating all components."""

    def __init__(self):
        # Components
        self.data_feed = DataFeed()
        self.paper_trader = PaperTrader()
        self.session_mgr = SessionManager()
        self.volume_profile = VolumeProfile()
        self.trend_model = TrendModel()
        self.range_model = RangeModel()

        # State
        self.prev_day_vp: VolumeProfileResult = VolumeProfileResult()
        self.current_vp: VolumeProfileResult = VolumeProfileResult()
        self.running = False
        self._iteration = 0

        # Setup logging
        self._setup_logging()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _setup_logging(self):
        """Configure logging to console and file."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL))

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        console.setFormatter(console_fmt)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_fmt)

        root_logger.addHandler(console)
        root_logger.addHandler(file_handler)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received. Closing positions...")
        self.running = False
        if self.paper_trader.has_open_position:
            last_price = self._get_current_price()
            if last_price:
                self.paper_trader.close_all_positions(last_price, "SHUTDOWN")
        self._print_summary()

    def _get_current_price(self) -> float:
        """Get the latest price from data feed."""
        bar = self.data_feed.get_latest_bar_1m()
        if bar is not None:
            return bar["close"]
        return 0.0

    def initialize(self) -> bool:
        """Initialize the bot: fetch data and compute initial profiles."""
        logger.info("=" * 60)
        logger.info("  NQ Futures Paper Trading Bot")
        logger.info("  Balance: $%.2f", self.paper_trader.balance)
        logger.info("=" * 60)

        logger.info("Fetching initial market data...")
        if not self.data_feed.fetch_data():
            logger.error("Failed to fetch initial data. Check internet connection.")
            return False

        # Compute previous day's Volume Profile
        prev_day = self.data_feed.get_previous_day_data_5m()
        if not prev_day.empty:
            self.prev_day_vp = self.volume_profile.calculate(prev_day)
            logger.info(
                "Previous day VP: POC=%.2f, VAH=%.2f, VAL=%.2f",
                self.prev_day_vp.poc,
                self.prev_day_vp.vah,
                self.prev_day_vp.val,
            )
        else:
            logger.warning("No previous day data available for Volume Profile")

        # Compute current session VP
        today_data = self.data_feed.get_today_data_1m()
        if not today_data.empty:
            self.current_vp = self.volume_profile.calculate(today_data)

        logger.info("Session: %s", self.session_mgr.get_status())
        logger.info("Bot initialized successfully.")
        return True

    def _update_data(self) -> bool:
        """Fetch fresh market data."""
        return self.data_feed.fetch_data()

    def _update_profiles(self):
        """Update volume profiles with latest data."""
        # Update current session VP
        today_data = self.data_feed.get_today_data_1m()
        if not today_data.empty and len(today_data) >= 5:
            self.current_vp = self.volume_profile.calculate(today_data)

    def _run_strategies(self):
        """Run strategy evaluation based on current session."""
        if not self.paper_trader.can_trade:
            return

        session = self.session_mgr.get_session_for_model()
        if session is None:
            return

        # Get data with order flow estimates
        data_1m = self.data_feed.get_recent_1m(60)
        data_5m = self.data_feed.get_recent_5m(50)

        if data_1m.empty or data_5m.empty:
            return

        data_1m_flow = self.data_feed.estimate_order_flow(data_1m)

        # Use previous day VP as reference for breakout/range detection
        ref_vp = self.prev_day_vp if self.prev_day_vp.poc > 0 else self.current_vp
        if ref_vp.poc == 0:
            return

        # --- Model 1: Trend Following (NY session) ---
        if self.session_mgr.can_run_trend_model():
            trend_signal = self.trend_model.evaluate(
                data_5m, data_1m, data_1m_flow, ref_vp, session="NY"
            )
            if trend_signal.active:
                self._execute_signal(trend_signal, "TREND")
                return  # Only one trade at a time

        # --- Model 2: Mean Reverting (London or range in NY) ---
        if self.session_mgr.can_run_range_model():
            range_signal = self.range_model.evaluate(
                data_5m, data_1m, data_1m_flow, ref_vp, session=session
            )
            if range_signal.active:
                self._execute_signal(range_signal, "RANGE")

    def _execute_signal(self, signal_obj, model: str):
        """Execute a trading signal by opening a paper position."""
        side = OrderSide.LONG if signal_obj.direction == "LONG" else OrderSide.SHORT

        position = self.paper_trader.open_position(
            side=side,
            entry_price=signal_obj.entry_price,
            stop_loss=signal_obj.stop_loss,
            take_profit=signal_obj.take_profit,
            model=model,
        )

        if position:
            logger.info(
                ">>> TRADE EXECUTED: %s %s @ %.2f | %s",
                model,
                signal_obj.direction,
                signal_obj.entry_price,
                signal_obj.reason,
            )

    def _manage_positions(self):
        """Check open positions for SL/TP/breakeven updates."""
        current_price = self._get_current_price()
        if current_price > 0:
            self.paper_trader.update_positions(current_price)

    def _check_force_close(self):
        """Force close positions near session end."""
        if self.session_mgr.should_force_close() and self.paper_trader.has_open_position:
            current_price = self._get_current_price()
            if current_price > 0:
                logger.info("FORCE CLOSE: End of session approaching")
                self.paper_trader.close_all_positions(current_price, "SESSION_END")

    def _check_new_day(self):
        """Handle new trading day transitions."""
        if self.session_mgr.is_new_day():
            logger.info("=" * 40)
            logger.info("  NEW TRADING DAY")
            logger.info("=" * 40)
            self.paper_trader.reset_daily()
            self.range_model.reset_state()

            # Refresh previous day VP
            prev_day = self.data_feed.get_previous_day_data_5m()
            if not prev_day.empty:
                self.prev_day_vp = self.volume_profile.calculate(prev_day)

    def _print_status(self):
        """Print periodic status update."""
        if self._iteration % 5 == 0:  # Every 5 iterations
            current_price = self._get_current_price()
            vp = self.current_vp
            logger.info(
                "Status | Price: %.2f | POC: %.2f | VAH: %.2f | VAL: %.2f | "
                "Balance: $%.2f | %s",
                current_price,
                vp.poc,
                vp.vah,
                vp.val,
                self.paper_trader.balance,
                self.session_mgr.get_status(),
            )

    def _print_summary(self):
        """Print trading summary."""
        print(self.paper_trader.get_summary())

    def run(self):
        """Main bot loop."""
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return

        self.running = True
        logger.info("Bot is RUNNING. Press Ctrl+C to stop.")
        logger.info("Fetching data every %d seconds.", FETCH_INTERVAL_SECONDS)

        while self.running:
            try:
                self._iteration += 1

                # 1. Check for new day
                self._check_new_day()

                # 2. Fetch fresh data
                if not self._update_data():
                    logger.warning("Data fetch failed, retrying next cycle...")
                    time.sleep(FETCH_INTERVAL_SECONDS)
                    continue

                # 3. Update volume profiles
                self._update_profiles()

                # 4. Check for force close
                self._check_force_close()

                # 5. Manage existing positions
                self._manage_positions()

                # 6. Run strategies (only if session allows trading)
                if self.session_mgr.is_trading_allowed():
                    self._run_strategies()

                # 7. Status update
                self._print_status()

                # Wait for next cycle
                time.sleep(FETCH_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(FETCH_INTERVAL_SECONDS)

        # Cleanup
        if self.paper_trader.has_open_position:
            current_price = self._get_current_price()
            if current_price > 0:
                self.paper_trader.close_all_positions(current_price, "BOT_STOPPED")

        self._print_summary()
        logger.info("Bot stopped.")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
