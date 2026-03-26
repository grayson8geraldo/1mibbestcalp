"""
Paper Trading Engine.
Manages virtual balance, positions, orders, and P&L tracking.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd

from config.settings import (
    INITIAL_BALANCE,
    MAX_DAILY_LOSS,
    MAX_POSITIONS,
    MAX_TRADES_PER_DAY,
    MICRO_MULTIPLIER,
    POSITION_SIZE,
    TICK_SIZE,
    TRADE_LOG_FILE,
)

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Represents an open or closed trading position."""

    id: int
    side: OrderSide
    entry_price: float
    entry_time: datetime
    size: int  # number of micro contracts
    stop_loss: float
    take_profit: float
    model: str  # "TREND" or "RANGE"
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    breakeven_moved: bool = False


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0


class PaperTrader:
    """Paper trading engine with virtual balance management."""

    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.position_counter = 0
        self.daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            peak_balance=initial_balance,
        )
        self.peak_balance = initial_balance

        logger.info("Paper Trader initialized with $%.2f balance", initial_balance)

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.status == PositionStatus.OPEN]

    @property
    def has_open_position(self) -> bool:
        return len(self.open_positions) > 0

    @property
    def daily_pnl(self) -> float:
        return self.daily_stats.total_pnl

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits."""
        if self.has_open_position:
            return False
        if self.daily_stats.trades >= MAX_TRADES_PER_DAY:
            logger.warning("Max daily trades (%d) reached", MAX_TRADES_PER_DAY)
            return False
        if self.daily_stats.total_pnl <= -MAX_DAILY_LOSS:
            logger.warning("Max daily loss ($%.2f) reached", MAX_DAILY_LOSS)
            return False
        if self.balance <= 0:
            logger.warning("Balance depleted")
            return False
        return True

    def open_position(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        model: str,
        size: int = POSITION_SIZE,
    ) -> Optional[Position]:
        """Open a new paper trading position."""
        if not self.can_trade:
            return None

        self.position_counter += 1
        position = Position(
            id=self.position_counter,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model=model,
        )

        self.positions.append(position)
        self.daily_stats.trades += 1

        # Calculate risk in dollars
        if side == OrderSide.LONG:
            risk_ticks = (entry_price - stop_loss) / TICK_SIZE
        else:
            risk_ticks = (stop_loss - entry_price) / TICK_SIZE

        risk_usd = risk_ticks * TICK_SIZE * MICRO_MULTIPLIER * size

        logger.info(
            "OPENED %s #%d: %s @ %.2f | SL: %.2f | TP: %.2f | Risk: $%.2f | Model: %s",
            side.value,
            position.id,
            "NQ",
            entry_price,
            stop_loss,
            take_profit,
            risk_usd,
            model,
        )
        return position

    def close_position(
        self, position: Position, exit_price: float, reason: str = ""
    ) -> float:
        """Close an open position and calculate P&L."""
        if position.status != PositionStatus.OPEN:
            return 0.0

        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.status = PositionStatus.CLOSED

        # Calculate P&L
        if position.side == OrderSide.LONG:
            price_diff = exit_price - position.entry_price
        else:
            price_diff = position.entry_price - exit_price

        pnl = price_diff * MICRO_MULTIPLIER * position.size
        position.pnl = pnl

        # Update balance
        self.balance += pnl
        self.daily_stats.total_pnl += pnl

        if pnl > 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

        # Track peak balance and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = self.peak_balance - self.balance
        if drawdown > self.daily_stats.max_drawdown:
            self.daily_stats.max_drawdown = drawdown

        self.closed_positions.append(position)

        logger.info(
            "CLOSED %s #%d @ %.2f | P&L: $%.2f | Reason: %s | Balance: $%.2f",
            position.side.value,
            position.id,
            exit_price,
            pnl,
            reason,
            self.balance,
        )

        self._log_trade(position)
        return pnl

    def update_positions(self, current_price: float) -> None:
        """
        Check all open positions against current price for:
        - Stop loss hit
        - Take profit hit
        - Breakeven adjustment
        """
        for position in self.open_positions:
            if position.side == OrderSide.LONG:
                # Check stop loss
                if current_price <= position.stop_loss:
                    self.close_position(position, position.stop_loss, "STOP_LOSS")
                    continue

                # Check take profit
                if current_price >= position.take_profit:
                    self.close_position(position, position.take_profit, "TAKE_PROFIT")
                    continue

                # Move to breakeven
                if not position.breakeven_moved:
                    profit_ticks = (current_price - position.entry_price) / TICK_SIZE
                    from config.settings import TREND_BREAKEVEN_TICKS

                    if profit_ticks >= TREND_BREAKEVEN_TICKS:
                        position.stop_loss = position.entry_price
                        position.breakeven_moved = True
                        logger.info(
                            "Position #%d: Stop moved to BREAKEVEN @ %.2f",
                            position.id,
                            position.entry_price,
                        )

            elif position.side == OrderSide.SHORT:
                # Check stop loss
                if current_price >= position.stop_loss:
                    self.close_position(position, position.stop_loss, "STOP_LOSS")
                    continue

                # Check take profit
                if current_price <= position.take_profit:
                    self.close_position(position, position.take_profit, "TAKE_PROFIT")
                    continue

                # Move to breakeven
                if not position.breakeven_moved:
                    profit_ticks = (position.entry_price - current_price) / TICK_SIZE
                    from config.settings import TREND_BREAKEVEN_TICKS

                    if profit_ticks >= TREND_BREAKEVEN_TICKS:
                        position.stop_loss = position.entry_price
                        position.breakeven_moved = True
                        logger.info(
                            "Position #%d: Stop moved to BREAKEVEN @ %.2f",
                            position.id,
                            position.entry_price,
                        )

    def close_all_positions(self, current_price: float, reason: str = "END_OF_DAY"):
        """Close all open positions (e.g., end of day)."""
        for position in self.open_positions:
            self.close_position(position, current_price, reason)

    def reset_daily(self):
        """Reset daily statistics for a new trading day."""
        self.daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            peak_balance=self.balance,
        )
        logger.info("Daily stats reset. Balance: $%.2f", self.balance)

    def get_summary(self) -> str:
        """Get a summary of current trading state."""
        total_trades = len(self.closed_positions)
        wins = sum(1 for p in self.closed_positions if p.pnl > 0)
        losses = sum(1 for p in self.closed_positions if p.pnl < 0)
        total_pnl = sum(p.pnl for p in self.closed_positions)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        return (
            f"\n{'='*50}\n"
            f"  PAPER TRADING SUMMARY\n"
            f"{'='*50}\n"
            f"  Initial Balance: ${self.initial_balance:.2f}\n"
            f"  Current Balance: ${self.balance:.2f}\n"
            f"  Total P&L:       ${total_pnl:.2f}\n"
            f"  Return:          {(total_pnl/self.initial_balance)*100:.1f}%\n"
            f"{'─'*50}\n"
            f"  Total Trades:    {total_trades}\n"
            f"  Wins:            {wins}\n"
            f"  Losses:          {losses}\n"
            f"  Win Rate:        {win_rate:.1f}%\n"
            f"  Max Drawdown:    ${self.daily_stats.max_drawdown:.2f}\n"
            f"{'─'*50}\n"
            f"  Open Positions:  {len(self.open_positions)}\n"
            f"  Today's P&L:     ${self.daily_pnl:.2f}\n"
            f"  Today's Trades:  {self.daily_stats.trades}\n"
            f"{'='*50}\n"
        )

    def _log_trade(self, position: Position):
        """Log trade to CSV file."""
        file_exists = os.path.exists(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "id", "side", "model", "entry_time", "entry_price",
                    "exit_time", "exit_price", "stop_loss", "take_profit",
                    "pnl", "exit_reason", "balance_after",
                ])
            writer.writerow([
                position.id,
                position.side.value,
                position.model,
                position.entry_time.isoformat(),
                f"{position.entry_price:.2f}",
                position.exit_time.isoformat() if position.exit_time else "",
                f"{position.exit_price:.2f}" if position.exit_price else "",
                f"{position.stop_loss:.2f}",
                f"{position.take_profit:.2f}",
                f"{position.pnl:.2f}",
                position.exit_reason,
                f"{self.balance:.2f}",
            ])
