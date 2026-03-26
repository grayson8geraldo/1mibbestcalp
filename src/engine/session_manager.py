"""
Session Manager.
Handles trading session timing, determines active session,
and enforces time-based trading rules.
"""

import logging
from datetime import datetime, time
from enum import Enum
from typing import Optional

import pandas as pd

from config.settings import (
    FORCE_CLOSE_TIME,
    LONDON_SESSION_END,
    LONDON_SESSION_START,
    NY_SESSION_END,
    NY_SESSION_START,
    NY_SKIP_MINUTES,
)

logger = logging.getLogger(__name__)


class Session(Enum):
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    CLOSED = "CLOSED"
    NY_SKIP = "NY_SKIP"  # First 20 min after NY open


def _parse_time(t: str) -> time:
    parts = t.split(":")
    return time(int(parts[0]), int(parts[1]))


LONDON_START = _parse_time(LONDON_SESSION_START)
LONDON_END = _parse_time(LONDON_SESSION_END)
NY_START = _parse_time(NY_SESSION_START)
NY_END = _parse_time(NY_SESSION_END)
FORCE_CLOSE = _parse_time(FORCE_CLOSE_TIME)

# NY skip period: 09:30 to 09:50
NY_SKIP_END = time(
    NY_START.hour,
    NY_START.minute + NY_SKIP_MINUTES,
)


class SessionManager:
    """Manages trading session timing and rules."""

    def __init__(self):
        self._last_session: Optional[Session] = None

    def get_current_time_et(self) -> datetime:
        """Get current time in US/Eastern."""
        return pd.Timestamp.now(tz="US/Eastern").to_pydatetime()

    def get_current_session(self) -> Session:
        """Determine the current active trading session."""
        now = self.get_current_time_et()
        current = now.time()

        # Check if it's within NY skip period (first 20 min)
        if NY_START <= current < NY_SKIP_END:
            return Session.NY_SKIP

        # Check NY session (after skip)
        if NY_SKIP_END <= current <= NY_END:
            return Session.NEW_YORK

        # Check London session
        if LONDON_START <= current < LONDON_END:
            return Session.LONDON

        return Session.CLOSED

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed based on session time."""
        session = self.get_current_session()
        return session in (Session.NEW_YORK, Session.LONDON)

    def should_force_close(self) -> bool:
        """Check if we should force-close all positions (near session end)."""
        now = self.get_current_time_et().time()
        return now >= FORCE_CLOSE

    def get_session_for_model(self) -> Optional[str]:
        """
        Determine which model should be active.
        - Model 1 (Trend): Only during NY session
        - Model 2 (Range): During London or summer consolidation
        Returns session name string or None if closed.
        """
        session = self.get_current_session()
        if session == Session.NEW_YORK:
            return "NY"
        elif session == Session.LONDON:
            return "LONDON"
        return None

    def can_run_trend_model(self) -> bool:
        """Model 1 only runs during NY session."""
        return self.get_current_session() == Session.NEW_YORK

    def can_run_range_model(self) -> bool:
        """Model 2 runs during London session (or low-vol NY periods)."""
        session = self.get_current_session()
        return session in (Session.LONDON, Session.NEW_YORK)

    def is_new_day(self) -> bool:
        """Check if it's a new trading day (session changed from CLOSED to active)."""
        current = self.get_current_session()
        was_closed = self._last_session == Session.CLOSED
        self._last_session = current
        return was_closed and current != Session.CLOSED

    def get_status(self) -> str:
        """Get human-readable session status."""
        session = self.get_current_session()
        now_et = self.get_current_time_et()
        return (
            f"Session: {session.value} | "
            f"Time (ET): {now_et.strftime('%H:%M:%S')} | "
            f"Trading: {'YES' if self.is_trading_allowed() else 'NO'}"
        )
