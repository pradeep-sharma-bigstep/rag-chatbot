"""Clock tool: the current date and time, so the model never has to guess "today"."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

_LOCAL = {"", "local", "none", "null"}  # models sometimes send these instead of omitting the argument


def _now(zone: ZoneInfo | None) -> datetime:
    """The current time. A separate function so tests can freeze it."""
    return datetime.now(zone) if zone is not None else datetime.now().astimezone()


def current_time(timezone: str | None = None) -> str:
    """The current time as ISO 8601 text, or a string starting with "Error"."""
    name = (timezone or "").strip()
    zone: ZoneInfo | None = None
    if name.lower() not in _LOCAL:
        try:
            zone = ZoneInfo(name)
        except (KeyError, ValueError, OSError):  # unknown name, malformed name, no tz database
            return _UNKNOWN_ZONE.format(name=name)
    return _now(zone).isoformat(timespec="seconds")


_ARGUMENT_ERROR = "Error: `timezone` must be text such as 'UTC' or 'Asia/Kolkata', or left out"
_UNKNOWN_ZONE = "Error: unknown timezone {name!r}; use an IANA name such as 'UTC', 'Asia/Kolkata' or 'America/New_York'"


@tool
def current_datetime(timezone: str | None = None) -> str:
    """Get the current date and time in ISO 8601 format, for example 2026-11-12T10:30:00+05:30.

    Use this whenever the answer depends on today's date or the current time. Optionally pass an
    IANA timezone name such as "UTC", "Asia/Kolkata" or "America/New_York"; without it the
    server's local timezone is used.
    """
    return current_time(timezone)


current_datetime.handle_validation_error = _ARGUMENT_ERROR
