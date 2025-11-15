"""Logging utilities for route planning module."""
from __future__ import annotations

from datetime import datetime
import os

from loguru import logger as _logger


def _default_log_path() -> str:
    """Return the platform specific log file path."""
    if os.name == "nt":
        return f"Log/log_{datetime.now().strftime('%Y%m%d')}.log"
    return "/app/logs/log_{time:YYYY-MM-DD}.log"


def configure_logger() -> "loguru.Logger":
    """Configure and return a shared loguru logger instance."""
    log_path = _default_log_path()
    try:
        _logger.add(
            log_path,
            rotation="00:00",
            encoding="utf-8",
            retention="7 days",
            enqueue=True,
            mode="w",
            backtrace=True,
            diagnose=True,
        )
    except OSError:
        fallback_path = "app/logs/log_{time:YYYY-MM-DD}.log"
        _logger.add(
            fallback_path,
            rotation="00:00",
            encoding="utf-8",
            retention="7 days",
            enqueue=True,
            mode="w",
            backtrace=True,
            diagnose=True,
        )
    return _logger


LOGGER = configure_logger()

__all__ = ["LOGGER", "configure_logger"]
