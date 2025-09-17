"""Logging utilities for the ESI agents platform."""
from __future__ import annotations

import logging
from typing import Optional


_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "esi_agents") -> logging.Logger:
    """Return a shared configured logger.

    Parameters
    ----------
    name:
        Name of the logger to retrieve.
    """
    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
        _LOGGER = logging.getLogger(name)
    return logging.getLogger(name)


__all__ = ["get_logger"]
