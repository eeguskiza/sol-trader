"""Logging configuration using Loguru."""

import sys

from loguru import logger

from config.settings import settings


def setup_logging(console_output: bool = True):
    """Configure application logging with console and file output.

    Args:
        console_output: If False, suppress stderr output (useful during Rich progress bars).
    """
    logger.remove()
    if console_output:
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="<dim>{time:HH:mm:ss}</dim> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )
    logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    return logger
