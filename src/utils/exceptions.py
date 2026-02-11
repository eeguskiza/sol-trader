"""Custom exception hierarchy for the trading system."""


class TradingError(Exception):
    """Base exception for all trading-related errors."""
    pass


class DataError(TradingError):
    """Error related to data fetching or processing."""
    pass


class ExecutionError(TradingError):
    """Error related to trade execution."""
    pass


class InsufficientDataError(DataError):
    """Not enough data available for the requested operation."""
    pass


class ConnectionError(TradingError):
    """Error connecting to an external service."""
    pass
