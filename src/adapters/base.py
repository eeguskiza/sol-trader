"""Abstract base adapter for execution backends."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Optional

from src.data.replay.stream import Candle


class BaseAdapter(ABC):
    """Interface that backtest, paper, and live adapters must implement."""

    @abstractmethod
    async def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> AsyncIterator[Candle]:
        """Yield candles for the given symbol and timeframe."""
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Return current available balance."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[dict]:
        """Return current position for a symbol, or None."""
        ...

    @abstractmethod
    async def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[dict]:
        """Submit an order and return fill details, or None on failure."""
        ...
