"""Backtest adapter -- routes orders through the exchange simulator."""

from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from config.settings import settings
from src.adapters.base import BaseAdapter
from src.data.replay.clock import SimulatedClock
from src.data.replay.stream import DataReplay, Candle
from src.backtest.simulator import ExchangeSimulator


class BacktestAdapter(BaseAdapter):
    """Adapter that replays historical data and simulates execution."""

    def __init__(self):
        self.clock = SimulatedClock(speed=settings.backtest_speed)
        self.replay = DataReplay(self.clock)
        self.simulator = ExchangeSimulator(initial_capital=settings.initial_capital)
        self.current_prices: dict[str, float] = {}

    async def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> AsyncIterator[Candle]:
        start_ts = int(datetime.strptime(settings.backtest_start, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(settings.backtest_end, "%Y-%m-%d").timestamp() * 1000)

        async for candle in self.replay.stream_candles(
            symbol, timeframe, start_ts, end_ts, on_progress=on_progress
        ):
            self.current_prices[symbol] = candle.close
            self.simulator.check_stops(symbol, candle.high, candle.low, candle.timestamp)
            yield candle

    async def get_balance(self) -> float:
        return self.simulator.balance

    async def get_position(self, symbol: str) -> Optional[dict]:
        pos = self.simulator.positions.get(symbol)
        if pos:
            return {
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
            }
        return None

    async def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[dict]:
        price = self.current_prices.get(symbol)
        if not price:
            return None

        direction = "long" if side in ("buy", "long") else "short"
        pos = self.simulator.open_position(
            symbol, direction, size, price, self.clock.now(), stop_loss, take_profit
        )

        if pos:
            return {
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
            }
        return None

    async def close_position(self, symbol: str) -> Optional[dict]:
        """Close the current position for a symbol."""
        price = self.current_prices.get(symbol)
        if price:
            return self.simulator.close_position(symbol, price, self.clock.now())
        return None

    def get_trade_history(self) -> list[dict]:
        return self.simulator.get_trade_history()

    def get_equity(self) -> float:
        return self.simulator.get_equity(self.current_prices)
