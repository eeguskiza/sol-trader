"""Backtest orchestrator that ties together replay, strategy, and simulation."""

import asyncio
from datetime import datetime

from loguru import logger

from config.settings import settings
from src.data.replay.clock import SimulatedClock
from src.data.replay.stream import DataReplay
from src.backtest.simulator import ExchangeSimulator
from src.backtest.metrics import calculate_metrics, BacktestMetrics


class BacktestEngine:
    """Run a strategy against historical data with simulated execution."""

    def __init__(self):
        self.clock = SimulatedClock(speed=settings.backtest_speed)
        self.replay = DataReplay(self.clock)
        self.simulator = ExchangeSimulator(initial_capital=settings.initial_capital)

    async def run(self, strategy) -> BacktestMetrics:
        """Execute the backtest loop.

        Args:
            strategy: An object with ``on_candle(candle, simulator)`` and
                ``execute_signal(signal, candle, simulator)`` coroutines.

        Returns:
            BacktestMetrics summarising the run.
        """
        start_ts = int(datetime.strptime(settings.backtest_start, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(settings.backtest_end, "%Y-%m-%d").timestamp() * 1000)

        logger.info(f"Starting backtest: {settings.backtest_start} to {settings.backtest_end}")
        logger.info(f"Initial capital: {settings.initial_capital}")

        candle = None
        async for candle in self.replay.stream_candles(
            settings.symbol, settings.timeframes[2], start_ts, end_ts
        ):
            self.simulator.check_stops(candle.symbol, candle.high, candle.low, candle.timestamp)

            signal = await strategy.on_candle(candle, self.simulator)
            if signal:
                await strategy.execute_signal(signal, candle, self.simulator)

        # Close any remaining positions at last known price
        if candle:
            for symbol in list(self.simulator.positions.keys()):
                self.simulator.close_position(symbol, candle.close, candle.timestamp)

        metrics = calculate_metrics(self.simulator.get_trade_history(), settings.initial_capital)

        logger.info(f"Backtest complete: {metrics.total_trades} trades")
        return metrics
