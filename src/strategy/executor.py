"""Strategy executor -- the main loop that connects signals to orders."""

from typing import Callable, Optional

from loguru import logger

from config.settings import settings
from src.adapters.base import BaseAdapter
from src.data.replay.stream import Candle
from src.strategy.signals import SignalGenerator, Signal
from src.strategy.risk import RiskManager


class StrategyExecutor:
    """Consume candles, generate signals, and execute trades through an adapter."""

    def __init__(self, adapter: BaseAdapter):
        self.adapter = adapter
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.threshold = settings.signal_threshold

    async def on_candle(self, candle: Candle) -> Optional[Signal]:
        """Process a candle and return a signal if strong enough."""
        signal = self.signal_generator.generate(candle)
        if signal:
            logger.debug(
                f"Signal: {signal.direction} strength={signal.strength:.3f} "
                f"(threshold={self.threshold}) components={signal.components}"
            )
            if signal.strength >= self.threshold:
                return signal
        return None

    async def execute_signal(self, signal: Signal, candle: Candle):
        """Open or flip a position based on the signal."""
        position = await self.adapter.get_position(signal.symbol)

        if position:
            if position["side"] != signal.direction:
                await self.adapter.close_position(signal.symbol)
                logger.info(f"Closed {position['side']} position due to signal reversal")
            else:
                return

        balance = await self.adapter.get_balance()
        risk_params = self.risk_manager.calculate(balance, candle.close, signal.direction)

        result = await self.adapter.execute_order(
            symbol=signal.symbol,
            side=signal.direction,
            size=risk_params.position_size,
            stop_loss=risk_params.stop_loss,
            take_profit=risk_params.take_profit,
        )

        if result:
            logger.info(
                f"Opened {signal.direction} {signal.symbol}: "
                f"size={risk_params.position_size:.4f}, "
                f"SL={risk_params.stop_loss:.2f}, TP={risk_params.take_profit:.2f}"
            )

    async def run(self, on_progress: Optional[Callable[[int, int], None]] = None):
        """Main loop -- stream candles and act on signals."""
        async for candle in self.adapter.stream_candles(
            settings.symbol, settings.timeframes[2], on_progress=on_progress
        ):
            signal = await self.on_candle(candle)
            if signal:
                await self.execute_signal(signal, candle)
