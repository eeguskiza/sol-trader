"""Position sizing and risk management."""

from dataclasses import dataclass

from config.settings import settings


@dataclass
class RiskParams:
    """Computed risk parameters for a trade."""

    position_size: float
    stop_loss: float
    take_profit: float


class RiskManager:
    """Calculate position size and SL/TP levels based on configured risk limits."""

    def __init__(self):
        self.max_position_pct = settings.max_position_pct
        self.stop_loss_pct = settings.stop_loss_pct
        self.take_profit_ratio = settings.take_profit_ratio

    def calculate(self, balance: float, entry_price: float, direction: str) -> RiskParams:
        """Derive position size, stop-loss, and take-profit for a trade.

        Args:
            balance: Current available balance.
            entry_price: Expected entry price.
            direction: "long" or "short".

        Returns:
            RiskParams with computed values.
        """
        position_value = balance * self.max_position_pct
        position_size = position_value / entry_price

        if direction == "long":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.stop_loss_pct * self.take_profit_ratio)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.stop_loss_pct * self.take_profit_ratio)

        return RiskParams(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
