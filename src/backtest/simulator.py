"""Exchange simulator with realistic fills, slippage, and position management."""

import random
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class Position:
    """An open trading position."""

    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_time: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Fill:
    """A single order fill."""

    symbol: str
    side: str
    size: float
    price: float
    timestamp: int
    slippage_pct: float
    fee: float


class ExchangeSimulator:
    """Simulates exchange order matching with fees and slippage."""

    def __init__(
        self,
        initial_capital: float,
        fee_pct: float = 0.001,
        slippage_bps: float = 5,
    ):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.fee_pct = fee_pct
        self.slippage_bps = slippage_bps
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict] = []

    def execute_market_order(
        self, symbol: str, side: str, size: float, price: float, timestamp: int
    ) -> Optional[Fill]:
        """Execute a market order with random slippage and fees."""
        slippage = random.uniform(0, self.slippage_bps) / 10000
        if side == "buy":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        cost = size * fill_price
        fee = cost * self.fee_pct

        if side == "buy":
            if self.balance < cost + fee:
                logger.warning(f"Insufficient balance for {side} {size} {symbol}")
                return None
            self.balance -= cost + fee
        else:
            self.balance += cost - fee

        fill = Fill(
            symbol=symbol,
            side=side,
            size=size,
            price=fill_price,
            timestamp=timestamp,
            slippage_pct=slippage * 100,
            fee=fee,
        )

        logger.debug(
            f"Fill: {side} {size} {symbol} @ {fill_price:.4f} (slippage: {slippage * 100:.3f}%)"
        )
        return fill

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        timestamp: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """Open a new position via a market order."""
        order_side = "buy" if side == "long" else "sell"
        fill = self.execute_market_order(symbol, order_side, size, price, timestamp)
        if not fill:
            return None

        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=fill.price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions[symbol] = position
        return position

    def close_position(self, symbol: str, price: float, timestamp: int) -> Optional[dict]:
        """Close an existing position and record the trade."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        exit_side = "sell" if position.side == "long" else "buy"

        fill = self.execute_market_order(symbol, exit_side, position.size, price, timestamp)
        if not fill:
            return None

        if position.side == "long":
            pnl = (fill.price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - fill.price) * position.size

        pnl_pct = pnl / (position.entry_price * position.size) * 100

        trade = {
            "symbol": symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "exit_price": fill.price,
            "entry_time": position.entry_time,
            "exit_time": timestamp,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
        self.trade_history.append(trade)
        del self.positions[symbol]

        logger.info(f"Closed {position.side} {symbol}: PnL {pnl:.2f} ({pnl_pct:.2f}%)")
        return trade

    def check_stops(
        self, symbol: str, high: float, low: float, timestamp: int
    ) -> Optional[dict]:
        """Check if any stop-loss or take-profit levels have been hit."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        if position.side == "long":
            if position.stop_loss and low <= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, timestamp)
            if position.take_profit and high >= position.take_profit:
                return self.close_position(symbol, position.take_profit, timestamp)
        else:
            if position.stop_loss and high >= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, timestamp)
            if position.take_profit and low <= position.take_profit:
                return self.close_position(symbol, position.take_profit, timestamp)

        return None

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """Calculate total equity including unrealized PnL."""
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if position.side == "long":
                    equity += position.size * price
                else:
                    equity += position.size * (2 * position.entry_price - price)
        return equity

    def get_trade_history(self) -> list[dict]:
        """Return list of all closed trades."""
        return self.trade_history
