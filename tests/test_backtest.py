"""Tests for the backtest engine components."""

import numpy as np
import pytest

from src.backtest.simulator import ExchangeSimulator
from src.backtest.metrics import calculate_metrics


class TestExchangeSimulator:
    def test_initial_state(self):
        sim = ExchangeSimulator(initial_capital=10000.0)
        assert sim.balance == 10000.0
        assert sim.positions == {}
        assert sim.trade_history == []

    def test_open_and_close_long(self):
        sim = ExchangeSimulator(initial_capital=10000.0, fee_pct=0, slippage_bps=0)

        pos = sim.open_position("SOL/USDT", "long", 10.0, 100.0, timestamp=1000)
        assert pos is not None
        assert pos.side == "long"
        assert pos.size == 10.0
        assert sim.balance == 9000.0  # 10000 - 10*100

        trade = sim.close_position("SOL/USDT", 110.0, timestamp=2000)
        assert trade is not None
        assert trade["pnl"] == 100.0  # (110-100)*10
        assert len(sim.trade_history) == 1

    def test_open_and_close_short(self):
        sim = ExchangeSimulator(initial_capital=10000.0, fee_pct=0, slippage_bps=0)

        pos = sim.open_position("SOL/USDT", "short", 10.0, 100.0, timestamp=1000)
        assert pos is not None

        trade = sim.close_position("SOL/USDT", 90.0, timestamp=2000)
        assert trade is not None
        assert trade["pnl"] == 100.0  # (100-90)*10

    def test_insufficient_balance(self):
        sim = ExchangeSimulator(initial_capital=100.0, fee_pct=0, slippage_bps=0)

        pos = sim.open_position("SOL/USDT", "long", 10.0, 100.0, timestamp=1000)
        assert pos is None  # Not enough balance

    def test_stop_loss_triggered(self):
        sim = ExchangeSimulator(initial_capital=10000.0, fee_pct=0, slippage_bps=0)

        sim.open_position("SOL/USDT", "long", 10.0, 100.0, timestamp=1000, stop_loss=95.0)

        result = sim.check_stops("SOL/USDT", high=101.0, low=94.0, timestamp=2000)
        assert result is not None
        assert result["pnl"] < 0
        assert "SOL/USDT" not in sim.positions

    def test_take_profit_triggered(self):
        sim = ExchangeSimulator(initial_capital=10000.0, fee_pct=0, slippage_bps=0)

        sim.open_position("SOL/USDT", "long", 10.0, 100.0, timestamp=1000, take_profit=110.0)

        result = sim.check_stops("SOL/USDT", high=111.0, low=99.0, timestamp=2000)
        assert result is not None
        assert result["pnl"] > 0

    def test_equity_calculation(self):
        sim = ExchangeSimulator(initial_capital=10000.0, fee_pct=0, slippage_bps=0)

        sim.open_position("SOL/USDT", "long", 10.0, 100.0, timestamp=1000)
        equity = sim.get_equity({"SOL/USDT": 105.0})
        # balance (9000) + position value (10 * 105 = 1050)
        assert equity == 10050.0


class TestMetrics:
    def test_empty_trades(self):
        metrics = calculate_metrics([], 10000.0)
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0

    def test_basic_metrics(self):
        trades = [
            {"pnl": 100, "pnl_pct": 10.0},
            {"pnl": -50, "pnl_pct": -5.0},
            {"pnl": 75, "pnl_pct": 7.5},
        ]
        metrics = calculate_metrics(trades, 10000.0)

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(66.67, abs=0.1)
        assert metrics.total_pnl == 125.0
        assert metrics.profit_factor == pytest.approx(3.5)

    def test_drawdown_calculation(self):
        trades = [
            {"pnl": 500, "pnl_pct": 5.0},
            {"pnl": -800, "pnl_pct": -8.0},
            {"pnl": 200, "pnl_pct": 2.0},
        ]
        metrics = calculate_metrics(trades, 10000.0)
        assert metrics.max_drawdown_pct > 0

    def test_all_winners(self):
        trades = [
            {"pnl": 100, "pnl_pct": 1.0},
            {"pnl": 200, "pnl_pct": 2.0},
        ]
        metrics = calculate_metrics(trades, 10000.0)
        assert metrics.win_rate == 100.0
        assert metrics.profit_factor == float("inf")
