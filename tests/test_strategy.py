"""Tests for the strategy components."""

import pytest

from src.strategy.risk import RiskManager, RiskParams


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()

    def test_long_risk_params(self):
        params = self.rm.calculate(balance=10000.0, entry_price=100.0, direction="long")

        assert isinstance(params, RiskParams)
        assert params.position_size > 0
        assert params.stop_loss < 100.0
        assert params.take_profit > 100.0

    def test_short_risk_params(self):
        params = self.rm.calculate(balance=10000.0, entry_price=100.0, direction="short")

        assert params.stop_loss > 100.0
        assert params.take_profit < 100.0

    def test_position_size_respects_limit(self):
        params = self.rm.calculate(balance=10000.0, entry_price=100.0, direction="long")

        # Position value should be max_position_pct of balance
        position_value = params.position_size * 100.0
        assert position_value == pytest.approx(10000.0 * self.rm.max_position_pct)

    def test_risk_reward_ratio(self):
        params = self.rm.calculate(balance=10000.0, entry_price=100.0, direction="long")

        risk = 100.0 - params.stop_loss
        reward = params.take_profit - 100.0

        assert reward / risk == pytest.approx(self.rm.take_profit_ratio, abs=0.01)
