"""Backtest performance metrics calculation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestMetrics:
    """Summary statistics for a completed backtest."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float


def calculate_metrics(trade_history: list[dict], initial_capital: float) -> BacktestMetrics:
    """Compute performance metrics from a list of closed trades.

    Args:
        trade_history: List of trade dicts with "pnl" and "pnl_pct" keys.
        initial_capital: Starting capital for return calculations.

    Returns:
        BacktestMetrics dataclass with all computed values.
    """
    if not trade_history:
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_pnl_pct=0,
            avg_pnl=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
        )

    pnls = [t["pnl"] for t in trade_history]
    pnl_pcts = [t["pnl_pct"] for t in trade_history]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Build equity curve for drawdown calculation
    equity_curve = [initial_capital]
    for pnl in pnls:
        equity_curve.append(equity_curve[-1] + pnl)

    peak = initial_capital
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Risk-adjusted returns
    returns = np.array(pnl_pcts) / 100
    sharpe = (
        np.mean(returns) / np.std(returns) * np.sqrt(252)
        if len(returns) > 1 and np.std(returns) > 0
        else 0
    )

    downside = returns[returns < 0]
    sortino = (
        np.mean(returns) / np.std(downside) * np.sqrt(252)
        if len(downside) > 1 and np.std(downside) > 0
        else 0
    )

    return BacktestMetrics(
        total_trades=len(trade_history),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trade_history) * 100 if trade_history else 0,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl / initial_capital * 100,
        avg_pnl=float(np.mean(pnls)),
        avg_win=float(np.mean(wins)) if wins else 0,
        avg_loss=float(np.mean(losses)) if losses else 0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        max_drawdown_pct=max_dd * 100,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
    )
