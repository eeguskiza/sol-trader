"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": range(n),
        "open": close + np.random.uniform(-0.5, 0.5, n),
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    })


def test_add_technical_indicators():
    from src.features.technical import add_technical_indicators

    df = _make_ohlcv()
    result = add_technical_indicators(df)

    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "bb_upper" in result.columns
    assert "atr_14" in result.columns
    assert "ema_9" in result.columns
    assert "volume_ratio" in result.columns

    # RSI should be between 0 and 100 for non-NaN values
    rsi_values = result["rsi_14"].dropna()
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()


def test_calculate_signals():
    from src.features.technical import add_technical_indicators, calculate_signals

    df = _make_ohlcv()
    df = add_technical_indicators(df)
    result = calculate_signals(df)

    assert "signal_technical" in result.columns

    # signal_technical should be in [-1, 1]
    sig = result["signal_technical"].dropna()
    assert (sig >= -1).all()
    assert (sig <= 1).all()


def test_build_features_pipeline():
    from src.features.pipeline import build_features

    df = _make_ohlcv()
    result = build_features(df)

    assert "signal_technical" in result.columns
    assert len(result) == len(df)
