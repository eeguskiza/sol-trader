"""Technical indicator feature engineering using the ``ta`` library."""

import numpy as np
import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns open, high, low, close, volume.

    Returns:
        DataFrame with additional indicator columns.
    """
    df = df.copy()

    # Momentum
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # Trend -- MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Volatility -- Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volatility -- ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # Trend -- EMAs
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    # Volume
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    return df


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rule-based signal scores from technical indicators.

    Each signal component is in [-1, 1].  The combined ``signal_technical``
    is a weighted average.

    Args:
        df: DataFrame with indicator columns from ``add_technical_indicators``.

    Returns:
        DataFrame with additional signal columns.
    """
    df = df.copy()

    df["signal_rsi"] = np.where(df["rsi_14"] < 30, 1, np.where(df["rsi_14"] > 70, -1, 0))
    df["signal_macd"] = np.where(df["macd_hist"] > 0, 1, -1)
    df["signal_bb"] = np.where(df["bb_pct"] < 0.2, 1, np.where(df["bb_pct"] > 0.8, -1, 0))
    df["signal_ema"] = np.where(df["ema_9"] > df["ema_21"], 1, -1)

    df["signal_technical"] = (
        df["signal_rsi"] * 0.25
        + df["signal_macd"] * 0.25
        + df["signal_bb"] * 0.25
        + df["signal_ema"] * 0.25
    )

    return df
