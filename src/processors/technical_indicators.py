"""
Technical Indicator Feature Engineering using Polars.

This module provides production-ready, vectorized implementations of technical
indicators using Polars expressions for maximum CPU performance.

All calculations are performed using native Polars operations (no loops, no pandas).
"""

import logging
from typing import List

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    High-performance feature engineering for OHLCV data.

    Implements technical indicators using vectorized Polars expressions
    for optimal CPU performance in trading systems.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        logger.info("FeatureEngineer initialized")

    def add_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add all technical indicators to the DataFrame.

        Args:
            df: Polars DataFrame with OHLCV data (timestamp, open, high, low, close, volume)

        Returns:
            DataFrame with added technical indicator columns

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

        logger.info("Adding all technical indicators")

        # Add indicators sequentially (order matters for some dependencies)
        df = self.add_rsi(df, period=14)
        df = self.add_atr(df, period=14)
        df = self.add_ema(df, periods=[50, 200])
        df = self.add_bollinger_bands(df, period=20, std_dev=2.0)

        logger.info(f"Added {len(df.columns) - 6} technical indicators")  # 6 original OHLCV + timestamp
        return df

    def add_rsi(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        Add Relative Strength Index (RSI) using Polars expressions.

        RSI measures momentum by comparing upward and downward price movements.
        Formula: RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss

        Args:
            df: DataFrame with 'close' column
            period: RSI period (default: 14)

        Returns:
            DataFrame with 'rsi_{period}' column added
        """
        logger.debug(f"Calculating RSI({period})")

        df = df.with_columns([
            # Calculate price changes
            (pl.col("close") - pl.col("close").shift(1)).alias("price_change"),
        ])

        df = df.with_columns([
            # Separate gains and losses
            pl.when(pl.col("price_change") > 0)
              .then(pl.col("price_change"))
              .otherwise(0.0)
              .alias("gain"),
            pl.when(pl.col("price_change") < 0)
              .then(pl.col("price_change").abs())
              .otherwise(0.0)
              .alias("loss"),
        ])

        # Calculate exponential moving average of gains and losses
        # Using Wilder's smoothing (same as EMA with alpha = 1/period)
        df = df.with_columns([
            pl.col("gain").ewm_mean(span=period, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(span=period, adjust=False).alias("avg_loss"),
        ])

        # Calculate RS and RSI
        df = df.with_columns([
            (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs"),
        ])

        df = df.with_columns([
            (100.0 - (100.0 / (1.0 + pl.col("rs")))).alias(f"rsi_{period}"),
        ])

        # Clean up intermediate columns
        df = df.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss", "rs"])

        return df

    def add_atr(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        Add Average True Range (ATR) using Polars expressions.

        ATR measures market volatility by decomposing the entire range of price
        movement. Critical for dynamic stop-loss positioning.

        Formula: ATR = EMA of True Range, where
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default: 14)

        Returns:
            DataFrame with 'atr_{period}' column added
        """
        logger.debug(f"Calculating ATR({period})")

        df = df.with_columns([
            # Calculate True Range components
            (pl.col("high") - pl.col("low")).alias("hl_range"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc_range"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc_range"),
        ])

        # True Range is the maximum of the three ranges
        df = df.with_columns([
            pl.max_horizontal(["hl_range", "hc_range", "lc_range"]).alias("true_range"),
        ])

        # ATR is the exponential moving average of True Range
        df = df.with_columns([
            pl.col("true_range").ewm_mean(span=period, adjust=False).alias(f"atr_{period}"),
        ])

        # Clean up intermediate columns
        df = df.drop(["hl_range", "hc_range", "lc_range", "true_range"])

        return df

    def add_ema(self, df: pl.DataFrame, periods: List[int] = [50, 200]) -> pl.DataFrame:
        """
        Add Exponential Moving Averages (EMA) using Polars expressions.

        EMA gives more weight to recent prices, making it more responsive to
        price changes than Simple Moving Average (SMA).

        Formula: EMA = (Close - EMA_prev) * multiplier + EMA_prev
        where multiplier = 2 / (period + 1)

        Args:
            df: DataFrame with 'close' column
            periods: List of EMA periods (default: [50, 200])

        Returns:
            DataFrame with 'ema_{period}' columns added
        """
        logger.debug(f"Calculating EMAs: {periods}")

        for period in periods:
            df = df.with_columns([
                pl.col("close").ewm_mean(span=period, adjust=False).alias(f"ema_{period}"),
            ])

        return df

    def add_bollinger_bands(
        self,
        df: pl.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pl.DataFrame:
        """
        Add Bollinger Bands using Polars expressions.

        Bollinger Bands measure volatility and potential price extremes.
        Consists of a middle band (SMA) and upper/lower bands (SMA ± std_dev * σ).

        Args:
            df: DataFrame with 'close' column
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)

        Returns:
            DataFrame with 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width' columns added
        """
        logger.debug(f"Calculating Bollinger Bands (period={period}, std={std_dev})")

        # Calculate middle band (Simple Moving Average)
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=period).alias("bb_middle"),
        ])

        # Calculate rolling standard deviation
        df = df.with_columns([
            pl.col("close").rolling_std(window_size=period).alias("bb_std"),
        ])

        # Calculate upper and lower bands
        df = df.with_columns([
            (pl.col("bb_middle") + (pl.col("bb_std") * std_dev)).alias("bb_upper"),
            (pl.col("bb_middle") - (pl.col("bb_std") * std_dev)).alias("bb_lower"),
        ])

        # Calculate band width (measure of volatility)
        df = df.with_columns([
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias("bb_width"),
        ])

        # Calculate %B (price position within bands)
        df = df.with_columns([
            ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower")))
            .alias("bb_percent"),
        ])

        # Clean up intermediate column
        df = df.drop(["bb_std"])

        return df

    def add_macd(
        self,
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pl.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) using Polars expressions.

        MACD shows the relationship between two moving averages of prices.

        Args:
            df: DataFrame with 'close' column
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns added
        """
        logger.debug(f"Calculating MACD({fast_period}, {slow_period}, {signal_period})")

        # Calculate fast and slow EMAs
        df = df.with_columns([
            pl.col("close").ewm_mean(span=fast_period, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=slow_period, adjust=False).alias("ema_slow"),
        ])

        # Calculate MACD line
        df = df.with_columns([
            (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd"),
        ])

        # Calculate signal line (EMA of MACD)
        df = df.with_columns([
            pl.col("macd").ewm_mean(span=signal_period, adjust=False).alias("macd_signal"),
        ])

        # Calculate histogram (MACD - Signal)
        df = df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
        ])

        # Clean up intermediate columns
        df = df.drop(["ema_fast", "ema_slow"])

        return df

    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add volume-based features using Polars expressions.

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume features added
        """
        logger.debug("Calculating volume features")

        # Volume moving averages
        df = df.with_columns([
            pl.col("volume").rolling_mean(window_size=20).alias("volume_sma_20"),
            pl.col("volume").rolling_mean(window_size=50).alias("volume_sma_50"),
        ])

        # Volume ratio (current volume vs average)
        df = df.with_columns([
            (pl.col("volume") / pl.col("volume_sma_20")).alias("volume_ratio"),
        ])

        return df


if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np
    from datetime import datetime, timedelta

    # Generate synthetic OHLCV data
    n_candles = 1000
    base_price = 100.0

    timestamps = [datetime.now() - timedelta(minutes=15*i) for i in range(n_candles)]
    timestamps.reverse()

    # Random walk price data
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n_candles)
    close_prices = base_price * np.cumprod(1 + returns)

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, n_candles)),
        "high": close_prices * (1 + np.random.uniform(0, 0.02, n_candles)),
        "low": close_prices * (1 + np.random.uniform(-0.02, 0, n_candles)),
        "close": close_prices,
        "volume": np.random.uniform(1000, 10000, n_candles),
    })

    # Apply feature engineering
    engineer = FeatureEngineer()
    df_with_features = engineer.add_all_features(df)

    print("Original columns:", df.columns)
    print("\nColumns after feature engineering:", df_with_features.columns)
    print(f"\nAdded {len(df_with_features.columns) - len(df.columns)} new features")

    # Display sample with features
    print("\nSample data with features:")
    print(df_with_features.select([
        "timestamp", "close", "rsi_14", "atr_14",
        "ema_50", "ema_200", "bb_upper", "bb_lower"
    ]).tail(10))

    # Performance test
    import time
    start = time.time()
    for _ in range(10):
        _ = engineer.add_all_features(df)
    elapsed = time.time() - start
    print(f"\nPerformance: {elapsed/10:.4f}s per run (avg over 10 runs)")
    print(f"Processing speed: {len(df) / (elapsed/10):.0f} candles/second")
