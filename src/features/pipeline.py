"""Feature engineering pipeline combining all feature sources."""

import pandas as pd

from src.features.technical import add_technical_indicators, calculate_signals
from src.features.orderflow import add_orderflow_features


def build_features(ohlcv_df: pd.DataFrame, ob_df: pd.DataFrame = None) -> pd.DataFrame:
    """Run the full feature pipeline on raw OHLCV data.

    Args:
        ohlcv_df: DataFrame with OHLCV columns.
        ob_df: Optional order book snapshots.

    Returns:
        Feature-enriched DataFrame.
    """
    df = add_technical_indicators(ohlcv_df)
    df = calculate_signals(df)

    if ob_df is not None:
        df = add_orderflow_features(df, ob_df)

    return df
