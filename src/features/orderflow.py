"""Order flow feature engineering (placeholder for phase 2)."""

import pandas as pd


def add_orderflow_features(df: pd.DataFrame, ob_df: pd.DataFrame) -> pd.DataFrame:
    """Merge order-book imbalance features into the candle DataFrame.

    Args:
        df: OHLCV DataFrame with a ``timestamp`` column.
        ob_df: Order book snapshots with ``timestamp`` and ``bid_ask_imbalance``.

    Returns:
        DataFrame with order flow columns joined.
    """
    if ob_df.empty:
        df["bid_ask_imbalance"] = 0.0
        df["spread_bps"] = 0.0
        return df

    ob_df = ob_df[["timestamp", "bid_ask_imbalance", "spread_bps"]].copy()
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        ob_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return df
