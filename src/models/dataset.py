"""Dataset and DataLoader creation for price action prediction."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from loguru import logger

from src.data.storage.duckdb_client import DuckDBClient
from src.features.technical import add_technical_indicators


class PriceActionDataset(Dataset):
    """Dataset for price action prediction with ATR-based labeling."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        sequence_length: int = 20,
        lookahead: int = 4,
        atr_multiplier: float = 1.5,
    ):
        self.sequence_length = sequence_length
        self.lookahead = lookahead
        self.atr_multiplier = atr_multiplier

        duckdb = DuckDBClient()
        df = duckdb.get_ohlcv(symbol, timeframe, start_ts, end_ts)

        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        logger.info(f"Loaded {len(df)} candles for dataset")

        df = add_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)

        logger.info(f"After dropna: {len(df)} candles")

        df = self._create_labels(df)

        self.feature_columns = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_pct", "atr_14", "volume_ratio",
            "ema_9", "ema_21", "ema_50",
        ]

        # z-score normalization
        for col in self.feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std if std > 0 else 0

        df["close_norm"] = (df["close"] - df["close"].mean()) / df["close"].std()
        df["volume_norm"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        self.feature_columns.extend(["close_norm", "volume_norm"])

        self.df = df
        self.valid_indices = list(range(self.sequence_length, len(self.df)))

        logger.info(
            f"Dataset created: {len(self.valid_indices)} samples, "
            f"{len(self.feature_columns)} features"
        )

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels: 1 if price rises by ATR threshold within lookahead."""
        df = df.copy()

        df["future_max"] = (
            df["high"]
            .rolling(window=self.lookahead, min_periods=1)
            .max()
            .shift(-self.lookahead)
        )

        threshold = df["atr_14"] * self.atr_multiplier
        df["label"] = ((df["future_max"] - df["close"]) >= threshold).astype(int)

        df = df.iloc[: -self.lookahead].copy()

        pos_count = df["label"].sum()
        total = len(df)
        logger.info(
            f"Labels: {pos_count} positive ({100 * pos_count / total:.1f}%), "
            f"{total - pos_count} negative"
        )

        return df

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple:
        actual_idx = self.valid_indices[idx]
        start_idx = actual_idx - self.sequence_length

        features = self.df[self.feature_columns].iloc[start_idx:actual_idx].values
        label = self.df["label"].iloc[actual_idx]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def get_class_weights(self) -> torch.Tensor:
        """Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance."""
        labels = self.df["label"].iloc[self.valid_indices].values
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count

        if pos_count == 0 or neg_count == 0:
            return torch.tensor([1.0])

        return torch.tensor([neg_count / pos_count], dtype=torch.float32)


def create_dataloaders(
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    sequence_length: int = 20,
) -> tuple:
    """Create train and validation dataloaders with chronological split."""
    dataset = PriceActionDataset(
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        sequence_length=sequence_length,
    )

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    train_subset = Subset(dataset, list(range(train_size)))
    val_subset = Subset(dataset, list(range(train_size, total_size)))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")

    return train_loader, val_loader, dataset.get_class_weights()
