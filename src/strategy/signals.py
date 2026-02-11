"""Signal generation combining technical analysis and model predictions."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.replay.stream import Candle
from src.features.technical import add_technical_indicators, calculate_signals
from src.models.inference import ModelInference


@dataclass
class Signal:
    """A trading signal with direction and strength."""

    timestamp: int
    symbol: str
    direction: str  # "long" or "short"
    strength: float  # 0..1
    components: dict


class SignalGenerator:
    """Fuses rule-based technical signals with model predictions."""

    def __init__(self):
        self.model = ModelInference()
        self.model_loaded = self.model.load()
        self.candle_buffer: list[dict] = []
        self.buffer_size = 100

    def add_candle(self, candle: Candle):
        """Append a candle to the rolling buffer."""
        self.candle_buffer.append({
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        })
        if len(self.candle_buffer) > self.buffer_size:
            self.candle_buffer.pop(0)

    def generate(self, candle: Candle) -> Optional[Signal]:
        """Produce a signal from the latest market state.

        Returns None when there is insufficient data or when the signal
        is too weak.
        """
        self.add_candle(candle)

        if len(self.candle_buffer) < 50:
            return None

        df = pd.DataFrame(self.candle_buffer)
        df = add_technical_indicators(df)
        df = calculate_signals(df)

        latest = df.iloc[-1]

        signal_technical = float(latest["signal_technical"])
        signal_quant = 0.0

        if self.model_loaded:
            import torch

            feature_cols = [
                "rsi_14", "macd", "macd_signal", "macd_hist",
                "bb_pct", "atr_14", "volume_ratio",
                "ema_9", "ema_21", "ema_50",
            ]

            # z-score normalization using the buffer stats (matches training)
            feat_df = df[feature_cols].copy()
            for col in feature_cols:
                mean = feat_df[col].mean()
                std = feat_df[col].std()
                feat_df[col] = (feat_df[col] - mean) / std if std > 0 else 0

            feat_df["close_norm"] = (df["close"] - df["close"].mean()) / df["close"].std()
            feat_df["volume_norm"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()

            features = torch.tensor(
                feat_df.iloc[-20:].values, dtype=torch.float32
            )
            signal_quant = self.model.predict(features)
            weight_technical = 0.5
            weight_quant = 0.5
        else:
            weight_technical = 1.0
            weight_quant = 0.0

        combined = signal_technical * weight_technical + signal_quant * weight_quant
        direction = "long" if combined > 0 else "short"

        return Signal(
            timestamp=candle.timestamp,
            symbol=candle.symbol,
            direction=direction,
            strength=abs(combined),
            components={"technical": signal_technical, "quant": signal_quant, "combined": combined},
        )
