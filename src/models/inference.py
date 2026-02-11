"""Optimized inference wrapper for the trained Transformer model."""

from pathlib import Path

import torch
from loguru import logger

from config.settings import settings


class ModelInference:
    """Load a trained checkpoint and run predictions."""

    def __init__(self, checkpoint_path: str = None):
        self.device = self._get_device()
        self.model = None
        self.checkpoint_path = checkpoint_path or str(
            settings.data_path / "checkpoints" / "best_model.pt"
        )

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> bool:
        """Load model weights from a checkpoint file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not Path(self.checkpoint_path).exists():
            logger.warning(f"No checkpoint found at {self.checkpoint_path}")
            return False

        from src.models.transformer import PriceActionTransformer

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = PriceActionTransformer(
            input_dim=checkpoint.get("input_dim", 12),
            d_model=checkpoint.get("d_model", 128),
            nhead=checkpoint.get("nhead", 4),
            num_layers=checkpoint.get("num_layers", 2),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {self.checkpoint_path}")
        return True

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> float:
        """Run inference and return a signal in [-1, 1].

        Args:
            features: Tensor of shape (seq_len, n_features) or (batch, seq_len, n_features).

        Returns:
            Signal strength mapped to [-1, 1].
        """
        if self.model is None:
            return 0.0

        features = features.to(self.device)
        if features.dim() == 2:
            features = features.unsqueeze(0)

        logits = self.model(features)
        prob = torch.sigmoid(logits).item()

        return (prob - 0.5) * 2
