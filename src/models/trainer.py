"""Training loop for the PriceActionTransformer."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from config.settings import settings


def get_device() -> torch.device:
    """Get best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class Trainer:
    """Train a PriceActionTransformer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: torch.Tensor,
        lr: float = 1e-4,
        checkpoint_dir: str = None,
    ):
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.class_weights = class_weights.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )

        self.checkpoint_dir = Path(checkpoint_dir or settings.data_path / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return avg_loss, accuracy, precision, recall, f1

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "input_dim": self.model.input_dim,
            "d_model": self.model.d_model,
            "nhead": self.model.nhead,
            "num_layers": self.model.num_layers,
        }

        torch.save(checkpoint, self.checkpoint_dir / "latest_model.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
            logger.info(f"Saved best model with val_loss={val_loss:.4f}")

    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc, precision, recall, f1 = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(f1)

            self.scheduler.step(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            if on_progress:
                on_progress(epoch, epochs)

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Acc: {val_acc:.3f} | "
                f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}"
            )

            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return history
