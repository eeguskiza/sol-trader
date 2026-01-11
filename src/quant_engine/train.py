"""
Training pipeline for Transformer-based trading model.

Handles data preprocessing, sliding window creation, training loop,
and model evaluation with proper time series handling and class imbalance correction.
"""

import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .model import PriceActionTransformer


class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset for time series data.

    Creates sequences of length window_size to predict target at window_size+1.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 64
    ):
        """
        Initialize dataset.

        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            window_size: Length of input sequence
        """
        self.features = features
        self.targets = targets
        self.window_size = window_size

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and target.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (sequence, target)
        """
        # Get sequence from idx to idx+window_size
        x = self.features[idx:idx + self.window_size]

        # Target is the label at the end of the window
        y = self.targets[idx + self.window_size]

        return torch.FloatTensor(x), torch.FloatTensor([y])


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training if validation loss doesn't improve for patience epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class Trainer:
    """
    Training pipeline for price action prediction model.

    Handles data loading, preprocessing, training, validation, and checkpointing.
    Includes class imbalance correction via weighted loss.
    """

    def __init__(
        self,
        data_path: str = "data/processed/training_dataset.parquet",
        checkpoint_dir: str = "data/checkpoints",
        window_size: int = 64,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        early_stopping_patience: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            data_path: Path to training dataset
            checkpoint_dir: Directory to save checkpoints
            window_size: Length of input sequences
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.data_path = Path(data_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

        # Placeholders
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_columns = None
        self.pos_weight = None

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset and perform preprocessing.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        print("\nLoading dataset...")
        df = pl.read_parquet(self.data_path)
        print(f"Loaded {len(df)} samples")

        # Define feature columns (exclude target and metadata)
        exclude_cols = {
            'timestamp', 'target', 'future_close', 'future_return',
            'future_return_pct', 'threshold'
        }
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        print(f"Using {len(self.feature_columns)} features: {self.feature_columns}")

        # Extract features and targets
        X = df.select(self.feature_columns).to_numpy()
        y = df['target'].to_numpy()

        # Chronological train/val split (80/20)
        split_idx = int(len(X) * 0.8)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")

        # Calculate class weights for imbalanced dataset
        print("\nCalculating class weights...")
        num_positives = np.sum(y_train == 1)
        num_negatives = np.sum(y_train == 0)

        print(f"  Class 0 (No Trade): {num_negatives} ({num_negatives/len(y_train)*100:.2f}%)")
        print(f"  Class 1 (Buy):      {num_positives} ({num_positives/len(y_train)*100:.2f}%)")

        # Calculate pos_weight for BCEWithLogitsLoss
        # pos_weight = num_negatives / num_positives
        self.pos_weight = num_negatives / num_positives
        print(f"  Calculated pos_weight: {self.pos_weight:.4f}")
        print(f"  This means Class 1 errors are penalized {self.pos_weight:.1f}x more than Class 0")

        # Global Z-score normalization
        print("\nNormalizing features (Z-score)...")
        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero

        X_train = (X_train - self.scaler_mean) / self.scaler_std
        X_val = (X_val - self.scaler_mean) / self.scaler_std

        print("Normalization complete")

        return X_train, y_train, X_val, y_val

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders with sliding window.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"\nCreating sliding window datasets (window_size={self.window_size})...")

        train_dataset = TimeSeriesDataset(X_train, y_train, self.window_size)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.window_size)

        print(f"Train sequences: {len(train_dataset)}")
        print(f"Val sequences: {len(val_dataset)}")

        # DataLoaders (no shuffling for time series)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep chronological order
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )

        return train_loader, val_loader

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        y_pred_binary = (y_pred >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }

        return metrics

    def count_predictions(self, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
        """
        Count number of 0s and 1s in predictions.

        Args:
            y_pred: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Tuple of (num_zeros, num_ones)
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        num_zeros = np.sum(y_pred_binary == 0)
        num_ones = np.sum(y_pred_binary == 1)
        return num_zeros, num_ones

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, Dict[str, float], Tuple[int, int]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Tuple of (average_loss, metrics, prediction_counts)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass (model outputs logits, not probabilities)
            optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics (convert logits to probabilities for metrics)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.detach().cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        metrics = self.calculate_metrics(np.array(all_targets), np.array(all_preds))
        pred_counts = self.count_predictions(np.array(all_preds))

        return avg_loss, metrics, pred_counts

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict[str, float], Tuple[int, int]]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, metrics, prediction_counts)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass (model outputs logits)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)

                # Track metrics (convert logits to probabilities)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(np.array(all_targets), np.array(all_preds))
        pred_counts = self.count_predictions(np.array(all_preds))

        return avg_loss, metrics, pred_counts

    def save_checkpoint(self, filename: str = "best_model.pth"):
        """
        Save model checkpoint and scaler.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        scaler_path = self.checkpoint_dir / "scaler.pkl"

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_columns': self.feature_columns,
            'window_size': self.window_size,
            'input_dim': len(self.feature_columns),
            'pos_weight': self.pos_weight
        }, checkpoint_path)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'mean': self.scaler_mean,
                'std': self.scaler_std,
                'feature_columns': self.feature_columns
            }, f)

        print(f"  Saved checkpoint: {checkpoint_path}")
        print(f"  Saved scaler: {scaler_path}")

    def train(self):
        """
        Execute complete training pipeline with class imbalance correction.
        """
        print("="*70)
        print("TRAINING PIPELINE - PRICE ACTION TRANSFORMER")
        print("="*70)

        # Load and preprocess data
        X_train, y_train, X_val, y_val = self.load_and_preprocess_data()

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(X_train, y_train, X_val, y_val)

        # Initialize model (without sigmoid, we'll use BCEWithLogitsLoss)
        print("\nInitializing model...")
        input_dim = len(self.feature_columns)
        self.model = PriceActionTransformer(input_dim=input_dim, use_sigmoid=False)
        self.model = self.model.to(self.device)

        print(f"Model parameters: {self.model.get_num_params():,}")

        # Weighted BCE loss to handle class imbalance
        print(f"\nInitializing BCEWithLogitsLoss with pos_weight={self.pos_weight:.4f}")
        pos_weight_tensor = torch.tensor([self.pos_weight]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        # Training loop
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print("="*70)

        best_val_loss = float('inf')
        best_val_f1 = 0.0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_metrics, train_preds = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_loss, val_metrics, val_preds = self.validate(val_loader, criterion)

            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            # Print progress with prediction counts
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | Preds: [0s: {train_preds[0]}, 1s: {train_preds[1]}]")
            print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | "
                  f"Rec: {val_metrics['recall']:.4f} | Preds: [0s: {val_preds[0]}, 1s: {val_preds[1]}]")

            # Save best model based on F1 score (more important than loss for imbalanced data)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
                print("  New best model saved!")

            # Early stopping based on validation loss
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

            print("-" * 70)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"Scaler saved to: {self.checkpoint_dir / 'scaler.pkl'}")

        return history


if __name__ == "__main__":
    # Initialize and run training
    trainer = Trainer(
        data_path="data/processed/training_dataset.parquet",
        checkpoint_dir="data/checkpoints",
        window_size=64,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        early_stopping_patience=5
    )

    history = trainer.train()
