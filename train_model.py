"""
Main training script for the Transformer model.

Execute: python train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.quant_engine.train import Trainer


def main():
    """Execute model training."""

    # Configuration
    config = {
        'data_path': 'data/processed/training_dataset.parquet',
        'checkpoint_dir': 'data/checkpoints',
        'window_size': 64,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'early_stopping_patience': 5
    }

    # Initialize trainer
    trainer = Trainer(**config)

    # Run training
    history = trainer.train()

    print("\nTraining history saved. Model ready for inference.")


if __name__ == "__main__":
    main()
