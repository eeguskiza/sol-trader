"""
Test training setup without running full training.

Verifies data loading, preprocessing, and model initialization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.quant_engine.train import Trainer


def test_setup():
    """Test training setup."""

    print("Testing training setup...")
    print("="*70)

    # Initialize trainer with minimal config
    trainer = Trainer(
        data_path="data/processed/training_dataset.parquet",
        checkpoint_dir="data/checkpoints",
        window_size=64,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=1,  # Just 1 epoch for testing
        early_stopping_patience=5
    )

    # Test data loading
    print("\n1. Testing data loading...")
    X_train, y_train, X_val, y_val = trainer.load_and_preprocess_data()

    print(f"   Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val shape: X={X_val.shape}, y={y_val.shape}")

    # Test dataloader creation
    print("\n2. Testing dataloader creation...")
    train_loader, val_loader = trainer.create_dataloaders(X_train, y_train, X_val, y_val)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Test single batch
    print("\n3. Testing single batch...")
    batch_x, batch_y = next(iter(train_loader))
    print(f"   Batch X shape: {batch_x.shape}")
    print(f"   Batch y shape: {batch_y.shape}")

    # Test model initialization
    print("\n4. Testing model initialization...")
    from src.quant_engine.model import PriceActionTransformer

    input_dim = len(trainer.feature_columns)
    model = PriceActionTransformer(input_dim=input_dim)
    model = model.to(trainer.device)

    print(f"   Model parameters: {model.get_num_params():,}")
    print(f"   Device: {trainer.device}")

    # Test forward pass
    print("\n5. Testing forward pass...")
    batch_x = batch_x.to(trainer.device)
    output = model(batch_x)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n" + "="*70)
    print("All tests passed. Ready for training.")
    print("="*70)
    print("\nRun training with: python train_model.py")


if __name__ == "__main__":
    test_setup()
