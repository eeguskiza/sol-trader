# Model Training Guide

## Architecture

### PriceActionTransformer

```
Input (batch, 64, 14)
    ↓
Linear Projection (14 → 128)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
Global Average Pooling
    ↓
Linear Head (128 → 64 → 1)
    ↓
Sigmoid
    ↓
Output (batch, 1) - Buy probability
```

**Parameters:** 406,785 trainable parameters

**Features:**
- 2 Transformer encoder layers
- 4 attention heads
- 512-dim feedforward network
- 0.1 dropout for regularization
- Xavier uniform weight initialization

## Data Pipeline

### 1. Loading & Preprocessing

```python
# Load parquet file
df = pl.read_parquet("data/processed/training_dataset.parquet")

# Extract features (14 technical indicators)
features = ['open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'atr_14', 'ema_50', 'ema_200',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent']

# Z-score normalization
X_normalized = (X - mean_train) / std_train
```

### 2. Chronological Split

**Critical:** No shuffling to preserve time series order.

```
Total: 7,261 samples
├── Train: 80% (5,808 samples)
└── Val:   20% (1,453 samples)
```

### 3. Sliding Window

Creates sequences of 64 consecutive candles to predict the 65th candle's target.

```
Sequence:  [t-63, t-62, ..., t-1, t]
Target:    label at t
```

**Result:**
- Train sequences: 5,744
- Val sequences: 1,389

### 4. DataLoader

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,      # Preserve order
    pin_memory=True     # GPU optimization
)
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_size` | 64 | Input sequence length |
| `batch_size` | 32 | Samples per batch |
| `learning_rate` | 0.001 | AdamW learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `num_epochs` | 100 | Maximum epochs |
| `early_stopping` | 5 | Patience (epochs) |

### Loss Function

Binary Cross-Entropy (BCE):
```python
BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Optimizer

AdamW with weight decay for better generalization.

## Execution

### Install Dependencies

```bash
pip install torch scikit-learn tqdm
```

### Test Setup

```bash
python test_training_setup.py
```

**Output:**
```
Using device: cuda
GPU: NVIDIA GeForce RTX 5070 Ti
Train samples: 5808
Val samples: 1453
All tests passed. Ready for training.
```

### Run Training

```bash
python train_model.py
```

**Expected output:**
```
Epoch [1/100]
  Train Loss: 0.5234 | Acc: 0.7532 | F1: 0.3421
  Val Loss:   0.4987 | Acc: 0.7821 | F1: 0.4102
  New best model saved!
```

## Metrics

### Classification Metrics

- **Accuracy:** Overall correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall

### Interpretation

For trading, **Precision** is critical:
- High precision = Few false buy signals
- Low recall is acceptable = Miss some opportunities, but avoid bad trades

**Target:** Precision > 0.7, F1 > 0.5

## Output Files

### Checkpoints

Saved in `data/checkpoints/`:

**best_model.pth:**
```python
{
    'model_state_dict': state_dict,
    'feature_columns': list,
    'window_size': int,
    'input_dim': int
}
```

**scaler.pkl:**
```python
{
    'mean': np.ndarray,  # (14,)
    'std': np.ndarray,   # (14,)
    'feature_columns': list
}
```

## GPU Utilization

### RTX 5070 Ti Optimization

- Mixed precision training (optional): `torch.cuda.amp.autocast()`
- Batch size: Increase to 64 or 128 if memory allows
- Pin memory: `pin_memory=True` in DataLoader

### Memory Usage

- Model: ~2 MB
- Batch (32x64x14): ~1 MB
- Optimizer state: ~4 MB
- **Total:** ~10 MB per batch (negligible for 16GB VRAM)

Can safely increase batch size to 128 for faster training.

## Early Stopping

Prevents overfitting by monitoring validation loss:

```
If val_loss doesn't improve for 5 consecutive epochs:
    Stop training
    Load best checkpoint
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
batch_size = 16
```

### Poor Performance (F1 < 0.3)
```bash
# Increase model capacity
d_model = 256
num_layers = 4

# Or adjust learning rate
learning_rate = 0.0005
```

### Overfitting (train_loss << val_loss)
```bash
# Increase dropout
dropout = 0.2

# Increase weight decay
weight_decay = 0.05
```

## Advanced Configurations

### Hyperparameter Tuning

Edit `train_model.py`:

```python
config = {
    'window_size': 128,        # Longer context
    'batch_size': 64,          # Larger batches
    'learning_rate': 0.0005,   # Lower LR
    'd_model': 256,            # Bigger model
    'num_layers': 4,           # Deeper network
}
```

### Custom Training Loop

For full control, import Trainer directly:

```python
from src.quant_engine.train import Trainer

trainer = Trainer(...)
history = trainer.train()

# Access training history
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
plt.savefig('training_curve.png')
```

## Next Steps

After training completes:

1. Evaluate on test set
2. Implement inference pipeline
3. Backtest on historical data
4. Deploy to live trading (paper trading first)
