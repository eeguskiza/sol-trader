# Class Imbalance Fix - Weighted BCE Loss

## Problem Identified

**Majority Class Trap:**
- Model predicted only Class 0 (No Trade)
- Achieved 90% accuracy but 0.0 F1-Score
- Dataset is heavily imbalanced:
  - Class 0: ~90% (No Trade)
  - Class 1: ~10% (Buy Signals)

**Root cause:** Standard BCE loss treats all errors equally. The model learned to always predict 0 to minimize loss.

## Solution Implemented

### 1. Weighted Binary Cross Entropy

**BCEWithLogitsLoss with pos_weight parameter:**

```python
pos_weight = num_negatives / num_positives
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

For our data:
- num_negatives = 5,245
- num_positives = 563
- **pos_weight = 9.32**

This means Class 1 (Buy) errors are penalized 9.3x more than Class 0 (No Trade) errors.

### 2. Model Architecture Change

**Modified `model.py`:**
- Added `use_sigmoid` parameter to `PriceActionTransformer.__init__()`
- Made sigmoid activation optional in forward pass
- For training: `use_sigmoid=False` (outputs logits)
- For inference: `use_sigmoid=True` (outputs probabilities)

**Reason:** `BCEWithLogitsLoss` combines sigmoid + BCE for numerical stability. Requires logits as input, not probabilities.

### 3. Training Pipeline Updates

**Modified `train.py`:**

#### Class Weight Calculation (lines 214-226):
```python
num_positives = np.sum(y_train == 1)
num_negatives = np.sum(y_train == 0)

self.pos_weight = num_negatives / num_positives

print(f"  Class 0 (No Trade): {num_negatives} ({num_negatives/len(y_train)*100:.2f}%)")
print(f"  Class 1 (Buy):      {num_positives} ({num_positives/len(y_train)*100:.2f}%)")
print(f"  Calculated pos_weight: {self.pos_weight:.4f}")
```

#### Loss Function (lines 474-476):
```python
pos_weight_tensor = torch.tensor([self.pos_weight]).to(self.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

#### Prediction Counting (lines 314-328):
```python
def count_predictions(self, y_pred, threshold=0.5):
    """Count number of 0s and 1s in predictions."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    num_zeros = np.sum(y_pred_binary == 0)
    num_ones = np.sum(y_pred_binary == 1)
    return num_zeros, num_ones
```

#### Enhanced Logging (lines 517-522):
```python
print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
      f"F1: {train_metrics['f1']:.4f} | Preds: [0s: {train_preds[0]}, 1s: {train_preds[1]}]")
print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
      f"F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | "
      f"Rec: {val_metrics['recall']:.4f} | Preds: [0s: {val_preds[0]}, 1s: {val_preds[1]}]")
```

#### Best Model Selection (lines 524-529):
Changed from loss-based to F1-based:
```python
if val_metrics['f1'] > best_val_f1:
    best_val_f1 = val_metrics['f1']
    best_val_loss = val_loss
    self.save_checkpoint("best_model.pth")
    print("  New best model saved!")
```

## Expected Behavior After Fix

### Training Output:
```
Calculating class weights...
  Class 0 (No Trade): 5245 (90.31%)
  Class 1 (Buy):      563 (9.69%)
  Calculated pos_weight: 9.3162
  This means Class 1 errors are penalized 9.3x more than Class 0

Initializing BCEWithLogitsLoss with pos_weight=9.3162

Epoch [1/100]
  Train Loss: 0.4523 | Acc: 0.7234 | F1: 0.3456 | Preds: [0s: 4812, 1s: 932]
  Val Loss:   0.4612 | Acc: 0.7156 | F1: 0.3321 | Prec: 0.4523 | Rec: 0.2678 | Preds: [0s: 1203, 1s: 186]
  New best model saved!
```

### Key Indicators Model is Learning:
1. **Preds showing 1s:** Previously all 0s, now predicting some 1s
2. **Non-zero Recall:** Model detects some buy signals
3. **Increasing F1:** Balanced performance improving
4. **Precision vs Recall trade-off:** Can tune threshold later

## Mathematical Explanation

### Standard BCE Loss:
```
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
Equal penalty for false positives and false negatives.

### Weighted BCE Loss:
```
L = -[w*y*log(ŷ) + (1-y)*log(1-ŷ)]
```
where w = pos_weight = 9.32

**Effect:**
- False negative (missing Buy signal): Loss × 9.32
- False positive (false Buy signal): Loss × 1.0

Model learns to avoid missing Buy signals even at cost of some false positives.

## Validation

Run test to verify fix:
```bash
python test_training_setup.py
```

Expected output:
```
Calculating class weights...
  Class 0 (No Trade): 5245 (90.31%)
  Class 1 (Buy):      563 (9.69%)
  Calculated pos_weight: 9.3162
  This means Class 1 errors are penalized 9.3x more than Class 0
```

## Files Modified

1. **src/quant_engine/model.py**
   - Added `use_sigmoid` parameter
   - Made sigmoid optional in forward pass

2. **src/quant_engine/train.py**
   - Added class weight calculation
   - Switched to BCEWithLogitsLoss
   - Added prediction counting
   - Enhanced logging
   - Changed best model selection to F1-based

## Trade-offs

### Advantages:
- Solves majority class trap
- Model learns to predict both classes
- Better F1-Score for minority class
- Mathematically principled approach

### Considerations:
- May increase false positives initially
- Requires threshold tuning for production
- Precision may be lower than recall initially

## Threshold Tuning (Post-Training)

After training, adjust classification threshold:

```python
# Default: threshold = 0.5
# Conservative (higher precision): threshold = 0.6-0.7
# Aggressive (higher recall): threshold = 0.3-0.4

y_pred_probs = model(x)
y_pred_binary = (y_pred_probs >= 0.6).int()  # Conservative
```

For trading, prefer **high precision** (fewer false buy signals) over high recall.

## Monitoring During Training

Watch these metrics:
1. **Prediction counts:** Should show increasing 1s over epochs
2. **Recall:** Should be > 0.0 (was 0.0 before fix)
3. **F1-Score:** Should improve from 0.0 to 0.3-0.5+
4. **Precision:** Should be > 0.3 for usable model

If after 10 epochs recall is still 0:
- Increase pos_weight to 12-15
- Decrease learning rate to 0.0005
- Check for data quality issues
