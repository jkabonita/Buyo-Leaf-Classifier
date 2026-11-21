# ⚠️ Model Performance Issue Detected

## Problem

The model is showing **low confidence** on actual Buyo leaves:
- Real Buyo leaf classified as "NO BUYO" with only 29.40% confidence
- Expected: >50% confidence for valid Buyo leaves
- Actual: <30% confidence

## Possible Causes

### 1. **Model Quality** (Most Likely)
- The trained model (`buyo_best.pth`) may not be performing well
- Training data quality issues
- Insufficient training epochs
- Overfitting or underfitting

### 2. **Preprocessing Mismatch**
- Image normalization values may not match training
- Image size (224x224) may not be optimal
- Color space issues

### 3. **Dataset Issues**
- Not enough training examples
- Class imbalance
- Poor quality training images

## Immediate Solutions

### Option 1: Lower Thresholds (✅ Already Applied)
```python
DETECTION_THRESHOLD = 0.50  # From 0.60
BUYO_CONFIDENCE_MIN = 0.25  # From 0.45
```

This will show more detections, even with lower confidence.

### Option 2: Retrain the Model (Recommended)

#### Check Current Model Performance
```bash
python evaluate.py
```

This will show:
- Validation accuracy
- Per-class accuracy
- Confusion matrix

#### Retrain with Better Settings
```bash
# Option A: Single model (faster, 1-2 hours)
python train_convnext_optimized.py

# Option B: Ensemble (best accuracy, 3-4 hours)
python train_ensemble.py
```

#### Before Retraining:
1. **Check dataset quality:**
   ```bash
   python diagnose_dataset.py
   ```

2. **Remove duplicates:**
   ```bash
   python find_duplicates.py
   python remove_duplicates.py
   ```

3. **Augment if needed:**
   ```bash
   python augment_data.py
   ```

### Option 3: Use Different Model

Try loading a different checkpoint if available:
```python
# In laptop_client_gui.py, change MODEL_PATH to:
MODEL_PATH = "buyo_convnext.pth"  # Or buyo_effnet.pth or buyo_resnet50.pth
```

## Expected Performance

A well-trained model should show:
- **CLASS A leaves**: 70-85% confidence
- **CLASS B leaves**: 65-80% confidence
- **CLASS C leaves**: 60-75% confidence
- **CLASS D leaves**: 60-75% confidence
- **CLASS E leaves**: 65-80% confidence

## Testing the Model

### Test with Known Images
```bash
# Test single image
python laptop_client/laptop_client_gui.py
# Click "📷 Classify Single Image"
# Select a clear CLASS A image from CLASS A/ folder
```

### Check Model Info
```bash
python check_model.py
```

Shows:
- Model architecture
- Number of parameters
- Training metadata

## Current Temporary Fix

I've lowered the thresholds so the model will:
1. Show predictions even with 25%+ confidence
2. Still label as "NO BUYO" if confidence < 25%
3. Color-code based on confidence:
   - 🟢 Green: >50% (High confidence)
   - 🟡 Orange: 25-50% (Medium confidence)
   - 🔴 Red: <25% (Low confidence)

## What to Watch For

### Good Signs ✅
- Confidence >50% on Buyo leaves
- Correct class predictions
- Consistent results across similar images

### Bad Signs ❌
- Confidence <30% on clear Buyo images
- Random class predictions
- "NO BUYO" on actual Buyo leaves

## Next Steps

1. **Immediate**: Test with clear Buyo leaf images from your dataset
2. **Short-term**: Evaluate model with `evaluate.py`
3. **Long-term**: Retrain model if accuracy <70%

## Diagnosis Commands

```bash
# 1. Check dataset quality
python diagnose_dataset.py

# 2. Check model performance
python evaluate.py

# 3. Test inference speed
python test_inference.py

# 4. Inspect model
python check_model.py
```

## Contact

If issues persist:
1. Check console output for errors
2. Verify `buyo_best.pth` is not corrupted
3. Try retraining with ensemble method
4. Consider collecting more training data

---

**Status**: Thresholds lowered, model may need retraining
**Priority**: Medium (functional but low confidence)
**Action Required**: Evaluate model performance and consider retraining

Last Updated: November 21, 2026
