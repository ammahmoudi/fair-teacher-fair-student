# Knowledge Distillation in Time-LLM for Time Series Prediction

## 🎓 Overview

This project has separate knowledge-distillation objectives for continuous
time-series forecasting and ECG classification. The original sections below
describe forecasting regression. MIT-BIH classification uses a categorical
cross-entropy/KL objective and must not reuse the regression interpretation.

## ECG Classification Distillation

For student logits `z_s`, teacher logits `z_t`, labels `y`, and temperature
`T`, the ECG wrapper uses:

```text
L_ECG = alpha * CE(z_s, y)
            + beta * T^2 * KL(softmax(z_t / T) || softmax(z_s / T))
```

- `alpha` weights supervised classification.
- `beta` weights teacher-distribution matching.
- `T` controls class-probability softness; unlike BG regression, it is an
    operative scientific parameter.
- The locked binary-ectopy run uses `alpha=0.5`, `beta=0.5`, and `T=1.0`,
    selected by a validation-only screen.
- A weighted sampler and class-weighted CE are mutually exclusive to avoid
    applying the imbalance correction twice.
- O2 jointly learns group-conditional per-class affine scale/bias parameters
    on student logits; BG O2 instead calibrates continuous predictions.

Classification checkpoints are selected on validation only and evaluated with
accuracy, macro-F1, weighted-F1, class recall, and support-qualified EO. Test
predictions are final evaluation artifacts, not tuning inputs.

## 📊 Distillation Loss Function

The distillation uses a **simplified two-component loss function** specifically designed for time series regression:

```python
Total_Loss = α × Ground_Truth_Loss + β × Teacher_Output_Loss
```

### 🔧 Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `α (alpha)` | 0.5 | Weight for ground truth supervision |
| `β (beta)` | 0.5 | Weight for teacher output matching |

> **Note**: Previous KL divergence loss was removed as it's inappropriate for continuous time series regression tasks.

## 🔍 Loss Components Detailed

### 1. 📈 Ground Truth Loss (MSE)
```python
loss_gt = MSE(student_predictions, true_time_series)
```

**Purpose**: Ensures the student model learns the actual time series patterns.
- **Function**: Standard Mean Squared Error
- **Target**: Real time series values (ground truth)
- **Why MSE**: Time series prediction is a regression task where we want to minimize the numerical difference between predicted and actual values

### 2. 🎯 Teacher Output Loss (MSE)
```python
loss_teacher = MSE(student_predictions, teacher_predictions)
```

**Purpose**: Student directly mimics teacher's numerical predictions.
- **Function**: Mean Squared Error between outputs
- **Target**: Teacher model's time series predictions
- **Benefit**: Student learns teacher's prediction patterns and domain knowledge

## 🏗️ Time Series Specific Architecture

### Input Processing
```python
# Time series input shape: [batch_size, sequence_length, features]
batch_x = time_series_data  # Historical values
batch_y = target_values     # Future values to predict
batch_x_mark = time_marks   # Temporal encodings (day, hour, etc.)
```

### Prediction Process
1. **Teacher Forward Pass**: Generate predictions with frozen teacher model
2. **Student Forward Pass**: Generate predictions with trainable student model
3. **Loss Calculation**: Combine all three loss components
4. **Backpropagation**: Update only student model parameters

### Decoder Input Construction
```python
# For autoregressive prediction
dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])  # Future slots
dec_inp = torch.cat([batch_y[:, :context_len, :], dec_inp], dim=1)  # Known + unknown
```

## ⚙️ Configurable Parameters

You can adjust distillation behavior by modifying these parameters:

### Loss Weighting

- **High α (alpha)**: More emphasis on ground truth (safer, more conservative)
- **High β (beta)**: More emphasis on teacher matching (better teacher knowledge transfer)

**Note**: Temperature scaling and KL divergence parameters were removed for simplified time series distillation.
- **Default T (3.0)**: Balanced uncertainty transfer

## 🎯 Time Series Distillation Benefits

### 1. **Pattern Transfer**
- Teacher learns complex temporal patterns from large datasets
- Student inherits these patterns in a compressed form
- Maintains forecasting accuracy with reduced parameters

### 2. **Uncertainty Modeling**
- Teacher's confidence about different time periods gets transferred
- Student learns when to be uncertain (e.g., during volatile periods)
- Better calibrated predictions with confidence intervals

### 3. **Seasonal Knowledge**
- Teacher captures long-term seasonal patterns
- Student inherits seasonal awareness efficiently
- Improved forecasting for cyclical time series

### 4. **Multi-variate Relationships**
- Teacher learns complex inter-variable dependencies
- Student inherits these relationships in compressed form
- Better multivariate time series forecasting

## 📊 Example Configuration

```python
# Simplified distillation parameters for time series prediction
distillation_params = {
    "alpha": 0.6,           # Emphasize ground truth (safety critical)
    "beta": 0.4,            # Teacher guidance  
    "learning_rate": 1e-4,  # Conservative learning
    "epochs": 50            # Sufficient convergence
}
```

## 🚀 Performance Characteristics

| Aspect | Teacher (BERT) | Student (TinyBERT) | Improvement |
|--------|---------------|-------------------|-------------|
| Parameters | 110M | 14M | 87% reduction |
| Inference Speed | 1x | 8x | 8x faster |
| Memory Usage | 1x | 0.3x | 70% reduction |
| Accuracy Retention | 100% | ~95% | 5% trade-off |

## 🔧 Implementation Notes

### Teacher Model Requirements
- Must be **pre-trained** on the same task
- **Frozen parameters** during distillation
- Same input/output dimensions as student

### Student Model Requirements  
- **Smaller architecture** than teacher
- **Trainable parameters** during distillation
- Compatible input/output with teacher

### Hardware Considerations
- Teacher + Student models loaded simultaneously
- Requires ~1.5x memory of largest model
- GPU acceleration recommended for efficiency

## 📈 Optimization Tips

### For Better Accuracy

- Increase `alpha` (ground truth weight)
- More `epochs` for convergence
- Conservative learning rates

### For Better Efficiency  

- Increase `beta` (teacher matching weight)
- Gradient accumulation for large batches
- Early stopping based on validation loss

### For Balanced Performance

- Use balanced `alpha` and `beta` weights (0.5/0.5)
- Monitor both teacher and ground truth loss components
- Validate performance on held-out test sets

The simplified MSE framework is specifically optimized for forecasting
regression. ECG classification intentionally uses probability-distribution
matching because its outputs are categorical logits.