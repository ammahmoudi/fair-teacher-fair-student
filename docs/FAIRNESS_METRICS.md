# Fairness Metrics for Fair Teacher–Fair Student

This document covers the fairness analysis framework for evaluating demographic equity in blood glucose prediction models.

## Overview

The fairness framework provides two layers of analysis:

1. **Standard Fairness Metrics** (`fairness/metrics/fairness_metrics.py`): RMSE-based fairness ratios across demographic groups.
2. **Advanced Fairness Metrics** (`fairness/metrics/advanced_fairness_metrics.py`): Clinical-grade metrics (DP Gap, EO Gap, FVO) focused on critical health outcome equity.

---

## Advanced Fairness Metrics

Three metrics specifically designed for blood glucose prediction safety and equity.

### 1. Demographic Parity Gap (DP Gap)

Measures whether critical alerts are distributed equally across demographic groups, independent of actual outcomes.

$$\text{DP Gap} = |P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b)|$$

| Range | Rating |
|-------|--------|
| < 0.05 | ✅ EXCELLENT |
| 0.05 – 0.10 | ✓ GOOD |
| 0.10 – 0.20 | ⚠️ CONCERNING |
| > 0.20 | ❌ CRITICAL |

### 2. Equal Opportunity Gap (EO Gap)

Ensures the model detects **actual** high-risk events equally well for all groups. Critical for patient safety.

$$\text{EO Gap} = |P(\hat{Y}=1 \mid Y=1, A=a) - P(\hat{Y}=1 \mid Y=1, A=b)| = |\text{TPR}_a - \text{TPR}_b|$$

| Range | Rating |
|-------|--------|
| < 0.05 | ✅ EXCELLENT |
| 0.05 – 0.10 | ✓ GOOD |
| 0.10 – 0.20 | ⚠️ CONCERNING |
| > 0.20 | ❌ CRITICAL SAFETY ISSUE |

### 3. Fairness Violation Objective (FVO)

Maximum accuracy disparity between any two demographic subgroups.

$$\text{FVO} = \max_{i,j} |\text{Acc}_i - \text{Acc}_j|$$

| Range | Rating |
|-------|--------|
| < 0.02 | ✅ EXCELLENT |
| 0.02 – 0.05 | ✓ GOOD |
| 0.05 – 0.10 | ⚠️ CONCERNING |
| > 0.10 | ❌ POOR |

---

## Calculation Modes

All three metrics support three calculation modes:

| Mode | Description | Use When |
|------|-------------|----------|
| `simple` (default) | Direct calculation on flat prediction arrays | Standard evaluation |
| `timeline_reconstruction` | Average overlapping windows, then binarize | Continuous monitoring analysis |
| `window_majority` | Binarize each window, majority vote at each timestamp | Noise-robust evaluation |

---

## Quick Start

```bash
# Run quick demo with synthetic data
python3 fairness/quickstart_advanced_metrics.py

# Apply to experiment results
python3 fairness/apply_advanced_metrics_to_results.py --experiment-dir <path>

# Full comprehensive example
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/advanced_metrics_example.py --example

# Compare all three modes side-by-side
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/compare_calc_modes.py
```

---

## Usage

### Calculate All Metrics at Once

```python
from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics
import numpy as np

afm = AdvancedFairnessMetrics(
    hypoglycemia_threshold=70.0,   # mg/dL
    hyperglycemia_threshold=180.0  # mg/dL
)

results = afm.calculate_all_advanced_metrics(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    group_attribute='Gender',
    risk_type='hypoglycemia',
    calc_mode='simple'          # or 'timeline_reconstruction', 'window_majority'
)

afm.print_comprehensive_report(results)
```

### Individual Metrics

```python
# Demographic Parity Gap (no y_true needed)
dp = afm.demographic_parity_gap(y_pred, group_labels=gender, risk_type='hypoglycemia')
print(f"DP Gap: {dp['dp_gap']:.4f}  — {dp['interpretation']}")

# Equal Opportunity Gap
eo = afm.equal_opportunity_gap(y_true, y_pred, group_labels=gender)
print(f"EO Gap: {eo['eo_gap']:.4f}  — {eo['interpretation']}")

# Fairness Violation Objective
fvo = afm.fairness_violation_objective(y_true, y_pred, group_labels=gender,
                                       metric_type='classification')
print(f"FVO:    {fvo['fvo']:.4f}  — {fvo['interpretation']}")
```

### Windowed / Overlapping Window Mode

```python
# Organise predictions as dicts of window lists per group
y_pred_by_group = {'male': [window1, window2, ...], 'female': [...]}
y_true_by_group = {'male': [...], 'female': [...]}
window_starts   = {'male': [0, 6, 12, ...], 'female': [0, 6, ...]}
total_lengths   = {'male': 600, 'female': 400}

results = afm.calculate_all_advanced_metrics(
    y_true=y_true_by_group,
    y_pred=y_pred_by_group,
    group_labels={g: np.array([g]) for g in ['male', 'female']},
    group_attribute='Gender',
    risk_type='hypoglycemia',
    calc_mode='timeline_reconstruction',
    window_start_indices=window_starts,
    total_lengths=total_lengths
)
```

---

## Integration with Existing Analyzers

```python
from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics

class YourAnalyzer:
    def __init__(self):
        self.afm = AdvancedFairnessMetrics()

    def analyze(self, y_true, y_pred, demographics):
        for attr, labels in demographics.items():
            advanced = self.afm.calculate_all_advanced_metrics(
                y_true=y_true,
                y_pred=y_pred,
                group_labels=labels,
                group_attribute=attr,
                risk_type='hypoglycemia'
            )
            # store / visualize advanced results
```

---

## Files

| Path | Description |
|------|-------------|
| `fairness/metrics/advanced_fairness_metrics.py` | Core implementation (DP Gap, EO Gap, FVO) |
| `fairness/metrics/ADVANCED_METRICS_README.md` | Detailed in-module documentation |
| `fairness/advanced_metrics_example.py` | Comprehensive usage examples with visualizations |
| `fairness/quickstart_advanced_metrics.py` | Minimal demo with synthetic data |
| `fairness/apply_advanced_metrics_to_results.py` | Apply metrics to existing experiment directories |
| `fairness/compare_calc_modes.py` | Side-by-side comparison of the three calculation modes |
| `fairness/investigate_all_scenarios.py` | Integrated cross-scenario analysis (updated) |
| `notebooks/advanced_metrics_analysis.ipynb` | Interactive notebook |

---

## Comparison with Standard Fairness Metrics

| Metric | What it measures | Needs `y_true`? |
|--------|-----------------|-----------------|
| Fairness Ratio (existing) | Worst/best RMSE ratio | ✅ |
| **DP Gap** | Alert distribution equality | ❌ (predictions only) |
| **EO Gap** | Detection rate equality (TPR) | ✅ |
| **FVO** | Maximum accuracy disparity | ✅ |

---

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
- Dwork, C. et al. (2012). Fairness through awareness. *ITCS*.
- Adapted for continuous blood glucose prediction with clinical safety thresholds.
