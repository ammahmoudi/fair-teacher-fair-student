# Advanced Fairness Metrics for Blood Glucose Prediction

## Overview

This module implements three advanced fairness metrics specifically designed for evaluating fairness in blood glucose prediction models. These metrics go beyond traditional performance comparisons to assess whether critical health predictions are equitable across demographic groups.

## The Three Metrics

### 1. Demographic Parity Gap (DP Gap)

**What it measures:** Whether critical alerts are distributed equally across demographic groups, independent of actual outcomes.

**Mathematical Definition:**
```
DP Gap = |P(Ŷ=1 | A=a) - P(Ŷ=1 | A=b)|
```

Where:
- `Ŷ=1`: Predicted positive (critical event, e.g., hypoglycemia)
- `A`: Protected attribute (e.g., gender)
- `a, b`: Different groups (e.g., male, female)

**Why it matters:** Ensures that the model doesn't systematically over-alert or under-alert certain demographic groups, which could lead to alert fatigue or missed warnings.

**Interpretation:**
- `< 0.05`: ✅ **EXCELLENT** - Alerts distributed fairly
- `0.05 - 0.10`: ✓ **GOOD** - Minor disparity
- `0.10 - 0.20`: ⚠️ **CONCERNING** - Moderate disparity
- `> 0.20`: ❌ **CRITICAL** - Significant disparity

**Example:** If 15% of male patients get hypoglycemia warnings but only 8% of female patients do, the DP Gap is 0.07 (concerning).

---

### 2. Equal Opportunity Gap (EO Gap)

**What it measures:** Whether the model is equally effective at detecting **actual** high-risk events across groups.

**Mathematical Definition:**
```
EO Gap = |P(Ŷ=1 | Y=1, A=a) - P(Ŷ=1 | Y=1, A=b)|
       = |TPR_a - TPR_b|
```

Where:
- `TPR`: True Positive Rate (Sensitivity/Recall)
- `Y=1`: Actual positive (true critical event)
- `Ŷ=1`: Predicted positive

**Why it matters:** This is **CRITICAL** for patient safety. If the model misses hypoglycemic events more often for one group, those patients face higher health risks.

**Interpretation:**
- `< 0.05`: ✅ **EXCELLENT** - Equal detection rates
- `0.05 - 0.10`: ✓ **GOOD** - Minor difference
- `0.10 - 0.20`: ⚠️ **CONCERNING** - Some groups may miss alerts
- `> 0.20`: ❌ **CRITICAL SAFETY ISSUE** - Significant detection disparity

**Example:** If the model correctly detects 85% of hypoglycemic events in males but only 50% in females, the EO Gap is 0.35 - a critical safety issue.

---

### 3. Fairness Violation Objective (FVO)

**What it measures:** Maximum disparity in overall model accuracy between any two demographic subgroups.

**Mathematical Definition:**
```
FVO = max|Acc_i - Acc_j| for all pairs (i,j) of groups
```

Where:
- `Acc_i`: Accuracy for group i
- Can be calculated for classification (binary events) or regression (continuous predictions)

**Why it matters:** Ensures general model reliability is equitable. Even if detection rates are similar, one group might receive less accurate predictions overall.

**Interpretation:**
- `< 0.02`: ✅ **EXCELLENT** - Equal performance
- `0.02 - 0.05`: ✓ **GOOD** - Minor variation
- `0.05 - 0.10`: ⚠️ **CONCERNING** - Notable disparity
- `> 0.10`: ❌ **POOR** - Significant reliability concerns

**Example:** If the model has 95% accuracy for one age group but only 82% for another, the FVO is 0.13 (poor).

---

## Usage

### Basic Usage

```python
from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics
import numpy as np

# Initialize with clinical thresholds
afm = AdvancedFairnessMetrics(
    hypoglycemia_threshold=70.0,  # mg/dL
    hyperglycemia_threshold=180.0  # mg/dL
)

# Your data
y_true = np.array([...])  # True glucose values
y_pred = np.array([...])  # Predicted glucose values
gender = np.array(['male', 'female', ...])  # Demographic labels

# Calculate all metrics at once
results = afm.calculate_all_advanced_metrics(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    group_attribute='Gender',
    risk_type='hypoglycemia'
)

# Print comprehensive report
afm.print_comprehensive_report(results)
```

### Individual Metrics

```python
# 1. Demographic Parity Gap
dp_result = afm.demographic_parity_gap(
    y_pred=y_pred,
    group_labels=gender,
    risk_type='hypoglycemia'
)
print(f"DP Gap: {dp_result['dp_gap']:.4f}")
print(f"Interpretation: {dp_result['interpretation']}")

# 2. Equal Opportunity Gap
eo_result = afm.equal_opportunity_gap(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    risk_type='hypoglycemia'
)
print(f"EO Gap: {eo_result['eo_gap']:.4f}")
print(f"Male TPR: {eo_result['group_tpr']['male']:.4f}")
print(f"Female TPR: {eo_result['group_tpr']['female']:.4f}")

# 3. Fairness Violation Objective
fvo_result = afm.fairness_violation_objective(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    metric_type='classification'  # or 'regression'
)
print(f"FVO: {fvo_result['fvo']:.4f}")
```

### Running Examples

```bash
# Test with synthetic data
cd /path/to/fair-teacher-fair-student
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/metrics/advanced_fairness_metrics.py

# Run comprehensive example with visualizations
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/advanced_metrics_example.py --example
```

## Integration with Existing Analyzers

You can integrate these metrics into your existing fairness analyzers:

```python
from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics

class YourAnalyzer:
    def __init__(self):
        self.afm = AdvancedFairnessMetrics()
    
    def analyze(self, y_true, y_pred, demographics):
        # Your existing analysis...
        
        # Add advanced metrics
        for demo_name, demo_labels in demographics.items():
            advanced_results = self.afm.calculate_all_advanced_metrics(
                y_true=y_true,
                y_pred=y_pred,
                group_labels=demo_labels,
                group_attribute=demo_name,
                risk_type='hypoglycemia'
            )
            
            # Store or visualize results...
```

## Comparison with Existing Metrics

| Metric | What It Measures | When To Use |
|--------|------------------|-------------|
| **Fairness Ratio** (existing) | Ratio of worst to best RMSE | Overall performance disparity |
| **DP Gap** (new) | Alert distribution equality | Ensuring equal warning rates |
| **EO Gap** (new) | Detection rate equality | Critical for patient safety |
| **FVO** (new) | Maximum accuracy disparity | General reliability across groups |

## Clinical Significance

### Why EO Gap is Most Critical

In diabetes care, **missing a hypoglycemic event is far more dangerous** than a false alarm. The Equal Opportunity Gap directly measures whether some demographic groups are more likely to miss critical warnings.

**Example Scenario:**
- Model has EO Gap of 0.40 between genders
- Male patients: 85% of hypoglycemic events detected
- Female patients: 45% of hypoglycemic events detected
- **Result:** Female patients face 2x higher risk of undetected hypoglycemia

### Risk Types

Both `hypoglycemia` and `hyperglycemia` can be analyzed:

```python
# Analyze hypoglycemia risk (< 70 mg/dL)
hypo_results = afm.calculate_all_advanced_metrics(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    group_attribute='Gender',
    risk_type='hypoglycemia'
)

# Analyze hyperglycemia risk (> 180 mg/dL)
hyper_results = afm.calculate_all_advanced_metrics(
    y_true=y_true,
    y_pred=y_pred,
    group_labels=gender,
    group_attribute='Gender',
    risk_type='hyperglycemia'
)
```

## Output Format

The comprehensive results dictionary contains:

```python
{
    'group_attribute': 'Gender',
    'risk_type': 'hypoglycemia',
    'thresholds': {
        'hypoglycemia': 70.0,
        'hyperglycemia': 180.0
    },
    'sample_size': 1000,
    'unique_groups': ['male', 'female'],
    'metrics': {
        'demographic_parity_gap': {
            'dp_gap': 0.0705,
            'group_positive_rates': {'male': 0.15, 'female': 0.08},
            'interpretation': '...',
            'lower_is_better': True
        },
        'equal_opportunity_gap': {
            'eo_gap': 0.3965,
            'group_tpr': {'male': 0.85, 'female': 0.45},
            'group_confusion_matrices': {...},
            'interpretation': '...',
            'lower_is_better': True
        },
        'fairness_violation_objective': {
            'classification_based': {...},
            'regression_based': {...}
        }
    },
    'overall_assessment': {
        'status': '⚠️ MODERATE CONCERNS',
        'summary': '...',
        'issues': [...],
        'metrics_summary': {...}
    }
}
```

## Optimization Targets

When training or fine-tuning models for fairness:

| Metric | Target | Optimization | Priority |
|--------|--------|--------------|----------|
| **DP Gap** | Output Rate | **Minimize** | Medium |
| **EO Gap** | Detection Rate | **Minimize** | **HIGH** (Safety Critical) |
| **FVO** | Overall Accuracy | **Minimize** | Medium-High |

## Files

- `fairness/metrics/advanced_fairness_metrics.py` - Core implementation
- `fairness/metrics/__init__.py` - Module exports
- `fairness/advanced_metrics_example.py` - Usage examples and integration
- `fairness/metrics/ADVANCED_METRICS_README.md` - This file

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Dwork, C., et al. (2012). Fairness through awareness.
- Feldman, M., et al. (2015). Certifying and removing disparate impact.

## Testing

Run the test suite:

```bash
cd /path/to/fair-teacher-fair-student
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 -m pytest fairness/tests/test_advanced_metrics.py -v
```

## Support

For questions or issues:
1. Check the examples in `advanced_metrics_example.py`
2. Review the test cases
3. See the main fairness README at `fairness/README.md`

---

**Status:** ✅ Implemented and tested  
**Last Updated:** January 3, 2026
