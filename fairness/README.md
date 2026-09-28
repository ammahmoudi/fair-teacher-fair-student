# Fairness Analysis Framework

Comprehensive fairness analysis for distillation and inference scenarios in diabetes prediction models.

## 🆕 NEW: Advanced Fairness Metrics

Three new metrics specifically designed for blood glucose prediction fairness:

1. **Demographic Parity Gap (DP Gap)**: Ensures alerts are distributed equally across groups
2. **Equal Opportunity Gap (EO Gap)**: **CRITICAL for patient safety** - detects unequal risk detection rates  
3. **Fairness Violation Objective (FVO)**: Measures maximum accuracy disparity between groups

### Quick Start with Advanced Metrics

```bash
# Quick demo with synthetic data
python3 fairness/quickstart_advanced_metrics.py

# Apply to your experiment results (if you have prediction files)
python3 fairness/apply_advanced_metrics_to_results.py --experiment-dir <path>

# See comprehensive example
PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/advanced_metrics_example.py --example
```

📖 **Full Documentation**: [Advanced Metrics README](metrics/ADVANCED_METRICS_README.md)

## Overview

This framework analyzes fairness across demographic dimensions in three contexts:

1. **Distillation Analysis**: Teacher → Student → Distilled models during knowledge distillation
2. **Inference Scenarios**: Different training conditions (inference-only, standard, noisy, denoised)
3. **Advanced Clinical Metrics**: DP Gap, EO Gap, FVO for critical health outcome fairness

## Quick Start

**Note**: Run all commands from the project root directory (`/path/to/fair-teacher-fair-student/`)

### Distillation Analysis

```bash
# Per-patient experiments (original BERT distillation)
python fairness/run_distillation_analyzers.py --experiment-type per_patient

# All-patients experiments
python fairness/run_distillation_analyzers.py --experiment-type all_patients

# MiniLM distillation experiments
python fairness/run_minilm_analyzers.py
```

### Inference Scenarios Analysis

```bash
# All features
python fairness/run_inference_analyzers.py --feature all

# Specific feature
python fairness/run_inference_analyzers.py --feature gender
```

### Comprehensive Cross-Scenario Analysis

```bash
# Compare ALL scenarios (inference + distillation) - includes advanced metrics summary
python fairness/investigate_all_scenarios.py

# Detailed training impact analysis
python fairness/analyze_training_impact.py

# Investigate specific fairness issues
python fairness/investigate_fairness_issues.py
```

## Analyzers

### Distillation Analyzers (6)

Analyze fairness during knowledge distillation across 5 demographic features plus comprehensive analysis.

- **Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)
- **Phases Analyzed**: Teacher → Student Baseline → Distilled
- **Experiment Folders**: 
  - `distillation_experiments` (BERT-based distillation)
  - `minilm_distil_experiments` (MiniLM-based distillation)
- **Results Location**:
  - fairness/analysis_results/distillation_per_patient/
  - fairness/analysis_results/distillation_all_patients/
  - fairness/analysis_results/minilm_distillation_per_patient/

### Inference Scenario Analyzers (6)

Analyze fairness across different training/inference conditions.

- **Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)
- **Scenarios Analyzed**:
  1. Inference Only (no training)
  2. Trained on Standard Data
  3. Trained on Noisy Data
  4. Trained on Denoised Data
- **Results Location**: fairness/analysis_results/inference_scenarios/

## Output Files

Each analyzer generates:

- **JSON Report**: Structured fairness metrics and statistics
- **PNG Visualization**: Multi-panel charts with RMSE comparisons and fairness analysis
- **CSV Table** (Legendary only): Per-group performance summary across all scenarios

### Legendary Analyzer Visualization

Single comprehensive PNG file containing:

- **Top Panel**: Fairness heatmap across all features and scenarios/phases
- **Middle Panels**: RMSE comparison charts for each demographic feature
- **Bottom Panel**: Summary table with per-group RMSE values and best/worst scenarios

## Demographics

- **Gender**: Male, Female
- **Age Group**: 20-40, 40-60, 60-80 years
- **Pump Model**: 630G, 530G
- **Sensor Band**: Empatica, Basis
- **Cohort**: 2018, 2020

## Fairness Metrics

### Fairness Ratio

Fairness Ratio = max(group_rmse) / min(group_rmse)

Lower ratios indicate more fair performance across groups.

### Fairness Levels

- **1.00-1.10**: EXCELLENT
- **1.10-1.25**: GOOD
- **1.25-1.50**: ACCEPTABLE
- **1.50+**: POOR (requires attention)

## Project Structure

```
fairness/
├── README.md
├── run_distillation_analyzers.py
├── run_inference_analyzers.py
├── analyzers/
│   ├── base_analyzer.py
│   ├── base_inference_analyzer.py
│   ├── *_fairness_analyzer.py        (5 distillation analyzers)
│   ├── inference_*_analyzer.py       (5 inference analyzers)
│   ├── legendary_distillation_analyzer.py
│   └── inference_legendary_analyzer.py
└── analysis_results/
    ├── distillation_per_patient/
    ├── distillation_all_patients/
    └── inference_scenarios/
```

## Requirements

Install from main project requirements:

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, matplotlib, seaborn

## Advanced Usage

### Run Individual Analyzer

```bash
# Distillation
python fairness/analyzers/gender_fairness_analyzer.py --experiment-type per_patient

# Inference
python fairness/analyzers/inference_gender_analyzer.py
```

### Programmatic Access

```python
from fairness.analyzers.legendary_distillation_analyzer import LegendaryDistillationAnalyzer

analyzer = LegendaryDistillationAnalyzer(experiment_type='per_patient')
analyzer.analyze()
```

## Troubleshooting

- **No data found**: Ensure experiment results exist in distillation_experiments/ directory
- **Import errors**: Run commands from project root and verify all dependencies installed
- **Visualization issues**: Adjust figure size parameters in analyzer code if needed

## Advanced Metrics Scripts

### New Analysis Tools

1. **`quickstart_advanced_metrics.py`** - Simplest way to test advanced metrics
   ```bash
   python3 fairness/quickstart_advanced_metrics.py
   ```

2. **`advanced_metrics_example.py`** - Comprehensive examples with visualizations
   ```bash
   PYTHONPATH=/path/to/fair-teacher-fair-student:$PYTHONPATH python3 fairness/advanced_metrics_example.py --example
   ```

3. **`apply_advanced_metrics_to_results.py`** - Apply to your experiment results
   ```bash
   python3 fairness/apply_advanced_metrics_to_results.py --experiment-dir <path>
   ```

4. **`investigate_all_scenarios.py`** - Now includes advanced metrics summary!
   ```bash
   python3 fairness/investigate_all_scenarios.py
   ```

### Output Locations

- Standard fairness analysis: `fairness/analysis_results/`
- Advanced metrics results: `fairness/analysis_results/advanced_metrics/`
- Comprehensive reports: `fairness/analysis_results/comprehensive_fairness_report_*.png`

## Key Metrics Comparison

| Metric | Type | What It Measures | When To Use |
|--------|------|------------------|-------------|
| **Fairness Ratio** | Ratio | max(RMSE) / min(RMSE) | Overall performance disparity |
| **DP Gap** ✨ | Difference | Alert distribution equality | Ensuring equal warning rates |
| **EO Gap** ✨ | Difference | Detection rate equality | **Critical for patient safety** |
| **FVO** ✨ | Maximum | Accuracy disparity | General reliability across groups |

✨ = New advanced metrics

---

All analyzers tested and operational (12 original + 3 advanced metrics tools)
