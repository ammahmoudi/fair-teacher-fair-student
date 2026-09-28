# Scripts Directory

Organized collection of scripts for running experiments, processing data, and analyzing results.

## Directory Structure

```
scripts/
├── analysis/                      # Result analysis and reporting
│   └── regenerate_full_table.py
│
├── pipelines/                     # Pipeline orchestration scripts
│   ├── distill_pipeline.sh
│   ├── run_all_patients_distillation.sh
│   ├── run_cross_scenario_inference.sh
│   ├── run_time_llm_bert_cross_scenario.sh
│   └── test_distillation_pairs.sh
│
├── raw_data_replacement/          # True values replacement utilities
│   ├── README.md
│   ├── replace_true_values_with_raw.py
│   ├── verify_exact_replacement.py
│   └── run_replace_true_values.sh
│
├── chronos/                       # Chronos model scripts
│   ├── README.md
│   ├── USAGE_GUIDE.md
│   ├── config_generator.py                # Main config generator
│   ├── run_experiments.py                 # Main experiment runner
│   └── helpers/                           # Batch/convenience scripts
│       ├── batch_generate_all_configs.py
│       ├── batch_generate_trained_inference.py
│       └── examples.sh
│
├── time_llm/                      # Time-LLM model scripts
│   ├── README.md
│   ├── TIME_LLM_USAGE_GUIDE.md
│   ├── config_generator.py                # Main config generator
│   ├── run_experiments.py                 # Main experiment runner
│   └── helpers/                           # Batch/convenience scripts
│       ├── batch_generate_all_configs.py
│       ├── examples_quick_commands.sh
│       └── examples_common_configs.sh
│
├── data_formatting/               # Data preprocessing and formatting
│   ├── README.md
│   ├── core/                      # Core processing scripts
│   │   ├── standardize_data.py    # Convert raw data to standard format
│   │   ├── format_data.py         # Create windowed datasets (6_6, 6_9)
│   │   └── convert_to_arrow.py    # Convert to GluonTS Arrow format
│   └── runners/                   # Pipeline orchestrators
│       ├── complete_data_pipeline.py  # Full pipeline with control
│       └── quick_process.py       # Simple pipeline interface
│
├── missing_and_noise/             # Data corruption utilities
│   ├── apply_noise.py
│   └── apply_missings.py
│
├── d1namo_scripts/                # D1NAMO dataset specific scripts
│   ├── config_d1namo_generator_chronos_train.py
│   ├── config_d1namo_generator_time_llm.py
│   └── extract_metrics.py
│
├── utilities/                     # Shared utility functions
│   ├── extract_metrics.py
│   ├── extract_metrics_corrected.py
│   ├── fix_results.py
│   ├── smooth_data.py             # Smooth prediction results
│   ├── smooth_data_window.py      # Smooth windowed predictions
│   ├── convert_window_to_single_data.py  # Convert windowed to single-step
│   ├── update_logs_with_new_metrics.py
│   └── seeds.py
│
├── tests/                         # Test scripts
│   ├── system_test.py
│   ├── check_models.py
│   └── accelerator_test.py
│
├── import_utils.py                # Import path configuration
└── run_main.sh                    # Main experiment runner
```

## Quick Reference

### Analysis Scripts

```bash
# Regenerate comprehensive results table
python scripts/analysis/regenerate_full_table.py
```

Note: Distillation-specific analysis scripts have been moved to `distillation/scripts/`:
- `analyze_distillation_results.py` - Analyze and visualize distillation results
- `distillation_comparison.py` - Compare different teacher-student pairs

### Pipeline Scripts

```bash
# Run full distillation pipeline
bash scripts/pipelines/distill_pipeline.sh --teacher bert --student tinybert --dataset 570

# Run all patients distillation
bash scripts/pipelines/run_all_patients_distillation.sh

# Run cross-scenario inference
bash scripts/pipelines/run_cross_scenario_inference.sh
```

### Model-Specific Scripts

```bash
# Chronos experiments
python scripts/chronos/run_experiments.py

# Time-LLM experiments
python scripts/time_llm/run_experiments.py

# Generate configs
python scripts/chronos/config_generator.py --mode train
python scripts/time_llm/config_generator.py --mode train
```

### Data Processing

```bash
# Quick data processing (all steps)
python scripts/data_formatting/runners/quick_process.py all

# Full pipeline with control
python scripts/data_formatting/runners/complete_data_pipeline.py --dataset ohiot1dm --scenarios all

# Individual core scripts
python scripts/data_formatting/core/standardize_data.py --dataset ohiot1dm --scenario all
python scripts/data_formatting/core/format_data.py --dataset ohiot1dm --windows all
python scripts/data_formatting/core/convert_to_arrow.py --dataset ohiot1dm --scenario all
```

### Raw Data Replacement

```bash
# Replace true values with raw data
bash scripts/raw_data_replacement/run_replace_true_values.sh

# Verify replacement
python scripts/raw_data_replacement/verify_exact_replacement.py
```

## Notes

- All subdirectories have `__init__.py` for proper Python package structure
- Use `import_utils.py` at the start of scripts that need utilities imports
- Each model folder (chronos, time_llm) has its own USAGE_GUIDE.md
- Raw data replacement scripts have dedicated README.md with full documentation
