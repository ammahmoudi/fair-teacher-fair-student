# Data Formatting Scripts

This directory contains scripts for complete data processing pipeline including standardization, formatting, and Arrow conversion.

## Overview

The data processing pipeline consists of three main steps:

1. **Standardization** - Convert raw data to standard format (item_id, timestamp, target)
2. **Formatting** - Create window-based sequences (6_6 and 6_9 configurations)
3. **Arrow Conversion** - Convert to Arrow format for efficient training

## Available Scripts

### 1. `complete_data_pipeline.py` - Main Pipeline Script

Complete data processing pipeline with full control over all steps.

#### Features:
- Process all datasets (ohiot1dm, d1namo) and scenarios (raw, missing_periodic, missing_random, noisy, denoised)
- Support for both window configurations (6_6 and 6_9)
- Dry-run mode to preview what will be processed
- Skip specific steps if needed
- Comprehensive logging and error handling

#### Usage Examples:

```bash
# Process everything
python complete_data_pipeline.py --all

# Process specific dataset
python complete_data_pipeline.py --dataset ohiot1dm

# Process specific scenarios
python complete_data_pipeline.py --dataset d1namo --scenarios raw,noisy,denoised

# Dry run to see what would be processed
python complete_data_pipeline.py --dataset ohiot1dm --dry-run

# Skip specific steps
python complete_data_pipeline.py --dataset ohiot1dm --skip-standardize
python complete_data_pipeline.py --dataset ohiot1dm --skip-format --skip-arrow
```

#### Available Options:

- `--all` - Process all datasets and scenarios
- `--dataset {ohiot1dm,d1namo,all}` - Specific dataset to process
- `--scenarios SCENARIOS` - Comma-separated scenarios or "all"
- `--dry-run` - Show what would be processed without running
- `--skip-standardize` - Skip standardization step
- `--skip-format` - Skip formatting step  
- `--skip-arrow` - Skip Arrow conversion step

### 2. `quick_process.py` - Simplified Interface

Quick and easy interface for common processing tasks.

#### Usage Examples:

```bash
# Process ohiot1dm completely
python quick_process.py ohiot1dm

# Process d1namo completely
python quick_process.py d1namo

# Process all datasets
python quick_process.py all

# Process specific scenarios
python quick_process.py ohiot1dm --scenarios raw,missing_periodic

# Dry run
python quick_process.py ohiot1dm --dry-run
```

## Data Processing Steps Explained

#### 1. Standardization
Converts raw CSV files to standardized format (item_id, timestamp, target).
- Script: `core/standardize_data.py`

#### 2. Formatting  
Creates windowed datasets (6_6 and 6_9 configurations).
- Script: `core/format_data.py`

#### 3. Arrow Conversion
Converts to GluonTS Arrow format for efficient training.
- Script: `core/convert_to_arrow.py`

## Directory Structure After Processing

```
data/
├── ohiot1dm/
│   ├── raw/                           # Original raw data
│   ├── raw_standardized/              # Standardized raw data  
│   ├── raw_formatted/
│   │   ├── 6_6/                      # 6_6 window configuration
│   │   └── 6_9/                      # 6_9 window configuration
│   ├── missing_periodic/              # Original missing periodic data
│   ├── missing_periodic_standardized/ # Standardized missing periodic
│   ├── missing_periodic_formatted/
│   │   ├── 6_6/
│   │   └── 6_9/
│   └── ... (other scenarios)
└── d1namo/
    └── ... (similar structure)
```

Each formatted directory contains:
- `{patient_id}-ws-training.csv` - Training sequences
- `{patient_id}-ws-testing.csv` - Testing sequences  
- `{patient_id}-ws-training.arrow` - Training sequences in Arrow format

## Supported Datasets

### OhioT1DM Dataset
- **Scenarios**: raw, missing_periodic, missing_random, noisy, denoised
- **Patients**: 540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596

### D1NAMO Dataset  
- **Scenarios**: raw, missing_periodic, missing_random, noisy, denoised
- **Patients**: Various patient IDs in the dataset

## Window Configurations

### 6_6 Configuration
- **Input length**: 6 time steps
- **Prediction length**: 6 time steps
- **Use case**: Standard short-term prediction

### 6_9 Configuration
- **Input length**: 6 time steps  
- **Prediction length**: 9 time steps
- **Use case**: Extended prediction horizon

## Prerequisites

Make sure you have the required dependencies installed:

```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements (if not already installed)
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- gluonts
- pathlib

## Example Workflow

### Complete Processing for OhioT1DM

```bash
# 1. Quick complete processing
python quick_process.py ohiot1dm

# 2. Or step-by-step with full control
python complete_data_pipeline.py --dataset ohiot1dm --scenarios all

# 3. Check results
ls -la ../../data/ohiot1dm/raw_formatted/6_6/
ls -la ../../data/ohiot1dm/raw_standardized/*.arrow
```

### Process Only Specific Scenarios

```bash
# Process only noisy and denoised data
python complete_data_pipeline.py --dataset ohiot1dm --scenarios noisy,denoised

# Dry run first to check
python complete_data_pipeline.py --dataset ohiot1dm --scenarios noisy,denoised --dry-run
```

### Reprocess Only Formatting

```bash
# Skip standardization and arrow conversion, only reformat
python complete_data_pipeline.py --dataset ohiot1dm --skip-standardize --skip-arrow
```

## Troubleshooting

### Common Issues

1. **Script not found errors**
   - Make sure you're running from the correct directory
   - Check that `data_processing/` scripts exist

2. **Import errors**
   - Ensure virtual environment is activated
   - Check that `utils/path_utils.py` exists

3. **Data not found**
   - Verify that source data exists in `data/{dataset}/{scenario}/`
   - Check that previous processing steps completed successfully

4. **Permission errors**
   - Make sure you have write permissions to the data directories

### Logging

All scripts provide detailed logging. Check the console output for:
- ✅ Success messages
- ❌ Error messages with details
- 📋 Processing summaries
- 🚀 Progress indicators

### Dry Run Mode

Always use `--dry-run` first to see what will be processed:

```bash
python complete_data_pipeline.py --dataset ohiot1dm --dry-run
```

This shows exactly what files will be processed without making changes.

## Integration with Training Scripts

After processing, the data can be used with:

### Time-LLM Training
- Use formatted CSV files from `*_formatted/{window_config}/` directories
- Both 6_6 and 6_9 configurations supported

### Chronos Training  
- Use Arrow files from `*_standardized/` directories
- Arrow format provides efficient loading for large datasets

### Configuration Generators
- Use processed data paths in config generators
- Both relative and absolute paths supported via path utilities

## Performance Notes

- **Processing time**: Depends on dataset size, typically 5-30 minutes per dataset
- **Storage**: Processed data requires ~2-3x original data size
- **Memory**: Processing is done in chunks to handle large datasets
- **Parallel processing**: Scripts process files sequentially for reliability

## Next Steps

After processing data:

1. **Verify processed files**: Check that all expected files were created
2. **Update training configs**: Use processed data paths in training configurations  
3. **Run training**: Use the formatted data for model training
4. **Monitor results**: Check training logs and metrics

For more details on using the processed data, see:
- `docs/README_chronos_commands.md` - Chronos training setup
- `docs/README_timellmbert_commands.md` - Time-LLM training setup