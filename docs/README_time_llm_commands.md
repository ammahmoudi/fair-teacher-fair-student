# Time-LLM Training & Inference Command Guide

This guide provides all commands needed to generate Time-LLM training and inference configs, and to run experiments for both OhioT1DM and D1NAMO datasets. It covers all scenarios, cross-scenario inference, and window configurations.

## ✅ Current Status
- **Cross-Scenario Inference**: Fully supported with `--train_data_scenario` parameter
- **Multiple Datasets**: ohiot1dm and d1namo both supported  
- **All Data Scenarios**: standardized, noisy, denoised, missing_periodic, missing_random
- **Window Configurations**: Automatically generates both 6_6 and 6_9 configurations
- **Dynamic Paths**: All paths use relative references, works from any installation location
- **Comprehensive Parameters**: All required parameters included automatically
- **Prompt Files**: Available in all necessary data folders

---

## 🤖 Supported Models

Time-LLM supports a comprehensive range of language models from large teacher models to small efficient student models:

### Large Teacher Models
- **BERT** (110M params): Best general performance, most tested
- **GPT2** (117M params): Decoder-only architecture 
- **LLAMA** (6.7B params): High-capacity model (requires significant GPU memory)
- **DistilBERT** (66M params): Good balance of size and performance

### Small Efficient Models  
- **TinyBERT** (14M params): Purpose-built for knowledge distillation
- **BERT-tiny** (4.4M params): Ultra-small, fastest inference
- **MiniLM** (33M params): Good performance/size trade-off
- **MobileBERT** (25M params): Mobile-optimized architecture
- **ALBERT** (12-18M params): Compact with shared parameters
- **OPT-125M** (125M params): Meta's efficient decoder model

### Model Selection Guidelines

**For Training/Research**:
```bash
--llm_models BERT,DistilBERT,TinyBERT  # Balanced performance
```

**For Efficiency Focus**:
```bash
--llm_models BERT-tiny,TinyBERT,MiniLM  # Fast inference
```

**For High Performance**:
```bash
--llm_models BERT,GPT2,DistilBERT  # Best accuracy
```

**For Knowledge Distillation**:
```bash
# Use in distillation pipeline - see docs/DISTILLATION_README.md
bash distill_pipeline.sh --teacher BERT --student BERT-tiny --patients 570,584
```

---

## Parameter Reference

Time-LLM config generator supports the following parameters:

- `--mode`: Operation mode (train, inference, train_inference)
- `--dataset`: Dataset type (ohiot1dm, d1namo)
- `--data_scenario`: Data type for testing/inference (standardized, noisy, denoised, missing_periodic, missing_random)
- `--train_data_scenario`: Data type used for training (for cross-scenario evaluation)
- `--patients`: Comma-separated patient IDs (e.g., "570,584")
- `--llm_models`: Comma-separated LLM models (BERT, GPT2, LLAMA, DistilBERT, TinyBERT, BERT-tiny, MiniLM, MobileBERT, ALBERT, OPT-125M)
- `--seeds`: Comma-separated random seeds (e.g., "831363" or "831363,809906,427368")
- `--epochs`: Number of training epochs (default: 10 for train modes, 0 for inference)
- `--output_dir`: Custom output directory (optional)

**Note**: Unlike Chronos, Time-LLM automatically generates both 6_6 and 6_9 window configurations without needing a `--window_config` parameter.

---

## Quick Start Example

```bash
# 1. Generate training configs for standardized data
python scripts/time_llm/config_generator.py --mode train \
    --patients 570,584 --data_scenario standardized --llm_models BERT \
    --seeds 831363 --epochs 10

# 2. Run training experiments
python scripts/time_llm/run_experiments.py --modes train --datasets ohiot1dm

# 3. Generate cross-scenario inference configs (train standardized → test noisy)
python scripts/time_llm/config_generator.py --mode train_inference \
    --data_scenario noisy --train_data_scenario standardized \
    --patients 570,584 --llm_models BERT --seeds 831363 --epochs 10

# 4. Run inference experiments
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm
```

---

## 1. Generate Training Configs

### Train on Standardized Data
```bash
python scripts/time_llm/config_generator.py --mode train \
    --dataset ohiot1dm --data_scenario standardized \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT,DistilBERT,TinyBERT --seeds 831363 --epochs 10
```

### Train on Other Scenarios
```bash
# Missing Periodic
python scripts/time_llm/config_generator.py --mode train \
    --dataset ohiot1dm --data_scenario missing_periodic \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT,TinyBERT --seeds 831363 --epochs 10

# Missing Random  
python scripts/time_llm/config_generator.py --mode train \
    --dataset ohiot1dm --data_scenario missing_random \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT,TinyBERT --seeds 831363 --epochs 10

# Noisy
python scripts/time_llm/config_generator.py --mode train \
    --dataset ohiot1dm --data_scenario noisy \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT,DistilBERT --seeds 831363 --epochs 10

# Denoised
python scripts/time_llm/config_generator.py --mode train \
    --dataset ohiot1dm --data_scenario denoised \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT,MiniLM --seeds 831363 --epochs 10
```

---

## 2. Run Training Experiments
```bash
python scripts/time_llm/run_experiments.py --modes train --datasets ohiot1dm
```

---

## 3. Cross-Scenario Inference (Train on One Scenario, Test on Another)

### Key Parameters:
- `--train_data_scenario`: The scenario the model was trained on
- `--data_scenario`: The scenario to test on  
- `--mode`: Use `train_inference` for combined training and inference

### ✅ Working Examples:

#### Train on Standardized, Test on Noisy
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --dataset ohiot1dm --data_scenario noisy --train_data_scenario standardized \
    --patients 570,584 --llm_models GPT2 --seeds 831363 --epochs 10
```

#### Train on Standardized, Test on Missing Periodic
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --dataset ohiot1dm --data_scenario missing_periodic --train_data_scenario standardized \
    --patients 570,584 --llm_models GPT2 --seeds 831363 --epochs 10
```

#### Train on Standardized, Test on Missing Random
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --dataset ohiot1dm --data_scenario missing_random --train_data_scenario standardized \
    --patients 570,584 --llm_models GPT2 --seeds 831363 --epochs 10
```

---

## 4. Generate Inference-Only Configs

```bash
# Standardized data inference
python scripts/time_llm/config_generator.py --mode inference \
    --dataset ohiot1dm --data_scenario standardized \
    --patients 570,584 --llm_models GPT2 --seeds 831363 --epochs 0

# Noisy data inference  
python scripts/time_llm/config_generator.py --mode inference \
    --dataset ohiot1dm --data_scenario noisy \
    --patients 570,584 --llm_models GPT2 --seeds 831363 --epochs 0
```

---

## 5. Run All Experiments

```bash
# Run specific experiment types (automatically extracts CSV results)
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm

# Run with specific models
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models GPT2

# Parallel execution (for faster processing)
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --parallel --max_workers 2

# Dry run (see what would be executed)
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --dry_run

# Disable automatic CSV extraction
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --no_extract_metrics
```

**Note**: All experiments automatically extract metrics to CSV files after each completion. Results are saved to files like `time_llm_training_inference_ohiot1dm_results.csv` in the root directory.

---

## 6. D1NAMO Dataset Support

Time-LLM fully supports the D1NAMO dataset:

```bash
# Generate D1NAMO configs
python scripts/time_llm/config_generator.py --mode train_inference \
    --dataset d1namo --data_scenario standardized \
    --patients 001,002,003,004,005,006,007 --llm_models GPT2 --seeds 831363 --epochs 10

# Run D1NAMO experiments
python scripts/time_llm/run_experiments.py --modes train_inference --datasets d1namo
```

---

## Advanced Usage

### Multiple Seeds
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --patients 570,584 --llm_models GPT2 \
    --seeds 831363,809906,427368,238822,247659 --epochs 10
```

### Custom Output Directory
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --patients 570,584 --llm_models GPT2 --epochs 10 \
    --output_dir ./custom_experiments/
```

### All Models and Comprehensive Setup
```bash
python scripts/time_llm/config_generator.py --mode train_inference \
    --dataset ohiot1dm --data_scenario standardized \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models GPT2,LLAMA,BERT --seeds 831363 --epochs 10
```

---

## Window Configurations

Time-LLM automatically generates configurations for both window types:
- **6_6**: 6 input timesteps → 6 prediction timesteps  
- **6_9**: 6 input timesteps → 9 prediction timesteps

**Note**: Unlike Chronos which requires explicit `--window_config` parameter, Time-LLM automatically creates both window configurations during config generation. No additional parameter needed.

---

## Data Format

Time-LLM uses the standard CSV format with columns:
- `item_id`: Patient ID
- `timestamp`: Datetime stamp  
- `target`: Blood glucose value

Example:
```csv
item_id,timestamp,target
570,2022-01-17 00:04:00,135
570,2022-01-17 00:09:00,143
```

This is different from Chronos which uses windowed format (BG_{t-5}, BG_{t-4}, etc.).

---

## Performance Expectations

### Training Performance
- **GPT2**: ~30 minutes per patient (10 epochs, RTX 2060)
- **LLAMA**: ~60 minutes per patient (10 epochs, RTX 2060)  
- **BERT**: ~45 minutes per patient (10 epochs, RTX 2060)
- **Window Configs**: Both 6_6 and 6_9 generated automatically

### Inference Performance  
- **Batch Processing**: 64 samples per batch
- **Processing Speed**: Varies by model size
- **Memory Usage**: Depends on LLM model (GPT2 < BERT < LLAMA)

---

## Troubleshooting

### Common Issues:
1. **Missing prompt files**: Fixed! Prompt files now copied to all scenario folders
2. **Path issues**: All paths are relative and work from any installation location
3. **Missing data scenarios**: All scenarios (standardized, noisy, denoised, missing_periodic, missing_random) are supported
4. **Cross-scenario data paths**: Automatically handled by the unified generator

### Verification Commands:
```bash
# Check generated experiments
ls experiments/time_llm*/

# Verify data files exist
head -n 3 data/ohiot1dm/noisy/570-ws-testing.csv

# Check prompt files
find data/ohiot1dm -name "t1dm_prompt.txt"

# Test config generation
python scripts/time_llm/config_generator.py --mode train \
    --patients 570 --llm_models GPT2 --epochs 1
```

---

## Comprehensive Experiment Generation

For complete experiment generation across all scenarios:

```bash
# Use the comprehensive generator
python scripts/time_llm/generate_all_time_llm_configs.py
```

This script generates:
- Training configs for all scenarios
- Inference configs for all scenarios  
- Training+inference configs for all scenarios
- Cross-scenario configs (train clean, test on each scenario)
- Both ohiot1dm and d1namo datasets
- Multiple LLM models (GPT2, LLAMA, BERT)

---

## Results & CSV Extraction

### Automatic CSV Logging
- **After Each Experiment**: Metrics automatically extracted to individual CSV files
- **Comprehensive Results**: Combined CSV with all experiment results
- **File Naming**: `time_llm_{experiment_type}_results.csv`
- **Contents**: Configuration details, performance metrics (RMSE, MAE, MAPE), timestamps

### Manual Extraction
```bash
# Extract all Time-LLM metrics
python extract_all_metrics.py --time_llm_only

# Extract metrics from specific experiment directory
python -c "
from scripts.utilities.extract_metrics import extract_metrics_to_csv
extract_metrics_to_csv('./experiments/time_llm_training_inference_ohiot1dm/', './custom_results.csv')
"
```

### CSV Structure Example
```csv
seed,model,dim,seq,context,pred,patch,epochs,patient_id,log_datetime,rmse,mae,mape
831363,GPT2,768,6,6,6,6,10,570,2025-10-11_16-30-15,28.45,15.23,8.92
```

---

## Notes
- All commands assume you are in the project root directory
- Patient IDs (ohiot1dm): 540,544,552,559,563,567,570,575,584,588,591,596
- Patient IDs (d1namo): 001,002,003,004,005,006,007
- Seeds from utilities/seeds.py are used by default
- LLM Models: GPT2 (768 dim), LLAMA (4096 dim), BERT (768 dim)
- All paths are relative and work from any installation location
- Cross-scenario inference requires specifying both `--data_scenario` and `--train_data_scenario`