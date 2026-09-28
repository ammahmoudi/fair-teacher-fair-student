# Time-LLM Unified Configuration Generator - Usage Guide

## Overview
The unified Time-LLM configuration generator (`config_generator.py`) combines all Time-LLM config generators into one powerful tool that supports multiple datasets, data scenarios, cross-scenario evaluation, and different LLM models.

## Quick Start

### Basic Commands
```bash
# Generate training configurations (default: ohiot1dm, standardized, GPT2+LLAMA)
python3 config_generator.py --mode train

# Generate inference configurations (epochs=0)
python3 config_generator.py --mode inference

# Generate combined training+inference configurations
python3 config_generator.py --mode train_inference
```

### Using Different Datasets
```bash
# Use D1NAMO dataset instead of OhioT1DM
python3 config_generator.py --mode train --dataset d1namo

# Use specific patients
python3 config_generator.py --mode train --patients 570,584
```

### Data Scenarios
```bash
# Train on noisy data
python3 config_generator.py --mode train --data_scenario noisy

# Train on missing periodic data
python3 config_generator.py --mode train --data_scenario missing_periodic

# Inference on denoised data
python3 config_generator.py --mode inference --data_scenario denoised
```

### Cross-Scenario Evaluation
Train on one scenario, test on another:
```bash
# Train on standardized, test on missing_periodic
python3 config_generator.py --mode train_inference \
    --data_scenario missing_periodic --train_data_scenario standardized

# Train on standardized, test on noisy (D1NAMO dataset)
python3 config_generator.py --mode train_inference \
    --dataset d1namo --data_scenario noisy --train_data_scenario standardized
```

### Model-Specific Configurations
```bash
# Use only GPT2 model
python3 config_generator.py --mode train --llm_models GPT2

# Use only LLAMA model
python3 config_generator.py --mode train --llm_models LLAMA

# Use only BERT model  
python3 config_generator.py --mode train --llm_models BERT

# Use all three models
python3 config_generator.py --mode train --llm_models GPT2,LLAMA,BERT
```

## Parameters

### Required Parameters
- `--mode`: Operation mode
  - `train`: Generate training configurations (epochs=10)
  - `inference`: Generate inference configurations (epochs=0)
  - `train_inference`: Generate combined configurations

### Optional Parameters
- `--dataset`: Dataset type (default: `ohiot1dm`)
  - `ohiot1dm`: OhioT1DM dataset
  - `d1namo`: D1NAMO dataset

- `--data_scenario`: Data scenario for inference/testing (default: `standardized`)
  - `standardized`: Clean/raw data
  - `noisy`: Data with added noise
  - `denoised`: Data that has been denoised
  - `missing_periodic`: Data with periodic missing values
  - `missing_random`: Data with random missing values

- `--train_data_scenario`: Training data scenario for cross-scenario evaluation
  - If not specified, uses `--data_scenario` for both training and testing

- `--patients`: Comma-separated patient IDs (default: `570,584`)
- `--llm_models`: Comma-separated LLM models (default: `GPT2,LLAMA`)
- `--seeds`: Comma-separated seeds (default: first 2 from fixed_seeds)
- `--epochs`: Number of training epochs (default: 10 for train, 0 for inference)
- `--output_dir`: Custom output directory (default: auto-generated)

## LLM Models

### Supported Models
- **GPT2**: llm_dim=768
- **LLAMA**: llm_dim=4096  
- **BERT**: llm_dim=768

### Model Configurations
Each model generates configs for multiple sequence length combinations:
- seq_len=6, context_len=6, pred_len=6, patch_len=6
- seq_len=6, context_len=6, pred_len=9, patch_len=6

## Output Structure

### Directory Naming
```
experiments/
├── time_llm_training/                                    # Basic training
├── time_llm_training_d1namo/                            # D1NAMO dataset
├── time_llm_training_missing_periodic/                  # Missing periodic data
├── time_llm_inference_noisy/                           # Noisy inference
├── time_llm_training_inference_train_standardized_test_missing_periodic/  # Cross-scenario
└── ...
```

### Config Structure
```
experiment_folder/
├── seed_1_model_GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10/
│   ├── patient_570/
│   │   ├── config.gin
│   │   └── logs/
│   └── patient_584/
│       ├── config.gin
│       └── logs/
└── seed_1_model_LLAMA_dim_4096_seq_6_context_6_pred_9_patch_6_epochs_10/
    └── ...
```

## Configuration File Format

Each `config.gin` contains:
```python
run.log_dir = "path/to/logs"
run.data_settings = {
    'path_to_train_data': './data/dataset/scenario/patient-ws-training.csv',
    'path_to_test_data': './data/dataset/scenario/patient-ws-testing.csv',
    'input_features': ['target'],
    'labels': ['target'], 
    'prompt_path': './data/dataset/scenario/t1dm_prompt.txt',
    # ... other data settings
}

run.training_settings = {
    'llm_model': 'GPT2',
    'llm_dim': 768,
    'seq_len': 6,
    'pred_len': 6,
    'patch_len': 6,
    # ... other training settings
}

run.hardware_settings = {
    'use_gpu': True,
    # ... other hardware settings
}
```

## Convenience Scripts

### Quick Commands
Edit and run specific configurations:
```bash
bash quick_config_time_llm.sh
```

### Auto-Run Common Configs
Generate standard research configurations:
```bash
bash run_common_time_llm_configs.sh
```

## Examples

### Research Workflow Examples

1. **Basic Model Comparison**
```bash
# Compare all three LLM models on standardized data
python3 config_generator.py --mode train_inference --llm_models GPT2,LLAMA,BERT
```

2. **Data Robustness Study**  
```bash
# Test model trained on clean data against various test scenarios
python3 config_generator.py --mode train_inference --data_scenario missing_periodic --train_data_scenario standardized
python3 config_generator.py --mode train_inference --data_scenario noisy --train_data_scenario standardized
python3 config_generator.py --mode train_inference --data_scenario missing_random --train_data_scenario standardized
```

3. **Dataset Comparison**
```bash
# Compare same model on different datasets
python3 config_generator.py --mode train --dataset ohiot1dm --llm_models GPT2
python3 config_generator.py --mode train --dataset d1namo --llm_models GPT2
```

4. **Extended Training**
```bash
# Train for more epochs with specific seeds
python3 config_generator.py --mode train --epochs 20 --seeds 1,2,3,4,5
```

## Troubleshooting

### Common Issues
1. **Import Error**: Make sure you're in the `scripts/time_llm/` directory
2. **Python Command**: Use `python3` instead of `python` on some systems
3. **Permission Denied**: Run `chmod +x *.sh` to make scripts executable
4. **No Configs Generated**: Check that the data paths exist in your workspace

### Getting Help
```bash
python3 config_generator.py --help
```