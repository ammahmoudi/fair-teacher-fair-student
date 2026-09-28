# Chronos Config Generation Usage Guide

This guide explains how to use the unified Chronos configuration system for generating experiment configurations.

## Quick Start

### Method 1: Using the Python Script Directly

```bash
# Navigate to the chronos directory
cd /path/to/fair-teacher-fair-student/scripts/chronos

# Basic usage - generate training configs
python config_generator.py --mode train

# Basic usage - generate inference configs
python config_generator.py --mode inference

# Advanced usage - with data scenarios
python config_generator.py --mode train --data_scenario noisy
python config_generator.py --mode train --data_scenario missing_periodic
```

### Method 2: Using the Shell Script with Examples

```bash
# Navigate to the chronos directory
cd /path/to/fair-teacher-fair-student/scripts/chronos

# Make the script executable
chmod +x chronos_config_examples.sh

# Run examples for all modes
./chronos_config_examples.sh all

# Run examples for specific mode
./chronos_config_examples.sh train
./chronos_config_examples.sh inference
```

## Understanding the Four Modes

### 1. **Training Mode** (`--mode train`)
- **Purpose**: Generate configs for training Chronos models from scratch
- **Output**: Creates configs in `experiments/chronos_training/`
- **What it does**: Sets up LoRA fine-tuning configurations

```bash
# Train on default patients (570,584) with default models
python config_generator.py --mode train

# Train specific patient with specific model
python config_generator.py --mode train --patients 570 --models amazon/chronos-t5-tiny

# Train multiple patients with custom seeds
python config_generator.py --mode train --patients 540,570,584 --seeds 831363,809906
```

### 2. **Inference Mode** (`--mode inference`)
- **Purpose**: Generate configs for inference using pretrained Chronos models
- **Output**: Creates configs in `experiments/chronos_inference/`
- **What it does**: Uses Amazon's pretrained models directly

```bash
# Inference with default settings
python config_generator.py --mode inference

# Inference with specific model
python config_generator.py --mode inference --models amazon/chronos-t5-base --patients 570
```

### 3. **Trained Inference Mode** (`--mode trained_inference`)
- **Purpose**: Generate configs for inference using YOUR trained checkpoints
- **Output**: Creates configs in `experiments/chronos_trained_inference/`
- **Prerequisites**: You must have run training first and have checkpoint files
- **What it does**: Automatically finds your trained checkpoints and uses them

```bash
# Use trained checkpoints for inference
python config_generator.py --mode trained_inference --patients 570 --models amazon/chronos-t5-tiny
```

### 4. **LoRA Inference Mode** (`--mode lora_inference`)
- **Purpose**: Generate configs for inference using YOUR trained LoRA adapters
- **Output**: Creates configs in `experiments/chronos_lora_inference/`
- **Prerequisites**: You must have run training with LoRA first
- **What it does**: Loads your trained LoRA adapters for inference

```bash
# Use trained LoRA adapters for inference
python config_generator.py --mode lora_inference --patients 570 --models amazon/chronos-t5-tiny
```

## Understanding the Parameters

### Required Parameters
- `--mode`: Choose from `train`, `inference`, `trained_inference`, `lora_inference`

### Optional Parameters
- `--dataset`: Dataset type (default: ohiot1dm)
  - `ohiot1dm`: OhioT1DM dataset (default)
  - `d1namo`: D1NAMO dataset
- `--data_scenario`: Data type for inference/testing (default: standardized)
  - `standardized`: Clean/raw data (default)
  - `noisy`: Data with added noise
  - `denoised`: Data that has been denoised  
  - `missing_periodic`: Data with periodic missing values
  - `missing_random`: Data with random missing values
- `--train_data_scenario`: Data type used for training (optional, for cross-scenario evaluation)
  - Only used with `trained_inference` and `lora_inference` modes
  - If not specified, uses same as `--data_scenario`
  - Enables testing model robustness: train on clean, test on noisy/missing data
- `--patients`: Comma-separated patient IDs (default: 570,584)
- `--models`: Comma-separated model names (default: amazon/chronos-t5-tiny,amazon/chronos-t5-base)
- `--seeds`: Comma-separated seeds (default: uses fixed_seeds from utilities)
- `--output_dir`: Custom output directory (default: auto-generated based on mode, dataset, and data_scenario)

## Datasets Explained

### 1. **OhioT1DM Dataset** (Default)
- **Path**: `/path/to/fair-teacher-fair-student/data/ohiot1dm/`
- **Description**: OhioT1DM dataset with structured subfolders  
- **Subfolders**: `raw_standardized/`, `noisy/`, `denoised/`

### 2. **D1NAMO Dataset**
- **Path**: `/path/to/fair-teacher-fair-student/data/d1namo/`
- **Description**: D1NAMO dataset with structured subfolders
- **Subfolders**: `raw_standardized/`, `noisy/`, `missing_periodic/`, `missing_random/`

## Data Scenarios Explained

### 1. **Standardized** (Default)
- **Path**: `/path/to/fair-teacher-fair-student/data/{dataset}/raw_standardized/`
- **Description**: Clean, raw patient data without any modifications
- **Use**: Baseline experiments, standard model training

### 2. **Noisy**  
- **Path**: `/path/to/fair-teacher-fair-student/data/{dataset}/noisy/`
- **Description**: Data with artificially added noise
- **Use**: Testing model robustness to sensor noise

### 3. **Denoised**
- **Path**: `/path/to/fair-teacher-fair-student/data/{dataset}/denoised/`
- **Description**: Previously noisy data that has been cleaned/denoised
- **Use**: Testing denoising algorithm effectiveness (mainly available for ohiot1dm)

### 4. **Missing Periodic**
- **Path**: `/path/to/fair-teacher-fair-student/data/{dataset}/missing_periodic/`
- **Description**: Data with missing values in regular/periodic patterns
- **Use**: Testing model handling of systematic data gaps

### 5. **Missing Random**
- **Path**: `/path/to/fair-teacher-fair-student/data/{dataset}/missing_random/`
- **Description**: Data with randomly missing values
- **Use**: Testing model robustness to irregular data availability

### Available Models
- `amazon/chronos-t5-tiny` (fastest, smallest)
- `amazon/chronos-t5-small`
- `amazon/chronos-t5-base` (good balance)
- `amazon/chronos-t5-large` (best quality, slowest)

## Typical Workflow

### Step 1: Generate Training Configs
```bash
# Generate training configs for your patients
python config_generator.py --mode train --patients 570,584 --models amazon/chronos-t5-tiny
```

### Step 2: Run Training (outside this script)
```bash
# You would run your actual training here using the generated configs
# This creates checkpoint files in the training output directories
```

### Step 3: Generate Inference Configs Using Trained Models
```bash
# Generate inference configs that use your trained checkpoints
python config_generator.py --mode trained_inference --patients 570,584 --models amazon/chronos-t5-tiny
```

## Output Structure

After running the commands, you'll see this directory structure:

```
experiments/
├── chronos_training/           # Training configurations
├── chronos_inference/          # Pretrained model inference
├── chronos_trained_inference/  # Your trained model inference
└── chronos_lora_inference/     # Your LoRA adapter inference

# Each contains folders like:
seed_831363_model_amazon-chronos-t5-tiny_dtype_float32_mode_train_context_512_pred_64/
└── patient_570/
    ├── config.gin              # The actual config file
    └── logs/                   # Directory for training logs
```

## Examples from the Shell Script

The `chronos_config_examples.sh` script contains many examples. Run it to see them in action:

```bash
# See all examples
./chronos_config_examples.sh all

# See specific mode examples
./chronos_config_examples.sh train
./chronos_config_examples.sh inference
./chronos_config_examples.sh trained_inference
./chronos_config_examples.sh lora_inference
```

## Common Use Cases

### Research Experiment Setup
```bash
# Generate configs for multiple patients and models for comparison
python config_generator.py --mode train \
  --patients 540,544,552,570,584 \
  --models amazon/chronos-t5-tiny,amazon/chronos-t5-base \
  --seeds 831363,809906,123456
```

### Dataset Comparison Study
```bash
# Train same model on different datasets (ohiot1dm is default)
python config_generator.py --mode train --patients 570,584  # uses ohiot1dm by default
python config_generator.py --mode train --dataset d1namo --patients 570,584 
python config_generator.py --mode train --dataset ohiot1dm --patients 570,584  # explicit
```

### Robustness Testing
```bash  
# Test model on noisy data across datasets
python config_generator.py --mode train --dataset d1namo --data_scenario noisy --patients 570
python config_generator.py --mode train --dataset ohiot1dm --data_scenario noisy --patients 570
```

### Missing Data Experiments
```bash
# Compare periodic vs random missing patterns on D1NAMO
python config_generator.py --mode train --dataset d1namo --data_scenario missing_periodic --patients 570,584
python config_generator.py --mode train --dataset d1namo --data_scenario missing_random --patients 570,584
```

### Cross-Scenario Robustness Testing
This is a powerful feature for evaluating model robustness - train on one data condition, test on another!

```bash
# Train on clean data, test robustness on missing values
python config_generator.py --mode trained_inference --data_scenario missing_periodic --train_data_scenario standardized --patients 570

# Train on clean data, test robustness on noisy data  
python config_generator.py --mode trained_inference --data_scenario noisy --train_data_scenario standardized --patients 570

# Train on noisy data, test on denoised data (denoising effectiveness)
python config_generator.py --mode trained_inference --data_scenario denoised --train_data_scenario noisy --patients 570

# Cross-dataset robustness: train on D1NAMO clean, test on OhioT1DM missing data
python config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --train_data_scenario standardized --patients 570
```

**Directory Structure for Cross-Scenario:**
- `experiments/chronos_trained_inference_train_standardized_test_missing_periodic/` 
- `experiments/chronos_trained_inference_train_noisy_test_denoised/`
- `experiments/chronos_trained_inference_d1namo_train_standardized_test_missing_random/`

### Quick Single Patient Test
```bash
# Test with one patient and one model
python config_generator.py --mode train --patients 570 --models amazon/chronos-t5-tiny --seeds 831363
```

### Production Inference Setup
```bash
# Generate inference configs using your best trained model
python config_generator.py --mode trained_inference --dataset d1namo --patients 570,584 --models amazon/chronos-t5-base
```

## Troubleshooting

### "No checkpoint found" error
- Make sure you've run training mode first
- Check that training completed and created checkpoint-final files
- Verify the patient IDs and model names match between training and inference

### Permission denied
```bash
chmod +x chronos_config_examples.sh
```

### Import errors
- Make sure you're running from the correct directory: `/path/to/fair-teacher-fair-student/scripts/chronos`
- The script should automatically handle imports

## Getting Help

```bash
# Show all available options
python config_generator.py --help

# Show examples
./chronos_config_examples.sh help
```