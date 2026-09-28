# Chronos Scripts

Organized collection of scripts for Chronos time series forecasting experiments.

## Directory Structure

```
chronos/
├── README.md                                  # This file
├── USAGE_GUIDE.md                             # Comprehensive usage documentation
│
├── config_generator.py                        # 🔧 MAIN: Config generator
├── run_experiments.py                         # 🚀 MAIN: Experiment runner
│
└── helpers/                                   # Convenience/batch scripts
    ├── batch_generate_all_configs.py          # Generate all configs at once
    ├── batch_generate_trained_inference.py    # Generate trained inference configs
    └── examples.sh                            # Usage examples
```

## Main Scripts (Use These)

### 1. Config Generator - `config_generator.py`

**The main configuration generator for all Chronos experiments.**

```bash
# Generate training configs
python config_generator.py --mode train

# Generate inference configs  
python config_generator.py --mode inference

# Generate trained inference configs (uses existing checkpoints)
python config_generator.py --mode trained_inference

# Generate with specific parameters
python config_generator.py --mode train --patients 570,584 --models amazon/chronos-t5-tiny --seeds 831363
```

**Modes:**
- `train` - Generate training configurations
- `inference` - Generate inference configurations for pretrained models
- `trained_inference` - Generate inference configs using your trained checkpoints
- `lora_inference` - Generate LoRA inference configurations

### 2. Experiment Runner - `run_experiments.py`

**Automatically discover and run all Chronos experiments.**

```bash
# Run all experiments
python run_experiments.py

# Run with parallel processing
python run_experiments.py --parallel --max_workers 4

# Filter by mode
python run_experiments.py --modes training,inference

# Filter by dataset
python run_experiments.py --datasets d1namo,ohiot1dm

# Resume interrupted experiments
python run_experiments.py --resume

# Dry run (see what would be executed)
python run_experiments.py --dry_run
```

## Helper Scripts (Convenience Shortcuts)

### 1. Batch Generate All Configs - `helpers/batch_generate_all_configs.py`

**Convenience script to generate all training and inference configs in one command.**

```bash
python helpers/batch_generate_all_configs.py
```

This automatically generates:
- All training configs (all datasets, scenarios, patients)
- All inference configs (pretrained models)
- All cross-scenario configs (train on clean, test on corrupted)

### 2. Batch Generate Trained Inference - `helpers/batch_generate_trained_inference.py`

**Generate trained_inference configs after training is complete.**

```bash
python helpers/batch_generate_trained_inference.py
```

Use this AFTER training experiments have completed and checkpoints are available.

### 3. Examples Script - `helpers/examples.sh`

**Interactive shell script with usage examples.**

```bash
# Make executable
chmod +x helpers/examples.sh

# Run all examples
./helpers/examples.sh all

# Run specific mode examples
./helpers/examples.sh train
./helpers/examples.sh inference
```

## Quick Start Workflow

### Step 1: Generate Training Configs

```bash
# Option A: Generate specific configs
python config_generator.py --mode train --patients 570,584

# Option B: Generate all training configs
python helpers/batch_generate_all_configs.py
```

### Step 2: Run Training Experiments

```bash
# Run all training experiments
python run_experiments.py --modes training

# Or with parallel processing
python run_experiments.py --modes training --parallel --max_workers 4
```

### Step 3: Generate Trained Inference Configs (After Training)

```bash
# Generate configs that use your trained checkpoints
python helpers/batch_generate_trained_inference.py
```

### Step 4: Run Inference Experiments

```bash
# Run all inference experiments
python run_experiments.py --modes inference,trained_inference
```

## Configuration Modes Explained

### Training Mode
- Trains Chronos models from scratch using LoRA fine-tuning
- Generates configs in `experiments/chronos_training_*/`
- Saves checkpoints for later use

### Inference Mode  
- Uses pretrained Chronos models from HuggingFace
- No training required
- Generates configs in `experiments/chronos_inference_*/`

### Trained Inference Mode
- Uses YOUR trained checkpoints from training experiments
- Requires completed training experiments first
- Generates configs in `experiments/chronos_trained_inference_*/`

### LoRA Inference Mode
- Specifically for LoRA-adapted models
- Uses saved LoRA adapters
- Generates configs in `experiments/chronos_lora_inference_*/`

## Data Scenarios

All config generators support multiple data scenarios:

- `standardized` - Clean, standardized data (default)
- `noisy` - Data with added noise
- `denoised` - Noise-reduced data
- `missing_periodic` - Periodic missing values
- `missing_random` - Random missing values

```bash
# Generate configs for specific scenario
python config_generator.py --mode train --data_scenario noisy

# Cross-scenario: train on clean, test on corrupted
python config_generator.py --mode inference --data_scenario noisy
```

## Supported Datasets

- **OhioT1DM**: Diabetes blood glucose dataset (patients: 540, 544, 552, 559, 563, 570, 575, 584, 588, 591, 596)
- **D1NAMO**: Extended diabetes dataset (patients: 1, 3, 5, 6, 9, 13, 14, 15, 16, 17, 18)

```bash
# Generate for specific dataset
python config_generator.py --mode train --dataset d1namo
python config_generator.py --mode train --dataset ohiot1dm
```

## Models Supported

- `amazon/chronos-t5-tiny` - Smallest, fastest (default)
- `amazon/chronos-t5-mini` - Small model
- `amazon/chronos-t5-small` - Medium model
- `amazon/chronos-t5-base` - Large model
- `amazon/chronos-t5-large` - Largest model

```bash
# Use specific model
python config_generator.py --mode train --models amazon/chronos-t5-base
```

## Output Locations

- **Generated Configs**: `experiments/chronos_*/`
- **Experiment Results**: `experiments/chronos_*/logs/`
- **Checkpoints**: `experiments/chronos_training_*/checkpoints/`
- **Metrics**: `experiments/chronos_*/final_results.csv`

## Tips

1. **Start Small**: Test with 1-2 patients first before generating all configs
   ```bash
   python config_generator.py --mode train --patients 570
   ```

2. **Use Parallel Processing**: Speed up experiments significantly
   ```bash
   python run_experiments.py --parallel --max_workers 8
   ```

3. **Resume After Interruption**: Don't restart from scratch
   ```bash
   python run_experiments.py --resume
   ```

4. **Dry Run First**: See what will be executed without running
   ```bash
   python run_experiments.py --dry_run
   ```

## For More Details

See `USAGE_GUIDE.md` for comprehensive documentation including:
- Detailed parameter explanations
- Advanced configuration options
- Troubleshooting guides
- Complete examples for all modes
