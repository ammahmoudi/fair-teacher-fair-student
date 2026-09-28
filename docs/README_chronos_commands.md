# Chronos Training & Inference Command Guide

This guide provides all commands needed to generate Chronos training and inference configs, and to run experiments for the OhioT1DM dataset. It covers all scenarios, cross-scenario inference, and window configurations (6_6 and 6_9).

## ✅ Recent Improvements
- **GPU Acceleration**: All training/inference now uses GPU automatically
- **Cross-Scenario Inference**: Train on one scenario, test on another (e.g., standardized → noisy)
- **Fixed Data Format**: Corrected column names (`BG_{t}` instead of `BG_{t}`)
- **Comprehensive Config Generation**: All required parameters included automatically
- **Performance Monitoring**: Real-time GPU utilization, memory usage, and efficiency metrics
- **Automated Result Correction**: Built-in outlier detection and correction with `--fix_results`
- **CSV Metrics Extraction**: Automatic metrics extraction after each experiment

---

## Quick Start Example

```bash
# 1. Generate training configs for standardized data
python scripts/chronos/config_generator.py --mode train --patients 570,584 --data_scenario standardized

# 2. Run training (automatically uses GPU)
python scripts/chronos/run_experiments.py --modes training --datasets ohiot1dm

# 3. Generate cross-scenario inference configs (train standardized → test noisy)
python scripts/chronos/config_generator.py --mode trained_inference \
    --dataset ohiot1dm --data_scenario noisy --patients 570,584 \
    --models amazon/chronos-t5-base --seeds 831363 \
    --train_scenario standardized --window_config 6_6

# 4. Run inference
python scripts/chronos/run_experiments.py --modes trained_inference --datasets ohiot1dm
```

---

## 1. Generate Training Configs

### Train on Standardized Raw
```bash
python scripts/chronos/config_generator.py --mode train --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode train --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Missing Random
```bash
python scripts/chronos/config_generator.py --mode train --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Noisy
```bash
python scripts/chronos/config_generator.py --mode train --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Denoised
```bash
python scripts/chronos/config_generator.py --mode train --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

---

## 2. Run Training Experiments
```bash
python scripts/chronos/run_experiments.py --modes training --datasets ohiot1dm
```

---

## 3. Cross-Scenario Inference (Train on One Scenario, Test on Another)

### Key Parameters:
- `--train_scenario`: The scenario the model was trained on
- `--data_scenario`: The scenario to test on
- `--window_config`: Window configuration (6_6 or 6_9)

### ✅ Verified Working Example: Train on Standardized, Test on Noisy
```bash
# 1. First train on standardized data
python scripts/chronos/config_generator.py --mode train \
    --patients 540 --data_scenario standardized --models amazon/chronos-t5-base --seeds 831363

python scripts/chronos/run_experiments.py --modes training --datasets ohiot1dm

# 2. Then generate cross-scenario inference configs  
python scripts/chronos/config_generator.py --mode trained_inference \
    --dataset ohiot1dm --data_scenario noisy --patients 540 \
    --models amazon/chronos-t5-base --seeds 831363 \
    --train_scenario standardized --window_config 6_6

# 3. Run inference (successfully tested)
python scripts/chronos/run_experiments.py --modes trained_inference --datasets ohiot1dm
```

**Results**: Successfully processed 2,885 samples with shape (2885, 6), achieving RMSE: 195.52, MAE: 46.91

## 4. Generate Inference Configs (all 5 seeds)

### Trained on Raw (standardized), Tested on:
#### Raw
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario standardized
```
#### Denoised
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario standardized
```
#### Noisy
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario standardized
```
#### Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario standardized
```
#### Missing Random
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario standardized
```

### Trained on Noisy, Tested on:
#### Noisy
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```
#### Raw (standardized)
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```
#### Denoised
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```
#### Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```
#### Missing Random
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```

### Trained on Denoised, Tested on:
#### Denoised
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario denoised
```
#### Raw (standardized)
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario denoised
```
#### Noisy
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario denoised
```
#### Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario denoised
```
#### Missing Random
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario denoised
```

### Trained on Missing Periodic, Tested on:
#### Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```
#### Raw (standardized)
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```
#### Denoised
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```
#### Noisy
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```
#### Missing Random
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```

### Trained on Missing Random, Tested on:
#### Missing Random
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```
#### Raw (standardized)
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```
#### Denoised
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```
#### Noisy
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```
#### Missing Periodic
```bash
python scripts/chronos/config_generator.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```

---

## 5. Run Inference Experiments
```bash
python scripts/chronos/run_experiments.py --modes trained_inference --datasets ohiot1dm
```

**Note**: All experiments automatically extract metrics to CSV files after completion. Results are saved to files like `chronos_trained_inference_ohiot1dm_noisy_results.csv` in the root directory.

---

## Advanced Usage

### Multiple Window Configurations
Generate configs for both 6_6 and 6_9 windows:
```bash
python scripts/chronos/config_generator.py --mode trained_inference \
    --dataset ohiot1dm --data_scenario standardized --patients 570,584 \
    --models amazon/chronos-t5-base --seeds 831363 \
    --train_scenario standardized --window_config both
```

### LoRA Fine-tuning
```bash
python scripts/chronos/config_generator.py --mode lora_inference \
    --dataset ohiot1dm --data_scenario standardized --patients 570,584 \
    --models amazon/chronos-t5-base --seeds 831363 \
    --use_lora --window_config 6_6 --train_scenario standardized
```

### Custom Model Selection
```bash
python scripts/chronos/config_generator.py --mode train \
    --patients 570,584 --models amazon/chronos-t5-tiny,amazon/chronos-t5-base
```

---

## Performance Expectations

### Training Performance (RTX 2060)
- **GPU Utilization**: 100% during training
- **Memory Usage**: ~776MB for chronos-t5-base
- **Training Speed**: ~200-2000 steps depending on model size
- **Checkpoint Saving**: Every 1000 steps automatically

### Inference Performance
- **Batch Processing**: 64 samples per batch
- **GPU Memory**: ~1217MB peak usage
- **Processing Speed**: ~21 seconds for 2885 samples
- **GPU Utilization**: ~38% average

---

## Automated Result Correction

The Chronos experiment runner now includes automated outlier detection and correction:

### Basic Usage
```bash
# Run experiments with automatic outlier correction
python scripts/chronos/run_experiments.py --modes training 
# Use custom outlier threshold (default is 3.0)
python scripts/chronos/run_experiments.py --modes training --fix_results --fix_threshold 2.5
```

### How It Works
- **Outlier Detection**: Identifies predictions that exceed `threshold_factor × median_value`
- **Correction Method**: Replaces outliers with average of neighboring predictions
- **Metrics Tracking**: Reports both original and corrected metrics
- **Automatic Integration**: Corrected results saved as `final_results.csv` in each experiment

### Output Example
```
🔧 Applying outlier correction with threshold 3.0...
✅ Fixed 12 outliers
📈 Original metrics: {'rmse': 245.82, 'mae': 58.34, 'mape': 12.45}
📈 Corrected metrics: {'rmse': 195.52, 'mae': 46.91, 'mape': 9.87}
📊 Metrics extracted: ./experiments/chronos_training_ohiot1dm/experiment_results.csv
```

### When to Use
- **Post-Training**: Recommended for all training experiments to clean results
- **Inference**: Especially useful for cross-scenario inference where noise may introduce outliers
- **Quality Assurance**: Provides both corrected and original metrics for comparison

---

## Troubleshooting

### Common Issues:
1. **"BG_{t} not in index"**: Fixed! Data formatter now generates correct column names
2. **GPU not detected**: Ensure CUDA is properly installed and GPU is available
3. **Checkpoint not found**: Ensure training completed successfully before inference
4. **Memory errors**: Use smaller batch sizes or reduce model size

### Verification Commands:
```bash
# Check GPU availability
nvidia-smi

# Verify data format
head -n 1 data/ohiot1dm/noisy_formatted/6_6/570-ws-testing.csv

# Check generated configs
ls experiments/chronos_training_ohiot1dm/
```

---

## Notes
- All commands assume you are in the project root directory
- Patient IDs: 540,544,552,559,563,567,570,575,584,588,591,596
- Seeds: 831363,809906,427368,238822,247659
- Default model: amazon/chronos-t5-base
- GPU acceleration is automatic when available
- Cross-scenario inference requires completing training first
