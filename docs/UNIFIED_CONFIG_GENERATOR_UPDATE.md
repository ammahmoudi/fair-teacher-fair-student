# Unified Config Generator Update: Per-Patient Inference Support

## Overview
Added support for per-patient inference from all-patients checkpoint to the unified Time-LLM config generator (`scripts/time_llm/config_generator_time_llm_unified.py`). This allows you to train a model on combined data from all patients and then test it on individual patients.

## Changes Made

### 1. New Mode: `per_patient_inference`
Added a new operation mode to the unified config generator that generates inference configs for individual patients using a checkpoint trained on all patients.

### 2. New Parameters
- `--checkpoint-path`: Direct path to checkpoint file (e.g., `/path/to/checkpoint.pth`)
- `--checkpoint-dir`: Directory to search for matching checkpoint (automatically finds checkpoint matching experiment parameters)

### 3. Config Generation Features
When using `per_patient_inference` mode, the generated configs have:
- Empty `path_to_train_data` (no training data loaded)
- `mode: 'inference'` (inference only, no training)
- `restore_checkpoint_path` pointing to the all-patients checkpoint
- Individual patient test data paths

### 4. Updated Scripts
- **scripts/time_llm/config_generator_time_llm_unified.py**: Added per_patient_inference mode
- **scripts/run_all_patients_distillation.sh**: Updated to use unified config generator instead of separate script
- **Removed**: `distillation/scripts/generate_per_patient_configs.py` (obsolete)

## Usage

### Basic Usage with Checkpoint Path
```bash
python scripts/time_llm/config_generator_time_llm_unified.py \
    --mode per_patient_inference \
    --checkpoint-path /path/to/phase1/checkpoint.pth \
    --llm_models BERT \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --seeds 42 \
    --dataset ohiot1dm \
    --data_scenario standardized
```

### Auto-Discovery from Directory
```bash
python scripts/time_llm/config_generator_time_llm_unified.py \
    --mode per_patient_inference \
    --checkpoint-dir /path/to/phase1/ \
    --llm_models BERT \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --seeds 42 \
    --dataset ohiot1dm
```

The `--checkpoint-dir` option will automatically search for checkpoints matching the experiment parameters (seed, model, dimensions, sequence lengths, etc.).

### Integration with Distillation Pipeline
The distillation pipeline (`scripts/run_all_patients_distillation.sh`) now automatically uses the unified config generator for per-patient inference on:
1. Teacher model (after Phase 1)
2. Student baseline (after Phase 2)
3. Distilled student (after Phase 3)

Example distillation pipeline usage:
```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 5 \
    --student-epochs 5 \
    --distill-epochs 10
```

## Generated Config Structure
```python
run.data_settings = {
    'path_to_test_data': './data/ohiot1dm/raw_standardized/570-ws-testing.csv',
    'path_to_train_data': '',  # Empty - no training data
    ...
}

run.llm_settings = {
    'mode': 'inference',  # Inference only
    'restore_checkpoint_path': '/path/to/checkpoint.pth',  # All-patients checkpoint
    'llm_model': 'BERT',
    'llm_dim': 768,
    ...
}
```

## Benefits
1. **Unified System**: All config generation uses the same infrastructure
2. **Consistency**: Same parameter handling across all modes
3. **Maintainability**: Single source of truth for config generation
4. **Flexibility**: Supports both direct checkpoint path and auto-discovery
5. **Integration**: Seamlessly integrated with existing distillation pipeline

## Testing
Verified that generated configs:
- Have empty training data path
- Set mode to 'inference'
- Include correct checkpoint restoration path
- Contain per-patient test data paths
- Work with existing `run_all_time_llm_experiments.py` runner

## Example Workflow
1. **Train on all patients**:
   ```bash
   python distillation/scripts/train_teachers.py --all-patients --model bert
   ```

2. **Generate per-patient inference configs**:
   ```bash
   python scripts/time_llm/config_generator_time_llm_unified.py \
       --mode per_patient_inference \
       --checkpoint-dir ./phase1_teacher/ \
       --llm_models BERT \
       --patients 540,544,552,559,563,567,570,575,584,588,591,596
   ```

3. **Run per-patient inference**:
   ```bash
   python scripts/time_llm/run_all_time_llm_experiments.py \
       --config-dir ./experiments/time_llm_per_patient_inference_ohiot1dm/ \
       --fix-results
   ```

## Notes
- The unified generator automatically handles checkpoint path resolution when using `--checkpoint-dir`
- Checkpoint discovery searches for files matching experiment parameters (seed, model, dimensions, etc.)
- Empty training data path ensures no training data loading in inference mode (compatible with main.py fix)
- All existing modes (train, inference, train_inference) remain unchanged
