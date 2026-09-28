# Knowledge Distillation for Forecasting and ECG Classification

This directory contains task-specific knowledge-distillation implementations
for efficient forecasting models and MIT-BIH ECG classifiers.

## Task Paths

- Forecasting: continuous outputs, MSE ground-truth and teacher-matching terms,
  forecasting datasets and RMSE/MAE metrics.
- ECG classification: `distillation/core/ecg_classification_wrapper.py`,
  categorical CE plus temperature-scaled KL, two- or five-class heads,
  per-beat predictions, and subgroup fairness metrics.

The locked ECG pair is BERT -> TinyBERT. Other backbones may be constructible,
but are not five-seed classification results. Forecasting and ECG checkpoints
are not interchangeable.

## Overview

The distillation pipeline enables training smaller, more efficient student models that maintain performance close to larger teacher models by learning from the teacher's predictions (soft targets) in addition to the ground truth labels.

## All-Patients Distillation Pipeline

### Quick Start

Run the complete distillation pipeline on all 12 patients combined (134K samples):

```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 5 \
    --student-epochs 5 \
    --distill-epochs 10
```

### Pipeline Stages

The pipeline executes in two main stages:

**STAGE 1: ALL TRAININGS**
1. **Train Teacher Model**: Full-size model (e.g., BERT-base, 768 dim, 12 layers)
2. **Train Student Baseline**: Small model without distillation (e.g., BERT-tiny, 128 dim, 2 layers)
3. **Train Distilled Student**: Small model with knowledge distillation from teacher

**STAGE 2: ALL INFERENCES**
4. **Teacher Inference**: Per-patient evaluation on all 12 patients
5. **Student Baseline Inference**: Per-patient evaluation on all 12 patients
6. **Distilled Student Inference**: Per-patient evaluation on all 12 patients

### Execution Modes

#### 1. Full Pipeline (Default)
Runs both training and inference:

```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 10 \
    --student-epochs 10 \
    --distill-epochs 15
```

#### 2. Training Only
Runs all training phases, skips inference (useful for quick training iterations):

```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 5 \
    --student-epochs 5 \
    --distill-epochs 10 \
    --training-only
```

This creates a pipeline directory like:
```
distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_12-30-45/
```

#### 3. Inference Only
Runs per-patient inference on previously trained models:

```bash
bash scripts/run_all_patients_distillation.sh \
    --inference-only \
    --pipeline-dir distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_12-30-45
```

**Benefits:**
- Run inference multiple times without retraining
- Test different inference configurations
- Re-evaluate models after fixing bugs
- Generate results for different patient subsets

### Command-Line Options

#### Model Configuration
- `--teacher <model>`: Teacher model name (default: `bert`)
  - Options: `bert`, `bert-base-uncased`, `gpt2`, etc.
- `--student <model>`: Student model name (default: `prajjwal1/bert-tiny`)
  - Options: `prajjwal1/bert-tiny`, `distilbert-base-uncased`, etc.

#### Training Hyperparameters
- `--teacher-epochs <n>`: Teacher training epochs (default: 5)
- `--student-epochs <n>`: Student baseline training epochs (default: 5)
- `--distill-epochs <n>`: Distillation training epochs (default: 10)
- `--seed <n>`: Random seed for reproducibility (default: 42)
- `--lr <rate>`: Learning rate (default: 0.001)
- `--batch-size <size>`: Training batch size (default: 32)

#### Distillation Parameters
- `--alpha <weight>`: Ground truth loss weight (default: 0.5)
  - Controls how much the student learns from true labels
- `--beta <weight>`: Teacher loss weight (default: 0.5)
  - Controls how much the student learns from teacher predictions
  - Note: `alpha + beta = 1.0` typically

#### Stage Control
- `--training-only`: Run only training stages (no inference)
- `--inference-only`: Run only inference stages (requires `--pipeline-dir`)
- `--pipeline-dir <path>`: Path to existing pipeline directory for inference

#### Other
- `--help`: Show help message with all options

### Examples

#### Quick Test Run (1 epoch each)
```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 1 \
    --student-epochs 1 \
    --distill-epochs 1
```

#### Production Run (More Training)
```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 10 \
    --student-epochs 10 \
    --distill-epochs 20 \
    --lr 0.0005 \
    --batch-size 64
```

#### Different Distillation Weights
```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --alpha 0.3 \
    --beta 0.7 \
    --distill-epochs 15
```

#### Training Only, Then Inference Later
```bash
# Step 1: Train all models
bash scripts/run_all_patients_distillation.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --teacher-epochs 5 \
    --training-only

# Note the pipeline directory from output, e.g.:
# Created new pipeline directory: distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22

# Step 2: Run inference later (can run multiple times)
bash scripts/run_all_patients_distillation.sh \
    --inference-only \
    --pipeline-dir distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22
```

### Output Structure

The pipeline creates a timestamped directory with all results:

```
distillation_experiments/all_patients_pipeline/pipeline_YYYY-MM-DD_HH-MM-SS/
├── phase_1_teacher/
│   ├── bert_all_patients_5epochs/
│   │   ├── logs/
│   │   │   └── logs_TIMESTAMP/
│   │   │       ├── checkpoints/
│   │   │       │   └── checkpoint.pth
│   │   │       ├── *.log
│   │   │       └── *_summary.json
│   │   └── config.gin
│   └── per_patient_inference/
│       └── time_llm_per_patient_inference_ohiot1dm/
│           └── seed_42_model_BERT_dim_768_*/
│               ├── patient_540/
│               ├── patient_544/
│               └── ... (all 12 patients)
├── phase_2_student/
│   ├── bert-tiny_all_patients_5epochs/
│   │   └── ... (same structure as teacher)
│   └── per_patient_inference/
│       └── ... (per-patient results)
└── phase_3_distillation/
    ├── bert_to_bert-tiny_all_patients/
    │   └── ... (distilled model checkpoints)
    └── per_patient_inference/
        └── ... (distilled model per-patient results)
```

### Key Files

- **Checkpoints**: `phase_*/*/logs/logs_*/checkpoints/checkpoint.pth`
  - Trained model weights for inference
- **Training Logs**: `phase_*/*/logs/logs_*/*.log`
  - Detailed training progress
- **Metrics**: `phase_*/*/logs/logs_*/*_summary.json`
  - Training performance metrics (RMSE, MAE, MAPE)
- **Per-Patient Results**: `phase_*/per_patient_inference/*/patient_*/logs/`
  - Individual patient evaluation results

### Viewing Results

```bash
# Find all training logs
find distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22 -name "*.log"

# Find all summary metrics
find distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22 -name "*summary.json"

# Find all checkpoints
find distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22 -name "checkpoint.pth"

# View per-patient inference results
find distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-30-22 -path "*/per_patient_inference/*" -name "*.json"
```

## Dataset Information

### OhioT1DM Dataset
- **Total Patients**: 12 (540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596)
- **Combined Samples**: ~134,000 time series samples
- **Frequency**: 5-minute intervals
- **Task**: Blood glucose prediction for Type 1 Diabetes Management

## Model Comparison

| Model | Dimensions | Layers | Parameters | Typical Performance |
|-------|-----------|--------|-----------|-------------------|
| BERT Teacher | 768 | 12 | ~110M | RMSE: ~35-40 mg/dL |
| BERT-tiny Student (Baseline) | 128 | 2 | ~4M | RMSE: ~45-50 mg/dL |
| BERT-tiny Student (Distilled) | 128 | 2 | ~4M | RMSE: ~38-42 mg/dL |

**Size Reduction**: ~96% fewer parameters  
**Speed Improvement**: ~10-15x faster inference  
**Performance Retention**: ~85-95% of teacher performance

## Architecture Components

### Core Scripts

- **`scripts/run_all_patients_distillation.sh`**: Main pipeline orchestrator
- **`distillation/scripts/train_teachers.py`**: Teacher model training
- **`distillation/scripts/train_students.py`**: Student baseline training
- **`distillation/scripts/distill_students.py`**: Knowledge distillation training
- **`scripts/time_llm/config_generator_time_llm_unified.py`**: Inference config generation
- **`scripts/time_llm/run_all_time_llm_experiments.py`**: Batch inference runner

### Distillation Core

- **`distillation/core/distiller.py`**: Base distillation logic
- **`distillation/core/distiller_time_llm.py`**: Time-LLM specific distillation

## Troubleshooting

### Common Issues

**1. Accelerate Configuration Error**
```
ValueError: When using `deepspeed_config_file`, the following accelerate config variables will be ignored...
```
**Solution**: The accelerate config has been updated. If you still see this error:
```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```
Should reference the DeepSpeed config file without conflicting parameters.

**2. Checkpoint Not Found**
```
⚠️  No checkpoint found in phase_X_dir (skipping per-patient inference)
```
**Solution**: Training may have failed. Check training logs:
```bash
find distillation_experiments -name "*.log" | xargs grep -l "ERROR\|FAILED"
```

**3. Out of Memory**
```
CUDA out of memory
```
**Solution**: Reduce batch size:
```bash
--batch-size 16  # or even 8
```

**4. Pipeline Directory Not Found (Inference Only)**
```
❌ Error: Pipeline directory not found: ...
```
**Solution**: Verify the path exists and contains phase directories:
```bash
ls distillation_experiments/all_patients_pipeline/pipeline_*/
```

## Advanced Usage

### Custom Model Architectures

You can specify custom teacher/student models by name:

```bash
bash scripts/run_all_patients_distillation.sh \
    --teacher "microsoft/deberta-base" \
    --student "microsoft/deberta-v3-small" \
    --teacher-epochs 10 \
    --distill-epochs 20
```

### Tuning Distillation Balance

Experiment with different loss weights:

```bash
# More emphasis on ground truth
--alpha 0.7 --beta 0.3

# More emphasis on teacher knowledge
--alpha 0.3 --beta 0.7

# Balanced (default)
--alpha 0.5 --beta 0.5
```

### Running Multiple Experiments

```bash
# Experiment 1: Baseline
bash scripts/run_all_patients_distillation.sh --alpha 0.5 --beta 0.5 --distill-epochs 10 --training-only

# Experiment 2: Teacher-heavy
bash scripts/run_all_patients_distillation.sh --alpha 0.3 --beta 0.7 --distill-epochs 10 --training-only

# Experiment 3: Ground-truth heavy
bash scripts/run_all_patients_distillation.sh --alpha 0.7 --beta 0.3 --distill-epochs 10 --training-only

# Then compare results
```

## References

- **Time-LLM Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **OhioT1DM Dataset**: [Blood Glucose Level Prediction Challenge](http://smarthealth.cs.ohio.edu/bglp/bglevels.html)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in the pipeline directory
3. Run with `--help` to see all options
4. Refer to the main project README for environment setup

## License

See the main project LICENSE file for details.
