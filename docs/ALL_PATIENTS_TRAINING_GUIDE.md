# All Patients Training - Updated Guide

## Overview

The distillation training scripts now support **ALL patients combined** training mode via the `--all-patients` flag.

This means you can use the existing distillation pipeline scripts with a simple flag to train on all 134,790 samples from 12 patients instead of per-patient training.

## 📁 Data Preparation

### Step 1: Create Combined Datasets

```bash
python data_processing/combine_all_patients.py
```

This creates:

- `data/ohiot1dm/all_patients_combined/all_patients_training.csv` (134,790 rows, 12 patients)
- `data/ohiot1dm/all_patients_combined/all_patients_testing.csv` (31,743 rows, 12 patients)
- `data/ohiot1dm/all_patients_combined/all_patients_complete.csv` (166,533 rows, all data)

---

## 🚀 Quick Start - All Patients Distillation

### Use the Convenience Script

The easiest way to run the full pipeline on all patients:

```bash
# Run with defaults (BERT teacher, TinyBERT student)
./scripts/run_all_patients_distillation.sh

# Customize parameters
./scripts/run_all_patients_distillation.sh \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --teacher-epochs 10 \
  --student-epochs 10 \
  --distill-epochs 15 \
  --seed 42
```

This script runs the complete 3-phase pipeline:
1. Train teacher on all patients (134K samples)
2. Train student baseline on all patients
3. Distill knowledge from teacher to student

---

## 🔧 Manual Usage - Individual Training Scripts

### Phase 1: Train Teacher on All Patients

```bash
python distillation/scripts/train_teachers.py \
  --model bert \
  --all-patients \
  --dataset ohiot1dm \
  --epochs 10 \
  --seed 42 \
  --output-dir ./results/all_patients/teacher
```

**Key Points:**
- Add `--all-patients` flag to use combined data
- Remove `--patients` parameter (it's ignored when --all-patients is set)
- Trains on `all_patients_training.csv` automatically

### Phase 2: Train Student Baseline

```bash
python distillation/scripts/train_students.py \
  --model prajjwal1/bert-tiny \
  --all-patients \
  --dataset ohiot1dm \
  --epochs 10 \
  --seed 42 \
  --output-dir ./results/all_patients/student
```

### Phase 3: Knowledge Distillation

```bash
python distillation/scripts/distill_students.py \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --all-patients \
  --dataset ohiot1dm \
  --distill-epochs 15 \
  --seed 42 \
  --alpha 0.5 \
  --beta 0.5 \
  --teacher-checkpoint-dir ./results/all_patients/teacher \
  --student-config-dir ./results/all_patients/student \
  --output-dir ./results/all_patients/distillation
```

---

## 🆚 Per-Patient vs All-Patients

### Per-Patient Mode (Traditional)

```bash
# Train on single patient
python distillation/scripts/train_teachers.py \
  --model bert \
  --patients 570 \
  --dataset ohiot1dm \
  --epochs 10
```

### All-Patients Mode (New)

```bash
# Train on ALL patients combined
python distillation/scripts/train_teachers.py \
  --model bert \
  --all-patients \
  --dataset ohiot1dm \
  --epochs 10
```

**What Changes:**
- Data path: `data/ohiot1dm/raw_standardized/570-ws-training.csv` → `data/ohiot1dm/all_patients_combined/all_patients_training.csv`
- Dataset size: ~10K samples → 134K samples
- Patient diversity: Single patient → All 12 patients
- Training time: Faster per patient → Longer overall but single run

---

## 📊 Example Workflows

### Workflow 1: Quick Test (Fast)

```bash
# Fast test with 2 epochs each phase
./scripts/run_all_patients_distillation.sh \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --teacher-epochs 2 \
  --student-epochs 2 \
  --distill-epochs 3
```

### Workflow 2: Full Training (Recommended)

```bash
# Full training with proper epochs
./scripts/run_all_patients_distillation.sh \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --teacher-epochs 10 \
  --student-epochs 10 \
  --distill-epochs 15
```

### Workflow 3: Large Models

```bash
# Use larger teacher and student
./scripts/run_all_patients_distillation.sh \
  --teacher bert-large-uncased \
  --student distilbert-base-uncased \
  --teacher-epochs 8 \
  --student-epochs 8 \
  --distill-epochs 12
```

### Workflow 4: Custom Distillation Weights

```bash
# Adjust distillation loss weights
./scripts/run_all_patients_distillation.sh \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --alpha 0.3 \
  --beta 0.7 \
  --distill-epochs 15
```

---

## 🔍 Checking Results

### View Pipeline Output

```bash
# List all pipeline runs
ls -lh distillation_experiments/all_patients_pipeline/

# Check specific run
PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_2025-10-27_16-45-30"

# View teacher results
cat $PIPELINE_DIR/phase_1_teacher/teacher_training_summary.json

# View student baseline
cat $PIPELINE_DIR/phase_2_student/student_baseline_summary.json

# View distillation results
cat $PIPELINE_DIR/phase_3_distillation/distillation_summary.json
```

### Check Logs

```bash
# Find all log files
find distillation_experiments/all_patients_pipeline/ -name "*.log"

# View latest teacher training log
tail -f distillation_experiments/all_patients_pipeline/pipeline_*/phase_1_teacher/*/logs/train.log
```

### Compare Metrics

```python
import json
import pandas as pd

# Load summary files
with open('distillation_experiments/all_patients_pipeline/pipeline_*/phase_1_teacher/teacher_training_summary.json') as f:
    teacher_metrics = json.load(f)

with open('distillation_experiments/all_patients_pipeline/pipeline_*/phase_2_student/student_baseline_summary.json') as f:
    student_baseline = json.load(f)

with open('distillation_experiments/all_patients_pipeline/pipeline_*/phase_3_distillation/distillation_summary.json') as f:
    distilled_student = json.load(f)

# Compare
print("Teacher RMSE:", teacher_metrics['performance_metrics']['rmse'])
print("Student Baseline RMSE:", student_baseline['performance_metrics']['rmse'])
print("Distilled Student RMSE:", distilled_student['performance_metrics']['rmse'])
```

---

## 🐛 Troubleshooting

### Issue: "Training data not found"

**Solution:**
```bash
python data_processing/combine_all_patients.py
```

### Issue: Out of memory

**Solution:** Reduce batch size:
```bash
./scripts/run_all_patients_distillation.sh --batch-size 16  # or 8
```

### Issue: Training takes too long

**Solution:** Reduce epochs for testing:
```bash
./scripts/run_all_patients_distillation.sh \
  --teacher-epochs 2 \
  --student-epochs 2 \
  --distill-epochs 3
```

### Issue: Teacher checkpoint not found (Phase 3)

**Solution:** Phase 1 must complete successfully. Check:
```bash
ls distillation_experiments/all_patients_pipeline/pipeline_*/phase_1_teacher/
```

---

## 🎯 Integration with Existing Pipeline

The `--all-patients` flag works seamlessly with the existing distillation pipeline:

```bash
# Original per-patient pipeline (still works)
./scripts/distill_pipeline.sh \
  --teacher bert \
  --student prajjwal1/bert-tiny \
  --patients 570,584 \
  --dataset ohiot1dm

# All-patients mode (add --all-patients to each script in your custom pipeline)
python distillation/scripts/train_teachers.py --model bert --all-patients
python distillation/scripts/train_students.py --model prajjwal1/bert-tiny --all-patients
python distillation/scripts/distill_students.py --teacher bert --student prajjwal1/bert-tiny --all-patients
```

---

## 💡 Best Practices

1. **Start Small:** Test with 2-3 epochs first to verify everything works
2. **Monitor GPU:** Watch `nvidia-smi` during training
3. **Save Checkpoints:** Automatically saved in each phase directory
4. **Batch Size:** Start with 32, reduce if OOM errors occur
5. **Epochs:** 
   - Teacher: 10-15 epochs
   - Student baseline: 10-15 epochs
   - Distillation: 15-20 epochs (more epochs = better knowledge transfer)
6. **Distillation Weights:**
   - Alpha (ground truth): 0.3-0.5
   - Beta (teacher): 0.5-0.7
   - Higher beta = more teacher influence

---

## � Related Scripts

- `data_processing/combine_all_patients.py` - Create combined datasets
- `distillation/scripts/train_teachers.py` - Train teacher models (supports --all-patients)
- `distillation/scripts/train_students.py` - Train student models (supports --all-patients)
- `distillation/scripts/distill_students.py` - Perform distillation (supports --all-patients)
- `scripts/distill_pipeline.sh` - Original per-patient pipeline
- `scripts/run_all_patients_distillation.sh` - All-patients pipeline wrapper

---

**Happy Training! 🚀**

### Step 1: Create Combined Datasets

```bash
python data_processing/combine_all_patients.py
```

This creates:
- `data/ohiot1dm/all_patients_combined/all_patients_training.csv` (134,790 rows, 12 patients)
- `data/ohiot1dm/all_patients_combined/all_patients_testing.csv` (31,743 rows, 12 patients)
- `data/ohiot1dm/all_patients_combined/all_patients_complete.csv` (166,533 rows, all data)

---

## 🚀 TIME-LLM Runner

### Basic Usage

```bash
# Train teacher model on all patients
python scripts/run_all_patients_time_llm.py --mode train --model_type teacher

# Train student model (no distillation)
python scripts/run_all_patients_time_llm.py --mode train --model_type student

# Train student with distillation from teacher
python scripts/run_all_patients_time_llm.py --mode train --model_type student --distill

# Test teacher model
python scripts/run_all_patients_time_llm.py --mode test --model_type teacher

# Test student model
python scripts/run_all_patients_time_llm.py --mode test --model_type student
```

### Full Distillation Pipeline

Runs complete pipeline: train teacher → distill to student → test both

```bash
python scripts/run_all_patients_time_llm.py --mode full_distillation
```

### Advanced Options

```bash
python scripts/run_all_patients_time_llm.py \
  --mode train \
  --model_type teacher \
  --llm_model BERT \
  --context_length 512 \
  --prediction_length 64 \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 1e-4
```

#### Parameters:

- `--mode`: Execution mode
  - `train`: Train only
  - `test`: Test only
  - `train_test`: Train then test
  - `full_distillation`: Complete pipeline (teacher → student → test both)

- `--model_type`: Model to train/test
  - `teacher`: Larger model
  - `student`: Smaller model for distillation

- `--llm_model`: LLM backbone
  - `BERT` (default)
  - `GPT2`
  - `LLAMA`
  - `DISTILBERT`

- `--context_length`: Input sequence length (default: 512)
- `--prediction_length`: Forecast horizon (default: 64)
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--distill`: Enable distillation for student training
- `--data_folder`: Data folder name (default: `all_patients_combined`)

### Output Structure

```
experiments/time_llm_all_patients_bert/
├── teacher/
│   ├── checkpoints/
│   │   └── checkpoint-final/
│   └── logs/
└── student/
    ├── checkpoints/
    │   └── checkpoint-final/
    └── logs/
```

---

## 🕐 Chronos Runner

### Basic Usage

```bash
# Train teacher model (base)
python scripts/run_all_patients_chronos.py --mode train --model_type teacher

# Train student model (small)
python scripts/run_all_patients_chronos.py --mode train --model_type student

# Test teacher model
python scripts/run_all_patients_chronos.py --mode test --model_type teacher

# Test student model
python scripts/run_all_patients_chronos.py --mode test --model_type student
```

### Full Training Pipeline

Runs complete pipeline: train teacher → train student → test both

```bash
python scripts/run_all_patients_chronos.py --mode full_distillation
```

### Advanced Options

```bash
python scripts/run_all_patients_chronos.py \
  --mode train \
  --model_type teacher \
  --teacher_model amazon/chronos-t5-base \
  --student_model amazon/chronos-t5-small \
  --context_length 512 \
  --prediction_length 64 \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 1e-3
```

#### Parameters:

- `--mode`: Execution mode
  - `train`: Train only
  - `test`: Test only
  - `train_test`: Train then test
  - `full_distillation`: Complete pipeline (train both → test both)

- `--model_type`: Model to train/test
  - `teacher`: Larger Chronos model
  - `student`: Smaller Chronos model

- `--teacher_model`: Teacher model from HuggingFace
  - `amazon/chronos-t5-tiny`
  - `amazon/chronos-t5-mini`
  - `amazon/chronos-t5-small`
  - `amazon/chronos-t5-base` (default)
  - `amazon/chronos-t5-large`

- `--student_model`: Student model from HuggingFace
  - `amazon/chronos-t5-tiny`
  - `amazon/chronos-t5-mini`
  - `amazon/chronos-t5-small` (default)

- `--context_length`: Input sequence length (default: 512)
- `--prediction_length`: Forecast horizon (default: 64)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--data_folder`: Data folder name (default: `all_patients_combined`)

### Output Structure

```
experiments/chronos_all_patients_chronos-t5-base/
├── teacher/
│   ├── checkpoints/
│   │   └── checkpoint-final/
│   └── logs/
└── student/
    ├── checkpoints/
    │   └── checkpoint-final/
    └── logs/
```

---

## 📊 Example Workflows

### Workflow 1: Quick Test (TIME-LLM)

```bash
# 1. Create combined data
python data_processing/combine_all_patients.py

# 2. Train teacher (5 epochs for quick test)
python scripts/run_all_patients_time_llm.py \
  --mode train \
  --model_type teacher \
  --epochs 5

# 3. Test teacher
python scripts/run_all_patients_time_llm.py \
  --mode test \
  --model_type teacher
```

### Workflow 2: Full Distillation (TIME-LLM)

```bash
# Complete pipeline with distillation
python scripts/run_all_patients_time_llm.py --mode full_distillation
```

This will:
1. Train teacher on all 12 patients (134K samples)
2. Train student with distillation from teacher
3. Test teacher and get metrics
4. Test student and get metrics
5. Save results for comparison

### Workflow 3: Chronos Teacher-Student

```bash
# Train both models and compare
python scripts/run_all_patients_chronos.py --mode full_distillation
```

This will:
1. Train teacher (chronos-t5-base) on all patients
2. Train student (chronos-t5-small) on all patients
3. Test both models
4. Compare performance vs. model size

### Workflow 4: Custom Configuration

```bash
# TIME-LLM with GPT2 backbone, longer context
python scripts/run_all_patients_time_llm.py \
  --mode full_distillation \
  --llm_model GPT2 \
  --context_length 1024 \
  --prediction_length 96 \
  --epochs 15 \
  --learning_rate 5e-5

# Chronos with tiny model for fast experiments
python scripts/run_all_patients_chronos.py \
  --mode train_test \
  --model_type teacher \
  --teacher_model amazon/chronos-t5-tiny \
  --epochs 5 \
  --batch_size 64
```

---

## 🆚 Per-Patient vs All-Patients

### Traditional Approach (Per-Patient)

- **Train**: Each patient's training data separately
- **Test**: Each patient's test data separately
- **Pros**: Patient-specific models, fairness analysis possible
- **Cons**: Small datasets per patient, no generalization across patients

### New Approach (All-Patients)

- **Train**: Combined data from all 12 patients (134K samples)
- **Test**: Combined test data from all 12 patients (31K samples)
- **Pros**: Larger dataset, better generalization, faster training
- **Cons**: No patient-specific fine-tuning

### When to Use Each?

**Use Per-Patient:**
- Fairness analysis across demographics
- Patient-specific personalization
- Cross-patient generalization studies

**Use All-Patients:**
- General glucose prediction model
- Distillation experiments with large data
- Faster prototyping and testing
- When patient-specific tuning not needed

---

## 🔍 Checking Results

### View Logs

```bash
# TIME-LLM logs
tail -f experiments/time_llm_all_patients_bert/teacher/logs/train.log

# Chronos logs
tail -f experiments/chronos_all_patients_chronos-t5-base/teacher/logs/train.log
```

### Check Checkpoints

```bash
# TIME-LLM checkpoints
ls -lh experiments/time_llm_all_patients_bert/teacher/checkpoints/

# Chronos checkpoints
ls -lh experiments/chronos_all_patients_chronos-t5-base/teacher/checkpoints/
```

### Extract Metrics

The runners automatically extract metrics after each experiment. Results are saved in:
- JSON reports in experiment folders
- CSV files in project root
- TensorBoard logs in logs/ folders

---

## 🐛 Troubleshooting

### Issue: "Training data not found"

**Solution:**
```bash
python data_processing/combine_all_patients.py
```

### Issue: "Teacher checkpoint not found" (for distillation)

**Solution:** Train teacher first:
```bash
python scripts/run_all_patients_time_llm.py --mode train --model_type teacher
```

### Issue: Out of memory

**Solution:** Reduce batch size:
```bash
python scripts/run_all_patients_time_llm.py \
  --mode train \
  --batch_size 8  # or 4
```

### Issue: Config file errors

**Solution:** The runners auto-generate configs. Check:
- Data paths exist
- Output directories writable
- Model names correct

---

## 📝 Next Steps

After running experiments:

1. **Compare Results:**
   - Check teacher vs student metrics
   - Analyze training curves
   - Compare efficiency (speed, memory)

2. **Fairness Analysis:**
   - Run fairness analyzers on results
   - Compare per-patient performance within combined model
   - Analyze if combining patients affects fairness

3. **Further Experiments:**
   - Try different LLM backbones
   - Adjust context/prediction lengths
   - Experiment with distillation parameters

4. **Production Deployment:**
   - Export best model
   - Create inference pipeline
   - Deploy for real-time predictions

---

## 💡 Tips

- **Start small:** Use `--epochs 2` for quick tests
- **Monitor GPU:** Watch `nvidia-smi` during training
- **Save checkpoints:** Enabled by default, finds best model automatically
- **Batch size:** TIME-LLM needs smaller batch (16), Chronos can handle larger (32+)
- **Context length:** Longer = better but slower. Start with 512.
- **Distillation:** Teacher should be trained well before distilling to student

---

## 📚 Related Scripts

- `data_processing/combine_all_patients.py` - Create combined datasets
- `scripts/time_llm/run_experiments.py` - Per-patient TIME-LLM
- `scripts/chronos/run_experiments.py` - Per-patient Chronos
- `fairness/analyzers/` - Fairness analysis tools

---

**Happy Training! 🚀**
