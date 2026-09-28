# 🧠 Knowledge Distillation Pipeline

## Overview

The Knowledge Distillation Pipeline enables you to distill knowledge from larger, more capable teacher models (like BERT-base) to smaller, more efficient student models (like TinyBERT). This technique maintains high performance while significantly reducing model size and computational requirements.

Two task-specific paths are available:

- BG/time-series forecasting through `distill_pipeline.sh` and the forecasting
  wrappers documented in the original sections below;
- MIT-BIH ECG classification through
  `scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh`, using
  `time_llm_ecg_classifier` and `distillation_ecg_classifier` in `main.py`.

The paths share backbone configuration concepts but not heads, losses,
checkpoints, datasets, or evaluation metrics.

## ECG Classification Quick Start

The production ECG pipeline supports AAMI-5 and binary modes, record-disjoint
splits, class-balanced sampling, classification KD, optional RR features, and
T1/O2 fairness variants.

```bash
PIPELINE_DIR=experiments/mitbih_example \
SEEDS=831363 \
LABEL_MODE=aami5 \
bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
```

Classification outputs include validation/test per-beat probabilities,
accuracy, macro-F1, weighted-F1, per-class recall, and subgroup fairness
reports. The locked binary protocol uses BERT -> TinyBERT, five fixed seeds,
ten epochs, N versus S/V/F with Q excluded, and previous/next RR features.

## 🚀 Quick Start

### Single Patient Pipeline
```bash
# Run complete 3-phase pipeline for one patient
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1
```

### Multi-Patient Pipeline
```bash
# Run complete 3-phase pipeline for multiple patients
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570,584 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1
```

### Dry Run (Check Configuration)
```bash
# Preview what the pipeline will do without running training
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570,584 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1 \
  --dry-run
```

## 📋 Pipeline Phases

The distillation pipeline consists of three phases that run sequentially for each patient:

### Phase 1: Teacher Training 🎓
- Trains the large teacher model (e.g., BERT-base-uncased)
- Saves trained model checkpoints for knowledge transfer
- Generates comprehensive performance metrics (RMSE, MAE, MAPE)
- Creates training summaries and efficiency reports

### Phase 2: Student Baseline 👨‍🎓
- Trains the smaller student model independently (e.g., prajjwal1/bert-tiny)
- Establishes baseline performance without teacher knowledge
- Saves student model checkpoints and configuration
- Records baseline metrics for comparison

### Phase 3: Knowledge Distillation 🧠
- Transfers knowledge from trained teacher to student model
- Uses distillation parameters (alpha, beta, temperature, KL weight)
- Combines teacher predictions with ground truth for student training
- Produces final distilled model with improved performance

## 🔧 Parameters

### Required Parameters
- `--teacher`: Teacher model name (e.g., bert-base-uncased)
- `--student`: Student model name (e.g., prajjwal1/bert-tiny)
- `--patients`: Patient IDs (comma-separated for multiple patients)
- `--dataset`: Dataset name (ohiot1dm, d1namo)
- `--seed`: Random seed for reproducibility
- `--teacher-epochs`: Number of epochs for teacher training
- `--student-epochs`: Number of epochs for student baseline training
- `--distill-epochs`: Number of epochs for knowledge distillation

### Optional Parameters
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--alpha`: Distillation loss weight (default: 0.5)
- `--beta`: Ground truth loss weight (default: 0.5)
- `--kl-weight`: KL divergence loss weight (default: 0.1)
- `--temperature`: Softmax temperature for distillation (default: 3.0)
- `--dry-run`: Preview mode without actual training

## 🎯 Supported Models

### Teacher Models (HuggingFace Compatible)
- `bert-base-uncased` - Standard BERT base model
- `bert-large-uncased` - Large BERT model
- `distilbert-base-uncased` - DistilBERT model
- `roberta-base` - RoBERTa base model

### Student Models (HuggingFace Compatible)
- `prajjwal1/bert-tiny` - 2-layer, 128-hidden BERT
- `prajjwal1/bert-mini` - 4-layer, 256-hidden BERT  
- `prajjwal1/bert-small` - 4-layer, 512-hidden BERT
- `prajjwal1/bert-medium` - 8-layer, 512-hidden BERT

### Auto-Detection
The pipeline automatically handles HuggingFace model names and applies appropriate configurations. Model names with forward slashes are sanitized for safe file operations.

## 🗂️ Output Structure

For each pipeline run, the following directory structure is created:

```
distillation_experiments/
├── pipeline_runs/
│   └── pipeline_2025-10-16_20-45-27/          # Timestamped pipeline run
│       ├── patient_570/                        # Individual patient results
│       │   ├── phase_1_teacher/                # Teacher training outputs
│       │   │   ├── config_teacher_*.gin        # Teacher configuration
│       │   │   ├── teacher_training_summary.json
│       │   │   └── bert_570_1epochs/           # Model logs and checkpoints
│       │   ├── phase_2_student/                # Student baseline outputs
│       │   │   ├── config_student_*.gin        # Student configuration
│       │   │   ├── student_baseline_summary.json
│       │   │   └── tinybert_570_1epochs/       # Model logs and checkpoints
│       │   └── phase_3_distillation/           # Distillation outputs
│       │       ├── config_distill_*.gin        # Distillation configuration
│       │       ├── distillation_summary.json  # Final metrics
│       │       └── bert_to_bert_tiny_570/      # Distilled model and logs
│       └── patient_584/                        # Additional patients
├── pipeline_results.csv                        # Comprehensive CSV results
└── pipeline_csv_logger.py                     # CSV logging utility
```

## 📊 Results & Metrics

### Automatic CSV Logging
After each patient's complete pipeline run, results are automatically logged to `distillation_experiments/pipeline_results.csv` with the following information:

- **Pipeline Configuration**: Teacher/student models, hyperparameters, patient IDs
- **Phase 1 Metrics**: Teacher RMSE, MAE, MAPE performance
- **Phase 2 Metrics**: Student baseline RMSE, MAE, MAPE performance  
- **Phase 3 Metrics**: Distilled model RMSE, MAE, MAPE performance
- **Improvement Analysis**: Performance improvements vs teacher and baseline
- **Runtime Information**: Total pipeline execution time
- **Status Tracking**: Success/failure status for each phase

### Example Results
```csv
timestamp,patient_ids,teacher_rmse,student_baseline_rmse,distilled_rmse,teacher_to_distilled_improvement_pct
2025-10-16 20:51:59,570,17.234,17.418,16.463,4.47%
2025-10-16 20:57:50,584,26.608,26.822,26.430,0.67%
```

### Performance Expectations
- **Successful Distillation**: Distilled model typically achieves 3-8% RMSE improvement over teacher
- **Model Compression**: 80-95% reduction in model parameters (e.g., BERT-base 110M → TinyBERT 4.4M)
- **Speed Improvement**: 5-15x faster inference compared to teacher model
- **Memory Reduction**: 70-90% reduction in memory usage

## 🔬 Multi-Patient Execution

When multiple patients are specified (e.g., `--patients 570,584`), the pipeline:

1. **Sequential Processing**: Runs all 3 phases for patient 570, then all 3 phases for patient 584
2. **Individual Logging**: Each patient's results are logged to CSV immediately after completion
3. **Isolated Execution**: Each patient's training is completely independent
4. **Organized Storage**: Each patient gets their own directory structure
5. **Progress Tracking**: Clear progress indicators show which patient is being processed

This approach ensures:
- ✅ **Fault Tolerance**: If one patient fails, others can still complete
- ✅ **Real-time Results**: CSV is updated after each patient completion
- ✅ **Resource Management**: Memory is freed between patients
- ✅ **Easy Analysis**: Individual patient results are clearly separated

## 📊 Available Datasets

### OhioT1DM (Recommended for Research)
- **Patients**: 540, 559, 563, 570, 575, 588, 591, 596
- **Use Cases**: Research experiments, production training
- **Characteristics**: Larger datasets, longer training times, more robust results

### D1NAMO (Good for Testing)
- **Patients**: 001, 002, 003, 004, 005, 006, 007
- **Use Cases**: Quick testing, development, proof of concept
- **Characteristics**: Smaller datasets, faster training, good for validation

### Data Types
- `raw_standardized`: Standard preprocessed data (recommended for most use cases)
- `noisy_standardized`: Data with added noise for robustness testing
- `missing_data_*`: Data with simulated missing values for imputation research

## 🎛️ Advanced Usage

### Custom Hyperparameters
```bash
# Fine-tune distillation parameters
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570,584 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 3 \
  --student-epochs 2 \
  --distill-epochs 5 \
  --lr 0.0001 \
  --batch-size 16 \
  --alpha 0.7 \
  --beta 0.3 \
  --temperature 5.0
```

### Different Model Combinations
```bash
# BERT-base → BERT-mini
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-mini \
  --patients 570 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 2 \
  --student-epochs 2 \
  --distill-epochs 3

# DistilBERT → TinyBERT  
bash distill_pipeline.sh \
  --teacher distilbert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 584 \
  --dataset d1namo \
  --seed 238822 \
  --teacher-epochs 2 \
  --student-epochs 2 \
  --distill-epochs 3
```

### Development vs Production Workflows

#### Development Workflow
```bash
# 1. Start with small dataset and single epoch
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 001 \
  --dataset d1namo \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1 \
  --dry-run

# 2. Test with real training
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 001 \
  --dataset d1namo \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1
```

#### Production Workflow
```bash
# High-quality training with more epochs and multiple patients
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570,575,588,591 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 10 \
  --student-epochs 8 \
  --distill-epochs 5 \
  --lr 0.0001 \
  --batch-size 32
```

## 🐛 Troubleshooting

### Common Issues

#### Pipeline Fails to Start
```bash
# Check if virtual environment is activated
source venv/bin/activate

# Verify dependencies are installed
pip install -r requirements.txt

# Check GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Model Loading Errors
```bash
# For HuggingFace model access issues
pip install --upgrade transformers
pip install --upgrade torch

# For model name issues (forward slashes)
# The pipeline automatically sanitizes model names, but verify the model exists:
# https://huggingface.co/prajjwal1/bert-tiny
```

#### Memory Issues
```bash
# Reduce batch size for limited GPU memory
bash distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --batch-size 8

# Use CPU if GPU memory is insufficient (slower but works)
# Edit configs to set device: 'cpu' instead of 'cuda'
```

#### CSV Logging Issues
```bash
# Check permissions on distillation_experiments directory
ls -la distillation_experiments/

# Manually run CSV logger to test
python distillation/scripts/pipeline_csv_logger.py --help
```

### Performance Optimization

#### GPU Optimization
- Use batch sizes that maximize GPU utilization (16, 32, 64)
- Monitor GPU memory usage during training
- Consider mixed precision training for larger models

#### Training Speed
- Start with fewer epochs (1-2) for quick validation
- Use smaller student models (bert-tiny) for faster training
- Enable gradient checkpointing for memory efficiency

## 📈 Best Practices

### Experiment Design
1. **Start Small**: Test with single patient and 1 epoch first
2. **Validate Setup**: Use `--dry-run` to check configurations
3. **Monitor Progress**: Watch CSV updates to track success
4. **Save Results**: Keep CSV files for analysis and comparison

### Model Selection
1. **Teacher Size**: Larger teachers provide better knowledge but train slower
2. **Student Size**: Choose based on deployment constraints
3. **Architecture Match**: BERT family models work well together
4. **Domain Relevance**: Pre-trained models should match your domain

### Hyperparameter Tuning
1. **Temperature**: Start with 3-5, higher for softer distributions
2. **Alpha/Beta**: Balance distillation vs ground truth (0.5/0.5 baseline)
3. **Learning Rate**: Often lower than standard training (0.0001-0.001)
4. **Epochs**: Teacher > Student ≥ Distillation for best results

## 🔍 Monitoring & Analysis

### Real-time Monitoring
```bash
# Watch CSV file updates
tail -f distillation_experiments/pipeline_results.csv

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check pipeline logs
tail -f distillation_experiments/pipeline_runs/*/patient_*/phase_*/*/logs/*/main.log
```

### Post-Training Analysis
```bash
# Load and analyze CSV results
python -c "
import pandas as pd
df = pd.read_csv('distillation_experiments/pipeline_results.csv')
print(df[['patient_ids', 'teacher_rmse', 'distilled_rmse', 'teacher_to_distilled_improvement_pct']])
"

# Compare multiple experiments
python scripts/analyze_distillation_results.py
```

## 🚀 What's Next?

After successful knowledge distillation:

1. **Deploy Models**: Use distilled models in production for faster inference
2. **Further Compression**: Apply quantization or pruning techniques
3. **Ensemble Methods**: Combine multiple distilled models
4. **Transfer Learning**: Use distilled models as base for new tasks
5. **Continuous Learning**: Retrain with new data while maintaining efficiency

---

## 🎓 Getting Started Checklist

Before running your first distillation experiment:

- [ ] **Environment Setup**
  - [ ] Clone repository: `git clone --recursive https://github.com/ammahmoudi/fair-teacher-fair-student.git`
  - [ ] Create virtual environment: `python3 -m venv venv`
  - [ ] Activate environment: `source venv/bin/activate`
  - [ ] Install dependencies: `pip install -r requirements.txt`

- [ ] **Initial Testing**
  - [ ] Test pipeline setup: `bash distill_pipeline.sh --help`
  - [ ] Run dry run: `bash distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --dry-run`
  - [ ] Check GPU availability: `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"`

- [ ] **First Experiment**
  - [ ] Start with small dataset (D1NAMO patient 001)
  - [ ] Use 1 epoch for each phase
  - [ ] Verify results in `distillation_experiments/`
  - [ ] Check CSV logging works correctly

- [ ] **Scale Up**
  - [ ] Move to OhioT1DM dataset for production experiments
  - [ ] Increase epochs for better performance
  - [ ] Test multi-patient execution
  - [ ] Analyze performance improvements in CSV results

For more details on the underlying models and training procedures, see:
- [Time-LLM Documentation](README_timellm_commands.md)
- [Main Project README](../README.md)
- [Chronos Model Documentation](README_chronos_commands.md)

## 🚀 Quick Start

### Basic Usage
```bash
# Simple distillation with default parameters
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --seed 42

# Custom training configuration
./distill_pipeline.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570,584 \
    --dataset ohiot1dm \
    --seed 42 \
    --teacher-epochs 5 \
    --student-epochs 3 \
    --distill-epochs 3 \
    --lr 0.002 \
    --batch-size 16 \
    --alpha 0.6 \
    --beta 0.4 \
    --kl-weight 0.15 \
    --temperature 4.0

# Dry run (preview commands without execution)
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --dry-run
```

## 📋 Parameters

### Required Parameters
- `--teacher`: Teacher model name (see supported models below)
- `--student`: Student model name (see supported models below)
- `--patients`: Patient IDs (comma-separated or single, e.g., "570" or "570,584")
- `--dataset`: Dataset name ("ohiot1dm" or "d1namo")

### Optional Parameters

#### Training Configuration
- `--seed`: Random seed for reproducibility (default: 238822)
- `--teacher-epochs`: Teacher training epochs (default: 1)
- `--student-epochs`: Student baseline epochs (default: 1)
- `--distill-epochs`: Knowledge distillation epochs (default: 1)

#### Optimization Parameters
- `--lr`: Learning rate for all phases (default: 0.001)
- `--batch-size`: Batch size for all phases (default: 32)

#### Distillation Hyperparameters
- `--alpha`: Weight for ground truth loss (student learning from actual labels, default: 0.5)
- `--beta`: Weight for teacher output loss (student learning from teacher predictions, default: 0.5)
- `--kl-weight`: Weight for KL divergence loss (knowledge transfer strength, default: 0.1)
- `--temperature`: Temperature for knowledge distillation softmax (controls soft target smoothness, default: 3.0)

#### Utility
- `--dry-run`: Preview commands without execution

## 🤖 Supported Models

### Teacher Models (Large, High-Capacity)
- `bert` / `bert-base-uncased` / `bert-base-cased`
- `bert-large-uncased` / `bert-large-cased`
- `distilbert` / `distilbert-base-uncased` / `distilbert-base-cased`
- `tinybert`

### Student Models (Small, Efficient)
- `tinybert`
- `distilbert` / `distilbert-base-uncased` / `distilbert-base-cased`
- `prajjwal1/bert-tiny` (2 layers, 128 dim)
- `prajjwal1/bert-mini` (4 layers, 256 dim)
- `prajjwal1/bert-small` (4 layers, 512 dim)
- `prajjwal1/bert-medium` (8 layers, 512 dim)

## 📁 Output Structure

The pipeline creates organized output directories:
```
distillation_experiments/pipeline_runs/pipeline_YYYY-MM-DD_HH-MM-SS/
├── phase_1_teacher/          # Teacher model training
│   ├── config_teacher_*.gin   # Training configuration
│   ├── bert-base-uncased_570_1epochs/  # Model outputs
│   └── logs/                  # Training logs
├── phase_2_student/           # Student baseline training
│   ├── config_student_*.gin   # Training configuration  
│   ├── prajjwal1-bert-tiny_570_1epochs/  # Model outputs
│   └── logs/                  # Training logs
└── phase_3_distillation/      # Knowledge distillation
    ├── config_distill_*.gin   # Distillation configuration
    ├── distilled_models/      # Final distilled models
    └── logs/                  # Distillation logs
```

## 🎛️ Individual Script Usage

### 1. Teacher Training
```bash
python distillation/scripts/train_teachers.py \
    --model bert-base-uncased \
    --patients 570 \
    --dataset ohiot1dm \
    --epochs 5 \
    --lr 0.001 \
    --batch-size 32 \
    --seed 42
```

### 2. Student Baseline Training
```bash
python distillation/scripts/train_students.py \
    --model prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --epochs 3 \
    --lr 0.001 \
    --seed 42
```

### 3. Knowledge Distillation
```bash
python distillation/scripts/distill_students.py \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --distill-epochs 3 \
    --lr 0.001 \
    --batch-size 32 \
    --alpha 0.5 \
    --beta 0.5 \
    --kl-weight 0.1 \
    --temperature 3.0 \
    --seed 42
```

## 📊 Datasets

### Supported Datasets
- **ohiot1dm**: Ohio T1DM dataset for glucose forecasting
- **d1namo**: D1NAMO dataset for continuous glucose monitoring
- **mitbih**: MIT-BIH beat classification through the separate ECG pipeline;
  AAMI-5 is primary and binary ectopy is the locked secondary endpoint

### Data Path Auto-Detection
The system automatically detects data paths:
- `ohiot1dm` → `./data/ohiot1dm/raw_standardized/`
- `d1namo` → `./data/d1nemo/raw_standardized/`

Expected data structure:
```
data/
├── ohiot1dm/
│   └── raw_standardized/
│       ├── 570-ws-training.csv
│       ├── 570-ws-testing.csv
│       ├── 584-ws-training.csv
│       ├── 584-ws-testing.csv
│       └── t1dm_prompt.txt
└── d1namo/
    └── raw_standardized/
        ├── [patient files]
        └── t1dm_prompt.txt
```

## 🔧 Configuration

### Model Architecture Mapping
The system maps HuggingFace model names to internal configurations:

| Model Name | Internal Config | Layers | Hidden Size |
|------------|----------------|---------|-------------|
| bert-base-uncased | BERT | 12 | 768 |
| bert-large-uncased | BERT | 24 | 1024 |
| distilbert-base-uncased | DistilBERT | 6 | 768 |
| prajjwal1/bert-tiny | BERT | 2 | 128 |
| prajjwal1/bert-mini | BERT | 4 | 256 |
| prajjwal1/bert-small | BERT | 4 | 512 |
| prajjwal1/bert-medium | BERT | 8 | 512 |

### Distillation Loss Components

The knowledge distillation loss combines multiple components:

```
Total Loss = α × Ground_Truth_Loss + β × Teacher_Output_Loss + λ × KL_Divergence_Loss
```

- **Ground Truth Loss (α)**: Student learns from actual labels
- **Teacher Output Loss (β)**: Student mimics teacher predictions  
- **KL Divergence Loss (λ)**: Knowledge transfer via soft targets with temperature scaling

## 📈 Performance Tips

### Hyperparameter Tuning
- **Higher α**: More focus on task accuracy
- **Higher β**: More knowledge transfer from teacher
- **Higher temperature**: Softer probability distributions
- **Higher KL weight**: Stronger knowledge distillation effect

### Computational Efficiency
- Use smaller batch sizes for GPU memory constraints
- Start with fewer epochs and scale up based on convergence
- Use `--dry-run` to verify configurations before training

## 🐛 Troubleshooting

### Common Issues
1. **Model not found**: Ensure model names match supported choices exactly
2. **Data path errors**: Verify data files exist in expected structure
3. **Memory errors**: Reduce batch size or use smaller models
4. **Permission errors**: Ensure write access to output directories

### Debug Commands
```bash
# Check available models
python distillation/scripts/train_teachers.py --model all --list-checkpoints

# Verify data paths with dry run
./distill_pipeline.sh --teacher bert --student tinybert --patients 570 --dataset ohiot1dm --dry-run

# Test individual components
python distillation/scripts/train_teachers.py --model bert --patients 570 --dataset ohiot1dm --dry-run
```

## 📝 Examples

### Example 1: Quick Distillation
```bash
# Fast distillation for testing
./distill_pipeline.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --teacher-epochs 1 \
    --student-epochs 1 \
    --distill-epochs 1
```

### Example 2: High-Quality Distillation
```bash
# Comprehensive distillation with multiple patients
./distill_pipeline.sh \
    --teacher bert-large-uncased \
    --student prajjwal1/bert-small \
    --patients 570,584 \
    --dataset ohiot1dm \
    --teacher-epochs 10 \
    --student-epochs 5 \
    --distill-epochs 8 \
    --lr 0.0005 \
    --batch-size 64 \
    --alpha 0.3 \
    --beta 0.7 \
    --kl-weight 0.2 \
    --temperature 5.0
```

### Example 3: Multi-Dataset Experiments
```bash
# Train on ohiot1dm
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --seed 42

# Train on d1namo  
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset d1namo --seed 42
```

## 🏗️ Architecture

### Pipeline Flow
```
Input Data → Teacher Training → Student Baseline → Knowledge Distillation → Compressed Model
     ↓              ↓                ↓                     ↓                      ↓
   Raw CSV    Large Model      Small Model        Distilled Model         Efficient Model
   Patients     Training        Training           (Teacher→Student)      Ready for Deployment
```

### File Organization
```
distillation/
├── scripts/
│   ├── train_teachers.py      # Teacher model training
│   ├── train_students.py      # Student baseline training
│   └── distill_students.py    # Knowledge distillation
├── __init__.py
└── utils/                     # Utility functions
```

---

**Note**: The `distill_pipeline.sh` commands in this document remain specific
to temporal forecasting. MIT-BIH classification must use the ECG generators or
pipeline because it requires beat-index data, class labels, a classification
head, cross-entropy/KL losses, and classification/fairness metrics.