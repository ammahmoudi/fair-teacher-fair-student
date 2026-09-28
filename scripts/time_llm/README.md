# Time-LLM Scripts

Organized collection of scripts for Time-LLM experiments.

## Directory Structure

```
time_llm/
├── README.md                                  # This file
├── TIME_LLM_USAGE_GUIDE.md                    # Comprehensive usage documentation
│
├── config_generator.py                        # 🔧 MAIN: Config generator
├── run_experiments.py                         # 🚀 MAIN: Experiment runner
│
└── helpers/                                   # Batch/convenience scripts
    ├── batch_generate_all_configs.py          # Generate all configs at once
    ├── examples_quick_commands.sh             # Quick command examples
    └── examples_common_configs.sh             # Common config examples
```

## Main Scripts (Use These)

### Config Generator - `config_generator.py`

**The main unified configuration generator for all Time-LLM experiments.**

```bash
# Generate training configs
python config_generator.py --mode train

# Generate inference configs
python config_generator.py --mode inference

# With specific parameters
python config_generator.py --mode train --patients 570,584 --models bert-base-uncased
```

**Supported Models:**
- `bert-base-uncased` (BERT - 768 dim)
- `distilbert-base-uncased` (DistilBERT - 768 dim)
- `huawei-noah/TinyBERT_General_4L_312D` (TinyBERT - 312 dim)
- `prajjwal1/bert-tiny` (BERT-tiny - 128 dim)
- `microsoft/MiniLM-L12-H384-A12` (MiniLM - 384 dim)
- `google/mobilebert-uncased` (MobileBERT - 512 dim)
- `albert-base-v2` (ALBERT - 768 dim)
- `gpt2` (GPT2 - 768 dim)
- `facebook/opt-125m` (OPT - 768 dim)

### Experiment Runner - `run_experiments.py`

**Automatically discover and run all Time-LLM experiments.**

```bash
# Run all experiments
python run_experiments.py

# Run with parallel processing
python run_experiments.py --parallel --max_workers 4

# Filter by mode
python run_experiments.py --modes training,inference

# Resume interrupted experiments
python run_experiments.py --resume

# Dry run
python run_experiments.py --dry_run
```

## Helper Scripts (Convenience Shortcuts)

### 1. Batch Generate All Configs - `helpers/batch_generate_all_configs.py`

**Generate all Time-LLM configs across datasets and scenarios.**

```bash
python helpers/batch_generate_all_configs.py
```

Generates:
- All training configs (all datasets, scenarios, patients, models)
- All inference configs
- Cross-scenario configs

### 2. Examples Scripts

**Quick commands script:**

```bash
bash helpers/examples_quick_commands.sh
```

**Common configs script:**

```bash
bash helpers/examples_common_configs.sh
```

## Quick Start Workflow

### Step 1: Generate Training Configs

```bash
# Option A: Generate specific configs
python config_generator.py --mode train --patients 570,584 --models bert-base-uncased

# Option B: Generate all configs
python helpers/batch_generate_all_configs.py
```

### Step 2: Run Training Experiments

```bash
# Run all training experiments
python run_experiments.py --modes training

# Or with parallel processing
python run_experiments.py --modes training --parallel --max_workers 4
```

### Step 3: Run Inference Experiments

```bash
# Run all inference experiments
python run_experiments.py --modes inference
```

## Configuration Parameters

### Datasets
- **OhioT1DM**: Diabetes blood glucose dataset
- **D1NAMO**: Extended diabetes dataset

### Data Scenarios
- `standardized` - Clean, standardized data (default)
- `noisy` - Data with added noise
- `denoised` - Noise-reduced data
- `missing_periodic` - Periodic missing values
- `missing_random` - Random missing values

### Prediction Horizons
- Context windows: 6, 9 timesteps
- Prediction horizons: 6, 9 timesteps
- Common combinations: 6_6, 6_9, 9_9

## Output Locations

- **Generated Configs**: `experiments/time_llm_*/`
- **Training Results**: `experiments/time_llm_training_*/logs/`
- **Inference Results**: `experiments/time_llm_inference_*/logs/`
- **Checkpoints**: Stored in experiment directories

## Tips

1. **Start Small**: Test with 1-2 patients and 1 model first
   ```bash
   python config_generator.py --mode train --patients 570 --models bert-base-uncased
   ```

2. **Use Appropriate Models**: Match model size to your resources
   - Fast testing: `prajjwal1/bert-tiny` (128 dim)
   - Balanced: `huawei-noah/TinyBERT_General_4L_312D` (312 dim)
   - Best performance: `bert-base-uncased` (768 dim)

3. **Parallel Processing**: Speed up experiments
   ```bash
   python run_experiments.py --parallel --max_workers 8
   ```

4. **Resume After Interruption**:
   ```bash
   python run_experiments.py --resume
   ```

## Comparison: Legacy vs Current Generator

The unified `config_generator.py` is the main and only recommended tool. Legacy generators have been removed for simplicity.

**Features:**
- Multiple modes supported (train, inference, train_inference)
- Flexible parameters with good defaults
- Better for batch generation
- Comprehensive model support (10 LLMs)
- Cross-scenario evaluation support

## For More Details

See `TIME_LLM_USAGE_GUIDE.md` for comprehensive documentation including:
- Detailed parameter explanations
- Model comparison and selection guide
- Advanced configuration options
- Troubleshooting guides
