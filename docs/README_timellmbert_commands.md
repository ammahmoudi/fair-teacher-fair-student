# TimeLLM Training & Testing Command Guide

This guide provides all commands needed to generate and run TimeLLM experiments for the OhioT1DM dataset. TimeLLM supports a mode that runs training and testing together (`train_inference`), so you do not need to manage checkpoints separately.

---

## 1. Generate Train+Test Configs (1 seed)

### Train+Test on Standardized Raw
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363
```

### Train+Test on Missing Periodic
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363
```

### Train+Test on Missing Random
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363
```

### Train+Test on Noisy
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363
```

### Train+Test on Denoised
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363
```

---

## 2. Run Train+Test Experiments
```bash
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models BERT
```

---

## 3. Generate Cross-Scenario Train+Test Configs (all 5 seeds)

### Trained on Raw (standardized), Tested on:
#### Raw
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```
#### Denoised
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario denoised --train_data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```
#### Noisy
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario noisy --train_data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```
#### Missing Periodic
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_periodic --train_data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```
#### Missing Random
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_random --train_data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```

### Trained on Noisy, Tested on:
#### Noisy
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```
#### Raw
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario standardized --train_data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```

### Trained on Missing Periodic, Tested on:
#### Missing Periodic
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```

### Trained on Missing Random, Tested on:
#### Missing Random
```bash
python scripts/time_llm/config_generator.py --mode train_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --llm_models BERT --seeds 831363,809906,427368,238822,247659
```

---

## 4. Run Train+Test Experiments
```bash
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models BERT
```

---

## Advanced Options

### Run with Parallel Execution
```bash
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models BERT --parallel --max_workers 2
```

### Resume from Previous Run
```bash
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models BERT --resume
```

### Dry Run (Show Commands Without Execution)
```bash
python scripts/time_llm/run_experiments.py --modes train_inference --datasets ohiot1dm --models BERT --dry_run
```

---

## Notes
- All commands assume you are in the project root directory.
- Patient IDs: 540,544,552,559,563,567,570,575,584,588,591,596
- Seeds: 831363,809906,427368,238822,247659
- Model: BERT (llm_dim=768)
- For cross-scenario experiments, use `--train_data_scenario` to specify the training scenario.
- The script automatically generates configs for both 6_6 and 6_9 window configurations.
