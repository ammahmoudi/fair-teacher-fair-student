# MIT-BIH Time-LLM Classification Implementation Plan

## Goal

Turn the current repository into a working **Time-LLM AAMI 5-class ECG classification + distillation + fairness** pipeline.

## Completion Update — 2026-07-15

The internal implementation milestone and both locked five-seed ECG runs are
complete. AAMI-5 remains the primary classification benchmark; binary ectopy
(N versus S/V/F with Q excluded) is a reportable secondary stress test with RR
features. Results are indexed in:

- `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/MULTISEED_ANALYSIS.md`;
- `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`;
- `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`.

The external CNN comparator is still future work. Binary locked-test results
must not be used for post-hoc model selection because validation averages do
not show the same ranking.

## Implementation status on 2026-07-11

The internal ECG path is implemented end to end:

- MIT-BIH metadata, demographics, AAMI mapping, beat windows, and record-safe
  train/validation/test splits;
- Time-LLM ECG teacher/student classification through `main.py`;
- classification KD and the T1, O1, O2, K1, K3, K4, O3, and T2 fairness
  adaptations;
- per-beat validation/test probabilities and subgroup metadata;
- efficiency, classification, and fairness reports;
- resumable single-seed, seed-list, and all-fixed-seed pipeline execution.

The external CNN comparator remains future comparison work. It must use this
project's record-level splits before its results are considered comparable.

The corrected imbalance policy is mutually exclusive:

- weighted sampler active: ordinary cross-entropy;
- ordinary shuffled loader: inverse-frequency weighted cross-entropy.

Combining the sampler and inverse class weights caused a real BERT run to
collapse to class `F`; the pipeline now logs the selected objective and rejects
a teacher before distillation if macro-F1 or prediction diversity is too low.

The ECG model now also matches the BG Time-LLM freezing policy. The pretrained
LLM backbone is frozen by default. Training updates the ECG patch embedding,
continuous-input projection, layer normalization, and five-class classifier.
For full BERT this is `58,373` trainable task parameters and zero trainable
backbone parameters. `FREEZE_LLM=0` is an explicit ablation; it uses task LR
`1e-4` and backbone LR `1e-5` rather than applying the task LR to BERT.

Frozen ECG checkpoints store only those trainable task modules. The pretrained
backbone is reconstructed from the same Hugging Face model when loading. This
reduces full-BERT checkpoints from about `438 MB` to about `237 KB` and remains
backward-compatible with earlier full state dictionaries.

## Non-breaking implementation policy

The current BG forecasting system must remain usable while this ECG work is added.

So implementation should follow these rules:

1. **Add new files instead of rewriting BG pipeline files when possible.**
2. **Keep ECG logic under separate MIT-BIH-specific modules, runners, and configs.**
3. **Make classification mode opt-in, not the default path.**
4. **Do not break existing Time-LLM BG forecasting commands, configs, or distillation flows.**

### Recommended separation strategy

Use new or clearly separated files such as:

- `data_processing/ecg/*`
- `scripts/mitbih/*`
- `llms/time_llm_ecg.py`
- `models/ecg/*`
- `distillation/scripts/distill_mitbih_*`
- `fairness/analyzers/ecg_*`

If an existing shared module must be changed, prefer:

- adding a new `task_type` or `mode` flag
- preserving old defaults exactly
- isolating ECG classification paths behind explicit configuration

### Practical rule

The BG system should still work exactly as before if the user never touches the new MIT-BIH files or config modes.

This plan is implementation-focused. It assumes the research decisions are already made:

- Main task = MIT-BIH beat-centered AAMI 5-class classification
- Main architecture = Time-LLM-inspired classifier
- Secondary architecture = encoder + classifier head as fallback/ablation
- External comparator = friend’s ECG CNN classifier

---

## 1. Implementation targets

We need four connected subsystems:

1. **ECG data layer**
2. **Time-LLM classification model variant**
3. **Classification distillation pipeline**
4. **Fairness analysis for multi-class ECG outputs**

---

## 2. Data layer tasks

### 2.1 Metadata parsing

Create:

- `data_processing/ecg/metadata.py`

Responsibilities:

- Read `.hea` files
- Extract record ID, age, sex, leads, notes
- Create metadata table

Output:

- `data/mit-bih-arrhythmia/metadata_records.csv`

### 2.2 Label mapping

Create:

- `data_processing/ecg/label_map.py`

Responsibilities:

- Read raw MIT-BIH symbols
- Map to AAMI 5 classes:
  - `N`
  - `S`
  - `V`
  - `F`
  - `Q`
- Define exclusions and unknown-handling clearly

### 2.3 Beat-window dataset builder

Create:

- `data_processing/ecg/dataset.py`

Responsibilities:

- Load ECG waveform
- Choose primary lead, prefer `MLII`
- Normalize per record
- Extract beat-centered windows
- Attach class label and subgroup metadata
- Support record-level split
- Exclude duplicate subject record `202` by default

Expected sample output:

```python
{
    "record_id": ...,
    "beat_index": ...,
    "x": signal_window,
    "y": class_id,
    "sex": ...,
    "age_group": ...,
    "paced_group": ...,
    "difficulty_group": ...
}
```

### 2.4 Split policy

Implement record-level split builder:

- Train / val / test
- Seed-controlled
- No beat-level leakage
- Operate on the curated one-record-per-patient set, excluding `202`

Optional output:

- `beat_index.csv`

Example build commands:

```bash
# Curated fairness mode: 47 records
python scripts/mitbih/build_metadata.py
python scripts/mitbih/build_demographics.py
python scripts/mitbih/prepare_beat_dataset.py

# Full dataset mode: 48 records
python scripts/mitbih/build_metadata.py --include-duplicate-202
python scripts/mitbih/build_demographics.py --include-duplicate-202
python scripts/mitbih/prepare_beat_dataset.py --include-duplicate-202
```

---

## 3. Time-LLM classification model tasks

### 3.1 Add classification mode

Wherever the current Time-LLM code defines forecasting head behavior, add a classification mode.

Core change:

```text
forecast head -> classification head
```

### 3.2 Add classification head

Needed behavior:

- Input = hidden sequence output from encoder/backbone
- Pooling = mean pooling or center-aware pooling
- Head = linear logits for 5 classes

Recommended first version:

- Mean pooling
- Linear layer to 5 logits

### 3.3 Support fallback ablation path

Allow optional mode:

- Backbone frozen or semi-frozen
- Simple encoder + classifier head variant

This is for comparison and debugging.

### 3.4 Model output contract

The model should return:

- Logits
- Probabilities
- Predicted class

Not forecast windows.

---

## 4. Training pipeline tasks

### 4.1 Standard classifier training runner

Create:

- `main.py` using `llm_settings["method"] == "time_llm_ecg_classifier"`

Responsibilities:

- Load ECG dataset
- Build model in classification mode
- Train teacher or student variant
- Save checkpoint
- Save validation/test predictions

### 4.2 Config support

Need configuration options for:

- Window size
- Lead selection policy
- Number of classes
- Pooling mode
- Teacher/student model size
- Learning rate
- Class weighting

### 4.3 Loss

Main loss:

- Multi-class cross-entropy

Imbalance handling:

- Main supervised protocol: capped class-balanced sampler with ordinary
  cross-entropy.
- Fallback without a sampler: inverse-frequency weighted cross-entropy.
- Do not enable both corrections for the same training loader.

### 4.4 Metrics during training

Track:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class recall
- Confusion matrix summary

---

## 5. Distillation pipeline tasks

### 5.1 Add classification distillation mode

Extend the distillation pipeline to support:

- Classification teacher logits
- Classification student logits
- Classification distillation loss

### 5.2 Distillation runner

Create:

- `distillation/scripts/distill_mitbih_classifier.py`

Responsibilities:

- Load teacher checkpoint
- Load student config
- Compute supervised classification loss
- Compute distillation loss from teacher outputs
- Train distilled student

### 5.3 Distillation comparisons

We need three core systems:

- Teacher baseline
- Student baseline
- Distilled student

Optional later:

- Fairness-mitigated distilled student

---

## 6. External comparator tasks

### 6.1 Bring in friend baseline

Integrate friend’s model as external baseline.

Possible structure:

- `baselines/ecg_classifier_llm/`
- Or documented cloned repo location

### 6.2 Standardize evaluation

Make sure friend baseline uses:

- Our data split
- Our label mapping
- Our subgroup metadata
- Our fairness evaluation scripts

This is critical. Otherwise comparison is not meaningful.

---

## 7. Fairness analysis tasks

### 7.1 Build ECG classifier fairness analyzer

Create:

- `fairness/analyzers/ecg_classifier_fairness_analyzer.py`

Responsibilities:

- Load predictions and labels
- Compute subgroup performance
- Compute multi-class fairness summaries
- Compute one-vs-rest fairness for each class

### 7.2 Required fairness outputs

For each group:

- Support count
- Macro-F1
- Per-class recall
- TPR gaps
- EO Gap
- DP Gap
- FVO

### 7.3 Distillation comparison

Need fairness reports for:

- Teacher
- Student
- Distilled student
- External CNN baseline

---

## 8. Fairness mitigation tasks

After the core pipeline works, add mitigation experiments.

### Candidate transfers from BG work

1. Subgroup-aware weighting
2. Oversampling / replay ideas
3. Fairness-regularized distillation loss
4. Output-level calibration for classification probabilities
5. Per-class subgroup calibration studies

---

## 9. File creation order

### First batch

1. `data_processing/ecg/metadata.py`
2. `data_processing/ecg/label_map.py`
3. `data_processing/ecg/dataset.py`

### Second batch

1. Time-LLM classification head adaptation
2. `main.py` using `llm_settings["method"] == "time_llm_ecg_classifier"`
3. `scripts/time_llm/config_generator_mitbih.py`
4. existing `scripts/time_llm/run_experiments.py` discovery flow

### Third batch

1. External baseline integration hooks
2. `fairness/analyzers/ecg_classifier_fairness_analyzer.py`

### Fourth batch

1. `distillation/scripts/distill_mitbih_classifier.py`
2. Fairness mitigation experiment scripts
3. `scripts/time_llm/config_generator_mitbih_distillation.py`

---

## 10. Minimum viable milestone

The first real milestone is:

1. Beat-centered AAMI 5-class dataset exists
2. Time-LLM classification variant trains
3. Friend baseline trains on same split
4. Fairness report runs on both

The second milestone is:

1. Teacher/student/distilled comparison exists

---

## 11. Historical execution step (completed)

Run one corrected BERT teacher seed in a fresh directory so invalid checkpoints
cannot be resumed:

```bash
PIPELINE_DIR=experiments/mitbih_fairness_pipeline_frozen_backbone_20260711 \
SEED=831363 \
bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
```

The default teacher gate requires macro-F1 at least `0.35` and predictions from
at least four classes. It stops before student and distillation phases if the
teacher fails. Override only for a documented sensitivity analysis with
`MIN_TEACHER_MACRO_F1` or `MIN_TEACHER_PREDICTED_CLASSES`.

After one seed passed, the fixed seeds were run in fresh locked directories.
The earlier example command was:

```bash
PIPELINE_DIR=experiments/mitbih_fairness_pipeline_all_seeds_objective_fixed \
ALL_SEEDS=1 \
bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
```

Do not report either archived collapse run as a model result. Use the completed
directories listed in the completion update for all final tables.
