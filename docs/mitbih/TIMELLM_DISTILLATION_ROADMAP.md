# MIT-BIH Time-LLM Distillation Roadmap

## Goal

Build a **Time-LLM classification variant** for **MIT-BIH AAMI 5-class beat classification**, then study:

- Teacher performance
- Student performance
- Distillation behavior
- Fairness before and after distillation
- Transfer of fairness-mitigation methods from the BG work

This roadmap is the ECG-classification analogue of the Time-LLM BG distillation and fairness pipeline.

## Completion Update — 2026-07-15

The internal reportable program described here is complete for both ECG
endpoints. The AAMI-5 five-seed run is stored under
`experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/`.
The secondary binary-ectopy run (N versus S/V/F, Q excluded) is stored under
`experiments/mitbih_binary_ectopy_five_seed/` and completed all six variants
for all five seeds.

The authoritative synthesis is
`experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`.
AAMI-5 remains mixed; the locked binary test supports O2/T1+O2 portability,
but validation favors Baseline KD and the test result is not a basis for
retuning. The external CNN comparator remains pending.

## Current status and corrected experiment protocol

Phases 1 and 3-7 are implemented for the internal Time-LLM path. Phase 2, the
external CNN baseline under the same record-safe split, remains pending.

The production pipeline is:

`scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh`

It prepares data, trains a teacher and student baseline, runs baseline/T1/O2/
T1+O2 distillation by default, computes fairness comparisons, supports all
fixed seeds, and preserves configs, checkpoints, histories, predictions,
calibration heads, efficiency reports, classification reports, and logs.

### Imbalance correction

MIT-BIH requires imbalance handling, but sampling and loss weighting must not
apply the same inverse-frequency correction simultaneously:

- class- or group-balanced `WeightedRandomSampler` active: use ordinary CE;
- no weighted sampler: use inverse-frequency weighted CE;
- baseline KD and O2 use the same capped class-balanced sampler as supervised
  teacher/student training; T1 replaces it with the group/class sampler;
- O2 follows the BG lifecycle: its group-specific affine head is optimized
  jointly with the student under the ground-truth and KD losses. For ECG the
  scale and bias are per class because a scalar bias shared by all logits
  cancels under softmax.

As in BG Time-LLM, pretrained teacher and student LLM backbones are frozen by
default. The trainable ECG path consists of the patch embedding, continuous
input projection, and classification head. Full-backbone fine-tuning is an
explicit ablation with a separate `1e-5` backbone rate; the task modules retain
their `1e-4` rate.

### Failed-run diagnosis and acceptance rule

The initial run collapsed to majority class `N`. A subsequent run combined
up-to-50x class sampling with inverse-frequency CE and overcorrected: the BERT
teacher predicted class `F` for all `22,975` test beats, with macro-F1
`0.00030`. Both runs are diagnostic artifacts, not reportable results.

The pipeline now stops before distillation unless the teacher reaches the
configurable defaults:

- macro-F1 at least `0.35`;
- at least four predicted classes.

The acceptance gate uses validation predictions only. Test predictions are
reserved for final reporting.

The original random record split was also invalid for fairness calibration:
several validation sex/class cells were empty, including male `F` and `Q`.
Data preparation now performs deterministic record-level stratification over
sex, class coverage, and class counts. Records remain disjoint across splits.
Some cells remain intrinsically too sparse because female `F` beats occur in
only three records and `Q` is similarly record-concentrated. Equal-opportunity
gaps are therefore excluded from aggregate fairness summaries when any group
has fewer than 20 true examples for that class or best-group recall is below
`0.05`; support, raw gaps, and exclusion reasons remain explicit in the JSON
report.

A corrected one-epoch BERT-tiny smoke test predicted all five classes and
reached macro-F1 `0.4336`. A frozen full-BERT smoke test also predicted all five
classes and reached macro-F1 `0.3784` after one epoch. These historical smoke
checks were followed by the completed locked five-seed experiments cited in
the completion update.

### Historical launch sequence (completed)

```bash
PIPELINE_DIR=experiments/mitbih_fairness_pipeline_frozen_backbone_20260711 \
SEED=831363 \
bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
```

This gate-first sequence was followed before the fixed five-seed runs. Do not
rerun it as a new result unless the protocol changes and a new output directory
is used.

---

## Phase 0: Scope lock

### Phase 0 objective

Lock the exact research direction before implementation.

### Phase 0 decision

Main system:

- Time-LLM classification variant

Main task:

- Beat-centered MIT-BIH AAMI 5-class classification

Main labels:

- `N`
- `S`
- `V`
- `F`
- `Q`

Primary architecture choice:

- Time-LLM-inspired classifier

Secondary architecture choice:

- Time-LLM encoder plus classifier head, only as an ablation or fallback

External comparator:

- Friend's ECG-Classifier-LLM baseline repo

---

## Phase 1: Data layer

### Phase 1 objective

Prepare a distillation-ready and fairness-ready ECG classification dataset.

### Phase 1 tasks

1. Parse record metadata from `.hea`.
2. Parse beat annotations from `.atr`.
3. Choose a primary lead per record.
4. Normalize waveform per record.
5. Extract beat-centered windows.
6. Map raw symbols to AAMI 5 classes.
7. Attach subgroup metadata.
8. Exclude duplicate subject record `202` to keep one-record-per-patient accounting.
9. Create record-level splits.

### Phase 1 output artifacts

- `metadata_records.csv`
- `beat_index.csv`
- Reusable ECG classification dataset loader

### Phase 1 success criteria

Each sample has:

- Record ID
- Beat anchor index
- Input window
- AAMI 5-class label
- Fairness subgroup attributes
- Split assignment

---

## Phase 2: External baseline integration

### Phase 2 objective

Bring in the friend's ECG classifier as a comparison system.

### Phase 2 tasks

1. Clone or vendor the reference implementation in a controlled location.
2. Document exact preprocessing assumptions.
3. Adapt data loading to our record-safe protocol if needed.
4. Train and evaluate it on our split policy.
5. Save outputs for fairness comparison.

### Phase 2 role

This model is:

- A comparison baseline
- Not the main system architecture

### Phase 2 success criteria

- External baseline runs on our processed data.
- Outputs are comparable to the internal pipeline.
- Fairness can be computed on it too.

---

## Phase 3: Time-LLM classification variant

### Phase 3 objective

Adapt Time-LLM from forecasting to multi-class beat classification.

### Phase 3 changes

1. Replace the forecasting head with a classification head.
2. Add pooling or center-aware aggregation.
3. Change the loss to a multi-class classification loss.
4. Change the metrics to classification metrics.
5. Support AAMI 5-class outputs.

### Phase 3 primary design

```text
ECG beat window -> Time-LLM-style backbone -> pooling -> classification logits
```

### Phase 3 ablation design

```text
ECG beat window -> Time-LLM encoder -> pooled hidden states ->
lightweight classifier head
```

### Phase 3 success criteria

- Model trains end to end.
- Outputs 5-class logits.
- Classification metrics are stable.

---

## Phase 4: Teacher and student variants

### Phase 4 objective

Define the distillation pair inside the Time-LLM classification family.

### Phase 4 tasks

1. Select the teacher backbone size.
2. Select one or more smaller student variants.
3. Standardize the output label space across all variants.
4. Make training and evaluation scripts consistent.

### Phase 4 success criteria

- Teacher baseline classifier
- Student baseline classifier
- Comparable model interfaces for distillation

---

## Phase 5: Distillation pipeline

### Phase 5 objective

Run classification distillation experiments analogous to the BG distillation setup.

### Phase 5 tasks

1. Supervised teacher training
2. Supervised student baseline training
3. Teacher-to-student distillation
4. Save probabilities, logits, and predictions
5. Compare accuracy and fairness across teacher, student, and distilled systems

### Phase 5 loss structure

Expected components:

- Supervised classification loss
- Distillation loss on teacher outputs
- Optional fairness-aware regularization later

### Phase 5 success criteria

- Distilled student trains successfully.
- A teacher, student, and distilled comparison table is produced.

---

## Phase 6: Fairness evaluation

### Phase 6 objective

Run fairness analysis on all classification systems.

### Phase 6 primary groups

- Sex

### Phase 6 secondary groups

- Age group
- Paced versus non-paced
- Quality or difficulty subgroup
- Lead subgroup

### Phase 6 metrics

- Macro-F1 by group
- Subgroup recall or TPR
- EO Gap
- DP Gap
- FVO
- Per-class subgroup recall gaps

### Phase 6 success criteria

- Fairness report for teacher
- Fairness report for student
- Fairness report for distilled student
- Comparison summary across all systems

---

## Phase 7: Fairness mitigation transfer

### Phase 7 objective

Test whether fairness-fixing methods from the BG work generalize to ECG classification distillation.

### Phase 7 candidate directions

1. Output-level calibration ideas adapted for classification
2. Subgroup-aware loss weighting
3. Sample rebalancing or subgroup-aware replay
4. Distillation-time fairness regularization
5. Post-hoc thresholding or calibration analysis if needed

### Phase 7 success criteria

- At least one mitigation sweep is completed.
- The fairness-performance tradeoff is documented.

---

## Phase 8: Comparative report

### Phase 8 objective

Produce a clear research story across systems.

### Systems to compare

- External CNN baseline
- Time-LLM teacher classifier
- Time-LLM student classifier
- Time-LLM distilled classifier
- Fairness-mitigated variants

### Questions to answer

1. Can a Time-LLM-family model be adapted cleanly to ECG classification?
2. Does distillation change fairness in this new domain?
3. Do our prior fairness-fixing ideas transfer?
4. How does the Time-LLM family compare with a standard ECG CNN baseline?

---

## Recommended file plan

### Data layer

- `data_processing/ecg/metadata.py`
- `data_processing/ecg/label_map.py`
- `data_processing/ecg/dataset.py`

### Time-LLM classification

- `models/` or `llms/` adaptation for a classification head
- `main.py` using `llm_settings["method"] == "time_llm_ecg_classifier"`
- shared evaluation through `main.py` outputs and fairness analyzers
- `scripts/time_llm/config_generator_mitbih.py` for MIT-BIH config generation
- existing `scripts/time_llm/run_experiments.py` for execution

### Distillation

- `distillation/` extensions for classification mode
- `distillation/scripts/distill_mitbih_classifier.py`
- `scripts/time_llm/config_generator_mitbih_distillation.py`

### Baselines

- External CNN baseline integration scripts

### Fairness

- `fairness/analyzers/ecg_classifier_fairness_analyzer.py`
- `scripts/fairness/run_mitbih_classifier_fairness.py`

### Docs

- `docs/mitbih/TIMELLM_CLASSIFICATION_VARIANT.md`
- `docs/mitbih/TIMELLM_DISTILLATION_ROADMAP.md`

---

## Priority order

### Priority 1

- Data layer
- AAMI 5-class mapping
- Record-safe split

### Priority 2

- External CNN comparison baseline
- Time-LLM classification variant

### Priority 3

- Teacher and student setup
- Distillation pipeline

### Priority 4

- Fairness analysis
- Fairness mitigation transfer

---

## First milestone

The first milestone is complete when we have:

1. An AAMI 5-class beat dataset
2. The external baseline running on our split
3. The Time-LLM classification variant running
4. Basic fairness evaluation for both

---

## Final decision summary

Main system:

- Time-LLM classification variant

Main task:

- MIT-BIH AAMI 5-class beat classification

Main architecture choice:

- Time-LLM-inspired classifier

Secondary architecture choice:

- Time-LLM encoder plus classifier head as an ablation or fallback

External comparator:

- Friend's ECG classifier repo

Research goal:

- Fairness and distillation generalization from BG forecasting to ECG classification
