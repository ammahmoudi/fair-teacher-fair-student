# MIT-BIH Time-LLM Classification Variant

## Goal

Build a **Time-LLM variant for ECG beat classification on MIT-BIH**, then run the same kind of fairness and distillation analysis used for blood glucose forecasting.

This is the intended direction:

- Adapt the Time-LLM-style model from forecasting to classification.
- Support AAMI 5-class ECG beat classification.
- Support teacher → student distillation.
- Run fairness analysis on the classifier outputs.
- Test whether our fairness-fixing ideas generalize beyond BG forecasting.

This document replaces the earlier idea of starting from a generic classifier-only baseline as the main target.

A generic CNN classifier can still be added as an external comparison baseline, but the **main system goal** is a **Time-LLM classification variant**.

## Current implementation and validated training protocol

As of 2026-07-11, the AAMI 5-class data path, Time-LLM classifier, teacher and
student wrappers, distillation path, fairness analysis, gin generators, and
one-command pipeline are implemented.

The main preprocessing and training rules are:

- split by record, never by beat;
- exclude duplicate-subject record `202` by default;
- retain robust per-record median/MAD normalization with mean/std fallback;
- freeze the pretrained LLM backbone by default, matching BG Time-LLM;
- train the ECG patch embedding, input projection, and five-class head;
- use capped class-balanced sampling for supervised teacher/student training;
- use ordinary cross-entropy when a weighted sampler is active;
- use inverse-frequency weighted cross-entropy only when training with the
	ordinary shuffled loader;
- never combine class-balanced sampling with inverse-frequency weighted loss,
	because this applies the imbalance correction twice.

The friend's baseline use of SMOTE, class weights, and min-max normalization was
reviewed. SMOTE and random beat-level splitting were not adopted because they
do not preserve the record-safe evaluation protocol. The existing robust
normalization was retained because the observed failures were label-imbalance
failures, not waveform-scale failures.

Two invalid diagnostic runs are intentionally preserved:

- `experiments/mitbih_fairness_pipeline_BROKEN_teacher_collapsed_20260710/`:
	weighted loss without sufficient sampling caused majority-class `N` collapse;
- `experiments/mitbih_fairness_pipeline/`: capped class sampling plus inverse
	class weights overcorrected and caused the BERT teacher to predict `F` for
	every test beat.
- `experiments/mitbih_fairness_pipeline_objective_fixed_20260711/`: corrected
	sampling/loss interaction but full-backbone BERT fine-tuning at `1e-4`
	stalled near random CE and produced only class `N`.

After making sampling and loss weighting mutually exclusive, a one-epoch
BERT-tiny smoke run reached accuracy `0.6815`, macro-F1 `0.4336`, and produced
all five classes. A BG-style frozen full-BERT smoke run then reached accuracy
`0.6671`, macro-F1 `0.3784`, and produced all five classes after one epoch. These
are behavior checks, not final reported results.

Default optimization hyperparameters are:

- task-module learning rate: `1e-4`;
- frozen backbone learning rate: inactive;
- optional unfrozen-backbone learning rate: `1e-5`;
- batch size: `32` training and `64` prediction for BERT-family models;
- dropout: `0.1`;
- patch length: `16`, stride: `8`, pooling: mean;
- ten full-training epochs;
- class-balanced oversampling cap: `50x` with ordinary CE.

Because the backbone is frozen, checkpoints contain only the patch embedding,
input projection, and classifier state. Existing full checkpoints still load,
but new full-BERT task checkpoints are approximately `237 KB` rather than
`438 MB`.

---

## 1. What is the task now?

### Main task

**Beat-centered multi-class ECG classification** on MIT-BIH.

### Input

- One ECG window centered on an annotated beat.
- Single lead first, preferably `MLII`.
- Fixed length, default `256` samples.
- Curated one-record-per-patient set: keep `201`, exclude duplicate record `202`.

### Output

- One class label for the **center beat**.

### Why this task

The final research goal is not only MIT-BIH classification by itself. The real goal is:

1. Build a Time-LLM-like classification system.
2. Distill it to smaller student models.
3. Run fairness analysis on teacher, student, and distilled models.
4. Test whether the fairness-fixing methods from the BG work generalize here.

So the ECG task should support the same high-level pattern as the BG project:

- Teacher model.
- Student model.
- Distillation.
- Fairness audit.
- Fairness mitigation.

For the current fairness protocol, use the curated 47-record subset rather than
all 48 records, because records `201` and `202` come from the same subject.

### Dataset build modes

Curated fairness mode, default:

```bash
python scripts/mitbih/build_metadata.py
python scripts/mitbih/build_demographics.py
python scripts/mitbih/prepare_beat_dataset.py
```

Full 48-record mode, explicit opt-in:

```bash
python scripts/mitbih/build_metadata.py --include-duplicate-202
python scripts/mitbih/build_demographics.py --include-duplicate-202
python scripts/mitbih/prepare_beat_dataset.py --include-duplicate-202
```

---

## 2. Time-LLM: which variant should we use?

Two ideas were considered.

### Option 1: Time-LLM-inspired classifier

- Reuse encoder/backbone ideas.
- Replace forecasting head with a classification head.
- Output class logits instead of a forecast window.

### Option 2: Time-LLM encoder plus classifier head

- Use the model as a sequence encoder.
- Pool hidden states.
- Predict beat class.

### Recommended choice

Use **Option 1** as the main design.

Reason:

- It is a cleaner model variant.
- It is a full classification adaptation, not just feature extraction glued to a classifier.
- It better matches the idea of creating a true **Time-LLM classification variant**.
- It is easier to compare teacher, student, and distilled variants under one architecture family.

### Role of Option 2

Option 2 can still be used:

- As an ablation.
- As a simpler fallback if full head replacement is hard.
- As a quick prototype.

But the main system should be:

#### Main system form

Time-LLM-style backbone plus classification head as the native task head.

---

## 3. How the model should change

Current Time-LLM style in this repo:

```text
input time series -> encoder/backbone -> forecasting head -> future value window
```

Required ECG classification variant:

```text
beat-centered ECG window -> encoder/backbone -> pooling -> classification head -> class logits
```

### Required changes

1. Remove the forecasting output head.
2. Add sequence pooling or center-aware pooling.
3. Add a classification head.
4. Train with classification loss.
5. Evaluate with classification and fairness metrics.

### Pooling choices

Possible pooling methods:

- Mean pooling.
- Attention pooling.
- CLS-style learned token, if the architecture supports it.
- Center-region pooling around the anchor beat.

### Recommended pooling

Start with:

- **Mean pooling** or **center-region pooling**.

Center-region pooling is appealing because the task label belongs to the center beat.

---

## 4. What are the classes?

The classes should be **AAMI 5-class**, not binary only.

### Recommended main label space

- `N` = normal and related beats
- `S` = supraventricular ectopic beats
- `V` = ventricular ectopic beats
- `F` = fusion beats
- `Q` = unknown, paced, or unclassifiable beats

This should be the **main task**.

### Why AAMI 5-class

- Standard ECG classification target.
- Much more informative than binary normal/abnormal.
- Lets us study fairness by clinically distinct beat types.
- Better long-term fit for a publishable generalization story.

### Is the binary task still useful?

Yes, as a secondary endpoint rather than the primary benchmark:

- It began as a warm-up and debugging task.
- It was later locked as N versus S/V/F with Q excluded and augmented with RR
	features.
- Its completed five-seed result is a reportable cross-domain stress test.

It does not replace AAMI-5 as the main task, and its locked test results cannot
be used for post-hoc model selection.

---

## 5. How beats are represented

MIT-BIH provides:

- Continuous ECG waveform.
- Beat annotation sample index.
- Beat symbol.

A beat is:

- Anchored at one annotation sample index.
- Represented by a window around that index.

### Default representation

- Input window size = `256`
- Left context = `128`
- Right context = `128`

### If beats are close

If two beats occur close together:

- Windows may overlap.
- That is fine.
- The label belongs to the **center beat only**.
- Nearby beats inside the window are context.

This is standard for beat-centered ECG classification.

---

## 6. Input data work needed

A classification-ready dataset layer is needed.

### Required preprocessing

1. Parse `.hea` metadata.
2. Parse `.atr` beat annotations.
3. Choose the primary lead per record.
4. Normalize waveform per record.
5. Extract beat-centered windows.
6. Map raw symbols to AAMI 5-class labels.
7. Attach subgroup metadata.
8. Split at the record level.

### Output dataset unit

Each sample should contain:

- `record_id`
- `beat_sample_index`
- `signal_window`
- `class_label`
- `sex`
- `age_group`
- `paced/non_paced`
- `quality/difficulty group`
- `split`

---

## 7. Fairness goal

The fairness goal is to do for ECG classification what was done for BG forecasting:

- Train the main model.
- Compress it with distillation.
- Compare fairness before and after distillation.
- Test fairness mitigation methods.

### Main fairness questions

1. Does distillation worsen subgroup detection performance?
2. Are some beat classes much less detectable in certain groups?
3. Do our fairness-fixing methods help here too?

### Primary fairness groups

- Sex.

### Secondary groups

- Age group.
- Paced vs non-paced.
- Quality/difficulty subgroup.
- Lead subgroup.

### Main fairness metrics

- Subgroup recall / TPR
- EO Gap
- DP Gap
- FVO
- Macro-F1 by group
- Per-class recall gaps

Because this is multi-class, the analysis will need:

- Macro metrics.
- Per-class one-vs-rest fairness analysis.

---

## 8. Distillation goal

This is a **distillation project**, not only a classifier project.

So the ECG variant must support the following.

### Teacher model

- Stronger Time-LLM classification variant.

### Student model

- Smaller Time-LLM classification variant.

### Distillation

- Teacher logits / soft targets.
- Student supervised loss.
- Distillation loss.
- Fairness analysis across all stages.

### Comparison set

- Teacher baseline.
- Student baseline.
- Distilled student.
- Fairness-mitigated distilled student variants.

This matches the BG distillation story more closely than a standalone unrelated CNN classifier.

---

## 9. What about your friend’s ECG-Classifier-LLM repo?

That repo is still useful, but as a **comparison baseline**, not the main architecture family.

### What is useful from it

- Standard 1D CNN classifier idea.
- 5-class ECG classification framing.
- Segmented beat dataset assumption.
- Simple train/eval runner shape.

### What role it should play here

Use it as:

- An external baseline model.
- A sanity-check model.
- Possibly another teacher or comparison system.

### What not to do

Do not let that repo define the main system architecture.

The main system should remain:

- **Our Time-LLM classification variant**
- With the external CNN as a comparator

---

## 10. Recommended model stack

### Main architecture family

- Time-LLM classification variant.

### External comparison baseline

- Friend’s 1D CNN ECG classifier style baseline.

### Why both

This allows comparison across:

- An architecture family adapted from the core project.
- A standard ECG classifier baseline.
- Fairness and distillation behavior across both.

This strengthens the final report.

---

## 11. First supported label setup

### Main target

AAMI 5-class:

- `N`
- `S`
- `V`
- `F`
- `Q`

### Optional debug target

Binary ectopy secondary endpoint:

- N versus S/V/F, with Q excluded

### Recommendation

Document and build around AAMI 5-class as the primary benchmark. The binary
mode is now also documented as a completed secondary stress test; retain
separate output directories and report it without substituting it for AAMI-5.

---

## 12. Training and evaluation setup

### Input settings

- Lead: `MLII` preferred.
- Window size: `256`.
- Sampling rate: `360 Hz`.
- Record-level split.

### Loss

- Multi-class cross-entropy.
- Capped class-balanced sampling plus unweighted cross-entropy for the main
	supervised teacher/student runs.
- Inverse-frequency weighted cross-entropy only when no weighted sampler is
	active.
- Distillation loss for teacher/student experiments.

### Evaluation metrics

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Per-class recall.
- Confusion matrix.

### Fairness metrics

- EO Gap
- DP Gap
- FVO
- Per-class subgroup recall gaps

---

## 13. Recommended implementation order

### Stage 1

Build the data layer for beat-centered AAMI 5-class classification.

### Stage 2

Build a friend-style baseline CNN classifier as an external comparison model.

### Stage 3

Build the Time-LLM classification variant with a classification head.

### Stage 4

Train teacher and student variants.

### Stage 5

Run distillation.

### Stage 6

Run fairness analysis.

### Stage 7

Test fairness mitigation methods from the BG work.

---

## 14. Final decision

### What is the main system?

A **Time-LLM classification variant** for MIT-BIH AAMI 5-class beat classification.

### What is the main task?

Beat-centered ECG classification.

### What are the main classes?

AAMI 5-class:

- `N`
- `S`
- `V`
- `F`
- `Q`

### Which architecture choice should be used?

Use:

- **Time-LLM-inspired classifier** as the main design.

Use additionally:

- **Time-LLM encoder plus classifier head** only as an ablation or fallback.

### What about the friend repo?

Bring it in as:

- An external ECG classifier baseline.
- A comparison model inside the system.

### Why this is the right direction

The final research goal is:

- Fairness of distillation on a Time-LLM-family model.
- On a new dataset and task.
- To test whether fairness methods generalize beyond BG forecasting.
