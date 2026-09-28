# MIT-BIH ECG Tuning and Class-Limitations Roadmap

## Purpose

This document defines the next experiments for improving MIT-BIH Time-LLM
classification and fairness results. It separates:

1. model failures that can plausibly improve through tuning or better input
   representation;
2. subgroup limitations that cannot be repaired without additional independent
   records;
3. reportable claims that are already supported by the completed five-seed
   experiment.

The current reportable split contains 47 records: 28 train, 9 validation, and
10 test. All tuning and checkpoint selection must use training and validation
data only. The test set remains reserved for evaluation after the configuration
is locked.

## Executive Decision

- Tune the existing MIT-BIH model before moving immediately to another
  dataset.
- Prioritize S-class learning and T1/O2 stability; these are model-level
  problems with plausible remedies.
- Do not claim F/Q sex fairness or broad pacing fairness from this dataset.
  Their limiting problem is independent subgroup support, not optimization.
- After locking the tuned configuration, run targeted age-group mitigation and
  then add an external ECG dataset for independent support.

## Final Experiment Status — 2026-07-15

The validation-only KD screen and locked binary-ectopy five-seed run are now
complete. The selected KD objective is alpha `0.5`, beta `0.5`, temperature
`1.0`; all six model variants completed for all five seeds. The final reports
are:

- `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`;
- `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`.

The binary result is a reportable secondary endpoint, not a replacement for
AAMI-5. On the locked test split, O2 improves macro-F1 by `0.0127` and ectopy
EO by `0.0357`; T1+O2 improves macro-F1 by `0.0264` and ectopy EO by `0.0248`
versus Baseline KD. T1+O2 has 5/5 macro-F1 wins and 4/5 EO wins. Validation
averages still favor Baseline KD, so these test results cannot be used to
select or retune a method.

## Implementation Status — 2026-07-14

Completed:

- exposed the T1 group/class oversampling cap through CLI, generated gin,
  runtime loader, and pipeline environment variables;
- added a separate O2 calibration-head learning rate;
- added scale-to-one and bias-to-zero O2 regularization;
- added opt-in validation-only checkpoint selection using macro-F1, S recall,
  and support-qualified N/V EO;
- added an optional hard S-recall gate;
- added protocol tests for config propagation, regularization gradients, and
  checkpoint selection;
- added opt-in previous/next RR features with neutral fallback and a compact
  projection fused into the existing pooled representation;
- added `aami5`, `binary_ectopy`, and `binary_non_n` label modes while
  preserving original AAMI labels in prediction artifacts;
- added configurable 256, 720, and 1,080 sample contexts without changing the
  default 256/16 configuration;
- added a resumable 16-case server suite, phase filtering, disk checks,
  failure markers, per-case manifests, and incremental validation aggregation;
- completed 15 direct protocol tests, shell/Python syntax checks, a complete
  16-case dry run, and a real one-epoch binary-plus-RR run through `main.py`.
- aligned validation checkpoint fairness with the reportability rule so a
  class that every group fails cannot appear artificially fair;
- added cached-window validation, complete-variant status checks, and immutable
  case manifests to prevent stale indexes, partial completion, or unsafe resume
  with changed seeds, epochs, models, or hyperparameters.

The binary-plus-RR integration smoke completed Teacher, Student, Baseline KD,
checkpoint restoration, validation selection, prediction export, and sex
fairness analysis. Its one-epoch test macro-F1 values were 0.6399 for Teacher
and Student and 0.6663 for Baseline KD. These are implementation checks, not
reportable tuning evidence.

One-epoch BERT-tiny behavior check:

| T1 cap | Validation macro-F1 | Validation S recall | Validation N/V EO | Test accuracy | Test S recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4× | 0.3357 | 0.1118 | 0.1945 | 0.7441 | 0.0269 |
| 8× | 0.2918 | 0.1553 | 0.2993 | 0.6137 | 0.0269 |

This smoke comparison is not a reportable experiment and was not used to
select the cap. It demonstrates that stronger T1 sampling can raise validation
S recall while harming utility and fairness, and that validation S gains do
not automatically generalize to test S recall. The subsequent validation-only
KD screen and locked five-seed binary run are documented in the final status
section and source artifacts below.

## Current Evidence

### Class support by split

| Split | Records | N | S | V | F | Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 28 | 52,573 | 2,181 | 4,238 | 749 | 4,168 |
| Validation | 9 | 17,300 | 322 | 778 | 13 | 1,807 |
| Test | 10 | 18,697 | 223 | 2,201 | 40 | 2,068 |
| Total | 47 | 88,570 | 2,726 | 7,217 | 802 | 8,043 |

S has enough beats for learning and evaluation. Its near-zero recall is not
caused by an empty class. F and Q have more total beats than S, but their
sex-specific beats are concentrated in very few records.

### Structural subgroup support

| Comparison | Test support | Global record providers | Consequence |
| --- | --- | --- | --- |
| Female vs male F | 4 vs 36 beats | 3 female vs 13 male records | Female F test behavior comes from one record; sex EO is not reliable |
| Female vs male Q | 2,064 vs 4 beats | 5 female vs 4 male records | Male Q test behavior comes from one record and four beats |
| Paced vs non-paced V | 2 vs 2,199 beats | 2 paced V records globally | V pacing EO cannot be estimated reliably |
| Paced S | 0 beats globally | 0 records | The comparison is impossible in MIT-BIH |
| Paced F | 0 beats globally | 0 records | The comparison is impossible in MIT-BIH |

The current minimum support rule of 20 true samples per group/class correctly
excludes these comparisons. Lowering that threshold would make the report look
more complete but would not make the estimates reliable.

## Limitation 1: S Recall Is Near Zero

### What is happening

Five-seed mean S recall is:

| Model | S recall |
| --- | ---: |
| Teacher | 0.0439 ± 0.0422 |
| Student | 0.0422 ± 0.0473 |
| Baseline KD | 0.0081 ± 0.0058 |
| T1 | 0.0072 ± 0.0093 |
| O2 | 0.0135 ± 0.0071 |
| T1+O2 | 0.0072 ± 0.0087 |

Across all five test seeds, true S beats are predicted approximately as:

| Model | Predicted N | Predicted S | Predicted V | Predicted F/Q |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 40.2% | 4.4% | 51.0% | 4.4% |
| Baseline KD | 45.7% | 0.8% | 49.5% | 4.0% |
| T1 | 43.1% | 0.7% | 50.3% | 5.9% |
| O2 | 44.8% | 1.3% | 49.2% | 4.6% |
| T1+O2 | 43.5% | 0.7% | 50.0% | 5.8% |

This is a class-learning failure, not a fairness success. KD suppresses the
already weak S signal from the teacher.

### Why sampling alone is insufficient

The raw training distribution is 82.26% N and 3.41% S. The baseline 50×
class-balanced sampler produces an expected 20% from every class, yet baseline
KD still has only 0.8% S recall. T1's current 4× group/class sampler produces
the following expected class mixture:

| Sampler | N | S | V | F | Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline class-balanced, 50× cap | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| T1 group/class, 4× cap | 25.5% | 17.4% | 25.5% | 6.0% | 25.5% |
| T1 group/class, 8× cap | 22.9% | 20.7% | 22.9% | 10.7% | 22.9% |
| T1 group/class, 16× cap | 20.3% | 20.3% | 20.3% | 19.0% | 20.3% |

S is already sampled frequently enough that increasing its frequency alone is
unlikely to solve the confusion.

### Root-cause hypothesis

S beats often resemble N morphologically and are distinguished by premature
timing and neighboring RR intervals. The current input is a single
beat-centered 256-sample window, approximately 0.71 seconds at 360 Hz. It may
not provide enough preceding/following rhythm context. The observed N/V
confusion is consistent with missing timing context.

### Fix sequence for S

#### S1. Improve checkpoint selection

- Record validation macro-F1, per-class recall, and N/V EO at every epoch.
- Require validation S recall of at least 0.05 before accepting a checkpoint.
- Among checkpoints within 0.01 macro-F1 of the best validation result, select
  the one with the best utility-constrained fairness result.
- Keep test predictions completely outside this selection.

#### S2. Tune KD transfer

Run a sequential validation search:

| Parameter | Current | Values to test |
| --- | ---: | --- |
| Supervised/KD weights | 0.5/0.5 | 0.7/0.3, 0.5/0.5, 0.3/0.7 |
| Temperature | 2 | 2, 4 |
| Task learning rate | 1e-4 | 5e-5, 1e-4, 2e-4 |
| Epochs | 10 | 10, 15, 20 with best-validation restoration |

Because the teacher is fairer but has weak S recall, increasing KD weight may
help fairness while further harming S. S recall must therefore be an explicit
acceptance gate rather than an after-the-fact metric.

#### S3. Add rhythm context

Prioritize these changes over more aggressive oversampling:

1. derive previous and next RR intervals from beat sample indices;
2. concatenate normalized RR features with the pooled Time-LLM representation;
3. test multi-beat windows of approximately 720 and 1,080 samples;
4. compare mean and center pooling;
5. evaluate the second MIT-BIH lead or a two-lead input as an ablation.

The RR-feature experiment is the most targeted change because it adds the
information used clinically to distinguish supraventricular ectopy without
abandoning the current architecture.

#### S4. Compare one imbalance correction at a time

Permitted ablations are:

- weighted sampling plus ordinary cross-entropy;
- ordinary shuffled batches plus effective-number or focal loss;
- class-balanced batches with no additional inverse-frequency weighting.

Do not combine inverse-frequency sampling and inverse-frequency loss. That
double correction already caused a collapsed diagnostic run.

### S acceptance criteria

A tuned configuration advances only if it:

- predicts all five classes on validation;
- has validation macro-F1 at least 0.35;
- reaches validation S recall at least 0.05 in every development seed;
- improves mean validation S recall over baseline KD;
- does not worsen N/V validation EO by more than a predeclared tolerance;
- preserves record-disjoint evaluation.

An aspirational target is mean S recall of at least 0.10 before final five-seed
testing. This target must be locked before inspecting new test predictions.

## Binary Normal-vs-Abnormal Secondary Task

A binary task is scientifically useful as a secondary fairness experiment. It
does not fix five-class S recognition; it changes the question from subtype
classification to abnormal-beat screening. The AAMI-5 task should remain the
primary benchmark so that binary aggregation cannot hide class failures.

### Candidate binary definitions

| Definition | Negative | Positive | Treatment of Q | Recommended role |
| --- | --- | --- | --- | --- |
| Ectopy screening | N | S, V, F | Exclude Q | Preferred binary endpoint |
| Any non-N screening | N | S, V, F, Q | Include Q as positive | Sensitivity analysis |

Q contains paced, fusion, and unclassifiable beats. Treating all Q beats as
clinically abnormal makes the positive class heavily dependent on paced-record
membership. The cleaner primary binary endpoint is therefore N versus S/V/F,
with Q excluded by a predeclared rule. N versus all non-N can still be reported
as a sensitivity analysis.

### Binary support

| Split | N | S/V/F positive | All non-N positive |
| --- | ---: | ---: | ---: |
| Train | 52,573 | 7,168 | 11,336 |
| Validation | 17,300 | 1,113 | 2,920 |
| Test | 18,697 | 2,464 | 4,532 |

Test sex support is adequate for both definitions:

| Definition | Female positive beats | Male positive beats | Positive-record providers per sex |
| --- | ---: | ---: | --- |
| S/V/F positive | 381 | 2,083 | 4 female, 4 male |
| All non-N positive | 2,445 | 2,087 | 4 female, 4 male |

This makes binary sex EO substantially more supportable than class-specific
F/Q sex EO. The groups still have different subtype mixtures, so report the
S/V/F/Q composition alongside every binary fairness result.

### Why binary pacing fairness is still weak

Binary aggregation does not create independent paced records:

- S/V/F-positive test support is 2 paced versus 2,462 non-paced beats;
- all-non-N test support is 2,066 paced versus 2,466 non-paced beats, but the
  2,066 paced positives come from one test record and are almost entirely Q;
- the resulting comparison measures record and subtype composition more than
  general pacing fairness.

Binary sex fairness is therefore recommended; binary pacing fairness remains
a descriptive sensitivity analysis only.

### Binary experiment design

Run the same five-seed matrix:

1. Teacher;
2. standalone Student;
3. Baseline KD;
4. T1;
5. O2;
6. T1+O2.

Use the same record split and validation-only selection. Retrain every model
because changing from five logits to two changes the classifier and KD target.

Report:

- AUROC and AUPRC;
- balanced accuracy and macro-F1;
- sensitivity and specificity at a validation-selected threshold;
- sex EO as the abnormal-class TPR gap;
- FPR gap and equalized-odds maximum;
- group calibration or Brier score;
- positive support and positive-record providers per sex.

Keep both a fixed 0.5 threshold and a validation-selected operating point.
Never select the threshold using test fairness.

### Hierarchical alternative

A stronger long-term design preserves subtype information:

1. head A predicts N versus abnormal;
2. head B predicts S/V/F/Q conditional on abnormal;
3. a shared encoder is optimized with both binary and five-class losses.

This can improve abnormal detection while retaining evidence about which rare
class remains difficult. It is a new model variant and should be compared with,
not substituted silently for, the current AAMI-5 classifier.

### Binary implementation checklist

- Add an explicit `label_mode` with `aami5`, `binary_ectopy`, and
  `binary_non_n` values.
- Apply label mapping before building samplers and class weights.
- Set `num_classes=2` for binary teacher, student, KD, and O2 heads.
- Preserve raw AAMI labels in prediction CSVs for subtype-composition audits.
- Add binary utility, fairness, calibration, and support reports.
- Add tests proving Q exclusion/inclusion and record-safe split preservation.
- Store binary outputs in a new experiment directory; never resume AAMI-5
  checkpoints into the binary task.

### Binary claim boundary

A successful binary result supports:

> The fairness intervention improves sex parity for abnormal-beat screening.

It does not support:

> The intervention fixes S, F, or Q subtype recognition or class-specific
> fairness.

## Limitation 2: F/Q Sex Fairness Is Non-Reportable

### Why tuning cannot fix it

Training augmentation, oversampling, O2 calibration, or a lower support
threshold cannot create independent test records. Female F has only three
record providers in the full curated dataset, forcing roughly one provider
into each train/validation/test split. Test female F therefore has four beats
from one record. Test male Q similarly has four beats from one record.

Thousands of beats from one record do not provide thousands of independent
patients. Beat-level bootstrap confidence intervals would exaggerate the
effective sample size.

### Valid remedies

1. Add an external ECG dataset with F/Q annotations and sex metadata.
2. Use patient/record-level grouped cross-validation as a sensitivity analysis,
   while reporting that provider counts remain small.
3. Keep raw F/Q gaps and exclusion reasons in appendices, but exclude them from
   aggregate EO.
4. Consider a clinically justified secondary label grouping only as an
   additional task; do not silently merge F/Q in the primary AAMI benchmark.
5. Require both a beat threshold and a provider-record threshold for future
   fairness reporting. A recommended gate is at least 20 true beats and at
   least 3 independent test records per group/class.

### Invalid remedies

- lowering the 20-beat threshold to make F/Q reportable;
- generating synthetic test beats;
- splitting beats from the same record across train and test;
- treating beats from one patient as independent demographic evidence;
- claiming zero or small EO when both groups fail to recognize the class.

## Limitation 3: Pacing Fairness Is Structurally Unsupported

Only two paced records exist in the curated dataset. Across all 47 records,
paced beats contain 262 N, 6 V, 4,148 Q, and no S or F beats. The current test
set has only two paced V beats.

No learning-rate, sampler, calibration, or fairness-loss setting can make a
two-beat test cell reliable. Resplitting cannot create more paced V providers,
and paced S/F comparisons are impossible because those cells are globally
empty.

### Valid pacing actions

- Keep N-only pacing EO as a narrow sensitivity analysis.
- Report paced/non-paced utility as robustness or domain-shift evidence, not as
  broad demographic fairness.
- Add an external paced ECG cohort with multiple records per class.
- Match or stratify by class when comparing paced and non-paced utility so that
  the Q-heavy paced distribution is not mistaken for model unfairness.

### Pacing acceptance gate

Do not report class-specific pacing EO unless both groups contain at least 20
true beats from at least 3 independent records for that class. Under the
current MIT-BIH data, only N can pass the beat-support part of this gate.

## T1/O2 Stability Roadmap

For AAMI-5, T1 has the best directional mean distilled fairness result, while
T1+O2 improves accuracy but has a smaller and inconsistent EO benefit. For the
locked binary-ectopy test, O2 has the lowest mean ectopy EO and T1+O2 has the
highest mean macro-F1; neither ranking appears on the binary validation
averages. The stability question is therefore endpoint- and split-dependent,
not evidence for one universally strongest ECG method.

### T1 tuning

- Expose the group/class sampler cap in the public MIT-BIH configuration.
- Test caps of 2×, 4×, and 8× after locking the KD parameters.
- Retain ordinary cross-entropy whenever the weighted sampler is active.
- Select the cap using validation utility, S recall, and N/V EO together.

### O2 tuning

The O2 affine scales and biases are not exploding, but their effect is
seed-sensitive. Add and tune:

| Parameter | Current | Values to test |
| --- | ---: | --- |
| Calibration-head learning rate | Same 1e-4 task LR | 2.5e-5, 5e-5, 1e-4 |
| Scale-to-one regularization | 0 | 0, 1e-4, 1e-3 |
| Bias-to-zero regularization | 0 | 0, 1e-4, 1e-3 |
| Calibration warm-up | 0 epochs | 0, 2 epochs |

The regularized O2 objective is:

$$
\mathcal{L}=\mathcal{L}_{KD}
+\lambda_s\lVert s-1\rVert_2^2
+\lambda_b\lVert b\rVert_2^2.
$$

This keeps O2 close to the identity transformation unless validation evidence
supports a larger group-specific correction.

## Efficient Experiment Order

Do not run one large Cartesian search. Use sequential phases:

### Phase A: Baseline KD tuning

1. Tune supervised/KD weights and temperature.
2. Tune task learning rate and epochs.
3. Select using validation macro-F1, S recall, and N/V EO.

### Phase B: S representation

1. Add RR-before and RR-after features.
2. Compare 256-sample versus multi-beat context.
3. Lock the best representation before fairness-method tuning.

### Phase C: T1 and O2

1. Tune the T1 group/class cap.
2. Tune O2 learning rate and identity regularization.
3. Compare Baseline KD, T1, O2, and T1+O2 under the same locked backbone,
   representation, split, and checkpoint-selection rule.

### Phase D: Alternate protected attribute

1. Retrain T1/O2 targeting age group.
2. Audit every model on both age and sex.
3. Treat pacing as robustness and difficulty as data-quality robustness.

### Phase E: Independent support

1. Replicate on a larger ECG dataset with age and sex metadata.
2. Prefer PTB-XL for demographic power and official folds.
3. Consider INCART if preserving beat-level AAMI classification is the main
   requirement, while acknowledging its smaller demographic sample.

## Unattended Server Suite

The entrypoint is `scripts/pipelines/run_mitbih_tuning_suite.sh`. It preserves
the existing `main.py`, gin generators, fixed record split, compact
checkpoints, and per-seed pipeline. All new behavior is opt-in; existing BG and
default AAMI-5 runs are unchanged.

The suite contains 16 fixed screening cases:

- four KD weight/temperature cases;
- three T1 sampler caps;
- three O2 learning-rate/regularization cases;
- RR fusion at 256 samples and RR fusion with 720/1,080 sample contexts;
- both binary endpoints;
- one age-targeted fairness case.

Profiles:

| Profile | Seeds | Epochs | Intended use |
| --- | ---: | ---: | --- |
| `screen` | 1 | 2 | Fast wiring and gross-failure screen; never report |
| `full` | 3 development seeds | 10 | Validation-only candidate comparison |

Run a no-training validation first:

```bash
DRY_RUN=1 PROFILE=screen \
SUITE_DIR=/tmp/mitbih_tuning_suite_dryrun \
bash scripts/pipelines/run_mitbih_tuning_suite.sh
```

To screen only the preferred normal-versus-ectopy endpoint before running the
other cases:

```bash
SUITE_DIR=experiments/mitbih_binary_ectopy_screen
nohup env SUITE_DIR="$SUITE_DIR" PROFILE=screen CASES=binary_ectopy \
  CONTINUE_ON_ERROR=0 \
  bash scripts/pipelines/run_mitbih_tuning_suite.sh \
  > "$SUITE_DIR.launcher.log" 2>&1 &
echo $!
```

This runs one development seed for two epochs and includes Teacher, standalone
Student, Baseline KD, T1, O2, and T1+O2. It is an implementation and gross-
failure screen, not reportable evidence.

Preflight the server before launching. The runner requires at least 20 GB free
by default and stops before training if the threshold is not met:

```bash
df -h .
test -x venv/bin/python
```

Override `MIN_FREE_GB` only after confirming that another filesystem contains
the suite directory and enough space for generated indexes, logs, compact
checkpoints, and predictions. A full suite should not be started on a volume
that is already near capacity.

Launch the complete development suite unattended:

```bash
SUITE_DIR=experiments/mitbih_tuning_suite_server
mkdir -p "$SUITE_DIR"
nohup env SUITE_DIR="$SUITE_DIR" PROFILE=full CONTINUE_ON_ERROR=1 \
  bash scripts/pipelines/run_mitbih_tuning_suite.sh \
  > "$SUITE_DIR/launcher.log" 2>&1 &
echo $!
```

Monitor without interrupting the run:

```bash
tail -f experiments/mitbih_tuning_suite_server/suite.log
```

Resume after interruption by running the same command with the same
`SUITE_DIR`. Cases with `.complete` markers are skipped, failures retain
`.failed` markers and continue by default, and existing per-seed pipeline
artifacts are reused. Set `PHASES=kd,t1` or another comma-separated subset to
run only selected phases. Set `CASES=binary_ectopy` or another comma-separated
case list for a narrower run. `SUITE_SEEDS` and `SUITE_EPOCHS` may override a
profile when a deliberate smaller diagnostic is needed. Each case manifest
locks its seeds, epoch count, teacher/student models, label mode, context,
variants, aggregation split, and tuning variables. Reusing a case directory
with different settings fails immediately; use a new `SUITE_DIR` for a new
protocol.

Candidate aggregation is validation-only by default. The generated
`SUITE_RESULTS.md`, `suite_per_seed.csv`, and `suite_aggregate.csv` therefore
read `val_predictions.csv`, recompute support/performance-qualified EO, and do
not rank candidates using test metrics. `suite_status.json`, each
`case_manifest.json`, `suite.log`, and `.complete`/`.failed` markers provide
the resumability audit trail. The pipeline may produce test artifacts as part
of its existing train/inference lifecycle; they must remain uninspected until
the configuration is locked.

After selecting and locking one configuration from development validation,
run that configuration once over all five fixed seeds and report the held-out
test results. Do not promote the 16-case development suite itself to a
five-seed test search.

## Locked Binary-Ectopy Five-Seed Run

Use `scripts/pipelines/run_mitbih_binary_ectopy_five_seed.sh` for the final
binary experiment. This is not a tuning suite: it wraps the same core
`run_mitbih_fairness_distillation_pipeline.sh` used by the AAMI-5 experiment
and locks the successful screen protocol in an immutable manifest.

Before the final run, select the classification KD objective with the dedicated
validation-only grid:

```bash
SUITE_DIR=experiments/mitbih_binary_ectopy_kd_grid
mkdir -p "$SUITE_DIR"
nohup env SUITE_DIR="$SUITE_DIR" \
  bash scripts/pipelines/run_mitbih_binary_ectopy_kd_grid.sh \
  > "$SUITE_DIR/launcher.log" 2>&1 &
echo $!
```

This fast grid uses seed `831363`, five epochs, binary ectopy plus RR, and six
predeclared Baseline KD objectives. It reuses one Teacher and one standalone
Student across the candidates:

| Supervised alpha | KD beta | Temperature |
| ---: | ---: | ---: |
| 0.3 | 0.3 | 2 |
| 0.7 | 0.3 | 2 |
| 0.5 | 0.5 | 2 |
| 0.3 | 0.7 | 2 |
| 0.5 | 0.5 | 1 |
| 0.5 | 0.5 | 4 |

The selector admits candidates within 0.01 validation macro-F1 of the best, then
uses lower support-qualified ectopy sex EO and higher ectopy sensitivity as
tie-breakers. It never reads test metrics and writes
`selected_kd_protocol.json` plus `KD_SELECTION.md`.

Older BG patient-level runs used alpha/beta 0.3/0.3, while the later canonical
all-patient fairness run used 0.5/0.5. In the BG trainer these are direct MSE
coefficients and no classification softmax temperature exists. In the ECG
classifier, alpha weights supervised cross-entropy, beta weights temperature-
scaled KL divergence, and temperature is applied to teacher/student logits.
Therefore 0.3/0.3 is included as a historical diagnostic, but it cannot be
assumed optimal for binary ECG.

The one-seed selector is intentionally a fast screen requested before the
expensive final run. It reduces cost but remains seed-sensitive; describe it as
a predeclared screening lock, not as multi-seed hyperparameter evidence.

The fixed protocol is:

- seeds `831363,809906,427368,238822,247659`;
- 10 epochs for Teacher, Student, and every distillation variant;
- BERT Teacher and TinyBERT Student with frozen backbones;
- N versus S/V/F with Q excluded;
- 256/16 waveform context plus previous/next RR features;
- training-only class balancing, T1 cap 4, and ordinary cross-entropy;
- KD alpha/beta and temperature loaded from the one-seed validation screen;
- validation-only utility/fairness checkpoint selection on class 1;
- Teacher, Student, Baseline KD, T1, O2, and T1+O2.

Validate the manifest without training:

```bash
DRY_RUN=1 PIPELINE_DIR=/tmp/mitbih_binary_final_dryrun \
  KD_PROTOCOL_FILE=experiments/mitbih_binary_ectopy_kd_grid/selected_kd_protocol.json \
  bash scripts/pipelines/run_mitbih_binary_ectopy_five_seed.sh
```

Launch the resumable final run:

```bash
PIPELINE_DIR=experiments/mitbih_binary_ectopy_five_seed
mkdir -p "$PIPELINE_DIR"
nohup env PIPELINE_DIR="$PIPELINE_DIR" \
  KD_PROTOCOL_FILE=experiments/mitbih_binary_ectopy_kd_grid/selected_kd_protocol.json \
  bash scripts/pipelines/run_mitbih_binary_ectopy_five_seed.sh \
  > "$PIPELINE_DIR/launcher.log" 2>&1 &
echo $!
```

Monitor with:

```bash
tail -f experiments/mitbih_binary_ectopy_five_seed/five_seed_runner.log
```

The runner verifies all validation/test predictions before marking training
complete, then writes:

- `MULTISEED_ANALYSIS.md` and validation/test aggregate CSVs;
- paired per-seed deltas versus Baseline KD;
- `BG_AAMI5_BINARY_COMPARISON.md` and its machine-readable CSV;
- `protocol_manifest.json`, `.training_complete`, and `.complete`.

Use `AGGREGATE_ONLY=1` with the same `PIPELINE_DIR` to regenerate reports
without rerunning training. A protocol mismatch or missing seed/model artifact
causes a hard failure rather than silently producing a partial comparison.

## Seed and Selection Protocol

1. Use the documented one-seed KD screen for speed; retain the explicit
  seed-sensitivity limitation. Prefer three development seeds when compute
  permits.
2. Never rank configurations using test EO, test accuracy, or test macro-F1.
3. Lock all hyperparameters and acceptance thresholds.
4. Run the final locked configuration over all five fixed seeds.
5. Report mean, sample standard deviation, per-seed wins, and paired deltas.
6. Add record-level bootstrap confidence intervals where the number of
   independent test records is adequate.

For multi-objective selection, first retain configurations within 0.01 of the
best validation macro-F1, then prefer higher S recall and lower performance-
qualified N/V EO. Publish the Pareto trade-off instead of hiding utility costs.

## Implementation Checklist

- Expose `fair_max_oversample` in the MIT-BIH configuration generator and
  pipeline.
- Add a separate `student_calibration_learning_rate`.
- Add O2 scale and bias identity-regularization weights.
- Save validation class metrics and fairness metrics at every epoch.
- Add a utility-constrained fairness checkpoint selector.
- Add RR-before/RR-after metadata and a small RR projection layer.
- Add configurable multi-beat window lengths.
- Preserve compact task-only checkpoints and calibration sidecars.
- Add tests for sampler caps, O2 regularization, RR feature shapes, and
  validation-only checkpoint selection.

## Report Guidance

The completed evidence supports:

> T1+O2 robustly improves BG fairness. ECG transfer is task-dependent: AAMI-5
> remains mixed, while the locked binary-ectopy test shows supportive utility
> and fairness improvements for O2 and T1+O2. These binary findings are
> descriptive because the five-seed validation ranking differs and the mean EO
> reduction is influenced by one high-gap Baseline KD seed.

For binary ectopy, the locked test result is:

- O2: macro-F1 `+0.0127`, ectopy EO `-0.0357`, and 3/5 EO wins versus Baseline KD;
- T1+O2: macro-F1 `+0.0264`, ectopy EO `-0.0248`, 5/5 macro-F1 wins, and 4/5 EO wins;
- validation does not improve over Baseline KD on average, so the test table
  must not be used for post-hoc model selection or retuning;
- all five seeds use the same ten held-out records, so seed dispersion is not
  uncertainty over an independent patient population.

This cross-domain evidence supports portability of the fairness-aware KD and
output-calibration framework, but not a universal ECG improvement claim or an
independent replication of the BG effect size.

F/Q sex fairness and broad pacing fairness must remain explicit dataset
limitations until additional independent records are available.

## Source Artifacts

- ECG five-seed analysis:
  `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/MULTISEED_ANALYSIS.md`
- ECG aggregate metrics:
  `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/multiseed_aggregate_summary.csv`
- Beat index used for support counts:
  `data/mit-bih-arrhythmia/beat_index.csv`
- ECG sampler implementation: `fairness/utils/ecg_sampling.py`
- ECG distillation implementation:
  `distillation/core/ecg_classification_wrapper.py`
- MIT-BIH distillation configuration:
  `scripts/time_llm/config_generator_mitbih_distillation.py`
- Cross-domain comparison:
  `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/BG_ECG_CROSS_DOMAIN_COMPARISON.md`
- Binary-ectopy five-seed analysis:
  `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`
- BG/AAMI-5/binary comparison:
  `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`
- Locked binary protocol and split:
  `experiments/mitbih_binary_ectopy_five_seed/protocol_manifest.json` and
  `experiments/mitbih_binary_ectopy_five_seed/data_split_manifest.json`
