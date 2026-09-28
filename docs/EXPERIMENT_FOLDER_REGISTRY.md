# Experiment Folder Registry

This registry covers the canonical BG fairness pipeline and the completed
MIT-BIH ECG fairness pipelines. Generated reports should be regenerated from
their scripts rather than edited without updating the generator.

## Canonical BG Pipeline

```
distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/
```

---

## Phase 1 — Teacher Model

| Dir | Checkpoint | Notes |
|-----|-----------|-------|
| `phase_1_teacher/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/` | `checkpoints/checkpoint.pth` | Baseline BERT teacher, 10 epochs, seed=42 |
| `phase_1_teacher/bert_all_patients_10epochs_fair_gender/bert_all_patients_10epochs/logs/logs_2026-06-13_11-13-20/` | `checkpoints/checkpoint.pth` | Fair-sampling BERT teacher (T1), seed=42 |

Inference (per-patient, all 12 patients):
```
phase_1_teacher/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

Fair-teacher comparable inference (per-patient, all 12 patients):
```
phase_1_teacher/per_patient_inference_fair_teacher/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

- **Teacher baseline**: RMSE 24.162 | EO Gap raw 0.2302 | EO Gap calibrated (holdout) 0.0761
- **Teacher fair sampling (T1 teacher)**: RMSE 22.564 | EO Gap raw 0.1925 | EO Gap calibrated (holdout) 0.0478

---

## Phase 2 — Student Baseline (no KD)

| Dir | Checkpoint | Notes |
|-----|-----------|-------|
| `phase_2_student/` | (see logs/ subdir) | BERT-tiny trained standalone, seed=42 |

Inference:
```
phase_2_student/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

---

## Phase 3 — Distillation Runs

All use: BERT (teacher) → BERT-tiny (student), all 12 OhioT1DM patients, seed=42, lr=0.001, batch=32, alpha=0.5, beta=0.5, epochs=10.

### Run 1 — Baseline Distillation (no fairness)
```
phase_3_distillation/bert_to_bert-tiny_all_patients/
  logs/logs_2025-10-28_15-53-41/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference (placed at phase_3 root due to early pipeline version):
```
phase_3_distillation/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.361 | **EO Gap raw**: 0.2171 | **EO Gap calibrated (holdout)**: 0.0766

---

### Run 2 v1 — EqualizedOddsLoss (fairness regularisation, v1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender/
  logs/logs_2026-06-09_12-14-26/student_distilled.pth     ← CANONICAL CHECKPOINT (v1)
  logs/logs_2026-06-10_10-07-22/student_distilled.pth     ← v2 overwrite (see Run 2 v2)
```
Inference (uses v1 checkpoint `logs_2026-06-09_12-14-26`):
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.672 | **EO Gap raw**: 0.2194 | **EO Gap calibrated (holdout)**: 0.0629
- Loss: `EqualizedOddsLoss`, fairness_weight=0.3

---

### Run 2 v2 — HypoglycemiaTPREqualityLoss (focal+soft, fairness regularisation v2)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_v2/
  (no checkpoint kept — model weights removed post-inference)
```
Inference (uses checkpoint from fairness_gender `logs_2026-06-10_10-07-22`):
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_v2/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.426 | **EO Gap raw**: 0.2300 | **EO Gap calibrated (holdout)**: 0.0764
- Loss: `HypoglycemiaTPREqualityLoss` (focal γ=3, soft sigmoid slope=0.1), fairness_weight=0.3

---

### Run 3 — Oversampling Only (no fairness loss)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_oversample_gender/
  logs/logs_2026-06-12_12-49-05/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_oversample_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.918 | **EO Gap raw**: 0.2326 | **EO Gap calibrated (holdout)**: 0.0752
- WeightedRandomSampler, 2.4× female hypo windows, no fairness loss

---

### Run 4 — Oversampling + HypoglycemiaTPREqualityLoss (combined)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_oversample/
  logs/logs_2026-06-12_17-57-56/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_oversample/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.141 | **EO Gap raw**: 0.2225 | **EO Gap calibrated (holdout)**: 0.0650
- WeightedRandomSampler (2.4×) + HypoglycemiaTPREqualityLoss, fairness_weight=0.3

---

### Run 5 — Distilled from Fair Teacher (T1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.828 | **EO Gap raw**: 0.2089 | **EO Gap calibrated (holdout)**: 0.0579

### Run 6 — Distilled from Fair Teacher + O1 Constraint
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/
  logs/logs_2026-06-15_09-20-01/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.141 | **EO Gap raw**: 0.2135 | **EO Gap calibrated (holdout)**: 0.0656
- O1 projected dual-ascent fairness constraint on top of the T1 fair teacher; operationally successful, scientifically still negative on raw EO

### Run 7 — Distilled from Fair Teacher + O2 Calibration Head
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/
  logs/logs_2026-06-15_16-10-49/student_distilled.pth          ← CANONICAL CHECKPOINT
  logs/logs_2026-06-15_16-10-49/student_calibration_head.json  ← CANONICAL O2 SIDECAR
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 22.645 | **EO Gap raw**: 0.1189 | **EO Gap calibrated (holdout)**: 0.0286
- T1 fair teacher distilled with a learned per-gender affine calibration head; strongest completed student-side fairness result so far

### Run 8 — K1 Calibrated Soft Labels
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 25.677 | **EO Gap raw**: 0.2326 | **EO Gap calibrated (holdout)**: 0.0706

### Run 9 — Distilled from Fair Teacher + K1 Calibrated Soft Labels (T1+K1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender_fair_teacher/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 22.654 | **EO Gap raw**: 0.1981 | **EO Gap calibrated (holdout)**: 0.0719
- T1+K1 completeness check: recovers most of the T1 fairness gain but does not beat O2 and leaves the calibrated gap higher

### Run 10 — Distilled from Fair Teacher + K3 Feature Alignment (T1+K3)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k3align_gender_w100_fair_teacher/   (one dir per weight)
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k3align_gender_w100_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **w=100**: RMSE 22.496 | EO Gap raw 0.1974 | EO Gap calibrated 0.0417
- **w=200**: RMSE 22.438 | EO Gap raw 0.1919 | EO Gap calibrated 0.0536  (dir `..._k3align_gender_w200_fair_teacher`)
- Scale-invariant CORAL, both weights engaged (K3 Align ~0.169 ≈ 2% of total loss). Best RMSE in the table, but the raw EO Gap stays at baseline at BOTH weights, and the K3 term was flat across epochs at both → ceiling, not undertuning. K3 helps accuracy, not the raw hypo-TPR gap. **Settled — no further weights needed.**
- Earlier weight=1.0 and absolute-CORAL weight=100 runs were inert and deleted. Runner weight-aware; comparison script auto-discovers `_w<weight>_` dirs.

### Run 11 — Distilled from Fair Teacher + K4 Selective KD Replay (T1+K4)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k4replay_gender_fair_teacher/
```
- **RMSE**: 23.413 | **EO Gap raw**: 0.2056 | **EO Gap calibrated (holdout)**: 0.0645
- Upweight female-hypo windows 4× in KD loss; engaged (Teacher Loss ~250–280 vs ~210–248 baseline). Raw gap unchanged from plain T1 (still CRITICAL), EO_cal slightly worse. Negative, as expected for the loss-reweighting family. Runner `run_t1_k4_kd_replay_experiment.sh`.

### Run 12 — Distilled from Fair Teacher + O3 Adversarial Group Erasure (T1+O3)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o3adv_gender_fair_teacher/
```
- **RMSE**: 23.048 | **EO Gap raw**: 0.2118 | **EO Gap calibrated (holdout)**: 0.0557
- Gradient-reversal discriminator, λ=1.0. **Mechanism worked** — O3 Adv ~0.685 ≈ ln(2): discriminator at chance, gender erased from the representation. Yet the raw gap did not move (still CRITICAL). Most informative negative: proves the disparity is data-prevalence, not representational bias. Runner `run_t1_o3_adv_erasure_experiment.sh`.

### Run 13 — Distilled from Per-Group Teachers (T2)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_t2pergroup_gender_fair_teacher/
```
Per-group teacher checkpoints:
```
phase_1_teacher/bert_male_patients_10epochs/.../checkpoints/checkpoint.pth     (7 patients)
phase_1_teacher/bert_female_patients_10epochs/.../checkpoints/checkpoint.pth   (5 patients)
```
- **RMSE**: 22.956 | **EO Gap raw**: 0.2197 | **EO Gap calibrated (holdout)**: 0.0732
- Male + female teachers, per-sample routing by gender. Verified both teachers trained on correct splits and loaded. Raw gap WORSE than plain T1 (0.2089) → CRITICAL. Per-group specialization bakes in each group's base rate rather than removing it. Runner `run_t2_pergroup_teachers_experiment.sh`. (Female teacher data-thin: 5 patients.)

---

## Multi-Seed Robustness (O2 headline validation)

```
pipeline_2025-10-28_14-20-17/multiseed_robustness_results.txt   ← human-readable mean±std across 5 seeds
pipeline_2025-10-28_14-20-17/multiseed_robustness_results.csv   ← machine-readable mean±std across 5 seeds
```
- Seeds: 831363, 809906, 427368, 238822, 247659 (from `scripts/utilities/seeds.py`). Run on an RTX 5090 host; missing per-patient trees were regenerated locally from checkpoints.
- Baseline KD: RMSE 23.528±0.270 | EO_raw 0.217±0.006 | EO_cal 0.067±0.008
- T1 (fair teacher): RMSE 22.845±0.331 | EO_raw 0.201±0.008 | EO_cal 0.057±0.006
- **T1+O2: RMSE 22.942±0.667 | EO_raw 0.118±0.033 | EO_cal 0.045±0.004**
- O2 per-seed EO_raw: [0.115, 0.155, 0.153, 0.072, 0.094]. **Beats baseline in all 5 seeds** (O2 worst 0.155 < baseline best 0.207); confirms the ~0.23→0.12 headline. O2 std larger (±0.033) — improvement magnitude is seed-sensitive but direction is 5/5 consistent.
- Current status: all 15 method×seed runs have auditable window-level `inference_results_reformatted.csv` files (12 patients each). Runner `regenerate_o2_multiseed_inference.sh`; original multiseed runner `run_multiseed_robustness_experiment.sh`; aggregator `aggregate_multiseed_results.py`.

---

## Comparison Results

```
pipeline_2025-10-28_14-20-17/fairness_comparison_results.csv   ← machine-readable
pipeline_2025-10-28_14-20-17/fairness_comparison_results.txt   ← human-readable
pipeline_2025-10-28_14-20-17/FAIRNESS_EXPERIMENTS_SUMMARY.md   ← analysis + findings
pipeline_2025-10-28_14-20-17/multiseed_robustness_results.txt  ← human-readable multi-seed validation
pipeline_2025-10-28_14-20-17/multiseed_robustness_results.csv  ← machine-readable multi-seed validation
```

**Key finding:** Most student-side fairness interventions remain weak, but O2 is a clear exception. The fair teacher reaches RMSE 22.564 / EO_raw 0.1925, the distilled T1 student only partially inherits that gain at RMSE 23.828 / EO_raw 0.2089, and T1+O1 improves RMSE to 23.141 without escaping the critical raw-gap range at EO_raw 0.2135. By contrast, T1+O2 reaches RMSE 22.645 / EO_raw 0.1189 / EO_cal 0.0286, making it the strongest completed student result and the first method to move the student well out of the worst raw-gap regime. Leakage-free patient-holdout calibration still matters because it prevents the older, overly optimistic near-zero calibrated gaps.

---

## Remaining Method Experiments

Method implementation and the single-seed intervention grid are complete, but the stronger
article scope requires confirmatory repeats. See
[FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md](research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md)
for the evidence gates, required five-seed BG matrix, statistical analysis, and stop rules.
See [FAIRNESS_SOLUTIONS_ROADMAP.md](FAIRNESS_SOLUTIONS_ROADMAP.md) for the completed method
grid and out-of-scope ideas.

Closed scope:
- Completed: T1, T2, K1, K3, K4, O1, O2, O3, plus baseline KD and teacher/student baselines.
- Excluded by rationale: K2 (not meaningful for regression) and T3 (counterfactual/GAN augmentation is a separate project).
- Next step: build the manifest-driven BG confirmatory suite, reuse completed artifacts, and run
  only the missing method-by-seed cells before writing the article.

---

## MIT-BIH AAMI-5 Five-Seed Pipeline

Canonical directory:

```text
experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/
```

Key artifacts:

- `MULTISEED_ANALYSIS.md`: AAMI-5 utility/fairness analysis;
- `multiseed_aggregate_summary.csv`: five-seed model aggregates;
- `multiseed_per_seed_summary.csv`: paired seed-level results;
- `BG_ECG_CROSS_DOMAIN_COMPARISON.md`: detailed BG/AAMI-5 report, now including the completed binary secondary endpoint;
- `alternate_group_fairness_aggregate.csv`: sex-trained model audits for age, pacing, and signal difficulty.

AAMI-5 remains the primary ECG endpoint. Its fairness result is mixed: T1 has
the strongest directional N/V EO change (`-0.0191`, 3/5 wins), but no paired
fairness comparison is statistically significant and distilled S recall is
near zero.

## MIT-BIH Binary-Ectopy KD Selection

Development screen:

```text
experiments/mitbih_binary_ectopy_kd_grid/
```

Key artifacts:

- `selected_kd_protocol.json`: validation-only KD lock;
- `KD_SELECTION.md`: six-candidate one-seed screen and caveat;
- selected objective: alpha `0.5`, beta `0.5`, temperature `1.0`.

This is screening evidence, not independent multi-seed tuning evidence.

## MIT-BIH Binary-Ectopy Five-Seed Pipeline

Canonical directory:

```text
experiments/mitbih_binary_ectopy_five_seed/
```

Protocol: five fixed seeds, ten epochs, frozen BERT teacher and TinyBERT
student, 256/16 waveform context, previous/next RR features, N versus S/V/F,
Q excluded, validation-only checkpoint selection.

Key artifacts:

- `protocol_manifest.json` and `data_split_manifest.json`: locked protocol and exact records;
- `validation_multiseed_aggregate.csv` and `test_multiseed_aggregate.csv`;
- `MULTISEED_ANALYSIS.md`: complete validation/test interpretation;
- `BG_AAMI5_BINARY_COMPARISON.md`: canonical compact three-task report;
- `bg_aami5_binary_comparison.csv`: machine-readable three-task table;
- `.training_complete` and `.complete`: successful completion markers.

Locked-test headline versus Baseline KD:

| Method | Macro-F1 delta | Ectopy EO delta | EO wins |
|---|---:|---:|---:|
| T1 | -0.0111 | -0.0081 | 4/5 |
| O2 | +0.0127 | -0.0357 | 3/5 |
| T1+O2 | +0.0264 | -0.0248 | 4/5 |

These binary results are supportive cross-domain evidence. Validation averages
favor Baseline KD, one high-gap seed influences mean test EO reductions, and
all seeds reuse the same ten held-out records. Do not use the test table for
method selection or describe MIT-BIH as an independent BG cohort.
