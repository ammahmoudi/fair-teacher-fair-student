# Fairness in Clinical KD — Solutions Roadmap

> Context: Standard fairness interventions (loss regularization, oversampling) fail in our
> BERT→BERT-tiny glucose forecasting distillation pipeline because the raw EO Gap is driven
> by a clinically real 3:1 male:female hypoglycemia prevalence imbalance, not model bias.
> This document tracks proposed solutions for future experiments / paper contributions.

---

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [—] Removed from committed scope (see "Considered but not pursued")

---

## Intervention Point 1 — Fix the Teacher

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| T1 | [x] | **Fair teacher retraining** | Retrained the teacher with fair sampling and evaluated both the teacher and its distilled student. The teacher improves substantially, but the student only inherits part of that gain. | Teacher-side source fix works, but KD transfer is lossy. |
| T2 | [x] | **Per-group teachers** | Trained a male teacher (7 patients) and a female teacher (5 patients), distilled both into one student with per-sample routing by gender. Verified: both teachers trained on the correct splits, both loaded and frozen, routing active. Result: RMSE 22.956, EO_raw 0.2197, EO_cal 0.0732 — the raw gap is *worse* than plain T1 distillation (0.2089) and back in CRITICAL. | Negative, and instructive: specializing teachers per group does not remove the prevalence problem, it bakes it in — each teacher learns its group's base rate and the student faithfully reproduces both, preserving the disparity. Confirms the O3 finding that the gap is base-rate-driven, not fixable from the data/teacher side. (Female teacher data-thin: 5 patients.) |
| T3 | [—] | **Counterfactual data augmentation** | Synthesize additional female hypoglycemia windows (interpolation or GAN) to match male hypo prevalence. **Removed from committed scope — it is a separate research project, not a single experiment.** A GAN on clinical glucose time-series plus the required validation that synthetic hypos are physiologically realistic is multi-day work with its own correctness questions; the interpolation shortcut can produce physiologically impossible glucose traces and would be scientifically weak. Left as genuine future work. | Most principled fix in theory, but out of scope as a clean result here. |

---

## Intervention Point 2 — Fix the Knowledge Transfer

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| K1 | [x] | **Calibrated soft labels** | Implemented calibrated soft-label KD using per-gender teacher offsets and evaluated it end to end, both standalone (K1-only) and on the fair teacher (T1+K1). K1-only is the worst run in the sweep (RMSE 25.677). T1+K1 recovers most of the T1 fairness gain (EO_raw 0.1981) but does not beat O2 and leaves the calibrated gap higher (EO_cal 0.0719). | Negative result. Source-level label shifting alone is not enough; even on a fair teacher it underperforms the learned calibration head. |
| K2 | [—] | **Group-conditional KD temperature** | ~~Use higher softmax temperature for male batches during KD.~~ **Removed — not applicable.** Softmax temperature only exists for classification logits; this is a point-valued glucose *regression* model with no softmax over the output, so there is nothing to apply a temperature to. The only "implementation" would be to silently substitute group-conditional KD-loss weighting, which is just a relabeling of oversampling / K4 and would be dishonest to call K2. See "Considered but not pursued" below. | n/a — does not exist in a regression model. |
| K3 | [x] | **Fairness-aware feature alignment** | Scale-invariant CORAL alignment of male/female student hidden states (standardize by pooled per-dim std → correlation-matrix difference), via a forward hook on the student LLM. Ran at two engaged weights (w=100 and w=200; earlier inert runs deleted). Both give the **best RMSE in the table (22.50 / 22.44)** but leave the raw EO Gap at baseline (0.197 / 0.192, vs O2's 0.119). Decisively, the K3 term stayed flat at ~0.169 across all epochs at *both* weights — doubling the pressure did not reduce it, so the student pays the penalty as a fixed tax it cannot lower; the representation is pinned by the dominant GT+teacher losses. | Robust negative on raw-gap-via-alignment: confirmed across weights, mean/correlation feature alignment does not close the hypo-TPR gap. Helps accuracy; does not help raw fairness. O2 remains the only method that moves the raw gap. |
| K4 | [x] | **Selective KD replay** | Upweight the KD loss on minority-group (female) hypoglycemic windows (factor=4, weights normalized to mean 1). Engaged properly — Teacher Loss rose to ~250–280 vs ~210–248 baseline. Result: RMSE 23.413, EO_raw 0.2056, EO_cal 0.0645. Raw gap essentially unchanged from plain T1 (0.2089) and still CRITICAL; EO_cal slightly worse. | Negative result, as expected for the loss-reweighting family: the penalty is active but does not close the raw hypo-TPR gap. |

---

## Intervention Point 3 — Fix the Student Objective

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| O1 | [x] | **Projected dual-ascent EO constraint** | Implemented a constrained distillation objective on top of the fair teacher with an adaptive dual variable and EO-gap violation penalty. The run completed end to end, but the student still stayed in the critical raw-gap range. | Scientifically negative result: operationally valid, but not sufficient to close the fairness gap. |
| O2 | [x] | **Learned calibration head** | Trained a small per-group linear calibration layer jointly with distillation (`y_final = W_group * y_student + b_group`) on top of the fair teacher. The learned head is persisted alongside the student checkpoint and applied automatically at inference time. | Best completed student-side method so far. Improves both RMSE and EO Gap materially relative to T1-only and T1+O1. |
| O3 | [x] | **Adversarial group erasure** | Gradient-reversal discriminator predicting gender from student hidden states (λ=1.0). **The mechanism worked**: the O3 Adv term sat at ~0.685 ≈ ln(2)=0.693 across all epochs — the discriminator is at chance, i.e. gender is provably erased from the representation. Yet the result (RMSE 23.048, EO_raw 0.2118, EO_cal 0.0557) leaves the raw gap unchanged from plain T1 (still CRITICAL). | **Most informative negative.** Erasing group information does NOT close the gap — proving the disparity is a data-prevalence phenomenon (real 3:1 male:female hypo imbalance), not representational bias. Strongly supports why only a post-hoc per-group correction (O2) works. |

---

## Considered but Not Pursued

These were originally listed as candidate methods but removed from committed scope after
assessing fit with this setup (point-valued glucose **regression** distillation, BERT→BERT-tiny):

- **K2 — group-conditional KD temperature.** Softmax temperature is a *classification* concept
  acting on logits. This model emits a single glucose value per step with no output softmax, so
  there is no temperature to set. The only realizable variant is group-conditional weighting of
  the KD loss, which is functionally the oversampling/K4 idea under a different name — so calling
  it "K2" would misrepresent it. Excluded rather than relabeled.
- **T3 — counterfactual data augmentation.** Generating synthetic female-hypoglycemia windows
  (GAN or interpolation) and validating that they are physiologically realistic is a standalone
  research effort, not a single experiment; interpolation alone can yield impossible glucose
  traces. Genuine future work, out of scope for the current comparison.

Methods still committed and being run: **K4** (selective KD replay), **O3** (adversarial group
erasure), **T2** (per-group teachers).

---

## Recommended Experiment Order for Paper

### Phase 1 (Diagnosis) — DONE

- [x] Run baseline + 4 fairness variants
- [x] Show all training-time interventions fail (EO Gap unchanged)
- [x] Re-evaluate Fix B with leakage-free patient-holdout calibration

### Phase 2 (Source Fix)

- [x] **T1**: Retrain fair teacher → teacher improves, student inherits only part of the gain
- [x] **K1**: Calibrated soft labels → negative result, worse than baseline KD and T1

### Phase 3 (Best Combined)

- [x] **T1 + K1**: Fair teacher + calibrated soft labels → completed as completeness check; recovers most of the T1 fairness gain (EO_raw 0.1981) but does not beat O2 and leaves the calibrated gap higher (EO_cal 0.0719)
- [x] **T1 + O1**: Fair teacher + projected dual-ascent EO constraint → completed, but still critical on raw EO Gap
- [x] **T1 + O2**: Fair teacher + learned calibration head → completed, best student-side fairness result so far
- [x] Compare completed Phase 2+3 runs on RMSE vs EO Gap tradeoff curve

### Current Combined Findings

Full comparison (OhioT1DM, BERT→BERT-tiny, all-patients pipeline, leakage-free
patient-holdout calibration, folds=2). EO Gap = |Male_TPR − Female_TPR| for
hypoglycemia detection; lower is fairer.

| Model | RMSE | EO_raw | EO_cal | Assessment (raw) |
| --- | --- | --- | --- | --- |
| Teacher baseline (BERT) | 24.162 | 0.2302 | 0.0761 | ❌ CRITICAL |
| Teacher fair sampling (T1 teacher) | 22.564 | 0.1925 | 0.0478 | ⚠️ CONCERNING |
| Student baseline (no KD) | 21.989 | 0.1948 | 0.0534 | ⚠️ CONCERNING |
| Distilled — no fairness | 23.361 | 0.2171 | 0.0766 | ❌ CRITICAL |
| Distilled + HypoglycemiaTPR loss (v1, EqualizedOdds) | 23.672 | 0.2194 | 0.0629 | ❌ CRITICAL |
| Distilled + HypoglycemiaTPR loss (v2, focal+soft) | 24.426 | 0.2300 | 0.0764 | ❌ CRITICAL |
| Distilled + Oversampling only | 24.918 | 0.2326 | 0.0752 | ❌ CRITICAL |
| Distilled + Oversampling + HypoglycemiaTPR loss | 24.141 | 0.2225 | 0.0650 | ❌ CRITICAL |
| Distilled from Fair Teacher (T1) | 23.828 | 0.2089 | 0.0579 | ❌ CRITICAL |
| Distilled from Fair Teacher + O1 Constraint | 23.141 | 0.2135 | 0.0656 | ❌ CRITICAL |
| Distilled from Fair Teacher + K1 Calibrated Soft Labels | 22.654 | 0.1981 | 0.0719 | ⚠️ CONCERNING |
| **Distilled from Fair Teacher + O2 Calibration Head** | **22.645** | **0.1189** | **0.0286** | ⚠️ MODERATE |
| Distilled from Fair Teacher + K3 Feature Alignment (w=100) | 22.496 | 0.1974 | 0.0417 | ⚠️ CONCERNING |
| Distilled from Fair Teacher + K3 Feature Alignment (w=200) | 22.438 | 0.1919 | 0.0536 | ⚠️ CONCERNING |
| Distilled from Fair Teacher + K4 Selective KD Replay | 23.413 | 0.2056 | 0.0645 | ❌ CRITICAL |
| Distilled from Fair Teacher + O3 Adversarial Erasure | 23.048 | 0.2118 | 0.0557 | ❌ CRITICAL |
| Distilled from Per-Group Teachers (T2) | 22.956 | 0.2197 | 0.0732 | ❌ CRITICAL |
| Distilled + K1 Calibrated Soft Labels | 25.677 | 0.2326 | 0.0706 | ❌ CRITICAL |

Interpretation: T1 is effective at the teacher stage, but most KD variants still fail
to preserve the full teacher fairness/accuracy gain. Every training-time intervention
(TPR losses v1/v2, oversampling, combinations, K4 selective replay) leaves the raw EO Gap
in the critical range. The single most informative negative is O3: adversarial group
erasure provably succeeded at its objective (the discriminator sat at chance, ~ln 2, so
gender is erased from the representation) yet the raw EO Gap did not move (0.2118, still
CRITICAL). This is direct evidence that the disparity is NOT representational bias — it is
a data-prevalence phenomenon (the real 3:1 male:female hypoglycemia imbalance). You cannot
fix a base-rate disparity by hiding the group variable, which is precisely why a post-hoc
per-group output correction (O2) is the only thing that works. T2 (per-group teachers)
closes the argument from the source side: training a dedicated teacher per group and routing
each sample to its own teacher makes the raw gap *worse* (0.2197), because each teacher
faithfully learns its group's base rate and the student reproduces both — the disparity is
transmitted, not removed. Source-side specialization, representation erasure, loss penalties,
and reweighting all leave the gap CRITICAL; only O2's output-level group correction moves it. O1 improved RMSE relative to baseline KD and T1-only distillation, but it still
failed to reduce the raw EO Gap below the critical range. T1+K1 recovers most of the T1
fairness gain (EO_raw 0.1981) yet does not beat O2 and leaves the calibrated gap higher,
while K1-only is the worst run overall. O2 is the only completed student-side method that
materially improves both fairness and accuracy at once, lowering the raw EO Gap to 0.1189
and the leakage-free calibrated EO Gap to 0.0286 while keeping RMSE near the fair teacher.
K3 feature alignment is now settled across two engaged weights. After fixing the loss to be
scale-invariant (standardize features by pooled per-dim std → correlation-matrix difference;
earlier absolute-covariance runs were inert at ~1e-4 and were deleted), both the w=100 and
w=200 runs gave the best RMSE in the table (22.50 / 22.44) but left the raw EO Gap at
baseline (0.197 / 0.192). The decisive observation: the K3 term stayed flat at ~0.169 across
all epochs at *both* weights — doubling the penalty did not reduce it, so this is a ceiling,
not undertuning. The student pays the alignment penalty as a fixed tax it cannot lower
because the dominant GT+teacher losses pin the representation. So K3, like every other
in-training method, does not move the raw hypo-TPR gap; it only helps accuracy. O2 remains
the only method that moves the raw gap. A second
quiet finding: per-gender threshold calibration flattens almost every model's EO_cal into
the 0.05–0.08 band regardless of training-time intervention, which further undercuts the
loss-based methods — only O2 stands apart on both raw and calibrated gaps. The current
evidence is that teacher quality matters, but the transfer mechanism itself must also
encode group-aware calibration to recover a substantial fairness gain.

### Experiments Complete — Including Multi-Seed Validation

**The full method grid is complete AND the headline is multi-seed validated.** Every committed
method is run: T1, K1, K3, O1, O2, K4, O3, T2 — spanning all three intervention points
(teacher: T1/T2; transfer: K1/K3/K4; student objective: O1/O2/O3). Every method except O2
leaves the raw gap CRITICAL or unchanged; O2 is the sole intervention that moves it. K2 and T3
were excluded with documented rationale (don't fit a regression model / out-of-scope GAN sub-project).

**Multi-seed robustness — DONE (5 seeds: 831363, 809906, 427368, 238822, 247659), mean ± std:**

| Method | RMSE | EO_raw | EO_cal |
| --- | --- | --- | --- |
| Baseline KD (no fairness) | 23.528 ± 0.270 | 0.217 ± 0.006 | 0.067 ± 0.008 |
| Distilled from Fair Teacher (T1) | 22.845 ± 0.331 | 0.201 ± 0.008 | 0.057 ± 0.006 |
| **Distilled from Fair Teacher + O2** | 22.942 ± 0.667 | **0.118 ± 0.033** | **0.045 ± 0.004** |

O2 per-seed EO_raw: `[0.115, 0.155, 0.153, 0.072, 0.094]`. **O2 beats baseline on the raw gap
in all 5 seeds** — even O2's worst seed (0.155) is below baseline's best (0.207) — confirming the
~0.23→0.12 headline. Honest caveat: O2's std (±0.033) is larger than baseline's (±0.006), so the
*magnitude* of the improvement is seed-sensitive (range 0.072–0.155) even though its *direction*
is 5/5 consistent. Run artifacts now include both `multiseed_robustness_results.txt` and
`multiseed_robustness_results.csv`, with all 15 method×seed per-patient/window outputs present.

**Method development is complete, but article-level confirmation is not.** O2 has five-seed
validation, while most mechanistically diverse negative methods remain single-seed results.
Before manuscript drafting, repeat the representative BG method families and mechanism checks
under the evidence gates in
[`research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md`](research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md).
Until then, the full grid is exploratory evidence and should not be described as robust proof
that every intervention fails or that the gap is caused only by base rates.

### Paper Narrative

> "Under real subgroup prevalence imbalance, a broad family of fairness interventions fails to
> close the raw hypoglycemia-detection TPR gap through clinical knowledge distillation. This
> holds across the full intervention spectrum: output-level loss regularization (equalized-odds
> and soft-TPR penalties), data-level oversampling, transfer-level calibrated soft labels (K1)
> and selective KD replay (K4), a projected dual-ascent EO constraint (O1), representation-level
> feature alignment (K3), and adversarial group erasure (O3). The O3 result is decisive: the
> adversary provably drove the student's representation to be gender-invariant (discriminator at
> chance), yet the gap was unchanged — demonstrating that the disparity is a data-prevalence
> phenomenon (a real ~3:1 subgroup base-rate imbalance), not representational bias that can be
> regularized away. Fair-teacher retraining improves the teacher materially but transfers only
> partially. The single intervention that works is a jointly learned per-group calibration head
> (O2): it is the only method that substantially reduces both the raw and the leakage-free
> calibrated EO gap without sacrificing accuracy. Because the gap originates in base rates rather
> than in the representation, closing it requires an explicit group-aware output correction in the
> transfer mechanism — in-training penalties, data reweighting, and group erasure are not sufficient."

---

## Implementation Notes

### T1: Fair Teacher Retraining

- Teacher script: `scripts/time_llm/run_experiments.py` + `config_generator.py`
- Add `--fairness-feature gender --fairness-weight <w>` to teacher training
- Or: add WeightedRandomSampler to teacher DataLoader in `data_processing/data_sets.py`
- Teacher checkpoint dir: `pipeline_2025-10-28_14-20-17/phase_1_teacher/`
- Completed artifact: `pipeline_2025-10-28_14-20-17/phase_1_teacher/bert_all_patients_10epochs_fair_gender/`
- Comparable inference artifact: `pipeline_2025-10-28_14-20-17/phase_1_teacher/per_patient_inference_fair_teacher/`

### K1: Calibrated Soft Labels

- Completed script: `scripts/fairness/compute_k1_teacher_offsets.py`
- Distillation implementation: `distillation/core/distillation_trainer.py`
- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/`
- Outcome: negative result; K1 worsens RMSE and does not beat T1 on fairness

### O1: Projected Dual-Ascent KD Constraint

- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/`
- Student checkpoint: `logs/logs_2026-06-15_09-20-01/student_distilled.pth`
- Outcome: RMSE 23.141, EO_raw 0.2135, leakage-free EO_cal 0.0656
- Training note: the run completed stably, but the fairness term did not materially pull the student out of the raw-gap regime

### O2: Learned Calibration Head

- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/`
- Student checkpoint: `logs/logs_2026-06-15_16-10-49/student_distilled.pth`
- Calibration sidecar: `logs/logs_2026-06-15_16-10-49/student_calibration_head.json`
- Outcome: RMSE 22.645, EO_raw 0.1189, leakage-free EO_cal 0.0286
- Training note: this is the strongest completed student result so far and is the main positive method result from the current roadmap

### K3: Fairness-Aware Feature Alignment

- Loss implementation: `FeatureAlignmentLoss` (CORAL/MMD) in `fairness/loss_functions/fairness_losses.py`
- Trainer hook: forward hook on `student.llm_model` in `distillation/core/distillation_trainer.py` captures the last hidden state; aligned across groups per step
- Runner: `scripts/pipelines/run_t1_k3_feature_alignment_experiment.sh` (CLI flags `--feature-alignment[-weight/-divergence/-layer]` in `distillation/scripts/distill_students.py`)
- Loss is scale-invariant: `FeatureAlignmentLoss._coral` standardizes features by the pooled
  per-dim std, then compares correlation matrices (earlier absolute-covariance form was inert
  at ~1e-4 and those runs were deleted). Real-data magnitude is now ~0.17; verified scale-invariant.
- Artifacts: `phase_3_distillation/bert_to_bert-tiny_all_patients_k3align_gender_w100_fair_teacher/`
  and `..._w200_fair_teacher/`
- Outcomes (scale-invariant CORAL, last layer): w=100 → RMSE 22.496, EO_raw 0.1974, EO_cal 0.0417;
  w=200 → RMSE 22.438, EO_raw 0.1919, EO_cal 0.0536. K3 Align ~0.169 (≈2% of total loss) at both,
  and flat across all epochs at both weights → the gap does not close; this is a ceiling, settled.
- Runner weight-aware (one dir per weight, `..._k3align_gender_w<weight>_fair_teacher`); comparison
  script auto-discovers weight dirs. K3 is done — no further weights needed.

---

## Related Files

- `fairness/loss_functions/fairness_losses.py` — current fairness loss implementations
- `distillation/core/distillation_trainer.py` — KD training loop
- `distillation/core/distillation_wrapper.py` — wraps training, applies demographics
- `fairness/utils/analyzer_utils.py` — demographics + shared utilities
- `scripts/fairness/compute_fairness_comparison.py` — generates comparison table
- `scripts/pipelines/run_fairness_distillation_experiments.sh` — runs all 4 variants
