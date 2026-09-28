# Fairness Issue A: Distillation Worsens Equalized Odds Gap

**Priority:** HIGH — directly impacts clinical safety (hypoglycemia detection equity)  
**Status:** RESOLVED — characterized and addressed; see Resolution below.  
**Related file:** `distillation/core/distillation_trainer.py`  
**Full program:** [FAIRNESS_SOLUTIONS_ROADMAP.md](FAIRNESS_SOLUTIONS_ROADMAP.md) · results in [EXPERIMENT_FOLDER_REGISTRY.md](EXPERIMENT_FOLDER_REGISTRY.md) and the pipeline's `FAIRNESS_EXPERIMENTS_SUMMARY.md`

---

## Resolution (summary)

This issue triggered a full intervention program across all three points (teacher, transfer,
student objective). The headline correction to this doc's original plan: **the EqualizedOddsLoss
fix proposed below does NOT work**, and neither does any other in-training method. The reason is
now understood and proven.

**Key finding — the gap is a data-prevalence phenomenon, not model/representation bias.**
OhioT1DM has a clinically real ~3:1 male:female hypoglycemia prevalence imbalance. The decisive
evidence is O3 (adversarial group erasure): it provably drove the student's representation to be
gender-invariant (discriminator held at chance, ~ln 2) yet the raw EO gap did **not** move. You
cannot fix a base-rate disparity by hiding the group variable.

**What was tried (all on the all-patients gender pipeline, leakage-free patient-holdout EO):**

| Method | Raw EO Gap | Verdict |
|---|---|---|
| Baseline KD (no fairness) | 0.217 | ❌ the problem |
| TPR/EqualizedOdds loss (v1, v2) | 0.219 / 0.230 | ❌ no effect (the fix this doc proposed) |
| Oversampling (± loss) | 0.233 / 0.222 | ❌ |
| Fair teacher (T1) distilled | 0.209 | ❌ partial transfer only |
| K1 calibrated soft labels | 0.198 | ❌ |
| K3 feature alignment (w=100/200) | 0.197 / 0.192 | ❌ (best RMSE, gap unmoved) |
| K4 selective KD replay | 0.206 | ❌ |
| O1 dual-ascent EO constraint | 0.214 | ❌ |
| O3 adversarial group erasure | 0.212 | ❌ erasure works, gap unmoved |
| T2 per-group teachers | 0.220 | ❌ bakes in base rates |
| **O2 learned per-group calibration head** | **0.119** | ✅ **the only fix** |

**The fix that works: O2 — a jointly learned per-group affine calibration head** on the student
output, on top of the fair teacher. It is the only method that materially reduces both the raw
gap (0.217 → 0.119) and the leakage-free calibrated gap (→ 0.029) without hurting accuracy.
**Multi-seed validated** (5 seeds): O2 EO_raw 0.118 ± 0.033 vs baseline 0.218 ± 0.007 — beats
baseline in all 5 seeds. Because the disparity originates in base rates, the cure is an explicit
group-aware *output correction*, not an in-training penalty.

Everything below this line is the **original (2026-01) problem statement and fix plan**, kept for
history. Note its "Expected Outcome" (EqualizedOddsLoss drops the gap below baseline) was
empirically refuted — see the table above.

---

## Problem

Knowledge Distillation (KD) consistently **worsens Equalized Odds (EO) Gap** compared to the
baseline student model.  The student, before distillation, already has a moderate EO Gap —
but after being trained to mimic the teacher, the gap grows.  This means the distilled model
is less fair at detecting hypoglycemia (BG ≤ 70 mg/dL) equally across demographic groups.

### Observed Numbers

| Scenario | Feature | Student EO Gap | Distilled EO Gap | Change |
|---|---|---|---|---|
| All-patients (bert-tiny) | Age | 0.163 | 0.207 | **+27%** 🔴 |
| All-patients (bert-tiny) | Gender | 0.195 | 0.217 | **+11%** 🔴 |
| All-patients (bert-tiny) | Pump type | 0.082 | 0.106 | **+29%** 🔴 |
| Per-patient (tinybert) | Age | — | — | **+14%** 🔴 |
| Per-patient (tinybert) | Pump type | — | — | **+26%** 🔴 |

The pattern is consistent: distillation copies teacher biases into the student.

---

## Root Cause

The KD loss in `distillation/core/distillation_trainer.py` (line 72) is:

```python
loss = self.alpha * loss_gt + self.beta * loss_teacher
```

Where:
- `loss_gt` = MSE against ground truth (treats all patients equally)
- `loss_teacher` = MSE against teacher's predictions

The `loss_teacher` term penalises the student for *disagreeing with the teacher*, even when the
teacher is wrong for a specific demographic group.  If `bert-base-uncased` (teacher) has a
bias towards better True-Positive Rate for some groups, `loss_teacher` forces the student to
replicate that bias.

There is **no fairness regularisation** in the current loss.

---

## Fix Plan

### Step 1 — Understand the current trainer

Read `distillation/core/distillation_trainer.py` lines 1–105 (the full file).  
Pay attention to:
- how `self.dataloader` yields batches (does it include patient/group metadata?)
- the signature of `DistillationTrainer.__init__`

### Step 2 — Check whether group labels are available at train time

The dataloader must supply a `group_labels` tensor alongside `(X, y_true, y_teacher)`.  
Check `distillation/core/distillation_driver.py` and `distillation/scripts/distill_students.py`
to see what information is passed to the trainer.

If group labels are **not** currently in the batch, they must be added:
- Patient ID is already known (per-patient training loop)
- For all-patients mode, the CSV has an `id` column that maps to demographic group
- Add a `demographic_feature` argument (e.g., `"age"`, `"gender"`, `"pump"`) to the driver

### Step 3 — Import the fairness loss

In `distillation/core/distillation_trainer.py`, add:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fairness.loss_functions.fairness_losses import EqualizedOddsLoss
```

### Step 4 — Add fairness loss to the trainer constructor

Extend `DistillationTrainer.__init__` with two new optional parameters:

```python
def __init__(
    self,
    ...,
    fairness_weight: float = 0.0,          # 0 = disabled (safe default)
    fairness_feature: str = None,          # e.g. "age", "gender", "pump"
    target_threshold: float = 70.0,        # hypoglycemia boundary (mg/dL)
    pred_threshold: float = 70.0,
):
    ...
    self.fairness_weight = fairness_weight
    self.fairness_feature = fairness_feature
    if fairness_weight > 0 and fairness_feature is not None:
        self.eo_loss_fn = EqualizedOddsLoss(
            base_loss=nn.MSELoss(),
            fairness_weight=fairness_weight,
            target_threshold=target_threshold,
            pred_threshold=pred_threshold,
        )
    else:
        self.eo_loss_fn = None
```

### Step 5 — Add fairness term in the training loop

In the training loop (currently line 72), replace:

```python
loss = self.alpha * loss_gt + self.beta * loss_teacher
```

with:

```python
loss = self.alpha * loss_gt + self.beta * loss_teacher
if self.eo_loss_fn is not None and group_labels is not None:
    loss_fairness = self.eo_loss_fn(y_student, y_true, group_labels)
    loss = loss + loss_fairness
```

`group_labels` must be retrieved from the batch in the same loop iteration.

### Step 6 — Expose `fairness_weight` and `fairness_feature` in the CLI

In `distillation/scripts/distill_students.py`, add:

```python
parser.add_argument("--fairness_weight", type=float, default=0.0,
                    help="Weight for EqualizedOddsLoss. 0=disabled.")
parser.add_argument("--fairness_feature", type=str, default=None,
                    choices=["age", "gender", "pump", "sensor"],
                    help="Demographic feature to equalise TPR across.")
parser.add_argument("--target_threshold", type=float, default=70.0,
                    help="BG threshold for positive class (default: hypo=70).")
```

### Step 7 — Re-run the all-patients distillation pipeline

```bash
cd /path/to/fair-teacher-fair-student
python distillation/scripts/distill_students.py \
    --fairness_weight 0.2 \
    --fairness_feature age \
    --target_threshold 70.0
```

Start with `fairness_weight=0.2`.  If EO Gap is still rising, increase to 0.4.

### Step 8 — Re-run the fairness analysis

```bash
python fairness/scripts/apply_advanced_metrics_to_results.py
```

Compare the new EO Gap values against the baseline in:
`fairness/analysis_results/advanced_metrics_raw_window_majority_20260104_181211.csv`

---

## Expected Outcome

- All-patients distilled EO Gap for Age should drop from **0.207 → below 0.163** (below student)
- All-patients distilled EO Gap for Gender should drop from **0.217 → below 0.195**
- Prediction accuracy (RMSE) should not degrade by more than 5%

---

## Risk

The `EqualizedOddsLoss` requires group labels at training time.  If group labels are not
currently passed through the dataloader, adding them is a prerequisite — this may require
touching `distillation_driver.py` and the dataset class in `data_processing/data_sets.py`.

Suggested approach: start with the all-patients pipeline (single model, demographic CSV
available) before tackling per-patient.
