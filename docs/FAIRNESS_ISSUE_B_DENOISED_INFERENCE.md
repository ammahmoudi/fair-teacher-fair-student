# Fairness Issue B: Denoised Data Causes Severe Inference Unfairness

**Priority:** HIGH — largest raw disparity numbers; affects deployed Chronos model  
**Status:** Open  
**Related files:** `scripts/data_formatting/core/format_data.py`, `fairness/loss_functions/fairness_losses.py`

---

## Problem

When the Chronos model is run on **denoised** input data, fairness collapses completely.
The same model is fair on noisy data (RMSE ratios ≈ 1.0) but extremely unfair on denoised data.
This means the denoising preprocessing step benefits some demographic groups far more than others.

### Observed Numbers

| Feature | Metric | Noisy data | Denoised data | Change |
|---|---|---|---|---|
| Gender | RMSE ratio | ~1.07 🟢 | **1.506** 🔴 | +40% |
| Cohort | RMSE ratio | ~1.07 🟢 | **1.900** 🔴 | +78% |
| Age | RMSE ratio | ~1.07 🟢 | **1.304** 🔴 | +22% |
| Age | EO Gap | 0.157 🟢 | **0.623** 🔴 | +297% |

> RMSE ratio = worst-group RMSE / best-group RMSE.  1.0 = perfectly equal.  
> The Cohort ratio of **1.90** means one patient cohort has errors almost **twice as large** as another.

---

## Root Cause

Denoising (smoothing) removes high-frequency glucose fluctuations before feeding data to the model.
The problem is that **not all patients have the same fluctuation patterns**:

- Patients with **630G pump** (closed-loop insulin delivery) have already-stable glucose → smoothing
  barely changes their data → model predictions already accurate → smoothing gives little benefit.
- Patients with **Empatica E4 sensor** (higher noise floor) or older pumps have noisier raw
  signals → smoothing removes *real clinical signal* mixed with noise → model predictions worsen.
- Similar asymmetry applies to Age and Gender groups based on their activity levels and physiology.

The current denoising pipeline in `scripts/data_formatting/` applies an **identical smoothing
filter** to all patients with no per-group calibration.

---

## Fix Plan

### Step 1 — Profile the denoising effect per demographic group

Before any code changes, verify the hypothesis with data:

```bash
python - <<'EOF'
import pandas as pd, numpy as np, glob

# Load noisy and denoised CSVs for all patients
noisy_files   = glob.glob("data/ohiot1dm/*/noisy/*.csv")
denoised_files = glob.glob("data/ohiot1dm/*/denoised/*.csv")

for nf, df_path in zip(sorted(noisy_files), sorted(denoised_files)):
    pid = nf.split("/")[2]
    n = pd.read_csv(nf)["glucose"].values
    d = pd.read_csv(df_path)["glucose"].values
    diff = np.std(n) - np.std(d)          # how much variance was removed
    print(f"Patient {pid}: std_removed={diff:.2f}")
EOF
```

Group the `std_removed` values by Age, Gender, Pump from the OhioT1DM patient metadata.
Expected: 630G pump group will show smaller `std_removed` than non-630G group.

### Step 2 — Add per-group fairness validation to the denoising pipeline

In `scripts/data_formatting/core/format_data.py`, after the denoising step is applied, add a
check that computes RMSE difference between noisy and denoised per patient:

```python
def check_denoising_fairness(noisy_df, denoised_df, patient_meta):
    """Warn if denoising effect is highly unequal across demographic groups."""
    diffs = {}
    for pid in noisy_df["id"].unique():
        n = noisy_df[noisy_df["id"] == pid]["glucose"].values
        d = denoised_df[denoised_df["id"] == pid]["glucose"].values
        diffs[pid] = np.sqrt(np.mean((n - d) ** 2))  # per-patient smoothing magnitude

    # Group by demographic (from patient_meta dict: {pid: {"age_group", "gender", "pump"}})
    for feature in ["age_group", "gender", "pump"]:
        groups = {}
        for pid, mag in diffs.items():
            grp = patient_meta.get(pid, {}).get(feature, "unknown")
            groups.setdefault(grp, []).append(mag)
        means = {g: np.mean(v) for g, v in groups.items()}
        ratio = max(means.values()) / (min(means.values()) + 1e-8)
        if ratio > 1.5:
            print(f"WARNING: denoising effect unequal for {feature}: {means}, ratio={ratio:.2f}")
```

### Step 3 — Option A: Group-adaptive denoising (recommended for data preprocessing)

Instead of applying the same smoothing window to all patients, use a **per-patient adaptive
window** that preserves the same proportion of variance across groups:

In `scripts/data_formatting/core/format_data.py`, replace the fixed-window smoothing with:

```python
def adaptive_smooth(series, target_snr_ratio=0.85):
    """Smooth a glucose series while preserving target fraction of original variance."""
    original_std = np.std(series)
    target_std = original_std * target_snr_ratio
    # Binary search for window size that achieves target_std
    lo, hi = 1, 60
    while lo < hi:
        mid = (lo + hi) // 2
        smoothed = pd.Series(series).rolling(mid, center=True, min_periods=1).mean().values
        if np.std(smoothed) > target_std:
            lo = mid + 1
        else:
            hi = mid
    return pd.Series(series).rolling(lo, center=True, min_periods=1).mean().values
```

This ensures each patient retains the same fraction of glucose variability, regardless of
their device or physiology.

### Step 4 — Option B: GroupRegularizedLoss during Chronos fine-tuning (recommended for model)

If the model is fine-tuned on denoised data, add `GroupRegularizedLoss` to the training
objective to prevent it from fitting more strongly to any one group.

`GroupRegularizedLoss` is already implemented in `fairness/loss_functions/fairness_losses.py`
(line 162).  Usage:

```python
from fairness.loss_functions.fairness_losses import GroupRegularizedLoss

# In Chronos fine-tuning script:
fairness_loss_fn = GroupRegularizedLoss(
    base_loss=nn.MSELoss(),
    fairness_weight=0.3,
    regularization_type="mse_difference",   # penalise MSE spread across groups
)

# In training loop, replace loss computation:
loss = fairness_loss_fn(predictions, targets, group_labels)
```

Find the Chronos fine-tuning entry point in `scripts/chronos/run_experiments.py` and wire
this loss in.

### Step 5 — Re-generate denoised data with adaptive smoothing

```bash
cd /path/to/fair-teacher-fair-student
python scripts/data_formatting/runners/complete_data_pipeline.py \
    --dataset ohiot1dm --scenarios denoised
```

### Step 6 — Re-run fairness analysis on new denoised results

```bash
python fairness/scripts/run_inference_analyzers.py
python fairness/scripts/apply_advanced_metrics_to_results.py
```

Target: denoised RMSE ratios should drop to the same level as noisy (≤ 1.1 for all groups).

---

## Which Option First?

| Option | Effort | Impact | Recommendation |
|---|---|---|---|
| A — Adaptive denoising | Medium (1–2 days) | Fixes root cause permanently | **Start here** |
| B — GroupRegularized fine-tuning | Small (half day) | Partial mitigation at model level | After A, or if A insufficient |

Do A first: fixing the data preprocessing is more principled and avoids a "fairness patch" on top of biased data.

---

## Expected Outcome

- Denoised RMSE ratio for Cohort: **1.900 → ≤ 1.2** 🟡
- Denoised RMSE ratio for Gender: **1.506 → ≤ 1.1** 🟢
- Denoised EO Gap for Age: **0.623 → ≤ 0.20** 🟡
- Noisy data fairness must remain unchanged (validate as a regression test)

---

## Risk

Adaptive smoothing may slightly reduce overall prediction accuracy on denoised data because
it preserves more noise in noisier patients.  Run a standard RMSE regression test before
and after to ensure overall accuracy doesn't degrade more than 3%.
