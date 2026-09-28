#!/usr/bin/env python3
"""
Compute full fairness comparison table across all distillation runs.

Produces:
  - fairness_comparison_results.csv  — machine-readable table
  - fairness_comparison_results.txt  — human-readable table for paper

Usage:
    python scripts/fairness/compute_fairness_comparison.py \
        --pipeline-dir distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Patient demographics — single source of truth from fairness module ───────
from fairness.utils.analyzer_utils import get_ohiot1dm_default_data as _get_demo
GENDER = {pid: d['gender'] for pid, d in _get_demo().items()}
HYPO_THRESHOLD = 70.0  # mg/dL


# ── Data loading ─────────────────────────────────────────────────────────────

def load_window_data(inference_dir: Path):
    """Load all per-patient inference_results_reformatted.csv into a flat dataframe."""
    records = []
    for csv_path in sorted(inference_dir.rglob('inference_results_reformatted.csv')):
        patient_id = next(
            (p.replace('patient_', '') for p in csv_path.parts if p.startswith('patient_')),
            None
        )
        if not patient_id:
            continue
        gender = GENDER.get(patient_id, 'Unknown')
        df = pd.read_csv(csv_path)
        true_cols = [c for c in df.columns if c.endswith('_true')]
        pred_cols = [c for c in df.columns if c.endswith('_pred')]
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for tc, pc in zip(true_cols, pred_cols):
                records.append({
                    'patient_id': patient_id,
                    'group': gender,
                    'row_idx': row_idx,
                    'target': float(row[tc]),
                    'pred': float(row[pc]),
                })
    return pd.DataFrame.from_records(records)


def arrays_from_frame(window_df: pd.DataFrame):
    """Extract prediction, target, and group arrays from a window dataframe."""
    return (
        window_df['pred'].to_numpy(dtype=float),
        window_df['target'].to_numpy(dtype=float),
        window_df['group'].to_numpy(dtype=object),
    )


def load_patient_rmse(inference_dir: Path):
    """Load per-patient RMSE from experiment_results.csv."""
    csv = next(inference_dir.rglob('experiment_results.csv'), None)
    if csv is None:
        return None
    df = pd.read_csv(csv)
    df['patient_id'] = df['patient_id'].astype(str)
    df['gender'] = df['patient_id'].map(GENDER)
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def tpr_at_threshold(preds, targets, thresh):
    hypo = targets < HYPO_THRESHOLD
    if hypo.sum() == 0:
        return 0.0
    return float((preds[hypo] < thresh).mean())


def tpr_at_row_thresholds(preds, targets, thresholds):
    hypo = targets < HYPO_THRESHOLD
    if hypo.sum() == 0:
        return 0.0
    return float((preds[hypo] < thresholds[hypo]).mean())


def eo_gap(preds, targets, groups, thresh_male=HYPO_THRESHOLD, thresh_female=HYPO_THRESHOLD):
    t = {'Male': thresh_male, 'Female': thresh_female}
    tprs = {g: tpr_at_threshold(preds[groups == g], targets[groups == g], t[g])
            for g in ['Male', 'Female']}
    return abs(tprs['Male'] - tprs['Female']), tprs


def fit_equalized_thresholds(preds, targets, groups):
    """Fit per-gender thresholds that equalize TPR to the midpoint target."""
    tpr_m = tpr_at_threshold(preds[groups == 'Male'],   targets[groups == 'Male'],   HYPO_THRESHOLD)
    tpr_f = tpr_at_threshold(preds[groups == 'Female'], targets[groups == 'Female'], HYPO_THRESHOLD)
    target_tpr = (tpr_m + tpr_f) / 2.0

    def find_t(g, lo=40, hi=200):
        pm, tm = preds[groups == g], targets[groups == g]
        try:
            return brentq(lambda t: tpr_at_threshold(pm, tm, t) - target_tpr, lo, hi)
        except ValueError:
            return HYPO_THRESHOLD

    t_male = find_t('Male')
    t_female = find_t('Female')
    return t_male, t_female, target_tpr


def calibrate_thresholds_in_sample(preds, targets, groups):
    """Legacy in-sample calibration for direct comparison with older reports."""
    t_male, t_female, _ = fit_equalized_thresholds(preds, targets, groups)
    gap_cal, tprs_cal = eo_gap(preds, targets, groups, t_male, t_female)
    return {
        'method': 'in_sample',
        'thresh_male_cal': t_male,
        'thresh_female_cal': t_female,
        'thresh_male_std': 0.0,
        'thresh_female_std': 0.0,
        'gap_cal': gap_cal,
        'tprs_cal': tprs_cal,
    }


def calibrate_thresholds_holdout(window_df: pd.DataFrame, num_folds: int = 2):
    """Fit thresholds on held-out patients and evaluate only on disjoint patients."""
    if num_folds < 2:
        raise ValueError('patient-holdout calibration requires at least 2 folds')

    patient_folds = {}
    for group in ['Male', 'Female']:
        patient_ids = sorted(window_df.loc[window_df['group'] == group, 'patient_id'].unique())
        if len(patient_ids) < num_folds:
            raise ValueError(
                f'Not enough {group} patients ({len(patient_ids)}) for {num_folds}-fold holdout calibration'
            )
        patient_folds[group] = [list(fold) for fold in np.array_split(patient_ids, num_folds)]

    fold_frames = []
    male_thresholds = []
    female_thresholds = []

    for fold_idx in range(num_folds):
        eval_mask = pd.Series(False, index=window_df.index)
        for group in ['Male', 'Female']:
            eval_patients = patient_folds[group][fold_idx]
            eval_mask |= (window_df['group'] == group) & window_df['patient_id'].isin(eval_patients)

        calib_df = window_df.loc[~eval_mask]
        eval_df = window_df.loc[eval_mask].copy()
        if calib_df.empty or eval_df.empty:
            raise ValueError(f'Fold {fold_idx + 1} produced an empty calibration or evaluation split')

        t_male, t_female, _ = fit_equalized_thresholds(*arrays_from_frame(calib_df))
        male_thresholds.append(t_male)
        female_thresholds.append(t_female)

        eval_df['applied_threshold'] = np.where(eval_df['group'] == 'Male', t_male, t_female)
        eval_df['calibration_fold'] = fold_idx + 1
        fold_frames.append(eval_df)

    eval_df = pd.concat(fold_frames, ignore_index=True)
    tprs_cal = {}
    for group in ['Male', 'Female']:
        group_df = eval_df.loc[eval_df['group'] == group]
        tprs_cal[group] = tpr_at_row_thresholds(
            group_df['pred'].to_numpy(dtype=float),
            group_df['target'].to_numpy(dtype=float),
            group_df['applied_threshold'].to_numpy(dtype=float),
        )
    gap_cal = abs(tprs_cal['Male'] - tprs_cal['Female'])

    return {
        'method': f'patient_holdout_{num_folds}fold',
        'thresh_male_cal': float(np.mean(male_thresholds)),
        'thresh_female_cal': float(np.mean(female_thresholds)),
        'thresh_male_std': float(np.std(male_thresholds)),
        'thresh_female_std': float(np.std(female_thresholds)),
        'gap_cal': gap_cal,
        'tprs_cal': tprs_cal,
    }


def calibrate_thresholds(window_df: pd.DataFrame, mode: str, num_folds: int):
    preds, targets, groups = arrays_from_frame(window_df)
    if mode == 'in-sample':
        return calibrate_thresholds_in_sample(preds, targets, groups)
    if mode == 'patient-holdout':
        return calibrate_thresholds_holdout(window_df, num_folds=num_folds)
    raise ValueError(f'Unsupported calibration mode: {mode}')


def assess(eo):
    if eo < 0.05:  return "✅ EXCELLENT"
    if eo < 0.10:  return "✅ GOOD"
    if eo < 0.15:  return "⚠️ MODERATE"
    if eo < 0.20:  return "⚠️ CONCERNING"
    return "❌ CRITICAL"


# ── Run analysis for one model ────────────────────────────────────────────────

def analyze_run(label, inference_dir, calibration_mode, calibration_folds):
    print(f"\n  Analyzing: {label}")
    inference_dir = Path(inference_dir)
    if not inference_dir.exists():
        print(f"    ⚠️  Directory not found: {inference_dir}")
        return None

    window_df = load_window_data(inference_dir)
    if window_df.empty:
        print(f"    ⚠️  No window data found in {inference_dir}")
        return None

    preds, targets, groups = arrays_from_frame(window_df)

    overall_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    overall_mae  = float(np.mean(np.abs(preds - targets)))

    # Per-gender RMSE
    rmse_male   = float(np.sqrt(np.mean((preds[groups=='Male']   - targets[groups=='Male'])   ** 2)))
    rmse_female = float(np.sqrt(np.mean((preds[groups=='Female'] - targets[groups=='Female']) ** 2)))

    # Raw EO Gap
    gap_raw, tprs_raw = eo_gap(preds, targets, groups)

    # Leakage-free calibrated EO Gap
    calibration = calibrate_thresholds(window_df, calibration_mode, calibration_folds)

    result = {
        'label':             label,
        'calibration_method': calibration['method'],
        'rmse':              round(overall_rmse, 3),
        'mae':               round(overall_mae,  3),
        'rmse_male':         round(rmse_male,   3),
        'rmse_female':       round(rmse_female, 3),
        'tpr_male_raw':      round(tprs_raw['Male'],   4),
        'tpr_female_raw':    round(tprs_raw['Female'], 4),
        'eo_gap_raw':        round(gap_raw, 4),
        'assess_raw':        assess(gap_raw),
        'thresh_male_cal':   round(calibration['thresh_male_cal'], 1),
        'thresh_female_cal': round(calibration['thresh_female_cal'], 1),
        'thresh_male_std':   round(calibration['thresh_male_std'], 3),
        'thresh_female_std': round(calibration['thresh_female_std'], 3),
        'tpr_male_cal':      round(calibration['tprs_cal']['Male'],   4),
        'tpr_female_cal':    round(calibration['tprs_cal']['Female'], 4),
        'eo_gap_calibrated': round(calibration['gap_cal'], 4),
        'assess_cal':        assess(calibration['gap_cal']),
    }
    print(
        f"    RMSE={overall_rmse:.3f}  EO_raw={gap_raw:.4f}  "
        f"EO_cal={calibration['gap_cal']:.4f} ({calibration['method']})"
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute fairness comparison table")
    parser.add_argument("--pipeline-dir", required=True,
                        help="Path to pipeline directory (e.g. distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17)")
    parser.add_argument(
        "--calibration-mode",
        choices=["patient-holdout", "in-sample"],
        default="patient-holdout",
        help="How to fit per-gender thresholds before reporting calibrated EO",
    )
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=2,
        help="Number of patient-holdout folds per gender when using patient-holdout calibration",
    )
    args = parser.parse_args()

    pipeline = Path(args.pipeline_dir)
    phase1   = pipeline / "phase_1_teacher"
    phase2   = pipeline / "phase_2_student"
    phase3   = pipeline / "phase_3_distillation"
    p1_inf   = phase1 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"
    p1_fair_inf = phase1 / "per_patient_inference_fair_teacher" / "time_llm_per_patient_inference_ohiot1dm"
    p2_inf   = phase2 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"

    # All runs to compare
    runs = [
        ("Teacher baseline (BERT)",
         p1_inf),
        ("Teacher fair sampling (T1 teacher)",
         p1_fair_inf),
        ("Student baseline (no KD)",
         p2_inf),
        ("Distilled — no fairness",
         phase3 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + HypoglycemiaTPR loss (v1, EqualizedOdds)",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + HypoglycemiaTPR loss (v2, focal+soft)",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender_v2" / "per_patient_inference"),
        ("Distilled + Oversampling only",
         phase3 / "bert_to_bert-tiny_all_patients_oversample_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + Oversampling + HypoglycemiaTPR loss",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender_oversample" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled from Fair Teacher (T1)",
         phase3 / "bert_to_bert-tiny_all_patients_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled from Fair Teacher + O1 Constraint",
         phase3 / "bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled from Fair Teacher + K1 Calibrated Soft Labels",
         phase3 / "bert_to_bert-tiny_all_patients_k1cal_gender_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled from Fair Teacher + O2 Calibration Head",
         phase3 / "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
    ]

    # K3 feature alignment: one run directory per alignment weight
    # (..._k3align_gender_w<weight>_fair_teacher). Discover them all and add a
    # labeled row per weight, sorted by weight, so a weight sweep self-populates
    # the table. The older unweighted dir (if present) is included as "w?".
    def _k3_weight_key(run_dir):
        m = re.search(r"_w([0-9p]+)_fair_teacher$", run_dir.name)
        if not m:
            return float("inf")  # legacy unweighted dir sorts last
        return float(m.group(1).replace("p", "."))

    k3_dirs = sorted(
        list(phase3.glob("bert_to_bert-tiny_all_patients_k3align_gender_w*_fair_teacher"))
        + list(phase3.glob("bert_to_bert-tiny_all_patients_k3align_gender_fair_teacher")),
        key=_k3_weight_key,
    )
    for d in k3_dirs:
        m = re.search(r"_w([0-9p]+)_fair_teacher$", d.name)
        weight_str = m.group(1).replace("p", ".") if m else "?"
        runs.append((
            f"Distilled from Fair Teacher + K3 Feature Alignment (w={weight_str})",
            d / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
        ))

    runs.append((
        "Distilled from Fair Teacher + K4 Selective KD Replay",
        phase3 / "bert_to_bert-tiny_all_patients_k4replay_gender_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
    ))
    runs.append((
        "Distilled from Fair Teacher + O3 Adversarial Erasure",
        phase3 / "bert_to_bert-tiny_all_patients_o3adv_gender_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
    ))
    runs.append((
        "Distilled from Per-Group Teachers (T2)",
        phase3 / "bert_to_bert-tiny_all_patients_t2pergroup_gender_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
    ))
    runs.append((
        "Distilled + K1 Calibrated Soft Labels",
        phase3 / "bert_to_bert-tiny_all_patients_k1cal_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
    ))

    print("\n" + "="*70)
    print("FAIRNESS COMPARISON TABLE")
    print("="*70)

    results = []
    for label, inf_dir in runs:
        r = analyze_run(label, inf_dir, args.calibration_mode, args.calibration_folds)
        if r:
            results.append(r)

    if not results:
        print("❌ No results found!")
        return

    df = pd.DataFrame(results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = pipeline / "fairness_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV saved: {csv_path}")

    # ── Print human-readable table ────────────────────────────────────────────
    txt_path = pipeline / "fairness_comparison_results.txt"
    lines = []
    lines.append("FAIRNESS COMPARISON — Gender EO Gap (Hypoglycemia TPR Disparity)")
    lines.append("All-Patients Distillation Pipeline  |  OhioT1DM  |  BERT→BERT-tiny")
    lines.append(f"Calibration method: {args.calibration_mode} (folds={args.calibration_folds})")
    lines.append("="*90)
    header = f"{'Model':<48} {'RMSE':>6}  {'EO_raw':>8}  {'EO_cal':>8}  {'Assessment (raw)'}"
    lines.append(header)
    lines.append("-"*90)
    for r in results:
        line = (f"{r['label']:<48} {r['rmse']:6.3f}  {r['eo_gap_raw']:8.4f}  "
                f"{r['eo_gap_calibrated']:8.4f}  {r['assess_raw']}")
        lines.append(line)
    lines.append("-"*90)
    lines.append("\nNotes:")
    lines.append("  EO_raw = EO Gap with uniform threshold=70 mg/dL for all groups")
    lines.append("  EO_cal = EO Gap after per-gender threshold calibration")
    lines.append("  patient-holdout mode fits thresholds on one patient fold and evaluates on different patients")
    lines.append("  EO Gap = |Male_TPR - Female_TPR| for hypoglycemia detection")
    lines.append("  Lower EO Gap = fairer model")
    lines.append("")
    lines.append("Per-gender thresholds after calibration:")
    for r in results:
        lines.append(
            f"  {r['label'][:45]:<45}  Male thresh={r['thresh_male_cal']:5.1f}±{r['thresh_male_std']:.2f}  "
            f"Female thresh={r['thresh_female_cal']:5.1f}±{r['thresh_female_std']:.2f}"
        )

    txt = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(txt)

    print("\n" + txt)
    print(f"\n✅ Text table saved: {txt_path}")


if __name__ == "__main__":
    main()
