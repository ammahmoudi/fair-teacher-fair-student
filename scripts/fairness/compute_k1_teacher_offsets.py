#!/usr/bin/env python3
"""
Compute K1 teacher calibration offsets from an inference directory.

K1 idea: during distillation, shift teacher outputs by group so that a fixed
hypoglycemia decision threshold (70 mg/dL) better matches equalized TPR behavior.

Offset convention (used by distillation trainer):
  shifted_teacher = teacher_output + group_offset

If calibrated threshold for a group is t_g, then equivalent offset at base
threshold t0 is:
  offset_g = t0 - t_g
because (pred + offset_g < t0) <=> (pred < t_g)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fairness.utils.analyzer_utils import get_ohiot1dm_default_data

HYPO_THRESHOLD = 70.0


def _group_from_patient(pid: str, feature: str, demo: dict) -> str:
    val = demo.get(pid, {}).get(feature)
    if val is None:
        return "Unknown"
    return str(val)


def load_window_data(inference_dir: Path, feature: str):
    demo = get_ohiot1dm_default_data()
    preds, targets, groups = [], [], []

    for csv_path in sorted(inference_dir.rglob("inference_results_reformatted.csv")):
        patient_id = next((p.replace("patient_", "") for p in csv_path.parts if p.startswith("patient_")), None)
        if not patient_id:
            continue

        group = _group_from_patient(patient_id, feature, demo)
        if group == "Unknown":
            continue

        df = pd.read_csv(csv_path)
        true_cols = [c for c in df.columns if c.endswith("_true")]
        pred_cols = [c for c in df.columns if c.endswith("_pred")]

        for _, row in df.iterrows():
            for tc, pc in zip(true_cols, pred_cols):
                targets.append(float(row[tc]))
                preds.append(float(row[pc]))
                groups.append(group)

    return np.array(preds), np.array(targets), np.array(groups)


def tpr_at_threshold(preds, targets, thresh):
    hypo = targets < HYPO_THRESHOLD
    if hypo.sum() == 0:
        return 0.0
    return float((preds[hypo] < thresh).mean())


def calibrate_threshold(preds, targets, target_tpr, lo=40.0, hi=200.0):
    try:
        return float(brentq(lambda t: tpr_at_threshold(preds, targets, t) - target_tpr, lo, hi))
    except ValueError:
        return HYPO_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="Compute K1 teacher group offsets")
    parser.add_argument("--inference-dir", required=True,
                        help="Teacher inference directory containing per-patient inference_results_reformatted.csv files")
    parser.add_argument("--feature", default="gender", choices=["gender", "age", "pump", "sensor", "cohort"],
                        help="Demographic feature used for K1 groups (default: gender)")
    parser.add_argument("--group0", default="Female",
                        help="Name of group mapped to label 0 in distillation (default: Female)")
    parser.add_argument("--group1", default="Male",
                        help="Name of group mapped to label 1 in distillation (default: Male)")
    parser.add_argument("--base-threshold", type=float, default=70.0,
                        help="Base hypo threshold used in fairness evaluation (default: 70)")
    parser.add_argument("--output", required=True,
                        help="Path to output JSON with offsets")
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    preds, targets, groups = load_window_data(inference_dir, args.feature)
    if len(preds) == 0:
        raise RuntimeError(f"No inference windows found in {inference_dir}")

    g0_mask = groups == args.group0
    g1_mask = groups == args.group1
    if g0_mask.sum() == 0 or g1_mask.sum() == 0:
        raise RuntimeError(
            f"Missing one group in inference data: group0={args.group0} ({g0_mask.sum()}), "
            f"group1={args.group1} ({g1_mask.sum()})"
        )

    tpr0 = tpr_at_threshold(preds[g0_mask], targets[g0_mask], args.base_threshold)
    tpr1 = tpr_at_threshold(preds[g1_mask], targets[g1_mask], args.base_threshold)
    target_tpr = (tpr0 + tpr1) / 2.0

    t0 = calibrate_threshold(preds[g0_mask], targets[g0_mask], target_tpr)
    t1 = calibrate_threshold(preds[g1_mask], targets[g1_mask], target_tpr)

    offset0 = args.base_threshold - t0
    offset1 = args.base_threshold - t1

    out = {
        "feature": args.feature,
        "group0_name": args.group0,
        "group1_name": args.group1,
        "group0_label": 0,
        "group1_label": 1,
        "base_threshold": args.base_threshold,
        "group0_calibrated_threshold": round(t0, 4),
        "group1_calibrated_threshold": round(t1, 4),
        "group0_offset": round(offset0, 4),
        "group1_offset": round(offset1, 4),
        "raw_tpr_group0": round(tpr0, 6),
        "raw_tpr_group1": round(tpr1, 6),
        "target_tpr": round(target_tpr, 6),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("K1 teacher offsets computed")
    print(f"  feature={args.feature}")
    print(f"  group0 ({args.group0}) threshold={t0:.3f} -> offset={offset0:+.3f}")
    print(f"  group1 ({args.group1}) threshold={t1:.3f} -> offset={offset1:+.3f}")
    print(f"  saved={out_path}")


if __name__ == "__main__":
    main()
