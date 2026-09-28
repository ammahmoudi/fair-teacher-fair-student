#!/usr/bin/env python3
"""Aggregate multi-seed robustness results for the fairness comparison.

For each (method, seed) inference directory, this reuses analyze_run() from
compute_fairness_comparison.py to compute RMSE / EO_raw / EO_cal, then reports
mean ± std across seeds per method. Used to confirm the O2 headline result is
not an artifact of a single seed.

Seeds come from scripts/utilities/seeds.py (fixed_seeds) — the canonical seed
reference for the project — unless --seeds is given.

Layout expected (one inference dir per method per seed):
  <phase3>/<run_name>_seed<seed>/per_patient_inference/time_llm_per_patient_inference_ohiot1dm
"""
import sys
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "utilities"))

from scripts.fairness.compute_fairness_comparison import analyze_run  # noqa: E402

# Methods to validate. Each entry's stem is the full run-dir name a seed run
# produces, INCLUDING the _seed<seed> tag appended via --dir-suffix in the runner.
# The {seed} placeholder is filled per seed. Stems mirror the dir_suffix logic in
# distill_students.py: baseline → no fairness suffix; T1/O2 → _fair_teacher.
METHODS = [
    ("Baseline KD (no fairness)", "bert_to_bert-tiny_all_patients_seed{seed}"),
    ("Distilled from Fair Teacher (T1)", "bert_to_bert-tiny_all_patients_fair_teacher_seed{seed}"),
    ("Standalone O2 Calibration Head", "bert_to_bert-tiny_all_patients_o2_gender_seed{seed}"),
    ("Distilled from Fair Teacher + O2 Calibration Head", "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed{seed}"),
]

INFER_SUBPATH = Path("per_patient_inference") / "time_llm_per_patient_inference_ohiot1dm"


def load_fixed_seeds():
    from seeds import fixed_seeds
    return list(fixed_seeds)


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", required=True)
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds (default: fixed_seeds from scripts/utilities/seeds.py)")
    ap.add_argument("--calibration-mode", default="patient-holdout")
    ap.add_argument("--calibration-folds", type=int, default=2)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else load_fixed_seeds()
    phase3 = Path(args.pipeline_dir) / "phase_3_distillation"

    print("=" * 78)
    print("MULTI-SEED ROBUSTNESS — Fairness Comparison")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Calibration: {args.calibration_mode} (folds={args.calibration_folds})")
    print("=" * 78)

    summary_rows = []
    for label, stem in METHODS:
        per_seed = {"rmse": [], "eo_raw": [], "eo_cal": []}
        used_seeds = []
        for seed in seeds:
            inf_dir = phase3 / stem.format(seed=seed) / INFER_SUBPATH
            r = analyze_run(f"{label} [seed {seed}]", inf_dir,
                            args.calibration_mode, args.calibration_folds)
            if r is None:
                continue
            per_seed["rmse"].append(r["rmse"])
            per_seed["eo_raw"].append(r["eo_gap_raw"])
            per_seed["eo_cal"].append(r["eo_gap_calibrated"])
            used_seeds.append(seed)

        n = len(used_seeds)
        if n == 0:
            print(f"\n  {label}: no seed runs found (looked for {stem.format(seed='<seed>')})")
            continue

        def ms(key):
            xs = per_seed[key]
            mean = statistics.mean(xs)
            std = statistics.pstdev(xs) if n > 1 else 0.0
            return mean, std

        rmse_m, rmse_s = ms("rmse")
        eor_m, eor_s = ms("eo_raw")
        eoc_m, eoc_s = ms("eo_cal")
        summary_rows.append((label, n, rmse_m, rmse_s, eor_m, eor_s, eoc_m, eoc_s,
                             per_seed["eo_raw"]))
        print(f"\n  {label}  (n={n} seeds: {used_seeds})")
        print(f"    RMSE   = {fmt(rmse_m, rmse_s)}")
        print(f"    EO_raw = {fmt(eor_m, eor_s)}   per-seed: {[round(x,4) for x in per_seed['eo_raw']]}")
        print(f"    EO_cal = {fmt(eoc_m, eoc_s)}")

    # Compact table
    # Fixed 16-wide metric columns so mean±std strings (~13 chars) never touch.
    hdr = f"{'Method':<46}{'RMSE':>16}{'EO_raw':>16}{'EO_cal':>16}"
    print("\n" + "=" * 94)
    print(hdr)
    print("-" * 94)
    for label, n, rmse_m, rmse_s, eor_m, eor_s, eoc_m, eoc_s, _ in summary_rows:
        print(f"{label[:45]:<46}{fmt(rmse_m,rmse_s):>16}{fmt(eor_m,eor_s):>16}{fmt(eoc_m,eoc_s):>16}")
    print("-" * 94)
    print("Values are mean ± std (population) across seeds. EO_raw is the headline fairness metric.")

    out = Path(args.pipeline_dir) / "multiseed_robustness_results.txt"
    csv_out = Path(args.pipeline_dir) / "multiseed_robustness_results.csv"
    with open(out, "w") as f:
        f.write(f"Multi-seed robustness — seeds={seeds}, calibration={args.calibration_mode}/{args.calibration_folds}\n\n")
        f.write(hdr + "\n")
        for label, n, rmse_m, rmse_s, eor_m, eor_s, eoc_m, eoc_s, eo_raw_list in summary_rows:
            f.write(f"{label[:45]:<46}{fmt(rmse_m,rmse_s):>16}{fmt(eor_m,eor_s):>16}{fmt(eoc_m,eoc_s):>16}\n")
            f.write(f"    (n={n}) per-seed EO_raw: {[round(x,4) for x in eo_raw_list]}\n")
    print(f"\n✅ Saved: {out}")

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "n_seeds",
            "rmse_mean", "rmse_std",
            "eo_raw_mean", "eo_raw_std",
            "eo_cal_mean", "eo_cal_std",
            "eo_raw_per_seed",
        ])
        for label, n, rmse_m, rmse_s, eor_m, eor_s, eoc_m, eoc_s, eo_raw_list in summary_rows:
            writer.writerow([
                label, n,
                f"{rmse_m:.6f}", f"{rmse_s:.6f}",
                f"{eor_m:.6f}", f"{eor_s:.6f}",
                f"{eoc_m:.6f}", f"{eoc_s:.6f}",
                ";".join(f"{x:.6f}" for x in eo_raw_list),
            ])
    print(f"✅ Saved: {csv_out}")


if __name__ == "__main__":
    main()
