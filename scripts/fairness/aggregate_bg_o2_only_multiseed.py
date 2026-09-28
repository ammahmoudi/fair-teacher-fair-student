#!/usr/bin/env python3
"""Aggregate the standalone O2-only five-seed BG ablation."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "utilities"))

from scripts.fairness.compute_fairness_comparison import analyze_run


METHOD = "Standalone O2 calibration head"
RUN_NAME = "bert_to_bert-tiny_all_patients_o2_gender_seed{seed}"
INFERENCE_SUBPATH = Path("per_patient_inference") / "time_llm_per_patient_inference_ohiot1dm"


def load_fixed_seeds() -> list[int]:
    from seeds import fixed_seeds

    return list(fixed_seeds)


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the standalone O2-only BG ablation.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--calibration-mode", default="patient-holdout")
    parser.add_argument("--calibration-folds", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else load_fixed_seeds()
    phase3_dir = args.suite_dir / "phase_3_distillation"
    values = {"rmse": [], "eo_raw": [], "eo_cal": []}
    used_seeds: list[int] = []

    for seed in seeds:
        inference_dir = phase3_dir / RUN_NAME.format(seed=seed) / INFERENCE_SUBPATH
        result = analyze_run(
            f"{METHOD} [seed {seed}]",
            inference_dir,
            args.calibration_mode,
            args.calibration_folds,
        )
        if result is None:
            continue
        values["rmse"].append(result["rmse"])
        values["eo_raw"].append(result["eo_gap_raw"])
        values["eo_cal"].append(result["eo_gap_calibrated"])
        used_seeds.append(seed)

    if not used_seeds:
        raise SystemExit("No completed standalone O2-only inference trees were found.")

    count = len(used_seeds)

    def summarize(metric: str) -> tuple[float, float]:
        entries = values[metric]
        return statistics.mean(entries), statistics.pstdev(entries) if count > 1 else 0.0

    rmse_mean, rmse_std = summarize("rmse")
    eo_raw_mean, eo_raw_std = summarize("eo_raw")
    eo_cal_mean, eo_cal_std = summarize("eo_cal")

    print(f"Standalone O2-only BG ablation: {count}/{len(seeds)} seeds complete")
    print(f"RMSE: {format_metric(rmse_mean, rmse_std)}")
    print(f"EO_raw: {format_metric(eo_raw_mean, eo_raw_std)}")
    print(f"EO_cal: {format_metric(eo_cal_mean, eo_cal_std)}")

    text_path = args.suite_dir / "o2_only_multiseed_results.txt"
    csv_path = args.suite_dir / "o2_only_multiseed_results.csv"
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        "Standalone O2-only BG ablation\n"
        f"Seeds: {used_seeds}\n"
        f"Calibration: {args.calibration_mode} (folds={args.calibration_folds})\n\n"
        f"RMSE: {format_metric(rmse_mean, rmse_std)}\n"
        f"EO_raw: {format_metric(eo_raw_mean, eo_raw_std)}\n"
        f"EO_cal: {format_metric(eo_cal_mean, eo_cal_std)}\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "n_seeds",
                "seeds",
                "rmse_mean",
                "rmse_std",
                "eo_raw_mean",
                "eo_raw_std",
                "eo_cal_mean",
                "eo_cal_std",
                "eo_raw_per_seed",
            ]
        )
        writer.writerow(
            [
                METHOD,
                count,
                ";".join(str(seed) for seed in used_seeds),
                f"{rmse_mean:.6f}",
                f"{rmse_std:.6f}",
                f"{eo_raw_mean:.6f}",
                f"{eo_raw_std:.6f}",
                f"{eo_cal_mean:.6f}",
                f"{eo_cal_std:.6f}",
                ";".join(f"{value:.6f}" for value in values["eo_raw"]),
            ]
        )
    print(f"Saved: {text_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()