#!/usr/bin/env python3
"""Aggregate five-seed BG results for the original Teacher and no-KD Student."""

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


INFERENCE_SUBPATH = Path("per_patient_inference") / "time_llm_per_patient_inference_ohiot1dm"
CONDITIONS = (
    ("Original Teacher (BERT)", Path("phase_1_teacher")),
    ("Student trained without distillation (BERT-tiny)", Path("phase_2_student")),
)


def load_fixed_seeds() -> list[int]:
    from seeds import fixed_seeds

    return list(fixed_seeds)


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def summarize(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate five-seed BG original Teacher and no-KD Student results."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--calibration-mode", default="patient-holdout")
    parser.add_argument("--calibration-folds", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else load_fixed_seeds()
    rows: list[dict[str, str | int | float]] = []
    summaries: list[str] = []

    for method, phase in CONDITIONS:
        metric_values = {"rmse": [], "eo_raw": [], "eo_cal": []}
        completed_seeds: list[int] = []
        for seed in seeds:
            inference_dir = args.suite_dir / f"seed_{seed}" / phase / INFERENCE_SUBPATH
            result = analyze_run(
                f"{method} [seed {seed}]",
                inference_dir,
                args.calibration_mode,
                args.calibration_folds,
            )
            if result is None:
                continue
            metric_values["rmse"].append(result["rmse"])
            metric_values["eo_raw"].append(result["eo_gap_raw"])
            metric_values["eo_cal"].append(result["eo_gap_calibrated"])
            completed_seeds.append(seed)

        if not completed_seeds:
            continue

        rmse_mean, rmse_std = summarize(metric_values["rmse"])
        eo_raw_mean, eo_raw_std = summarize(metric_values["eo_raw"])
        eo_cal_mean, eo_cal_std = summarize(metric_values["eo_cal"])
        rows.append(
            {
                "method": method,
                "n_seeds": len(completed_seeds),
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "eo_raw_mean": eo_raw_mean,
                "eo_raw_std": eo_raw_std,
                "eo_cal_mean": eo_cal_mean,
                "eo_cal_std": eo_cal_std,
                "eo_raw_per_seed": ";".join(f"{value:.6f}" for value in metric_values["eo_raw"]),
            }
        )
        summaries.extend(
            [
                f"{method}: {len(completed_seeds)}/{len(seeds)} seeds complete",
                f"  Seeds: {completed_seeds}",
                f"  RMSE: {format_metric(rmse_mean, rmse_std)}",
                f"  EO_raw: {format_metric(eo_raw_mean, eo_raw_std)}",
                f"  EO_cal: {format_metric(eo_cal_mean, eo_cal_std)}",
                "",
            ]
        )

    if not rows:
        raise SystemExit("No completed original Teacher or no-KD Student inference trees were found.")

    args.suite_dir.mkdir(parents=True, exist_ok=True)
    text_path = args.suite_dir / "teacher_student_multiseed_results.txt"
    csv_path = args.suite_dir / "teacher_student_multiseed_results.csv"
    text_path.write_text(
        "BG five-seed reference baselines\n"
        "Methods: original BERT Teacher; independently trained BERT-tiny Student\n"
        f"Calibration: {args.calibration_mode} (folds={args.calibration_folds})\n\n"
        + "\n".join(summaries),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "n_seeds",
                "rmse_mean",
                "rmse_std",
                "eo_raw_mean",
                "eo_raw_std",
                "eo_cal_mean",
                "eo_cal_std",
                "eo_raw_per_seed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {text_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()