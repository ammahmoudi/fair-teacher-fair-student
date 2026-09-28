#!/usr/bin/env python3
"""Prepare reproducible paired BG prediction inputs for validated CEG/SEG analysis.

This script intentionally does not assign Clarke or Surveillance zones. The
repository's existing Plotly helpers use schematic bands, not a verified clinical
grid implementation. It discovers the new five-seed pipeline's per-patient
``inference_results_reformatted.csv`` files, validates their paired true/pred
columns, and writes pooled method/seed CSVs for a separately validated CEG/SEG
calculator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHODS = {
    "standard_kd": "phase_3_distillation_baseline",
    "ebtd": "phase_3_distillation_t1",
    "ebtd_gcoa": "phase_3_distillation_t1_o2",
}


def paired_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return matching t_<step>_true / t_<step>_pred column pairs."""
    pairs: list[tuple[str, str]] = []
    for column in frame.columns:
        if not column.endswith("_true"):
            continue
        prediction_column = column.removesuffix("_true") + "_pred"
        if prediction_column in frame.columns:
            pairs.append((column, prediction_column))
    if not pairs:
        raise ValueError("No paired t_<step>_true/t_<step>_pred columns found.")
    return pairs


def flatten_predictions(path: Path, method: str, seed: str, patient: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[pd.DataFrame] = []
    for true_column, prediction_column in paired_columns(frame):
        step = true_column.removeprefix("t_").removesuffix("_true")
        paired = frame[[true_column, prediction_column]].rename(
            columns={true_column: "true_glucose", prediction_column: "pred_glucose"}
        )
        paired["forecast_step"] = int(step)
        rows.append(paired)
    long_frame = pd.concat(rows, ignore_index=True).dropna()
    long_frame = long_frame.assign(method=method, seed=seed, patient=patient)
    return long_frame[["method", "seed", "patient", "forecast_step", "true_glucose", "pred_glucose"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare paired BG forecast inputs for validated CEG/SEG analysis."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        required=True,
        help="Five-seed BG pipeline directory containing seed_<value> directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for pooled method/seed paired-prediction CSVs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    written = 0

    for seed_dir in sorted(args.pipeline_dir.glob("seed_*")):
        for method, phase_dir in METHODS.items():
            files = sorted(seed_dir.glob(f"{phase_dir}/**/inference_results_reformatted.csv"))
            if not files:
                missing.append(f"{seed_dir.name}/{phase_dir}")
                continue
            patient_frames = []
            for prediction_file in files:
                patient = prediction_file.parent.parent.name
                patient_frames.append(
                    flatten_predictions(prediction_file, method, seed_dir.name, patient)
                )
            result = pd.concat(patient_frames, ignore_index=True)
            output = args.output_dir / f"{method}_{seed_dir.name}_paired_predictions.csv"
            result.to_csv(output, index=False)
            print(f"Wrote {len(result):,} paired values to {output}")
            written += 1

    if missing:
        print("Missing inference artifacts (pipeline may still be running):")
        for item in missing:
            print(f"  - {item}")
    if not written:
        raise SystemExit("No paired prediction files were found.")


if __name__ == "__main__":
    main()
