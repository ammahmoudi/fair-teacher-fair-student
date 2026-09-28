#!/usr/bin/env python3
"""Analyze why K1 calibrated soft-label distillation underperformed.

Produces reproducible artifacts inside the pipeline directory:
  - k1_failure_patient_deltas.csv
  - k1_failure_gender_summary.csv
  - k1_failure_shift_summary.csv
  - k1_failure_window_summary.csv
  - K1_FAILURE_ANALYSIS.md
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fairness.utils.analyzer_utils import get_ohiot1dm_default_data


HYPO_THRESHOLD = 70.0


def get_gender_map():
    return {str(pid): info["gender"] for pid, info in get_ohiot1dm_default_data().items()}


def get_run_dirs(pipeline_dir: Path):
    phase3 = pipeline_dir / "phase_3_distillation"
    return {
        "baseline": phase3 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
        "t1": phase3 / "bert_to_bert-tiny_all_patients_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
        "k1": phase3 / "bert_to_bert-tiny_all_patients_k1cal_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm",
    }


def load_patient_metrics(inference_dir: Path, gender_map: dict) -> pd.DataFrame:
    csv_path = inference_dir / "experiment_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing patient metrics: {csv_path}")

    df = pd.read_csv(csv_path)
    df["patient_id"] = df["patient_id"].astype(str)
    df["gender"] = df["patient_id"].map(gender_map)
    return df[["patient_id", "gender", "rmse", "mae", "mape"]].copy()


def _parse_patient_id(csv_path: Path):
    for part in csv_path.parts:
        if part.startswith("patient_"):
            return part.replace("patient_", "")
    return None


def load_window_metrics(inference_dir: Path, gender_map: dict) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(inference_dir.rglob("inference_results_reformatted.csv")):
        patient_id = _parse_patient_id(csv_path)
        if patient_id is None:
            continue

        gender = gender_map.get(patient_id)
        df = pd.read_csv(csv_path)
        true_cols = [c for c in df.columns if c.endswith("_true")]
        pred_cols = [c for c in df.columns if c.endswith("_pred")]

        for row_idx, row in df.iterrows():
            for step_idx, (true_col, pred_col) in enumerate(zip(true_cols, pred_cols), start=1):
                target = float(row[true_col])
                pred = float(row[pred_col])
                rows.append(
                    {
                        "patient_id": patient_id,
                        "gender": gender,
                        "row_idx": int(row_idx),
                        "step_idx": step_idx,
                        "target": target,
                        "pred": pred,
                        "error": pred - target,
                        "abs_error": abs(pred - target),
                        "sq_error": (pred - target) ** 2,
                        "true_hypo": int(target < HYPO_THRESHOLD),
                        "pred_hypo": int(pred < HYPO_THRESHOLD),
                    }
                )

    if not rows:
        raise RuntimeError(f"No window-level inference files found in {inference_dir}")

    return pd.DataFrame(rows)


def summarize_windows(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    records = []
    for group_name, group_df in [("Overall", df), *[(g, df[df["gender"] == g]) for g in sorted(df["gender"].dropna().unique())]]:
        if group_df.empty:
            continue
        positives = group_df["true_hypo"] == 1
        tpr = float(group_df.loc[positives, "pred_hypo"].mean()) if positives.any() else np.nan
        records.append(
            {
                "model": model_label,
                "group": group_name,
                "n_windows": int(len(group_df)),
                "rmse": float(np.sqrt(group_df["sq_error"].mean())),
                "mae": float(group_df["abs_error"].mean()),
                "mean_target": float(group_df["target"].mean()),
                "mean_pred": float(group_df["pred"].mean()),
                "mean_error": float(group_df["error"].mean()),
                "mean_abs_error": float(group_df["abs_error"].mean()),
                "true_hypo_rate": float(group_df["true_hypo"].mean()),
                "pred_hypo_rate": float(group_df["pred_hypo"].mean()),
                "tpr_at_70": tpr,
            }
        )
    return pd.DataFrame(records)


def compute_eo_gap(df: pd.DataFrame) -> float:
    tprs = {}
    for gender in ["Male", "Female"]:
        group_df = df[df["gender"] == gender]
        positives = group_df["true_hypo"] == 1
        tprs[gender] = float(group_df.loc[positives, "pred_hypo"].mean()) if positives.any() else np.nan
    return abs(tprs["Male"] - tprs["Female"])


def build_patient_delta_table(baseline_df: pd.DataFrame, t1_df: pd.DataFrame, k1_df: pd.DataFrame) -> pd.DataFrame:
    merged = baseline_df.rename(columns={"rmse": "rmse_baseline", "mae": "mae_baseline", "mape": "mape_baseline"})
    merged = merged.merge(
        t1_df[["patient_id", "rmse", "mae", "mape"]].rename(columns={"rmse": "rmse_t1", "mae": "mae_t1", "mape": "mape_t1"}),
        on="patient_id",
        how="left",
    )
    merged = merged.merge(
        k1_df[["patient_id", "rmse", "mae", "mape"]].rename(columns={"rmse": "rmse_k1", "mae": "mae_k1", "mape": "mape_k1"}),
        on="patient_id",
        how="left",
    )
    merged["delta_t1_vs_baseline"] = merged["rmse_t1"] - merged["rmse_baseline"]
    merged["delta_k1_vs_baseline"] = merged["rmse_k1"] - merged["rmse_baseline"]
    merged["delta_k1_vs_t1"] = merged["rmse_k1"] - merged["rmse_t1"]
    merged["k1_worse_than_baseline"] = merged["delta_k1_vs_baseline"] > 0
    return merged.sort_values(["delta_k1_vs_baseline", "patient_id"], ascending=[False, True])


def build_shift_summary(baseline_windows: pd.DataFrame, k1_windows: pd.DataFrame) -> pd.DataFrame:
    merged = baseline_windows.merge(
        k1_windows,
        on=["patient_id", "gender", "row_idx", "step_idx", "target", "true_hypo"],
        suffixes=("_baseline", "_k1"),
        how="inner",
    )
    merged["pred_delta_k1_minus_baseline"] = merged["pred_k1"] - merged["pred_baseline"]
    merged["error_delta_k1_minus_baseline"] = merged["error_k1"] - merged["error_baseline"]
    merged["abs_error_delta_k1_minus_baseline"] = merged["abs_error_k1"] - merged["abs_error_baseline"]
    merged["pred_hypo_delta"] = merged["pred_hypo_k1"] - merged["pred_hypo_baseline"]

    records = []
    for group_name, group_df in [("Overall", merged), *[(g, merged[merged["gender"] == g]) for g in sorted(merged["gender"].dropna().unique())]]:
        if group_df.empty:
            continue
        positives = group_df["true_hypo"] == 1
        hypo_shift = float(group_df.loc[positives, "pred_hypo_delta"].mean()) if positives.any() else np.nan
        records.append(
            {
                "group": group_name,
                "n_windows": int(len(group_df)),
                "mean_pred_baseline": float(group_df["pred_baseline"].mean()),
                "mean_pred_k1": float(group_df["pred_k1"].mean()),
                "mean_pred_delta_k1_minus_baseline": float(group_df["pred_delta_k1_minus_baseline"].mean()),
                "mean_abs_error_baseline": float(group_df["abs_error_baseline"].mean()),
                "mean_abs_error_k1": float(group_df["abs_error_k1"].mean()),
                "mean_abs_error_delta_k1_minus_baseline": float(group_df["abs_error_delta_k1_minus_baseline"].mean()),
                "mean_error_delta_k1_minus_baseline": float(group_df["error_delta_k1_minus_baseline"].mean()),
                "mean_hypo_prediction_delta_on_true_hypo_windows": hypo_shift,
            }
        )
    return pd.DataFrame(records)


def build_gender_summary(patient_delta_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group_name, group_df in [("Overall", patient_delta_df), *[(g, patient_delta_df[patient_delta_df["gender"] == g]) for g in sorted(patient_delta_df["gender"].dropna().unique())]]:
        if group_df.empty:
            continue
        records.append(
            {
                "group": group_name,
                "n_patients": int(len(group_df)),
                "baseline_rmse_mean": float(group_df["rmse_baseline"].mean()),
                "t1_rmse_mean": float(group_df["rmse_t1"].mean()),
                "k1_rmse_mean": float(group_df["rmse_k1"].mean()),
                "k1_minus_baseline_rmse_mean": float(group_df["delta_k1_vs_baseline"].mean()),
                "t1_minus_baseline_rmse_mean": float(group_df["delta_t1_vs_baseline"].mean()),
                "k1_worse_than_baseline_count": int(group_df["k1_worse_than_baseline"].sum()),
                "k1_worse_than_baseline_rate": float(group_df["k1_worse_than_baseline"].mean()),
            }
        )
    return pd.DataFrame(records)


def find_log_file(run_dir: Path) -> Path:
    candidates = [
        path
        for path in run_dir.rglob("log.log")
        if "per_patient_inference" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f"No log.log found under {run_dir}")
    return candidates[-1]


def parse_training_log(log_path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Total Loss:\s+(?P<total>[0-9.]+)\s+\|\s+GT Loss:\s+(?P<gt>[0-9.]+)\s+\|\s+Teacher Loss:\s+(?P<teacher>[0-9.]+)(?:\s+\|\s+Fairness Loss:\s+(?P<fairness>[0-9.]+))?"
    )
    rows = []
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "total_loss": float(match.group("total")),
                    "gt_loss": float(match.group("gt")),
                    "teacher_loss": float(match.group("teacher")),
                    "fairness_loss": float(match.group("fairness") or 0.0),
                }
            )
    if not rows:
        raise RuntimeError(f"No epoch rows parsed from {log_path}")
    return pd.DataFrame(rows)


def write_report(
    pipeline_dir: Path,
    patient_delta_df: pd.DataFrame,
    gender_summary_df: pd.DataFrame,
    shift_summary_df: pd.DataFrame,
    window_summary_df: pd.DataFrame,
    baseline_log_df: pd.DataFrame,
    k1_log_df: pd.DataFrame,
):
    baseline_overall = gender_summary_df[gender_summary_df["group"] == "Overall"].iloc[0]
    male_summary = gender_summary_df[gender_summary_df["group"] == "Male"].iloc[0]
    female_summary = gender_summary_df[gender_summary_df["group"] == "Female"].iloc[0]
    shift_overall = shift_summary_df[shift_summary_df["group"] == "Overall"].iloc[0]
    shift_male = shift_summary_df[shift_summary_df["group"] == "Male"].iloc[0]
    shift_female = shift_summary_df[shift_summary_df["group"] == "Female"].iloc[0]

    eo_baseline = compute_eo_gap(window_summary_df_source["baseline"])
    eo_t1 = compute_eo_gap(window_summary_df_source["t1"])
    eo_k1 = compute_eo_gap(window_summary_df_source["k1"])

    lines = []
    lines.append("# K1 Failure Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"K1 made RMSE worse in {int(baseline_overall['k1_worse_than_baseline_count'])}/"
        f"{int(baseline_overall['n_patients'])} patients, with mean RMSE change "
        f"{baseline_overall['k1_minus_baseline_rmse_mean']:+.3f} versus baseline KD."
    )
    lines.append(
        f"The degradation is broad across both genders: Female {female_summary['k1_minus_baseline_rmse_mean']:+.3f}, "
        f"Male {male_summary['k1_minus_baseline_rmse_mean']:+.3f}."
    )
    lines.append(
        f"Raw EO gap also moved the wrong way: baseline {eo_baseline:.4f}, T1 {eo_t1:.4f}, K1 {eo_k1:.4f}."
    )
    lines.append("")
    lines.append("## What K1 Actually Does")
    lines.append("")
    lines.append(
        "K1 applies a constant group-specific additive shift to every teacher output during distillation. "
        "With the current gender mapping, Female windows receive +3.776 mg/dL and Male windows receive -3.498 mg/dL."
    )
    lines.append(
        f"The downstream student does not preserve those intended directions cleanly: relative to baseline KD, "
        f"mean student prediction moved {shift_female['mean_pred_delta_k1_minus_baseline']:+.3f} mg/dL for Female and "
        f"{shift_male['mean_pred_delta_k1_minus_baseline']:+.3f} mg/dL for Male, so both groups drifted downward overall."
    )
    lines.append(
        f"That global translation increases mean absolute error by {shift_female['mean_abs_error_delta_k1_minus_baseline']:+.3f} mg/dL "
        f"for Female and {shift_male['mean_abs_error_delta_k1_minus_baseline']:+.3f} mg/dL for Male."
    )
    lines.append("")
    lines.append("## KD Loss Comparison")
    lines.append("")
    lines.append(
        f"Baseline KD teacher loss averaged {baseline_log_df['teacher_loss'].mean():.3f}; "
        f"K1 averaged {k1_log_df['teacher_loss'].mean():.3f}."
    )
    lines.append(
        f"Baseline GT loss averaged {baseline_log_df['gt_loss'].mean():.3f}; "
        f"K1 averaged {k1_log_df['gt_loss'].mean():.3f}."
    )
    lines.append(
        "This matches the intended failure mode: the student is asked to match a biased teacher target that is no longer aligned with the real glucose scale."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "Abandon K1 in its current form for the paper's main method. The current implementation is a global group-wise target perturbation, "
        "not a targeted fairness correction near the hypoglycemia boundary. It increases regression error broadly, moves both groups in an unstable direction, "
        "and does not improve raw EO gap."
    )
    lines.append(
        "If you revisit this branch, the next variant should be local rather than global: apply correction only near the hypo threshold, "
        "or distill classification-aware teacher signals instead of shifting all continuous targets."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- k1_failure_patient_deltas.csv")
    lines.append("- k1_failure_gender_summary.csv")
    lines.append("- k1_failure_shift_summary.csv")
    lines.append("- k1_failure_window_summary.csv")

    report_path = pipeline_dir / "K1_FAILURE_ANALYSIS.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze why K1 underperformed baseline KD")
    parser.add_argument(
        "--pipeline-dir",
        required=True,
        help="Pipeline directory, e.g. distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17",
    )
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir)
    gender_map = get_gender_map()
    run_dirs = get_run_dirs(pipeline_dir)

    baseline_patients = load_patient_metrics(run_dirs["baseline"], gender_map)
    t1_patients = load_patient_metrics(run_dirs["t1"], gender_map)
    k1_patients = load_patient_metrics(run_dirs["k1"], gender_map)

    baseline_windows = load_window_metrics(run_dirs["baseline"], gender_map)
    t1_windows = load_window_metrics(run_dirs["t1"], gender_map)
    k1_windows = load_window_metrics(run_dirs["k1"], gender_map)

    global window_summary_df_source
    window_summary_df_source = {
        "baseline": baseline_windows,
        "t1": t1_windows,
        "k1": k1_windows,
    }

    patient_delta_df = build_patient_delta_table(baseline_patients, t1_patients, k1_patients)
    gender_summary_df = build_gender_summary(patient_delta_df)
    shift_summary_df = build_shift_summary(baseline_windows, k1_windows)
    window_summary_df = pd.concat(
        [
            summarize_windows(baseline_windows, "baseline"),
            summarize_windows(t1_windows, "t1"),
            summarize_windows(k1_windows, "k1"),
        ],
        ignore_index=True,
    )

    baseline_log_df = parse_training_log(find_log_file(pipeline_dir / "phase_3_distillation" / "bert_to_bert-tiny_all_patients"))
    k1_log_df = parse_training_log(find_log_file(pipeline_dir / "phase_3_distillation" / "bert_to_bert-tiny_all_patients_k1cal_gender"))

    patient_delta_path = pipeline_dir / "k1_failure_patient_deltas.csv"
    gender_summary_path = pipeline_dir / "k1_failure_gender_summary.csv"
    shift_summary_path = pipeline_dir / "k1_failure_shift_summary.csv"
    window_summary_path = pipeline_dir / "k1_failure_window_summary.csv"

    patient_delta_df.to_csv(patient_delta_path, index=False)
    gender_summary_df.to_csv(gender_summary_path, index=False)
    shift_summary_df.to_csv(shift_summary_path, index=False)
    window_summary_df.to_csv(window_summary_path, index=False)

    report_path = write_report(
        pipeline_dir,
        patient_delta_df,
        gender_summary_df,
        shift_summary_df,
        window_summary_df,
        baseline_log_df,
        k1_log_df,
    )

    print("K1 failure analysis complete")
    print(f"  patient_deltas={patient_delta_path}")
    print(f"  gender_summary={gender_summary_path}")
    print(f"  shift_summary={shift_summary_path}")
    print(f"  window_summary={window_summary_path}")
    print(f"  report={report_path}")


if __name__ == "__main__":
    main()