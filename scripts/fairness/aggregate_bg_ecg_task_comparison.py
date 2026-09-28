#!/usr/bin/env python3
"""Build a compact BG, AAMI-5 ECG, and binary-ectopy comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

METHODS = ["Teacher", "Student", "Baseline KD", "T1", "O2", "T1+O2"]
ECG_NAMES = {
    "teacher": "Teacher",
    "student": "Student",
    "KD": "Baseline KD",
    "T1": "T1",
    "O2": "O2",
    "T1+O2": "T1+O2",
}
BG_MULTI_NAMES = {
    "Baseline KD (no fairness)": "Baseline KD",
    "Distilled from Fair Teacher (T1)": "T1",
    "Distilled from Fair Teacher + O2 Calibration Head": "T1+O2",
}
BG_SINGLE_NAMES = {
    "Teacher baseline (BERT)": "Teacher",
    "Student baseline (no KD)": "Student",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bg-root", type=Path, required=True)
    parser.add_argument("--aami5-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bg_rows(root: Path) -> list[dict[str, Any]]:
    output = []
    single_path = root / "fairness_comparison_results.csv"
    if single_path.exists():
        for row in read_csv(single_path):
            method = BG_SINGLE_NAMES.get(row["label"])
            if method:
                output.append(
                    {
                        "task": "BG forecasting",
                        "evidence": "single seed",
                        "n_seeds": 1,
                        "method": method,
                        "utility_metric": "RMSE",
                        "utility_mean": float(row["rmse"]),
                        "utility_std": "",
                        "fairness_metric": "hypoglycemia EO",
                        "fairness_mean": float(row["eo_gap_raw"]),
                        "fairness_std": "",
                        "fairness_delta_vs_kd": "",
                        "wins_vs_kd": "",
                    }
                )
    multi = read_csv(root / "multiseed_robustness_results.csv")
    baseline = next(float(row["eo_raw_mean"]) for row in multi if row["method"] == "Baseline KD (no fairness)")
    baseline_per_seed = next(
        [float(value) for value in row["eo_raw_per_seed"].split(";")]
        for row in multi
        if row["method"] == "Baseline KD (no fairness)"
    )
    for row in multi:
        method = BG_MULTI_NAMES[row["method"]]
        values = [float(value) for value in row["eo_raw_per_seed"].split(";")]
        fairness = float(row["eo_raw_mean"])
        output.append(
            {
                "task": "BG forecasting",
                "evidence": "5 seeds",
                "n_seeds": 5,
                "method": method,
                "utility_metric": "RMSE",
                "utility_mean": float(row["rmse_mean"]),
                "utility_std": float(row["rmse_std"]),
                "fairness_metric": "hypoglycemia EO",
                "fairness_mean": fairness,
                "fairness_std": float(row["eo_raw_std"]),
                "fairness_delta_vs_kd": fairness - baseline,
                "wins_vs_kd": sum(value < reference for value, reference in zip(values, baseline_per_seed)),
            }
        )
    return output


def ecg_rows(root: Path, task: str, fairness_metric: str, legacy_aami5: bool) -> list[dict[str, Any]]:
    rows = read_csv(root / "multiseed_aggregate_summary.csv")
    output = []
    normalized = {ECG_NAMES[row["variant"]]: row for row in rows}
    baseline = normalized["Baseline KD"]
    baseline_eo = float(baseline["eo_NV_mean"] if legacy_aami5 else baseline["selected_eo_mean"])

    per_seed_path = root / "multiseed_per_seed_summary.csv"
    per_seed = read_csv(per_seed_path)
    baseline_by_seed = {
        int(row["seed"]): float(row["eo_NV"] if legacy_aami5 else row["selected_eo"])
        for row in per_seed
        if ECG_NAMES[row["variant"]] == "Baseline KD"
        and (row["eo_NV"] if legacy_aami5 else row["selected_eo"]) not in {"", "None"}
    }
    for method in METHODS:
        row = normalized[method]
        eo_field = "eo_NV" if legacy_aami5 else "selected_eo"
        fairness = float(row[f"{eo_field}_mean"])
        variant_values = [
            item for item in per_seed if ECG_NAMES[item["variant"]] == method
        ]
        wins = sum(
            float(item[eo_field]) < baseline_by_seed[int(item["seed"])]
            for item in variant_values
            if item[eo_field] not in {"", "None"}
            and int(item["seed"]) in baseline_by_seed
        )
        output.append(
            {
                "task": task,
                "evidence": f"{int(row['n_seeds']) if 'n_seeds' in row else 5} seed"
                f"{'s' if (int(row['n_seeds']) if 'n_seeds' in row else 5) != 1 else ''}",
                "n_seeds": int(row["n_seeds"]) if "n_seeds" in row else 5,
                "method": method,
                "utility_metric": "macro-F1",
                "utility_mean": float(row["macro_f1_mean"]),
                "utility_std": float(row["macro_f1_std"]),
                "fairness_metric": fairness_metric,
                "fairness_mean": fairness,
                "fairness_std": float(row[f"{eo_field}_std"]),
                "fairness_delta_vs_kd": fairness - baseline_eo,
                "wins_vs_kd": wins,
            }
        )
    return output


def value(mean: Any, std: Any) -> str:
    if mean == "" or mean is None:
        return "NA"
    if std == "" or std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def build_report(rows: list[dict[str, Any]]) -> str:
    by_task_method = {(row["task"], row["method"]): row for row in rows}
    tasks = ["BG forecasting", "ECG AAMI-5", "ECG binary ectopy"]
    lines = [
        "# BG, AAMI-5 ECG, and Binary-Ectopy Comparison",
        "",
        "Lower fairness values are better. Utility and fairness definitions differ across tasks and must only be compared within a task.",
        "All ECG rows are locked-test results. Negative fairness deltas and fairness wins are paired comparisons against Baseline KD within the same task.",
        "",
        "| Task | Evidence | Method | Utility | Fairness | Fairness delta vs KD | Fairness wins vs KD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for method in METHODS:
            row = by_task_method.get((task, method))
            if row is None:
                lines.append(f"| {task} | not run | {method} | NA | NA | NA | NA |")
                continue
            delta = row["fairness_delta_vs_kd"]
            delta_text = "NA" if delta == "" else f"{float(delta):+.4f}"
            wins = row["wins_vs_kd"]
            wins_text = "NA" if wins == "" else f"{wins}/{row['n_seeds']}"
            lines.append(
                f"| {task} | {row['evidence']} | {method} | "
                f"{value(row['utility_mean'], row['utility_std'])} {row['utility_metric']} | "
                f"{value(row['fairness_mean'], row['fairness_std'])} {row['fairness_metric']} | "
                f"{delta_text} | {wins_text} |"
            )
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- BG utility is RMSE; lower is better. BG fairness is the sex TPR gap for hypoglycemia.",
            "- AAMI-5 utility is macro-F1; higher is better. Its fairness endpoint is mean sex recall gap over support/performance-qualified N and V.",
            "- Binary utility is macro-F1; higher is better. Its fairness endpoint is the female/male ectopy-recall gap for S/V/F versus N, with Q excluded.",
            "- BG Teacher and Student are single-seed references because the canonical BG five-seed artifact contains KD, T1, and T1+O2 only.",
            "- Negative fairness deltas indicate improvement over Baseline KD.",
            "- The BG O2 head calibrates continuous BG outputs, whereas ECG O2 is a jointly trained group-conditional affine head on class logits. They instantiate the same output-calibration idea but are not identical estimators.",
            "",
            "## Cross-Domain Interpretation",
            "",
            "- BG provides the strongest result: T1+O2 lowers mean hypoglycemia EO from 0.2174 to 0.1179 (45.8%) and improves in 5/5 seeds while also lowering RMSE.",
            "- AAMI-5 is mixed: T1 has the best directional fairness result (3/5 wins), while O2 and T1+O2 do not show robust fairness transfer and S-class recognition remains poor.",
            "- Binary ectopy is supportive but not decisive: on the locked test split, O2 and T1+O2 improve both mean macro-F1 and mean ectopy EO versus Baseline KD; T1+O2 improves utility in 5/5 seeds and fairness in 4/5.",
            "- Binary validation does not show the same ranking, and its mean EO reductions are influenced by one high-gap Baseline KD seed. Therefore these results support cross-task portability of the intervention family, not universal superiority or independent replication of the BG effect size.",
            "- MIT-BIH is a different dataset and classification task, but it is not an external BG forecasting cohort. It strengthens the broader claim that fairness-aware KD must be audited per task; it does not enlarge the OhioT1DM patient sample.",
            "",
            "## Reporting Guardrail",
            "",
            "Do not compare absolute RMSE, macro-F1, or EO magnitudes across tasks. Do not select an ECG method from these locked-test rows. Report the BG result as the primary finding and the ECG experiments as a cross-domain stress test with mixed AAMI-5 and supportive binary-ectopy evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.binary_root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = bg_rows(args.bg_root)
    rows.extend(ecg_rows(args.aami5_root, "ECG AAMI-5", "N/V sex EO", True))
    rows.extend(ecg_rows(args.binary_root, "ECG binary ectopy", "ectopy sex EO", False))
    write_csv(output_dir / "bg_aami5_binary_comparison.csv", rows)
    (output_dir / "BG_AAMI5_BINARY_COMPARISON.md").write_text(
        build_report(rows),
        encoding="utf-8",
    )
    print(output_dir / "BG_AAMI5_BINARY_COMPARISON.md")


if __name__ == "__main__":
    main()
