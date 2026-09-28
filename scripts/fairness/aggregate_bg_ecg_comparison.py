#!/usr/bin/env python3
"""Aggregate BG/ECG fairness results and alternate-group ECG audits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
ECG_VARIANTS = [
    "teacher",
    "student_baseline",
    "distilled_baseline",
    "distilled_t1",
    "distilled_o2",
    "distilled_t1_o2",
]
DISPLAY_NAMES = {
    "teacher": "Teacher",
    "student_baseline": "Student",
    "distilled_baseline": "Baseline KD",
    "distilled_t1": "T1",
    "distilled_o2": "O2",
    "distilled_t1_o2": "T1+O2",
}
ECG_AGGREGATE_NAMES = {
    "teacher": "teacher",
    "student_baseline": "student",
    "distilled_baseline": "KD",
    "distilled_t1": "T1",
    "distilled_o2": "O2",
    "distilled_t1_o2": "T1+O2",
}
GROUP_ATTRIBUTES = ["sex", "age_group", "paced_group", "difficulty_group"]
BG_ALTERNATE_GROUP_DIAGNOSTICS = [
    ("Age", 0.163, 0.207),
    ("Gender", 0.195, 0.217),
    ("Pump type", 0.082, 0.106),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-root", type=Path, required=True)
    parser.add_argument("--bg-root", type=Path, required=True)
    parser.add_argument(
        "--binary-root",
        type=Path,
        default=Path("experiments/mitbih_binary_ectopy_five_seed"),
        help="Locked binary-ectopy five-seed directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_ecg_alternate_group_results(
    ecg_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    reports: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)

    for seed_dir in sorted(ecg_root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for attribute in GROUP_ATTRIBUTES:
            report_path = seed_dir / f"fairness_comparison_{attribute}.json"
            data = json.loads(report_path.read_text(encoding="utf-8"))
            for variant in ECG_VARIANTS:
                reports[attribute].append((seed, variant, data[variant]))

    per_seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    for attribute in GROUP_ATTRIBUTES:
        attribute_reports = reports[attribute]
        common_classes: set[int] | None = None
        groups: set[str] = set()
        for _, _, report in attribute_reports:
            reportable = {int(value) for value in report["summary"]["reportable_eo_classes"]}
            common_classes = reportable if common_classes is None else common_classes & reportable
            groups.update(report["overall"])

        qualified_classes = sorted(common_classes or [])
        if not qualified_classes:
            raise ValueError(f"No common performance-qualified classes for {attribute}")

        metadata[attribute] = {
            "groups": sorted(groups),
            "qualified_classes": [CLASS_NAMES[class_id] for class_id in qualified_classes],
        }

        base_by_seed: dict[int, float] = {}
        pending_rows: list[dict[str, Any]] = []
        for seed, variant, report in attribute_reports:
            classwise = report["classwise_one_vs_rest"]
            eo_gap = statistics.mean(
                float(classwise[str(class_id)]["raw_eo_gap"])
                for class_id in qualified_classes
            )
            if variant == "distilled_baseline":
                base_by_seed[seed] = eo_gap
            pending_rows.append(
                {
                    "group_attribute": attribute,
                    "seed": seed,
                    "variant": DISPLAY_NAMES[variant],
                    "groups": ";".join(sorted(groups)),
                    "qualified_classes": ";".join(
                        CLASS_NAMES[class_id] for class_id in qualified_classes
                    ),
                    "eo_gap": eo_gap,
                }
            )

        for row in pending_rows:
            baseline = base_by_seed[int(row["seed"])]
            delta = float(row["eo_gap"]) - baseline
            row["eo_delta_vs_baseline_kd"] = delta
            row["better_than_baseline_kd"] = delta < 0
            per_seed_rows.append(row)

        for variant in ECG_VARIANTS:
            display_name = DISPLAY_NAMES[variant]
            variant_rows = [row for row in pending_rows if row["variant"] == display_name]
            values = [float(row["eo_gap"]) for row in variant_rows]
            deltas = [
                value - base_by_seed[int(row["seed"])]
                for value, row in zip(values, variant_rows)
            ]
            aggregate_rows.append(
                {
                    "group_attribute": attribute,
                    "variant": display_name,
                    "groups": ";".join(sorted(groups)),
                    "qualified_classes": ";".join(
                        CLASS_NAMES[class_id] for class_id in qualified_classes
                    ),
                    "n_seeds": len(values),
                    "eo_gap_mean": statistics.mean(values),
                    "eo_gap_std": sample_std(values),
                    "eo_delta_vs_baseline_kd_mean": statistics.mean(deltas),
                    "eo_change_vs_baseline_kd_pct": (
                        statistics.mean(deltas)
                        / statistics.mean(base_by_seed.values())
                        * 100.0
                    ),
                    "wins_vs_baseline_kd": sum(delta < 0 for delta in deltas),
                }
            )

    return per_seed_rows, aggregate_rows, metadata


def build_primary_comparison(
    ecg_root: Path,
    bg_root: Path,
    alternate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bg_single = read_csv(bg_root / "fairness_comparison_results.csv")
    for row in bg_single:
        output.append(
            {
                "domain": "BG",
                "evidence_scope": "single_seed",
                "method": row["label"],
                "utility_metric": "RMSE",
                "utility_mean": float(row["rmse"]),
                "utility_std": "",
                "fairness_feature": "gender",
                "fairness_metric": "hypoglycemia EO raw",
                "fairness_mean": float(row["eo_gap_raw"]),
                "fairness_std": "",
                "fairness_delta_vs_baseline_kd": "",
                "fairness_change_vs_baseline_kd_pct": "",
                "fairness_wins_vs_baseline_kd": "",
            }
        )

    bg_multiseed = read_csv(bg_root / "multiseed_robustness_results.csv")
    bg_baseline = next(
        float(row["eo_raw_mean"])
        for row in bg_multiseed
        if row["method"] == "Baseline KD (no fairness)"
    )
    bg_baseline_per_seed = next(
        [float(value) for value in row["eo_raw_per_seed"].split(";")]
        for row in bg_multiseed
        if row["method"] == "Baseline KD (no fairness)"
    )
    for row in bg_multiseed:
        fairness = float(row["eo_raw_mean"])
        per_seed = [float(value) for value in row["eo_raw_per_seed"].split(";")]
        output.append(
            {
                "domain": "BG",
                "evidence_scope": "five_seed",
                "method": row["method"],
                "utility_metric": "RMSE",
                "utility_mean": float(row["rmse_mean"]),
                "utility_std": float(row["rmse_std"]),
                "fairness_feature": "gender",
                "fairness_metric": "hypoglycemia EO raw",
                "fairness_mean": fairness,
                "fairness_std": float(row["eo_raw_std"]),
                "fairness_delta_vs_baseline_kd": fairness - bg_baseline,
                "fairness_change_vs_baseline_kd_pct": (
                    (fairness - bg_baseline) / bg_baseline * 100.0
                ),
                "fairness_wins_vs_baseline_kd": sum(
                    value < baseline
                    for value, baseline in zip(per_seed, bg_baseline_per_seed)
                ),
            }
        )

    ecg_aggregate = {
        row["variant"]: row
        for row in read_csv(ecg_root / "multiseed_aggregate_summary.csv")
    }
    sex_rows = {
        row["variant"]: row
        for row in alternate_rows
        if row["group_attribute"] == "sex"
    }
    for variant in ECG_VARIANTS:
        name = DISPLAY_NAMES[variant]
        aggregate = ecg_aggregate[ECG_AGGREGATE_NAMES[variant]]
        fairness = sex_rows[name]
        output.append(
            {
                "domain": "ECG",
                "evidence_scope": "five_seed",
                "method": name,
                "utility_metric": "accuracy",
                "utility_mean": float(aggregate["accuracy_mean"]),
                "utility_std": float(aggregate["accuracy_std"]),
                "fairness_feature": "sex",
                "fairness_metric": "N/V EO raw",
                "fairness_mean": float(fairness["eo_gap_mean"]),
                "fairness_std": float(fairness["eo_gap_std"]),
                "fairness_delta_vs_baseline_kd": float(
                    fairness["eo_delta_vs_baseline_kd_mean"]
                ),
                "fairness_change_vs_baseline_kd_pct": float(
                    fairness["eo_change_vs_baseline_kd_pct"]
                ),
                "fairness_wins_vs_baseline_kd": int(fairness["wins_vs_baseline_kd"]),
            }
        )

    return output


def fmt(value: float) -> str:
    return f"{value:.4f}"


def mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"

def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def fairness_delta(value: float, reference: float) -> str:
    delta = value - reference
    return f"{delta:+.4f} ({delta / reference * 100.0:+.1f}%)"


def markdown_table_row(cells: list[str], bold: bool = False) -> str:
    values = [f"**{cell}**" for cell in cells] if bold else cells
    return "| " + " | ".join(values) + " |"


def generate_report(
    ecg_root: Path,
    bg_root: Path,
    binary_root: Path,
    primary_rows: list[dict[str, Any]],
    alternate_rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> str:
    bg_five = {
        row["method"]: row
        for row in primary_rows
        if row["domain"] == "BG" and row["evidence_scope"] == "five_seed"
    }
    ecg_five = {
        row["method"]: row
        for row in primary_rows
        if row["domain"] == "ECG" and row["evidence_scope"] == "five_seed"
    }
    bg_single = [
        row
        for row in primary_rows
        if row["domain"] == "BG" and row["evidence_scope"] == "single_seed"
    ]
    alternate = {
        (row["group_attribute"], row["variant"]): row for row in alternate_rows
    }
    binary_aggregate = {
        row["variant"]: row
        for row in read_csv(binary_root / "multiseed_aggregate_summary.csv")
    }
    binary_per_seed = read_csv(binary_root / "multiseed_per_seed_summary.csv")
    binary_kd_by_seed = {
        int(row["seed"]): float(row["selected_eo"])
        for row in binary_per_seed
        if row["variant"] == "KD"
    }

    bg_single_by_name = {row["method"]: row for row in bg_single}
    bg_single_baseline = bg_single_by_name["Distilled — no fairness"]
    overview_methods = [
        ("Teacher", "Teacher baseline (BERT)", "single_seed"),
        ("Student", "Student baseline (no KD)", "single_seed"),
        ("Baseline KD", "Baseline KD (no fairness)", "five_seed"),
        ("T1", "Distilled from Fair Teacher (T1)", "five_seed"),
        ("O2", None, "not_run"),
        ("T1+O2", "Distilled from Fair Teacher + O2 Calibration Head", "five_seed"),
    ]
    lines = [
        "# Detailed BG, AAMI-5 ECG, and Binary-Ectopy Fairness Distillation Comparison",
        "",
        "## Reading the comparison",
        "",
        "- Lower fairness gaps are better in both domains.",
        "- BG utility is RMSE (lower is better); ECG utility is accuracy or macro-F1 (higher is better). These task metrics are not directly comparable.",
        "- BG EO is the male/female true-positive-rate gap for hypoglycemia detection. AAMI-5 EO is averaged over performance-qualified N/V recall gaps; binary EO is the S/V/F ectopy-recall sex gap with Q excluded.",
        "- BG five-seed evidence covers baseline KD, T1, and T1+O2. Both ECG endpoints additionally have standalone Teacher, Student, and pure O2 five-seed results.",
        "- All binary rows are from the locked test split. They must not be used to select or retune an ECG method.",
        "- Existing AAMI-5 alternate-group results are audits of models trained with sex as the fairness feature; they are not targeted age/pacing/difficulty mitigations.",
        "",
        "## Cross-domain overview",
        "",
        "BG values marked **single seed** are included for completeness but do not have the same evidence strength as five-seed mean ± standard deviation results. Pure O2 was not run for BG; the BG O2 experiment always used T1+O2.",
        "",
        "| Method | BG RMSE | BG gender EO | BG EO change vs KD | AAMI-5 accuracy | AAMI-5 sex N/V EO | AAMI-5 EO change vs KD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ecg_name, bg_name, bg_scope in overview_methods:
        ecg = ecg_five[ecg_name]
        if bg_scope == "five_seed":
            bg = bg_five[str(bg_name)]
            bg_rmse = mean_std(float(bg["utility_mean"]), float(bg["utility_std"]))
            bg_eo = mean_std(float(bg["fairness_mean"]), float(bg["fairness_std"]))
            bg_change = (
                f"{float(bg['fairness_change_vs_baseline_kd_pct']):+.1f}% "
                f"({bg['fairness_wins_vs_baseline_kd']}/5)"
            )
        elif bg_scope == "single_seed":
            bg = bg_single_by_name[str(bg_name)]
            bg_rmse = f"{float(bg['utility_mean']):.3f} **(single seed)**"
            bg_eo = f"{float(bg['fairness_mean']):.4f} **(single seed)**"
            bg_change_pct = (
                (float(bg["fairness_mean"]) - float(bg_single_baseline["fairness_mean"]))
                / float(bg_single_baseline["fairness_mean"])
                * 100.0
            )
            bg_change = f"{bg_change_pct:+.1f}% **(single seed)**"
        else:
            bg_rmse = "Not run alone"
            bg_eo = "Not run alone"
            bg_change = "Not run alone"
        lines.append(
            f"| {ecg_name} | "
            f"{bg_rmse} | {bg_eo} | {bg_change} | "
            f"{mean_std(float(ecg['utility_mean']), float(ecg['utility_std']))} | "
            f"{mean_std(float(ecg['fairness_mean']), float(ecg['fairness_std']))} | "
            f"{float(ecg['fairness_change_vs_baseline_kd_pct']):+.1f}% "
            f"({ecg['fairness_wins_vs_baseline_kd']}/5) |"
        )

    lines.extend(
        [
            "",
            "For the AAMI-5 endpoint, BG T1+O2 reduces mean EO by 45.8% and wins all five seeds, whereas AAMI-5 T1+O2 reduces mean EO by 7.0% and wins two of five seeds. T1 has the best directional AAMI-5 fairness result, but its effect is not statistically significant at five seeds.",
            "",
            "## Locked Binary-Ectopy Five-Seed Result",
            "",
            "Binary ectopy is N versus S/V/F, with Q excluded. Utility is macro-F1 and fairness is the female/male ectopy-recall gap. Values are locked-test mean ± sample standard deviation across the same five training seeds.",
            "",
            "| Method | Macro-F1 | Ectopy sex EO | Macro-F1 delta vs KD | EO delta vs KD | EO wins vs KD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    binary_kd = binary_aggregate["KD"]
    for variant in ("teacher", "student", "KD", "T1", "O2", "T1+O2"):
        row = binary_aggregate[variant]
        variant_rows = [item for item in binary_per_seed if item["variant"] == variant]
        wins = sum(
            float(item["selected_eo"]) < binary_kd_by_seed[int(item["seed"])]
            for item in variant_rows
        )
        display = {"teacher": "Teacher", "student": "Student", "KD": "Baseline KD"}.get(
            variant, variant
        )
        lines.append(
            f"| {display} | {mean_std(float(row['macro_f1_mean']), float(row['macro_f1_std']))} | "
            f"{mean_std(float(row['selected_eo_mean']), float(row['selected_eo_std']))} | "
            f"{float(row['macro_f1_mean']) - float(binary_kd['macro_f1_mean']):+.4f} | "
            f"{float(row['selected_eo_mean']) - float(binary_kd['selected_eo_mean']):+.4f} | "
            f"{wins}/5 |"
        )
    lines.extend(
        [
            "",
            "Binary ectopy is supportive but not decisive. O2 improves mean macro-F1 by 0.0127 and lowers EO by 0.0357; T1+O2 improves macro-F1 by 0.0264 in 5/5 seeds and lowers EO by 0.0248 in 4/5 seeds. However, Baseline KD is strongest on the five-seed validation averages, and seed 427368 strongly influences the mean test EO reductions. The binary results therefore support task-specific portability of the intervention family, not universal superiority or independent replication of the BG effect size.",
            "",
            "## Fairness change against each baseline",
            "",
            "Negative deltas mean a lower EO gap and therefore better fairness than the reference. Positive deltas mean worse fairness. Comparisons are valid within each domain only because BG and ECG use different prediction tasks and EO definitions.",
            "",
            "BG Teacher and Student are single-run references; BG Baseline KD, T1, and T1+O2 use five-seed means ± standard deviation. All AAMI-5 rows use five-seed means ± standard deviation. Pure O2 was not evaluated separately from T1+O2 for BG. Deltas across different evidence scopes are descriptive rather than paired statistical comparisons.",
            "",
            "Bold rows mark the baseline KD reference and the strongest distilled fairness result for BG and AAMI-5 in this subsection.",
            "",
            "| Domain | Method | EO gap | Δ vs teacher | Δ vs student | Δ vs baseline KD | Fairness conclusion |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    bg_delta_values = {
        "Teacher": float(bg_single_by_name["Teacher baseline (BERT)"]["fairness_mean"]),
        "Student": float(bg_single_by_name["Student baseline (no KD)"]["fairness_mean"]),
        "Baseline KD": float(bg_five["Baseline KD (no fairness)"]["fairness_mean"]),
        "T1": float(bg_five["Distilled from Fair Teacher (T1)"]["fairness_mean"]),
        "T1+O2": float(
            bg_five["Distilled from Fair Teacher + O2 Calibration Head"]["fairness_mean"]
        ),
    }
    bg_delta_stds = {
        "Baseline KD": float(bg_five["Baseline KD (no fairness)"]["fairness_std"]),
        "T1": float(bg_five["Distilled from Fair Teacher (T1)"]["fairness_std"]),
        "T1+O2": float(
            bg_five["Distilled from Fair Teacher + O2 Calibration Head"]["fairness_std"]
        ),
    }
    bg_teacher = bg_delta_values["Teacher"]
    bg_student = bg_delta_values["Student"]
    bg_kd = bg_delta_values["Baseline KD"]
    bg_conclusions = {
        "Teacher": "Reference teacher; worse than student and baseline KD",
        "Student": "Better than teacher and baseline KD",
        "Baseline KD": "Better than teacher; worse than student",
        "T1": "Better than teacher and baseline KD; worse than student",
        "T1+O2": "Better than teacher, student, and baseline KD",
    }
    for method in ("Teacher", "Student", "Baseline KD", "T1"):
        value = bg_delta_values[method]
        domain = "BG (1 seed)" if method in {"Teacher", "Student"} else "BG (5 seeds)"
        value_text = (
            f"{value:.4f}"
            if method in {"Teacher", "Student"}
            else mean_std(value, bg_delta_stds[method])
        )
        lines.append(
            markdown_table_row(
                [
                    domain,
                    method,
                    value_text,
                    fairness_delta(value, bg_teacher),
                    fairness_delta(value, bg_student),
                    fairness_delta(value, bg_kd),
                    bg_conclusions[method],
                ],
                bold=method == "Baseline KD",
            )
        )
    lines.append(
        "| BG (not run alone) | O2-only | Not run alone | Not run alone | Not run alone | Not run alone | O2 exists only in the T1+O2 experiment |"
    )
    value = bg_delta_values["T1+O2"]
    lines.append(
        markdown_table_row(
            [
                "BG (5 seeds)",
                "T1+O2",
                mean_std(value, bg_delta_stds["T1+O2"]),
                fairness_delta(value, bg_teacher),
                fairness_delta(value, bg_student),
                fairness_delta(value, bg_kd),
                bg_conclusions["T1+O2"],
            ],
            bold=True,
        )
    )
    ecg_delta_values = {
        method: float(ecg_five[method]["fairness_mean"])
        for method in ("Teacher", "Student", "Baseline KD", "T1", "O2", "T1+O2")
    }
    ecg_teacher = ecg_delta_values["Teacher"]
    ecg_student = ecg_delta_values["Student"]
    ecg_kd = ecg_delta_values["Baseline KD"]
    ecg_conclusions = {
        "Teacher": "Fairest mean result",
        "Student": "Worse than teacher and baseline KD",
        "Baseline KD": "Worse than teacher; better than student",
        "T1": "Worse than teacher; better than student and baseline KD",
        "O2": "Worse than teacher and baseline KD; better than student",
        "T1+O2": "Worse than teacher; better than student and baseline KD",
    }
    for method in ("Teacher", "Student", "Baseline KD", "T1", "O2", "T1+O2"):
        value = ecg_delta_values[method]
        lines.append(
            markdown_table_row(
                [
                    "AAMI-5 ECG (5 seeds)",
                    method,
                    mean_std(value, float(ecg_five[method]["fairness_std"])),
                    fairness_delta(value, ecg_teacher),
                    fairness_delta(value, ecg_student),
                    fairness_delta(value, ecg_kd),
                    ecg_conclusions[method],
                ],
                bold=method in {"Baseline KD", "T1"},
            )
        )

    lines.extend(
        [
            "",
            "### Focused intervention comparison",
            "",
            "This duplicate view removes the standalone Student row and all student-relative deltas. It retains only Teacher, Baseline KD, T1, O2, and T1+O2.",
            "",
            "| Domain | Method | EO gap | Δ vs teacher | Δ vs baseline KD | Fairness conclusion |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    bg_focused_conclusions = {
        "Teacher": "Reference teacher",
        "Baseline KD": "Better than teacher",
        "T1": "Better than teacher and baseline KD",
        "T1+O2": "Better than teacher and baseline KD",
    }
    for method in ("Teacher", "Baseline KD", "T1"):
        value = bg_delta_values[method]
        domain = "BG (1 seed)" if method == "Teacher" else "BG (5 seeds)"
        value_text = (
            f"{value:.4f}"
            if method == "Teacher"
            else mean_std(value, bg_delta_stds[method])
        )
        lines.append(
            markdown_table_row(
                [
                    domain,
                    method,
                    value_text,
                    fairness_delta(value, bg_teacher),
                    fairness_delta(value, bg_kd),
                    bg_focused_conclusions[method],
                ],
                bold=method == "Baseline KD",
            )
        )
    lines.append(
        "| BG (not run alone) | O2-only | Not run alone | Not run alone | Not run alone | O2 exists only in the T1+O2 experiment |"
    )
    value = bg_delta_values["T1+O2"]
    lines.append(
        markdown_table_row(
            [
                "BG (5 seeds)",
                "T1+O2",
                mean_std(value, bg_delta_stds["T1+O2"]),
                fairness_delta(value, bg_teacher),
                fairness_delta(value, bg_kd),
                bg_focused_conclusions["T1+O2"],
            ],
            bold=True,
        )
    )

    ecg_focused_conclusions = {
        "Teacher": "Fairest mean result",
        "Baseline KD": "Worse than teacher",
        "T1": "Worse than teacher; better than baseline KD",
        "O2": "Worse than teacher and baseline KD",
        "T1+O2": "Worse than teacher; better than baseline KD",
    }
    for method in ("Teacher", "Baseline KD", "T1", "O2", "T1+O2"):
        value = ecg_delta_values[method]
        lines.append(
            markdown_table_row(
                [
                    "AAMI-5 ECG (5 seeds)",
                    method,
                    mean_std(value, float(ecg_five[method]["fairness_std"])),
                    fairness_delta(value, ecg_teacher),
                    fairness_delta(value, ecg_kd),
                    ecg_focused_conclusions[method],
                ],
                bold=method in {"Baseline KD", "T1"},
            )
        )

    lines.extend(
        [
            "",
            "BG fairness path: the single-run standalone student improves EO over the single-run teacher by 0.0354. The five-seed baseline KD mean is 0.0226 worse than that student reference; five-seed T1 partially repairs baseline KD, while five-seed T1+O2 clearly beats the teacher, student, and baseline KD references.",
            "",
            "AAMI-5 fairness path: compression without KD worsens EO by 0.1063 relative to the teacher. Baseline KD recovers 0.0737 of that loss, and T1 recovers a further 0.0191 versus baseline KD. However, no distilled AAMI-5 method reaches the teacher's mean EO; T1 is the closest.",
            "",
            "## AAMI-5 ECG Five-Seed Sections",
            "",
            "| Section | Accuracy | Macro-F1 | Weighted-F1 | Sex N/V EO | EO change vs KD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    ecg_aggregate = {
        row["variant"]: row
        for row in read_csv(ecg_root / "multiseed_aggregate_summary.csv")
    }
    for variant in ECG_VARIANTS:
        name = DISPLAY_NAMES[variant]
        utility = ecg_aggregate[ECG_AGGREGATE_NAMES[variant]]
        fairness = ecg_five[name]
        lines.append(
            f"| {name} | "
            f"{mean_std(float(utility['accuracy_mean']), float(utility['accuracy_std']))} | "
            f"{mean_std(float(utility['macro_f1_mean']), float(utility['macro_f1_std']))} | "
            f"{mean_std(float(utility['weighted_f1_mean']), float(utility['weighted_f1_std']))} | "
            f"{mean_std(float(fairness['fairness_mean']), float(fairness['fairness_std']))} | "
            f"{float(fairness['fairness_change_vs_baseline_kd_pct']):+.1f}% |"
        )

    student = ecg_five["Student"]
    kd = ecg_five["Baseline KD"]
    student_to_kd = (
        (float(kd["fairness_mean"]) - float(student["fairness_mean"]))
        / float(student["fairness_mean"])
        * 100.0
    )
    lines.extend(
        [
            "",
            f"Unlike BG, ECG baseline KD improves fairness relative to the standalone student: sex N/V EO falls from {fmt(float(student['fairness_mean']))} to {fmt(float(kd['fairness_mean']))} ({student_to_kd:+.1f}%), while accuracy rises from {fmt(float(student['utility_mean']))} to {fmt(float(kd['utility_mean']))}.",
            "",
            "## BG single-run method grid",
            "",
            "The full BG intervention grid is single-run evidence; only the three methods in the direct comparison were repeated over five seeds.",
            "",
            "| Section | RMSE | Gender EO raw | Gender EO calibrated |",
            "|---|---:|---:|---:|",
        ]
    )
    bg_source = {
        row["label"]: row for row in read_csv(bg_root / "fairness_comparison_results.csv")
    }
    for row in bg_single:
        source = bg_source[str(row["method"])]
        lines.append(
            f"| {row['method']} | {float(row['utility_mean']):.3f} | "
            f"{float(row['fairness_mean']):.4f} | {float(source['eo_gap_calibrated']):.4f} |"
        )

    lines.extend(
        [
            "",
            "In the matched single run, BG baseline KD is worse than the standalone student on both RMSE (23.361 vs 21.989) and raw EO (0.2171 vs 0.1948). This is the opposite of ECG, where baseline KD improves both mean utility and sex EO over the standalone student.",
            "",
            "## Other grouping attributes",
            "",
            "### BG documented diagnostic",
            "",
            "This diagnostic compares the standalone student with baseline KD only; it does not establish T1/O2 performance for age or pump type.",
            "",
            "| Grouping | Student EO | Baseline KD EO | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for grouping, student_eo, kd_eo in BG_ALTERNATE_GROUP_DIAGNOSTICS:
        lines.append(
            f"| {grouping} | {student_eo:.3f} | {kd_eo:.3f} | "
            f"{(kd_eo - student_eo) / student_eo * 100:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "### ECG audit of existing sex-trained models",
            "",
            "Each row uses the same class set across all six variants and all five seeds. The parenthesized value is the number of seeds with lower EO than baseline KD.",
            "",
            "| Grouping | Qualified classes | Teacher | Student | Baseline KD | T1 | O2 | T1+O2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for attribute in GROUP_ATTRIBUTES:
        model_cells = []
        for variant in ECG_VARIANTS:
            name = DISPLAY_NAMES[variant]
            row = alternate[(attribute, name)]
            cell = mean_std(float(row["eo_gap_mean"]), float(row["eo_gap_std"]))
            if variant.startswith("distilled_") and variant != "distilled_baseline":
                cell += f" ({row['wins_vs_baseline_kd']}/5)"
            model_cells.append(cell)
        qualified = ",".join(metadata[attribute]["qualified_classes"])
        groups = ", ".join(metadata[attribute]["groups"])
        label = f"{attribute} ({groups})"
        lines.append(f"| {label} | {qualified} | " + " | ".join(model_cells) + " |")

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "1. AAMI-5 sex: T1 has the best mean EO and improves three of five seeds; pure O2 is slightly worse than baseline KD on average. This statement does not apply to the binary endpoint.",
            "2. Age: baseline KD improves strongly over the standalone student, but all sex-trained mitigation variants have worse mean age EO than baseline KD.",
            "3. Pacing: only class N is support-qualified across every model and seed. V has only two paced test beats, so this is a narrow audit rather than a general pacing-fairness claim.",
            "4. Difficulty: O2 has the best mean mitigation result versus baseline KD, but wins only three of five seeds and the remaining EO gap is large.",
            "5. Age is a demographic fairness attribute; pacing status is a clinical/device subgroup; difficulty is a data-quality robustness subgroup. They should be described separately in a paper.",
            "",
            "## Targeted retraining",
            "",
            "The ECG pipeline can target `age_group`, `paced_group`, or `difficulty_group` by changing `FAIR_FEATURE`. That retrains T1 sampling and O2 calibration for the chosen attribute and is the valid test of attribute-specific mitigation. Auditing sex-trained models on another column only measures spillover/generalization.",
            "",
            "Recommended next experiments:",
            "",
            "1. Retrain five seeds with age as the first alternate target because all three age groups have N/V support and age is the closest analogue to a protected demographic attribute.",
            "2. Treat pacing as a limited clinical subgroup analysis unless the split/data design can provide sufficient paced V/S/F support.",
            "3. Treat difficulty as robustness under data quality shift, not demographic fairness.",
            "4. Compare targeted-versus-spillover matrices: train on sex and age separately, then audit both models on sex, age, pacing, and difficulty.",
            "",
            "## Source artifacts",
            "",
            f"- ECG five-seed aggregate: `{(ecg_root / 'multiseed_aggregate_summary.csv').name}`",
            f"- ECG alternate-group aggregate: `{(ecg_root / 'alternate_group_fairness_aggregate.csv').name}`",
            f"- ECG alternate-group per-seed results: `{(ecg_root / 'alternate_group_fairness_per_seed.csv').name}`",
            f"- Binary-ectopy five-seed analysis: `{display_path(binary_root / 'MULTISEED_ANALYSIS.md')}`",
            f"- Canonical three-task comparison: `{display_path(binary_root / 'BG_AAMI5_BINARY_COMPARISON.md')}`",
            f"- Binary locked protocol: `{display_path(binary_root / 'protocol_manifest.json')}`",
            "- MIT-BIH tuning and class-limitations roadmap: `../../docs/mitbih/ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`",
            f"- BG single-run grid: `{bg_root / 'fairness_comparison_results.csv'}`",
            f"- BG five-seed results: `{bg_root / 'multiseed_robustness_results.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    required_binary = [
        args.binary_root / "multiseed_aggregate_summary.csv",
        args.binary_root / "multiseed_per_seed_summary.csv",
    ]
    missing_binary = [str(path) for path in required_binary if not path.exists()]
    if missing_binary:
        raise SystemExit(f"Missing locked binary artifacts: {missing_binary}")
    per_seed_rows, alternate_rows, metadata = load_ecg_alternate_group_results(args.ecg_root)
    primary_rows = build_primary_comparison(args.ecg_root, args.bg_root, alternate_rows)

    write_csv(args.ecg_root / "alternate_group_fairness_per_seed.csv", per_seed_rows)
    write_csv(args.ecg_root / "alternate_group_fairness_aggregate.csv", alternate_rows)
    write_csv(args.ecg_root / "bg_ecg_primary_comparison.csv", primary_rows)
    report = generate_report(
        args.ecg_root,
        args.bg_root,
        args.binary_root,
        primary_rows,
        alternate_rows,
        metadata,
    )
    (args.ecg_root / "BG_ECG_CROSS_DOMAIN_COMPARISON.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
