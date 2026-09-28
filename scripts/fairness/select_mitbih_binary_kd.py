#!/usr/bin/env python3
"""Select binary-ectopy KD parameters from validation-only grid results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CASE_PARAMETERS = {
    "binary_kd_a03_b03_t2": {"alpha": 0.3, "beta": 0.3, "temperature": 2.0},
    "binary_kd_a07_b03_t2": {"alpha": 0.7, "beta": 0.3, "temperature": 2.0},
    "binary_kd_a05_b05_t2": {"alpha": 0.5, "beta": 0.5, "temperature": 2.0},
    "binary_kd_a03_b07_t2": {"alpha": 0.3, "beta": 0.7, "temperature": 2.0},
    "binary_kd_a05_b05_t1": {"alpha": 0.5, "beta": 0.5, "temperature": 1.0},
    "binary_kd_a05_b05_t4": {"alpha": 0.5, "beta": 0.5, "temperature": 4.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--macro-f1-tolerance", type=float, default=0.01)
    parser.add_argument("--min-seeds", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(value: str) -> float | None:
    return None if value in {"", "None", "NA"} else float(value)


def main() -> None:
    args = parse_args()
    aggregate_path = args.suite_dir / "suite_aggregate.csv"
    if not aggregate_path.exists():
        raise SystemExit(f"Missing suite aggregate: {aggregate_path}")
    rows = [
        row
        for row in read_rows(aggregate_path)
        if row["case"] in CASE_PARAMETERS and row["variant"] == "Baseline KD"
    ]
    by_case = {row["case"]: row for row in rows}
    missing = sorted(set(CASE_PARAMETERS).difference(by_case))
    if missing:
        raise SystemExit(f"Missing binary KD candidates: {missing}")

    candidates: list[dict[str, Any]] = []
    for case, parameters in CASE_PARAMETERS.items():
        row = by_case[case]
        if row["evaluation_split"] != "validation":
            raise SystemExit(f"Candidate {case} uses {row['evaluation_split']}, not validation")
        if row["label_mode"] != "binary_ectopy":
            raise SystemExit(f"Candidate {case} has label_mode={row['label_mode']}")
        n_seeds = int(row["n_seeds"])
        if n_seeds < args.min_seeds:
            raise SystemExit(
                f"Candidate {case} has only {n_seeds} seeds; need at least {args.min_seeds}"
            )
        candidates.append(
            {
                "case": case,
                **parameters,
                "n_seeds": n_seeds,
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "macro_f1_std": float(row["macro_f1_std"]),
                "sensitivity_mean": float(row["target_class_recall_mean"]),
                "sensitivity_std": float(row["target_class_recall_std"]),
                "selected_eo_mean": optional_float(row["selected_eo_mean"]),
                "selected_eo_std": optional_float(row["selected_eo_std"]),
            }
        )

    best_macro_f1 = max(item["macro_f1_mean"] for item in candidates)
    utility_eligible = [
        item
        for item in candidates
        if item["macro_f1_mean"] >= best_macro_f1 - args.macro_f1_tolerance
    ]
    selected = min(
        utility_eligible,
        key=lambda item: (
            item["selected_eo_mean"]
            if item["selected_eo_mean"] is not None
            else float("inf"),
            -item["sensitivity_mean"],
            -item["macro_f1_mean"],
        ),
    )
    payload = {
        "protocol": "mitbih_binary_ectopy_kd_selection_v1",
        "screening_evidence": len({item["n_seeds"] for item in candidates}) == 1
        and candidates[0]["n_seeds"] == 1,
        "selection_split": "validation",
        "label_mode": "binary_ectopy",
        "selection_rule": {
            "macro_f1_tolerance": args.macro_f1_tolerance,
            "primary_gate": "macro_f1 within tolerance of best",
            "tie_break_1": "lowest support/performance-qualified ectopy sex EO",
            "tie_break_2": "highest ectopy sensitivity",
            "test_metrics_used": False,
        },
        "selected": selected,
        "candidates": candidates,
    }
    output_path = args.suite_dir / "selected_kd_protocol.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Binary-Ectopy KD Validation Selection",
        "",
        "Selection uses validation only. Test predictions are not read by this selector.",
        "This is a fast one-seed screening lock when each candidate has one seed; it is not independent multi-seed tuning evidence.",
        "",
        "| Case | Alpha | Beta | Temperature | Seeds | Macro-F1 | Sensitivity | Ectopy sex EO | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    eligible_names = {item["case"] for item in utility_eligible}
    for item in candidates:
        eo = "NA" if item["selected_eo_mean"] is None else f"{item['selected_eo_mean']:.4f}"
        lines.append(
            f"| {item['case']} | {item['alpha']:.1f} | {item['beta']:.1f} | "
            f"{item['temperature']:.1f} | {item['n_seeds']} | "
            f"{item['macro_f1_mean']:.4f} | {item['sensitivity_mean']:.4f} | "
            f"{eo} | {'yes' if item['case'] in eligible_names else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Selected",
            "",
            f"- Case: `{selected['case']}`",
            f"- Alpha/beta: `{selected['alpha']}/{selected['beta']}`",
            f"- Temperature: `{selected['temperature']}`",
            "- This lock must be used unchanged for Baseline KD, T1, O2, and T1+O2 in the final five-seed run.",
        ]
    )
    (args.suite_dir / "KD_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print(args.suite_dir / "KD_SELECTION.md")


if __name__ == "__main__":
    main()
