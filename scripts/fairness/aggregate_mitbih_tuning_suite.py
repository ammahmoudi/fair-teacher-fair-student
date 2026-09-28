#!/usr/bin/env python3
"""Aggregate resumable MIT-BIH tuning-suite cases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


VARIANT_DIRS = {
    "teacher": "teacher_gen",
    "student_baseline": "student_baseline_gen",
    "distilled_baseline": "distill_baseline_gen",
    "distilled_t1": "distill_t1_gen",
    "distilled_o2": "distill_o2_gen",
    "distilled_t1_o2": "distill_t1_o2_gen",
}
DISPLAY_NAMES = {
    "teacher": "Teacher",
    "student_baseline": "Student",
    "distilled_baseline": "Baseline KD",
    "distilled_t1": "T1",
    "distilled_o2": "O2",
    "distilled_t1_o2": "T1+O2",
}
PREDICTION_SPLIT_NAMES = {
    "validation": "val",
    "test": "test",
}
DISTILL_VARIANT_KEYS = {
    "baseline": "distilled_baseline",
    "t1": "distilled_t1",
    "o2": "distilled_o2",
    "t1_o2": "distilled_t1_o2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def latest_predictions(seed_dir: Path, variant: str, evaluation_split: str) -> Path | None:
    directory = VARIANT_DIRS.get(variant)
    if directory is None:
        return None
    filename_split = PREDICTION_SPLIT_NAMES[evaluation_split]
    paths = sorted(
        (seed_dir / directory).glob(f"**/{filename_split}_predictions.csv")
    )
    return paths[-1] if paths else None


def selected_eo(
    predictions: pd.DataFrame,
    fairness_feature: str,
    class_ids: list[int],
    min_support: int = 20,
    min_best_group_recall: float = 0.05,
) -> float | None:
    if fairness_feature not in predictions.columns:
        return None
    groups = predictions[fairness_feature].dropna().unique()
    if len(groups) < 2:
        return None
    gaps = []
    for class_id in class_ids:
        recalls = []
        for group in groups:
            group_rows = predictions[predictions[fairness_feature] == group]
            positive = group_rows["y_true"] == class_id
            if int(positive.sum()) < min_support:
                return None
            recalls.append(
                float((group_rows.loc[positive, "y_pred"] == class_id).mean())
            )
        if max(recalls) < min_best_group_recall:
            return None
        gaps.append(max(recalls) - min(recalls))
    return statistics.mean(gaps) if gaps else None

def expected_variant_keys(manifest: dict[str, Any]) -> list[str]:
    variants = manifest.get("distill_variants", list(DISTILL_VARIANT_KEYS))
    unknown = sorted(set(variants).difference(DISTILL_VARIANT_KEYS))
    if unknown:
        raise ValueError(f"Unsupported distillation variants in case manifest: {unknown}")
    return ["teacher", "student_baseline"] + [
        DISTILL_VARIANT_KEYS[value] for value in variants
    ]

def seed_is_complete(found_variants: list[str], expected_variants: list[str]) -> bool:
    return set(expected_variants).issubset(found_variants)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    case_dirs = sorted((args.suite_dir / "cases").glob("*"))
    per_seed_rows: list[dict[str, Any]] = []
    status_rows = []

    for case_dir in case_dirs:
        manifest_path = case_dir / "case_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_name = manifest["case_name"]
        fairness_feature = manifest.get("fairness_feature", "sex")
        evaluation_split = manifest.get("aggregation_split", "validation")
        if evaluation_split not in {"validation", "test"}:
            raise ValueError(
                f"Unsupported aggregation_split={evaluation_split!r} for {case_name}"
            )
        fairness_classes = [int(value) for value in manifest["fairness_classes"]]
        expected_seeds = [int(value) for value in manifest["seeds"]]
        expected_variants = expected_variant_keys(manifest)
        completed_seeds = []
        seed_variant_status: dict[str, list[str]] = {}

        for seed in expected_seeds:
            seed_dir = case_dir / f"seed_{seed}"
            found_variants = []
            for variant in VARIANT_DIRS:
                prediction_path = latest_predictions(
                    seed_dir,
                    variant,
                    evaluation_split,
                )
                if prediction_path is None:
                    continue
                found_variants.append(variant)
                predictions = pd.read_csv(prediction_path)
                y_true = predictions["y_true"].to_numpy(dtype=int)
                y_pred = predictions["y_pred"].to_numpy(dtype=int)
                target_class = 1
                target_mask = y_true == target_class
                target_recall = (
                    float((y_pred[target_mask] == target_class).mean())
                    if target_mask.any()
                    else None
                )
                per_seed_rows.append(
                    {
                        "case": case_name,
                        "phase": manifest["phase"],
                        "label_mode": manifest["label_mode"],
                        "fairness_feature": fairness_feature,
                        "evaluation_split": evaluation_split,
                        "sequence_length": manifest["sequence_length"],
                        "use_rr_features": manifest["use_rr_features"],
                        "seed": seed,
                        "variant": DISPLAY_NAMES.get(variant, variant),
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                        "selected_eo": selected_eo(
                            predictions,
                            fairness_feature,
                            fairness_classes,
                        ),
                        "target_class_recall": target_recall,
                        "prediction_csv": str(prediction_path),
                    }
                )
            seed_variant_status[str(seed)] = found_variants
            if seed_is_complete(found_variants, expected_variants):
                completed_seeds.append(seed)

        status_rows.append(
            {
                "case": case_name,
                "expected_seeds": expected_seeds,
                "completed_seeds": completed_seeds,
                "expected_variants": expected_variants,
                "seed_variants": seed_variant_status,
                "complete": set(completed_seeds) == set(expected_seeds),
            }
        )

    per_seed_fields = [
        "case", "phase", "label_mode", "fairness_feature", "evaluation_split",
        "sequence_length", "use_rr_features", "seed", "variant", "accuracy",
        "macro_f1", "weighted_f1", "selected_eo", "target_class_recall",
        "prediction_csv",
    ]
    write_csv(args.suite_dir / "suite_per_seed.csv", per_seed_rows, per_seed_fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(str(row["case"]), str(row["variant"]))].append(row)

    aggregate_rows = []
    for (case, variant), rows in sorted(grouped.items()):
        baseline_by_seed = {
            int(row["seed"]): row
            for row in per_seed_rows
            if row["case"] == case and row["variant"] == "Baseline KD"
        }

        def aggregate(key: str) -> tuple[float | None, float | None]:
            values = [float(row[key]) for row in rows if row[key] is not None]
            if not values:
                return None, None
            return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

        accuracy_mean, accuracy_std = aggregate("accuracy")
        macro_mean, macro_std = aggregate("macro_f1")
        weighted_mean, weighted_std = aggregate("weighted_f1")
        eo_mean, eo_std = aggregate("selected_eo")
        target_mean, target_std = aggregate("target_class_recall")
        paired_eo_deltas = [
            float(row["selected_eo"]) - float(baseline_by_seed[int(row["seed"])]["selected_eo"])
            for row in rows
            if row["selected_eo"] is not None
            and int(row["seed"]) in baseline_by_seed
            and baseline_by_seed[int(row["seed"])]["selected_eo"] is not None
        ]
        first = rows[0]
        aggregate_rows.append(
            {
                "case": case,
                "phase": first["phase"],
                "label_mode": first["label_mode"],
                "fairness_feature": first["fairness_feature"],
                "evaluation_split": first["evaluation_split"],
                "sequence_length": first["sequence_length"],
                "use_rr_features": first["use_rr_features"],
                "variant": variant,
                "n_seeds": len(rows),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "macro_f1_mean": macro_mean,
                "macro_f1_std": macro_std,
                "weighted_f1_mean": weighted_mean,
                "weighted_f1_std": weighted_std,
                "selected_eo_mean": eo_mean,
                "selected_eo_std": eo_std,
                "target_class_recall_mean": target_mean,
                "target_class_recall_std": target_std,
                "eo_delta_vs_kd_mean": statistics.mean(paired_eo_deltas) if paired_eo_deltas else None,
                "eo_wins_vs_kd": sum(delta < 0 for delta in paired_eo_deltas),
                "eo_paired_seeds": len(paired_eo_deltas),
            }
        )

    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else [
        "case", "phase", "label_mode", "fairness_feature", "evaluation_split",
        "sequence_length", "use_rr_features", "variant", "n_seeds",
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "weighted_f1_std",
        "selected_eo_mean", "selected_eo_std", "target_class_recall_mean",
        "target_class_recall_std", "eo_delta_vs_kd_mean", "eo_wins_vs_kd",
        "eo_paired_seeds",
    ]
    write_csv(args.suite_dir / "suite_aggregate.csv", aggregate_rows, aggregate_fields)
    (args.suite_dir / "suite_status.json").write_text(
        json.dumps(status_rows, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# MIT-BIH Tuning Suite Results",
        "",
        "Candidate summaries use the manifest's evaluation split (validation by default). Lower selected EO is better. Target recall is S recall for AAMI-5 and positive-class recall for binary modes.",
        "",
        "| Case | Phase | Split | Label | Model | Seeds | Accuracy | Macro-F1 | Selected EO | Target Recall | EO Δ vs KD |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        def value(name: str) -> str:
            item = row[name]
            return "NA" if item is None else f"{float(item):.4f}"

        lines.append(
            f"| {row['case']} | {row['phase']} | {row['evaluation_split']} | "
            f"{row['label_mode']} | {row['variant']} | "
            f"{row['n_seeds']} | {value('accuracy_mean')} | {value('macro_f1_mean')} | "
            f"{value('selected_eo_mean')} | {value('target_class_recall_mean')} | "
            f"{value('eo_delta_vs_kd_mean')} |"
        )
    lines.extend(["", "## Completion Status", ""])
    for status in status_rows:
        marker = "complete" if status["complete"] else "incomplete"
        lines.append(
            f"- {status['case']}: **{marker}** — {len(status['completed_seeds'])}/{len(status['expected_seeds'])} seeds"
        )
    (args.suite_dir / "SUITE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Aggregated {len(per_seed_rows)} per-seed model rows across {len(status_rows)} cases")
    print(args.suite_dir / "SUITE_RESULTS.md")


if __name__ == "__main__":
    main()
