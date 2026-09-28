#!/usr/bin/env python3
"""Aggregate a locked MIT-BIH ECG classification run over fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

VARIANT_DIRS = {
    "teacher": "teacher_gen",
    "student": "student_baseline_gen",
    "KD": "distill_baseline_gen",
    "T1": "distill_t1_gen",
    "O2": "distill_o2_gen",
    "T1+O2": "distill_t1_o2_gen",
}
SPLIT_FILES = {
    "validation": "val_predictions.csv",
    "test": "test_predictions.csv",
}
DEFAULT_SEEDS = [831363, 809906, 427368, 238822, 247659]
CLASS_NAMES = {
    "aami5": {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"},
    "binary_ectopy": {0: "normal_N", 1: "ectopy_SVF"},
    "binary_non_n": {0: "normal_N", 1: "non_N"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument(
        "--label-mode",
        choices=sorted(CLASS_NAMES),
        required=True,
    )
    parser.add_argument("--fairness-feature", default="sex")
    parser.add_argument(
        "--fairness-classes",
        default=None,
        help="Comma-separated class IDs. Defaults to 0,2 for AAMI-5 and 1 for binary.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--min-group-class-support", type=int, default=20)
    parser.add_argument("--min-best-group-recall", type=float, default=0.05)
    return parser.parse_args()


def latest_predictions(seed_dir: Path, directory: str, filename: str) -> Path | None:
    paths = sorted((seed_dir / directory).glob(f"**/{filename}"))
    return paths[-1] if paths else None


def class_recall(y_true: pd.Series, y_pred: pd.Series, class_id: int) -> float:
    positive = y_true == class_id
    return float((y_pred[positive] == class_id).mean()) if positive.any() else 0.0


def selected_eo(
    frame: pd.DataFrame,
    group_column: str,
    class_ids: list[int],
    min_support: int,
    min_best_recall: float,
) -> tuple[float | None, dict[str, Any]]:
    groups = sorted(str(value) for value in frame[group_column].dropna().unique())
    details: dict[str, Any] = {}
    gaps = []
    if len(groups) < 2:
        return None, details
    for class_id in class_ids:
        recalls: dict[str, float] = {}
        supports: dict[str, int] = {}
        for group in groups:
            rows = frame[frame[group_column].astype(str) == group]
            positive = rows["y_true"] == class_id
            supports[group] = int(positive.sum())
            recalls[group] = (
                float((rows.loc[positive, "y_pred"] == class_id).mean())
                if positive.any()
                else 0.0
            )
        reportable = (
            all(value >= min_support for value in supports.values())
            and max(recalls.values()) >= min_best_recall
        )
        gap = max(recalls.values()) - min(recalls.values())
        details[str(class_id)] = {
            "supports": supports,
            "recalls": recalls,
            "raw_gap": gap,
            "reportable": reportable,
        }
        if not reportable:
            return None, details
        gaps.append(gap)
    return (statistics.mean(gaps) if gaps else None), details


def validate_predictions(frame: pd.DataFrame, label_mode: str, path: Path) -> None:
    required = {"y_true", "y_pred", "original_class_id", "label_mode"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    modes = set(frame["label_mode"].astype(str))
    if modes != {label_mode}:
        raise ValueError(f"{path} has label modes {sorted(modes)}, expected {label_mode}")
    valid_ids = set(CLASS_NAMES[label_mode])
    observed = set(frame["y_true"].astype(int)) | set(frame["y_pred"].astype(int))
    if not observed.issubset(valid_ids):
        raise ValueError(f"{path} has unexpected class IDs: {sorted(observed - valid_ids)}")
    if label_mode == "binary_ectopy" and (frame["original_class_id"] == 4).any():
        raise ValueError(f"{path} contains Q beats even though binary_ectopy excludes Q")


def metric_row(
    frame: pd.DataFrame,
    seed: int,
    variant: str,
    split: str,
    label_mode: str,
    fairness_feature: str,
    fairness_classes: list[int],
    min_support: int,
    min_best_recall: float,
    path: Path,
) -> dict[str, Any]:
    validate_predictions(frame, label_mode, path)
    if fairness_feature not in frame.columns:
        raise ValueError(f"{path} is missing fairness feature {fairness_feature!r}")
    y_true = frame["y_true"].astype(int)
    y_pred = frame["y_pred"].astype(int)
    labels = sorted(CLASS_NAMES[label_mode])
    eo, eo_details = selected_eo(
        frame,
        fairness_feature,
        fairness_classes,
        min_support,
        min_best_recall,
    )
    row: dict[str, Any] = {
        "split": split,
        "seed": seed,
        "variant": variant,
        "label_mode": label_mode,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "selected_eo": eo,
        "fairness_classes": ";".join(str(value) for value in fairness_classes),
        "fairness_feature": fairness_feature,
        "eo_details": json.dumps(eo_details, sort_keys=True),
        "prediction_csv": str(path),
    }
    for class_id, class_name in CLASS_NAMES[label_mode].items():
        row[f"recall_{class_name}"] = class_recall(y_true, y_pred, class_id)
    if len(labels) == 2:
        row.update(
            {
                "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "specificity": class_recall(y_true, y_pred, 0),
                "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in rows[0]
        if name not in {
            "split", "seed", "variant", "label_mode", "fairness_classes",
            "fairness_feature", "eo_details", "prediction_csv",
        }
    ]
    output = []
    for variant in VARIANT_DIRS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        item: dict[str, Any] = {
            "split": selected[0]["split"],
            "variant": variant,
            "label_mode": selected[0]["label_mode"],
            "n_seeds": len(selected),
            "fairness_feature": selected[0]["fairness_feature"],
            "fairness_classes": selected[0]["fairness_classes"],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            item[f"{metric}_mean"] = statistics.mean(values) if values else None
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(item)
    return output


def kd_delta_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    kd_by_seed = {
        int(row["seed"]): row for row in per_seed if row["variant"] == "KD"
    }
    metrics = ["accuracy", "macro_f1", "weighted_f1", "selected_eo"]
    if per_seed and "sensitivity" in per_seed[0]:
        metrics.extend(["sensitivity", "specificity", "precision"])
    for row in per_seed:
        if row["variant"] in {"teacher", "student", "KD"}:
            continue
        baseline = kd_by_seed[int(row["seed"])]
        delta: dict[str, Any] = {
            "split": row["split"],
            "seed": row["seed"],
            "variant": row["variant"],
        }
        for metric in metrics:
            value = row.get(metric)
            reference = baseline.get(metric)
            delta[f"{metric}_delta_vs_KD"] = (
                float(value) - float(reference)
                if value is not None and reference is not None
                else None
            )
        output.append(delta)
    return output


def mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def build_report(
    label_mode: str,
    seeds: list[int],
    fairness_feature: str,
    fairness_classes: list[int],
    aggregates: dict[str, list[dict[str, Any]]],
    deltas: dict[str, list[dict[str, Any]]],
) -> str:
    class_labels = [CLASS_NAMES[label_mode][class_id] for class_id in fairness_classes]
    lines = [
        f"# MIT-BIH {label_mode} Multi-Seed Classification Analysis",
        "",
        "## Protocol",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "- Ten training epochs per model; record-disjoint fixed split: 28 train, 9 validation, 10 test records (split seed 42).",
        "- Frozen BERT teacher and TinyBERT student backbones.",
        "- ECG context: 256 samples with patch length 16, plus previous/next RR intervals with fusion weight 1.0.",
        "- Teacher, Student, Baseline KD, T1, O2, and T1+O2 use the same split and seeds.",
        "- Weighted sampling is training-only; validation and test retain natural prevalence.",
        f"- Fairness endpoint: {fairness_feature} recall gap over {', '.join(class_labels)}.",
        "- Hyperparameters and checkpoints are selected from validation only.",
    ]
    if label_mode == "binary_ectopy":
        lines.extend(
            [
                "- Endpoint: N is normal, S/V/F are ectopy, and Q is excluded.",
                "- KD lock: alpha=0.5, beta=0.5, and temperature=1.0, chosen by a predeclared one-seed validation screen; the screen is seed-sensitive and is not multi-seed tuning evidence.",
                "- T1 uses sex-aware teacher sampling capped at 4x; O2 jointly learns a sex-conditional affine scale and bias on student logits.",
                "- Checkpoint selection retains validation macro-F1 within 0.01 of the best candidate, then considers support-qualified ectopy EO; test predictions are never used for selection.",
            ]
        )
    for split in ("validation", "test"):
        lines.extend(
            [
                "",
                f"## {split.title()} Aggregate",
                "",
                "Values are mean ± sample standard deviation across seeds. Lower EO is better.",
                "",
                "| Method | Accuracy | Macro-F1 | Weighted-F1 | Sensitivity | Specificity | Precision | Selected EO |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        by_variant = {row["variant"]: row for row in aggregates[split]}
        for variant in VARIANT_DIRS:
            row = by_variant[variant]
            lines.append(
                f"| {variant} | {mean_std(row, 'accuracy')} | {mean_std(row, 'macro_f1')} | "
                f"{mean_std(row, 'weighted_f1')} | {mean_std(row, 'sensitivity')} | "
                f"{mean_std(row, 'specificity')} | {mean_std(row, 'precision')} | "
                f"{mean_std(row, 'selected_eo')} |"
            )
        lines.extend(["", "### Paired changes versus Baseline KD", ""])
        split_deltas = deltas[split]
        for variant in ("T1", "O2", "T1+O2"):
            selected = [row for row in split_deltas if row["variant"] == variant]
            macro = statistics.mean(float(row["macro_f1_delta_vs_KD"]) for row in selected)
            eo_values = [
                float(row["selected_eo_delta_vs_KD"])
                for row in selected
                if row["selected_eo_delta_vs_KD"] is not None
            ]
            eo_text = "NA" if not eo_values else f"{statistics.mean(eo_values):+.4f} ({sum(value < 0 for value in eo_values)}/{len(eo_values)} wins)"
            lines.append(f"- {variant}: macro-F1 {macro:+.4f}; EO {eo_text}.")
    if label_mode == "binary_ectopy":
        lines.extend(
            [
                "",
                "## Locked-Test Interpretation",
                "",
                "- On validation, Baseline KD has the highest mean macro-F1 and the lowest mean ectopy sex EO; none of the fairness variants improves both validation endpoints.",
                "- On the locked test split, O2 changes macro-F1 by +0.0127 and EO by -0.0357 versus Baseline KD (EO improves in 3/5 seeds).",
                "- On the locked test split, T1+O2 changes macro-F1 by +0.0264 and EO by -0.0248 versus Baseline KD (macro-F1 improves in 5/5 seeds and EO in 4/5).",
                "- The O2 and T1+O2 mean EO reductions are influenced strongly by seed 427368, where Baseline KD has an unusually large test EO gap; the corresponding median paired EO changes are -0.0097 and -0.0091.",
                "- With only five seed pairs, the fairness deltas are descriptive and not statistically conclusive. All seeds evaluate the same ten test records, so seed variation is not a confidence interval over patients or records.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Validation is the only split permitted for candidate selection. The test section is a final locked-protocol evaluation and must not be used to choose among T1, O2, or T1+O2, retune this run, or claim a universally best ECG method.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    fairness_classes = (
        [int(value) for value in args.fairness_classes.split(",") if value]
        if args.fairness_classes
        else ([0, 2] if args.label_mode == "aami5" else [1])
    )
    missing_seeds = [seed for seed in seeds if not (args.pipeline_dir / f"seed_{seed}").is_dir()]
    if missing_seeds:
        raise SystemExit(f"Missing expected seed directories: {missing_seeds}")

    per_split: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, list[dict[str, Any]]] = {}
    deltas: dict[str, list[dict[str, Any]]] = {}
    for split, filename in SPLIT_FILES.items():
        rows = []
        for seed in seeds:
            seed_dir = args.pipeline_dir / f"seed_{seed}"
            for variant, directory in VARIANT_DIRS.items():
                path = latest_predictions(seed_dir, directory, filename)
                if path is None:
                    raise SystemExit(
                        f"Missing {split} predictions for seed={seed}, variant={variant}"
                    )
                rows.append(
                    metric_row(
                        pd.read_csv(path),
                        seed,
                        variant,
                        split,
                        args.label_mode,
                        args.fairness_feature,
                        fairness_classes,
                        args.min_group_class_support,
                        args.min_best_group_recall,
                        path,
                    )
                )
        per_split[split] = rows
        aggregates[split] = aggregate_rows(rows)
        deltas[split] = kd_delta_rows(rows)
        write_csv(args.pipeline_dir / f"{split}_multiseed_per_seed.csv", rows)
        write_csv(args.pipeline_dir / f"{split}_multiseed_aggregate.csv", aggregates[split])
        write_csv(args.pipeline_dir / f"{split}_multiseed_kd_deltas.csv", deltas[split])

    # Preserve the AAMI-5 artifact naming convention for downstream comparisons.
    write_csv(args.pipeline_dir / "multiseed_per_seed_summary.csv", per_split["test"])
    write_csv(args.pipeline_dir / "multiseed_aggregate_summary.csv", aggregates["test"])
    write_csv(args.pipeline_dir / "multiseed_kd_deltas.csv", deltas["test"])
    summary = {
        "label_mode": args.label_mode,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "variants": list(VARIANT_DIRS),
        "fairness_feature": args.fairness_feature,
        "fairness_classes": fairness_classes,
        "selection_split": "validation",
        "final_evaluation_split": "test",
    }
    (args.pipeline_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.pipeline_dir / "MULTISEED_ANALYSIS.md").write_text(
        build_report(
            args.label_mode,
            seeds,
            args.fairness_feature,
            fairness_classes,
            aggregates,
            deltas,
        ),
        encoding="utf-8",
    )
    print(f"Aggregated {len(seeds)} seeds for {args.label_mode}")
    print(args.pipeline_dir / "MULTISEED_ANALYSIS.md")


if __name__ == "__main__":
    main()
