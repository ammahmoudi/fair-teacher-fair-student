#!/usr/bin/env python3
"""Run fairness analysis for MIT-BIH ECG classifier predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fairness.analyzers.ecg_classifier_fairness_analyzer import ECGClassifierFairnessAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIT-BIH ECG classifier fairness analysis")
    parser.add_argument("--prediction-csv", default=None, type=Path)
    parser.add_argument(
        "--prediction-csvs",
        default=None,
        help="Comma-separated list of named CSVs in the form name=path,name2=path2 for teacher/student/distilled comparison",
    )
    parser.add_argument("--group-column", default="sex")
    parser.add_argument("--min-group-class-support", type=int, default=20)
    parser.add_argument("--min-best-group-recall", type=float, default=0.05)
    parser.add_argument("--output-json", default=None, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prediction_csvs:
        outputs = {}
        pairs = [item.strip() for item in args.prediction_csvs.split(",") if item.strip()]
        for pair in pairs:
            name, path_str = pair.split("=", 1)
            analyzer = ECGClassifierFairnessAnalyzer(
                Path(path_str),
                group_column=args.group_column,
                min_group_class_support=args.min_group_class_support,
                min_best_group_recall=args.min_best_group_recall,
            )
            outputs[name] = analyzer.analyze()
        output_json = args.output_json or Path("./fairness_mitbih_comparison.json")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(outputs, f, indent=2)
        print(f"Saved fairness comparison report to {output_json}")
    else:
        if args.prediction_csv is None:
            raise ValueError("Provide --prediction-csv or --prediction-csvs")
        analyzer = ECGClassifierFairnessAnalyzer(
            args.prediction_csv,
            group_column=args.group_column,
            min_group_class_support=args.min_group_class_support,
            min_best_group_recall=args.min_best_group_recall,
        )
        output_json = args.output_json or args.prediction_csv.parent / f"fairness_{args.group_column}.json"
        path = analyzer.save_json(output_json)
        print(f"Saved fairness report to {path}")
