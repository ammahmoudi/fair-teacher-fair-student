#!/usr/bin/env python3
"""Fairness analyzer for MIT-BIH ECG classification outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


class ECGClassifierFairnessAnalyzer:
    """Analyze subgroup fairness for AAMI 5-class ECG classification outputs."""

    def __init__(
        self,
        prediction_csv: str | Path,
        group_column: str = "sex",
        min_group_class_support: int = 20,
        min_best_group_recall: float = 0.05,
    ):
        self.prediction_csv = Path(prediction_csv)
        self.group_column = group_column
        self.min_group_class_support = int(min_group_class_support)
        self.min_best_group_recall = float(min_best_group_recall)
        self.df = pd.read_csv(self.prediction_csv)
        required = {"y_true", "y_pred", group_column}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")
        self.class_ids = sorted(int(v) for v in pd.unique(self.df["y_true"]))

    def analyze(self) -> Dict[str, object]:
        grouped = self._group_metrics()
        classwise = self._classwise_one_vs_rest(grouped)
        summary = self._overall_summary(grouped, classwise)
        return {
            "group_column": self.group_column,
            "min_group_class_support": self.min_group_class_support,
            "min_best_group_recall": self.min_best_group_recall,
            "overall": grouped,
            "classwise_one_vs_rest": classwise,
            "summary": summary,
        }

    def save_json(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.analyze(), f, indent=2)
        return output_path

    def _group_metrics(self) -> Dict[str, Dict[str, object]]:
        metrics: Dict[str, Dict[str, object]] = {}
        for group_value, group_df in self.df.groupby(self.group_column):
            y_true = group_df["y_true"].to_numpy()
            y_pred = group_df["y_pred"].to_numpy()
            metrics[str(group_value)] = {
                "count": int(len(group_df)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
            for class_id in self.class_ids:
                y_true_bin = (y_true == class_id).astype(int)
                y_pred_bin = (y_pred == class_id).astype(int)
                metrics[str(group_value)][f"support_class_{class_id}"] = int(y_true_bin.sum())
                metrics[str(group_value)][f"recall_class_{class_id}"] = float(
                    recall_score(y_true_bin, y_pred_bin, zero_division=0)
                )
        return metrics

    def _classwise_one_vs_rest(self, grouped_metrics: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        classwise: Dict[str, Dict[str, object]] = {}
        group_names = list(grouped_metrics.keys())
        for class_id in self.class_ids:
            recalls = [grouped_metrics[group][f"recall_class_{class_id}"] for group in group_names]
            if not recalls:
                continue
            supports = [int(grouped_metrics[group][f"support_class_{class_id}"]) for group in group_names]
            support_sufficient = all(support >= self.min_group_class_support for support in supports)
            performance_sufficient = max(recalls) >= self.min_best_group_recall
            eo_reportable = support_sufficient and performance_sufficient
            exclusion_reasons = []
            if not support_sufficient:
                exclusion_reasons.append("insufficient_group_support")
            if not performance_sufficient:
                exclusion_reasons.append("insufficient_class_recall")
            positive_rates = []
            accuracies = []
            for group_name in group_names:
                group_df = self.df[self.df[self.group_column] == group_name]
                y_true = (group_df["y_true"].to_numpy() == class_id).astype(int)
                y_pred = (group_df["y_pred"].to_numpy() == class_id).astype(int)
                positive_rates.append(float(np.mean(y_pred)) if len(y_pred) else 0.0)
                accuracies.append(float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0)
            classwise[str(class_id)] = {
                "group_support": dict(zip(group_names, supports)),
                "eo_reportable": eo_reportable,
                "eo_exclusion_reasons": exclusion_reasons,
                "raw_eo_gap": float(max(recalls) - min(recalls)),
                "eo_gap": float(max(recalls) - min(recalls)) if eo_reportable else None,
                "dp_gap": float(max(positive_rates) - min(positive_rates)) if positive_rates else 0.0,
                "fvo": float(max(accuracies) - min(accuracies)) if accuracies else 0.0,
                "max_recall_gap": float(max(recalls) - min(recalls)) if eo_reportable else None,
                "worst_group_recall": float(min(recalls)) if eo_reportable else None,
                "best_group_recall": float(max(recalls)) if eo_reportable else None,
            }
        if len(group_names) >= 2:
            macro_values = [grouped_metrics[group]["macro_f1"] for group in group_names]
            accuracy_values = [grouped_metrics[group]["accuracy"] for group in group_names]
            classwise["summary"] = {
                "macro_f1_gap": float(max(macro_values) - min(macro_values)),
                "accuracy_gap": float(max(accuracy_values) - min(accuracy_values)),
                "groups_compared": group_names,
            }
        return classwise

    def _overall_summary(self, grouped_metrics: Dict[str, Dict[str, object]], classwise: Dict[str, Dict[str, object]]) -> Dict[str, object]:
        class_entries = [value for key, value in classwise.items() if key != "summary"]
        if not class_entries:
            return {}
        reportable_eo_entries = [entry for entry in class_entries if entry["eo_reportable"]]
        reportable_eo_classes = [
            int(class_id)
            for class_id, entry in classwise.items()
            if class_id != "summary" and entry["eo_reportable"]
        ]
        excluded_eo_classes = [
            int(class_id)
            for class_id, entry in classwise.items()
            if class_id != "summary" and not entry["eo_reportable"]
        ]
        return {
            "avg_eo_gap": (
                float(np.mean([entry["eo_gap"] for entry in reportable_eo_entries]))
                if reportable_eo_entries else None
            ),
            "avg_dp_gap": float(np.mean([entry["dp_gap"] for entry in class_entries])),
            "avg_fvo": float(np.mean([entry["fvo"] for entry in class_entries])),
            "worst_class_eo_gap": (
                float(max(entry["eo_gap"] for entry in reportable_eo_entries))
                if reportable_eo_entries else None
            ),
            "worst_class_dp_gap": float(max(entry["dp_gap"] for entry in class_entries)),
            "worst_class_fvo": float(max(entry["fvo"] for entry in class_entries)),
            "reportable_eo_classes": reportable_eo_classes,
            "excluded_eo_classes": excluded_eo_classes,
            "groups": list(grouped_metrics.keys()),
        }
