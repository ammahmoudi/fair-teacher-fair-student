"""Sampling helpers for MIT-BIH ECG classification fairness experiments."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_group_labels_from_samples(dataset, feature: str) -> np.ndarray:
    """Extract one group label per ECG beat sample from dataset metadata."""
    valid_features = {"sex", "age_group", "paced_group", "difficulty_group"}
    if feature not in valid_features:
        raise ValueError(f"Unsupported ECG fairness feature: {feature}. Expected one of {sorted(valid_features)}")
    return np.asarray([getattr(sample, feature) for sample in dataset.samples], dtype=object)


def build_group_class_sampler(
    dataset,
    group_labels: np.ndarray,
    max_oversample: float = 4.0,
) -> Tuple[WeightedRandomSampler, Dict[str, object]]:
    """Create a sampler that upweights rare group/class combinations.

    This is the ECG classification analogue of fair teacher sampling.
    """
    if len(group_labels) != len(dataset.samples):
        raise ValueError("group_labels length must match number of ECG samples")

    class_ids = np.asarray([sample.class_id for sample in dataset.samples], dtype=int)
    combo_counter = Counter((str(group), int(class_id)) for group, class_id in zip(group_labels, class_ids))
    mean_count = float(np.mean(list(combo_counter.values()))) if combo_counter else 1.0

    weights = []
    for group, class_id in zip(group_labels, class_ids):
        combo_count = combo_counter[(str(group), int(class_id))]
        raw_weight = mean_count / max(1.0, float(combo_count))
        weights.append(min(max_oversample, raw_weight))

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    info = {
        "group_class_counts": {f"{group}|{class_id}": count for (group, class_id), count in combo_counter.items()},
        "max_oversample": float(max_oversample),
        "mean_weight": float(weights.mean()),
    }
    return sampler, info


def build_class_balanced_sampler(
    dataset,
    max_oversample: float = 50.0,
) -> Tuple[WeightedRandomSampler, Dict[str, object]]:
    """Create a sampler that upweights rare AAMI classes only (no demographic
    grouping). Intended for teacher/student baseline training, where MIT-BIH's
    extreme class imbalance (e.g. class F is ~0.07% of beats, ~1000x rarer
    than class N) can otherwise cause the model to collapse to always
    predicting the majority class, even with class-weighted loss alone.

    `max_oversample` defaults higher than the group-aware `max_oversample=4.0`
    used for T1 fairness sampling, since class imbalance here is far more
    extreme than the demographic imbalance T1 was designed for.
    """
    class_ids = np.asarray([sample.class_id for sample in dataset.samples], dtype=int)
    class_counter = Counter(int(class_id) for class_id in class_ids)
    mean_count = float(np.mean(list(class_counter.values()))) if class_counter else 1.0

    weights = []
    for class_id in class_ids:
        class_count = class_counter[int(class_id)]
        raw_weight = mean_count / max(1.0, float(class_count))
        weights.append(min(max_oversample, raw_weight))

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    info = {
        "class_counts": {str(class_id): count for class_id, count in class_counter.items()},
        "max_oversample": float(max_oversample),
        "mean_weight": float(weights.mean()),
    }
    return sampler, info
