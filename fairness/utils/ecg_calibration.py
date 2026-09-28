"""Groupwise calibration utilities for MIT-BIH ECG classification.

This is the ECG classification analogue of output-level calibration ideas used
in the BG fairness work.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def fit_groupwise_logit_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    group_labels: Iterable,
    num_classes: int,
    steps: int = 200,
    lr: float = 0.1,
    class_weights: np.ndarray | None = None,
) -> Dict[str, object]:
    """Fit a small additive logit bias per group on validation data.

    The fitted parameters can later be applied as:
        adjusted_logits = logits + group_bias[group]

    `class_weights` (length `num_classes`, e.g. inverse class frequency) is
    strongly recommended for imbalanced multi-class problems like MIT-BIH
    AAMI classification: fitting with plain (unweighted) cross-entropy on a
    dataset where one class is >100x more frequent than others causes the
    bias to chase majority-class accuracy, actively re-collapsing the model
    toward the majority class instead of calibrating it fairly.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    group_labels = np.asarray(list(group_labels), dtype=object)
    unique_groups = [str(g) for g in sorted(pd_unique(group_labels))]
    group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
    group_idx = torch.tensor([group_to_idx[str(g)] for g in group_labels], dtype=torch.long)

    class_weight_t = None
    if class_weights is not None:
        class_weight_t = torch.tensor(class_weights, dtype=torch.float32)

    bias = torch.zeros((len(unique_groups), num_classes), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=lr)

    for _ in range(steps):
        adjusted = logits_t + bias[group_idx]
        loss = F.cross_entropy(adjusted, labels_t, weight=class_weight_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "groups": unique_groups,
        "group_biases": bias.detach().cpu().numpy().tolist(),
        "num_classes": num_classes,
        "steps": steps,
        "lr": lr,
        "class_weighted": class_weights is not None,
    }


def apply_groupwise_logit_bias(logits: np.ndarray, group_labels: Iterable, metadata: Dict[str, object]) -> np.ndarray:
    groups = metadata["groups"]
    group_to_idx = {str(group): idx for idx, group in enumerate(groups)}
    biases = np.asarray(metadata["group_biases"], dtype=np.float32)
    adjusted = []
    for logit, group in zip(logits, group_labels):
        adjusted.append(logit + biases[group_to_idx[str(group)]])
    return np.asarray(adjusted, dtype=np.float32)


def pd_unique(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
