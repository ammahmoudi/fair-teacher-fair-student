"""
Fair sampling utilities for training DataLoaders.

Provides WeightedRandomSampler construction that oversamples minority-group
hypoglycemic windows to reduce between-group prevalence imbalance.

Used by:
 - data_processing/data_loader.py   (teacher fair training, T1)
 - distillation/core/distillation_wrapper.py  (student distillation, existing)
"""
import logging
import numpy as np
from typing import Optional, Dict


# Canonical demographics are loaded from the fairness module (single source of truth)
try:
    from fairness.utils.analyzer_utils import get_ohiot1dm_default_data
    _DEFAULT_DEMOGRAPHICS = get_ohiot1dm_default_data()
except ImportError:
    _DEFAULT_DEMOGRAPHICS = {}

# Binary label mappings per feature
_GROUP_LABEL_MAPS = {
    'gender':  lambda v: 1 if v == 'Male' else 0,
    'age':     lambda v: 1 if v == '60-80' else 0,
    'pump':    lambda v: 1 if v == '630G'  else 0,
    'sensor':  lambda v: 1 if v == 'Basis' else 0,
    'cohort':  lambda v: 1 if v == '2020'  else 0,
}


def build_group_labels(
    data_path: str,
    seq_len: int,
    pred_len: int,
    val_split: float = 0,
    percent: int = 100,
    feature: str = 'gender',
    demographics: Optional[Dict] = None,
) -> Optional[np.ndarray]:
    """Return per-window integer group labels (0 or 1) for a training CSV.

    Each window is assigned the label of its majority patient.  Replicates
    the same sorting / slicing logic as Dataset_T1DM.__read_data__ so that
    window indices align perfectly.

    Args:
        data_path:    Path to the training CSV (must have item_id, timestamp, target).
        seq_len:      Sequence length (look-back) used by the model.
        pred_len:     Prediction horizon used by the model.
        val_split:    Validation fraction in percent (same as Dataset_T1DM).
        percent:      Data percentage to use (same as Dataset_T1DM).
        feature:      Demographic feature to split on (default: 'gender').
        demographics: Override for patient demographics dict.  Defaults to the
                      canonical OhioT1DM demographics from analyzer_utils.

    Returns:
        np.ndarray of shape (n_windows,) with values 0 or 1, or None on error.
    """
    import pandas as pd

    if feature not in _GROUP_LABEL_MAPS:
        logging.warning(f"[fair-sampling] Unknown feature '{feature}'. Skipping.")
        return None

    demo = demographics or _DEFAULT_DEMOGRAPHICS
    label_fn = _GROUP_LABEL_MAPS[feature]

    try:
        df = pd.read_csv(data_path)
        if 'item_id' not in df.columns:
            logging.warning("[fair-sampling] CSV has no 'item_id' column. Skipping.")
            return None

        # Replicate Dataset_T1DM timestamp parsing + sort
        for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                break
            except Exception:
                pass
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.iloc[:int(len(df) * percent / 100)]

        num_samples = len(df)
        num_train = int(num_samples * (100 - val_split) / 100)
        df_train = df.iloc[0:num_train].reset_index(drop=True)

        patient_ids = df_train['item_id'].astype(str).values

        def pid_to_label(pid):
            info = demo.get(pid, {})
            val = info.get(feature)
            return label_fn(val) if val is not None else -1

        row_labels = np.array([pid_to_label(pid) for pid in patient_ids])

        n_windows = len(df_train) - seq_len - pred_len + 1
        if n_windows <= 0:
            logging.warning("[fair-sampling] Not enough rows for windows. Skipping.")
            return None

        window_labels = np.empty(n_windows, dtype=np.int64)
        for i in range(n_windows):
            window_rows = row_labels[i: i + seq_len]
            valid = window_rows[window_rows >= 0]
            if len(valid) == 0:
                window_labels[i] = 0
            else:
                counts = np.bincount(valid)
                window_labels[i] = int(np.argmax(counts))

        n0 = int(np.sum(window_labels == 0))
        n1 = int(np.sum(window_labels == 1))
        logging.info(
            f"[fair-sampling] Group labels built: {n_windows} windows, "
            f"feature='{feature}', label_0={n0}, label_1={n1}"
        )
        return window_labels

    except Exception as e:
        logging.warning(f"[fair-sampling] Could not build group labels: {e}. Skipping.")
        return None


def build_hypo_sampler(
    dataset,
    group_labels: np.ndarray,
    hypo_threshold: float = 70.0,
    max_oversample: float = 2.4,
):
    """Build a WeightedRandomSampler that oversamples minority-group hypo windows.

    Windows from the group with *lower* hypo prevalence get upweighted so that
    both groups see a similar expected number of hypo events per epoch.

    Args:
        dataset:         A Dataset_T1DM (or any Dataset) whose __getitem__ returns
                         (seq_x, seq_y, ...) where seq_y contains the label values.
        group_labels:    1-D int array from build_group_labels (length == len(dataset)).
        hypo_threshold:  Glucose threshold below which a window is labelled hypo.
        max_oversample:  Cap on per-window weight multiplier to avoid extreme bias.

    Returns:
        (torch.utils.data.WeightedRandomSampler, dict) — sampler + info dict for logging.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    n = len(dataset)
    # Identify hypo windows: label at prediction horizon < threshold
    # seq_y shape: (context_len + pred_len, 1)  — last pred_len rows are the forecast targets
    hypo_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            sample = dataset[i]
            seq_y = sample[1]                          # (context+pred, 1) or similar
            if hasattr(seq_y, 'numpy'):
                seq_y = seq_y.numpy()
            seq_y = np.asarray(seq_y).flatten()
            hypo_flags[i] = np.any(seq_y < hypo_threshold)
        except Exception:
            pass

    # Per-group hypo rates
    groups = np.unique(group_labels[group_labels >= 0])
    group_hypo_rates = {}
    for g in groups:
        mask = group_labels == g
        rate = hypo_flags[mask].mean() if mask.sum() > 0 else 0.0
        group_hypo_rates[g] = max(rate, 1e-6)

    max_rate = max(group_hypo_rates.values())

    # Per-window weight: upweight minority-group hypo windows
    weights = np.ones(n, dtype=float)
    for i in range(n):
        g = int(group_labels[i])
        if g < 0:
            continue
        if hypo_flags[i]:
            # Oversample this window proportionally to rate gap, capped at max_oversample
            rate_ratio = max_rate / group_hypo_rates[g]
            weights[i] = min(rate_ratio, max_oversample)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=n,
        replacement=True,
    )

    info = {
        'group_hypo_rates': {int(g): f"{r:.4f}" for g, r in group_hypo_rates.items()},
        'max_oversample': max_oversample,
        'n_hypo_windows': int(hypo_flags.sum()),
    }
    return sampler, info
