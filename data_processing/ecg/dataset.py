"""Beat-centered MIT-BIH dataset scaffold for Time-LLM classification.

This is an ECG-specific dataset module added alongside the BG forecasting data
modules so the original system remains untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_processing.ecg.label_map import MitBihAamiMapper
from data_processing.ecg.metadata import (
    DEFAULT_MITBIH_DIR,
    MitBihMetadataParser,
    resolve_excluded_record_ids,
)


try:
    import wfdb  # type: ignore
except ImportError:  # pragma: no cover - dependency may be installed later
    wfdb = None


@dataclass
class EcgBeatSample:
    record_id: str
    beat_sample_index: int
    raw_symbol: str
    aami_class: str
    original_class_id: int
    class_id: int
    signal_window: np.ndarray
    sex: str
    age_group: str
    paced_group: str
    difficulty_group: str
    split: str
    rr_prev_seconds: float
    rr_next_seconds: float


class MitBihBeatDataset(Dataset):
    """Beat-centered MIT-BIH dataset.

    Current scope:
    - single lead
    - beat-centered windows
    - AAMI 5-class labels
    - record-level split metadata

    This scaffold is intentionally separate from the existing Dataset_T1DM so the
    BG forecasting workflow stays unchanged.
    """

    def __init__(
        self,
        dataset_dir: Path | str = DEFAULT_MITBIH_DIR,
        split: str = "train",
        window_size: int = 256,
        primary_lead_only: bool = True,
        metadata_csv: Optional[Path | str] = None,
        beat_index_csv: Optional[Path | str] = None,
        normalize: bool = True,
        split_assignments: Optional[Dict[str, str]] = None,
        include_duplicate_202: bool = False,
        label_mode: str = "aami5",
        sampling_rate_hz: float = 360.0,
        rr_clip_seconds: float = 3.0,
    ):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of train/val/test/all")
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.window_size = window_size
        self.primary_lead_only = primary_lead_only
        self.normalize = normalize
        self.mapper = MitBihAamiMapper()
        self.metadata_parser = MitBihMetadataParser(dataset_dir=self.dataset_dir)
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else self.dataset_dir / "metadata_records.csv"
        self.beat_index_csv = Path(beat_index_csv) if beat_index_csv is not None else self.dataset_dir / "beat_index.csv"
        self.split_assignments = split_assignments or {}
        self.include_duplicate_202 = include_duplicate_202
        if label_mode not in {"aami5", "binary_ectopy", "binary_non_n"}:
            raise ValueError(
                "label_mode must be one of aami5/binary_ectopy/binary_non_n"
            )
        self.label_mode = label_mode
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.rr_clip_seconds = float(rr_clip_seconds)

        self.metadata = self._load_metadata()
        self.samples: List[EcgBeatSample] = []

        if self.beat_index_csv.exists():
            self.samples = self._load_from_index(self.beat_index_csv)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = sample.signal_window.astype(np.float32).reshape(-1, 1)
        y = np.int64(sample.class_id)
        meta = {
            "record_id": sample.record_id,
            "beat_sample_index": sample.beat_sample_index,
            "raw_symbol": sample.raw_symbol,
            "aami_class": sample.aami_class,
            "original_class_id": sample.original_class_id,
            "label_mode": self.label_mode,
            "sex": sample.sex,
            "age_group": sample.age_group,
            "paced_group": sample.paced_group,
            "difficulty_group": sample.difficulty_group,
            "split": sample.split,
            "rr_prev_seconds": sample.rr_prev_seconds,
            "rr_next_seconds": sample.rr_next_seconds,
        }
        return x, y, meta

    def _load_metadata(self) -> pd.DataFrame:
        if self.metadata_csv.exists():
            return pd.read_csv(self.metadata_csv)
        rows = [
            vars(row)
            for row in self.metadata_parser.parse_all(
                record_ids=self.metadata_parser.load_record_ids(
                    exclude_record_ids=resolve_excluded_record_ids(
                        include_duplicate_202=self.include_duplicate_202
                    )
                )
            )
        ]
        return pd.DataFrame(rows)

    def _load_from_index(self, index_path: Path) -> List[EcgBeatSample]:
        df = pd.read_csv(index_path)
        df["record_id"] = df["record_id"].astype(str)
        df = df.sort_values(["record_id", "beat_sample_index"]).reset_index(drop=True)
        grouped_indices = df.groupby("record_id")["beat_sample_index"]
        rr_prev = grouped_indices.diff() / self.sampling_rate_hz
        rr_next = -grouped_indices.diff(-1) / self.sampling_rate_hz
        record_medians = rr_prev.where(rr_prev > 0).groupby(df["record_id"]).transform("median")
        global_median = float(rr_prev[rr_prev > 0].median())
        if not np.isfinite(global_median):
            global_median = 1.0
        df["rr_prev_seconds"] = rr_prev.fillna(record_medians).fillna(global_median)
        df["rr_next_seconds"] = rr_next.fillna(record_medians).fillna(global_median)
        df["rr_prev_seconds"] = df["rr_prev_seconds"].clip(0.0, self.rr_clip_seconds)
        df["rr_next_seconds"] = df["rr_next_seconds"].clip(0.0, self.rr_clip_seconds)
        samples: List[EcgBeatSample] = []
        for _, row in df.iterrows():
            row_split = row.get("split", "all")
            if self.split != "all" and row_split != self.split:
                continue
            original_class_id = int(row["class_id"])
            if self.label_mode == "binary_ectopy" and original_class_id == 4:
                continue
            if self.label_mode == "aami5":
                class_id = original_class_id
            elif self.label_mode == "binary_ectopy":
                class_id = int(original_class_id in {1, 2, 3})
            else:
                class_id = int(original_class_id != 0)
            signal = np.fromstring(str(row["signal_window"]), sep=" ")
            samples.append(
                EcgBeatSample(
                    record_id=str(row["record_id"]),
                    beat_sample_index=int(row["beat_sample_index"]),
                    raw_symbol=str(row["raw_symbol"]),
                    aami_class=str(row["aami_class"]),
                    original_class_id=original_class_id,
                    class_id=class_id,
                    signal_window=signal,
                    sex=str(row.get("sex", "unknown")),
                    age_group=str(row.get("age_group", "unknown")),
                    paced_group=str(row.get("paced_group", "non_paced")),
                    difficulty_group=str(row.get("difficulty_group", "unknown")),
                    split=str(row_split),
                    rr_prev_seconds=float(row["rr_prev_seconds"]),
                    rr_next_seconds=float(row["rr_next_seconds"]),
                )
            )
        return samples

    def build_index(self, output_path: Optional[Path | str] = None) -> Path:
        if wfdb is None:
            raise ImportError("wfdb is required to build MIT-BIH beat windows. Install with `pip install wfdb`.")

        output_path = Path(output_path) if output_path is not None else self.beat_index_csv
        records = self.metadata_parser.load_record_ids(
            exclude_record_ids=resolve_excluded_record_ids(
                include_duplicate_202=self.include_duplicate_202
            )
        )
        rows: List[Dict[str, object]] = []
        half_left = self.window_size // 2
        half_right = self.window_size - half_left

        metadata_by_record = {str(row["record_id"]): row for _, row in self.metadata.iterrows()}

        for record_id in records:
            signal, lead_name = self._load_primary_signal(record_id)
            signal = self._normalize_signal(signal) if self.normalize else signal
            ann = wfdb.rdann(str(self.dataset_dir / record_id), "atr")
            record_meta = metadata_by_record.get(record_id, {})
            assigned_split = self.split_assignments.get(record_id, "all")

            for beat_idx, symbol in zip(ann.sample, ann.symbol):
                mapped = self.mapper.map_symbol(symbol)
                if mapped is None:
                    continue
                window = self._extract_window(signal, beat_idx, half_left, half_right)
                rows.append(
                    {
                        "record_id": record_id,
                        "beat_sample_index": int(beat_idx),
                        "raw_symbol": symbol,
                        "aami_class": mapped.aami_class,
                        "class_id": mapped.class_id,
                        "primary_lead_used": lead_name,
                        "sex": record_meta.get("sex", "unknown"),
                        "age_group": record_meta.get("age_group", "unknown"),
                        "paced_group": record_meta.get("paced_group", "non_paced"),
                        "difficulty_group": record_meta.get("difficulty_group", "unknown"),
                        "split": assigned_split,
                        "signal_window": " ".join(map(str, window.astype(np.float32))),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        self.samples = self._load_from_index(output_path)
        return output_path

    def _load_primary_signal(self, record_id: str) -> Tuple[np.ndarray, str]:
        if wfdb is None:
            raise ImportError("wfdb is required to load MIT-BIH signals.")
        record = wfdb.rdrecord(str(self.dataset_dir / record_id))
        sig_name = list(record.sig_name)
        lead_idx = sig_name.index("MLII") if self.primary_lead_only and "MLII" in sig_name else 0
        signal = np.asarray(record.p_signal[:, lead_idx], dtype=np.float32)
        return signal, sig_name[lead_idx]

    @staticmethod
    def _normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad > eps:
            return (signal - median) / mad
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        return (signal - mean) / max(std, eps)

    @staticmethod
    def _extract_window(signal: np.ndarray, center: int, left: int, right: int) -> np.ndarray:
        start = center - left
        end = center + right
        if start >= 0 and end <= len(signal):
            return signal[start:end]
        pad_left = max(0, -start)
        pad_right = max(0, end - len(signal))
        clipped = signal[max(0, start):min(len(signal), end)]
        if clipped.size == 0:
            clipped = np.zeros((0,), dtype=np.float32)
        return np.pad(clipped, (pad_left, pad_right), mode="reflect" if clipped.size > 1 else "constant")[: left + right]


def build_record_level_split_assignments(
    record_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, str]:
    """Create a simple deterministic record-level split map.

    This keeps the ECG path separate from the BG pipeline and provides a
    leakage-safe default split for MIT-BIH classification experiments.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    ids = list(dict.fromkeys(str(record_id) for record_id in record_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0

    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    if not test_ids and ids:
        moved = ids[-1]
        train_ids.discard(moved)
        test_ids.add(moved)

    split_map: Dict[str, str] = {}
    for record_id in ids:
        if record_id in train_ids:
            split_map[record_id] = "train"
        elif record_id in val_ids:
            split_map[record_id] = "val"
        else:
            split_map[record_id] = "test"

    return split_map


def build_stratified_record_level_split_assignments(
    index_rows: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    search_iterations: int = 100_000,
) -> Dict[str, str]:
    """Create a record-safe split balanced across sex and AAMI classes.

    Random record splitting can leave entire sex/class cells absent from
    validation or test data. This deterministic search preserves record-level
    isolation while balancing record sex, per-class record coverage, overall
    class counts, and sex-by-class counts. A sex/class cell is required in every split when
    at least three records provide that cell globally.
    """
    required_columns = {"record_id", "class_id", "sex"}
    missing_columns = required_columns.difference(index_rows.columns)
    if missing_columns:
        raise ValueError(f"Missing split-stratification columns: {sorted(missing_columns)}")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("val_ratio must be non-negative and train_ratio + val_ratio must be < 1")

    rows = index_rows.loc[:, ["record_id", "class_id", "sex"]].copy()
    rows["record_id"] = rows["record_id"].astype(str)
    rows["class_id"] = rows["class_id"].astype(int)
    rows["sex"] = rows["sex"].astype(str)
    records = sorted(rows["record_id"].unique().tolist())
    if len(records) < 3:
        return build_record_level_split_assignments(records, seed, train_ratio, val_ratio)

    n_total = len(records)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    split_sizes = [n_train, n_val, n_total - n_train - n_val]
    split_ratios = np.asarray(split_sizes, dtype=np.float64) / n_total

    record_sex = rows.groupby("record_id")["sex"].first().to_dict()
    class_counts = rows.groupby(["record_id", "class_id"]).size().to_dict()
    sexes = sorted(set(record_sex.values()))
    num_classes = max(5, int(rows["class_id"].max()) + 1)

    feature_rows = []
    for record_id in records:
        sex_features = [float(record_sex[record_id] == sex) for sex in sexes]
        presence_features = [float(class_counts.get((record_id, class_id), 0) > 0) for class_id in range(num_classes)]
        count_features = [float(class_counts.get((record_id, class_id), 0)) for class_id in range(num_classes)]
        group_count_features = [
            float(class_counts.get((record_id, class_id), 0))
            if record_sex[record_id] == sex else 0.0
            for sex in sexes
            for class_id in range(num_classes)
        ]
        feature_rows.append(sex_features + presence_features + count_features + group_count_features)
    features = np.asarray(feature_rows, dtype=np.float64)
    features /= np.maximum(features.sum(axis=0, keepdims=True), 1.0)

    required_cells = []
    for sex in sexes:
        for class_id in range(num_classes):
            providers = np.asarray(
                [
                    record_sex[record_id] == sex and class_counts.get((record_id, class_id), 0) > 0
                    for record_id in records
                ],
                dtype=bool,
            )
            if int(providers.sum()) >= 3:
                required_cells.append(providers)

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_groups = None
    first_end = split_sizes[0]
    second_end = split_sizes[0] + split_sizes[1]
    for _ in range(search_iterations):
        order = rng.permutation(n_total)
        groups = [order[:first_end], order[first_end:second_end], order[second_end:]]
        if any(not all(bool(providers[group].any()) for group in groups) for providers in required_cells):
            continue
        score = 0.0
        for split_index, group in enumerate(groups):
            actual = features[group].sum(axis=0)
            score += float(np.mean(np.square(actual - split_ratios[split_index])))
        if score < best_score:
            best_score = score
            best_groups = [group.copy() for group in groups]

    if best_groups is None:
        raise RuntimeError(
            "Could not find a record-level split with complete feasible sex/class coverage; "
            "increase search_iterations or inspect record metadata."
        )

    split_map: Dict[str, str] = {}
    for split_name, group in zip(("train", "val", "test"), best_groups):
        for record_index in group:
            split_map[records[int(record_index)]] = split_name
    return split_map
