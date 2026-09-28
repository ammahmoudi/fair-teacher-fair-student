"""Build MIT-BIH beat-centered index CSV.

Usage:
    python scripts/mitbih/prepare_beat_dataset.py
    python scripts/mitbih/prepare_beat_dataset.py --window-size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.ecg.metadata import (
    DEFAULT_MITBIH_DIR,
    MitBihMetadataParser,
    resolve_excluded_record_ids,
)
from data_processing.ecg.dataset import (
    MitBihBeatDataset,
    build_record_level_split_assignments,
    build_stratified_record_level_split_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-BIH beat-centered index CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_MITBIH_DIR,
        help="Path to MIT-BIH dataset directory",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Beat-centered window size in samples",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset-dir>/beat_index.csv",
    )
    parser.add_argument(
        "--include-duplicate-202",
        action="store_true",
        help="Include duplicate record 202 instead of using the default curated 47-record set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for record-level split assignment")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Record-level training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Record-level validation split ratio")
    parser.add_argument(
        "--split-strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="Record-level split strategy (default: stratified by sex and AAMI class coverage)",
    )
    parser.add_argument(
        "--resplit-existing",
        action="store_true",
        help="Reuse existing ECG windows and only update their record-level split assignments",
    )
    return parser.parse_args()


def rewrite_split_column(index_path: Path, split_assignments: dict[str, str]) -> None:
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with index_path.open(newline="") as source, temporary_path.open("w", newline="") as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "record_id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"Beat index is missing record_id/split columns: {index_path}")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["split"] = split_assignments[str(row["record_id"])]
            writer.writerow(row)
    os.replace(temporary_path, index_path)

def validate_existing_index_window_size(index_path: Path, expected_size: int) -> None:
    """Reject a cached beat index built for a different waveform context."""
    rows = pd.read_csv(index_path, usecols=["signal_window"], nrows=1)
    if rows.empty:
        raise ValueError(f"Beat index has no rows: {index_path}")
    actual_size = len(str(rows.iloc[0]["signal_window"]).split())
    if actual_size != expected_size:
        raise ValueError(
            f"Cached beat index {index_path} has window size {actual_size}, "
            f"but --window-size requested {expected_size}. Rebuild it or use another output path."
        )


def write_split_manifest(
    index_path: Path,
    split_assignments: dict[str, str],
    strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Path:
    records_by_split = {
        split: sorted(record for record, assigned in split_assignments.items() if assigned == split)
        for split in ("train", "val", "test")
    }
    manifest = {
        "strategy": strategy,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "records_by_split": records_by_split,
    }
    manifest_path = index_path.with_name("split_manifest.json")
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path or Path(args.dataset_dir) / "beat_index.csv"
    metadata_parser = MitBihMetadataParser(dataset_dir=args.dataset_dir)
    record_ids = metadata_parser.load_record_ids(
        exclude_record_ids=resolve_excluded_record_ids(
            include_duplicate_202=args.include_duplicate_202
        )
    )
    if not (args.resplit_existing and output_path.exists()):
        initial_assignments = (
            build_record_level_split_assignments(
                record_ids,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            if args.split_strategy == "random"
            else {}
        )
        dataset = MitBihBeatDataset(
            dataset_dir=args.dataset_dir,
            split="all",
            window_size=args.window_size,
            include_duplicate_202=args.include_duplicate_202,
            split_assignments=initial_assignments,
        )
        dataset.build_index(output_path=output_path)
    else:
        validate_existing_index_window_size(output_path, args.window_size)

    split_rows = pd.read_csv(output_path, usecols=["record_id", "class_id", "sex"])
    split_assignments = (
        build_stratified_record_level_split_assignments(
            split_rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        if args.split_strategy == "stratified"
        else build_record_level_split_assignments(
            split_rows["record_id"].astype(str).unique().tolist(),
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    )
    rewrite_split_column(output_path, split_assignments)
    manifest_path = write_split_manifest(
        output_path,
        split_assignments,
        args.split_strategy,
        args.seed,
        args.train_ratio,
        args.val_ratio,
    )
    split_counts = {
        split: sum(assigned == split for assigned in split_assignments.values())
        for split in ("train", "val", "test")
    }
    print(f"Saved beat index CSV to {output_path}")
    print(f"Record split counts: {split_counts}")
    print(f"Split manifest: {manifest_path}")
