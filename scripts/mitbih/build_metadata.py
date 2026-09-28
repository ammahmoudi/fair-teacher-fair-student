"""Build MIT-BIH metadata CSV.

Usage:
    python scripts/mitbih/build_metadata.py
    python scripts/mitbih/build_metadata.py --dataset-dir data/mit-bih-arrhythmia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.ecg.metadata import DEFAULT_MITBIH_DIR, build_metadata_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-BIH metadata CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_MITBIH_DIR,
        help="Path to MIT-BIH dataset directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset-dir>/metadata_records.csv",
    )
    parser.add_argument(
        "--include-duplicate-202",
        action="store_true",
        help="Include duplicate record 202 instead of using the default curated 47-record set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = build_metadata_csv(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path or Path(args.dataset_dir) / "metadata_records.csv",
        include_duplicate_202=args.include_duplicate_202,
    )
    print(f"Saved metadata CSV to {output}")
