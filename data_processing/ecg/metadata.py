"""MIT-BIH metadata parsing utilities.

This module adds an ECG-specific data layer without changing the existing BG
forecasting pipeline. It reads MIT-BIH header files, extracts record metadata,
and can export a repository-local metadata table for downstream dataset
construction and fairness analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import re
from typing import Iterable, List, Optional


DEFAULT_MITBIH_DIR = Path(__file__).resolve().parents[2] / "data" / "mit-bih-arrhythmia"

# MIT-BIH contains 48 records from 47 subjects. Records 201 and 202 come from
# the same subject. Current project policy keeps one-record-per-patient for
# fairness accounting, so record 202 is excluded by default.
DEFAULT_EXCLUDED_RECORD_IDS = {"202"}


def resolve_excluded_record_ids(include_duplicate_202: bool = False) -> set[str]:
    """Return excluded MIT-BIH record IDs under the current protocol."""
    if include_duplicate_202:
        return set()
    return set(DEFAULT_EXCLUDED_RECORD_IDS)


_HEADER_DEMOGRAPHICS_RE = re.compile(
    r"^#\s*(?P<age>\d+|\?|\-1)\s+(?P<sex>[MF\?])\b(?P<rest>.*)$"
)


@dataclass
class MitBihRecordMetadata:
    record_id: str
    sampling_rate_hz: int
    signal_length: int
    num_channels: int
    lead_1: Optional[str]
    lead_2: Optional[str]
    age: Optional[int]
    sex: str
    medications: str
    notes: str
    primary_lead_used: Optional[str]
    age_group: str
    paced_group: str
    difficulty_group: str


class MitBihMetadataParser:
    """Parse MIT-BIH header files into structured metadata rows."""

    def __init__(self, dataset_dir: Path | str = DEFAULT_MITBIH_DIR):
        self.dataset_dir = Path(dataset_dir)

    def load_record_ids(self, exclude_record_ids: Optional[Iterable[str]] = None) -> List[str]:
        records_file = self.dataset_dir / "RECORDS"
        if not records_file.exists():
            raise FileNotFoundError(f"RECORDS file not found: {records_file}")
        excluded = {
            str(record_id).strip()
            for record_id in (exclude_record_ids if exclude_record_ids is not None else DEFAULT_EXCLUDED_RECORD_IDS)
            if str(record_id).strip()
        }
        return [
            line.strip()
            for line in records_file.read_text().splitlines()
            if line.strip() and line.strip() not in excluded
        ]

    def parse_record(self, record_id: str) -> MitBihRecordMetadata:
        header_path = self.dataset_dir / f"{record_id}.hea"
        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")

        lines = header_path.read_text().splitlines()
        if not lines:
            raise ValueError(f"Header file is empty: {header_path}")

        first = lines[0].split()
        if len(first) < 4:
            raise ValueError(f"Malformed MIT-BIH header first line: {lines[0]}")

        num_channels = int(first[1])
        sampling_rate = int(float(first[2]))
        signal_length = int(first[3])

        lead_names: List[Optional[str]] = []
        for line in lines[1:1 + num_channels]:
            parts = line.split()
            lead_names.append(parts[-1] if parts else None)

        age = None
        sex = "unknown"
        medications = ""
        notes_lines: List[str] = []

        for line in lines[1 + num_channels:]:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue

            demo_match = _HEADER_DEMOGRAPHICS_RE.match(stripped)
            if demo_match:
                age_raw = demo_match.group("age")
                age = int(age_raw) if age_raw.isdigit() else None
                sex_raw = demo_match.group("sex")
                sex = {"M": "M", "F": "F"}.get(sex_raw, "unknown")
                continue

            content = stripped.lstrip("#").strip()
            if content.lower().startswith("medications:"):
                medications = content.split(":", 1)[1].strip()
            elif content:
                notes_lines.append(content)

        primary_lead = self._choose_primary_lead(lead_names)
        age_group = self._infer_age_group(age)
        notes = " | ".join(notes_lines)
        paced_group = self._infer_paced_group(notes, lead_names)
        difficulty_group = self._infer_difficulty_group(notes)

        return MitBihRecordMetadata(
            record_id=record_id,
            sampling_rate_hz=sampling_rate,
            signal_length=signal_length,
            num_channels=num_channels,
            lead_1=lead_names[0] if len(lead_names) > 0 else None,
            lead_2=lead_names[1] if len(lead_names) > 1 else None,
            age=age,
            sex=sex,
            medications=medications,
            notes=notes,
            primary_lead_used=primary_lead,
            age_group=age_group,
            paced_group=paced_group,
            difficulty_group=difficulty_group,
        )

    def parse_all(self, record_ids: Optional[Iterable[str]] = None) -> List[MitBihRecordMetadata]:
        ids = list(record_ids) if record_ids is not None else self.load_record_ids()
        return [self.parse_record(record_id) for record_id in ids]

    def export_csv(
        self,
        output_path: Path | str,
        record_ids: Optional[Iterable[str]] = None,
    ) -> Path:
        output_path = Path(output_path)
        rows = self.parse_all(record_ids=record_ids)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        return output_path

    def export_demographics_csv(
        self,
        output_path: Path | str,
        record_ids: Optional[Iterable[str]] = None,
    ) -> Path:
        output_path = Path(output_path)
        rows = self.parse_all(record_ids=record_ids)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        demographic_rows = [
            {
                "record_id": row.record_id,
                "sex": row.sex,
                "age": row.age,
                "age_group": row.age_group,
                "lead_1": row.lead_1,
                "lead_2": row.lead_2,
                "primary_lead_used": row.primary_lead_used,
                "paced_group": row.paced_group,
                "difficulty_group": row.difficulty_group,
                "medications": row.medications,
                "notes": row.notes,
            }
            for row in rows
        ]

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(demographic_rows[0].keys()))
            writer.writeheader()
            for row in demographic_rows:
                writer.writerow(row)

        return output_path

    @staticmethod
    def _choose_primary_lead(lead_names: List[Optional[str]]) -> Optional[str]:
        if "MLII" in lead_names:
            return "MLII"
        for lead in lead_names:
            if lead is not None:
                return lead
        return None

    @staticmethod
    def _infer_age_group(age: Optional[int]) -> str:
        if age is None:
            return "unknown"
        if age < 50:
            return "<50"
        if age <= 69:
            return "50-69"
        return "70+"

    @staticmethod
    def _infer_paced_group(notes: str, lead_names: List[Optional[str]]) -> str:
        notes_l = notes.lower()
        if "paced" in notes_l or any(lead == "/" for lead in lead_names):
            return "paced"
        return "non_paced"

    @staticmethod
    def _infer_difficulty_group(notes: str) -> str:
        notes_l = notes.lower()
        keywords = ["noise", "artifact", "difficult", "unreadable", "baseline shift"]
        return "noisy_or_difficult" if any(word in notes_l for word in keywords) else "clean_or_mostly_clean"


def build_metadata_csv(
    dataset_dir: Path | str = DEFAULT_MITBIH_DIR,
    output_path: Path | str = DEFAULT_MITBIH_DIR / "metadata_records.csv",
    include_duplicate_202: bool = False,
) -> Path:
    parser = MitBihMetadataParser(dataset_dir=dataset_dir)
    return parser.export_csv(
        output_path=output_path,
        record_ids=parser.load_record_ids(
            exclude_record_ids=resolve_excluded_record_ids(
                include_duplicate_202=include_duplicate_202
            )
        ),
    )


def build_demographics_csv(
    dataset_dir: Path | str = DEFAULT_MITBIH_DIR,
    output_path: Path | str = DEFAULT_MITBIH_DIR / "demographics_records.csv",
    include_duplicate_202: bool = False,
) -> Path:
    parser = MitBihMetadataParser(dataset_dir=dataset_dir)
    return parser.export_demographics_csv(
        output_path=output_path,
        record_ids=parser.load_record_ids(
            exclude_record_ids=resolve_excluded_record_ids(
                include_duplicate_202=include_duplicate_202
            )
        ),
    )
