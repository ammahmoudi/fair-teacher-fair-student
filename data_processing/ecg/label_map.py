"""MIT-BIH to AAMI 5-class label mapping.

This module keeps ECG classification labeling separate from the existing BG
forecasting code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set


AAMI_CLASS_TO_ID: Dict[str, int] = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}


RAW_TO_AAMI: Dict[str, str] = {
    # N class
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S class
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V class
    "V": "V",
    "E": "V",
    # F class
    "F": "F",
    # Q class
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


EXCLUDED_SYMBOLS: Set[str] = {
    "[",
    "]",
    "!",
    "|",
    "~",
    "+",
    "x",
    "(",
    ")",
    "p",
    "t",
    "u",
    "`",
    "'",
    "^",
    "=",
    '"',
    "@",
}


@dataclass(frozen=True)
class AamiLabel:
    raw_symbol: str
    aami_class: str
    class_id: int


class MitBihAamiMapper:
    """Map raw MIT-BIH beat symbols into AAMI 5-class labels."""

    def __init__(self, raw_to_aami: Optional[Dict[str, str]] = None):
        self.raw_to_aami = raw_to_aami or RAW_TO_AAMI

    def map_symbol(self, symbol: str) -> Optional[AamiLabel]:
        if symbol in EXCLUDED_SYMBOLS:
            return None
        mapped = self.raw_to_aami.get(symbol)
        if mapped is None:
            return None
        return AamiLabel(raw_symbol=symbol, aami_class=mapped, class_id=AAMI_CLASS_TO_ID[mapped])

    def is_supported(self, symbol: str) -> bool:
        return self.map_symbol(symbol) is not None

    def class_names_in_id_order(self):
        return [label for label, _ in sorted(AAMI_CLASS_TO_ID.items(), key=lambda x: x[1])]
