"""
Comparison report -- data structures for TDG vs Catala comparison.

A ComparisonReport records the outcome of comparing one document's
TDG extraction against its Catala-computed ground truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class FieldComparison:
    """
    Comparison of a single temporal variable between TDG and Catala.

    status values:
        match         -- both sides have the same value
        mismatch      -- both sides have a value but they differ
        tdg_only      -- TDG extracted it, Catala did not output it
        catala_only   -- Catala computed it, TDG did not extract it
        off_by_one    -- date values differ by exactly 1 day (calendar arithmetic)
    """
    variable_name: str
    status: str                          # match | mismatch | tdg_only | catala_only | off_by_one
    tdg_value: Optional[str] = None
    catala_value: Optional[str] = None
    delta_days: Optional[int] = None     # only set when both are dates and status is mismatch/off_by_one


@dataclass
class ComparisonReport:
    """
    Full comparison report for one document.

    Aggregates per-field comparisons and computes summary statistics.
    """
    document_id: str
    catala_status: str                   # success | typecheck_error | repair_failed | interpret_error
    scope_name: Optional[str] = None
    repair_attempts: int = 0
    fields: list[FieldComparison] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        return sum(1 for f in self.fields
                   if f.status in ("match", "off_by_one", "semantic_match", "value_match", "duration_match"))

    @property
    def mismatch_count(self) -> int:
        return sum(1 for f in self.fields
                   if f.status in ("mismatch", "semantic_mismatch", "duration_mismatch"))

    @property
    def placeholder_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "placeholder")

    @property
    def type_mismatch_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "type_mismatch")

    @property
    def tdg_only_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "tdg_only")

    @property
    def catala_only_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "catala_only")

    @property
    def match_rate(self) -> Optional[float]:
        """Fraction of aligned fields that match. None if no aligned fields."""
        aligned = self.match_count + self.mismatch_count
        if aligned == 0:
            return None
        return self.match_count / aligned

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "catala_status": self.catala_status,
            "scope_name": self.scope_name,
            "repair_attempts": self.repair_attempts,
            "summary": {
                "match": self.match_count,
                "mismatch": self.mismatch_count,
                "tdg_only": self.tdg_only_count,
                "catala_only": self.catala_only_count,
                "match_rate": self.match_rate,
            },
            "fields": [asdict(f) for f in self.fields],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)