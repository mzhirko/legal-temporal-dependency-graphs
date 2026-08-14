"""v3-E: an additive edge must be stated, not computed (TODO 1.1).

The failure this closes: an extractor sees two dates, computes the gap
between them, and writes it down as though the document stated a rule
("hearing = dismissal + 53 days"). Such an edge is arithmetic laundered
into causation. Typed `ordering` it is harmless; one mistyping puts
fabricated causation into what-if cascades and derived dates.

The check is deterministic and text-grounded: an additive edge's offset
must be recoverable from the sentences of its own endpoints. If neither
endpoint sentence yields an offset phrase that computes the same shift
as the edge claims, the edge is downgraded to `ordering` -- never
deleted -- and the downgrade is reported with its reason, the same way
the recall audit reports extraction misses.

Comparison is by effect, not by string: "within 28 days of" in the
sentence supports delta_days=28, and "three months" supports "+3m",
because both are applied to a probe date and compared by result. Probe
dates are chosen so month-length aliasing cannot produce a false match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from tdg_pipeline.tdg import TemporalDependencyGraph, TemporalDependency
from tdg_pipeline.entailment import _offset_from_text

# Two probes with different month lengths: an offset of "1 month" and an
# offset of "30 days" agree on 2001-03-01 but disagree on 2001-01-31,
# so requiring agreement on both separates calendar from day arithmetic.
_PROBES = (date(2001, 1, 31), date(2001, 3, 1))


def _shift(offset, delta_days: Optional[int]):
    """The edge's effect as a pair of probe results, or None."""
    if offset is not None:
        return tuple(offset.apply(p) for p in _PROBES)
    if delta_days is not None:
        return tuple(p + timedelta(days=delta_days) for p in _PROBES)
    return None


@dataclass
class EdgeFinding:
    from_id: str
    to_id: str
    claimed: str                 # the offset as the edge states it
    status: str                  # "supported" | "downgraded" | "unverifiable"
    evidence: str = ""           # the sentence that supports it, when supported
    reason: str = ""


@dataclass
class EdgeAudit:
    document_id: str
    findings: list[EdgeFinding] = field(default_factory=list)

    @property
    def downgraded(self) -> list[EdgeFinding]:
        return [f for f in self.findings if f.status == "downgraded"]

    def summary(self) -> str:
        n = len(self.findings)
        return (f"{self.document_id}: {n} additive edge(s), "
                f"{sum(1 for f in self.findings if f.status == 'supported')} supported, "
                f"{len(self.downgraded)} downgraded, "
                f"{sum(1 for f in self.findings if f.status == 'unverifiable')} unverifiable")


def validate_additive_edges(tdg: TemporalDependencyGraph) -> EdgeAudit:
    """Audit every additive dependency against its endpoints' sentences.

    Pure function: the TDG is not modified. Apply the downgrades with
    apply_edge_audit, so the decision and the mutation stay separable --
    the replay measures with and without application.
    """
    audit = EdgeAudit(document_id=tdg.document_id)
    facts = {f.id: f for f in tdg.facts}
    for dep in tdg.dependencies:
        if dep.constraint_type != "additive":
            continue
        claimed_offset = _offset_from_text(dep.constraint_expr or "")
        claimed = _shift(claimed_offset, dep.delta_days)
        label = dep.constraint_expr or f"delta_days={dep.delta_days}"
        if claimed is None:
            audit.findings.append(EdgeFinding(
                dep.from_id, dep.to_id, label, "unverifiable",
                reason="edge carries no computable offset"))
            continue
        evidence = ""
        for fid in (dep.from_id, dep.to_id):
            sent = (facts[fid].sentence or "") if fid in facts else ""
            stated = _offset_from_text(sent)
            if stated is not None and _shift(stated, None) == claimed:
                evidence = sent.strip()
                break
        if evidence:
            audit.findings.append(EdgeFinding(
                dep.from_id, dep.to_id, label, "supported", evidence=evidence))
        else:
            audit.findings.append(EdgeFinding(
                dep.from_id, dep.to_id, label, "downgraded",
                reason="offset not stated in either endpoint sentence; "
                       "gap between observed dates is not a rule"))
    return audit


def apply_edge_audit(tdg: TemporalDependencyGraph, audit: EdgeAudit) -> int:
    """Downgrade the flagged edges in place: additive -> ordering,
    verified -> False. Nothing is deleted. Returns the count changed."""
    flagged = {(f.from_id, f.to_id) for f in audit.downgraded}
    n = 0
    for dep in tdg.dependencies:
        if dep.constraint_type == "additive" and (dep.from_id, dep.to_id) in flagged:
            dep.constraint_type = "ordering"
            dep.verified = False
            n += 1
    return n
