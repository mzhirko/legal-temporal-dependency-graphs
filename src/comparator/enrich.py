"""
Catala -> TDG enrichment.

(!) EXPERIMENTAL -- NOT WIRED INTO ANY PIPELINE. No entry point calls this module,
so its reported results ("22/45 enriched") are not reproducible from committed
code. Self-consistency checks here are circular (enriched dates are derived from
the same anchors they are checked against). Do not cite results from this module
until it has been validated against held-out data. See CLEANUP_MANIFEST.md.

Feeds Catala's verified computations back into the TDG as new facts WITH
dependencies, using the Catala parser for dependency chains and provenance
for source clause text.

This solves two problems:
  1. Contracts become self-checkable (enriched facts have additive deps)
  2. Cross-doc matching uses real legal clause text (not boilerplate tags)

Usage:
    from comparator.enrich import enrich_tdg

    enriched = enrich_tdg(tdg, comparison, catala_file=Path("seed10.catala_en"))
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from tdg_pipeline.tdg import (
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
    TimexSpan,
)
from tdg_pipeline.embeddings import normalise_entity


# --- Main entry point ----------------------------------------------------

def enrich_tdg(
    tdg: TemporalDependencyGraph,
    comparison: dict,
    catala_file: Optional[Path] = None,
) -> tuple[TemporalDependencyGraph, list[dict]]:
    """Enrich a TDG with Catala-computed values, dependencies, and provenance.

    Args:
        tdg: the TDG to enrich (modified in place)
        comparison: dict loaded from a comparison report JSON
        catala_file: path to .catala_en file (enables deps + provenance)

    Returns:
        (enriched_tdg, conflicts) where conflicts is a list of dicts
        describing verified disagreements between TDG extraction and
        Catala computation for the SAME concept.
    """
    fields = comparison.get("fields", [])
    scope_name = comparison.get("scope_name", "catala")
    catala_status = comparison.get("catala_status", "unknown")

    if catala_status != "success":
        return tdg, []

    # Parse catala file for dependency chains and provenance
    resolved_outputs = {}
    provenance = {}
    if catala_file and catala_file.exists():
        try:
            from catala_pipeline.catala_parser import analyse_catala_file
            from comparator.provenance import extract_provenance
            scope_decl, resolved_outputs = analyse_catala_file(catala_file)
            provenance = extract_provenance(catala_file)
        except Exception:
            pass  # fall back to no deps/provenance

    # Build lookup of existing facts by value
    existing_by_value = {}
    for fact in tdg.facts:
        val = fact.timex.date_parsed.isoformat() if fact.timex.date_parsed else fact.timex.value
        if val:
            existing_by_value[val] = fact

    next_id = _next_catala_id(tdg)
    new_facts = []  # collect new facts for dependency creation
    conflicts = []  # genuine TDG-vs-Catala disagreements

    for field in fields:
        status = field.get("status", "")
        var_name = field.get("variable_name", "")
        catala_value = field.get("catala_value")
        tdg_value = field.get("tdg_value")

        if status in ("placeholder", "tdg_only", "type_mismatch"):
            continue

        if status == "catala_only" and catala_value:
            fact = _make_catala_fact(
                fact_id=f"c_{next_id}",
                variable_name=var_name,
                value=catala_value,
                scope_name=scope_name,
                sentence=provenance.get(var_name, ""),
                resolved=resolved_outputs.get(var_name),
            )
            if fact:
                tdg.facts.append(fact)
                new_facts.append((var_name, fact))
                next_id += 1

        elif status in ("match", "value_match", "duration_match", "off_by_one"):
            if tdg_value and tdg_value in existing_by_value:
                existing_by_value[tdg_value].confidence = min(
                    existing_by_value[tdg_value].confidence + 0.1, 1.0
                )

        elif status in ("mismatch", "semantic_mismatch", "duration_mismatch"):
            if catala_value:
                # Verify alignment: is this a real conflict or a misalignment?
                # Find the TDG fact that was matched
                tdg_fact = existing_by_value.get(tdg_value) if tdg_value else None
                is_real_conflict = False

                if tdg_fact:
                    # Check if Catala variable and TDG entity refer to same concept
                    var_natural = var_name.replace("_", " ").lower()
                    entity_natural = normalise_entity(tdg_fact.entity)
                    # Token overlap between variable name and entity name
                    var_tokens = set(var_natural.split()) - {"date", "the", "of", "a"}
                    ent_tokens = set(entity_natural.split()) - {"date", "the", "of", "a"}
                    if var_tokens and ent_tokens:
                        overlap = len(var_tokens & ent_tokens) / len(var_tokens | ent_tokens)
                        is_real_conflict = overlap >= 0.2

                fact = _make_catala_fact(
                    fact_id=f"c_{next_id}",
                    variable_name=var_name,
                    value=catala_value,
                    scope_name=scope_name,
                    sentence=provenance.get(var_name, ""),
                    resolved=resolved_outputs.get(var_name),
                )
                if fact:
                    tdg.facts.append(fact)
                    # Misaligned comparisons are treated as catala_only
                    # (new fact with dep). Real conflicts get both the
                    # new fact and a record in conflicts list.
                    new_facts.append((var_name, fact))
                    if is_real_conflict and tdg_fact:
                        conflicts.append({
                            "variable": var_name,
                            "tdg_fact_id": tdg_fact.id,
                            "tdg_entity": tdg_fact.entity,
                            "tdg_value": tdg_value,
                            "catala_fact_id": fact.id,
                            "catala_value": catala_value,
                            "delta_days": field.get("delta_days"),
                        })
                    next_id += 1

    # Create dependencies for new facts using catala_parser chains
    if resolved_outputs and new_facts:
        _create_dependencies(tdg, new_facts, resolved_outputs)

    return tdg, conflicts


def enrich_tdg_from_file(
    tdg: TemporalDependencyGraph,
    comparison_path: str,
    catala_file: Optional[Path] = None,
) -> tuple[TemporalDependencyGraph, list[dict]]:
    """Load a comparison report from file and enrich the TDG."""
    with open(comparison_path) as f:
        comparison = json.load(f)
    return enrich_tdg(tdg, comparison, catala_file=catala_file)


# --- Dependency creation -------------------------------------------------

def _create_dependencies(
    tdg: TemporalDependencyGraph,
    new_facts: list[tuple[str, TemporalFact]],
    resolved_outputs: dict,
) -> None:
    """Create additive dependencies connecting enriched facts to TDG anchors.

    Matching strategy: for each enriched fact, try every dated TDG fact as
    a potential anchor. Pick the one where (enriched_date - anchor_date)
    best matches a DURATION fact in the TDG, or is a round number of
    years/months. This avoids the entity-name vocabulary gap between
    TDG extraction and Catala generation.
    """
    dated_facts = [f for f in tdg.facts
                   if f.timex.date_parsed and not f.id.startswith("c_")]
    duration_days_set = {f.timex.duration_days for f in tdg.facts
                         if f.timex.duration_days is not None}

    for var_name, new_fact in new_facts:
        resolved = resolved_outputs.get(var_name)
        if not resolved or not resolved.root_input:
            continue
        if not new_fact.timex.date_parsed:
            continue

        best_anchor = None
        best_score = -1

        for candidate in dated_facts:
            if not candidate.timex.date_parsed:
                continue
            delta = (new_fact.timex.date_parsed - candidate.timex.date_parsed).days
            if delta <= 0:
                continue  # anchor must be before enriched date

            score = _delta_plausibility(delta, duration_days_set)
            if score > best_score:
                best_score = score
                best_anchor = (candidate, delta)

        if best_anchor and best_score > 0:
            anchor_fact, delta = best_anchor
            tdg.dependencies.append(TemporalDependency(
                from_id=anchor_fact.id,
                to_id=new_fact.id,
                constraint_type="additive",
                constraint_expr=f"{resolved.root_input} -> {var_name} (catala verified)",
                delta_days=delta,
                verified=True,
                confidence=0.95 if best_score >= 2 else 0.7,
            ))


def _delta_plausibility(delta_days: int, known_durations: set[int]) -> float:
    """Score how plausible a delta is as a temporal rule.

    Higher = more likely this delta represents a real legal duration.
    """
    score = 0.0

    # Best: delta matches a known DURATION fact in the TDG
    for known in known_durations:
        if abs(delta_days - known) <= 5:
            score += 3.0
            break

    # Good: delta is a round number of years (within 5 days)
    for years in range(1, 30):
        if abs(delta_days - years * 365) <= 5:
            score += 2.0
            break

    # OK: delta is a round number of months (within 3 days)
    for months in range(1, 360):
        if abs(delta_days - months * 30) <= 3:
            score += 1.0
            break

    return score


# --- Fact creation --------------------------------------------------------

def _next_catala_id(tdg: TemporalDependencyGraph) -> int:
    max_id = 0
    for fact in tdg.facts:
        if fact.id.startswith("c_"):
            try:
                n = int(fact.id[2:])
                max_id = max(max_id, n)
            except ValueError:
                pass
    return max_id + 1


def _make_catala_fact(
    fact_id: str,
    variable_name: str,
    value: str,
    scope_name: str,
    sentence: str = "",
    resolved: Optional[object] = None,
) -> Optional[TemporalFact]:
    """Create a TDG fact from a Catala-computed value.

    Returns None if the value can't be parsed as a date.
    """
    parsed_date = None
    try:
        parsed_date = date.fromisoformat(value)
    except (ValueError, TypeError):
        pass

    if parsed_date is None:
        return None

    role = _infer_role_from_structure(resolved)
    entity = variable_name.replace("_", " ")

    return TemporalFact(
        id=fact_id,
        entity=entity,
        role=role,
        timex=TimexSpan(
            text=value,
            timex_type="DATE",
            value=value,
            start_char=0,
            end_char=0,
            date_parsed=parsed_date,
        ),
        sentence=sentence,
        confidence=1.0,
    )


def _infer_role_from_structure(
    resolved: Optional["ResolvedOutput"] = None,
) -> str:
    """Infer temporal role from Catala computation structure.

    Uses the computation direction and chain properties -- not variable names.

    - output = anchor + duration (direction=+1) -> END (computed after anchor)
    - output = anchor - duration (direction=-1) -> START (computed before anchor)
    - output = literal date, no computation -> CONTAINS (no structural info)
    - output = identity (output == input) -> CONTAINS (no structural info)
    """
    if resolved is None:
        return "CONTAINS"

    if resolved.literal_date is not None:
        return "CONTAINS"

    if resolved.is_identity:
        return "CONTAINS"

    # Has computation chain -- direction tells us the role
    if resolved.root_input is not None:
        if resolved.direction >= 0:
            return "END"    # anchor + duration -> computed after anchor
        else:
            return "START"  # anchor - duration -> computed before anchor

    return "CONTAINS"