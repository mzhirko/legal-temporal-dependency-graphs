"""
Aligner -- matches TDG facts to Catala output variables with semantic comparison.

Comparison strategy:
  1. Like-for-like (date vs date): direct comparison, off-by-one detection
  2. Semantic (TDG duration vs Catala date): check whether
       Catala_output == anchor_input + TDG_duration
     using the resolved dependency chain from catala_parser
  3. Placeholder: output depends on a placeholder input -> not comparable
  4. tdg_only / catala_only: one side missing

Field statuses:
  match             -- both are dates, values equal
  off_by_one        -- both are dates, differ by exactly 1 day
  mismatch          -- both are dates, values differ (delta_days set)
  semantic_match    -- TDG has duration, Catala has date, they are consistent
  semantic_mismatch -- TDG has duration, Catala has date, they are inconsistent
  placeholder       -- Catala output derives from a placeholder input
  tdg_only          -- TDG extracted it, Catala has no counterpart
  catala_only       -- Catala computed it, TDG has no counterpart
  type_mismatch     -- both sides have values but incompatible types
"""

from __future__ import annotations

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Optional

from catala_pipeline.catala_parser import (
    ResolvedOutput, analyse_catala_file, parse_definitions)
from comparator.provenance import extract_provenance
from comparator.report import ComparisonReport, FieldComparison
from tdg_pipeline.embeddings import EmbeddingSimilarity, normalise_entity

# Minimum pairing confidence (provenance/name/embedding score from
# _find_tdg_fact) required to report a date-vs-date *disagreement*. Below this,
# a "mismatch" is treated as a mis-pairing of two unrelated dates rather than a
# genuine conflict, and the pairing is discarded (see the guard in
# align_and_compare). Exact matches are always kept regardless of score. This
# gates on pairing confidence only -- never on the day delta, which would be
# circular. 0.35 sits above the loose 0.15 provenance floor that lets weak
# pairings through, while staying below the score of a genuinely shared clause.
_DATE_MISMATCH_CONF = 0.35


# --- Text overlap for provenance matching ---------------------------------

_STOP = {"the", "of", "a", "an", "in", "on", "to", "for", "and", "or",
         "is", "are", "was", "were", "be", "been", "shall", "will",
         "that", "this", "by", "with", "from", "at", "as", "its",
         "it", "not", "no", "any", "all", "each", "such", "may"}


def _text_overlap(text_a: str, text_b: str) -> float:
    """Compute token overlap (Jaccard) between two text passages.

    Used to match Catala source quotes against TDG fact sentences.
    Higher overlap = more likely they refer to the same clause.
    """
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.lower().split()) - _STOP
    tokens_b = set(text_b.lower().split()) - _STOP
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


def align_and_compare(
    document_id: str,
    tdg_facts: list,
    catala_outputs: dict,
    catala_status: str,
    scope_name: Optional[str],
    repair_attempts: int,
    catala_file: Optional[Path] = None,
    placeholder_fields: Optional[set[str]] = None,
    embedder: Optional[EmbeddingSimilarity] = None,
    catala_inputs: Optional[dict] = None,
) -> ComparisonReport:
    """
    Align TDG facts with Catala outputs and produce a ComparisonReport.

    Args:
        document_id:        Document identifier.
        tdg_facts:          List of TemporalFact from TDG pipeline.
        catala_outputs:     Dict of {variable_name: value} from Catala execution.
        catala_status:      Status string from CatalaResult.
        scope_name:         Name of the executed scope.
        repair_attempts:    Number of LLM repair rounds needed.
        catala_file:        Path to the .catala_en file (for dependency analysis).
        placeholder_fields: Set of input field names that were placeholders.
        catala_inputs:      Dict of {input_var: value} actually fed to the Catala
                            scope (from input_extractor). Required for semantic
                            comparison: the anchor input's date value is needed to
                            reconstruct anchor + TDG_duration. Without it, every
                            duration-vs-date pair degrades to type_mismatch.
    """
    report = ComparisonReport(
        document_id=document_id,
        catala_status=catala_status,
        scope_name=scope_name,
        repair_attempts=repair_attempts,
    )

    if catala_status != "success":
        return report

    placeholder_fields = placeholder_fields or set()

    # Extract source text provenance from .catala_en file
    provenance: dict[str, str] = {}
    if catala_file and catala_file.exists():
        provenance = extract_provenance(catala_file)

    # Parse Catala file for dependency chain analysis
    resolved_outputs: dict[str, ResolvedOutput] = {}
    # Anchor input values for semantic comparison. Only string-valued inputs
    # (i.e. dates "YYYY-MM-DD") can serve as arithmetic anchors; duration
    # inputs arrive as dicts and are not anchors. Placeholder anchors are
    # filtered separately below via placeholder_fields.
    catala_inputs_used: dict[str, str] = {
        k: v for k, v in (catala_inputs or {}).items() if isinstance(v, str)
    }
    definitions: dict = {}
    if catala_file is not None and catala_file.exists():
        _, resolved_outputs = analyse_catala_file(catala_file)
        try:
            _content = catala_file.read_text(encoding="utf-8")
            definitions = {d.output_var: d for d in parse_definitions(_content)}
        except Exception:
            definitions = {}

    matched_fact_ids: set[str] = set()

    for var_name, catala_value in catala_outputs.items():
        catala_value_str = str(catala_value) if catala_value is not None else None
        resolved = resolved_outputs.get(var_name)

        # Check if this output depends on a placeholder input
        if resolved and resolved.root_input in placeholder_fields:
            tdg_match = _find_tdg_fact(var_name, tdg_facts, provenance, catala_value_str, embedder)
            if tdg_match:
                fact_id, tdg_value_str, tdg_fact, _score = tdg_match
                matched_fact_ids.add(fact_id)
            report.fields.append(FieldComparison(
                variable_name=var_name,
                status="placeholder",
                catala_value=catala_value_str,
                tdg_value=tdg_value_str if tdg_match else None,
            ))
            continue

        # Find matching TDG fact
        tdg_match = _find_tdg_fact(var_name, tdg_facts, provenance, catala_value_str, embedder)

        if tdg_match is None:
            # Semantic bridge: a Catala DATE computed from a real (non-placeholder)
            # anchor + offset may correspond to a TDG *duration* fact for the same
            # clause (TDG says "P4Y", Catala computed the resulting date). The
            # primary matcher above is type-filtered to dates, so duration facts
            # are searched here explicitly before giving up.
            #
            # The bridge fires when EITHER the chain roots in a usable input
            # (resolved.root_input) OR the output has a usable immediate anchor
            # (definition.anchor_var) -- the latter covers subtractions and
            # computed/literal anchors (e.g. deadline = expiry - notice_period),
            # which have root_input=None and were previously dropped.
            _defn = definitions.get(var_name)
            _anchor_ok = (_defn is not None and _defn.anchor_var is not None
                          and _defn.anchor_var not in placeholder_fields)
            _root_ok = (resolved is not None and resolved.root_input
                        and resolved.root_input not in placeholder_fields)
            if _is_date_string(catala_value_str) and (_root_ok or _anchor_ok):
                dur_match = _find_tdg_duration_fact(
                    var_name, tdg_facts, provenance, matched_fact_ids
                )
                if dur_match is not None:
                    fid, tdg_dur_str, _ = dur_match
                    # Prefer the empirical offset Catala actually applied
                    # (output_value - anchor_value), which captures direction,
                    # computed anchors and multi-step chains. Fall back to the
                    # root-anchor reconstruction when the anchor value is absent.
                    applied = _applied_offset_days(
                        _defn, catala_outputs, catala_inputs_used)
                    if applied is not None:
                        status, delta = _semantic_compare_offset(tdg_dur_str, applied)
                    else:
                        status, delta = _semantic_compare(
                            tdg_duration_str=tdg_dur_str,
                            catala_date_str=catala_value_str,
                            resolved=resolved,
                            catala_inputs=catala_inputs_used,
                        )
                    if status != "type_mismatch":
                        matched_fact_ids.add(fid)
                        report.fields.append(FieldComparison(
                            variable_name=var_name,
                            status=status,
                            tdg_value=tdg_dur_str,
                            catala_value=catala_value_str,
                            delta_days=delta,
                        ))
                        continue

            # Strategy 3: value-based fallback
            value_match = find_tdg_fact_by_value(
                catala_value_str, tdg_facts, matched_fact_ids
            )
            if value_match:
                fact_id, tdg_value_str, tdg_fact = value_match
                matched_fact_ids.add(fact_id)
                report.fields.append(FieldComparison(
                    variable_name=var_name,
                    status="value_match",
                    tdg_value=tdg_value_str,
                    catala_value=catala_value_str,
                    delta_days=0,
                ))
            else:
                report.fields.append(FieldComparison(
                    variable_name=var_name,
                    status="catala_only",
                    catala_value=catala_value_str,
                ))
            continue

        fact_id, tdg_value_str, tdg_fact, match_score = tdg_match
        matched_fact_ids.add(fact_id)

        # Determine comparison type based on what TDG and Catala have
        tdg_is_date = _is_date_string(tdg_value_str)
        tdg_is_duration = _is_iso_duration(tdg_value_str)
        catala_is_date = _is_date_string(catala_value_str)

        if tdg_is_date and catala_is_date:
            # Like-for-like date comparison
            status, delta = _compare_dates(tdg_value_str, catala_value_str)

            # Confidence guard: a *disagreement* is only credible when the
            # pairing evidence is strong. The primary matcher can pair a Catala
            # date with an unrelated TDG date on weak embedding/name similarity;
            # reporting that as a "mismatch" manufactures false conflicts (e.g.
            # a 17-year delta) AND consumes the TDG fact so the semantic-duration
            # bridge never runs. On a weak-evidence mismatch we therefore DISCARD
            # the pairing: free the TDG fact, try the duration bridge, else fall
            # through to catala_only. Exact matches / off-by-one are kept
            # regardless of score (a value match is self-evidently correct). The
            # guard never turns a real agreement into a non-match; it only
            # refuses to call a low-confidence pairing a disagreement. The
            # threshold is on pairing confidence, NOT on the day delta (tuning on
            # the delta would be circular).
            if status == "mismatch" and match_score < _DATE_MISMATCH_CONF:
                matched_fact_ids.discard(fact_id)
                bridged = _try_duration_bridge(
                    var_name, catala_value_str, tdg_facts, provenance,
                    matched_fact_ids, definitions, catala_outputs,
                    catala_inputs_used, resolved, placeholder_fields, report)
                if not bridged:
                    report.fields.append(FieldComparison(
                        variable_name=var_name,
                        status="catala_only",
                        catala_value=catala_value_str,
                    ))
                continue

            report.fields.append(FieldComparison(
                variable_name=var_name,
                status=status,
                tdg_value=tdg_value_str,
                catala_value=catala_value_str,
                delta_days=delta,
            ))

        elif tdg_is_duration and (
            _is_catala_duration(catala_value_str) or _is_iso_duration(catala_value_str)
        ):
            # Duration vs duration comparison
            status, delta = _compare_durations(tdg_value_str, catala_value_str)
            report.fields.append(FieldComparison(
                variable_name=var_name,
                status=status,
                tdg_value=tdg_value_str,
                catala_value=catala_value_str,
                delta_days=delta,
            ))

        else:
            # Incompatible types or unknown format
            report.fields.append(FieldComparison(
                variable_name=var_name,
                status="type_mismatch",
                tdg_value=tdg_value_str,
                catala_value=catala_value_str,
            ))

    # TDG facts with no Catala counterpart
    for fact in tdg_facts:
        if fact.id not in matched_fact_ids:
            tdg_value_str = _fact_value_str(fact)
            if tdg_value_str:
                report.fields.append(FieldComparison(
                    variable_name=f"tdg:{fact.entity}:{fact.role}",
                    status="tdg_only",
                    tdg_value=tdg_value_str,
                ))

    return report


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def _compare_dates(
    tdg_value: str,
    catala_value: str,
) -> tuple[str, Optional[int]]:
    """Direct date comparison with off-by-one detection."""
    if tdg_value == catala_value:
        return "match", None
    try:
        d1 = date.fromisoformat(tdg_value)
        d2 = date.fromisoformat(catala_value)
        delta = abs((d2 - d1).days)
        if delta == 1:
            return "off_by_one", delta
        return "mismatch", delta
    except ValueError:
        return "mismatch", None


def _applied_offset_days(defn, catala_outputs: dict,
                         catala_inputs: dict) -> Optional[int]:
    """Signed days Catala actually applied for an output: output - anchor.

    Reads the anchor's value (an input date, or another computed output's date)
    and the output's own value from the executed Catala results, so the offset
    is whatever Catala truly computed -- direction, computed/literal anchors and
    multi-step chains all fall out of the value difference, with no need to
    reconstruct the arithmetic. Returns None when either value is missing or not
    a date (e.g. a placeholder or a duration-typed input).
    """
    if defn is None or getattr(defn, "anchor_var", None) is None:
        return None
    vals: dict[str, str] = {}
    for src in (catala_inputs or {}), (catala_outputs or {}):
        for k, v in src.items():
            if isinstance(v, str):
                vals[k] = v
    anchor_v = vals.get(defn.anchor_var)
    out_v = vals.get(defn.output_var)
    if not anchor_v or not out_v:
        return None
    try:
        a = date.fromisoformat(anchor_v)
        o = date.fromisoformat(out_v)
    except ValueError:
        return None
    return (o - a).days


def _semantic_compare_offset(
    tdg_duration_str: str,
    applied_offset_days: int,
) -> tuple[str, Optional[int]]:
    """Compare a TDG duration to the magnitude of the offset Catala applied.

    The TDG states a clause period (e.g. P6M); Catala applied some signed
    day-offset between the clause's anchor and result. If the magnitudes agree
    within a small tolerance (absorbing month/year length variation), the two
    descriptions of the same clause are consistent. Sign is intentionally
    ignored here -- "expiry - 6 months" and "start + 6 months" both confirm a
    six-month clause; whether it is added or subtracted is a separate concern.
    """
    months, days = _iso_duration_to_cal(tdg_duration_str)
    if months is None and days is None:
        return "type_mismatch", None
    m = months or 0
    d = days or 0
    lo = m * 30 + d - 3
    hi = m * 31 + d + 3
    mag = abs(applied_offset_days)
    if lo <= mag <= hi:
        return "semantic_match", applied_offset_days
    return "semantic_mismatch", applied_offset_days


def _semantic_compare(
    tdg_duration_str: str,
    catala_date_str: str,
    resolved: ResolvedOutput,
    catala_inputs: dict[str, str],
) -> tuple[str, Optional[int]]:
    """
    Semantic comparison: does Catala's computed date agree with TDG's duration?

    Reconstructs the expected date as anchor_input + tdg_duration using CALENDAR
    arithmetic (relativedelta), matching how Catala evaluates "+ N month/year".
    Using 30-day months / 365-day years here instead produces false mismatches:
    e.g. seed14 issue_date(1975-07-19) + 6 month = 1976-01-19 in Catala, but
    6*30=180 days = 1976-01-15, a spurious 4-day "disagreement".

    Caveat: only the leaf TDG duration is added to the resolved root anchor.
    Multi-step Catala chains with intermediate offsets (e.g.
    deadline = entry + 14 day, entry = notif + period) are NOT reconstructed
    faithfully and should be treated as unreliable.
    """
    months, days = _iso_duration_to_cal(tdg_duration_str)
    if months is None and days is None:
        return "type_mismatch", None

    anchor_value = catala_inputs.get(resolved.root_input) if resolved.root_input else None
    if anchor_value is None:
        return "type_mismatch", None

    try:
        anchor_date = date.fromisoformat(anchor_value)
        catala_date = date.fromisoformat(catala_date_str)
    except ValueError:
        return "type_mismatch", None

    expected = anchor_date + relativedelta(months=months or 0, days=days or 0)
    delta = abs((catala_date - expected).days)

    # Calendar arithmetic removes month/year approximation drift, so the
    # tolerance can be tight. Allow 1 day for off-by-one rounding conventions.
    if delta <= 1:
        return "semantic_match", delta
    return "semantic_mismatch", delta


# ---------------------------------------------------------------------------
# TDG fact matching
# ---------------------------------------------------------------------------

def _try_duration_bridge(
    var_name, catala_value_str, tdg_facts, provenance, matched_fact_ids,
    definitions, catala_outputs, catala_inputs_used, resolved,
    placeholder_fields, report,
) -> bool:
    """Attempt the date-vs-duration semantic bridge for one Catala date output.

    Mirrors the inline bridge used when the primary matcher finds nothing: if
    the output has a usable anchor and a TDG duration fact describes the same
    clause, compare the offset Catala applied against that duration. On a
    non-type-mismatch result, append the field, mark the duration fact matched,
    and return True. Otherwise return False (caller records catala_only).
    """
    if not _is_date_string(catala_value_str):
        return False
    _defn = definitions.get(var_name)
    _anchor_ok = (_defn is not None and _defn.anchor_var is not None
                  and _defn.anchor_var not in placeholder_fields)
    _root_ok = (resolved is not None and resolved.root_input
                and resolved.root_input not in placeholder_fields)
    if not (_root_ok or _anchor_ok):
        return False
    dur_match = _find_tdg_duration_fact(
        var_name, tdg_facts, provenance, matched_fact_ids)
    if dur_match is None:
        return False
    fid, tdg_dur_str, _ = dur_match
    applied = _applied_offset_days(_defn, catala_outputs, catala_inputs_used)
    if applied is not None:
        status, delta = _semantic_compare_offset(tdg_dur_str, applied)
    else:
        status, delta = _semantic_compare(
            tdg_duration_str=tdg_dur_str, catala_date_str=catala_value_str,
            resolved=resolved, catala_inputs=catala_inputs_used)
    if status == "type_mismatch":
        return False
    matched_fact_ids.add(fid)
    report.fields.append(FieldComparison(
        variable_name=var_name, status=status, tdg_value=tdg_dur_str,
        catala_value=catala_value_str, delta_days=delta))
    return True


def _find_tdg_fact(
    var_name: str,
    tdg_facts: list,
    provenance: dict[str, str],
    catala_value: Optional[str] = None,
    embedder: Optional[EmbeddingSimilarity] = None,
    provenance_threshold: float = 0.15,
) -> Optional[tuple[str, str, object]]:
    """
    Find the TDG fact that best matches a Catala variable.

    Returns (fact_id, value_string, fact) or None.

    Strategy:
      1. Provenance matching: compare the Catala variable's source quote
         (from the .catala_en prose section) against TDG fact sentences.
         Highest text overlap wins.
      2. Fallback: embedding/token similarity on variable name vs entity name
         (only used when no provenance quote is available).

    Type filtering is applied in both strategies: Catala date outputs
    only match TDG date facts, duration outputs only match duration facts.
    """
    catala_is_date = _is_date_string(catala_value)
    catala_is_duration = (_is_iso_duration(catala_value)
                         or _is_catala_duration(catala_value))

    # Strategy 1: provenance matching
    source_quote = provenance.get(var_name)
    if source_quote:
        best_score = 0.0
        best_match = None

        for fact in tdg_facts:
            value = _fact_value_str(fact)
            if not value:
                continue

            # Type filter
            fact_is_date = _is_date_string(value)
            fact_is_duration = _is_iso_duration(value)
            if catala_is_date and not fact_is_date:
                continue
            if catala_is_duration and not fact_is_duration:
                continue

            # Compare source quote against TDG fact sentence
            score = _text_overlap(source_quote, fact.sentence)
            if score > best_score:
                best_score = score
                best_match = (fact.id, value, fact)

        if best_match and best_score >= provenance_threshold:
            return (*best_match, best_score)

    # Strategy 2: fallback to name similarity (when no provenance available)
    var_natural = var_name.lower().replace("_", " ")
    best_score = 0.0
    best_match = None

    for fact in tdg_facts:
        value = _fact_value_str(fact)
        if not value:
            continue

        fact_is_date = _is_date_string(value)
        fact_is_duration = _is_iso_duration(value)
        if catala_is_date and not fact_is_date:
            continue
        if catala_is_duration and not fact_is_duration:
            continue

        entity_norm = normalise_entity(fact.entity)

        if embedder is not None:
            score = embedder.similarity(var_natural, entity_norm)
        else:
            stop = {"date", "start", "end", "the", "of", "a", "an"}
            var_tokens = set(var_natural.split()) - stop
            entity_tokens = set(entity_norm.split()) - stop
            if var_tokens and entity_tokens:
                overlap = len(var_tokens & entity_tokens)
                union = len(var_tokens | entity_tokens)
                score = overlap / union if union else 0.0
            else:
                score = 0.0

        if score > best_score:
            best_score = score
            best_match = (fact.id, value, fact)

    if best_match and best_score >= 0.5:
        return (*best_match, best_score)

    return None


def _find_tdg_duration_fact(
    var_name: str,
    tdg_facts: list,
    provenance: dict[str, str],
    already_matched: set[str],
    threshold: float = 0.15,
) -> Optional[tuple[str, str, object]]:
    """
    Find a TDG *duration* fact that corresponds to a Catala variable, for
    semantic (duration-vs-computed-date) comparison.

    Only considers facts whose value is an ISO duration (P…) and that are not
    already matched. Ranks by source-clause overlap (provenance) when available,
    else by variable-name/entity token overlap. This is the duration-side
    counterpart of _find_tdg_fact, which is type-filtered to dates and therefore
    cannot bridge the duration-vs-date representation gap on its own.

    Returns (fact_id, value_string, fact) or None.
    """
    source_quote = provenance.get(var_name)
    var_natural = var_name.lower().replace("_", " ")
    _stop = {"date", "start", "end", "the", "of", "a", "an", "period", "duration"}

    best_score = 0.0
    best_match = None
    for fact in tdg_facts:
        if fact.id in already_matched:
            continue
        value = _fact_value_str(fact)
        if not value or not _is_iso_duration(value):
            continue  # duration facts only

        # Score on clause identity: take the best of (a) provenance-quote
        # overlap and (b) variable-name vs entity-name token overlap. Quote
        # overlap alone is purely lexical and misses paraphrases (e.g. a quote
        # saying "denunciation" against a fact sentence saying "denounce"),
        # whereas the entity name often still carries the clause label. Using
        # the max recovers those; a loose match here is safe because the caller
        # validates the magnitude of the offset before accepting a semantic match.
        quote_score = _text_overlap(source_quote, fact.sentence) if source_quote else 0.0
        vt = set(var_natural.split()) - _stop
        et = set(normalise_entity(fact.entity).split()) - _stop
        name_score = len(vt & et) / len(vt | et) if (vt and et) else 0.0
        score = max(quote_score, name_score)

        if score > best_score:
            best_score = score
            best_match = (fact.id, value, fact)

    if best_match and best_score >= threshold:
        return best_match
    return None


def find_tdg_fact_by_value(
    catala_value: str,
    tdg_facts: list,
    already_matched: set[str],
) -> Optional[tuple[str, str, object]]:
    """
    Value-based fallback: find a TDG fact whose value matches the Catala output.

    Used after name-based matching fails. Reports as catala_only if no
    value match found, or as a special value_match if found.

    Only matches date values -- duration strings are too ambiguous.
    Only considers facts not already matched by name-based alignment.
    """
    if not _is_date_string(catala_value):
        return None
    for fact in tdg_facts:
        if fact.id in already_matched:
            continue
        value = _fact_value_str(fact)
        if value and value == catala_value and _is_date_string(value):
            return fact.id, value, fact
    return None


def _fact_value_str(fact) -> Optional[str]:
    """Extract the best string representation of a fact's value."""
    if fact.timex.date_parsed:
        return fact.timex.date_parsed.isoformat()
    if fact.timex.value:
        return fact.timex.value
    return None


# ---------------------------------------------------------------------------
# Type detection helpers
# ---------------------------------------------------------------------------

def _is_date_string(value: Optional[str]) -> bool:
    """True if value looks like a YYYY-MM-DD date."""
    if not value:
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _is_iso_duration(value: Optional[str]) -> bool:
    """True if value looks like an ISO 8601 duration (P...)."""
    return bool(value and value.startswith("P") and len(value) > 1)


def _is_catala_duration(value: Optional[str]) -> bool:
    """True if value looks like a Catala duration dict string e.g. {'days': 730.0}."""
    return bool(value and "days" in value and "{" in value)


def _catala_duration_to_days(value: str) -> Optional[int]:
    """Parse a Catala duration dict string like {'days': 730.0} to days."""
    import re
    total = 0
    for key, multiplier in [("years", 365), ("months", 30), ("days", 1)]:
        pattern = r"""['"']?""" + key + r"""['"']?\s*:\s*([\d.]+)"""
        m = re.search(pattern, value)
        if m:
            total += int(float(m.group(1)) * multiplier)
    return total if total > 0 else None


def _iso_duration_to_cal(duration_str: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse an ISO 8601 duration into (months, days) for CALENDAR arithmetic.

    Years and months collapse into a month count; weeks and days into a day
    count. This lets the caller apply relativedelta(months=…, days=…) so that
    "+ N month/year" matches Catala's calendar semantics exactly, instead of
    the lossy 30-day-month / 365-day-year approximation in
    _iso_duration_to_days (which is fine for coarse duration-vs-duration
    comparison but wrong for reconstructing a specific calendar date).

    Returns (months, days); either element may be None if that component is
    absent. (None, None) signals an unparseable / empty duration.
    """
    import re
    months = 0
    days = 0
    seen = False
    for m in re.finditer(r"(\d+(?:\.\d+)?)([YMWDH])", duration_str):
        n = float(m.group(1))
        unit = m.group(2)
        if unit == "Y":
            months += int(n * 12); seen = True
        elif unit == "M":
            months += int(n); seen = True
        elif unit == "W":
            days += int(n * 7); seen = True
        elif unit == "D":
            days += int(n); seen = True
        elif unit == "H":
            # hours can't move a calendar date meaningfully; treat as 0 days
            seen = True
    if not seen:
        return None, None
    return (months or None), (days or None)


def _iso_duration_to_days(duration_str: str) -> Optional[int]:
    """Convert ISO 8601 duration string to approximate days."""
    total = 0
    import re
    for m in re.finditer(r"(\d+(?:\.\d+)?)([YMWD])", duration_str):
        n = float(m.group(1))
        unit = m.group(2)
        if unit == "Y":
            total += int(n * 365)
        elif unit == "M":
            total += int(n * 30)
        elif unit == "W":
            total += int(n * 7)
        elif unit == "D":
            total += int(n)
    return total if total > 0 else None


def _compare_durations(tdg_value: str, catala_value: str) -> tuple[str, Optional[int]]:
    """
    Compare a TDG ISO duration against a Catala duration dict string.
    Both represent durations but in different formats.
    Returns (status, delta_days).
    """
    tdg_days = _iso_duration_to_days(tdg_value)
    if _is_catala_duration(catala_value):
        catala_days = _catala_duration_to_days(catala_value)
    elif _is_iso_duration(catala_value):
        catala_days = _iso_duration_to_days(catala_value)
    else:
        return "type_mismatch", None

    if tdg_days is None or catala_days is None:
        return "type_mismatch", None

    delta = abs(tdg_days - catala_days)
    # Allow small tolerance for month/year approximation (up to 5 days)
    if delta <= 5:
        return "duration_match", delta
    return "duration_mismatch", delta