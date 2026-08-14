"""
Catala file parser -- extracts dependency chains from generated .catala_en files.

Parses the simple subset of Catala that the LLM generates:
  - declaration scope blocks (inputs vs outputs)
  - definition rules: output = expr
  - expr forms: input + duration, input - duration, literal_date, other_output + duration

Resolves chains: if output A = B + 1 month, and B = C + 2 months,
then A's root anchor is C with accumulated duration of 3 months.

This is intentionally simple -- it handles the patterns the LLM actually produces,
not the full Catala language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScopeDeclaration:
    """Input and output variables declared in a scope."""
    scope_name: str
    inputs: set[str] = field(default_factory=set)
    outputs: set[str] = field(default_factory=set)


@dataclass
class Definition:
    """A single definition rule: output_var = anchor_var + duration."""
    output_var: str
    anchor_var: Optional[str]       # None if the value is a literal
    duration_days: Optional[int]    # None when duration is a variable (var + var)
    literal_date: Optional[str]     # set if output = |YYYY-MM-DD|
    duration_var: Optional[str] = None  # set when duration is a variable name
    direction: int = 1              # +1 = anchor + dur, -1 = anchor - dur


@dataclass
class ResolvedOutput:
    """
    Fully resolved output variable: traced back to a root input.

    root_input:    the input variable this output ultimately depends on,
                   or None if it's a literal date
    total_days:    accumulated duration in days from root_input to this output,
                   or None if duration involves variable inputs (can't compute statically)
    literal_date:  set if this output is a fixed date literal
    is_identity:   True if output == root_input with zero duration
    duration_inputs: set of duration variable names involved in the chain
    direction:     +1 if output = anchor + duration (forward),
                   -1 if output = anchor - duration (backward)
    """
    output_var: str
    root_input: Optional[str]
    total_days: Optional[int]
    literal_date: Optional[str]
    is_identity: bool
    duration_inputs: set[str] = field(default_factory=set)
    direction: int = 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches: definition var_name equals ...
_DEF_RE = re.compile(r"^\s*definition\s+(\w+)\s+equals\s+(.+)$", re.MULTILINE)

# Matches: var_name + N unit  OR  var_name - N unit
_ANCHOR_PLUS_DUR = re.compile(
    r"^(\w+)\s*([+-])\s*(\d+)\s*(year|month|day|week)s?$"
)

# Matches: var_name + var_name  OR  var_name - var_name (duration is a variable)
_ANCHOR_PLUS_VAR = re.compile(
    r"^(\w+)\s*([+-])\s*(\w+)$"
)

# Matches: |YYYY-MM-DD| + N unit  OR  |YYYY-MM-DD| - N unit
_LITERAL_PLUS_DUR = re.compile(
    r"^\|(\d{4}-\d{2}-\d{2})\|\s*([+-])\s*(\d+)\s*(year|month|day|week)s?$"
)

# Matches: (expr) + N unit  OR  (expr) - N unit  (multi-step arithmetic)
_PAREN_PLUS_DUR = re.compile(
    r"^\((.+)\)\s*([+-])\s*(\d+)\s*(year|month|day|week)s?$"
)

# Matches: |YYYY-MM-DD|
_DATE_LITERAL = re.compile(r"^\|(\d{4}-\d{2}-\d{2})\|$")

# Matches: declaration scope Name:
_SCOPE_DECL_RE = re.compile(r"^\s*declaration\s+scope\s+(\w+)\s*:", re.MULTILINE)

# Matches input/output lines inside declaration blocks
_INPUT_RE = re.compile(r"^\s*input\s+(\w+)\s+content", re.MULTILINE)
_OUTPUT_RE = re.compile(r"^\s*output\s+(\w+)\s+content", re.MULTILINE)


def _duration_to_days(n: int, unit: str) -> int:
    """Convert a duration expression to approximate days."""
    unit = unit.lower().rstrip("s")
    if unit == "year":
        return n * 365
    if unit == "month":
        return n * 30
    if unit == "week":
        return n * 7
    if unit == "day":
        return n
    return 0


def _extract_code_blocks(catala_content: str) -> str:
    """Extract all code inside ```catala fences, concatenated."""
    blocks = re.findall(r"```catala\s*(.*?)```", catala_content, re.DOTALL)
    return "\n".join(blocks)


def parse_scope_declaration(catala_content: str) -> Optional[ScopeDeclaration]:
    """
    Parse the first scope declaration from a .catala_en file.

    Returns a ScopeDeclaration with input and output variable names,
    or None if no declaration found.
    """
    code = _extract_code_blocks(catala_content)

    scope_match = _SCOPE_DECL_RE.search(code)
    if not scope_match:
        return None

    scope_name = scope_match.group(1)
    decl = ScopeDeclaration(scope_name=scope_name)

    # Find the declaration block -- text between 'declaration scope Name:'
    # and the next blank line or next 'scope' keyword
    decl_start = scope_match.end()
    # Find end of declaration block (next declaration or scope rule)
    next_block = re.search(r"\n\s*(?:declaration\s+scope|scope\s+\w+\s*:)", code[decl_start:])
    decl_end = decl_start + next_block.start() if next_block else len(code)
    decl_block = code[decl_start:decl_end]

    for m in _INPUT_RE.finditer(decl_block):
        decl.inputs.add(m.group(1))
    for m in _OUTPUT_RE.finditer(decl_block):
        decl.outputs.add(m.group(1))

    return decl


def parse_definitions(catala_content: str) -> list[Definition]:
    """
    Parse all definition rules from a .catala_en file.

    Returns a list of Definition objects covering the patterns:
      output = input + N unit           (date + literal duration)
      output = input + var              (date + variable duration)
      output = input                    (identity)
      output = |YYYY-MM-DD|             (literal date)
      output = |YYYY-MM-DD| + N unit    (literal date ± duration)
      output = (expr) + N unit          (parenthesised multi-step)
    """
    code = _extract_code_blocks(catala_content)
    definitions = []

    for m in _DEF_RE.finditer(code):
        var_name = m.group(1)
        expr = m.group(2).strip()

        # Strip 'under condition ... consequence' from conditional rules
        consequence_match = re.search(r"\bconsequence\s+equals\s+(.+)$", expr)
        if consequence_match:
            expr = consequence_match.group(1).strip()

        # Try literal date: |YYYY-MM-DD|
        lit_match = _DATE_LITERAL.match(expr)
        if lit_match:
            definitions.append(Definition(
                output_var=var_name,
                anchor_var=None,
                duration_days=0,
                literal_date=lit_match.group(1),
            ))
            continue

        # Try literal date ± duration: |YYYY-MM-DD| + 6 month
        lit_dur_match = _LITERAL_PLUS_DUR.match(expr)
        if lit_dur_match:
            lit_date = lit_dur_match.group(1)
            sign = 1 if lit_dur_match.group(2) == "+" else -1
            n = int(lit_dur_match.group(3))
            unit = lit_dur_match.group(4)
            days = sign * _duration_to_days(n, unit)
            # Compute the resulting date
            try:
                base = date.fromisoformat(lit_date)
                result = base + timedelta(days=days)
                definitions.append(Definition(
                    output_var=var_name,
                    anchor_var=None,
                    duration_days=0,
                    literal_date=result.isoformat(),
                ))
            except (ValueError, OverflowError):
                # Bad date -- skip
                pass
            continue

        # Try multi-step: (inner_expr) ± N unit
        paren_match = _PAREN_PLUS_DUR.match(expr)
        if paren_match:
            inner = paren_match.group(1).strip()
            sign = 1 if paren_match.group(2) == "+" else -1
            n = int(paren_match.group(3))
            unit = paren_match.group(4)
            outer_days = sign * _duration_to_days(n, unit)

            # Parse inner expression as var + N unit
            inner_dur = _ANCHOR_PLUS_DUR.match(inner)
            if inner_dur:
                anchor = inner_dur.group(1)
                inner_sign = 1 if inner_dur.group(2) == "+" else -1
                inner_n = int(inner_dur.group(3))
                inner_unit = inner_dur.group(4)
                inner_days = inner_sign * _duration_to_days(inner_n, inner_unit)
                definitions.append(Definition(
                    output_var=var_name,
                    anchor_var=anchor,
                    duration_days=inner_days + outer_days,
                    literal_date=None,
                ))
                continue

        # Try anchor + literal duration: var + 180 day
        dur_match = _ANCHOR_PLUS_DUR.match(expr)
        if dur_match:
            anchor = dur_match.group(1)
            sign = 1 if dur_match.group(2) == "+" else -1
            n = int(dur_match.group(3))
            unit = dur_match.group(4)
            days = sign * _duration_to_days(n, unit)
            definitions.append(Definition(
                output_var=var_name,
                anchor_var=anchor,
                duration_days=days,
                literal_date=None,
            ))
            continue

        # Try anchor + variable duration: var + var
        var_match = _ANCHOR_PLUS_VAR.match(expr)
        if var_match:
            anchor = var_match.group(1)
            sign = 1 if var_match.group(2) == "+" else -1
            dur_var = var_match.group(3)
            definitions.append(Definition(
                output_var=var_name,
                anchor_var=anchor,
                duration_days=None,  # unknown -- duration is a variable
                literal_date=None,
                duration_var=dur_var,
                direction=sign,
            ))
            continue

        # Try identity: output = input (single word, no arithmetic)
        if re.match(r"^\w+$", expr):
            definitions.append(Definition(
                output_var=var_name,
                anchor_var=expr,
                duration_days=0,
                literal_date=None,
            ))
            continue

        # Unrecognised expression -- skip (conservative)

    return definitions


def resolve_outputs(
    scope_decl: ScopeDeclaration,
    definitions: list[Definition],
) -> dict[str, ResolvedOutput]:
    """
    Resolve each output variable to its root input and total duration.

    Chains are followed: if A = B + 30 days and B = C + 30 days,
    then A resolves to root=C, total_days=60.

    Cycles are broken after max_depth steps (indicates LLM error).
    """
    def_map: dict[str, Definition] = {d.output_var: d for d in definitions}
    resolved: dict[str, ResolvedOutput] = {}

    for output_var in scope_decl.outputs:
        root, days, literal, is_identity, dur_inputs, direction = _resolve_chain(
            output_var, def_map, scope_decl.inputs, depth=0, max_depth=10
        )
        resolved[output_var] = ResolvedOutput(
            output_var=output_var,
            root_input=root,
            total_days=days,
            literal_date=literal,
            is_identity=(days == 0 and root is not None and literal is None),
            duration_inputs=dur_inputs,
            direction=direction,
        )

    return resolved


def _resolve_chain(
    var: str,
    def_map: dict[str, Definition],
    inputs: set[str],
    depth: int,
    max_depth: int,
) -> tuple[Optional[str], Optional[int], Optional[str], bool, set[str], int]:
    """
    Recursively resolve a variable to
    (root_input, total_days, literal_date, is_identity, duration_inputs, direction).

    total_days is None when any step involves a variable duration.
    direction is +1 (forward/additive) or -1 (backward/subtractive),
    accumulated as a product across chain steps.
    """
    if depth > max_depth:
        return None, 0, None, False, set(), 1

    # Base case: var is a declared input
    if var in inputs:
        return var, 0, None, True, set(), 1

    defn = def_map.get(var)
    if defn is None:
        return None, 0, None, False, set(), 1

    # Literal date -- no input anchor
    if defn.literal_date is not None:
        return None, 0, defn.literal_date, False, set(), 1

    # Has an anchor -- recurse
    if defn.anchor_var is not None:
        root, acc_days, literal, _, dur_inputs, inner_dir = _resolve_chain(
            defn.anchor_var, def_map, inputs, depth + 1, max_depth
        )

        # Track duration variables
        if defn.duration_var:
            dur_inputs = dur_inputs | {defn.duration_var}

        # Accumulate days: None if any step is unknown
        if defn.duration_days is None or acc_days is None:
            total = None
        else:
            total = acc_days + defn.duration_days

        # Direction: product of this step's direction and inner direction
        direction = defn.direction * inner_dir

        return root, total, literal, False, dur_inputs, direction

    return None, 0, None, False, set(), 1


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def analyse_catala_file(catala_file: Path) -> tuple[
    Optional[ScopeDeclaration],
    dict[str, ResolvedOutput],
]:
    """
    Parse a .catala_en file and return its scope declaration and resolved outputs.

    Returns (None, {}) if the file cannot be parsed.
    """
    content = catala_file.read_text(encoding="utf-8")
    scope_decl = parse_scope_declaration(content)
    if scope_decl is None:
        return None, {}
    definitions = parse_definitions(content)
    resolved = resolve_outputs(scope_decl, definitions)
    return scope_decl, resolved