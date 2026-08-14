"""
Catala provenance parser.

Parses a .catala_en file and extracts source text quotes associated with
each variable definition. Used for provenance-based alignment between
TDG facts and Catala outputs.

A .catala_en file alternates prose sections and ```catala code blocks.
With the updated encoder prompt, each rule section has a blockquote (>)
containing the exact source clause that the rule formalizes.

Structure:
    ## Rule name
    > Exact clause from source text.
    ```catala
    scope MyScope:
      definition my_var equals ...
    ```

This parser extracts: my_var -> "Exact clause from source text."
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def extract_provenance(catala_path: str | Path) -> dict[str, str]:
    """Parse a .catala_en file and return variable -> source quote mapping.

    Returns dict mapping variable names to the blockquoted source text
    that precedes their definition block. Variables without a preceding
    blockquote are omitted.
    """
    with open(catala_path) as f:
        content = f.read()

    return extract_provenance_from_text(content)


def extract_provenance_from_text(content: str) -> dict[str, str]:
    """Parse .catala_en content string and return variable -> source quote mapping."""
    provenance: dict[str, str] = {}

    # Split into sections by code fences
    # Each section is either prose or code
    parts = re.split(r"```catala\s*\n", content)

    for i in range(1, len(parts)):
        # parts[i] starts inside a code block, ends at closing ```
        code_and_rest = parts[i].split("```", 1)
        code_block = code_and_rest[0].strip()

        # The prose section is the end of parts[i-1]
        prose = parts[i - 1]

        # Extract blockquote lines from the prose section
        quote = _extract_blockquote(prose)

        # Extract variable definitions from the code block
        variables = _extract_definitions(code_block)

        # Map each defined variable to the source quote
        if quote:
            for var in variables:
                provenance[var] = quote

    return provenance


def _extract_blockquote(prose: str) -> Optional[str]:
    """Extract source clause text from a prose section.

    Looks for lines starting with 'Source clause:' (the format requested
    by the encoder prompt). Also handles:
    - Blockquote format (> prefix) for backward compat
    - Quoted text after "Source clause:" with or without quote marks

    Returns the extracted quote text, or None if no quote found.
    """
    lines = prose.strip().split("\n")

    # Strategy 1: "Source clause:" prefix
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.lower().startswith("source clause:"):
            quote = stripped[len("source clause:"):].strip()
            # Strip surrounding quotes if present
            if len(quote) >= 2 and quote[0] == '"' and quote[-1] == '"':
                quote = quote[1:-1]
            if quote:
                return quote

    # Strategy 2: blockquote format (> prefix) -- backward compat
    quote_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(">"):
            quote_lines.insert(0, stripped[1:].strip())
        elif quote_lines:
            break

    if quote_lines:
        return " ".join(quote_lines)

    # Strategy 3: prose fallback for files without any markers
    return _extract_prose_fallback(prose)


def _extract_prose_fallback(prose: str) -> Optional[str]:
    """For .catala_en files without blockquotes, use the last prose paragraph.

    This handles files generated before the prompt was updated to require
    blockquotes. The prose sections are typically paraphrases of the source
    text, which still provide some matching signal.
    """
    lines = prose.strip().split("\n")
    # Skip heading lines (start with #) and empty lines from the end
    paragraphs = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        elif not stripped.startswith("#"):
            current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    # Return the last non-heading paragraph
    return paragraphs[-1] if paragraphs else None


def _extract_definitions(code_block: str) -> list[str]:
    """Extract variable names from 'definition X equals' patterns in a code block."""
    variables = []
    for m in re.finditer(r"definition\s+(\w+)\s+equals", code_block):
        variables.append(m.group(1))
    return variables