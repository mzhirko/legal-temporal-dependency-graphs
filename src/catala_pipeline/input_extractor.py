"""
Input extractor -- Call 2 of the Catala pipeline.

After a .catala_en file has passed typecheck, this module:
  1. Gets the scope's JSON input schema via clerk json-schema
  2. Asks Gemma 4 to extract concrete input values from the legal text
  3. Validates the extracted values against the schema
  4. Returns the inputs dict AND a set of placeholder field names

Placeholder fields are inputs the LLM could not extract from the text --
they are filled with neutral values so clerk can run, but flagged so the
comparator can mark derived outputs as non-comparable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from catala_pipeline.catala_runner import json_schema


# Neutral placeholder values by type -- used when LLM cannot extract a value.
# These are chosen to be clearly non-real (Unix epoch) so outputs derived
# from them are easily identifiable in logs, but the actual value does not
# matter because placeholder_fields tracking makes comparison skip them.
_PLACEHOLDER_DATE = "1970-01-01"
_PLACEHOLDER_DURATION = {"years": 0, "months": 0, "days": 0}
_PLACEHOLDER_BOOL = False
_PLACEHOLDER_INT = 0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal data extraction assistant. "
    "You extract specific values from legal text and return them as JSON. "
    "Output ONLY valid JSON. No explanation, no markdown, no code fences."
)

USER_PROMPT_TEMPLATE = """Extract input values from the legal text below.
You must return a JSON object with ALL of these fields:

{input_schema}

Rules for value formats:
  date:     "YYYY-MM-DD" string, e.g. "2005-09-08"
  duration: {{"years": int, "months": int, "days": int}}, e.g. {{"years": 0, "months": 6, "days": 0}}
  boolean:  true or false
  integer:  whole number

Instructions:
- Return a value for EVERY field -- never omit a field.
- If a date is explicitly stated in the text, use it exactly.
- If a date can be inferred from context (e.g. "day of signature" with a
  known document date), infer it.
- If you cannot determine a value at all, return null for that field.
  The system will handle nulls -- do not guess randomly.

Legal text:
{legal_text}
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class InputExtractor:
    """
    Extracts concrete input values for a Catala scope from legal text.

    Returns both the inputs dict (ready for clerk run) and a set of
    placeholder_fields -- inputs that could not be extracted and were
    filled with neutral placeholder values. Outputs derived from
    placeholder inputs are not meaningfully comparable to TDG values.
    """

    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = model
        self.temperature = temperature

    def extract(
        self,
        catala_file: Path,
        scope_name: str,
        legal_text: str,
    ) -> tuple[Optional[dict], set[str], Optional[str]]:
        """
        Extract input values for the given scope from the legal text.

        Args:
            catala_file:  Path to the typechecked .catala_en file.
            scope_name:   Name of the scope to extract inputs for.
            legal_text:   Raw legal text to extract values from.

        Returns:
            (inputs_dict, placeholder_fields, None)   on success
            (None, set(), error_message)               on failure

        placeholder_fields: set of field names that could not be extracted
            and were filled with neutral placeholder values.
        """
        schemas = json_schema(catala_file, scope_name)
        if schemas is None:
            return None, set(), f"Failed to get JSON schema for scope {scope_name}"

        input_schema = schemas[0]
        input_properties = (
            input_schema
            .get("definitions", {})
            .get(f"{scope_name}_in", {})
            .get("properties", {})
        )

        if not input_properties:
            # Scope has no inputs -- run directly with empty dict
            return {}, set(), None

        simplified = {
            var: _describe_type(spec)
            for var, spec in input_properties.items()
        }

        user_message = USER_PROMPT_TEMPLATE.format(
            input_schema=json.dumps(simplified, indent=2),
            legal_text=legal_text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = _strip_json_fences(raw)

        try:
            inputs = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, set(), f"LLM returned invalid JSON: {e}\nRaw: {raw}"

        # Track which fields the LLM could not extract (returned null)
        placeholder_fields: set[str] = set()

        # Fill nulls and missing fields with typed placeholders
        required = set(input_properties.keys())
        for field_name in required:
            value = inputs.get(field_name)
            type_desc = simplified.get(field_name, "")

            # Detect null at top level
            is_null = value is None

            # Detect duration with null fields: {"years": null, ...}
            if not is_null and "duration" in type_desc and isinstance(value, dict):
                if all(v is None for v in value.values()):
                    is_null = True
                else:
                    # Replace individual null fields with 0
                    inputs[field_name] = {
                        "years":  value.get("years")  if value.get("years")  is not None else 0,
                        "months": value.get("months") if value.get("months") is not None else 0,
                        "days":   value.get("days")   if value.get("days")   is not None else 0,
                    }

            if is_null:
                placeholder_fields.add(field_name)
                if "date" in type_desc:
                    inputs[field_name] = _PLACEHOLDER_DATE
                elif "duration" in type_desc:
                    inputs[field_name] = _PLACEHOLDER_DURATION
                elif "boolean" in type_desc:
                    inputs[field_name] = _PLACEHOLDER_BOOL
                else:
                    inputs[field_name] = _PLACEHOLDER_INT
            elif "date" in type_desc and isinstance(value, str):
                # GROUNDING CHECK: an extracted date must be traceable to the
                # source text. LLMs sometimes confabulate an anchor (typically
                # near their training present) instead of returning null --
                # e.g. seed21: last_report_date invented as ~2023 in a 2006
                # document, producing a fluent but ungrounded computation the
                # placeholder mechanism cannot catch. Ungrounded dates keep
                # their value (so clerk can still run) but are flagged as
                # placeholders, which marks derived outputs non-comparable.
                if not _date_grounded(value, legal_text):
                    placeholder_fields.add(field_name)

        return inputs, placeholder_fields, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_grounded(iso_value: str, legal_text: str) -> bool:
    """
    Return True if an ISO date extracted by the LLM is plausibly grounded in
    the source text. Checks common legal renderings of the full date
    ('27 November 1992', 'November 27, 1992', ISO, dotted/slashed numeric),
    falling back to a year-presence check -- weak, but it catches the
    high-damage case of a confabulated anchor whose year never occurs in the
    document at all (e.g. a 2023 date injected into a 2006 agreement).
    Inferred dates (e.g. 'day of signature' resolved against a stated date)
    pass via the year check, so legitimate inference is not penalised.
    """
    from datetime import date as _date

    try:
        d = _date.fromisoformat(iso_value)
    except (ValueError, TypeError):
        return False

    text = legal_text.lower()
    month = d.strftime("%B").lower()
    variants = [
        iso_value,
        f"{d.day} {month} {d.year}",
        f"{d.day:02d} {month} {d.year}",
        f"{month} {d.day}, {d.year}",
        f"{d.day}.{d.month}.{d.year}",
        f"{d.day:02d}.{d.month:02d}.{d.year}",
        f"{d.day}/{d.month}/{d.year}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
    ]
    if any(v in text for v in variants):
        return True
    return str(d.year) in text


def _describe_type(schema_spec: dict) -> str:
    """Convert a JSON schema type spec into a human-readable description."""
    ref = schema_spec.get("$ref", "")
    if "date" in ref:
        return 'date string "YYYY-MM-DD"'
    if "duration" in ref:
        return 'duration object {"years": int, "months": int, "days": int}'
    if "money" in ref:
        return 'money object {"cents": int}'
    type_ = schema_spec.get("type", "unknown")
    if type_ == "boolean":
        return "boolean (true or false)"
    if type_ == "integer":
        return "integer"
    return type_


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences if the model added them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()