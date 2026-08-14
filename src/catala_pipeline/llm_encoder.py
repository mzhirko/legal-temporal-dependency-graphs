"""
LLM encoder -- Call 1 of the Catala pipeline.

Takes raw legal text and produces a .catala_en file by prompting Gemma 4
with few-shot examples retrieved from the RAG index.

The LLM is shown:
  - The Catala type system (concise reference, ~30 lines)
  - 2-3 retrieved examples of legal text -> valid Catala scope
  - The target document text

It is asked to output ONLY the .catala_en file content, nothing else.
The result is saved to disk. Typecheck validation happens in repair_loop.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Catala type reference (injected into every prompt)
# ---------------------------------------------------------------------------

CATALA_TYPE_REFERENCE = """
CATALA 1.1 SYNTAX REFERENCE -- use only what is listed here.

## Types
  date       -- a calendar date
  duration   -- a length of time (years, months, days)
  boolean    -- true or false
  integer    -- whole number
  money      -- monetary amount

## Date literals
  |YYYY-MM-DD|          e.g. |2005-09-08|

## Duration expressions
  N year / N month / N day    e.g. 6 month, 30 day, 1 year

## Arithmetic
  date + duration  -> date
  date - duration  -> date
  date - date      -> duration
  duration + duration -> duration

## Comparison
  date < date, date <= date, date > date, date >= date

## Scope declaration
  declaration scope MyScopeName:
    input  var_name content type
    output var_name content type

## Scope rules
  scope MyScopeName:
    date round down
    definition var_name equals <expression>
    definition var_name under condition <boolean_expr> consequence equals <expression>

## File structure -- CRITICAL RULES
  A .catala_en file alternates prose and ```catala fences.
  ALL code (declarations AND scope rules) must be inside ```catala fences.
  Prose lines are NEVER code. Never write declarations outside fences.

  CORRECT structure:
    # Title
    Description of the scope.
    ```catala
    declaration scope MyScope:
      input x content date
      output y content date
    ```
    Description of the rule.
    ```catala
    scope MyScope:
      date round down
      definition y equals x + 1 month
    ```

  WRONG -- declaration outside fence (this will fail typecheck):
    declaration scope MyScope:
      input x content date

  Each scope must be declared EXACTLY ONCE.
  Do NOT repeat the declaration block anywhere in the file.
  Multiple scope rule blocks are fine (they merge), but the
  declaration block must appear only once.
""".strip()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal formalization assistant. "
    "You translate natural language legal text into valid Catala 1.1 programs. "
    "Output ONLY the .catala_en file content, nothing else. "
    "No explanation. No markdown wrapper around the whole output. "
    "Every single line of Catala code MUST be inside ```catala fences. "
    "Prose lines outside fences describe the law. Code lines outside fences are forbidden."
)

USER_PROMPT_TEMPLATE = """{type_reference}

## Examples of legal text -> Catala scope

{examples}

---

## Your task

Write a Catala 1.1 .catala_en file for the legal document below.

You MUST follow this exact structure -- no exceptions:

# Title

One sentence describing what this scope computes.

```catala
declaration scope MyScopeName:
  input var1 content date
  input var2 content duration
  output var3 content date
```

## Rule 1 name

One sentence from the legal text describing this rule.

```catala
scope MyScopeName:
  date round down
  definition var3 equals var1 + var2
```

RULES:
- The declaration block MUST be inside a ```catala fence.
- Every scope block MUST be inside a ```catala fence.
- NEVER write Catala code outside a fence.
- Declare each scope EXACTLY ONCE.
- Use only: date, duration, boolean, integer as types.
- Date literals: |YYYY-MM-DD|
- Duration: 1 month, 30 day, 6 month, 1 year
- NEVER invent a date literal. If a specific date does not appear explicitly
  in the legal text, declare it as an input variable instead of a literal.
  Only use |YYYY-MM-DD| for dates that are verbatim in the document.

Legal text:
{legal_text}
"""


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LLMEncoder:
    """
    Generates a .catala_en file from raw legal text using Gemma 4 via Ollama.

    Uses RAG-retrieved examples as few-shot context.
    Typecheck validation and repair are handled by repair_loop.py.
    """

    def __init__(
        self,
        model: str = "gemma4:12b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.1,
    ):
        # Ollama uses OpenAI-compatible API
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = model
        self.temperature = temperature

    def encode(
        self,
        legal_text: str,
        retrieved_examples: list[str],
        output_path: Path,
    ) -> str:
        """
        Generate a .catala_en file for the given legal text.

        Args:
            legal_text:          Raw text of the legal document.
            retrieved_examples:  List of .catala_en file contents from RAG.
            output_path:         Where to save the generated file.

        Returns:
            The generated .catala_en content as a string.
        """
        examples_block = "\n\n---\n\n".join(retrieved_examples) if retrieved_examples else "(no examples available)"

        user_message = USER_PROMPT_TEMPLATE.format(
            type_reference=CATALA_TYPE_REFERENCE,
            examples=examples_block,
            legal_text=legal_text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content.strip()
        content = _strip_outer_fences(content)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        return content

    def encode_with_error_feedback(
        self,
        legal_text: str,
        retrieved_examples: list[str],
        previous_attempt: str,
        error_message: str,
        output_path: Path,
    ) -> str:
        """
        Repair a previously generated .catala_en file given a compiler error.

        Used by repair_loop.py. The LLM receives its previous output and the
        exact compiler error, and is asked to fix only the error.

        Returns:
            The repaired .catala_en content as a string.
        """
        repair_message = (
            f"The Catala program you wrote failed typecheck with this error:\n\n"
            f"{error_message}\n\n"
            f"Here is the program you wrote:\n\n"
            f"{previous_attempt}\n\n"
            f"Fix only the error. Output the corrected .catala_en file. "
            f"Do not explain. Output ONLY the file content."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    type_reference=CATALA_TYPE_REFERENCE,
                    examples="\n\n---\n\n".join(retrieved_examples) if retrieved_examples else "(no examples)",
                    legal_text=legal_text,
                )},
                {"role": "assistant", "content": previous_attempt},
                {"role": "user", "content": repair_message},
            ],
        )

        content = response.choices[0].message.content.strip()
        content = _strip_outer_fences(content)

        output_path.write_text(content, encoding="utf-8")
        return content


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _strip_outer_fences(text: str) -> str:
    """
    Remove any outer markdown code fences the model may add despite instructions.
    Internal ```catala fences (part of the literate program) are preserved.
    """
    text = text.strip()
    # Remove outer ```catala_en or ```markdown or plain ``` wrapper
    if text.startswith("```"):
        text = re.sub(r"^```[a-z_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()