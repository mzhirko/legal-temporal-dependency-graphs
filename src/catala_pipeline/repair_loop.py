"""
Repair loop -- typecheck -> LLM feedback -> retry.

Orchestrates the self-repair cycle for LLM-generated Catala programs:
  1. Typecheck the generated .catala_en file
  2. If it passes, return success
  3. If it fails, send the error back to the LLM for repair
  4. Repeat up to max_attempts times
  5. If still failing, save the file with error log and return repair_failed

This is the core research loop: the success/failure rate and number of
repair rounds needed are measurable findings.
"""

from __future__ import annotations

import json
from pathlib import Path

from catala_pipeline.catala_result import CatalaResult
from catala_pipeline.catala_runner import typecheck
from catala_pipeline.llm_encoder import LLMEncoder


def run_repair_loop(
    catala_file: Path,
    legal_text: str,
    retrieved_examples: list[str],
    encoder: LLMEncoder,
    scope_name: str,
    document_id: str,
    max_attempts: int = 3,
) -> CatalaResult:
    """
    Run the typecheck -> repair loop on a generated .catala_en file.

    The file must already exist (written by LLMEncoder.encode() before calling
    this function). This function only handles the repair cycle.

    Args:
        catala_file:         Path to the already-generated .catala_en file.
        legal_text:          Original legal text (passed to encoder for context).
        retrieved_examples:  RAG examples (passed to encoder for context).
        encoder:             LLMEncoder instance for repair calls.
        scope_name:          Name of the scope declared in the file.
        document_id:         Document identifier for the result record.
        max_attempts:        Maximum number of repair attempts (default 3).

    Returns:
        CatalaResult with status 'success', 'typecheck_error', or 'repair_failed'.
        On success, status is 'success' and repair_attempts records how many
        rounds were needed (0 = passed on first try).
    """
    current_content = catala_file.read_text(encoding="utf-8")

    for attempt in range(max_attempts):
        passed, error = typecheck(catala_file)

        if passed:
            return CatalaResult(
                document_id=document_id,
                scope_name=scope_name,
                catala_file=str(catala_file.resolve()),
                status="success",
                repair_attempts=attempt,
            )

        # Typecheck failed -- attempt repair if we have tries left
        if attempt < max_attempts - 1:
            print(f"  [repair_loop] attempt {attempt + 1}/{max_attempts - 1} failed, repairing...")
            current_content = encoder.encode_with_error_feedback(
                legal_text=legal_text,
                retrieved_examples=retrieved_examples,
                previous_attempt=current_content,
                error_message=error,
                output_path=catala_file,
            )
        else:
            # Final attempt failed -- save error log alongside the file
            error_log_path = catala_file.with_suffix(".error.txt")
            error_log_path.write_text(
                f"Failed after {max_attempts} attempts.\n\nLast error:\n{error}\n\n"
                f"Last file content:\n{current_content}",
                encoding="utf-8",
            )
            return CatalaResult(
                document_id=document_id,
                scope_name=scope_name,
                catala_file=str(catala_file.resolve()),
                status="repair_failed",
                error_message=error,
                repair_attempts=max_attempts,
            )

    # Should not reach here, but satisfy type checker
    return CatalaResult(
        document_id=document_id,
        scope_name=scope_name,
        catala_file=str(catala_file.resolve()),
        status="repair_failed",
        error_message="Unexpected end of repair loop",
        repair_attempts=max_attempts,
    )
