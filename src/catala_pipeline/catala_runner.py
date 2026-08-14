"""
Catala runner -- subprocess wrapper around clerk commands.

Handles three operations:
  1. typecheck   -- validate syntax and types of a .catala_en file
  2. json_schema -- get the input/output JSON schema of a scope
  3. run_scope   -- execute a scope with JSON inputs, return parsed outputs

All clerk commands require:
  - A catala.toml file in the working directory (created by setup_project_dir)
  - The opam environment activated (OPAMROOT set, eval $(opam env) called)

The caller is responsible for activating the environment before use:
    export OPAMROOT=/data/$USER/.opam
    eval $(opam env --root=/data/$USER/.opam --switch=default)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def setup_project_dir(directory: Path) -> None:
    """
    Ensure a directory is a valid Catala project (has catala.toml + stdlib).

    Creates catala.toml if missing, then runs 'clerk start' to initialise
    the standard library. Safe to call multiple times.
    """
    toml_path = directory / "catala.toml"
    if not toml_path.exists():
        toml_path.write_text('[project]\nname = "catala_generated"\n')

    build_dir = directory / "_build" / "libcatala"
    if not build_dir.exists():
        result = subprocess.run(
            ["clerk", "start"],
            cwd=directory,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"clerk start failed in {directory}:\n{result.stderr}"
            )


def validate_fences(catala_file: Path) -> tuple[bool, str]:
    """
    Check that the generated .catala_en file contains at least one ```catala fence.

    This catches the common LLM mistake of writing bare Catala code without
    fences, which causes clerk json-schema to fail even if typecheck passes.

    Returns:
        (True, "")              if fences are present
        (False, error_message)  if no fences found
    """
    content = catala_file.read_text(encoding="utf-8")
    if "```catala" not in content:
        return False, (
            "Generated file contains no ```catala fences. "
            "All Catala code must be inside ```catala ... ``` blocks. "
            "Rewrite the file so every declaration and scope block is "
            "inside a ```catala fence."
        )
    return True, ""


def typecheck(catala_file: Path) -> tuple[bool, str]:
    """
    Run 'clerk typecheck' on a .catala_en file.

    Checks for ```catala fences first, then runs the compiler.

    Returns:
        (True, "")              on success
        (False, error_message)  on failure
    """
    # Pre-check: fences must be present before calling the compiler
    ok, err = validate_fences(catala_file)
    if not ok:
        return False, err

    result = subprocess.run(
        ["clerk", "typecheck", catala_file.name],
        cwd=catala_file.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    # Clerk writes errors to stdout (not stderr)
    error = result.stdout.strip() or result.stderr.strip()
    return False, error


def json_schema(catala_file: Path, scope_name: str) -> Optional[list]:
    """
    Run 'clerk json-schema' to get the input/output schema of a scope.

    Returns a list of two dicts: [input_schema, output_schema],
    or None if the command fails.
    """
    result = subprocess.run(
        ["clerk", "json-schema", catala_file.name, f"--scope={scope_name}"],
        cwd=catala_file.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def run_scope(
    catala_file: Path,
    scope_name: str,
    inputs: dict,
) -> tuple[bool, dict, str]:
    """
    Run 'clerk run' on a scope with the given JSON inputs.

    Writes inputs to a temporary file (absolute path required by clerk).

    Returns:
        (True, outputs_dict, "")       on success
        (False, {}, error_message)     on failure
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=catala_file.parent,
    ) as tmp:
        json.dump(inputs, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "clerk", "run", catala_file.name,
                f"--scope={scope_name}",
                f"--input={tmp_path.resolve()}",
                "--output-format=json",
            ],
            cwd=catala_file.parent,
            capture_output=True,
            text=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        error = result.stdout.strip() or result.stderr.strip()
        return False, {}, error

    try:
        outputs = json.loads(result.stdout.strip())
        return True, outputs, ""
    except json.JSONDecodeError as e:
        return False, {}, f"Failed to parse clerk output: {e}\nRaw: {result.stdout}"
    