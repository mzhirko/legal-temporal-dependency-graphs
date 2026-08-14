"""
Catala execution result -- data structures.

Represents the output of running a Catala scope, whether successful
or failed at typecheck / interpretation stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class CatalaResult:
    """
    Output of running a single Catala scope on a document.

    status values:
        success          -- scope ran and produced outputs
        typecheck_error  -- compiler rejected the .catala_en file (fixable by hand)
        interpret_error  -- scope compiled but failed at runtime (e.g. missing rule)
        repair_failed    -- typecheck failed after max repair attempts
    """
    document_id: str
    scope_name: str
    catala_file: str                        # absolute path to the .catala_en file
    status: str                             # success | typecheck_error | interpret_error | repair_failed
    outputs: dict = field(default_factory=dict)       # variable_name -> value (on success)
    error_message: Optional[str] = None    # compiler/interpreter error text (on failure)
    repair_attempts: int = 0               # how many LLM repair rounds were needed

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
