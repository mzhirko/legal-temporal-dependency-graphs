#!/usr/bin/env python3
"""Neutrality gate for the v3 refactors. Run from code/src. Must print
ALL NEUTRALITY CHECKS PASS before any v3 replay is scored; after it,
rerun the gold upper bound and byte-compare against the archived run.
"""
import sys
sys.path.insert(0, ".")

# 1. counting phrases: data file must equal the previous literals exactly
from tdg_pipeline.entailment import _INCLUSIVE_CONN, _EXCLUSIVE_CONN
assert _INCLUSIVE_CONN == ("beginning with", "starting with",
                           "commencing with", "beginning on", "starting on")
assert _EXCLUSIVE_CONN == ("from the date on which", "from the date",
                           "after the date", "following the date",
                           "from", "after", "following")
print("ok: counting phrases identical to pre-v3 literals")

# 2. matter/parties: absent fields must not change serialization
from tdg_pipeline.tdg import TemporalDependencyGraph
d = TemporalDependencyGraph(document_id="d", document_type="x",
                            source_text="s").to_dict()
assert "matter" not in d and "parties" not in d
print("ok: serialization unchanged when matter/parties unset")

# 3. composed linking must be OFF by default
from tdg_pipeline.cross_doc import CrossDocLinker
assert CrossDocLinker().composed is False
print("ok: composed linking default OFF (v2 path is the default)")

print("ALL NEUTRALITY CHECKS PASS")
