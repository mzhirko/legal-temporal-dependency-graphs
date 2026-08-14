#!/usr/bin/env python3
"""
Apply two fixes to tdg_pipeline/timex_extractor.py.

  1. strict mode  - HeidelTimeExtractor raises instead of silently
                    returning regex output. This is what let
                    results_heidel_raw/ be named for a backend that
                    never ran.
  2. span offsets - use the character offsets py-heideltime returns
                    instead of original_text.find(raw), which maps every
                    repeated expression onto its first occurrence.

Idempotent: running twice is a no-op. Fails loudly if the source does
not look like what it expects, rather than corrupting the file.

Usage:
    python apply_patches.py path/to/src/tdg_pipeline/timex_extractor.py
    python apply_patches.py --revert path/to/.../timex_extractor.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- patch 1

OLD_INIT = """        self.document_type = document_type
        self.language = language
        self.heideltime_path = heideltime_path
        self._fallback = RegexExtractor()
"""

NEW_INIT = """        self.document_type = document_type
        self.language = language
        self.heideltime_path = heideltime_path
        self.strict = strict
        self._fallback = RegexExtractor()
        self.backend_used: Optional[str] = None

        if self.strict:
            import importlib.util
            if importlib.util.find_spec("py_heideltime") is None:
                raise RuntimeError(
                    "py_heideltime is not installed and strict=True.\\n"
                    "  pip install py-heideltime   (needs a JRE 11+)\\n"
                    "Construct with strict=False to accept the regex "
                    "fallback. Output produced by the fallback must not "
                    "be reported as HeidelTime."
                )
"""

OLD_SIG = """        document_type: str = "narrative",
        language: str = "english",
        heideltime_path: Optional[str] = None,
    ):"""

NEW_SIG = """        document_type: str = "narrative",
        language: str = "english",
        heideltime_path: Optional[str] = None,
        strict: bool = True,
    ):"""

OLD_EXCEPT = """        except Exception as e:
            print(f"[HeidelTimeExtractor] Failed: {e}. Falling back to regex.")
            return self._fallback.extract(text, reference_date)"""

NEW_EXCEPT = """        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"HeidelTime extraction failed and strict=True: {e}"
                ) from e
            import warnings
            warnings.warn(
                f"HeidelTime unavailable ({e}); using RegexExtractor. "
                f"This output is NOT HeidelTime.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.backend_used = "regex-fallback"
            return self._fallback.extract(text, reference_date)"""

OLD_RETURN = """            return self._parse_result(result, text)"""
NEW_RETURN = """            self.backend_used = "heideltime"
            return self._parse_result(result, text)"""

# ---------------------------------------------------------------- patch 2

OLD_CALL = """                raw = item.get("text", "")
                ttype = item.get("type", "DATE")
                value = item.get("value", "")
                spans.append(self._make_span(raw, ttype, value, original_text, seen))"""

NEW_CALL = """                raw = item.get("text", "")
                ttype = item.get("type", "DATE")
                value = item.get("value", "")
                spans.append(self._make_span(
                    raw, ttype, value, original_text, seen,
                    span=item.get("span"),
                ))"""

OLD_MAKE_SIG = """        original_text: str,
        seen: set[tuple[int, int]],
    ) -> Optional[TimexSpan]:
        if not raw:
            return None

        # Find position in original text
        idx = original_text.find(raw)
        start = idx if idx >= 0 else 0
        end = start + len(raw)"""

NEW_MAKE_SIG = """        original_text: str,
        seen: set[tuple[int, int]],
        span: Optional[list] = None,
    ) -> Optional[TimexSpan]:
        if not raw:
            return None

        # Prefer the offsets HeidelTime reports. original_text.find()
        # maps every repeat of an expression onto its first occurrence,
        # which then loses spans to the `seen` de-duplication below and
        # hands the role classifier the wrong sentence context.
        start = end = None
        if span and len(span) == 2:
            try:
                start, end = int(span[0]), int(span[1])
                if original_text[start:end].strip() != raw.strip():
                    start = end = None          # offsets disagree, fall back
            except (TypeError, ValueError, IndexError):
                start = end = None
        if start is None:
            idx = original_text.find(raw)
            start = idx if idx >= 0 else 0
            end = start + len(raw)"""

PATCHES = [
    ("signature: strict kwarg", OLD_SIG, NEW_SIG),
    ("__init__: strict guard", OLD_INIT, NEW_INIT),
    ("extract: fail loudly", OLD_EXCEPT, NEW_EXCEPT),
    ("extract: stamp backend", OLD_RETURN, NEW_RETURN),
    ("_parse_result: pass span", OLD_CALL, NEW_CALL),
    ("_make_span: use offsets", OLD_MAKE_SIG, NEW_MAKE_SIG),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path: Path = args.path
    backup = path.with_suffix(path.suffix + ".prepatch")

    if args.revert:
        if not backup.exists():
            print(f"no backup at {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"reverted {path} from {backup}")
        return 0

    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    src = path.read_text()

    if all(new in src for _, _, new in PATCHES):
        print("already patched, nothing to do")
        return 0

    problems = []
    for name, old, new in PATCHES:
        if new in src:
            continue
        n = src.count(old)
        if n != 1:
            problems.append(f"  {name}: expected 1 match, found {n}")
    if problems:
        print("SOURCE DOES NOT MATCH -- not touching the file:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nYour timex_extractor.py has diverged from the version "
              "this patch was written against. Apply the changes by hand.",
              file=sys.stderr)
        return 2

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup -> {backup}")

    for name, old, new in PATCHES:
        if new not in src:
            src = src.replace(old, new, 1)
            print(f"applied: {name}")

    path.write_text(src)

    import py_compile
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(backup, path)
        print(f"SYNTAX ERROR, reverted: {e}", file=sys.stderr)
        return 3

    print(f"\npatched {path} (compiles clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
