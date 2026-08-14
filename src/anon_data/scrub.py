#!/usr/bin/env python3
"""
scrub.py — pre-publication identifier gate.

    python scrub.py scan  <tree>            # what identifiers are where
    python scrub.py plan  <tree>            # exclude-list + rekey map
    python scrub.py rekey <tree> <out>      # apply, into a fresh copy

Two categories, treated differently:

  TEXT      judgment text (.txt corpora, prompt files, quoted_sentence columns).
            Never published. Excluded, not scrubbed — free text cannot be
            reliably de-identified, and it is Licensed Material regardless.

  METADATA  results, manifests, graphs keyed by case id. Published, after the
            key is rewritten to a citation-only or pseudonymous form.

The mapping file this produces is PRIVATE. Keep it gitignored. It is the only
thing that reverses the pseudonyms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import secrets
import shutil
import sys
from collections import defaultdict
from pathlib import Path, PurePath

# --- identifier patterns ---------------------------------------------------

PATTERNS = {
    "et_case_number":   re.compile(r"\b\d{7}[./\-]\d{4}\b"),
    "eat_appeal_number": re.compile(r"\bEA-\d{4}-\d{6}(?:-[A-Z]{2,4})?\b"),
    "titled_name":      re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)[_ .]+[A-Z][A-Za-z\-']+"),
    "v_caption":        re.compile(r"\b[A-Z][A-Za-z\-']{2,}\s+(?:v\.?|vs\.?|-v-)\s+[A-Z][A-Za-z\-']{2,}"),
}

# Files whose *content* is judgment text -> exclude wholesale.
# Matched on path components, so this catches prompts/ and raw/ anywhere in the
# tree, not just under the paths that happened to exist when this was written.
TEXT_DIR_NAMES = {"caselaw", "cases", "prompts", "raw", "inputs", "txt", "xml", "pdf"}
TEXT_COLUMNS = {"quoted_sentence", "quote", "sentence", "text", "excerpt",
                "based_on_sentences", "supporting_text", "context"}

# Named columns only catch the ones you thought of. Any column whose values
# average this many characters is treated as prose and hashed too.
CSV_TEXT_MEAN_LEN = 120

SCAN_EXT = {".py", ".json", ".csv", ".md", ".txt", ".tex", ".yaml", ".yml", ".jsonl",
            ".log", ".sh", ".catala_en"}

# Containers and opaque stores. Excluded outright: their contents bypass every
# path-based rule, so an archive of an excluded directory would sail through.
OPAQUE_EXCLUDE = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar",
                  ".sqlite3", ".db", ".pkl", ".pyc", ".npz"}


def surnames_from_tree(root: Path) -> set[str]:
    """Harvest surname tokens from case-id-shaped filenames and keys."""
    out = set()
    pat = re.compile(r"\b([a-z][a-z\-']{2,})_(?:\d{4}_EAT_\d+|\d{7}_\d{4})")
    for p in root.rglob("*"):
        if p.is_file():
            for m in pat.finditer(p.name):
                out.add(m.group(1))
            if p.suffix in {".json", ".csv", ".py", ".sh", ".md", ".log",
                             ".tex", ".catala_en"} and p.stat().st_size < 5_000_000:
                try:
                    for m in pat.finditer(p.read_text(encoding="utf-8", errors="ignore")):
                        out.add(m.group(1))
                except OSError:
                    pass
    return out - {"gold", "pool", "case", "test", "eval", "full", "raw"}


def is_text_artefact(rel: str, hits: dict | None = None) -> bool:
    """TEXT by directory name, or by content signature.

    Path rules only catch directories you thought of: 'prompts_anchor' is not
    'prompts'. So directory names match on prefix, and any file carrying both a
    case caption and an appeal number is treated as judgment text regardless of
    where it lives.
    """
    parts = [q.lower() for q in Path(rel.replace("\\", "/")).parts[:-1]]
    if any(q.startswith(tuple(TEXT_DIR_NAMES)) for q in parts):
        return True
    if hits and hits.get("v_caption") and hits.get("eat_appeal_number"):
        return True
    return False


def scan_file(path: Path, surnames: set[str]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    try:
        body = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return hits
    for kind, pat in PATTERNS.items():
        for m in pat.findall(body):
            hits[kind].add(m if isinstance(m, str) else m[0])
    low = body.lower()
    for s in surnames:
        if re.search(rf"\b{re.escape(s)}\b", low):
            hits["surname"].add(s)
    if path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
                hdr = next(csv.reader(fh), [])
            for col in hdr:
                if col.strip().lower() in TEXT_COLUMNS:
                    hits["text_column"].add(col.strip())
        except OSError:
            pass
    return hits


def resolve_tree(arg: str) -> Path:
    """Fail loudly on a missing or empty tree.

    A silent zero here reads as 'clean' when it actually means 'nothing was
    looked at'. That is the most dangerous failure this tool could have.
    """
    root = Path(arg).resolve()
    if not root.exists():
        sys.exit(f"error: no such path: {root}\n"
                 f"       (cwd is {Path.cwd()}) — paths are relative to where you run this")
    if not root.is_dir():
        sys.exit(f"error: not a directory: {root}")
    n = sum(1 for p in root.rglob("*") if p.is_file())
    if n == 0:
        sys.exit(f"error: {root} contains no files")
    print(f"scanning {root}  ({n} files)")
    return root


def walk(root: Path, surnames: set[str]):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git/" in str(p) or p.suffix not in SCAN_EXT:
            continue
        rel = str(p.relative_to(root))
        hits = scan_file(p, surnames)
        if hits:
            yield rel, hits, is_text_artefact(rel, hits)


def cmd_scan(args) -> int:
    root = resolve_tree(args.tree)
    surnames = surnames_from_tree(root)
    print(f"harvested {len(surnames)} surname token(s): {', '.join(sorted(surnames))}\n")

    n_text = n_meta = 0
    by_kind: dict[str, int] = defaultdict(int)
    meta_examples = []
    for rel, hits, is_text in walk(root, surnames):
        for k, v in hits.items():
            by_kind[k] += len(v)
        if is_text:
            n_text += 1
        else:
            n_meta += 1
            if len(meta_examples) < 25:
                meta_examples.append((rel, hits))

    print(f"TEXT artefacts with identifiers  : {n_text}   -> exclude from publication")
    print(f"METADATA files with identifiers  : {n_meta}   -> rekey before publication\n")
    print("identifier occurrences by kind:")
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"   {k:20s} {v}")
    if meta_examples:
        print("\nmetadata files needing rekey (first 25):")
        for rel, hits in meta_examples:
            kinds = ",".join(sorted(hits))
            print(f"   {rel}  [{kinds}]")
    return 1 if (n_text or n_meta) else 0


# --- rekeying --------------------------------------------------------------

RE_EAT_KEY = re.compile(r"\b([a-z][a-z\-']{2,})_(\d{4}_EAT_\d+(?:_s\d+)?)\b")
RE_ET_KEY = re.compile(r"\b([a-z][a-z\-']{2,})_(\d{7})_(\d{4})\b")


def build_map(root: Path, salt: str) -> dict[str, str]:
    """old key -> new key. EAT: drop surname. ET: salted pseudonym."""
    mapping: dict[str, str] = {}
    for p in root.rglob("*"):
        blob = p.name
        if p.is_file() and p.suffix in SCAN_EXT and p.stat().st_size < 5_000_000:
            try:
                blob += "\n" + p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
        for m in RE_EAT_KEY.finditer(blob):
            mapping[m.group(0)] = m.group(2)
        for m in RE_ET_KEY.finditer(blob):
            digest = hashlib.sha256((salt + m.group(0)).encode()).hexdigest()[:8]
            mapping[m.group(0)] = f"et_{digest}"
    return mapping


def cmd_plan(args) -> int:
    root = resolve_tree(args.tree)
    salt = secrets.token_hex(16)
    mapping = build_map(root, salt)
    excludes = sorted({str(Path(rel).parent) + "/"
                       for rel, _, is_text in walk(root, surnames_from_tree(root)) if is_text})

    Path(args.map_out).write_text(json.dumps(
        {"_warning": "PRIVATE. Gitignore this file. It reverses the pseudonyms.",
         "tree": str(root), "n_keys": len(mapping),
         "salt": salt, "mapping": mapping}, indent=1), encoding="utf-8")

    print(f"rekey map: {len(mapping)} key(s) -> {args.map_out}  (KEEP PRIVATE)\n")
    for old, new in sorted(mapping.items())[:15]:
        print(f"   {old:34s} -> {new}")
    if len(mapping) > 15:
        print(f"   ... and {len(mapping) - 15} more")
    print("\nadd to .gitignore:")
    for e in excludes:
        print(f"   {e}")
    print(f"   {args.map_out}")
    return 0


def text_column_indices(rows: list[list[str]], policy_cols: dict[str, str],
                        threshold: int) -> tuple[list[int], list[str]]:
    """Indices to hash, and the names caught by the length heuristic."""
    hdr = rows[0]
    idx, heuristic = [], []
    for i, col in enumerate(hdr):
        name = col.strip().lower()
        action = policy_cols.get(name)
        if action == "keep":
            continue
        if action == "hash" or name in TEXT_COLUMNS:
            idx.append(i)
            continue
        vals = [r[i] for r in rows[1:] if i < len(r) and r[i].strip()]
        if vals and sum(map(len, vals)) / len(vals) >= threshold:
            idx.append(i)
            heuristic.append(col.strip())
    return idx, heuristic


def rewrite_csv_text_columns(body: str, policy_cols: dict[str, str] | None = None,
                             threshold: int = CSV_TEXT_MEAN_LEN):
    """Replace any text-bearing column with a SHA-256 of its value.

    Keeps the annotation (concept, type, value, flags, comments) and keeps the
    row verifiable against the source, while removing the text itself.
    """
    rows = list(csv.reader(body.splitlines()))
    if not rows:
        return body, 0, []
    hdr = rows[0]
    idx, heuristic = text_column_indices(rows, policy_cols or {}, threshold)
    if not idx:
        return body, 0, []
    for i in idx:
        hdr[i] = hdr[i].strip() + "_sha256"
    for row in rows[1:]:
        for i in idx:
            if i < len(row) and row[i]:
                row[i] = hashlib.sha256(row[i].encode("utf-8")).hexdigest()[:32]
    out = []
    w = csv.writer(_Sink(out), lineterminator="\n")
    w.writerows(rows)
    return "".join(out), len(idx), heuristic


class _Sink:
    def __init__(self, buf): self.buf = buf
    def write(self, s): self.buf.append(s)


# --- field-level policy ----------------------------------------------------
# Key rewriting only reaches keys. Free-text fields carry captions, quotes and
# case numbers that no key rule will ever touch, so they need their own rules.
#
# actions:  keep | drop | hash | citation_only
#   citation_only  '[YYYY] EAT NN, Name v Organisation' -> '[YYYY] EAT NN'

DEFAULT_POLICY = {
    "exclude_files": ["**/verified_quotes.json"],
    "json_fields": {
        "source": "citation_only",
        "case_name": "citation_only",
        "caption": "citation_only",
        "quote": "hash",
        "quoted_sentence": "hash",
        "sentence": "hash",
        "excerpt": "hash",
        "note": "drop",
        "comment": "drop",
    },
}

RE_CITATION = re.compile(r"\[\d{4}\]\s+[A-Z]{2,6}(?:\s+(?:Civ|Crim|Admin))?\s+\d+")


def apply_field(value, action: str):
    """Returns (new_value, drop_flag)."""
    if action == "keep" or not isinstance(value, str) or not value:
        return value, False
    if action == "drop":
        return None, True
    if action == "hash":
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32], False
    if action == "citation_only":
        m = RE_CITATION.search(value)
        return (m.group(0) if m else None), (m is None)
    return value, False


def apply_json_policy(obj, fields: dict[str, str], counts: dict[str, int]):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            action = fields.get(k.strip().lower())
            if action and not isinstance(v, (dict, list)):
                new, dropped = apply_field(v, action)
                counts[f"{k}:{action}"] += 1
                if not dropped:
                    out[k] = new
                continue
            out[k] = apply_json_policy(v, fields, counts)
        return out
    if isinstance(obj, list):
        return [apply_json_policy(x, fields, counts) for x in obj]
    return obj


def resolve_policy_path(path: str) -> Path:
    """Try cwd first, then next to this script — that is where it ships."""
    for cand in (Path(path), Path(__file__).parent / Path(path).name):
        if cand.is_file():
            return cand
    sys.exit(f"error: policy file not found: {path}\n"
             f"       looked in {Path.cwd()} and {Path(__file__).parent}")


def load_policy(path: str | None) -> dict:
    if not path:
        return DEFAULT_POLICY
    spec = json.loads(resolve_policy_path(path).read_text(encoding="utf-8"))
    merged = {**DEFAULT_POLICY, **spec}
    merged["json_fields"] = {**DEFAULT_POLICY["json_fields"],
                             **spec.get("json_fields", {})}
    return merged


def cmd_rekey(args) -> int:
    # Preflight: everything cheap and fallible, before the expensive tree walk.
    dst = Path(args.out)
    if dst.exists():
        if any(dst.iterdir()):
            sys.exit(f"error: {dst.resolve()} exists and is not empty — "
                     f"remove it or choose another output path")
        dst.rmdir()  # empty leftover from an aborted run; harmless to clear
    if not Path(args.map).is_file():
        sys.exit(f"error: rekey map not found: {args.map}\n"
                 f"       run 'scrub.py plan' first")
    policy = load_policy(getattr(args, "policy", None))
    spec = json.loads(Path(args.map).read_text(encoding="utf-8"))
    mapping = spec["mapping"]

    src = resolve_tree(args.tree)

    # An empty mapping compiles to an empty pattern that matches everywhere.
    # More importantly, a map built from a different tree silently under-
    # rekeys this one, which leaks rather than crashes.
    if not mapping:
        sys.exit(f"error: {args.map} contains no keys\n"
                 f"       re-run: scrub.py plan {args.tree} --map-out {args.map}")
    map_tree = spec.get("tree")
    if map_tree and Path(map_tree).resolve() != src:
        sys.exit(f"error: rekey map was built from a different tree\n"
                 f"       map:  {map_tree}\n"
                 f"       here: {src}\n"
                 f"       a stale map under-rekeys silently — re-run 'plan' on this tree")

    # longest first, so '2026_EAT_64_s111' beats '2026_EAT_64'
    field_counts: dict[str, int] = defaultdict(int)
    excluded_policy: list[str] = []
    surnames = surnames_from_tree(src)
    keys = sorted(mapping, key=len, reverse=True)
    rx = re.compile("|".join(re.escape(k) for k in keys))

    n_files = n_subs = n_hashed = 0
    opaque: set[str] = set()
    excluded_opaque: list[str] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or ".git/" in str(p):
            continue
        rel = str(p.relative_to(src))
        hits = scan_file(p, surnames) if p.suffix in SCAN_EXT else {}
        if is_text_artefact(rel, hits):
            continue  # excluded, never copied
        if p.suffix in OPAQUE_EXCLUDE:
            excluded_opaque.append(rel)
            continue
        if any(PurePath(rel.replace("\\", "/")).match(g)
               for g in policy["exclude_files"]):
            excluded_policy.append(rel)
            continue
        new_rel = rx.sub(lambda m: mapping[m.group(0)], rel)
        target = dst / new_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in SCAN_EXT:
            try:
                body = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                shutil.copy2(p, target)
                continue
            if p.suffix == ".json":
                try:
                    obj = json.loads(body)
                    obj = apply_json_policy(obj, policy["json_fields"], field_counts)
                    body = json.dumps(obj, indent=1, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            if p.suffix == ".csv":
                body, n_cols, heur = rewrite_csv_text_columns(
                    body, policy.get("csv_columns", {}),
                    policy.get("csv_text_mean_len", CSV_TEXT_MEAN_LEN))
                if n_cols:
                    n_hashed += n_cols
                    extra = f"  (by length: {', '.join(heur)})" if heur else ""
                    print(f"   hashed {n_cols} column(s) in {new_rel}{extra}")
            new_body, k = rx.subn(lambda m: mapping[m.group(0)], body)
            target.write_text(new_body, encoding="utf-8")
            n_subs += k
        else:
            shutil.copy2(p, target)
            opaque.add(p.suffix or "(no ext)")
        n_files += 1

    print(f"\nwrote {n_files} file(s) to {dst}, {n_subs} substitution(s), "
          f"{n_hashed} text column(s) hashed")
    print("text artefacts excluded entirely.")
    if field_counts:
        print("field policy applied:")
        for k, v in sorted(field_counts.items()):
            print(f"   {k}  x{v}")
    if excluded_policy:
        print(f"excluded by policy: {', '.join(excluded_policy)}")
    if excluded_opaque:
        print(f"\nexcluded {len(excluded_opaque)} archive/opaque file(s) "
              f"(contents bypass path rules):")
        for r in excluded_opaque[:10]:
            print(f"   {r}")
        if len(excluded_opaque) > 10:
            print(f"   ... and {len(excluded_opaque) - 10} more")
    if opaque:
        print(f"WARNING: copied unscanned by extension: {', '.join(sorted(opaque))}")
        print("         these were NOT rekeyed and NOT checked for identifiers.")
    print(f"now re-run: python scrub.py scan {dst}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("tree"); s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("plan"); s.add_argument("tree")
    s.add_argument("--map-out", default="rekey_map.PRIVATE.json"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("rekey"); s.add_argument("tree"); s.add_argument("out")
    s.add_argument("--map", default="rekey_map.PRIVATE.json")
    s.add_argument("--policy", help="JSON policy file; merged over the defaults")
    s.set_defaults(fn=cmd_rekey)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
    