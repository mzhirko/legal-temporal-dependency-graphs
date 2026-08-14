#!/usr/bin/env python3
"""Collect thesis metrics from archived counterfactual results.

Run from the root of the experiments repo:

    python3 collect_thesis_metrics.py [SEARCH_DIR ...]

With no arguments it searches the current directory tree for candidate
results files (any *.json / *.jsonl whose name or parent mentions
counterfactual / cf / sweep / results). It prints what it found and why it
kept or skipped each file, so nothing is silently included.

Outputs (written next to this script):
    thesis_metrics.json   -- everything, machine-readable
    thesis_metrics.md     -- human preview of the same numbers

If your field names differ, edit FIELD_MAP below -- one place only.
Every metric is recomputed from raw records; nothing is trusted from
pre-aggregated summaries.
"""

import json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Field mapping: edit here if your keys differ. Each entry lists accepted
#    aliases, first match wins. A record is one (item x system) prediction.
# ---------------------------------------------------------------------------
FIELD_MAP = {
    "case":        ["case", "case_id", "case_name", "gold_case", "docket"],
    "k":           ["k", "offset", "shift", "shift_days", "delta_k"],
    "model":       ["model", "model_name", "system", "extractor"],
    "condition":   ["condition", "config", "mode", "input_condition", "run"],
    "pred":        ["pred", "prediction", "verdict", "predicted_verdict",
                    "answer", "model_verdict"],
    "truth":       ["truth", "gold", "label", "gold_verdict", "true_verdict"],
    "abstained":   ["abstained", "abstain", "abstention", "no_answer"],
    "abstain_cause": ["abstain_cause", "cause", "abstention_cause", "reason"],
    "anchor_correct": ["anchor_correct", "anchor_match", "anchor_ok",
                       "extracted_anchor_correct"],
    "boundary_k":  ["boundary_k", "b", "true_boundary", "boundary_offset"],
}

TIMELY = {"timely", "in_time", "in time", "true", "1", "yes", "on_time"}
LATE   = {"late", "out_of_time", "out of time", "false", "0", "no"}


def get(rec: dict, field: str, default=None):
    for alias in FIELD_MAP[field]:
        if alias in rec:
            return rec[alias]
        # tolerate one level of nesting, e.g. rec["gold"]["verdict"]
        for k, v in rec.items():
            if isinstance(v, dict) and alias in v:
                return v[alias]
    return default


def norm_verdict(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "timely" if v else "late"
    s = str(v).strip().lower()
    if s in TIMELY:
        return "timely"
    if s in LATE:
        return "late"
    return None  # unparseable => treated as no answer, counted separately


# ---------------------------------------------------------------------------
# 2. Discovery
# ---------------------------------------------------------------------------
NAME_HINT = re.compile(r"counterfactual|(^|[_\-/])cf([_\-/]|$)|sweep|results",
                       re.IGNORECASE)


def discover(roots):
    files = []
    for root in roots:
        for p in Path(root).rglob("*.json*"):
            if p.suffix not in (".json", ".jsonl"):
                continue
            if NAME_HINT.search(str(p)):
                files.append(p)
    return sorted(set(files))


def path_label(path: Path) -> str:
    """Derive run label from directory layout when records lack model/condition.
    external_bench/out/<run>/results.json      -> <run>
    out/v3_replay/<run>/rescored_<v>.json      -> <run>:<v>
    """
    parts = path.parts
    if "v3_replay" in parts:
        i = parts.index("v3_replay")
        run = parts[i + 1] if i + 1 < len(parts) else "replay"
        variant = path.stem.replace("rescored_", "")
        return f"replay/{run}:{variant}"
    return path.parent.name if path.parent.name != "." else path.stem


def load_records(path: Path):
    try:
        if path.suffix == ".jsonl":
            recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        else:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                recs = data
            elif isinstance(data, dict):
                # common shapes: {"items": [...]}, {"results": [...]},
                # or {case: [...]} -- flatten one level of lists of dicts
                recs = []
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        for r in v:
                            r.setdefault("_parent_key", k)
                            recs.append(r)
                if not recs:
                    return None, "dict with no list-of-dicts inside"
            else:
                return None, "not a list or dict"
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable ({e})"
    # keep only files that look like per-item predictions
    probe = recs[0] if recs else {}
    if get(probe, "k") is None or get(probe, "case") is None:
        return None, "records lack case/k fields (edit FIELD_MAP if wrong)"
    label = path_label(path)
    for r in recs:
        r.setdefault("_run", label)
    return recs, None


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def flip_stats(seq):
    """seq: list of (k, pred) sorted by k, preds may contain None."""
    answered = [(k, p) for k, p in seq if p is not None]
    flips = []
    for (k0, p0), (k1, p1) in zip(answered, answered[1:]):
        if p0 != p1:
            flips.append(k1)  # flip happens entering k1
    return {
        "n_answered": len(answered),
        "n_flips": len(flips),
        "flip_positions": flips,
        "single_flip": len(flips) == 1,
        "monotonicity_violations": max(0, len(flips) - 1),
    }


def summarise(records):
    # group: (model, condition) -> case -> list[(k, rec)]
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        m = get(r, "model")
        c = get(r, "condition")
        key = (str(m), str(c)) if (m and c) else (r.get("_run", "?"), "")
        groups[key][str(get(r, "case", "?"))].append((int(get(r, "k")), r))

    out = {}
    for (model, cond), cases in sorted(groups.items()):
        n_total = n_answered = n_correct = 0
        n_anchor_known = n_anchor_ok = n_anchor_ok_correct = 0
        n_anchor_bad_correct = n_anchor_bad = 0
        causes = Counter()
        per_case = {}
        for case, items in sorted(cases.items()):
            items.sort(key=lambda t: t[0])
            seq = []
            boundary = None
            # infer boundary from gold: first k whose truth is "late";
            # also verify the gold sweep itself is monotone (harness self-check)
            gold_seq = [(k, norm_verdict(get(r, "truth"))) for k, r in items]
            late_ks = [k for k, t in gold_seq if t == "late"]
            inferred_boundary = min(late_ks) if late_ks else None
            gold_flips = sum(
                1 for (_, a), (_, b) in zip(gold_seq, gold_seq[1:]) if a != b)
            for k, r in items:
                n_total += 1
                pred = norm_verdict(get(r, "pred"))
                truth = norm_verdict(get(r, "truth"))
                abst = bool(get(r, "abstained", False)) or pred is None
                if boundary is None:
                    boundary = get(r, "boundary_k")
                if abst:
                    causes[str(get(r, "abstain_cause", "unspecified"))] += 1
                    seq.append((k, None))
                    continue
                n_answered += 1
                ok = (pred == truth)
                n_correct += int(ok)
                seq.append((k, pred))
                a = get(r, "anchor_correct")
                if a is not None:
                    n_anchor_known += 1
                    if bool(a):
                        n_anchor_ok += 1
                        n_anchor_ok_correct += int(ok)
                    else:
                        n_anchor_bad += 1
                        n_anchor_bad_correct += int(ok)
            fs = flip_stats(seq)
            if boundary is None:
                boundary = inferred_boundary
                fs["boundary_source"] = "inferred_from_gold"
            fs["boundary_k"] = boundary
            fs["gold_monotone_single_flip"] = (gold_flips == 1)
            if fs["single_flip"] and boundary is not None:
                fs["flip_minus_boundary"] = fs["flip_positions"][0] - int(boundary)
            per_case[case] = fs

        gname = f"{model} | {cond}" if cond else model
        out[gname] = {
            "items": n_total,
            "coverage": round(n_answered / n_total, 4) if n_total else None,
            "accuracy_over_all": round(n_correct / n_total, 4) if n_total else None,
            "accuracy_over_answered": round(n_correct / n_answered, 4) if n_answered else None,
            "anchor_split": {
                "rows_with_anchor_flag": n_anchor_known,
                "anchor_correct_rows": n_anchor_ok,
                "accuracy_on_anchor_correct": round(n_anchor_ok_correct / n_anchor_ok, 4) if n_anchor_ok else None,
                "accuracy_on_anchor_wrong": round(n_anchor_bad_correct / n_anchor_bad, 4) if n_anchor_bad else None,
                "preregistered_expectation": 0.923,
            },
            "abstention_causes": dict(causes),
            "cases_single_flip": sum(1 for c in per_case.values() if c["single_flip"]),
            "n_cases": len(per_case),
            "per_case": per_case,
        }
    return out


def to_markdown(summary):
    lines = ["# Counterfactual metrics (auto-collected)\n"]
    for group, s in summary.items():
        lines.append(f"## {group}")
        lines.append(f"- items {s['items']}, coverage {s['coverage']}, "
                     f"acc/all {s['accuracy_over_all']}, "
                     f"acc/answered {s['accuracy_over_answered']}, "
                     f"single-flip cases {s['cases_single_flip']}/{s['n_cases']}")
        a = s["anchor_split"]
        if a["rows_with_anchor_flag"]:
            lines.append(f"- anchor-correct rows: {a['anchor_correct_rows']}"
                         f" -> acc {a['accuracy_on_anchor_correct']}"
                         f" (pre-registered 0.923); anchor-wrong acc "
                         f"{a['accuracy_on_anchor_wrong']}")
        if s["abstention_causes"]:
            lines.append(f"- abstentions: {s['abstention_causes']}")
        lines.append("")
        lines.append("| case | flips | violations | single | flip-boundary gap |")
        lines.append("|---|---|---|---|---|")
        for case, c in s["per_case"].items():
            gap = c.get("flip_minus_boundary", "")
            lines.append(f"| {case} | {c['n_flips']} | "
                         f"{c['monotonicity_violations']} | "
                         f"{'yes' if c['single_flip'] else 'no'} | {gap} |")
        lines.append("")
    return "\n".join(lines)


def main():
    roots = sys.argv[1:] or ["."]
    here = Path.cwd()
    found = discover(roots)
    if not found:
        print("No candidate *.json(l) files found under", roots)
        print("Pass the directory that holds the archived results explicitly.")
        sys.exit(1)
    all_records, kept = [], []
    for f in found:
        recs, why = load_records(f)
        if recs is None:
            print(f"skip  {f}  ({why})")
        else:
            print(f"keep  {f}  ({len(recs)} records)")
            kept.append(str(f))
            all_records.extend(recs)
    if not all_records:
        print("\nFiles were found but none parsed as per-item predictions.")
        print("Open one and adjust FIELD_MAP at the top of this script.")
        sys.exit(2)
    schema = {}
    for f in kept:
        for r in all_records:
            if r.get("_run") == path_label(Path(f)) or True:
                pass
        # first record per source file
    # simpler: reload first record of each kept file for its key list
    for f in kept:
        try:
            d = json.loads(Path(f).read_text()) if f.endswith(".json") else None
            first = d[0] if isinstance(d, list) and d else (
                next((v[0] for v in d.values()
                      if isinstance(v, list) and v and isinstance(v[0], dict)), None)
                if isinstance(d, dict) else None)
            if first:
                keys = {k: (sorted(v.keys()) if isinstance(v, dict) else type(v).__name__)
                        for k, v in first.items()}
                schema[f] = keys
        except Exception:  # noqa: BLE001
            pass
    summary = summarise(all_records)
    (here / "thesis_metrics.json").write_text(
        json.dumps({"sources": kept, "record_schemas": schema,
                    "groups": summary}, indent=2))
    (here / "thesis_metrics.md").write_text(to_markdown(summary))
    print(f"\nWrote thesis_metrics.json and thesis_metrics.md "
          f"({len(all_records)} records, {len(summary)} model|condition groups)")


if __name__ == "__main__":
    main()
    