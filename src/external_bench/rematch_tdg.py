#!/usr/bin/env python3
"""Re-derive TDG predictions OFFLINE from dumped graphs -- no LLM, no endpoint.

Extraction (the 4-hour part) writes one graph per item to <run>/graphs/ when you
run run_external.py --system tdg. This script reads those graphs and re-runs only
the cheap downstream steps: matching the queried event pair to graph nodes, then
ordering them by date or graph path. So matcher and threshold experiments cost
seconds, not another full extraction.

The whole point: the diagnosis said ~46% of items abstain at `unresolved-mention`,
i.e. the mention matcher can't line the queried events up with extracted nodes.
This lets you try a better matcher / different threshold and measure the effect
immediately, then feed the result straight into compare_lextime.py.

Matchers:
  lexical : the original -- stemmed-token containment (reproduces run_external).
  char    : character 3-gram cosine -- robust to morphology / partial overlap /
            typos (recovers the near-miss bucket) with NO extra dependency.

Usage:
  # single config
  python rematch_tdg.py --graphs out/lextime_tdg_llama3.1_fair/graphs \
                        --match char --thr 0.42 \
                        --out out/lextime_tdg_llama3.1_char
  python compare_lextime.py --llm out/lextime_llm_llama3.1_fair \
                            --tdg out/lextime_tdg_llama3.1_char

  # sweep thresholds for one matcher (prints a coverage/accuracy table, writes nothing)
  python rematch_tdg.py --graphs out/lextime_tdg_llama3.1_fair/graphs \
                        --match char --sweep 0.30 0.55 0.05
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "was", "is", "with",
         "for", "by", "at", "his", "her", "their"}
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


# ----------------------------------------------------------------------------
# matchers: each maps (query_event_string, fact_text) -> similarity in [0,1]
# ----------------------------------------------------------------------------
def _lex_tokens(s: str) -> set:
    out = set()
    for t in re.findall(r"[a-z0-9]+", s.lower()):
        if t in _STOP:
            continue
        for suf in ("ment", "ing", "ion", "al", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[:len(t) - len(suf)]
                break
        out.add(t)
    return out


def _lex_score(event: str, fact_text: str) -> float:
    et, ft = _lex_tokens(event), _lex_tokens(fact_text)
    return len(et & ft) / len(et) if et and ft else 0.0


def _char_ngrams(s: str, n: int = 3) -> Counter:
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    s = f"  {s}  "
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _char_score(event: str, fact_text: str) -> float:
    a, b = _char_ngrams(event), _char_ngrams(fact_text)
    if not a or not b:
        return 0.0
    dot = sum(a[g] * b[g] for g in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


_SCORERS = {"lexical": _lex_score, "char": _char_score}


def _fact_text(f: dict) -> str:
    return " ".join(str(v) for v in f.values() if isinstance(v, str))


def _match_pair(e1: str, e2: str, facts: list, score, thr: float):
    """Joint assignment of BOTH events to DISTINCT facts, maximising total
    similarity (mirrors run_external's v3 matcher, generalised over the scorer)."""
    s1 = [score(e1, _fact_text(f)) for f in facts]
    s2 = [score(e2, _fact_text(f)) for f in facts]
    best, best_sum = (None, None), -1.0
    for i in range(len(facts)):
        for j in range(len(facts)):
            if i == j:
                continue
            if s1[i] >= thr and s2[j] >= thr and s1[i] + s2[j] > best_sum:
                best, best_sum = (facts[i], facts[j]), s1[i] + s2[j]
    return best, (max(s1, default=0.0), max(s2, default=0.0))


def derive(rec: dict, score, thr: float) -> tuple[Optional[str], str]:
    """Re-run match + order for one dumped graph record. Same logic as
    run_external.TDGOrderSystem.predict, minus the extraction call."""
    facts = rec.get("facts", [])
    rels = rec.get("dependencies", [])
    e1, e2 = rec.get("event1") or "", rec.get("event2") or ""
    kind = rec.get("kind", "order")
    if not facts:
        return None, "empty-graph"
    (f1, f2), (bc1, bc2) = _match_pair(e1, e2, facts, score, thr)
    note = f"matched={bool(f1)},{bool(f2)} best_c={bc1:.2f},{bc2:.2f}"
    order = None
    if f1 and f2:
        d1 = f1.get("date_parsed") or f1.get("value") or f1.get("date")
        d2 = f2.get("date_parsed") or f2.get("value") or f2.get("date")
        if d1 and d2 and _ISO.match(str(d1)) and _ISO.match(str(d2)) \
                and str(d1)[:10] != str(d2)[:10]:
            order = "BEFORE" if str(d1)[:10] < str(d2)[:10] else "AFTER"
            note += " by-date"
        else:
            id1 = f1.get("id") or f1.get("fact_id")
            id2 = f2.get("id") or f2.get("fact_id")
            adj: dict = {}
            for r in rels:
                adj.setdefault(r.get("from_id"), set()).add(r.get("to_id"))

            def reach(a, b):
                seen, stack = set(), [a]
                while stack:
                    x = stack.pop()
                    if x == b:
                        return True
                    if x in seen:
                        continue
                    seen.add(x)
                    stack.extend(adj.get(x, ()))
                return False
            if reach(id1, id2):
                order, note = "BEFORE", note + " by-path"
            elif reach(id2, id1):
                order, note = "AFTER", note + " by-path"
    if order is None:
        return None, note + (" unresolved-mention" if not (f1 and f2)
                             else " no-date-no-edge")
    if kind == "order":
        return order, note
    rel = rec.get("relation")
    if rel == "SIM":
        return None, note + " simultaneous-statement"
    return ("YES" if order == rel else "NO"), note + f" derived={order}"


def score_rows(rows: list) -> dict:
    n = len(rows)
    ans = [r for r in rows if r["pred"] is not None]
    correct = sum(1 for r in ans if r["pred"] == r["gold"])
    return {"n": n, "answered": len(ans),
            "coverage": round(len(ans) / n, 4) if n else None,
            "accuracy_answered": round(correct / len(ans), 4) if ans else None,
            "accuracy_all": round(correct / n, 4) if n else None,
            "correct": correct}


def load_graphs(gdir: Path) -> list:
    recs = [json.loads(p.read_text()) for p in sorted(gdir.glob("*.json"))
            if p.name != "results.json"]
    if not recs:
        raise SystemExit(f"no graph files in {gdir} -- run run_external.py "
                         f"--system tdg first (it dumps them by default).")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", required=True, help="dir of dumped graph json")
    ap.add_argument("--match", choices=list(_SCORERS), default="char")
    ap.add_argument("--thr", type=float, default=0.42)
    ap.add_argument("--out", help="write a compare-compatible results.json here")
    ap.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                    help="print coverage/accuracy for thr in [LO,HI) and exit")
    args = ap.parse_args()

    recs = load_graphs(Path(args.graphs))
    score = _SCORERS[args.match]

    if args.sweep:
        lo, hi, step = args.sweep
        print(f"matcher={args.match}  n={len(recs)}\n"
              f"  thr   coverage  acc_answered  acc_all  correct")
        t = lo
        while t < hi + 1e-9:
            rows = []
            for rec in recs:
                pred, _ = derive(rec, score, t)
                rows.append({"gold": rec["gold"], "pred": pred})
            s = score_rows(rows)
            print(f"  {t:4.2f}  {s['coverage']:7.1%}   "
                  f"{(s['accuracy_answered'] or 0):9.1%}   "
                  f"{(s['accuracy_all'] or 0):6.1%}   {s['correct']}")
            t += step
        return

    rows = []
    for rec in recs:
        pred, note = derive(rec, score, args.thr)
        rows.append({"id": rec["id"], "kind": rec.get("kind", "order"),
                     "gold": rec["gold"], "pair_type": rec.get("pair_type", ""),
                     "pred": pred, "note": note, "fallback": False,
                     "match": pred == rec["gold"] if pred is not None else False})
    summary = score_rows(rows)
    causes = Counter()
    for r in rows:
        if r["pred"] is not None:
            causes["answered"] += 1
        else:
            causes[r["note"].split()[-1]] += 1
    print(f"matcher={args.match} thr={args.thr}  " +
          "  ".join(f"{k}={v}" for k, v in summary.items()))
    print("  causes:", dict(causes.most_common()))

    if args.out:
        split_key = "\n".join(sorted(
            f"{r['id']}\t{r['gold']}\t{r.get('pair_type', '')}" for r in rows))
        meta = {"source_graphs": str(Path(args.graphs).resolve()),
                "matcher": args.match, "match_thr": args.thr,
                "n": len(rows), "system": "tdg-rematch",
                "gold_distribution": dict(Counter(r["gold"] for r in rows)),
                "split_sha256": hashlib.sha256(split_key.encode()).hexdigest()[:16]}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(
            {"dataset": "lextime", "mode": "tdg-rematch", "meta": meta,
             "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
        print(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
    