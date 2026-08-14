#!/usr/bin/env python3
"""Offline v3 replay over archived TDGs -- per PREREG_matcher_v3.md.

Frozen inputs, no model calls. Four conditions per input set:

  v2    the engine exactly as published (baseline replay; must equal the
        archived results -- that equality is the gate, checked here).
  v3E   edge_provenance.validate_additive_edges + apply_edge_audit on each
        INSTANCE TDG (deep copy; archives untouched). Statute TDGs are
        audited in REPORT-ONLY mode (prereg scopes application to the
        frozen instance sets).
  v3L   composed-evidence anchor recovery for ABSTAINED rows only. Rows the
        v2 engine answered are structurally untouchable here, so prereg P2
        can only fail via rounding, not via override. A conversion happens
        iff EventLinker (statute+instance bundle, composed evidence) links
        the discovered rule's anchor fact to exactly ONE dated instance
        fact; the deadline then comes from the same engine primitives the
        v2 path uses (rule.offset.apply, _apply_conciliation,
        _select_action). Two candidate links with different dates = still
        abstained (the prereg's design principle: below-threshold or
        contested evidence never merges).
  v3EL  both.

Modes:
  cf    a counterfactual run dir (raw/*.tdg.json) scored against the
        archived oracle sweep -- same row schema as rescore_cf.py.
  x2    a directory of 2x2 pipeline TDGs scored against the gold facts
        ground truth -- same row schema, per-directory aggregate.

Usage:
  python run_v3_replay.py cf out/cf_anchor_pipeline_gemma \\
      ../data/experiments/counterfactual/anchor_sweep.json \\
      ../data/results_uk --conditions v2 v3E v3L v3EL \\
      --out out/v3_replay/cf_gemma

  python run_v3_replay.py x2 ../data/results_pipeline/gemma_fulltext/era \\
      ../data/ground_truth/ground_truth_gold.json ../data/results_uk \\
      --statute era_1996_s111 --out out/v3_replay/x2_gemma_full_era

Every output file records the condition, the edge-audit summary, and each
v3L conversion with its composed-evidence string, so the thesis table and
its per-case diagnosis both read straight out of the JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdg_pipeline.io import load_tdg                                # noqa: E402
from tdg_pipeline.entailment import (                               # noqa: E402
    check_entailment, find_rules, _apply_conciliation, _select_action,
)
from tdg_pipeline.edge_provenance import (                          # noqa: E402
    validate_additive_edges, apply_edge_audit,
)
from tdg_pipeline.linking import EventLinker                        # noqa: E402
from rescore_cf import CASE_STATUTE, _NAME                          # noqa: E402

CONDITIONS = ("v2", "v3E", "v3L", "v3EL")


# -- shared: score one instance TDG under one condition --------------------

def _score_one(statute_tdg, instance_tdg, da, db, cond, embedder=None):
    """Row dict in rescore_cf's schema + v3 bookkeeping."""
    edge_report = None
    tdg = instance_tdg
    if "E" in cond and cond != "v2":
        tdg = copy.deepcopy(instance_tdg)
        audit = validate_additive_edges(tdg)
        applied = apply_edge_audit(tdg, audit)
        edge_report = {"summary": audit.summary(), "applied": applied,
                       "downgraded": [f.claimed for f in audit.downgraded]}

    res = check_entailment(statute_tdg, tdg, embedder=embedder,
                           acas_day_a=da, acas_day_b=db)
    r = res[0] if res else None
    verdict = r.verdict if r else "INDETERMINATE"
    row = {
        "pred": verdict if verdict in ("TIMELY", "LATE") else None,
        "note": "answered" if verdict in ("TIMELY", "LATE") else "indeterminate",
        "explanation": r.explanation if r else "no rule discovered",
        "pred_anchor": r.anchor_date if r else None,
        "pred_deadline": r.deadline_computed if r else None,
        "edge_audit": edge_report,
        "v3l": None,
    }

    if "L" in cond and row["pred"] is None:
        conv = _recover_anchor(statute_tdg, tdg, da, db, embedder)
        if conv is not None:
            row.update(pred=conv["verdict"], note="v3L-converted",
                       pred_anchor=conv["anchor"], pred_deadline=conv["deadline"],
                       explanation=conv["explanation"], v3l=conv)
        else:
            row["v3l"] = {"converted": False}
    return row


def _recover_anchor(statute_tdg, instance_tdg, da, db, embedder):
    """Composed-evidence anchor recovery. None = stay abstained."""
    rules = find_rules(statute_tdg, embedder=embedder)
    if not rules:
        return None
    rule = rules[0]
    linker = EventLinker({statute_tdg.document_id: statute_tdg,
                          instance_tdg.document_id: instance_tdg},
                         embedder=embedder)
    anchor_links = []
    for doc_a, fa, doc_b, fb, ev, rel in linker.link_all():
        pair = {(doc_a, fa.id): fb, (doc_b, fb.id): fa}
        hit = pair.get((statute_tdg.document_id, rule.anchor_fact.id))
        if hit is not None and hit.timex.date_parsed:
            anchor_links.append((hit, ev))
    dates = {f.timex.date_parsed for f, _ in anchor_links}
    if len(dates) != 1:
        return None            # zero evidence, or a contested anchor: abstain
    fact, ev = anchor_links[0]
    anchor = fact.timex.date_parsed
    deadline = rule.offset.apply(anchor)
    deadline, acas = _apply_conciliation(deadline, da, db)
    post = [f for f in instance_tdg.facts
            if f.timex.date_parsed and f.timex.date_parsed > anchor
            and f.id != fact.id]
    if not post:
        return None            # anchor recovered but nothing to judge
    action, aconf, how = _select_action(rule, post, embedder)
    days_over = (action.timex.date_parsed - deadline).days
    return {
        "converted": True,
        "verdict": "TIMELY" if days_over <= 0 else "LATE",
        "anchor": anchor.isoformat(), "deadline": deadline.isoformat(),
        "acas_applied": acas,
        "link_evidence": str(ev),
        "linked_entity": fact.entity,
        "action_via": how,
        "explanation": (f"v3L: anchor '{fact.entity}' ({anchor.isoformat()}) "
                        f"via composed link [{ev}]; deadline "
                        f"{deadline.isoformat()}{' (ACAS)' if acas else ''}; "
                        f"action {action.timex.date_parsed.isoformat()} "
                        f"via {how} -> "
                        f"{'TIMELY' if days_over <= 0 else 'LATE'}"),
    }


# -- cf mode ---------------------------------------------------------------

def replay_cf(run: Path, sweep_path: Path, statute_dir: Path,
              conditions, outdir: Path) -> None:
    sweep = json.loads(sweep_path.read_text())
    oracle = {(cid, it["k"]): it
              for cid, c in sweep["cases"].items() for it in c["items"]}
    statutes = {s: load_tdg(statute_dir / f"{s}.json")
                for s in set(CASE_STATUTE.values())}
    _report_statute_audits(statutes.values(), outdir)

    files = sorted((run / "raw").glob("*.tdg.json"))
    for cond in conditions:
        rows = []
        for f in files:
            pid = f.name[: -len(".tdg.json")]
            m = _NAME.match(pid)
            cid, k = m["c"], int(m["k"])
            it = oracle[(cid, k)]
            da = date.fromisoformat(it["acas_day_a"]) if it.get("acas_day_a") else None
            db = date.fromisoformat(it["acas_day_b"]) if it.get("acas_day_b") else None
            row = _score_one(statutes[CASE_STATUTE[cid]], load_tdg(f), da, db, cond)
            row.update(id=pid, case=cid, k=k, gold=it["verdict"],
                       match=(row["pred"] == it["verdict"]))
            rows.append(row)
        _write(rows, cond, outdir, key="case")


# -- x2 mode ---------------------------------------------------------------

def replay_x2(tdg_dir: Path, gold_path: Path, statute_dir: Path,
              statute_key: str, conditions, outdir: Path) -> None:
    gold = json.loads(gold_path.read_text())["cases"]
    statute = load_tdg(statute_dir / f"{statute_key}.json")
    _report_statute_audits([statute], outdir)
    files = sorted(tdg_dir.glob("*.json"))
    for cond in conditions:
        rows = []
        for f in files:
            cid = f.stem
            c = gold.get(cid, {})
            da = date.fromisoformat(c["acas_day_a"]) if c.get("acas_day_a") else None
            db = date.fromisoformat(c["acas_day_b"]) if c.get("acas_day_b") else None
            gv = {"in_time": "TIMELY", "out_of_time": "LATE"}.get(
                str(c.get("verdict", "")).lower())
            row = _score_one(statute, load_tdg(f), da, db, cond)
            row.update(id=cid, case=cid, gold=gv,
                       match=(gv is not None and row["pred"] == gv))
            rows.append(row)
        _write(rows, cond, outdir, key="case")


# -- output ----------------------------------------------------------------

def _report_statute_audits(statute_tdgs, outdir: Path) -> None:
    """Prereg scopes edge application to instance TDGs; statutes are audited
    report-only, because a downgraded statute edge would delete the rule
    itself -- a data-quality finding to report, never a silent mutation."""
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [validate_additive_edges(t).summary() for t in statute_tdgs]
    (outdir / "statute_edge_audit.txt").write_text("\n".join(lines) + "\n")


def _write(rows, cond, outdir: Path, key: str) -> None:
    answered = [r for r in rows if r["pred"]]
    correct = [r for r in answered if r["match"]]
    converted = [r for r in rows if r["note"] == "v3L-converted"]
    conv_correct = [r for r in converted if r["match"]]
    per_case = {}
    for r in rows:
        d = per_case.setdefault(r[key], {"n": 0, "answered": 0, "correct": 0,
                                         "converted": 0, "conv_correct": 0})
        d["n"] += 1
        d["answered"] += bool(r["pred"])
        d["correct"] += bool(r["match"])
        d["converted"] += (r["note"] == "v3L-converted")
        d["conv_correct"] += (r["note"] == "v3L-converted" and r["match"])
    summary = {
        "condition": cond,
        "n": len(rows),
        "coverage": round(len(answered) / len(rows), 4) if rows else None,
        "acc_answered": round(len(correct) / len(answered), 4) if answered else None,
        "acc_all": round(len(correct) / len(rows), 4) if rows else None,
        "converted": len(converted),
        "acc_converted": (round(len(conv_correct) / len(converted), 4)
                          if converted else None),
        "edges_downgraded": sum((r["edge_audit"] or {}).get("applied", 0)
                                for r in rows),
        "by_case": per_case,
    }
    out = outdir / f"rescored_{cond}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"[{cond}] cov={summary['coverage']} acc_ans={summary['acc_answered']} "
          f"acc_all={summary['acc_all']} converted={summary['converted']} "
          f"(acc {summary['acc_converted']}) edges_downgraded="
          f"{summary['edges_downgraded']}  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cf", "x2"])
    ap.add_argument("inputs", nargs=3,
                    help="cf: RUN_DIR SWEEP_JSON STATUTE_DIR | "
                         "x2: TDG_DIR GOLD_JSON STATUTE_DIR")
    ap.add_argument("--statute", help="x2 only: statute key (e.g. era_1996_s111)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=CONDITIONS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b, c = (Path(p) for p in args.inputs)
    outdir = Path(args.out)
    if args.mode == "cf":
        replay_cf(a, b, c, args.conditions, outdir)
    else:
        if not args.statute:
            ap.error("x2 mode needs --statute")
        replay_x2(a, b, c, args.statute, args.conditions, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
