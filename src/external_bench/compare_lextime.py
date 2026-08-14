#!/usr/bin/env python3
"""Honest side-by-side comparison of an LLM baseline run and a TDG-pipeline run
on LexTime (or TRACIE).

The whole point of this script is that it will *refuse* to compare two runs
unless they were scored on the IDENTICAL set of items (same id -> gold mapping).
That single guard would have caught the original mistake, where the baseline was
scored on an all-`entailment` (single-class) slice and the pipeline on a balanced
slice, making "0.79 vs 0.21" meaningless.

It reports six views:

  1. Raw per-system:  coverage, accuracy_answered, accuracy_all, by pair_type,
     plus the TDG's abstention causes (an `empty-graph` rate far above the other
     model's is an EXTRACTION FAILURE being counted as a principled abstention).
  2. Selective-prediction (risk/coverage): credits the pipeline for knowing when
     it cannot ground an answer. TDG is reported at its own coverage; the LLM at
     coverage 1.0. This is the honest framing for a system that abstains.
  2b. CONDITIONAL: the LLM scored on EXACTLY the items the TDG answered. This is
     the view that decides whether the pipeline has any skill, and it is the one
     the 09/07 notes were missing. `acc(ans) 90% vs LLM 62%` compares different
     item sets; if the LLM also scores ~90% on the TDG's own subset, then the
     pipeline's precision is the difficulty of the items its matcher can reach,
     not skill. Reported overall and per pair_type.
  3. Forced-answer / hybrid: TDG where it commits, else the LLM's answer on the
     SAME item. Gives an accuracy_all that is head-to-head comparable with the
     LLM baseline, and isolates the pipeline's marginal value.
  4. TWIN CONSISTENCY: LexTime states most event pairs twice, once as
     "A precedes B" and once as "A follows B", with the gold flipped. Exactly one
     of a twin can be YES, so answering YES to both (or NO to both) is a logical
     self-contradiction, independent of which answer is right. A system that
     derives one order and reports it is 0% self-contradictory by construction.
     Needs the dumped graphs (they carry the parsed event pair).
  5. EFFECTIVE N + CLUSTER BOOTSTRAP: LexTime's items are not independent. They
     are event pairs (each answered by ONE derivation that decides both twins)
     drawn from a handful of paragraphs. Item-level tests over-report. This view
     prints the real denominators and recomputes the headline delta with a
     bootstrap over paragraphs. Needs the source csv.

On the subset where BOTH systems answer, it prints the 2x2 agreement table and a
two-sided exact McNemar test, so any claimed difference comes with a p-value
instead of a vibe -- reported at BOTH the item and the event-pair level, because
the item-level test double-counts twins.

Usage:
    python compare_lextime.py --llm out/lextime_llm_llama \
                              --tdg out/lextime_tdg_llama_v4
    # (pass either the run directory or its results.json)

Views 4 and 5 need the graphs / the source csv. Both are found automatically
(<tdg>/graphs or meta.source_graphs; meta.data_path); override with --graphs /
--data. If they are unavailable the view is skipped with a reason, never faked.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

_ABSTENTION_CAUSES = ("call-error", "empty-graph-unparseable",
                      "empty-graph-no-events", "empty-graph",
                      "unresolved-mention", "no-date-no-edge",
                      "simultaneous-statement", "hypothesis-unparsed")


def _load(p: str) -> dict:
    path = Path(p)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        sys.exit(f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_graphs(tdg_arg: str, res: dict, override: str | None) -> Path | None:
    """<run>/graphs for a --system tdg run; meta.source_graphs for a rematch."""
    for cand in (override,
                 str(Path(tdg_arg) / "graphs") if Path(tdg_arg).is_dir() else None,
                 res.get("meta", {}).get("source_graphs")):
        if cand and Path(cand).is_dir() and any(Path(cand).glob("*.json")):
            return Path(cand)
    return None


def _load_pairs(gdir: Path) -> dict[str, tuple]:
    """id -> (event1, event2) from the dumped graphs. The graph carries the TR
    already parsed into (event, relation, event), which is why we read it from
    here rather than re-parsing the csv's prose."""
    out = {}
    for p in sorted(gdir.glob("*.json")):
        if p.name == "results.json":
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("event1") and r.get("event2"):
            out[r["id"]] = (r["event1"], r["event2"])
    return out


def _load_paragraphs(res: dict, override: str | None) -> dict[str, int] | None:
    """id -> paragraph cluster index, from the csv this run actually scored.
    ids are positional (`lextime_%04d` over the csv rows; see
    run_external.load_lextime), so the join is exact, not fuzzy."""
    src = override or res.get("meta", {}).get("data_path", "")
    if not src or not Path(src).exists():
        return None
    rows = list(csv.DictReader(Path(src).open(encoding="utf-8")))
    if not rows:
        return None
    keys = {k.lower().strip(): k for k in rows[0]}
    if "paragraph" not in keys:
        return None
    ids: dict[str, int] = {}
    seen: dict[str, int] = {}
    for i, r in enumerate(rows):
        para = r[keys["paragraph"]]
        ids[f"lextime_{i:04d}"] = seen.setdefault(para, len(seen))
    return ids


def _cause(note: str) -> str:
    # Longest match first: "empty-graph-no-events" contains "empty-graph", so a
    # naive left-to-right scan collapses the diagnosis back into the coarse
    # bucket it was added to replace.
    for c in sorted(_ABSTENTION_CAUSES, key=len, reverse=True):
        if c in (note or ""):
            return c
    return "answered"


def _rows_by_id(res: dict) -> dict[str, dict]:
    return {r["id"]: r for r in res["rows"]}


def _gold_map(res: dict) -> dict[str, str]:
    return {r["id"]: r["gold"] for r in res["rows"]}


def _acc(rows: list[dict]) -> tuple[int, int, int, int]:
    """returns (n, answered, correct_answered, correct_over_all)"""
    n = len(rows)
    answered = [r for r in rows if r["pred"] is not None]
    correct = sum(1 for r in answered if r["pred"] == r["gold"])
    return n, len(answered), correct, correct


def _fmt_block(name: str, rows: list[dict]) -> str:
    n, ans, correct, _ = _acc(rows)
    cov = ans / n if n else 0.0
    acc_ans = correct / ans if ans else float("nan")
    acc_all = correct / n if n else float("nan")
    return (f"  {name:<26} n={n:<4} coverage={cov:6.1%}  "
            f"acc_answered={acc_ans:6.1%}  acc_all={acc_all:6.1%}  "
            f"(correct={correct})")


def _by_pair_type(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("pair_type", "") or "(none)", []).append(r)
    return out


def _mcnemar_exact(b: int, c: int) -> float:
    """two-sided exact McNemar p-value over discordant pairs (b, c)."""
    nn = b + c
    if nn == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(nn, i) for i in range(0, k + 1)) * (0.5 ** nn)
    return min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", required=True, help="LLM baseline run dir or results.json")
    ap.add_argument("--tdg", required=True, help="TDG pipeline run dir or results.json")
    ap.add_argument("--allow-subset", action="store_true",
                    help="compare on the intersection of ids if the two runs "
                         "are not identical (still requires gold to agree on "
                         "the overlap). Off by default: mismatches should be "
                         "fixed, not silently tolerated.")
    ap.add_argument("--graphs", help="dumped graphs dir for the twin-consistency "
                                     "view (default: <tdg>/graphs, else "
                                     "meta.source_graphs)")
    ap.add_argument("--data", help="source csv for the paragraph-cluster view "
                                   "(default: meta.data_path)")
    ap.add_argument("--boot", type=int, default=5000,
                    help="cluster-bootstrap resamples (0 = skip)")
    args = ap.parse_args()

    llm, tdg = _load(args.llm), _load(args.tdg)

    # ---- comparability guard -------------------------------------------------
    lg, tg = _gold_map(llm), _gold_map(tdg)
    lm, tm = llm.get("meta", {}), tdg.get("meta", {})
    sha_l, sha_t = lm.get("split_sha256"), tm.get("split_sha256")
    common = set(lg) & set(tg)
    disagree = {i for i in common if lg[i] != tg[i]}
    same_ids = set(lg) == set(tg)

    print("=" * 78)
    print("COMPARABILITY CHECK")
    print("=" * 78)
    print(f"  LLM run : {len(lg)} items  gold={lm.get('gold_distribution', '?')}  "
          f"sha={sha_l}  data={lm.get('data_path', '?')}")
    print(f"  TDG run : {len(tg)} items  gold={tm.get('gold_distribution', '?')}  "
          f"sha={sha_t}  data={tm.get('data_path', '?')}")
    for tag, res in (("LLM", llm), ("TDG", tdg)):
        gd = res.get("meta", {}).get("gold_distribution")
        if gd and len(gd) < 2:
            print(f"  !! {tag} run is SINGLE-CLASS {gd}: its accuracy is a "
                  f"base-rate, not skill. Re-run it with --seed and --limit 0.")

    # A balanced GOLD set does not make a baseline meaningful: a model that
    # answers YES to everything scores the majority base-rate while measuring
    # nothing. The seeded shuffle fixed the gold imbalance; this catches the
    # PREDICTION imbalance, which is the same error one layer in and is what
    # made llama3.1:8b's 62% look like a task difficulty rather than a bias.
    print()
    for tag, res in (("LLM", llm), ("TDG", tdg)):
        preds = [r["pred"] for r in res["rows"] if r["pred"] is not None]
        golds = [r["gold"] for r in res["rows"]]
        if not preds:
            continue
        pd_, gd_ = Counter(preds), Counter(golds)
        maj = max(gd_.values()) / len(golds)
        line = (f"  {tag} predicted={dict(pd_)}  gold={dict(gd_)}  "
                f"majority-baseline={maj:.1%}")
        print(line)
        for lab, k in pd_.items():
            # only flag the OVER-predicted label: in a binary task the mirrored
            # under-prediction of the other label is the same fact twice.
            p_rate, g_rate = k / len(preds), gd_.get(lab, 0) / len(golds)
            if p_rate - g_rate > 0.15:
                print(f"     !! {tag} predicts {lab} on {p_rate:.1%} of answers "
                      f"but gold is {g_rate:.1%} {lab}: this baseline is "
                      f"{lab}-BIASED. Beating it is not evidence about the "
                      f"other system. Report this next to its accuracy.")

    aligned = same_ids and not disagree
    if sha_l and sha_t:
        aligned = aligned and (sha_l == sha_t)

    if aligned:
        print("  OK: identical item set and gold labels. Comparison is valid.\n")
        rows_scope = sorted(common)
    else:
        print("  MISMATCH:")
        if not same_ids:
            only_l, only_t = set(lg) - set(tg), set(tg) - set(lg)
            print(f"     ids only in LLM: {len(only_l)}  only in TDG: {len(only_t)}"
                  f"  shared: {len(common)}")
        if disagree:
            ex = list(disagree)[:5]
            print(f"     {len(disagree)} shared ids have DIFFERENT gold labels, "
                  f"e.g. {[(i, lg[i], tg[i]) for i in ex]}")
        if sha_l and sha_t and sha_l != sha_t:
            print("     split_sha256 differs -> the two runs are not the same "
                  "evaluation.")
        if not args.allow_subset:
            print("\n  REFUSING to report a head-to-head number: these runs are "
                  "not comparable.\n  Fix: run BOTH conditions with the same "
                  "--data, --seed and --limit, then rerun this script.\n  "
                  "(Or pass --allow-subset to compare only the aligned "
                  "overlap, if any.)")
            sys.exit(2)
        rows_scope = sorted(i for i in common if lg[i] == tg[i])
        if not rows_scope:
            sys.exit("  no aligned overlap to compare; aborting.")
        print(f"\n  --allow-subset: comparing on {len(rows_scope)} aligned "
              f"items only.\n")

    lb, tb = _rows_by_id(llm), _rows_by_id(tdg)
    llm_rows = [lb[i] for i in rows_scope]
    tdg_rows = [tb[i] for i in rows_scope]

    # ---- view 1: raw per-system ---------------------------------------------
    print("=" * 78)
    print("VIEW 1 - RAW PER-SYSTEM")
    print("=" * 78)
    print(_fmt_block("LLM baseline", llm_rows))
    print(_fmt_block("TDG pipeline", tdg_rows))
    print("\n  by pair_type (TDG pipeline):")
    for pt, rs in sorted(_by_pair_type(tdg_rows).items()):
        print(_fmt_block("    " + pt, rs))
    print("\n  by pair_type (LLM baseline):")
    for pt, rs in sorted(_by_pair_type(llm_rows).items()):
        print(_fmt_block("    " + pt, rs))

    causes = Counter(_cause(r.get("note", "")) for r in tdg_rows)
    print("\n  TDG abstention causes:")
    for cse, k in causes.most_common():
        print(f"    {cse:24s} {k:4d}  ({k/len(tdg_rows):5.1%})")
    infra = causes.get("call-error", 0)
    unparse = causes.get("empty-graph-unparseable", 0)
    genuine = causes.get("empty-graph-no-events", 0)
    unknown = causes.get("empty-graph", 0)
    n = max(1, len(tdg_rows))
    if infra:
        print(f"\n  !! {infra} items ({infra/n:.1%}) are INFRA: the extraction call "
              f"failed and the pipeline swallowed it.\n     These are not model "
              f"behaviour and must not be reported as coverage. Rerun them "
              f"(--redo-empty).")
    if unparse:
        print(f"\n  !! {unparse} items ({unparse/n:.1%}) returned UNPARSEABLE JSON. "
              f"A real model failure, but of\n     serialisation, not of reading "
              f"the text -- a different sentence in the chapter.")
    if genuine:
        print(f"\n  -> {genuine} items ({genuine/n:.1%}) are empty-graph-no-events: "
              f"the model read the text and\n     found no temporal content. This "
              f"one IS a genuine abstention and is reportable as such.")
    if unknown / n > 0.05:
        print(f"\n  !! {unknown} items ({unknown/n:.1%}) are bare 'empty-graph' -- "
              f"cause NOT recorded, so this run\n     predates the instrument fix. "
              f"The pipeline swallows a failed extraction call and returns an\n"
              f"     empty graph, and the old guard was an endpoint ping fired "
              f"AFTER the fact: it detects a DEAD\n     endpoint, not a failed "
              f"call, so call_errors_infra=0 does not clear these. Rerun with "
              f"--redo-empty\n     to find out which of the three they are.")

    # ---- view 2: selective prediction ---------------------------------------
    n = len(rows_scope)
    t_ans = [r for r in tdg_rows if r["pred"] is not None]
    t_cov = len(t_ans) / n if n else 0.0
    t_acc_ans = (sum(1 for r in t_ans if r["pred"] == r["gold"]) / len(t_ans)
                 if t_ans else float("nan"))
    l_correct = sum(1 for r in llm_rows if r["pred"] == r["gold"])
    l_acc = l_correct / n if n else float("nan")
    print("\n" + "=" * 78)
    print("VIEW 2 - SELECTIVE PREDICTION (credits the pipeline for abstaining)")
    print("=" * 78)
    print(f"  TDG pipeline : answers {t_cov:.1%} of items, and is "
          f"{t_acc_ans:.1%} accurate WHEN IT ANSWERS.")
    print(f"  LLM baseline : answers 100% of items at {l_acc:.1%} accuracy.")
    print("  Read as a risk/coverage trade-off: the pipeline is a high-precision,"
          "\n  low-coverage predictor; the LLM is full-coverage. In a legal "
          "setting an\n  abstention ('cannot ground this') is not equivalent to "
          "a wrong answer.")
    print("  !! Those two numbers are computed on DIFFERENT item sets and are "
          "NOT a\n     head-to-head result. See view 2b before quoting them.")

    # ---- view 2b: conditional (the LLM on the TDG's own subset) --------------
    # The comparison that view 2 invites is invalid: the TDG's coverage is not a
    # random sample, it is the set its matcher could reach. The only way to know
    # whether abstaining bought anything is to score the LLM on exactly that set.
    print("\n" + "=" * 78)
    print("VIEW 2b - CONDITIONAL: the LLM on EXACTLY the items the TDG answered")
    print("=" * 78)
    S = [i for i in rows_scope if tb[i]["pred"] is not None]
    if not S:
        print("  TDG answered nothing; no conditional comparison possible.")
    else:
        def _pair(ids_):
            t = [i for i in ids_ if tb[i]["pred"] is not None]
            l = [i for i in ids_ if lb[i]["pred"] is not None]
            at = sum(1 for i in t if tb[i]["pred"] == tg[i]) / len(t) if t else float("nan")
            al = sum(1 for i in l if lb[i]["pred"] == lg[i]) / len(l) if l else float("nan")
            return at, al

        at, al = _pair(S)
        print(f"  TDG on its own subset      : {at:6.1%}  (n={len(S)})")
        print(f"  LLM on the SAME subset     : {al:6.1%}")
        print(f"  LLM on ALL items           : {l_acc:6.1%}")
        print(f"  TDG - LLM, same items      : {at - al:+6.1%}   <- the honest delta")
        if al > l_acc + 0.02:
            print(f"\n  !! The LLM scores {al:.1%} on the TDG's subset vs {l_acc:.1%} "
                  f"overall: the items the TDG\n     commits to are EASIER than "
                  f"average. Its accuracy_answered is partly item\n     selection, "
                  f"not skill. Quote view 2b's delta, never acc_answered alone.")
        if at < al:
            print(f"\n  !! The TDG is BEHIND the LLM on the items it chose to "
                  f"answer. Abstention is\n     buying nothing here: the LLM "
                  f"already handles this subset better, unaided.")
        print("\n  by pair_type (cov | TDG | LLM same items | LLM all of type | delta):")
        for pt in sorted({r.get("pair_type", "") or "(none)" for r in tdg_rows}):
            ids_pt = [i for i in rows_scope
                      if (tb[i].get("pair_type", "") or "(none)") == pt]
            S_pt = [i for i in ids_pt if tb[i]["pred"] is not None]
            if not S_pt:
                print(f"    {pt:26s} cov=  0.0%   (never answers this pair type)")
                continue
            at_pt, al_pt = _pair(S_pt)
            base_pt = [i for i in ids_pt if lb[i]["pred"] is not None]
            allb = sum(1 for i in base_pt if lb[i]["pred"] == lg[i]) / len(base_pt)
            print(f"    {pt:26s} cov={len(S_pt)/len(ids_pt):6.1%}  TDG={at_pt:6.1%}  "
                  f"LLM_same={al_pt:6.1%}  LLM_all={allb:6.1%}  "
                  f"delta={at_pt - al_pt:+6.1%}")

    # ---- view 3: forced-answer / hybrid -------------------------------------
    # hybrid = TDG where it commits, else LLM's answer on the same id.
    hy_correct = 0
    used_fallback = 0
    for i in rows_scope:
        p = tb[i]["pred"]
        if p is None:
            p = lb[i]["pred"]
            used_fallback += 1
        if p == lg[i]:
            hy_correct += 1
    hy_acc = hy_correct / n if n else float("nan")
    t_acc_all = sum(1 for r in tdg_rows if r["pred"] == r["gold"]) / n if n else float("nan")
    print("\n" + "=" * 78)
    print("VIEW 3 - FORCED-ANSWER / HYBRID (head-to-head accuracy_all)")
    print("=" * 78)
    print(f"  LLM baseline  acc_all            = {l_acc:6.1%}")
    print(f"  TDG (abstain=wrong) acc_all       = {t_acc_all:6.1%}   "
          f"<- what the naive comparison used")
    print(f"  TDG + LLM-fallback (hybrid) acc_all = {hy_acc:6.1%}   "
          f"(fell back on {used_fallback}/{n} items)")
    delta = hy_acc - l_acc
    print(f"  hybrid - LLM                      = {delta:+.1%}   "
          f"(is the pipeline adding value over the raw LLM?)")

    # ---- significance on the both-answered subset ---------------------------
    both = [i for i in rows_scope if lb[i]["pred"] is not None and tb[i]["pred"] is not None]
    b = sum(1 for i in both if tb[i]["pred"] == tg[i] and lb[i]["pred"] != lg[i])  # tdg right, llm wrong
    c = sum(1 for i in both if tb[i]["pred"] != tg[i] and lb[i]["pred"] == lg[i])  # tdg wrong, llm right
    both_agree = sum(1 for i in both if tb[i]["pred"] == lb[i]["pred"])
    p = _mcnemar_exact(b, c)
    print("\n" + "=" * 78)
    print("SIGNIFICANCE - items BOTH systems answered")
    print("=" * 78)
    print(f"  both-answered items      : {len(both)}")
    print(f"  prediction agreement     : {both_agree}/{len(both)}"
          f" ({(both_agree/len(both) if both else 0):.1%})")
    print(f"  TDG right / LLM wrong (b): {b}")
    print(f"  TDG wrong / LLM right (c): {c}")
    print(f"  McNemar exact (2-sided) p: {p:.4f}"
          f"   {'(no significant difference)' if p > 0.05 else '(significant)'}"
          f"   [ITEM level]")
    print("\n  Note: this subset is where the TDG chose to commit, so it is not a"
          "\n  random sample - report it as a conditional comparison, not the "
          "headline.")

    # ---- views 4 & 5: the item is not the unit of observation ---------------
    gdir = _find_graphs(args.tdg, tdg, args.graphs)
    pairs = _load_pairs(gdir) if gdir else None
    paras = _load_paragraphs(tdg, args.data) or _load_paragraphs(llm, args.data)

    print("\n" + "=" * 78)
    print("VIEW 4 - TWIN CONSISTENCY (logical coherence, independent of accuracy)")
    print("=" * 78)
    if not pairs:
        print("  SKIPPED: no dumped graphs found (looked in <tdg>/graphs and "
              "meta.source_graphs).\n  Re-run the TDG condition without "
              "--no-dump-graphs, or pass --graphs DIR.")
    else:
        twins: dict[tuple, list] = {}
        for i in rows_scope:
            if i in pairs:
                twins.setdefault(pairs[i], []).append(i)
        tw = [v for v in twins.values()
              if len(v) == 2 and lg[v[0]] != lg[v[1]]]
        print(f"  flipped twins in scope: {len(tw)}  (same event pair, relation "
              f"reversed, gold reversed)")
        print("  exactly one of a twin can be YES, so YES/YES or NO/NO is a "
              "self-contradiction.\n")
        for tag, byid in (("LLM baseline", lb), ("TDG pipeline", tb)):
            coh = inc = 0
            for a, b_ in tw:
                pa, pb = byid[a]["pred"], byid[b_]["pred"]
                if pa is None or pb is None:
                    continue
                if (pa == "YES") != (pb == "YES"):
                    coh += 1
                else:
                    inc += 1
            tot = coh + inc
            if not tot:
                print(f"  {tag:14s} answers no complete twin.")
                continue
            print(f"  {tag:14s} both-answered twins={tot:4d}  coherent={coh:4d} "
                  f"({coh/tot:5.1%})  SELF-CONTRADICTORY={inc:4d} ({inc/tot:5.1%})")
        print("\n  A system that derives one order per pair and reports it cannot "
              "contradict\n  itself: 0% here is BY CONSTRUCTION, not a tuned "
              "result, and it holds\n  whatever the accuracy is. This is the "
              "LexTime instance of C8.")

    print("\n" + "=" * 78)
    print("VIEW 5 - EFFECTIVE N + PARAGRAPH CLUSTER BOOTSTRAP")
    print("=" * 78)
    n_pairs = len({pairs[i] for i in rows_scope if i in pairs}) if pairs else None
    n_para = len({paras[i] for i in rows_scope if i in paras}) if paras else None
    print(f"  items                : {len(rows_scope)}")
    print(f"  distinct event pairs : {n_pairs if n_pairs else '? (need graphs)'}"
          f"   <- the TDG makes ONE derivation per pair; it decides both twins")
    print(f"  distinct paragraphs  : {n_para if n_para else '? (need --data)'}"
          f"   <- the real cluster")
    if n_pairs:
        # The observation is the EVENT PAIR, and the question underneath
        # LexTime's yes/no framing is "did the system recover the true ordering
        # of these two events?". A system recovers it iff it answers EVERY
        # statement about that pair, and gets them all right -- answering YES to
        # both "A precedes B" and "A follows B" is not half-knowledge, it is no
        # knowledge. Scoring twins separately gives a system credit for the coin
        # landing its way on one of the two.
        grp: dict[tuple, list] = {}
        for i in rows_scope:
            if i in pairs:
                grp.setdefault(pairs[i], []).append(i)

        def _recovered(byid, gold, k):
            its = grp[k]
            if any(byid[i]["pred"] is None for i in its):
                return None
            return all(byid[i]["pred"] == gold[i] for i in its)

        b2 = c2 = 0
        both_pairs = 0
        for k in grp:
            t_ok = _recovered(tb, tg, k)
            l_ok = _recovered(lb, lg, k)
            if t_ok is None or l_ok is None:
                continue
            both_pairs += 1
            b2 += t_ok and not l_ok
            c2 += l_ok and not t_ok
        print(f"\n  ordering RECOVERED (all statements about the pair answered "
              f"and correct):")
        for tag, byid, gold in (("LLM", lb, lg), ("TDG", tb, tg)):
            ok = sum(1 for k in grp if _recovered(byid, gold, k))
            ans = sum(1 for k in grp if _recovered(byid, gold, k) is not None)
            print(f"    {tag}: {ok}/{ans} answered pairs ({ok/ans:.1%})"
                  f"   coverage {ans}/{len(grp)} = {ans/len(grp):.1%}")
        print(f"\n  McNemar on EVENT PAIRS: both-answered={both_pairs} "
              f"b={b2} c={c2} p={_mcnemar_exact(b2, c2):.4f}   [PAIR level]")
        print("  The item-level p above double-counts every twin. Quote this one.")
    if paras and args.boot:
        byp: dict[int, list] = {}
        for i in rows_scope:
            if i in paras:
                byp.setdefault(paras[i], []).append(i)
        keys = list(byp)

        def _delta(ids_):
            t = [i for i in ids_ if tb[i]["pred"] is not None]
            if not t:
                return None
            at_ = sum(1 for i in t if tb[i]["pred"] == tg[i]) / len(t)
            l = [i for i in t if lb[i]["pred"] is not None]
            if not l:
                return None
            al_ = sum(1 for i in l if lb[i]["pred"] == lg[i]) / len(l)
            return len(t) / len(ids_), at_, at_ - al_

        random.seed(0)
        covs, accs, dels = [], [], []
        for _ in range(args.boot):
            samp = [i for k in random.choices(keys, k=len(keys)) for i in byp[k]]
            d = _delta(samp)
            if d:
                covs.append(d[0]); accs.append(d[1]); dels.append(d[2])
        q = lambda v, pr: sorted(v)[int(pr * len(v))]
        pt_ = _delta(rows_scope)
        print(f"\n  bootstrap over {len(keys)} paragraphs, {args.boot} resamples:")
        print(f"    coverage      = {pt_[0]:6.1%}  95% CI [{q(covs,.025):.1%}, "
              f"{q(covs,.975):.1%}]")
        print(f"    acc(answered) = {pt_[1]:6.1%}  95% CI [{q(accs,.025):.1%}, "
              f"{q(accs,.975):.1%}]")
        print(f"    delta vs LLM  = {pt_[2]:+6.1%}  95% CI [{q(dels,.025):+.1%}, "
              f"{q(dels,.975):+.1%}]   P(delta<=0) = "
              f"{sum(1 for d in dels if d <= 0)/len(dels):.3f}")
        print("\n  Item-level intervals on this benchmark are too narrow: the "
              "items are twins of\n  each other inside a handful of documents. "
              "Quote the clustered interval, and\n  state n_paragraphs next to "
              "n_items.")


if __name__ == "__main__":
    main()
    