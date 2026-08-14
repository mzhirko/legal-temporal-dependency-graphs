#!/usr/bin/env python3
"""Re-score a finished cf pipeline run OFFLINE from its saved raw/*.tdg.json.

Extraction (the GPU-day) is reused; matching + gates + engine replay in
seconds under the CURRENT entailment code. Writes results.json in the same
shape as run_counterfactual_pipeline.py, plus a contamination split using
residual_refs from the sweep.

Usage (from code/):
  python src/rescore_cf.py src/external_bench/out/cf_anchor_pipeline_gemma \
      data/experiments/counterfactual/anchor_sweep.json data/results_uk
"""
import sys, json, re
from datetime import date
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lives in src/
from tdg_pipeline.entailment import check_entailment
from tdg_pipeline.io import load_tdg

CASE_STATUTE = {
    "2026_EAT_64_s111": "era_1996_s111",
    "2026_EAT_64_s123": "eqa_2010_s123",
    "2025_EAT_155": "era_1996_s111",
    "2026_EAT_14": "era_1996_s111",
    "2026_EAT_76": "eqa_2010_s123",
    "2026_EAT_59": "era_1996_s111",
    "kj_2026_EAT_46": "eqa_2010_s123",
}
_NAME = re.compile(r"^(?P<c>.+)__k(?P<k>[+-]\d+)$")


def main():
    run, sweep_p, stat_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sweep = json.load(open(sweep_p))
    oracle = {(cid, it["k"]): it
              for cid, c in sweep["cases"].items() for it in c["items"]}
    tdgs = {s: load_tdg(f"{stat_dir}/{s}.json")
            for s in set(CASE_STATUTE.values())}

    rows = []
    for f in sorted((run / "raw").glob("*.tdg.json")):
        pid = f.name[:-len(".tdg.json")]
        m = _NAME.match(pid)
        cid, k = m["c"], int(m["k"])
        it = oracle[(cid, k)]
        da = date.fromisoformat(it["acas_day_a"]) if it.get("acas_day_a") else None
        db = date.fromisoformat(it["acas_day_b"]) if it.get("acas_day_b") else None
        res = check_entailment(tdgs[CASE_STATUTE[cid]], load_tdg(f),
                               acas_day_a=da, acas_day_b=db)
        v = res[0].verdict if res else "INDETERMINATE"
        pred = v if v in ("TIMELY", "LATE") else None
        rows.append(dict(
            id=pid, case=cid, k=k, gold=it["verdict"], pred=pred,
            match=(pred == it["verdict"]),
            note="answered" if pred else "indeterminate",
            explanation=res[0].explanation if res else None,
            pred_anchor=res[0].anchor_date if res else None,
            pred_deadline=res[0].deadline_computed if res else None,
            gold_deadline=it["effective_deadline"],
            residual_refs=it.get("residual_refs", 0),
            anchor_correct=(res[0].anchor_date == it["anchor"]) if (res and pred) else None,
        ))

    def block(rs, label):
        a = [r for r in rs if r["pred"] is not None]
        c = sum(r["match"] for r in a)
        return {f"{label}_n": len(rs), f"{label}_answered": len(a),
                f"{label}_coverage": round(len(a) / len(rs), 4) if rs else None,
                f"{label}_acc_answered": round(c / len(a), 4) if a else None,
                f"{label}_acc_all": round(c / len(rs), 4) if rs else None}

    clean = [r for r in rows if not r["residual_refs"]]
    poisoned = [r for r in rows if r["residual_refs"]]
    anchor_ok = [r for r in rows if r["anchor_correct"]]
    anchor_bad = [r for r in rows if r["anchor_correct"] is False]

    summary = {**block(rows, "all"), **block(clean, "clean"),
               **block(poisoned, "contaminated")}
    summary["anchor_correct_rows"] = len(anchor_ok)
    summary["anchor_correct_acc"] = round(
        sum(r["match"] for r in anchor_ok) / len(anchor_ok), 4) if anchor_ok else None
    summary["anchor_wrong_rows"] = len(anchor_bad)
    summary["anchor_wrong_acc"] = round(
        sum(r["match"] for r in anchor_bad) / len(anchor_bad), 4) if anchor_bad else None
    reason = Counter()
    for r in rows:
        if r["pred"] is None:
            reason[(r["explanation"] or r["note"]).split(";")[0]
                   .split("--")[0].strip()[:60]] += 1
    summary["abstention_reasons"] = dict(reason.most_common())

    by_case = defaultdict(Counter)
    for r in rows:
        by_case[r["case"]]["answered" if r["pred"] else "abstain"] += 1
        if r["pred"]:
            by_case[r["case"]]["correct"] += r["match"]

    out = {"system": "pipeline-counterfactual, RESCORED offline from saved "
                     "extractions (raw/*.tdg.json) under current entailment.py",
           "summary": summary,
           "by_case": {c: dict(v) for c, v in by_case.items()},
           "rows": rows}
    (run / "results.json").write_text(json.dumps(out, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {run/'results.json'}")


if __name__ == "__main__":
    main()
