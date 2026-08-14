import sys, os, json, glob, importlib
sys.path.insert(0, os.getcwd())
from pathlib import Path
from tdg_pipeline.tdg import (TimexSpan, TemporalFact, TemporalDependencyGraph)

CMP   = "../data/experiments/comparison_50"
TDGS  = "../data/results_contracts_50"
CAT   = "../data/experiments/catala_50"

def load_tdg_facts(seed):
    d = json.load(open(f"{TDGS}/{seed}.json"))
    facts = []
    for f in d["facts"]:
        dp = f.get("date_parsed")
        from datetime import date
        dpv = date.fromisoformat(dp) if dp else None
        tx = TimexSpan(text=f.get("raw_text",""), timex_type=f.get("timex_type","DATE"),
                       value=f.get("value"), start_char=0, end_char=0, date_parsed=dpv,
                       duration_days=f.get("duration_days"))
        facts.append(TemporalFact(id=f["id"], entity=f["entity"], role=f.get("role","UNKNOWN"),
                                  timex=tx, sentence=f.get("sentence","")))
    return facts

def catala_outputs_for(seed):
    d = json.load(open(f"{CMP}/{seed}_comparison.json"))
    return {fld["variable_name"]: fld["catala_value"] for fld in d["fields"]
            if fld.get("catala_value") is not None and not fld["variable_name"].startswith("tdg:")}, d.get("catala_status")

def run(label):
    import comparator.align as A
    importlib.reload(A)
    agg = {}
    statuses = {}
    seeds = [os.path.basename(p).replace("_comparison.json","")
             for p in sorted(glob.glob(f"{CMP}/*_comparison.json"))]
    for seed in seeds:
        outs, status = catala_outputs_for(seed)
        cat_file = Path(f"{CAT}/{seed}.catala_en")
        if not cat_file.exists() or not os.path.exists(f"{TDGS}/{seed}.json"):
            continue
        facts = load_tdg_facts(seed)
        rep = A.align_and_compare(
            document_id=seed, tdg_facts=facts, catala_outputs=outs,
            catala_status="success", scope_name=None, repair_attempts=0,
            catala_file=cat_file, catala_inputs={})
        for fld in rep.fields:
            agg[fld.status] = agg.get(fld.status, 0) + 1
    return agg

# BEFORE: original align.py
import shutil
shutil.copy("/tmp/align_orig.py", "comparator/align.py")
before = run("before")
# AFTER: patched align.py
shutil.copy("/tmp/align_patched.py", "comparator/align.py")
after = run("after")

def overlap(a):
    semantic = a.get("semantic_match",0)+a.get("semantic_mismatch",0)
    direct   = a.get("match",0)+a.get("mismatch",0)+a.get("off_by_one",0)+a.get("value_match",0)
    matched  = direct+semantic
    catala_only = a.get("catala_only",0); tdg_only = a.get("tdg_only",0)
    tot = matched+catala_only+tdg_only+a.get("type_mismatch",0)
    return matched, catala_only, tdg_only, tot

for label, a in (("BEFORE (original)", before), ("AFTER (patched)", after)):
    m, co, to, tot = overlap(a)
    print(f"\n{label}:")
    print(f"  statuses: {dict(sorted(a.items()))}")
    print(f"  matched(any)={m}  catala_only={co}  tdg_only={to}  total_fields={tot}")
    print(f"  semantic_match={a.get('semantic_match',0)} semantic_mismatch={a.get('semantic_mismatch',0)}")