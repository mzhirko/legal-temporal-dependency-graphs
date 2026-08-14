# Run from code/src:  python /tmp/hopdecay.py ../data/evaluation_results/proof_head2head_v2.json
import json, sys, os
sys.path.insert(0, os.getcwd())
from proof.solvers import _parse_date_flexible as P

d = json.load(open(sys.argv[1]))
gold = {r["item_id"]: r for r in d["benchmark"]}
# chain order of event names (depth 1,2,3)
ORDER = ["notice deadline", "grievance window", "appeal deadline", "final review"]

for col in [c for c in d["answers"] if c.startswith("llm:")]:
    per_hop = {1:[0,0], 2:[0,0], 3:[0,0]}   # name-depth -> [correct, total]
    echoed_original = 0      # answer equals a date that is NOT the corrected one but parses
    full = parts = zero = 0
    for a in d["answers"][col]:
        if a["task"] != "cascade": continue
        g = gold[a["item_id"]]["gold"]["updates"]
        ups = a.get("updates") or {}
        right = 0
        for name, iso in g.items():
            depth = ORDER.index(name) + 1 if name in ORDER else 1
            ok = (P(ups.get(name)) == P(iso))
            per_hop[depth][1] += 1
            per_hop[depth][0] += ok
            right += ok
        n = len(g)
        full += (right == n); zero += (right == 0); parts += (0 < right < n)
    print(f"\n[{col}] cascade items: full={full} partial={parts} zero={zero}")
    for h in (1,2,3):
        c,t = per_hop[h]
        if t: print(f"   hop {h} (event #{h} in chain): {c}/{t} = {c/t:.0%} correct")