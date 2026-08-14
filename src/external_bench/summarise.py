#!/usr/bin/env python3
"""One table over every benchmark run in out/. No scrolling.

    python src/external_bench/summarise.py            # all runs in out/
    python src/external_bench/summarise.py out/*_v2   # just the reruns

For each run: n, coverage, the accuracy metric that fits the task (exact-match
for numeric, accuracy for binary), the majority baseline (so a score means
something), and call_errors_infra with a loud flag when it is missing or nonzero
-- because a run whose infra field is absent predates the instrumentation fix
and its abstention counts are not trustworthy (our own rule).
"""
import json
import sys
from collections import Counter
from pathlib import Path


def majority(rows) -> float:
    g = Counter(r["gold"] for r in rows)
    return max(g.values()) / len(rows) if rows else 0.0


def main() -> int:
    args = sys.argv[1:]
    if args:
        dirs = [Path(a) for a in args]
    else:
        base = Path("out")
        dirs = sorted(p for p in base.iterdir()
                      if (p / "results.json").exists()) if base.is_dir() else []
    if not dirs:
        sys.exit("no runs found (looked in out/ or the globs you passed)")

    print(f"{'run':30s} {'n':>4s} {'cov':>5s} {'metric':>8s} {'maj':>5s} "
          f"{'vs':>5s} {'infra':>6s}")
    print("-" * 74)
    flags = []
    for d in dirs:
        try:
            j = json.loads((d / "results.json").read_text())
        except Exception as e:
            print(f"{d.name:30s}  unreadable: {e}"); continue
        s = j.get("summary", {})
        rows = j.get("rows", [])
        n = s.get("n", len(rows))
        cov = s.get("coverage")
        maj = majority(rows) if rows else None

        # pick the metric the task actually uses
        if "exact_match" in s:
            em = s["exact_match"]
            metric = f"{em}/{s.get('answered', n)}ex"
            score = em / s.get("answered", n) if s.get("answered") else None
            vs = "-"
        else:
            score = s.get("accuracy_all")
            metric = f"{score:.3f}" if score is not None else "-"
            vs = ("above" if (score is not None and maj is not None and score > maj)
                  else "below" if score is not None else "-")

        infra = s.get("call_errors_infra", "MISSING")
        if infra == "MISSING":
            flags.append((d.name, "infra field absent -> predates the fix; "
                                  "abstention counts not trustworthy"))
            infra_s = "MISS!"
        elif infra:
            flags.append((d.name, f"{infra} infra errors -> rerun before reporting"))
            infra_s = str(infra)
        else:
            infra_s = "0"

        cov_s = f"{cov:.2f}" if isinstance(cov, (int, float)) else "-"
        maj_s = f"{maj:.3f}" if maj is not None else "-"
        print(f"{d.name:30s} {n:>4} {cov_s:>5} {metric:>8s} {maj_s:>5} "
              f"{vs:>5s} {infra_s:>6s}")

    if flags:
        print("\n!! attention:")
        for name, why in flags:
            print(f"   {name}: {why}")
    print("\nexact-match is the honest metric for numeric tasks (a deadline or a "
          "fee is right or it is not);\nwithin-10% is a secondary column, pinned "
          "BEFORE looking, never chosen after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
