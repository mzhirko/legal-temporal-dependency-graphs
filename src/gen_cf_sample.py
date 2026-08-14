#!/usr/bin/env python3
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--k", required=True, type=int)
    ap.add_argument("--gold", required=True, choices=["timely", "late"])
    ap.add_argument("--anchor-date", required=True)
    ap.add_argument("--replace", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--window", type=int, default=350)
    a = ap.parse_args()
    text = open(a.text, encoding="utf-8").read()
    pos = text.find(a.anchor_date)
    if pos < 0:
        sys.exit(f"anchor date {a.anchor_date!r} not found in {a.text}")
    lo, hi = max(0, pos - a.window), min(len(text), pos + a.window)
    snippet = text[lo:hi]
    pairs = [r.split("=", 1) for r in a.replace]
    for old, new in pairs:
        snippet = snippet.replace(old, new)
    for old, _ in pairs:
        if old in snippet:
            sys.exit(f"replacement failed: {old!r} still present")
    caps = [w for w in snippet.split() if w.istitle() and len(w) > 3]
    print("% check for missed names:", sorted(set(caps))[:15], file=sys.stderr)
    ell_l = "[\\ldots] " if lo > 0 else ""
    ell_r = " [\\ldots]" if hi < len(text) else ""
    print(f"""One perturbed instance, anonymised. Case {a.case.replace('_', chr(92)+'_')},
offset $k={a.k:+d}$ days; the shifted anchor date is {a.anchor_date} and
the recomputed gold verdict is \\emph{{{a.gold}}}. The passage around the
shifted anchor:

\\begin{{quote}}\\small
{ell_l}{snippet.strip()}{ell_r}
\\end{{quote}}""")

if __name__ == "__main__":
    main()
