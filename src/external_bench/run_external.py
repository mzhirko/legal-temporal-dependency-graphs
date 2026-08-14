#!/usr/bin/env python3
"""run_external.py -- evaluate on external open benchmarks:
LexTime (legal event ordering), SARA binary/numeric (statutory tax
reasoning), and DeonticBench (airline / housing / USCIS rule reasoning).

Design mirrors src/baseline/run_baseline.py:
  * prompts are FROZEN (identical across models; do not edit between runs)
  * temperature 0 everywhere
  * every raw model response is archived to disk
  * all reported numbers are computed by this script, never hand-counted

Datasets
--------
  lextime       Pairwise temporal ordering of two legal events (BEFORE/AFTER).
                NOTE: github.com/clairebarale/LexTime returns 404 as of
                2026-07-07; obtain the file from the authors (EMNLP 2025
                Findings) and pass it with --data FILE (json/jsonl/csv;
                field names are auto-detected).
  sara_binary   SARA entailment cases (labels 1/0), via the DeonticBench repo.
  sara_numeric  SARA tax computation (integer answer), via DeonticBench.
  airline       DeonticBench airline fee computation (integer answer).
  housing       DeonticBench housing yes/no questions (statutes embedded).
  uscis         DeonticBench USCIS appeals, Accepted/Dismissed.

For the DeonticBench-hosted sets, clone the repo and pass its root:
    git clone https://github.com/guangyaodou/DeonticBench
    python run_external.py --dataset sara_numeric --data DeonticBench \\
        --split smoke --emit-prompts out/prompts_sara

Modes (exactly one)
-------------------
  --emit-prompts DIR      write one prompt file per item; run elsewhere.
  --endpoint URL          run now against an OpenAI-compatible endpoint
                          (Ollama: http://localhost:11434/v1). --model NAME.
                          For api.openai.com set OPENAI_API_KEY.
  --score-dir DIR         score saved responses: <id>.txt or <id>.json
                          (json may be {"response": "..."} or raw string).

LexTime-only extra system
-------------------------
  --system tdg --repo PATH_TO_THESIS_CODE
      Answers ordering questions from the TDG instead of asking the LLM
      directly: extract a TDG from the context with LLMPipeline, resolve
      the two event mentions to facts, and order by (1) parsed dates,
      (2) directed additive/ordering edges. Abstains when unresolved.
      This evaluates the extraction layer on an open benchmark.

Outputs
-------
  <out>/raw/<id>.json      archived raw response per item
  <out>/results.json       per-item rows + summary (machine-readable)
  stdout                   summary table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------
# Frozen prompts. DO NOT EDIT BETWEEN MODELS.
# --------------------------------------------------------------------------

PROMPT_LEXTIME_ENT = """\
You are given a paragraph from a U.S. federal legal complaint and a statement
about the temporal relation between two events. One event may be implicit
(not stated directly, but inferable). Decide whether the statement is true
given the paragraph.

## Paragraph
{context}

## Statement
{statement}

Think step by step about the temporal cues in the paragraph. Then give your
final answer on the last line, exactly in this form:

ANSWER: YES
(if the statement is true / entailed)
or
ANSWER: NO
(if the statement is false / contradicted)
"""

PROMPT_LEXTIME = """\
You are given a passage from a U.S. federal legal complaint and two events
mentioned in it. Decide the temporal order of the two events as described
by the passage.

## Passage
{context}

## Event 1
{event1}

## Event 2
{event2}

Think step by step about the temporal cues in the passage. Then give your
final answer on the last line, exactly in this form:

ANSWER: BEFORE
(if Event 1 happens before Event 2)
or
ANSWER: AFTER
(if Event 1 happens after Event 2)
"""

PROMPT_SARA_BINARY = """\
You are given excerpts of the U.S. Internal Revenue Code and a tax case.
Decide whether the statement is entailed by the statutes and the case facts.

## Statutes
{statutes}

## Case
{text}

## Statement
{question}

Apply the statutes to the facts step by step, showing your arithmetic where
relevant. Then give your final answer on the last line, exactly in this form:

ANSWER: YES
(if the statement is entailed / true)
or
ANSWER: NO
(if the statement is contradicted / false)
"""

PROMPT_NUMERIC = """\
You are given the governing rules and a case. Compute the numeric answer to
the question by applying the rules to the facts.

## Rules
{statutes}

## Case
{text}

## Question
{question}

Apply the rules step by step, showing every arithmetic step. Round to the
nearest whole number. Then give your final answer on the last line, exactly
in this form (digits only, no currency symbol, no commas):

ANSWER: <number>
"""

PROMPT_YESNO = """\
You are given the governing statute text and a question about it.

## Statutes
{statutes}

## Question
{question}

Reason step by step from the statute text only. Then give your final answer
on the last line, exactly in this form:

ANSWER: YES
or
ANSWER: NO
"""

PROMPT_USCIS = """\
You are given the governing eligibility rules and the record of a USCIS
Administrative Appeals Office case. Decide the outcome of the appeal.

## Rules
{statutes}

## Case record
{text}

## Question
{question}

Apply the rules to the record step by step. Then give your final answer on
the last line, exactly in this form:

ANSWER: ACCEPTED
or
ANSWER: DISMISSED
"""

PROMPT_TRACIE = """\
You are given a short story and a statement about the temporal relation
between two events. One event may be implicit (not stated directly, but
inferable). Decide whether the statement is true given the story.

## Story
{context}

## Statement
{hypothesis}

Think step by step about when each event starts and ends in the story.
Then give your final answer on the last line, exactly in this form:

ANSWER: YES
(if the statement is true / entailed)
or
ANSWER: NO
(if the statement is false / contradicted)
"""

ANSWER_RE = re.compile(r"ANSWER\s*[:\-]\s*([^\n]+)", re.IGNORECASE)

PROMPT_CASEHOLD = """\
You are given an excerpt from a U.S. court decision with a citation whose
holding has been masked, followed by five candidate holding statements.
Choose the holding that the cited case actually stands for in this context.

## Excerpt
{context}

## Candidate holdings
A. {e0}
B. {e1}
C. {e2}
D. {e3}
E. {e4}

Reason briefly about which holding fits the citation context. Then give
your final answer on the last line, exactly in this form:

ANSWER: <letter A-E>
"""

UNFAIR_TOS_LABELS = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]

PROMPT_UNFAIR_TOS = """\
You are given a sentence from a consumer Terms of Service contract.
Decide which, if any, of the following potentially-unfair clause types
the sentence contains:

A. Limitation of liability
B. Unilateral termination
C. Unilateral change
D. Content removal
E. Contract by using
F. Choice of law
G. Jurisdiction
H. Arbitration

## Sentence
{text}

Reason briefly. Then give your final answer on the last line, exactly in
this form: either

ANSWER: NONE
(if no unfair clause type applies)
or

ANSWER: <comma-separated letters, e.g. A,G>
"""

PROMPT_SINGLE_LABEL = """\
You are given a legal text and a fixed list of categories. Choose the ONE
category that best describes the text.

## Categories
{labels}

## Text
{text}

Reason briefly. Then give your final answer on the last line, exactly in
this form:

ANSWER: <category number>
"""


# --------------------------------------------------------------------------
# Loaders. Each returns a list of dicts:
#   {id, prompt, gold, kind}  with kind in {binary, numeric, order}
# --------------------------------------------------------------------------

def _read_statute_dir(d: Path) -> str:
    parts = []
    for f in sorted(d.iterdir()):
        if f.is_file():
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def load_deonticbench(root: Path, domain: str, split: str) -> list[dict]:
    """domain in {sara_binary, sara_numeric, airline, housing, uscis-aao}."""
    data_file = root / "data" / domain / f"{split}.json"
    if not data_file.exists():
        sys.exit(f"not found: {data_file}\n"
                 f"clone https://github.com/guangyaodou/DeonticBench and pass "
                 f"its root with --data")
    items = json.loads(data_file.read_text(encoding="utf-8"))

    statutes = ""
    if domain in ("sara_binary", "sara_numeric"):
        statutes = _read_statute_dir(root / "statutes" / "sara")
    elif domain == "airline":
        statutes = _read_statute_dir(root / "statutes" / "airline")

    out = []
    for it in items:
        iid = str(it["id"])
        if domain == "sara_binary":
            prompt = PROMPT_SARA_BINARY.format(
                statutes=statutes, text=it["text"], question=it["question"])
            gold = "YES" if int(it["label"]) == 1 else "NO"
            kind = "binary"
        elif domain in ("sara_numeric", "airline"):
            prompt = PROMPT_NUMERIC.format(
                statutes=statutes, text=it["text"], question=it["question"])
            gold = float(it["label"])
            kind = "numeric"
        elif domain == "housing":
            prompt = PROMPT_YESNO.format(
                statutes=it["statutes"], question=it["question"])
            gold = str(it["label"]).strip().upper()          # YES / NO
            kind = "binary"
        elif domain == "uscis-aao":
            prompt = PROMPT_USCIS.format(
                statutes=it["statutes"], text=it["text"],
                question=it["question"])
            gold = str(it["label"]).strip().upper()          # ACCEPTED / DISMISSED
            kind = "binary"
        else:
            sys.exit(f"unknown DeonticBench domain: {domain}")
        out.append({"id": iid, "prompt": prompt, "gold": gold, "kind": kind,
                    "raw_item": it})
    return out


# LexTime field aliases (the public schema may differ; auto-detect).
_LT_CONTEXT = ("context", "passage", "text", "sentence", "input", "paragraph")
_LT_E1 = ("event1", "event_1", "e1", "arg1", "event_a", "first_event")
_LT_E2 = ("event2", "event_2", "e2", "arg2", "event_b", "second_event")
_LT_LABEL = ("label", "relation", "answer", "temporal_relation", "gold", "order")
_LT_ID = ("id", "idx", "instance_id", "uid")


def _pick(d: dict, names: tuple) -> Optional[str]:
    lower = {k.lower(): k for k in d}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _norm_order_label(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("before", "b", "precedes", "earlier", "<", "0"):
        return "BEFORE"
    if s in ("after", "a", "follows", "later", ">", "1"):
        return "AFTER"
    return None


# Connectives for parsing the official LexTime "TR" statement into
# (event A, relation, event B). Order matters: "is followed by" before "follows".
_LX_CONNECTIVES = [
    (r"\bis\s+not\s+simultaneous\b", None),          # complex negation: abstain in TDG
    (r"\bis\s+simultaneous\s+with\b", "SIM"),
    (r"\bis\s+followed\s+by\b", "BEFORE"),
    (r"\bprecedes\b", "BEFORE"),
    (r"\b(?:happens|occurs|starts|comes|is)\s+before\b", "BEFORE"),
    (r"\bfollows\b", "AFTER"),
    (r"\b(?:happens|occurs|starts|comes)\s+after\b", "AFTER"),
]


def _parse_tr(statement: str):
    for pat, rel in _LX_CONNECTIVES:
        m = re.search(pat, statement, re.IGNORECASE)
        if m:
            if rel is None:
                return None
            return {"event1": statement[:m.start()].strip(" ,."),
                    "event2": statement[m.end():].strip(" ,."),
                    "relation": rel}
    return None


def load_lextime(path: Path) -> list[dict]:
    """Official Zenodo release (record 17157439): Entailment_Dataset.csv with
    columns paragraph, TR, label, pair_type. Falls back to a generic
    context/event1/event2/label schema for other files."""
    if not path.exists():
        sys.exit(f"not found: {path}\n"
                 "Download Entailment_Dataset.csv from "
                 "https://zenodo.org/records/17157439 and pass it here.")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", data.get("instances", []))
    if not rows:
        sys.exit("LexTime file parsed but contains no rows")

    keys = {k.lower(): k for k in rows[0]}
    if "paragraph" in keys and "tr" in keys and "label" in keys:
        # official Zenodo schema
        from collections import Counter
        labels = Counter(str(r[keys["label"]]).strip().lower() for r in rows)
        print(f"[lextime] official schema; label distribution: {dict(labels)}",
              file=sys.stderr)
        out = []
        for i, r in enumerate(rows):
            ctx = r[keys["paragraph"]]
            stmt = str(r[keys["tr"]]).strip()
            lab = str(r[keys["label"]]).strip().lower()
            gold = "YES" if lab == "entailment" else "NO"
            raw_item = {"context": ctx, "statement": stmt,
                        "pair_type": str(r.get(keys.get("pair_type", ""), "")).strip()}
            parsed = _parse_tr(stmt)
            if parsed:
                raw_item.update(parsed)
            out.append({"id": f"lextime_{i:04d}",
                        "prompt": PROMPT_LEXTIME_ENT.format(context=ctx,
                                                            statement=stmt),
                        "gold": gold, "kind": "binary", "raw_item": raw_item})
        unparsed = sum(1 for it in out if "relation" not in it["raw_item"])
        if unparsed:
            print(f"[lextime] {unparsed}/{len(out)} TR statements not parsed "
                  f"into (A, rel, B); the LLM condition still covers them, "
                  f"the TDG condition abstains on them", file=sys.stderr)
        return out

    # generic fallback: context/event1/event2/label with BEFORE/AFTER labels
    kc = _pick(rows[0], _LT_CONTEXT)
    k1 = _pick(rows[0], _LT_E1)
    k2 = _pick(rows[0], _LT_E2)
    kl = _pick(rows[0], _LT_LABEL)
    kid = _pick(rows[0], _LT_ID)
    if not all((kc, k1, k2, kl)):
        sys.exit(f"could not auto-detect LexTime fields; found keys: "
                 f"{sorted(rows[0].keys())}")
    out, skipped = [], 0
    for i, r in enumerate(rows):
        gold = _norm_order_label(r[kl])
        if gold is None:
            skipped += 1
            continue
        out.append({
            "id": str(r[kid]) if kid else f"lextime_{i:04d}",
            "prompt": PROMPT_LEXTIME.format(
                context=r[kc], event1=r[k1], event2=r[k2]),
            "gold": gold, "kind": "order",
            "raw_item": {"context": r[kc], "event1": r[k1], "event2": r[k2]},
        })
    if skipped:
        print(f"[lextime] skipped {skipped} rows with labels outside "
              f"BEFORE/AFTER (kept {len(out)})", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# Model call (OpenAI-compatible; works for Ollama /v1 and api.openai.com)
# --------------------------------------------------------------------------

def _http_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode(errors="replace")[:400]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from None


def call_chat(endpoint: str, model: str, prompt: str,
              max_tokens: int = 4096, timeout: int = 600,
              num_ctx: int = 32768, num_batch: int = 0,
              native_only: bool = False) -> str:
    """Try the OpenAI-compatible /v1 endpoint; if that fails and this looks
    like Ollama, fall back to the native /api/chat (the API run_baseline.py
    uses). Error bodies are surfaced, not swallowed."""
    base = endpoint.rstrip("/")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if "api.openai.com" in base:
        if not key:
            sys.exit("OPENAI_API_KEY not set")
        headers["Authorization"] = f"Bearer {key}"
    elif key:
        headers["Authorization"] = f"Bearer {key}"

    msg = [{"role": "user", "content": prompt}]
    oai_url = base + ("/chat/completions" if base.endswith("/v1")
                      else "/v1/chat/completions" if "api.openai.com" not in base
                      else "/chat/completions")
    errors = []
    # 1) OpenAI-compatible, modern token field then legacy
    for tok_field in (() if native_only else
                      ("max_completion_tokens", "max_tokens")):
        try:
            data = _http_json(oai_url, {
                "model": model, "temperature": 0, "messages": msg,
                tok_field: max_tokens}, headers, timeout)
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            errors.append(str(e))
    # 2) native Ollama /api/chat (only for non-OpenAI endpoints)
    if "api.openai.com" not in base:
        native = base[:-3] if base.endswith("/v1") else base
        try:
            data = _http_json(native + "/api/chat", {
                "model": model, "stream": False, "messages": msg,
                "options": {"temperature": 0, "num_ctx": num_ctx,
                            "num_predict": max_tokens,
                            **({"num_batch": num_batch} if num_batch else {})}},
                headers, timeout)
            return data.get("message", {}).get("content", "") or ""
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(" | ".join(errors[-2:]))


# --------------------------------------------------------------------------
# Answer extraction and scoring
# --------------------------------------------------------------------------

def extract_answer(raw: str, kind: str) -> Optional[Any]:
    m = None
    for m in ANSWER_RE.finditer(raw):      # take the LAST "ANSWER:" line
        pass
    if not m:
        return None
    val = m.group(1).strip()
    if kind == "order":
        return _norm_order_label(val)
    if kind == "binary":
        s = val.strip().strip(".").upper()
        s = s.split()[0] if s.split() else s
        if s in ("YES", "TRUE", "ENTAILED", "ACCEPTED", "ACCEPT", "REMAND",
                 "REMANDED"):
            return "ACCEPTED" if s in ("ACCEPTED", "ACCEPT", "REMAND",
                                       "REMANDED") else "YES"
        if s in ("NO", "FALSE", "CONTRADICTED", "DISMISSED", "DISMISS"):
            return "DISMISSED" if s in ("DISMISSED", "DISMISS") else "NO"
        return None
    if kind == "numeric":
        s = val.replace("$", "").replace(",", "").strip()
        m2 = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m2.group()) if m2 else None
    return None


def score_rows(rows: list[dict]) -> dict:
    n = len(rows)
    call_errors = sum(1 for r in rows
                      if str(r.get("note", "")).startswith("call-error"))
    answered = [r for r in rows if r["pred"] is not None]
    abstained = n - len(answered) - call_errors
    summary: dict[str, Any] = {"n": n, "answered": len(answered),
                               "coverage": round(len(answered) / n, 4) if n else None,
                               "call_errors_infra": call_errors,
                               "unparseable_or_abstained": abstained}
    if call_errors:
        summary["warning"] = ("call_errors_infra > 0: rerun these items "
                              "before reporting (infrastructure, not model, "
                              "failures)")
    kinds = {r["kind"] for r in rows}
    if kinds <= {"binary", "order"}:
        correct = sum(1 for r in answered if r["pred"] == r["gold"])
        summary["accuracy_answered"] = round(correct / len(answered), 4) if answered else None
        summary["accuracy_all"] = round(correct / n, 4) if n else None
        summary["correct"] = correct
    else:  # numeric
        def rel_err(r):
            g, p = float(r["gold"]), float(r["pred"])
            return abs(p - g) / max(1.0, abs(g))
        exact = sum(1 for r in answered if abs(float(r["pred"]) - float(r["gold"])) < 0.5)
        within1 = sum(1 for r in answered if rel_err(r) <= 0.01)
        within10 = sum(1 for r in answered if rel_err(r) <= 0.10)
        summary.update({
            # no official tolerance is pinned in the DeonticBench repo, so
            # all three are reported; state which one you use in the thesis.
            "exact_match": exact,
            "within_1pct": within1,
            "within_10pct": within10,
            "exact_match_rate_all": round(exact / n, 4) if n else None,
        })
    return summary


# --------------------------------------------------------------------------
# TDG system for LexTime (evaluates the extraction layer)
# --------------------------------------------------------------------------

class TDGOrderSystem:
    """Order two events using the thesis TDG pipeline instead of direct QA.

    Resolution: extract facts+relations from the context; match each event
    string to a fact by token overlap; order by parsed dates, else by a
    directed edge/path between the matched facts. Abstain otherwise.
    """

    def __init__(self, repo: Path, model: str, endpoint: Optional[str]):
        sys.path.insert(0, str(repo / "src"))
        try:
            from tdg_pipeline.llm_pipeline import LLMPipeline  # noqa: E501
        except Exception as e:
            sys.exit(f"could not import LLMPipeline from {repo}/src: {e}\n"
                     f"pass --repo pointing at thesis-personal-progress/code "
                     f"and install its requirements (pip install openai).")
        base_url = endpoint if endpoint and "api.openai.com" not in endpoint else None
        self.pipe = LLMPipeline(model=model, base_url=base_url)
        self.endpoint = endpoint
        self.model = model
        if endpoint and not self._endpoint_ok():
            sys.exit(f"extraction endpoint {endpoint} is not answering for "
                     f"{model}. Do NOT run another model concurrently "
                     f"(GPU contention triggers the GGML crash); restart "
                     f"Ollama and rerun this alone.")

    def _endpoint_ok(self) -> bool:
        try:
            call_chat(self.endpoint, self.model, "Say OK.", max_tokens=8,
                      timeout=120)
            return True
        except Exception:
            return False

    @staticmethod
    def _tokens(s: str) -> set:
        toks = re.findall(r"[a-z0-9]+", s.lower())
        stop = {"the", "a", "an", "of", "to", "in", "on", "and", "was",
                "is", "with", "for", "by", "at", "his", "her", "their"}
        out = set()
        for t in toks:
            if t in stop:
                continue
            # crude stem: granted/granting/grants -> grant, unions -> union
            for suf in ("ment", "ing", "ion", "al", "ed", "es", "s"):
                if t.endswith(suf) and len(t) - len(suf) >= 3:
                    t = t[:len(t) - len(suf)]
                    break
            out.add(t)
        return out

    def _scores(self, event: str, facts: list) -> list:
        et = self._tokens(event)
        out = []
        for f in facts:
            ft = self._tokens(" ".join(str(v) for v in f.values()
                                       if isinstance(v, str)))
            out.append(len(et & ft) / len(et) if et and ft else 0.0)
        return out

    def _match_pair(self, e1: str, e2: str, facts: list, thr: float = 0.5):
        """Containment matcher (v3): joint assignment of BOTH events to
        DISTINCT facts, maximizing total containment. v2 matched each
        event independently and could map both onto the same fact (e.g.
        an event phrase quoted inside another fact's sentence), losing
        the edge between the true pair."""
        s1, s2 = self._scores(e1, facts), self._scores(e2, facts)
        best, best_sum = (None, None), -1.0
        for i, f1 in enumerate(facts):
            for j, f2 in enumerate(facts):
                if i == j:
                    continue
                if s1[i] >= thr and s2[j] >= thr and s1[i] + s2[j] > best_sum:
                    best, best_sum = (f1, f2), s1[i] + s2[j]
        return best

    def predict(self, item: dict) -> tuple[Optional[str], str]:
        ri = item["raw_item"]
        self.last_raw = ""
        if "event1" not in ri:                      # unparsed TRACIE hypothesis
            return None, "hypothesis-unparsed"
        # Resume: if this item's graph was already dumped by a previous run,
        # reuse it instead of paying for extraction again. Lets a stalled or
        # killed run pick up where it left off (only the missing items cost
        # LLM calls) rather than restarting all 514 from scratch.
        dd = getattr(self, "dump_dir", None)
        cached = (dd / f"{item['id']}.json") if dd is not None else None
        if cached is not None and cached.exists():
            try:
                rec = json.loads(cached.read_text())
                gd = {"facts": rec.get("facts", []),
                      "dependencies": rec.get("dependencies", [])}
            except Exception:
                gd = None
            if gd is not None:
                facts = gd["facts"]
                rels = gd["dependencies"]
                if not facts:
                    # An empty cached graph is exactly the case we may need to
                    # re-extract: it records that the graph was empty, not WHY,
                    # so replaying it from cache would re-assert the unknown
                    # instead of resolving it. --redo-empty falls through to a
                    # real extraction call for these (and only these) items.
                    if getattr(self, "redo_empty", False):
                        pass
                    else:
                        return None, "empty-graph (cached)"
                else:
                    return self._derive(ri, facts, rels, item)
        g = self.pipe.process(ri["context"], document_id=item["id"],
                              generate_scenarios=False)
        # The extractor's own account of the call. Kept so the harness can
        # archive it and tell a failed call apart from a genuine empty graph.
        self.last_raw = getattr(self.pipe, "last_raw", "") or ""
        status = getattr(self.pipe, "last_status", "ok")
        gd = g.to_dict() if hasattr(g, "to_dict") else g
        facts = gd.get("facts", [])
        rels = gd.get("dependencies", gd.get("relations", []))
        # Persist the extracted graph + the query, so matcher/threshold/edge
        # experiments can be re-run OFFLINE (rematch_tdg.py) in seconds instead
        # of paying for a full 4h re-extraction each time. Extraction is the
        # only expensive, non-deterministic-cost step; everything after it is
        # cheap and should never need re-running the LLM.
        if getattr(self, "dump_dir", None) is not None:
            (self.dump_dir / f"{item['id']}.json").write_text(json.dumps({
                "id": item["id"], "gold": item["gold"], "kind": item["kind"],
                "event1": ri.get("event1"), "event2": ri.get("event2"),
                "relation": ri.get("relation"),
                "pair_type": ri.get("pair_type", ""),
                "facts": facts, "dependencies": rels,
            }, indent=1), encoding="utf-8")
        if not facts:
            # Three different findings, previously all reported as
            # "empty-graph":
            #   extract-call-error  -> INFRA. The call failed and the pipeline
            #                          swallowed it. Not a model result. Rerun.
            #   extract-json-error  -> the model answered but emitted
            #                          unparseable JSON. A real model failure,
            #                          but of serialisation, not of reading.
            #   extract-zero-events -> the model read the text and found no
            #                          temporal content. The only one of the
            #                          three that is a genuine abstention.
            # The old guard pinged the endpoint AFTER the fact, which only
            # detects a DEAD endpoint -- a per-call failure leaves it alive and
            # was silently scored as a model abstention.
            if status.startswith("extract-call-error"):
                return None, f"call-error: {status} - infra, rerun"
            if status.startswith("extract-json-error"):
                return None, f"empty-graph-unparseable: {status}"
            if status == "extract-zero-events":
                return None, "empty-graph-no-events"
            if self.endpoint and not self._endpoint_ok():
                return None, ("call-error: extraction endpoint failing "
                              "(GGML crash / GPU contention) - infra, rerun")
            return None, "empty-graph"
        return self._derive(ri, facts, rels, item)

    def _derive(self, ri: dict, facts: list, rels: list,
                item: dict) -> tuple[Optional[str], str]:
        f1, f2 = self._match_pair(ri["event1"], ri["event2"], facts)
        s1 = max(self._scores(ri["event1"], facts), default=0.0)
        s2 = max(self._scores(ri["event2"], facts), default=0.0)
        note = f"matched={bool(f1)},{bool(f2)} best_c={s1:.2f},{s2:.2f}"
        order: Optional[str] = None
        if f1 and f2:
            d1 = f1.get("date_parsed") or f1.get("value") or f1.get("date")
            d2 = f2.get("date_parsed") or f2.get("value") or f2.get("date")
            iso = re.compile(r"^\d{4}-\d{2}-\d{2}")
            if d1 and d2 and iso.match(str(d1)) and iso.match(str(d2)) \
                    and str(d1)[:10] != str(d2)[:10]:
                order = "BEFORE" if str(d1)[:10] < str(d2)[:10] else "AFTER"
                note += f" by-date {d1}|{d2}"
            else:
                id1 = f1.get("id") or f1.get("fact_id")
                id2 = f2.get("id") or f2.get("fact_id")
                adj = {}
                for r in rels:
                    adj.setdefault(r.get("from_id"), set()).add(r.get("to_id"))
                def reachable(a, b):
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
                if reachable(id1, id2):
                    order, note = "BEFORE", note + " by-path"
                elif reachable(id2, id1):
                    order, note = "AFTER", note + " by-path"
        if order is None:
            return None, note + (" unresolved-mention" if not (f1 and f2)
                                 else " no-date-no-edge")
        if item["kind"] == "order":                 # generic: order IS the answer
            return order, note
        # entailment items (tracie, official lextime): compare derived order
        # against the stated relation. SIM statements cannot be affirmed from
        # a directed order; the TDG abstains on them.
        rel = ri.get("relation")
        if rel == "SIM":
            return None, note + " simultaneous-statement"
        return ("YES" if order == rel else "NO"), note + f" derived={order}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

DOMAIN_ALIASES = {"uscis": "uscis-aao", "uscis-aao": "uscis-aao",
                  "sara_binary": "sara_binary", "sara_numeric": "sara_numeric",
                  "airline": "airline", "housing": "housing"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["lextime", "tracie", "sara_binary", "sara_numeric",
                             "airline", "housing", "uscis"])
    ap.add_argument("--data", required=True,
                    help="lextime: dataset file; tracie: repo root or .txt; others: DeonticBench repo root")
    ap.add_argument("--split", default="smoke", choices=["smoke", "hard"],
                    help="DeonticBench split (default smoke: 5 items)")
    ap.add_argument("--limit", type=int, default=0, help="cap item count")
    ap.add_argument("--out", default="out/external", help="output directory")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-prompts", metavar="DIR")
    mode.add_argument("--endpoint", metavar="URL",
                      help="OpenAI-compatible /v1 endpoint")
    mode.add_argument("--score-dir", metavar="DIR")
    ap.add_argument("--model", help="model name for --endpoint / --system tdg")
    ap.add_argument("--system", choices=["llm", "tdg"], default="llm",
                    help="tdg (lextime only): order events via the thesis "
                         "TDG pipeline")
    ap.add_argument("--repo", help="thesis code root (for --system tdg)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--num-ctx", type=int, default=32768,
                    help="Ollama context window (native fallback)")
    ap.add_argument("--num-batch", type=int, default=0,
                    help="Ollama batch size (native API). Set e.g. 128 to "
                         "work around the GGML_SCHED_MAX_SPLIT_INPUTS crash "
                         "on Gemma models with long prompts.")
    ap.add_argument("--native", action="store_true",
                    help="skip the /v1 attempt; call Ollama /api/chat "
                         "directly (needed for --num-batch to apply)")
    ap.add_argument("--seed", type=int, default=20240517,
                    help="deterministic shuffle seed applied to lextime/tracie "
                         "BEFORE --limit, so --system llm and --system tdg see "
                         "the IDENTICAL subset. Set 0 to disable shuffling "
                         "(reproduces the old positional-slice behaviour; not "
                         "recommended - the official LexTime CSV is sorted by "
                         "label, so slicing yields a single-class subset).")
    ap.add_argument("--no-dump-graphs", dest="dump_graphs", action="store_false",
                    help="tdg only: by default the extracted graph for every "
                         "item is saved to <out>/graphs/ so matcher/threshold "
                         "experiments can be re-run offline with rematch_tdg.py "
                         "(no 4h re-extraction). Pass this to disable.")
    ap.set_defaults(dump_graphs=True)
    ap.add_argument("--redo-empty", action="store_true",
                    help="re-extract items whose CACHED graph is empty instead "
                         "of replaying the empty result. An empty cached graph "
                         "records that extraction produced nothing, not why; "
                         "with the raw response now archived, re-running only "
                         "those items tells a swallowed call-error (infra) "
                         "apart from a genuine no-temporal-content abstention. "
                         "Costs one LLM call per empty item; every non-empty "
                         "graph is still served from cache.")
    ap.add_argument("--tdg-fallback", choices=["none", "llm"], default="none",
                    help="tdg only: on a genuine abstention (not an infra "
                         "call-error), back off to the raw LLM on the same "
                         "prompt. Produces a TDG-where-grounded / LLM-elsewhere "
                         "hybrid so accuracy_all is head-to-head comparable "
                         "with the LLM baseline (forced-answer framing).")
    args = ap.parse_args()

    if args.dataset == "lextime":
        items = load_lextime(Path(args.data))
    elif args.dataset == "tracie":
        items = load_tracie(Path(args.data))
    else:
        items = load_deonticbench(Path(args.data),
                                  DOMAIN_ALIASES[args.dataset], args.split)
    # Deterministic shuffle BEFORE limiting. The official LexTime CSV is sorted
    # with every `entailment` row first, so items[:limit] off the raw file is an
    # all-YES subset on which "accuracy" is only the model's YES-rate, not skill.
    # A pinned seed makes --system llm and --system tdg draw the IDENTICAL subset
    # (each item carries its own id+gold, so the two runs stay aligned).
    if args.dataset in ("lextime", "tracie") and args.seed:
        import random as _random
        _random.Random(args.seed).shuffle(items)
        print(f"shuffled {len(items)} items with seed={args.seed} "
              f"(disable with --seed 0)")
    if args.limit:
        items = items[:args.limit]
    print(f"loaded {len(items)} items from {args.dataset}")

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    if args.emit_prompts:
        pdir = Path(args.emit_prompts)
        pdir.mkdir(parents=True, exist_ok=True)
        for it in items:
            (pdir / f"{it['id']}.txt").write_text(it["prompt"], encoding="utf-8")
        manifest = [{"id": it["id"], "gold": it["gold"], "kind": it["kind"]}
                    for it in items]
        (pdir / "_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {len(items)} prompts to {pdir} (+ _manifest.json). "
              f"Score the responses later with --score-dir.")
        return

    if args.system == "tdg":
        if args.dataset not in ("lextime", "tracie"):
            sys.exit("--system tdg is only defined for lextime/tracie")
        if not (args.repo and args.model):
            sys.exit("--system tdg needs --repo and --model")
        tdg = TDGOrderSystem(Path(args.repo), args.model, args.endpoint)
        tdg.redo_empty = args.redo_empty
        if args.dump_graphs:
            tdg.dump_dir = out / "graphs"
            tdg.dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"dumping extracted graphs to {tdg.dump_dir} "
                  f"(reuse offline with rematch_tdg.py -- no re-extraction)")
        if args.dataset == "tracie":
            print("note: TRACIE 'ends' comparator is approximated by "
                  "start-time order; abstentions are reported, not hidden.")

    rows = []
    for i, it in enumerate(items, 1):
        raw, pred, note = "", None, ""
        fallback_used = False
        if args.score_dir:
            sd = Path(args.score_dir)
            f = next((sd / f"{it['id']}{ext}" for ext in (".txt", ".json")
                      if (sd / f"{it['id']}{ext}").exists()), None)
            if f is None:
                note = "missing-response-file"
            else:
                raw = f.read_text(encoding="utf-8", errors="replace")
                if f.suffix == ".json":
                    try:
                        j = json.loads(raw)
                        raw = j.get("response", j.get("content", raw)) \
                            if isinstance(j, dict) else raw
                    except json.JSONDecodeError:
                        pass
                pred = extract_answer(raw, it["kind"])
        elif args.system == "tdg":
            try:
                pred, note = tdg.predict(it)
            except Exception as e:               # archive failures too
                note = f"tdg-error: {e}"
            # Archive the EXTRACTOR's raw response, not just the parsed graph.
            # Without this an empty graph is undiagnosable after the run: the
            # call may have failed, the JSON may have been unparseable, or the
            # model may genuinely have found nothing, and the three are
            # different findings. Empty when the graph came from cache.
            raw = getattr(tdg, "last_raw", "") or ""
            # Forced-answer framing: back off a genuine abstention to the raw
            # LLM on the same prompt. Never override an infra call-error (that
            # would launder an infrastructure failure into a model prediction).
            if (pred is None and args.tdg_fallback == "llm"
                    and not note.startswith("call-error")
                    and not note.startswith("tdg-error")):
                try:
                    raw = call_chat(args.endpoint, args.model, it["prompt"],
                                    args.max_tokens, num_ctx=args.num_ctx,
                                    num_batch=args.num_batch,
                                    native_only=args.native)
                    fb = extract_answer(raw, it["kind"])
                    if fb is not None:
                        pred, fallback_used = fb, True
                        note += " ->llm-fallback"
                except Exception as e:
                    note += f" | fallback-error: {e}"
        else:
            if not args.model:
                sys.exit("--endpoint needs --model")
            try:
                raw = call_chat(args.endpoint, args.model, it["prompt"],
                                args.max_tokens, num_ctx=args.num_ctx,
                                num_batch=args.num_batch,
                                native_only=args.native)
                pred = extract_answer(raw, it["kind"])
            except Exception as e:
                note = f"call-error: {e}"

        (out / "raw" / f"{it['id']}.json").write_text(json.dumps(
            {"id": it["id"], "gold": it["gold"], "pred": pred,
             "note": note, "fallback": fallback_used, "raw": raw}, indent=1),
            encoding="utf-8")
        rows.append({"id": it["id"], "kind": it["kind"], "gold": it["gold"],
                     "pair_type": it["raw_item"].get("pair_type", ""),
                     "pred": pred, "note": note, "fallback": fallback_used,
                     "match": pred == it["gold"] if it["kind"] != "numeric"
                     else None})
        gm = "?" if pred is None else ("OK" if rows[-1]["match"] or (
            it["kind"] == "numeric" and pred is not None and
            abs(float(pred) - float(it["gold"])) < 0.5) else "X")
        print(f"[{i:>3}/{len(items)}] {it['id']:<40} gold={it['gold']} "
              f"pred={pred} {gm} {note}")

    summary = score_rows(rows)
    pts = sorted({r["pair_type"] for r in rows if r.get("pair_type")})
    if pts:
        summary["by_pair_type"] = {
            pt: score_rows([r for r in rows if r["pair_type"] == pt])
            for pt in pts}
    # Provenance: pin exactly which items were scored so two runs can be proven
    # comparable (or proven NOT comparable) after the fact. split_sha256 is a
    # hash of the (id, gold, pair_type) triples; two runs meant to be compared
    # head-to-head MUST share the same hash.
    import hashlib as _hashlib
    from collections import Counter as _Counter
    split_key = "\n".join(sorted(
        f"{r['id']}\t{r['gold']}\t{r.get('pair_type', '')}" for r in rows))
    gold_dist = dict(_Counter(r["gold"] for r in rows))
    meta = {"data_path": str(Path(args.data).resolve()),
            "n": len(rows), "seed": args.seed, "limit": args.limit,
            "system": args.system, "tdg_fallback": args.tdg_fallback,
            "fallback_used": sum(1 for r in rows if r.get("fallback")),
            "gold_distribution": gold_dist,
            "split_sha256": _hashlib.sha256(split_key.encode()).hexdigest()[:16]}
    if len(gold_dist) < 2 and args.dataset in ("lextime", "tracie"):
        meta["INVALID_SINGLE_CLASS"] = True
        print("\n*** WARNING: this evaluation set has a SINGLE gold class "
              f"{gold_dist}. 'accuracy' on a single-class set is only the "
              "model's base-rate for that class and is NOT a measure of skill "
              "(a constant classifier scores 1.0). Re-run with --seed <n> and "
              "--limit 0 (or a larger limit) to get a balanced set. ***\n")
    result = {"dataset": args.dataset, "split": args.split,
              "mode": ("score-dir" if args.score_dir else
                       "tdg" if args.system == "tdg" else "endpoint"),
              "model": args.model, "meta": meta,
              "summary": summary, "rows": rows}
    (out / "results.json").write_text(json.dumps(result, indent=1),
                                      encoding="utf-8")
    print("\n== summary ==")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print(f"\nresults -> {out/'results.json'}   raw -> {out/'raw'}/")
    if args.dataset == "lextime":
        print("reference: LexTime paper reports up to 80.8% accuracy for "
              "LLMs on implicit-explicit pairs (EMNLP 2025 Findings).")


if __name__ == "__main__":
    main()
    