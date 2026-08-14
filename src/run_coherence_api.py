#!/usr/bin/env python3
"""run_coherence_api.py -- send emitted coherence prompts to an OpenAI-compatible
API and archive the responses in the exact shape run_coherence.py --score-dir
expects.

Pairs with:
    run_coherence.py --out-dir DIR --emit-prompts DIR/prompts
    run_coherence.py --out-dir DIR --emit-prompts DIR/prompts --single

Reads DIR/prompts/<case>.<question>.txt, writes DIR/raw/<case>.<question>.json:

    {"case_id", "question", "model", "raw", "call_error",
     "prompt_sha", "latency_s", "finish_reason", "usage"}

Same discipline as the rest of the harness:
  - temperature 0, frozen prompts (this script never edits a prompt)
  - every raw archived, including failures
  - call_error recorded per call; the scorer counts them separately from
    model abstentions and self-flags the run if any are nonzero
  - already-written raws are skipped, so a rerun only re-calls failures

USAGE
  export OPENAI_API_KEY=sk-...
  python run_coherence_api.py --dir ../data/experiments/coherence/gpt_indep \\
      --model gpt-5.4

  # different provider / gateway:
  python run_coherence_api.py --dir ... --model ... \\
      --base-url https://your-endpoint/v1

  # see what would be sent, spend nothing:
  python run_coherence_api.py --dir ... --model ... --dry-run

  # resume after failures (default; --redo forces everything):
  python run_coherence_api.py --dir ... --model ...

NOTE ON REASONING MODELS
  Some endpoints reject `temperature` for reasoning-family models. If the
  first call fails with a 400 naming temperature, rerun with --no-temperature
  and SAY SO in the paper: the run then differs from the local ones in one
  stated respect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def parse_stem(stem: str) -> tuple[str, str]:
    """'<case>.<question>' -> (case, question). Case ids contain no dots."""
    case, _, question = stem.rpartition(".")
    if not case:
        raise ValueError(f"cannot split case/question from {stem!r}")
    return case, question


def call_api(base_url: str, api_key: str, model: str, prompt: str,
             timeout: int, use_temperature: bool,
             max_tokens: int | None) -> dict:
    """One call. Returns a record dict; never raises for API-level failures."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_temperature:
        payload["temperature"] = 0
    if max_tokens:
        payload["max_completion_tokens"] = max_tokens

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        choice = (body.get("choices") or [{}])[0]
        return {
            "raw": (choice.get("message") or {}).get("content") or "",
            "call_error": None,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "latency_s": round(time.time() - t0, 2),
        }
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        return {"raw": "", "call_error": f"HTTP {e.code}: {detail}",
                "finish_reason": None, "usage": None,
                "latency_s": round(time.time() - t0, 2)}
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError) as e:
        return {"raw": "", "call_error": f"{type(e).__name__}: {e}",
                "finish_reason": None, "usage": None,
                "latency_s": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="condition dir holding prompts/; raw/ is written here")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls (rate limiting)")
    ap.add_argument("--retries", type=int, default=2,
                    help="retries per call on failure")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--no-temperature", action="store_true",
                    help="omit temperature (some reasoning models reject it)")
    ap.add_argument("--redo", action="store_true",
                    help="re-call prompts whose raw already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    prompt_dir = root / "prompts"
    raw_dir = root / "raw"
    if not prompt_dir.is_dir():
        sys.exit(f"no prompts dir at {prompt_dir} -- run --emit-prompts first")

    prompts = sorted(prompt_dir.glob("*.txt"))
    if not prompts:
        sys.exit(f"no .txt prompts in {prompt_dir}")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        sys.exit(f"{args.api_key_env} not set")

    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prompts
            if args.redo or not (raw_dir / f"{p.stem}.json").exists()]
    print(f"{len(prompts)} prompts, {len(todo)} to call, model={args.model}")
    if args.dry_run:
        for p in todo[:3]:
            case, q = parse_stem(p.stem)
            print(f"  would call {case} / {q}  ({len(p.read_text())} chars)")
        print(f"  ... {len(todo)} total. No calls made.")
        return 0

    infra = 0
    for i, p in enumerate(todo, 1):
        case, question = parse_stem(p.stem)
        prompt = p.read_text()

        rec = None
        for attempt in range(args.retries + 1):
            rec = call_api(args.base_url, api_key, args.model, prompt,
                           args.timeout, not args.no_temperature,
                           args.max_tokens)
            if rec["call_error"] is None:
                break
            if attempt < args.retries:
                wait = 2 ** attempt * 5
                print(f"    retry {attempt + 1} in {wait}s "
                      f"({rec['call_error'][:80]})")
                time.sleep(wait)

        if rec["call_error"]:
            infra += 1

        (raw_dir / f"{p.stem}.json").write_text(json.dumps({
            "case_id": case,
            "question": question,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": None if args.no_temperature else 0,
            "prompt_sha": sha(prompt),
            "raw": rec["raw"],
            "call_error": rec["call_error"],
            "finish_reason": rec["finish_reason"],
            "usage": rec["usage"],
            "latency_s": rec["latency_s"],
        }, indent=1))

        flag = f"  ERROR {rec['call_error'][:60]}" if rec["call_error"] else ""
        trunc = ("  TRUNCATED" if rec["finish_reason"] == "length" else "")
        print(f"  [{i:3d}/{len(todo)}] {case:28s} {question:11s} "
              f"{rec['latency_s']:6.1f}s{flag}{trunc}")

        if args.sleep:
            time.sleep(args.sleep)

    print(f"\ninfrastructure errors: {infra}")
    if infra:
        print("NOTE: nonzero infra errors. Rerun this command to re-call only\n"
              "      the failures, and do not report any rate until it is 0.")
    truncated = sum(1 for f in raw_dir.glob("*.json")
                    if json.loads(f.read_text()).get("finish_reason") == "length")
    if truncated:
        print(f"WARNING: {truncated} responses hit the token cap. A cut-off\n"
              "         response loses its JSON object and scores as an\n"
              "         abstention, not an answer. Raise --max-tokens and redo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
