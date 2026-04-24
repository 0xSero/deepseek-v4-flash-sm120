"""Minimal Needle-In-A-Haystack (NIAH) probe at several context lengths.

For each target context length we
  1. Build a "haystack" of innocuous repeated text;
  2. Inject a unique factual needle sentence at a given fractional depth;
  3. Ask the model the question and compare the answer to the ground truth
     with case-insensitive substring match.

Results (depth x context => pass/fail + raw answer) are written to CSV.

Usage:
    python niah.py --base-url http://127.0.0.1:8000/v1 \
        --tokenizer-dir /mnt/llm_models/DeepSeek-V4-Flash-FP8 \
        --context-lengths 1000,8000,32000,128000 \
        --depths 0.1,0.5,0.9 \
        --out results/niah.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import List

import requests
from transformers import AutoTokenizer

NEEDLE_TEMPLATE = (
    "The secret code for the {city} archive is {code}. "
    "Keep this code safe; nobody else knows it."
)

QUESTION_TEMPLATE = (
    "What is the secret code for the {city} archive, "
    "as stated earlier in the passage? Reply with the code only."
)

CITIES = [
    "Lisbon", "Kyoto", "Reykjavik", "Cairo", "Montreal", "Dakar", "Hanoi",
    "Oslo", "Bogotá", "Perth", "Nairobi", "Tbilisi",
]


def _make_code(rng: random.Random) -> str:
    return f"{rng.randint(10_000, 99_999)}-" f"{rng.choice(['ALPHA','BETA','GAMMA','DELTA','OMEGA','SIGMA'])}"


def _haystack_text(tokenizer, target_ctx: int, needle_sentence: str,
                   depth: float) -> str:
    """Build a passage of ~target_ctx tokens with the needle at fractional
    position ``depth`` (0.0 = very beginning, 1.0 = very end)."""
    seed = (
        "In the long history of exploration, careful record-keeping has "
        "mattered more than flashy discovery. Quiet librarians, archivists, "
        "and registrars kept the notes that later mapmakers trusted. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    needle_ids = tokenizer.encode(" " + needle_sentence + " ",
                                  add_special_tokens=False)
    # Leave ~96 tokens for the question + chat template overhead.
    body = max(target_ctx - len(needle_ids) - 96, 64)
    pre = max(int(body * depth), 0)
    post = max(body - pre, 0)
    filler: List[int] = []
    while len(filler) < max(pre, post):
        filler += seed_ids
    pre_ids = filler[:pre]
    post_ids = filler[:post]
    full = pre_ids + needle_ids + post_ids
    return tokenizer.decode(full, skip_special_tokens=True)


def _ask(base_url: str, model: str, passage: str, question: str,
         timeout: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": [
            {"role": "user",
             "content": passage + "\n\n" + question},
        ],
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    j = r.json()
    return {
        "answer": j["choices"][0]["message"]["content"],
        "usage": j.get("usage", {}),
        "elapsed": elapsed,
    }


def run(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                        trust_remote_code=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    f = out.open("a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "timestamp", "target_ctx", "depth", "city", "code",
            "prompt_tokens", "elapsed_s", "pass", "answer_trim",
        ])
    rng = random.Random(args.seed)
    total = 0
    passes = 0
    print(f"{'ctx':>7s} {'dep':>5s} {'city':>10s} {'code':>12s} "
          f"{'pt':>6s} {'sec':>7s}  pass  answer")
    for target in args.context_lengths:
        for depth in args.depths:
            city = rng.choice(CITIES)
            code = _make_code(rng)
            needle = NEEDLE_TEMPLATE.format(city=city, code=code)
            question = QUESTION_TEMPLATE.format(city=city)
            passage = _haystack_text(tok, target, needle, depth)
            try:
                r = _ask(args.base_url, args.model, passage, question,
                         args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"{target:>7d} {depth:>5.2f} {city:>10s} "
                      f"{code:>12s}   ERROR {exc!r}")
                continue
            ans = r["answer"].strip()
            ok = code.lower() in ans.lower()
            if ok:
                passes += 1
            total += 1
            writer.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"), target, depth,
                city, code, r["usage"].get("prompt_tokens"),
                f"{r['elapsed']:.2f}", int(ok), ans[:120],
            ])
            f.flush()
            print(f"{target:>7d} {depth:>5.2f} {city:>10s} {code:>12s} "
                  f"{r['usage'].get('prompt_tokens'):>6d} "
                  f"{r['elapsed']:>7.2f}   "
                  f"{'Y' if ok else 'n'}   {ans[:60]!r}")
    f.close()
    if total:
        print(f"\nNIAH: {passes}/{total} passed ({100*passes/total:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get(
        "BENCH_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get(
        "BENCH_MODEL", "deepseek-v4-flash"))
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--context-lengths",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1000, 8000, 32000, 128000])
    p.add_argument("--depths", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.1, 0.5, 0.9])
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/niah.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
