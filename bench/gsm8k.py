"""Tiny GSM8K reasoning accuracy probe via the OpenAI chat API.

Loads 100 GSM8K test questions (default), enables the DeepSeek-V4 "thinking"
template, asks the model, then extracts the numeric answer with a simple
regex and compares against the gold label.

Usage
-----
    python gsm8k.py --base-url http://127.0.0.1:8000/v1 \
        --num-samples 100 --out results/gsm8k.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset  # type: ignore


_ANS_RE = re.compile(r"####\s*(-?[\d,\.]+)")
# fallback: pull the last signed decimal from the answer
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _gold(answer: str) -> float:
    m = _ANS_RE.search(answer)
    s = m.group(1) if m else _NUM_RE.findall(answer)[-1]
    return float(s.replace(",", ""))


def _predict(text: str) -> float | None:
    # Prefer a line like "Answer: 42" or "**42**" at end
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def _ask(url: str, model: str, question: str, timeout: int,
         thinking: bool, max_tokens: int) -> dict:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system",
             "content": "You are a careful math tutor. Solve the problem "
                        "and finish with 'Answer: <number>'."},
            {"role": "user", "content": question},
        ],
    }
    if thinking:
        body["chat_template_kwargs"] = {"thinking": True}
    t0 = time.perf_counter()
    r = requests.post(f"{url.rstrip('/')}/chat/completions",
                      json=body, timeout=timeout)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    j = r.json()
    msg = j["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "usage": j.get("usage", {}),
        "elapsed": elapsed,
    }


def run(args) -> None:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    f = out.open("a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "idx", "gold", "pred", "correct", "tokens",
            "reasoning_tokens", "elapsed_s", "content_trim",
        ])
    correct = 0
    for i, row in enumerate(ds):
        gold = _gold(row["answer"])
        try:
            r = _ask(args.base_url, args.model, row["question"],
                     args.timeout, args.thinking, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:03d}] ERROR {exc!r}")
            continue
        pred = _predict(r["content"])
        ok = pred is not None and abs(pred - gold) < 1e-6
        if ok:
            correct += 1
        rt = r["usage"].get("reasoning_tokens") or 0
        tot = r["usage"].get("completion_tokens") or 0
        writer.writerow([
            i, gold, pred if pred is not None else "", int(ok),
            tot, rt, f"{r['elapsed']:.2f}",
            r["content"].replace("\n", " ")[:200],
        ])
        f.flush()
        print(f"[{i:03d}] gold={gold:<10g} pred={pred!s:<10s} "
              f"{'OK' if ok else '  '}  elapsed={r['elapsed']:.1f}s "
              f"ct={tot} rt={rt}")
    f.close()
    if len(ds):
        print(f"\nGSM8K: {correct}/{len(ds)} correct "
              f"({100*correct/len(ds):.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get(
        "BENCH_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get(
        "BENCH_MODEL", "deepseek-v4-flash"))
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default="results/gsm8k.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
