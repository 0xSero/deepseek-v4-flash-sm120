"""Prefill and decode throughput sweep for an OpenAI-compatible endpoint.

For each target prompt length we:
  * build a prompt of exactly N tokens using the model's HF tokenizer;
  * send it with ``max_tokens=OUTPUT``, ``temperature=0`` and record:
      - prompt_tokens         (echoed by the server)
      - completion_tokens
      - TTFT  (time to first streamed token)
      - total wall time
      - prefill tok/s         = prompt_tokens / TTFT
      - decode  tok/s         = (completion_tokens - 1) / (total - TTFT)

Results are appended to a CSV; stdout is a human-readable table.

Usage
-----
    python throughput.py --base-url http://127.0.0.1:8000/v1 \
        --model deepseek-v4-flash \
        --tokenizer-dir /mnt/llm_models/DeepSeek-V4-Flash-FP8 \
        --output-tokens 64 \
        --context-lengths 100,1000,8000,32000,128000,262144 \
        --out results/throughput.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import requests
from transformers import AutoTokenizer


def _build_prompt(tokenizer, target_tokens: int) -> str:
    """Return a prompt that tokenizes to approximately ``target_tokens`` tokens.

    We repeat a small deterministic passage and then trim by token ids to
    match exactly.  The final user instruction is a short "reply with X"
    request so decode output is deterministic."""
    seed = (
        "The quick brown fox jumps over the lazy dog. "
        "All work and no play makes Jack a dull boy. "
        "In a hole in the ground there lived a hobbit. "
    )
    # Leave 48 tokens of headroom for the final instruction + chat template.
    room = max(target_tokens - 48, 16)
    text_tokens: List[int] = []
    while len(text_tokens) < room:
        text_tokens += tokenizer.encode(seed, add_special_tokens=False)
    text_tokens = text_tokens[:room]
    filler = tokenizer.decode(text_tokens, skip_special_tokens=True)
    # Open-ended instruction so the model actually decodes the full budget
    # rather than stopping after 2 tokens like "OK".
    suffix = (
        "\n\nIgnore the text above. Write a short factual paragraph "
        "about the boiling point of water. Keep writing until you are "
        "asked to stop."
    )
    return filler + suffix


def _stream_once(base_url: str, model: str, prompt: str, max_tokens: int,
                 timeout: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    first_tok_time = None
    completion_tokens = 0
    prompt_tokens = None
    last_reported = None
    with requests.post(url, json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage is not None:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens",
                                              completion_tokens)
                last_reported = usage
            choices = chunk.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if first_tok_time is None:
                        first_tok_time = time.perf_counter()
    t_end = time.perf_counter()
    return dict(
        ttft=(first_tok_time - t0) if first_tok_time else None,
        total=(t_end - t0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        last_usage=last_reported,
    )


def run(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    f = out_path.open("a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "timestamp", "target_ctx", "prompt_tokens", "completion_tokens",
            "ttft_s", "total_s", "prefill_tok_s", "decode_tok_s",
        ])
    f.flush()

    print(f"{'ctx':>8s} {'prompt':>7s} {'out':>5s} {'ttft':>8s} "
          f"{'total':>8s} {'prefill':>10s} {'decode':>9s}")
    for target in args.context_lengths:
        prompt = _build_prompt(tok, target)
        try:
            r = _stream_once(args.base_url, args.model, prompt,
                             args.output_tokens, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"{target:>8d}  ERROR  {exc!r}")
            continue
        prefill = (r["prompt_tokens"] or 0) / r["ttft"] if r["ttft"] else 0.0
        decode = 0.0
        if r["ttft"] is not None and (r["completion_tokens"] or 0) > 1:
            decode = (r["completion_tokens"] - 1) / max(
                r["total"] - r["ttft"], 1e-9)
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"), target, r["prompt_tokens"],
            r["completion_tokens"], f"{r['ttft']:.3f}" if r["ttft"] else "",
            f"{r['total']:.3f}", f"{prefill:.2f}", f"{decode:.2f}",
        ])
        f.flush()
        print(f"{target:>8d} {r['prompt_tokens']!s:>7s} "
              f"{r['completion_tokens']!s:>5s} "
              f"{(r['ttft'] or 0):>8.2f} {r['total']:>8.2f} "
              f"{prefill:>10.2f} {decode:>9.2f}")
    f.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get(
        "BENCH_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get(
        "BENCH_MODEL", "deepseek-v4-flash"))
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--context-lengths", type=lambda s: [int(x) for x in s.split(",")],
                   default=[100, 1000, 8000, 32000, 128000, 262144])
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="results/throughput.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
