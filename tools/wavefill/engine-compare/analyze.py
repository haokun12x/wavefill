#!/usr/bin/env python3
"""Model-based analysis of bench_engines.py results.

  analyze.py [--block engine=tokens ...] <dir> [engine ...]

cold   : least-squares fit t(N) = a*N + b*N^2 per engine (prompt_tokens as N). The N^2 term is the
         causal attention of the 16 full-attention layers: 393216 FLOP per (query, key) pair and N^2/2
         pairs, so the effective attention throughput is R = 196608 / b.
turns  : recomputed tokens per turn = prompt_tokens - cached_tokens, and a fit latency = c + k * recomputed.
         vLLM does not report cached_tokens; with `--block vllm=784` its reuse is modelled as whole blocks
         of the previous prompt (hybrid models run its prefix cache in mamba "align" mode, and the
         attention block size is raised to the mamba state page: 784 tokens for BF16 KV, 1568 for FP8).
decode : ms/token against the 11.4 ms bandwidth floor (19.35 GB per token at 1.70 TB/s).
"""
import json
import statistics
import sys

import numpy as np

FLOOR_MS = 19.35e9 / 1.70e12 * 1e3


def load(d, e):
    return json.load(open(f"{d}/{e}.json"))


def cold_fit(rows):
    pts = [(r["prompt_tokens"], r["seconds"]) for r in rows if "seconds" in r]
    by = {}
    for n, t in pts:
        by.setdefault(n, []).append(t)
    ns = sorted(by)
    ts = [statistics.median(by[n]) for n in ns]
    A = np.array([[n, n * n] for n in ns], float)
    (a, b), *_ = np.linalg.lstsq(A, np.array(ts), rcond=None)
    fit = A @ np.array([a, b])
    return ns, ts, a, b, fit


def recomputed(rows, block):
    """Tokens prefilled per turn: reported, or modelled from whole-block reuse of the previous prompt."""
    out = []
    for i, r in enumerate(rows):
        if r.get("cached") is not None:
            out.append(r["prompt_tokens"] - r["cached"])
        elif block:
            same = [x for x in rows if x["delta"] == r["delta"]]
            growth = same[1]["prompt_tokens"] - same[0]["prompt_tokens"]
            prev = r["prompt_tokens"] - growth
            out.append(r["prompt_tokens"] - prev // block * block)
        else:
            out.append(None)
    return out


def main():
    args = sys.argv[1:]
    blocks = {}
    while "--block" in args:
        i = args.index("--block")
        e, b = args[i + 1].split("=")
        blocks[e] = int(b)
        del args[i:i + 2]
    d = args[0]
    engines = args[1:] or ["wavefill", "ninfer", "vllm", "sglang"]
    print("== cold prefill: t(N) = a N + b N^2")
    print(f"{'engine':10s} {'a us/tok':>9s} {'R_attn TFLOP/s':>15s}  per-point (measured / fit)")
    for e in engines:
        ns, ts, a, b, fit = cold_fit(load(d, e).get("cold", []))
        pts = "  ".join(f"{n // 1024}k {t:.2f}/{f:.2f}" for n, t, f in zip(ns, ts, fit))
        r = 196608 / b / 1e12 if b > 0 else float("nan")
        print(f"{e:10s} {a * 1e6:9.1f} {r:15.0f}  {pts}")
    print("\n== turns: latency vs recomputed tokens (fit: c + k * recomputed)")
    for e in engines:
        rows = load(d, e).get("turns", [])
        rec = recomputed(rows, blocks.get(e))
        out = [f"d={r['delta']}: {r['seconds'] * 1e3:.0f} ms" + (f" ({n})" if n is not None else "")
               for r, n in zip(rows, rec)]
        print(f"{e:10s} " + "; ".join(out))
        if rows and None not in rec:
            A = np.array([[1, n] for n in rec], float)
            y = np.array([r["seconds"] * 1e3 for r in rows])
            (c, k), *_ = np.linalg.lstsq(A, y, rcond=None)
            print(f"{'':10s} c = {c:.0f} ms, k = {k * 1e3:.0f} us/token, r = {np.corrcoef(A[:, 1], y)[0, 1]:.3f}, "
                  f"max |residual| = {np.abs(y - A @ [c, k]).max():.0f} ms")
    print(f"\n== decode vs {FLOOR_MS:.1f} ms/token floor")
    for e in engines:
        rows = load(d, e).get("decode", [])
        if not rows:
            continue
        tps = statistics.median(r["decode_tps"] for r in rows)
        ms = 1e3 / tps
        print(f"{e:10s} {tps:6.1f} tok/s  {ms:5.1f} ms/token  {FLOOR_MS / ms * 100:5.1f}% of floor  "
              f"excess {ms - FLOOR_MS:4.1f} ms  ttft {statistics.median(r['ttft'] for r in rows) * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
