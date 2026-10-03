#!/usr/bin/env python3
"""Whole-prompt prefill time of Qwen3.8-27B from measured prompt-attention kernel times.

    python3 prefill_model.py measurements.csv [--other-ms 88] [--layers 16] [--chunk 1024]

measurements.csv rows: variant,kv,context,ms  (one 1024-token chunk of one attention layer whose
first row sees `context` earlier keys). A prompt of L tokens runs L/chunk chunks; chunk c sees
context c*chunk. Per chunk: other_ms (every non-attention kernel of the chunk, measured 88 ms in a
2048-token nsys trace and independent of context) + layers x attention(context), with attention
linearly interpolated between measured contexts (and extrapolated from the last two points).
"""
import argparse
import collections
import csv


def interp(points, x):
    points = sorted(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y1 + (y1 - y0) * (x - x1) / (x1 - x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--other-ms", type=float, default=88.0)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lengths", default="8192,32768,65536,131072,196608")
    args = ap.parse_args()
    data = collections.defaultdict(list)
    for row in csv.DictReader(open(args.csv)):
        data[(row["variant"], row["kv"])].append((int(row["context"]), float(row["ms"])))
    lengths = [int(x) for x in args.lengths.split(",")]
    kvs = sorted({kv for _, kv in data})
    print(f"{'kv':<5} {'prompt':>7} {'base s':>8} {'new s':>8} {'speedup':>8} {'base tok/s':>11} "
          f"{'new tok/s':>10} {'attn share base->new':>21}")
    for kv in kvs:
        base, new = data[("base", kv)], data[("new", kv)]
        if not base or not new:
            continue
        for length in lengths:
            tb = tn = ab = an = 0.0
            for c in range(length // args.chunk):
                ctx = c * args.chunk
                a0 = args.layers * interp(base, ctx)
                a1 = args.layers * interp(new, ctx)
                tb += args.other_ms + a0
                tn += args.other_ms + a1
                ab += a0
                an += a1
            print(f"{kv:<5} {length:>7} {tb / 1e3:8.2f} {tn / 1e3:8.2f} {tb / tn:7.2f}x "
                  f"{length / tb * 1e3:11.0f} {length / tn * 1e3:10.0f} "
                  f"{100 * ab / tb:9.0f}% -> {100 * an / tn:3.0f}%")


if __name__ == "__main__":
    main()
