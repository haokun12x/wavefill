#!/usr/bin/env python3
"""Same-workload comparison of OpenAI-compatible LLM servers (NInfer / Wavefill / vLLM / SGLang).

Every engine gets the identical chat requests (same text, same template options, temperature 0,
thinking disabled, one request at a time) over HTTP Chat Completions, so tokenization, template
rendering and HTTP overhead are part of every number, exactly as an agent client sees them.

  cold   : one user message of ~N tokens, max_tokens=1 -> time to first (only) token.
           A unique short header per request defeats prefix caches.
  turns  : a conversation whose history (~H tokens) the server already holds; every turn appends
           the previous assistant reply and a new user message of ~d tokens, max_tokens=1.
           usage.prompt_tokens_details.cached_tokens shows how much was reused.
  decode : short prompt, 512 output tokens streamed -> decode tokens/s (first token excluded).

Text comes from decoding a token-id corpus with the model's own tokenizer, so ~N tokens of text
re-tokenize to ~N tokens; the measured prompt_tokens are recorded.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return time.perf_counter() - t0, data


def stream(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage = [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            ev = json.loads(body)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    stamps.append(time.perf_counter() - t0)
    return stamps, usage


class Text:
    def __init__(self, tokenizer_json, corpus_ids):
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.ids = [int(x) for x in open(corpus_ids).read().split()]
        self.pos = 0

    def take(self, n):
        """Fresh text of ~n tokens (never reused within a run)."""
        if self.pos + n > len(self.ids):
            self.pos = 0
        s = self.tok.decode(self.ids[self.pos:self.pos + n])
        self.pos += n
        return s


def base_payload(model, messages, max_tokens, **extra):
    p = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
         "chat_template_kwargs": {"enable_thinking": False}}
    p.update(extra)
    return p


def cached(usage):
    d = (usage or {}).get("prompt_tokens_details") or {}
    return d.get("cached_tokens")


def run_cold(a, text, url, out):
    res = []
    # global warm-up (JIT, graph capture, autotuning)
    for n in (2048, 8192):
        post(url, base_payload(a.model, [{"role": "user", "content": f"[warmup {n}]\n" + text.take(n)}], 1))
    for n in a.cold:
        for rep in range(a.reps):
            msg = f"[cold {n} {rep} {time.time_ns()}]\n" + text.take(n)
            try:
                dt, data = post(url, base_payload(a.model, [{"role": "user", "content": msg}], 1))
            except urllib.error.HTTPError as e:
                row = dict(test="cold", n=n, rep=rep, error=f"HTTP {e.code}: {e.read()[:300]!r}")
                print(json.dumps(row), flush=True)
                res.append(row)
                break
            u = data.get("usage", {})
            row = dict(test="cold", n=n, rep=rep, seconds=dt, prompt_tokens=u.get("prompt_tokens"),
                       cached=cached(u))
            print(json.dumps(row), flush=True)
            res.append(row)
    out["cold"] = res


def run_turns(a, text, url, out):
    res = []
    for h in a.history:
        for d in a.deltas:
            msgs = [{"role": "user", "content": f"[history {h} {d} {time.time_ns()}]\n" + text.take(h)}]
            _, data = post(url, base_payload(a.model, msgs, 1))          # seed, untimed
            msgs.append({"role": "assistant", "content": data["choices"][0]["message"].get("content") or "好"})
            for t in range(-1, a.turns):
                msgs.append({"role": "user", "content": text.take(d)})
                dt, data = post(url, base_payload(a.model, msgs, 1))
                msgs.append({"role": "assistant",
                             "content": data["choices"][0]["message"].get("content") or "好"})
                u = data.get("usage", {})
                if t < 0:
                    continue
                row = dict(test="turn", history=h, delta=d, turn=t, seconds=dt,
                           prompt_tokens=u.get("prompt_tokens"), cached=cached(u))
                print(json.dumps(row), flush=True)
                res.append(row)
    out["turns"] = res


def run_decode(a, text, url, out):
    res = []
    for rep in range(a.reps + 1):
        msgs = [{"role": "user", "content": "请写一篇关于 GPU 体系结构发展史的长文，至少 3000 字，分章节详细展开。"}]
        extra = {"stream": True, "stream_options": {"include_usage": True}}
        if a.ignore_eos:
            extra["ignore_eos"] = True
        stamps, usage = stream(url, base_payload(a.model, msgs, a.decode_tokens, **extra))
        if rep == 0 or len(stamps) < 2:
            continue
        n = (usage or {}).get("completion_tokens") or len(stamps)
        tps = (n - 1) / (stamps[-1] - stamps[0])
        row = dict(test="decode", rep=rep, ttft=stamps[0], tokens=n, decode_tps=tps)
        print(json.dumps(row), flush=True)
        res.append(row)
    out["decode"] = res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cold", type=lambda s: [int(x) for x in s.split(",")], default=[32768, 65536, 131072])
    p.add_argument("--history", type=lambda s: [int(x) for x in s.split(",")], default=[65536])
    p.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[128, 512, 4096])
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--tests", default="cold,turns,decode")
    a = p.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"
    text = Text(a.tokenizer, a.corpus)
    out = {"engine": a.engine}
    for t in a.tests.split(","):
        {"cold": run_cold, "turns": run_turns, "decode": run_decode}[t](a, text, url, out)
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("summary", a.engine)
    by = {}
    for r in out.get("cold", []):
        if "seconds" in r:
            by.setdefault(r["n"], []).append(r["seconds"])
    for n, v in by.items():
        print(f"  cold {n}: median {statistics.median(v):.3f} s")
    by = {}
    for r in out.get("turns", []):
        by.setdefault((r["history"], r["delta"]), []).append((r["seconds"], r["cached"], r["prompt_tokens"]))
    for k, v in by.items():
        print(f"  turn h={k[0]} d={k[1]}: median {statistics.median(x[0] for x in v) * 1e3:.1f} ms, "
              f"cached {[x[1] for x in v]} of {[x[2] for x in v]}")
    for r in out.get("decode", []):
        print(f"  decode {r['decode_tps']:.1f} tok/s ({r['tokens']} tokens)")


if __name__ == "__main__":
    main()
