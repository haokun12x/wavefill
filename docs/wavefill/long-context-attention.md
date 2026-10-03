# Long-context prompt attention: cluster split-KV and FP16 PV accumulation

Wavefill's main change. It makes long-prompt prefill of Qwen3.8-27B on an RTX 5090 **1.12× faster
at 64k tokens and 1.18× at 128k end to end** (BF16 KV cache, MTP3). The causal prompt attention
kernels themselves get **1.29–1.40×** faster at 64k context. Every number below was measured on
one RTX 5090 (sm_120, 170 SMs, 600 W power limit, CUDA 13.3). Raw outputs are in
[`results/`](results/).

## Why attention dominates long prefill

NInfer prefills in 1024-token chunks. Chunk *c* attends to all `c × 1024` earlier keys, so the
attention cost of a chunk grows linearly with context while everything else in the chunk (the
GEMMs, the 48 Gated-DeltaNet layers) stays at about 88 ms. At 64k context the 16 full-attention
layers already cost more than the rest of the model, and at 256k they make up about 80% of
prefill.

## What limited the original kernels

**Measured tensor-core peaks** (`tools/wavefill/mma_peak.cu`, dense, at boost clock):

| `mma.sync` | TFLOP/s |
|---|---:|
| BF16/FP16 → FP32 accumulate (m16n8k16) | 251 |
| FP16 → **FP16 accumulate** | **501** |
| INT8 → INT32 (m16n8k32) | 997 |
| E4M3 → FP32 / FP16 accumulate | 499 / 998 |

QK and PV each carry half of the logical FLOPs. That puts the original kernels' ceiling at 251
TFLOP/s for the BF16 cache and 401 for the INT8 cache. They reached 150–172 TFLOP/s at every
context length.

1. **Wave quantization.** Each CTA owns a 64-row query block of one head and occupies a whole SM
   (96 KB of shared memory). A 1024-token chunk is therefore 16 × 24 = 384 CTAs on 170 SMs:
   three waves, the last one 26% full. That is about 25% lost, for every cache format.
2. **PV at half rate.** P·V used the FP32-accumulate MMA, half the throughput of FP16
   accumulation. In the BF16 kernel, 76% of warp stalls sit on the two MMA instructions.

## Change 1 — cluster split-KV (no workspace)

The S CTAs of a thread-block cluster share one tile. Rank *r* processes key blocks
`[r·n/S, (r+1)·n/S)`. After the key loop, the ranks merge through distributed shared memory:

1. Each row owner publishes `{m, l}` (raw score max, row sum relative to it) in its own shared
   memory.
2. `cluster.sync()`. Every thread reads all ranks' `{m, l}` for its rows, scales its accumulator
   by `exp(m_r − m)/l`, and stores it as FP32 in its own shared memory (Q/K/V staging is dead by
   then).
3. `cluster.sync()`. Rank *r* sums slice *r* of the tile over all ranks, in rank order, and
   writes BF16.
4. `cluster.sync()`, so no rank exits while another still reads its shared memory.

No global workspace is involved, so NInfer's startup memory planning does not change. The launcher
picks S on the host. It queries `cudaOccupancyMaxActiveClusters` once per kernel; this card has
11 GPCs and holds 170/85/52/41/30/22/22/19 clusters for S = 1…8. It then minimizes
`waves(S) × (keys per rank + overhead)` using the chunk's execution envelope. At short context
it picks S = 1, the original code path. The measured times match the wave model within 3%:

| S | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16, 64k, 1024 tokens (ms) | 10.81 | **9.08** | 9.66 | 9.14 | 9.52 | 11.02 | 9.50 | 9.77 |
| predicted (waves / S, relative to S=1) | 1 | 0.833 | 0.889 | 0.833 | 0.867 | 1.0 | 0.857 | 0.875 |

## Change 2 — FP16 PV accumulation that cannot overflow

Each 64-key tile accumulates P·V with `mma…f16.f16.f16.f16` and is then folded into the FP32
accumulator. Two choices make this safe:

* **P is taken relative to the tile's own row max and scaled by 2⁻⁶.** The largest P in a tile is
  exactly 2⁻⁶, so 64 keys sum to at most 1. A tile partial is therefore a sub-convex combination of
  V rows, and `|partial| ≤ max|V| ≤ 65504`: overflow is impossible.
* **The fold multiplies by β = exp(m_tile − m_running)** (l likewise), so the normalized result
  is unchanged. A tile far below the running max still has full FP16 relative precision. The
  simpler variant scales P by 2⁻⁶ relative to the *running* max. It was 4% faster, but it pushes
  small probabilities into FP16 subnormals in attention-sink-dominated long contexts, so it was
  rejected.

## Change 3 — BF16 decode kernel K/V staging

The BF16 small-T (decode / MTP verify) kernel was single-buffered: it staged the whole K+V tile,
waited, then computed. It now issues V(j) before QK(j) and K(j+1) before PV(j), so a load is
always in flight. Shared memory and grid are unchanged and the output is bit-identical.

## Results

**Kernel level** (`ninfer_causal_softmax_attention_bench`, append entry = the real prefill
path, CUDA graph, cold cache; 24 q / 4 kv heads, D = 256, one 1024-token chunk):

| KV cache | context 0 | 8k | 32k | 64k | TFLOP/s at 64k |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.13× | 1.35× | 1.39× | 11.01 → 7.85 ms, **1.40×** | 151 → 212 |
| INT8 | 1.06× | 1.25× | 1.29× | 10.89 → 8.42 ms, **1.29×** | 153 → 197 |
| FP8 | 1.11× | 1.30× | 1.34× | 9.89 → 7.34 ms, **1.35×** | 168 → 226 |

BF16 at 128k and 192k context: 1.41× and 1.40×. The other registered geometry (16 q / 2 kv heads)
gets 1.27–1.42× at 64k. BF16 decode attention with T=4 (MTP3 verify) at 64k: 200.7 → 190.5 µs.

**End to end** (`ninfer_bench`, Qwen3.8-27B NVFP4 artifact, BF16 KV, MTP3, 1024-token chunks,
before/after binaries alternated):

| prompt | before | Wavefill | speed-up |
|---:|---:|---:|---:|
| 8k | 10111 tok/s | 10150 tok/s | 1.00× |
| 32k | 4.25 s | 4.01 s | **1.06×** |
| 64k | 11.29 s | 10.11 s | **1.12×** |
| 128k | 33.8 s | 28.6 s | **1.18×** |
| MTP3 decode after a 64k prompt | 193.4 tok/s | 193.8 tok/s | +0.2% |

**Why end to end is less than the kernel numbers suggest: the power cap.** During a 64k prefill
the card sits at its 600 W limit (`SW power cap`). With the faster attention the median SM clock
drops from 2745 to 2595 MHz. Inside the engine the attention therefore speeds up 1.32× instead of
1.40×, and kernels that did not change at all get 3% slower
([`results/power_cap_64k_prefill.txt`](results/power_cap_64k_prefill.txt)). Kernel micro-benchmarks
run for milliseconds and never reach that steady state. At the same power the prefill finishes
sooner, so energy per prefill drops by about 9% (588 W × 11.50 s → 600 W × 10.28 s).

## Verification

* NInfer's FP64-oracle op test (`ninfer_softmax_attention_test`) passes. It now has four cases
  that split: history not aligned to a key block, a ragged last query block, CUDA-graph replay,
  both geometries, BF16/INT8 caches, append and cached entries. During development the whole test
  was also run with S forced to 2, 3, 5 and 8.
* Per-case error statistics over 611 cases: FP16 accumulation raises relative L2 error by about
  1.5% on average (BF16 1.562e-3 → 1.586e-3) and stays within 62% of every criterion (61% before).
* A stress harness, `tools/wavefill/attn_numerics`, covers activation regimes the op test does
  not reach. It runs the original kernel and the new one against FP64 at 64k keys with uniform,
  peaked, attention-sink, V-outlier (|V| up to 3·10⁴) and sink + outlier inputs. The new kernel is
  within 2% of the original everywhere, has no non-finite values, and is slightly *more* accurate
  with a sink ([`results/numerics_ctx65536.txt`](results/numerics_ctx65536.txt)).
* compute-sanitizer memcheck, racecheck, synccheck and initcheck: 48 runs, 0 errors, 0 hazards
  ([`results/sanitizers.txt`](results/sanitizers.txt)).
* Whole-engine SASS diff (`tools/wavefill/sass-diff.py`): only the 18 prompt-attention and
  112 BF16 small-T kernel instances changed; the other 2912 functions are identical.
* Full `ctest`: every test that ran passes, including the real-model `score_real` and
  `vision_workspace` tests. Not run: the real-model tests that need other artifacts, and
  `prefix_real`, which hard-codes Qwen3.6 prompt token counts and fails on the Qwen3.8 artifact
  without these changes too.

## Tried and rejected

| idea | result |
|---|---|
| 8 warps per CTA, 128 rows × 32 keys, so each scheduler has two warps | no faster than 4 × (64×64); the warps run in lock step, so softmax still does not overlap the MMAs |
| interleave 2 or 4 n-tile pairs in the FP16 PV chains | +1.6%, but spills 32 B |
| fold β in FP16 (`HMUL2`) | +0.4% for an extra FP16 rounding |
| running-max × 2⁻⁶ FP16 accumulation | +4%, precision loss with attention sinks |

## What is left

* **INT8 prompt kernel.** Its QK costs a quarter of the BF16 tensor time, yet it is slower than
  BF16 (8.42 vs 7.85 ms at 64k). Its producer/consumer phases are serialized by barriers, and
  overlapping QK(j+1) with PV(j) needs a second FP16 V stage that does not fit in 99 KB. That is
  a rewrite, worth up to ~2× on INT8 prompt attention.
* **BF16 kernel**, tensor pipe ~64% busy: needs FA3-style warp ping-pong to overlap softmax with
  MMA.
* **Decode at T=1**: about 13% below the 1.70 TB/s read ceiling, plus a separate 5 µs reduce
  kernel per layer.

## Reproduce

```bash
cmake --build build --target ninfer_causal_softmax_attention_bench ninfer_softmax_attention_test ninfer_bench
build/bench/ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 \
    --kv-dtype bf16 --tokens 1024 --context 0,8192,32768,65536 --execution graph --cache cold
NINFER_OP_REPORT_STATS=1 build/tests/ninfer_softmax_attention_test
tools/wavefill/build.sh && tools/wavefill/attn_numerics --context 65536 --splits 1,2,8
tools/wavefill/long-prefill-ab.sh qwen3_8_27b_nvfp4.ninfer <before>/ninfer_bench build/bench/ninfer_bench bf16 32768,65536 2
```
