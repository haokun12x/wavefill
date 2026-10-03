# NVFP4 down-projection GEMV: the cost of a cold instruction image

The Qwen3.8-27B MLP down projection (NVFP4 5120×17408 plus residual, 56 layers) at one token per
step. Measured on an RTX 5090.

**Symptom.** In a real decode trace the kernel took 35.1 µs against a 31.0 µs pure-read floor.
That looked like poor bandwidth efficiency (83%).

**Cause.** It was not the kernel's memory pipeline. Run back to back, the same kernel took
31.7 µs, 97% of the floor. All 34 K-phases were fully unrolled into a 6520-instruction (102 KiB)
image. In a decode step the kernel runs once per layer between a dozen other kernels, so every
launch fetched its code cold:

| same kernel | time |
|---|---:|
| back to back (warm instruction cache) | 31.7 µs |
| right after the real SwiGLU kernel | 33.4 µs |
| real decode | 35.1 µs |

**Change.** `Nvfp4GemvSchedule` gains a defaulted `PhaseUnroll` (0 = full unroll, so every
existing instance is unchanged). The K=17408 shape uses unroll 4 with direct scale loads: 1248
instructions, 20 KiB. The three conditions above become 31.7 / 32.2 / 32.6 µs.

| | before | after |
|---|---:|---:|
| kernel in real decode (nsys, 7224 launches) | 35.14 µs | 32.58 µs (−7.3%) |
| decode throughput (3 alternating rounds) | 76.92 tok/s | 77.88 tok/s (**+1.25%**) |
| MTP3 decode (T=4 kernel untouched) | 119.50 | 119.41 (noise) |

**Verification.** Output is bit-identical to the previous schedule on the real weights. The six
NVFP4 FP64-oracle ctests pass, the four compute-sanitizer tools report no errors, and only the two
K=17408 GEMV instances change SASS.

**Rejected.** An explicit register prefetch queue (depth 2–6), 16-byte loads per lane, other
CTA shapes, and rolling the T=2…7 SIMT kernels: each was equal or slower. The memory-level
parallelism was never the bottleneck.

**Lesson.** A micro-benchmark that launches one kernel back to back overestimates large,
fully-unrolled kernels. Time them after their real predecessor, or in the real trace.
