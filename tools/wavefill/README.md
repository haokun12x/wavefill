# Wavefill tools

Harnesses behind [`docs/wavefill/`](../../docs/wavefill/). They compile this tree's kernel
headers with NInfer's nvcc flags (`build.sh`, needs CUDA 13.x and an sm_120 GPU).

| file | what it does |
|---|---|
| `build.sh [target...]` | builds the `.cu` harnesses (`NVCC=` overrides the compiler) |
| `mma_peak.cu` | dense `mma.sync` throughput per shape and accumulator (BF16/FP16/INT8/E4M3) |
| `cluster_occupancy.cu` | clusters of S whole-SM CTAs the GPU holds at once, S = 1..16 |
| `attn_numerics.cu` | the upstream BF16 prompt kernel vs this tree's (S = 1, 2, 4, 8) vs an FP64 oracle on sampled rows, for uniform / peaked / attention-sink / V-outlier inputs |
| `attn_timing.cu` | kernel time of the upstream vs this tree's BF16 prompt attention up to 192k context (K/V generated in place); `--wide` times the rejected 8-warp prototypes |
| `prompt_bf16_head.cuh` | the upstream (NInfer `9e163eee`) BF16 prompt kernel, renamed into `ninfer::ops::ref` |
| `prompt_bf16_wide.cuh` | rejected prototype with rows/keys per tile as parameters (its 32-key variants index the 64-token page table wrongly above 16k context: timing only) |
| `prefill_model.py` | whole-prompt prefill time from per-chunk attention times + 88 ms of other work per 1024-token chunk |
| `sass-diff.py` | which device functions differ between two binaries, compared as SASS hash multisets |
| `long-prefill-ab.sh` | whole-model long-prompt prefill A/B of two `ninfer_bench` binaries (needs ≥ 26 GiB free GPU memory) |
