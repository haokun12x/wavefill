#!/usr/bin/env bash
# Build the Wavefill harnesses against this tree's kernel headers with NInfer's nvcc flags.
# -rdc=true matters: NInfer compiles its ops relocatable, and without it register allocation
# differs (the BF16 prompt kernel spills).
#
#   tools/wavefill/build.sh [target...]     default: every harness
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NVCC="${NVCC:-nvcc}"
FLAGS=(-forward-unknown-to-host-compiler -I"$ROOT/include" -I"$ROOT/src" -I"$HERE"
  -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]"
  -lineinfo -rdc=true)
for target in "${@:-mma_peak cluster_occupancy attn_numerics attn_timing bw_variants pdl_chain}"; do
  for t in $target; do
    "$NVCC" "${FLAGS[@]}" "$HERE/$t.cu" -o "$HERE/$t"
    echo "built $HERE/$t"
  done
done
