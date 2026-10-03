#!/usr/bin/env bash
# Whole-model long-prompt prefill A/B of two ninfer_bench binaries, alternating rounds.
# Needs the whole GPU (refuses with less than 26 GiB free: the model is 21.5 GB and BF16 KV for
# 64k keys another 4.3 GB). Prompts longer than the bundled 65536-token corpus use a tiled copy.
#
#   tools/wavefill/long-prefill-ab.sh ARTIFACT BASE_BIN NEW_BIN [kv] [lengths] [rounds]
#   tools/wavefill/long-prefill-ab.sh qwen3_8_27b_nvfp4.ninfer old/ninfer_bench build/bench/ninfer_bench bf16 32768,65536 2
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ART="$1"; BASE="$2"; NEW="$3"
KV="${4:-bf16}"; LENGTHS="${5:-32768,65536}"; ROUNDS="${6:-2}"
OUT="${OUT:-$ROOT/build/wavefill-ab}"
CORPUS="$OUT/corpus_262144.ids"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -lt 26000 ]; then
  echo "only ${free_mib} MiB of GPU memory free; stop other GPU processes first" >&2
  exit 2
fi
mkdir -p "$OUT"
if [ ! -f "$CORPUS" ]; then
  python3 - "$ROOT/bench/fixtures/bench_corpus.ids" "$CORPUS" <<'PY'
import sys
ids = open(sys.argv[1]).read().split()
open(sys.argv[2], "w").write(" ".join((ids * (262144 // len(ids) + 1))[:262144]) + "\n")
PY
fi
for round in $(seq "$ROUNDS"); do
  for tag in base new; do
    bin="$BASE"; [ "$tag" = new ] && bin="$NEW"
    csv="$OUT/long_prefill_${KV}_${tag}_${round}.csv"
    "$bin" --weights "$ART" --corpus "$CORPUS" -p "$LENGTHS" -r 2 --warmup 1 --kv-dtype "$KV" \
      --spec mtp --draft-tokens 3 -o csv --output-file "$csv" >/dev/null
    echo "round $round $tag:"; cut -d, -f1,3,30,37 "$csv"
  done
done
