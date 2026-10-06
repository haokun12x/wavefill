#!/usr/bin/env bash
# Same-workload comparison of Wavefill / NInfer / vLLM / SGLang on one GPU.
# Starts each engine in turn (localhost only), waits for /health, runs bench_engines.py, stops it.
#
#   run_all.sh [engine ...]     engines: wavefill ninfer vllm sglang (default: all)
#
# Common settings: Qwen3.8-27B NVFP4 (unsloth checkpoint / its NInfer conversion), BF16 KV,
# 1 concurrent request, 142336-token context, no speculative decoding, text only, prefix cache on.
#
# Required environment:
#   HF_MODEL_DIR     local snapshot of unsloth/Qwen3.8-27B-NVFP4 (tokenizer.json + weights)
#   NINFER_ARTIFACT  the same model converted to .ninfer
#   CORPUS_IDS       whitespace-separated token ids (>= 262144) used as prompt text
#   CLIENT_PY        python with the `tokenizers` package
# Per engine:
#   WAVEFILL_SERVE   ninfer-serve built from this tree
#   NINFER_SERVE     ninfer-serve built from upstream 9e163eee
#   VLLM             vllm executable (0.27.1 was measured); CUDA_HOME with nvcc for FlashInfer JIT
#   SGLANG_PY        python of an SGLang environment (0.5.17 was measured)
# Optional: OUT (results dir), BENCH_ARGS (extra client args), VLLM_KV_BYTES, VLLM_KV_DTYPE.
set -uo pipefail
set -m  # background jobs get their own process group (pgid = $!), so stop() can kill server trees
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/engine-compare-results}
mkdir -p "$OUT"
CTX=142336
CLIENT=(--tokenizer "$HF_MODEL_DIR/tokenizer.json" --corpus "$CORPUS_IDS")

wait_health() { # url pid
  for _ in $(seq 1 900); do
    kill -0 "$2" 2>/dev/null || return 1
    curl -sf "$1/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

stop() { # pid (= process group leader, see set -m)
  kill -INT -- "-$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$1" 2>/dev/null
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1500 ] && break
    sleep 2
  done
}

sample_clock() { # file pid-to-watch
  while kill -0 "$2" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,memory.used --format=csv,noheader >> "$1"
    sleep 0.5
  done
}

run_engine() { # name url model pid extra-client-args...
  local name=$1 url=$2 model=$3 pid=$4; shift 4
  if ! wait_health "$url" "$pid"; then echo "[$name] failed to start"; stop "$pid"; return 1; fi
  echo "[$name] ready $(date +%T)"
  "$CLIENT_PY" "$HERE/bench_engines.py" --url "$url" --model "$model" --engine "$name" \
    "${CLIENT[@]}" --out "$OUT/$name.json" "$@" ${BENCH_ARGS:-} > "$OUT/$name.client.log" 2>&1 &
  local cpid=$!
  sample_clock "$OUT/$name.clock.csv" "$cpid" &
  wait "$cpid"; echo "[$name] client rc=$? $(date +%T)"
  stop "$pid"
}

start_ninfer() { # name binary
  "$2" "$NINFER_ARTIFACT" --host 127.0.0.1 --port 8080 --max-context $CTX --kv-capacity auto \
    --kv-dtype bf16 --max-concurrency 1 --prefill-chunk 1024 > "$OUT/$1.server.log" 2>&1 &
  local pid=$!
  run_engine "$1" http://127.0.0.1:8080 qwen3.8-27b $pid
}

start_vllm() {
  # FlashInfer's FP4 autotuning runs out of memory on a 32 GB card at 2048 batched tokens, so it is
  # disabled and the KV cache size is fixed. The checkpoint's config asks for an FP8 KV cache, which
  # `--kv-cache-dtype auto` follows; BF16 is set explicitly to match the other engines.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$VLLM" serve "$HF_MODEL_DIR" --host 127.0.0.1 --port 8000 \
    --served-model-name qwen3.8-27b-nvfp4 --max-model-len $CTX --gpu-memory-utilization 0.95 \
    --kv-cache-memory-bytes ${VLLM_KV_BYTES:-9663676416} --max-num-batched-tokens 2048 \
    --no-enable-flashinfer-autotune --max-num-seqs 1 --kv-cache-dtype ${VLLM_KV_DTYPE:-bfloat16} \
    --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' --trust-remote-code \
    > "$OUT/vllm.server.log" 2>&1 &
  local pid=$!
  run_engine vllm http://127.0.0.1:8000 qwen3.8-27b-nvfp4 $pid --ignore-eos
}

start_sglang() {
  "$SGLANG_PY" -m sglang.launch_server --model-path "$HF_MODEL_DIR" \
    --served-model-name qwen3.8-27b-nvfp4 --host 127.0.0.1 --port 30000 --tp 1 \
    --context-length $CTX --max-running-requests 1 --mamba-full-memory-ratio 0.5 \
    --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.94 --max-mamba-cache-size 8 \
    --mm-feature-transport cpu --attention-backend flashinfer --fp4-gemm-backend flashinfer_cudnn \
    --kv-cache-dtype bf16 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1 \
    --cuda-graph-backend-prefill disabled --enable-cache-report --trust-remote-code \
    > "$OUT/sglang.server.log" 2>&1 &
  local pid=$!
  run_engine sglang http://127.0.0.1:30000 qwen3.8-27b-nvfp4 $pid --ignore-eos
}

for e in "${@:-wavefill ninfer vllm sglang}"; do
  for x in $e; do
    case $x in
      wavefill) start_ninfer wavefill "$WAVEFILL_SERVE" ;;
      ninfer) start_ninfer ninfer "$NINFER_SERVE" ;;
      vllm) start_vllm ;;
      sglang) start_sglang ;;
    esac
  done
done
