#!/usr/bin/env bash
# Smoke-test the GGUF pool members through the exact config path (gguf-serve.sh:
# hf download -> local llama-server). Run on the inference host:
#   curl -fsSL https://raw.githubusercontent.com/TKontu/llama-swap-deploy/main/scripts/smoke_gguf.sh -o /tmp/smoke.sh
#   HF_TOKEN=hf_... bash /tmp/smoke.sh
set -u

CARD_A=GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1   # 3090 #0
CARD_B=GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb   # 3090 #2 (A2000 is off-limits)
IMG=ghcr.io/tkontu/bonsai-llama:latest
PORT=8090
: "${HF_TOKEN:?export HF_TOKEN=hf_... first (needed for gated/Xet repos)}"

# Auto-pick whichever 3090 has the most free VRAM (llama-swap may have a model
# resident on the other one). The smoke bypasses llama-swap, so we just use a free card.
free_of() { nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F', ' -v u="$1" '$1==u{print $2}'; }
FA=$(free_of "$CARD_A"); FB=$(free_of "$CARD_B")
if [ "${FB:-0}" -ge "${FA:-0}" ]; then CARD0=$CARD_B; FREE=$FB; else CARD0=$CARD_A; FREE=$FA; fi
echo ">> using GPU $CARD0 (${FREE} MiB free)"
if [ "${FREE:-0}" -lt 12000 ]; then
  echo "!! WARNING: only ${FREE} MiB free — both 3090s look busy. Free one (e.g. stop a"
  echo "   leftover model container, or POST /unload to llama-swap) and re-run."
fi

smoke() {
  local label="$1" repo="$2" file="$3"
  echo
  echo "==================== $label : $file ===================="
  docker rm -f smoke >/dev/null 2>&1 || true
  if ! docker run -d --name smoke --pull always --gpus "device=$CARD0" \
      -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e GGUF_REPO="$repo" -e GGUF_FILE="$file" -e GGUF_CTX=32768 \
      -v /models/hf-cache:/root/.cache/huggingface -p "${PORT}:8080" \
      --entrypoint gguf-serve.sh "$IMG" --alias "$label" >/dev/null; then
    echo "!! docker run failed"; return
  fi
  echo "loading (first run downloads the GGUF)…"
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    docker ps -q -f name=smoke | grep -q . || { echo "!! container exited during load"; break; }
    sleep 10
  done
  echo "-- health --";     curl -s -m 10 "http://127.0.0.1:${PORT}/health"; echo
  echo "-- completion --"
  curl -s -m 60 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$label"'","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16,"temperature":0}' \
    | head -c 700
  echo
  echo "-- last log --";   docker logs smoke 2>&1 | tail -12
  docker rm -f smoke >/dev/null 2>&1 || true
}

smoke qwythos-v2 empero-ai/Qwythos-9B-v2-GGUF Qwythos-9B-v2-Q4_K_M.gguf
smoke fablevibes tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF Qwen3.6-14B-A3B-FableVibes-Q6_K.gguf

echo
echo ">> done. llama-swap reloads models on demand; nothing to restart."
