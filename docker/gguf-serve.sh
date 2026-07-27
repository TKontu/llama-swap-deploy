#!/bin/sh
# Generic GGUF server for the bonsai image. This build's llama-server has no HTTPS,
# so we download the file with the `hf` CLI (handles HTTPS + gated repos + Xet), then
# serve the LOCAL file with llama-server. Driven by env vars:
#   GGUF_REPO   (required)  e.g. empero-ai/Qwythos-9B-v2-GGUF
#   GGUF_FILE   (required)  e.g. Qwythos-9B-v2-Q4_K_M.gguf
#   GGUF_CTX    (optional, default 32768)  PER-SLOT context
#   GGUF_PARALLEL (optional, default 1)    concurrent request slots
#   GGUF_PORT   (optional, default 8080)
# llama-server's -c is the TOTAL KV cache, split evenly across --parallel slots, so
# we pass ctx*parallel to give each slot the full GGUF_CTX. Unlike vLLM there is no
# paging or preemption here: KV VRAM scales linearly with GGUF_PARALLEL.
# Extra args after the image (e.g. --alias) are forwarded to llama-server via "$@".
set -e

: "${GGUF_REPO:?set GGUF_REPO}"
: "${GGUF_FILE:?set GGUF_FILE}"
BASE="${GGUF_DIR:-/root/.cache/huggingface/gguf}"
DIR="$BASE/$(echo "$GGUF_REPO" | tr '/' '_')"
mkdir -p "$DIR"

if [ ! -f "$DIR/$GGUF_FILE" ]; then
    echo "gguf-serve: downloading $GGUF_REPO :: $GGUF_FILE -> $DIR"
    hf download "$GGUF_REPO" "$GGUF_FILE" --local-dir "$DIR"
fi

CTX="${GGUF_CTX:-32768}"
NP="${GGUF_PARALLEL:-1}"
TOTAL_CTX=$((CTX * NP))

echo "gguf-serve: llama-server -m $DIR/$GGUF_FILE (ctx ${CTX}/slot x ${NP} slots = ${TOTAL_CTX} total, port ${GGUF_PORT:-8080})"
exec llama-server -m "$DIR/$GGUF_FILE" \
    --host 0.0.0.0 --port "${GGUF_PORT:-8080}" \
    -ngl 99 -fa on -c "$TOTAL_CTX" --parallel "$NP" --jinja \
    "$@"
