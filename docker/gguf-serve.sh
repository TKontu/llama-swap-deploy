#!/bin/sh
# Generic GGUF server, shared by BOTH images (Dockerfile.bonsai and Dockerfile.llamacpp).
# This build's llama-server has no HTTPS, so we download files with the `hf` CLI (handles
# HTTPS + gated repos + Xet), then serve the LOCAL files with llama-server. Driven by env:
#   GGUF_REPO   (required)  e.g. empero-ai/Qwythos-9B-v2-GGUF
#   GGUF_FILE   (required)  e.g. Qwythos-9B-v2-Q4_K_M.gguf
#   GGUF_CTX    (optional, default 32768)  PER-SLOT context
#   GGUF_PARALLEL (optional, default 1)    concurrent request slots
#   GGUF_PORT   (optional, default 8080)
# Optional extras (all default to unset = previous single-file, single-GPU behaviour):
#   GGUF_MMPROJ       vision projector filename in the same repo   -> --mmproj
#   GGUF_DRAFT        speculative drafter filename in same repo    -> -md ... -ngld 99
#   GGUF_DRAFT_MAX    draft tokens per step                        -> --draft-max
#   GGUF_SPLIT_MODE   layer|row|none, for multi-GPU                -> -sm
#   GGUF_TENSOR_SPLIT proportion per GPU, e.g. "1,1"               -> -ts
#   GGUF_CACHE_TYPE   KV cache quant, e.g. q8_0                    -> --cache-type-k/-v
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

# Fetch each requested file once. Naming files explicitly (rather than --include) is
# deliberate: hf's --include is ignored when positional names are present, which has
# silently skipped the main model before — see docker/bonsai-serve.sh.
for f in "$GGUF_FILE" "$GGUF_MMPROJ" "$GGUF_DRAFT"; do
    [ -n "$f" ] || continue
    if [ ! -f "$DIR/$f" ]; then
        echo "gguf-serve: downloading $GGUF_REPO :: $f -> $DIR"
        hf download "$GGUF_REPO" "$f" --local-dir "$DIR"
    fi
done

CTX="${GGUF_CTX:-32768}"
NP="${GGUF_PARALLEL:-1}"
TOTAL_CTX=$((CTX * NP))

# Speculative decoding: offload all drafter layers to GPU too, else it runs on CPU and
# is slower than not drafting at all.
SPEC=""
if [ -n "$GGUF_DRAFT" ]; then
    SPEC="-md $DIR/$GGUF_DRAFT -ngld 99"
    [ -n "$GGUF_DRAFT_MAX" ] && SPEC="$SPEC --draft-max $GGUF_DRAFT_MAX"
fi

echo "gguf-serve: llama-server -m $DIR/$GGUF_FILE (ctx ${CTX}/slot x ${NP} slots = ${TOTAL_CTX} total, port ${GGUF_PORT:-8080})"
echo "gguf-serve: mmproj=${GGUF_MMPROJ:-none} draft=${GGUF_DRAFT:-none} split=${GGUF_SPLIT_MODE:-single} ts=${GGUF_TENSOR_SPLIT:-auto} kv=${GGUF_CACHE_TYPE:-f16}"

# shellcheck disable=SC2086
exec llama-server -m "$DIR/$GGUF_FILE" \
    --host 0.0.0.0 --port "${GGUF_PORT:-8080}" \
    -ngl 99 -fa on -c "$TOTAL_CTX" --parallel "$NP" --jinja \
    ${GGUF_MMPROJ:+--mmproj "$DIR/$GGUF_MMPROJ"} \
    ${GGUF_SPLIT_MODE:+-sm "$GGUF_SPLIT_MODE"} \
    ${GGUF_TENSOR_SPLIT:+-ts "$GGUF_TENSOR_SPLIT"} \
    ${GGUF_CACHE_TYPE:+--cache-type-k "$GGUF_CACHE_TYPE" --cache-type-v "$GGUF_CACHE_TYPE"} \
    $SPEC \
    "$@"
