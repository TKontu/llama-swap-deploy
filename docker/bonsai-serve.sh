#!/bin/sh
# Entrypoint for the Ternary-Bonsai-27B PrismML fork image.
# Auto-downloads the GGUF weights (like the vLLM models do) into the mounted HF
# cache if missing, then launches llama-server with the 27B flag set (vision +
# DSpark speculative + tool calling), matching PrismML-Eng/Bonsai-demo's
# scripts/start_llama_server.sh. Extra args after the image (e.g. --alias) are
# forwarded to llama-server via "$@".
set -e

REPO="prism-ml/Ternary-Bonsai-27B-gguf"
DIR="${BONSAI_MODEL_DIR:-/root/.cache/huggingface}"
SNAP="$DIR/models--prism-ml--Ternary-Bonsai-27B-gguf/snapshots"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-16384}"     # DSpark re-prefills each request → give it room

# Find the Q2_0 ternary model (exclude mmproj/dspark/kv-bias extras).
find_model() {
    for f in "$SNAP"/*/*Q2_0*.gguf; do
        [ -f "$f" ] || continue
        case "$f" in *mmproj*|*dspark*|*kv-bias*) continue ;; esac
        echo "$f"; return 0
    done
    return 1
}

MODEL="$(find_model || true)"
if [ -z "$MODEL" ]; then
    echo "Bonsai: weights not in cache — downloading $REPO (Q2_0 + mmproj + dspark)…"
    echo "        (one-time; ~10 GB into $DIR — first cold-start will be slow)"
    huggingface-cli download "$REPO" \
        --include "*Q2_0*.gguf" "*mmproj*.gguf" "*dspark-Q4_1*.gguf" \
        --cache-dir "$DIR"
    MODEL="$(find_model || true)"
fi
if [ -z "$MODEL" ]; then
    echo "ERROR: Ternary-Bonsai-27B Q2_0 GGUF still not found under $SNAP/*/ after download." >&2
    echo "       Check HF access to $REPO (HF_TOKEN) and disk space." >&2
    exit 1
fi

# Vision projector (27B is a VLM).
MMPROJ=""
for f in "$SNAP"/*/*mmproj*.gguf; do [ -f "$f" ] && MMPROJ="$f" && break; done

# DSpark speculative drafter (optional). --spec-draft-n-max MUST equal the
# drafter's block_size or llama-server assert-crashes; read it from GGUF (else 4).
MD=""
for f in "$SNAP"/*/*dspark-Q4_1*.gguf; do [ -f "$f" ] && MD="$f" && break; done
SPEC=""
if [ -n "$MD" ]; then
    NMAX="$(python3 - "$MD" <<'PY' 2>/dev/null || true
import sys
try:
    import gguf
    r = gguf.GGUFReader(sys.argv[1])
    fld = r.get_field('dspark.dspark.block_size')
    print(int(fld.contents()) if fld else '')
except Exception:
    print('')
PY
)"
    case "$NMAX" in ''|*[!0-9]*) NMAX=4 ;; esac
    SPEC="-md $MD --spec-type draft-dspark --spec-draft-n-max $NMAX -ngld 999 -np 1"
fi

echo "Bonsai serve: model=$(basename "$MODEL") mmproj=$( [ -n "$MMPROJ" ] && basename "$MMPROJ" || echo none ) draft=$( [ -n "$MD" ] && basename "$MD" || echo none ) ctx=$CTX port=$PORT"

# shellcheck disable=SC2086
exec llama-server -m "$MODEL" --host 0.0.0.0 --port "$PORT" \
    -ngl 99 -fa on -c "$CTX" \
    --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0 \
    --jinja \
    ${MMPROJ:+--mmproj "$MMPROJ"} \
    $SPEC \
    "$@"
