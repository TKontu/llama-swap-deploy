#!/bin/sh
# Entrypoint for the Ternary-Bonsai-27B PrismML fork image.
# Discovers the GGUF weights in the mounted HF cache and launches llama-server
# with the 27B flag set (vision + DSpark speculative + tool calling), matching
# PrismML-Eng/Bonsai-demo scripts/start_llama_server.sh. Any extra args passed
# after the image (e.g. --alias) are forwarded to llama-server via "$@".
set -e

DIR="${BONSAI_MODEL_DIR:-/root/.cache/huggingface}"
SNAP="$DIR/models--prism-ml--Ternary-Bonsai-27B-gguf/snapshots"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-16384}"     # DSpark re-prefills each request → give it room

# --- main weights: the Q2_0 ternary model (exclude mmproj/dspark/kv-bias) -----
MODEL=""
for f in "$SNAP"/*/*Q2_0*.gguf; do
    [ -f "$f" ] || continue
    case "$f" in *mmproj*|*dspark*|*kv-bias*) continue ;; esac
    MODEL="$f"; break
done
if [ -z "$MODEL" ]; then
    echo "ERROR: Ternary-Bonsai-27B Q2_0 GGUF not found under $SNAP/*/." >&2
    echo "       Download prism-ml/Ternary-Bonsai-27B-gguf into the HF cache (see README)." >&2
    exit 1
fi

# --- vision projector (27B is a VLM) -----------------------------------------
MMPROJ=""
for f in "$SNAP"/*/*mmproj*.gguf; do [ -f "$f" ] && MMPROJ="$f" && break; done

# --- DSpark speculative drafter (optional) -----------------------------------
MD=""
for f in "$SNAP"/*/*dspark-Q4_1*.gguf; do [ -f "$f" ] && MD="$f" && break; done
SPEC=""
if [ -n "$MD" ]; then
    # --spec-draft-n-max MUST equal the drafter's block_size or llama-server
    # assert-crashes on the first draft round. Read it from the GGUF (fallback 4).
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
