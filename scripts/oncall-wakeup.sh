#!/bin/sh
# On-call standby poller for llama-swap.
#
# Keeps ONCALL_MODEL resident whenever the box has genuinely stopped working, without
# ever blocking another model. It relies on two properties of the generated config:
#
#   1. Every pairNN group and every solo entry is `exclusive: true`, so a request to any
#      other model evicts the on-call model immediately. Nothing here has to unload it.
#   2. Symmetrically, the wakeup request below evicts whatever is squatting on the cards.
#      That is why this script never calls POST /api/models/unload.
#
# The on-call model carries `ttl: 0` (never idle-unload) so it stays put once loaded.
#
# Trigger: BOTH 3090s below IDLE_PCT utilisation for a continuous IDLE_SECONDS. GPU
# utilisation — not llamaswap_gpu_memory_used_bytes — is the signal, because a merely
# resident model keeps memory allocated but does no work, and reclaiming the cards from
# exactly that state is the whole point.
#
# Deliberately does NOT use `persistent: true`: llama-swap defines that as "other groups
# can never unload this group's members", which would starve every other model.
set -eu

LLAMASWAP_URL="${LLAMASWAP_URL:-http://127.0.0.1:9292}"
ONCALL_MODEL="${ONCALL_MODEL:-muse-glimmer}"
IDLE_SECONDS="${IDLE_SECONDS:-3600}"
IDLE_PCT="${IDLE_PCT:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# Match GPUs by UUID, not index: the README warns indices reorder across reboots, and the
# A2000 (id=1) must never be considered here. Defaults are the two 3090s (CARD0, CARD2).
GPU_UUIDS="${GPU_UUIDS:-GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1,GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb}"

quiet=0
fails=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) oncall: $*"; }

# Highest utilisation across the watched GPUs, or "" if metrics are unavailable.
peak_gpu_util() {
    metrics="$(curl -sf -m 10 "$LLAMASWAP_URL/metrics" 2>/dev/null)" || return 1
    echo "$metrics" | awk -v uuids="$GPU_UUIDS" '
        BEGIN { n = split(uuids, want, ","); peak = -1 }
        /^llamaswap_gpu_util_percent/ {
            for (i = 1; i <= n; i++)
                if (index($0, want[i]) > 0 && $NF + 0 > peak) peak = $NF + 0
        }
        END { if (peak >= 0) print peak }
    '
}

log "watching ${LLAMASWAP_URL} — wake '${ONCALL_MODEL}' after ${IDLE_SECONDS}s below ${IDLE_PCT}% GPU"

while :; do
    sleep "$POLL_SECONDS"

    util="$(peak_gpu_util || true)"
    if [ -z "$util" ]; then
        # Metrics unreachable (llama-swap restarting?). Back off, don't reset the counter.
        fails=$((fails + 1))
        backoff=$((POLL_SECONDS * fails))
        [ "$backoff" -gt 900 ] && backoff=900
        log "metrics unavailable (${fails}x) — backing off ${backoff}s"
        sleep "$backoff"
        continue
    fi
    fails=0

    if [ "$(awk -v u="$util" -v t="$IDLE_PCT" 'BEGIN { print (u > t) ? 1 : 0 }')" = "1" ]; then
        [ "$quiet" -gt 0 ] && log "GPU busy (${util}%) — resetting quiet timer"
        quiet=0
        continue
    fi

    quiet=$((quiet + POLL_SECONDS))
    [ "$quiet" -lt "$IDLE_SECONDS" ] && continue

    # Reset first: a failed wake must not retry every poll against a ~19 GB cold start.
    quiet=0

    if curl -sf -m 10 "$LLAMASWAP_URL/running" | grep -q "\"$ONCALL_MODEL\""; then
        log "'$ONCALL_MODEL' already resident — nothing to do"
        continue
    fi

    log "idle ${IDLE_SECONDS}s (peak ${util}%) — waking '$ONCALL_MODEL'"
    if curl -sf -m 900 -X POST "$LLAMASWAP_URL/v1/chat/completions" \
         -H 'Content-Type: application/json' \
         -d "{\"model\":\"$ONCALL_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\".\"}],\"max_tokens\":1}" \
         >/dev/null 2>&1; then
        log "'$ONCALL_MODEL' is on call"
    else
        log "wake failed — will retry after another ${IDLE_SECONDS}s idle"
    fi
done
