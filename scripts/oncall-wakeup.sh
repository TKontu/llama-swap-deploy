#!/bin/sh
# On-call standby poller for llama-swap.
#
# Keeps ONCALL_MODEL resident whenever the box has genuinely stopped working, without
# ever blocking another model. It relies on two properties of the generated config's
# matrix router:
#
#   1. A request for any other card-0 model, or any whole-box model, evicts the on-call
#      model immediately. Nothing here has to unload it.
#   2. Symmetrically, the wakeup request below evicts whatever is squatting on card 0 (or
#      a whole-box model); a model on card 2 stays loaded. That is why this script never
#      calls POST /api/models/unload.
#
# ONCALL_MODEL must be the REAL model ID, not an alias: /running reports real IDs, so the
# "already resident" check below would never match an alias and would re-wake every time.
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
ONCALL_MODEL="${ONCALL_MODEL:-c0.muse-glimmer}"
IDLE_SECONDS="${IDLE_SECONDS:-3600}"
IDLE_PCT="${IDLE_PCT:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# Comma-separated model IDs the poller must never evict (RAM-offload MoE; see SPEC-bigmoe.md
# §6). Empty = no-op. gen_config.py keeps the compose value in sync with the config.
BIGMOE_MODELS="${BIGMOE_MODELS:-}"
# Match GPUs by UUID, not index: indices DO reorder (they did on 2026-09-15), and the A2000s
# must never be considered here. Defaults are the two 3090s (CARD0, CARD2).
GPU_UUIDS="${GPU_UUIDS:-GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1,GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb}"

quiet=0
fails=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) oncall: $*"; }

# is_resident <running-json> <model-id>. Fixed-string match on the quoted ID: model IDs
# contain dots, which a regex grep would treat as wildcards.
is_resident() { echo "$1" | grep -qF "\"$2\""; }

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

    # Fail closed: if we cannot see what is loaded, we cannot rule out a bigmoe model.
    if ! running="$(curl -sf -m 10 "$LLAMASWAP_URL/running")"; then
        log "/running unavailable — not waking (retry after another ${IDLE_SECONDS}s idle)"
        continue
    fi

    if is_resident "$running" "$ONCALL_MODEL"; then
        log "'$ONCALL_MODEL' already resident — nothing to do"
        continue
    fi

    # GPU utilisation is not an idle signal for a RAM-offload MoE: while it decodes, the
    # cards mostly wait on host memory and read as idle. Waking here would kill a running
    # generation and throw away a ~145 GiB load for a 17 GB standby. See SPEC-bigmoe.md §6.
    bigmoe=""
    for m in $(echo "$BIGMOE_MODELS" | tr ',' ' '); do
        if is_resident "$running" "$m"; then bigmoe="$m"; break; fi
    done
    if [ -n "$bigmoe" ]; then
        log "'$bigmoe' is resident (BIGMOE_MODELS) — not evicting it"
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
