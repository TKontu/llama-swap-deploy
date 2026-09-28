#!/usr/bin/env bash
# Run the scheduler / edge / deploy-gate tests against the PINNED llama-swap release (version
# and checksum from the Dockerfile) and the pinned Caddy (version from Dockerfile.edge) running
# the real docker/edge/Caddyfile. Needs python3 with aiohttp; no GPU.
#   bash tests/sim/run.sh [case ...]
# The binary is cached outside the repo (SIM_BIN_DIR), which also works where the checkout's
# filesystem doesn't allow executing files.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Read the pin from the Dockerfile so the tests can't drift from what is deployed.
ver=$(sed -n 's/^ARG LLAMA_SWAP_VERSION=//p' "$repo/Dockerfile")
sha=$(sed -n 's/^ARG LLAMA_SWAP_SHA256=//p' "$repo/Dockerfile")
bindir="${SIM_BIN_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/llama-swap-sim}"
bin="$bindir/llama-swap-v$ver"
if [ ! -x "$bin" ]; then
    mkdir -p "$bindir"
    tgz="$bindir/llama-swap_${ver}.tgz"
    curl -fsSL -o "$tgz" \
        "https://github.com/mostlygeek/llama-swap/releases/download/v${ver}/llama-swap_${ver}_linux_amd64.tar.gz"
    echo "$sha  $tgz" | sha256sum -c -
    tar -xzf "$tgz" -C "$bindir" llama-swap
    mv "$bindir/llama-swap" "$bin"
    rm "$tgz"
fi
"$bin" --version

# Caddy: the version comes from Dockerfile.edge's FROM line; its release checksum (SHA-512) is
# pinned here and must be updated together with that line.
CADDY_VERSION=2.11.4
CADDY_SHA512=8220d1f013b6f27510247b2360c9e0ca9f018feebd82515f07635318b34ff9777ccc8fd0b6e6f2486ce3a33fe389fbb7db12d05baa474f4587509fb4f5ebf1c9
edge_ver=$(sed -n 's/^FROM caddy:\([0-9.]*\)-alpine.*/\1/p' "$repo/Dockerfile.edge")
if [ "$edge_ver" != "$CADDY_VERSION" ]; then
    echo "Dockerfile.edge uses Caddy $edge_ver but run.sh pins $CADDY_VERSION: update CADDY_VERSION + CADDY_SHA512" >&2
    exit 1
fi
caddy="$bindir/caddy-v$CADDY_VERSION"
if [ ! -x "$caddy" ]; then
    tgz="$bindir/caddy_${CADDY_VERSION}.tgz"
    curl -fsSL -o "$tgz" \
        "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz"
    echo "$CADDY_SHA512  $tgz" | sha512sum -c -
    tar -xzf "$tgz" -C "$bindir" caddy
    mv "$bindir/caddy" "$caddy"
    rm "$tgz"
fi
"$caddy" version
exec python3 "$here/run.py" "$bin" "$caddy" "$@"
