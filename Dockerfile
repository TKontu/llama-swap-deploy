# Custom llama-swap image = stock unified-cuda + the `docker` CLI.
#
# Why: the official `unified-cuda` image is nvidia/cuda:runtime + llama.cpp/whisper/sd
# binaries. It has NO `docker` CLI, but we launch vLLM by having llama-swap run
# `docker run …` against the mounted host Docker socket (Docker-out-of-Docker).
# So we add the CLI. llama.cpp models still run as bundled child processes.
#
# Both the base image and the llama-swap binary are PINNED. "Work in progress is never
# interrupted" depends on llama-swap's scheduler: a swap waits for the evicted model's
# in-flight requests, and the TTL skips while any are in flight. tests/sim checks exactly that
# against the same release binary (LLAMA_SWAP_VERSION + SHA256), so bump the pin and re-run
# tests/sim/run.sh together. The unified-cuda tag only carries a date, not the llama-swap
# version, hence the explicit binary install below.
FROM ghcr.io/mostlygeek/llama-swap:unified-cuda-2026-09-26

# docker.io provides the `docker` CLI (client only; it talks to the host daemon via the
# mounted /var/run/docker.sock). If you prefer the upstream Docker CE apt repo, swap this.
# curl is needed by the compose healthcheck (curl -f http://127.0.0.1:9293/health).
# python3-minimal runs scripts/deploy-gate.py (stdlib only).
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io ca-certificates curl python3-minimal \
 && rm -rf /var/lib/apt/lists/*

# v256 (6701d0d) is the build the host ran when the scheduler behaviour was verified.
ARG LLAMA_SWAP_VERSION=256
ARG LLAMA_SWAP_SHA256=557757336cde1a667f0b200d4a84c48c9f5fdf561fd8ae969a2bed9d8c94ec48
RUN set -eux; \
    curl -fsSL -o /tmp/ls.tgz \
      "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz"; \
    echo "${LLAMA_SWAP_SHA256}  /tmp/ls.tgz" | sha256sum -c -; \
    tar -xzf /tmp/ls.tgz -C /usr/local/bin llama-swap; \
    rm /tmp/ls.tgz; \
    llama-swap --version | grep -q "v${LLAMA_SWAP_VERSION} "

# Bake the model config into the image (instead of a bind mount). A single-file bind
# mount from a Portainer stack is fragile: if the file isn't present at container-create
# time, Docker auto-creates the source as a *directory* and the mount fails with
# "not a directory". Baking it in makes the container start reliably regardless of how
# the stack is deployed. CI rebuilds the image whenever config.yaml changes, so editing
# models stays a git push (see .github/workflows/build-and-push.yml).
COPY config.yaml /etc/llama-swap/config/config.yaml

# On-call standby poller. Baked in for the same reason as config.yaml above — the
# docker-compose `oncall-wakeup` service runs this image with an entrypoint override
# rather than bind-mounting the script, so a Portainer Git stack can't mangle it.
COPY scripts/oncall-wakeup.sh /usr/local/bin/oncall-wakeup.sh
RUN chmod +x /usr/local/bin/oncall-wakeup.sh

# Deploy gate (the `deploy-gate` service): redeploys via the Portainer webhook only when nothing
# is in flight. Baked in like the poller above.
COPY scripts/deploy-gate.py /usr/local/bin/deploy-gate.py

# Entrypoint/CMD are inherited from the base image; the compose file passes
# --config and --listen.
