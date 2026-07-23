# Custom llama-swap image = stock unified-cuda + the `docker` CLI.
#
# Why: the official `unified-cuda` image is nvidia/cuda:runtime + llama.cpp/whisper/sd
# binaries. It has NO `docker` CLI, but we launch vLLM by having llama-swap run
# `docker run …` against the mounted host Docker socket (Docker-out-of-Docker).
# So we add the CLI. llama.cpp models still run as bundled child processes.
FROM ghcr.io/mostlygeek/llama-swap:unified-cuda

# docker.io provides the `docker` CLI (client only; it talks to the host daemon via the
# mounted /var/run/docker.sock). If you prefer the upstream Docker CE apt repo, swap this.
# curl is needed by the compose healthcheck (curl -f http://127.0.0.1:9292/health).
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Entrypoint/CMD are inherited from the base image; the compose file passes
# --config and --listen.
