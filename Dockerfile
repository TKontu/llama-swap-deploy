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

# Bake the model config into the image (instead of a bind mount). A single-file bind
# mount from a Portainer stack is fragile: if the file isn't present at container-create
# time, Docker auto-creates the source as a *directory* and the mount fails with
# "not a directory". Baking it in makes the container start reliably regardless of how
# the stack is deployed. CI rebuilds the image whenever config.yaml changes, so editing
# models stays a git push (see .github/workflows/build-and-push.yml).
COPY config.yaml /etc/llama-swap/config/config.yaml

# Entrypoint/CMD are inherited from the base image; the compose file passes
# --config and --listen.
