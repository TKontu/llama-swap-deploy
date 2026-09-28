#!/bin/sh
# Entrypoint for the ComfyUI image (Dockerfile.comfyui). llama-swap starts it with `docker run`
# and appends the per-instance flags (--enable-manager etc.) as arguments.
#
# Per-instance state lives under COMFYUI_DATA (a host dir mounted per model ID, e.g.
# /fast/comfyui/c2.comfyui): user settings, saved workflows, the comfyui.db SQLite file,
# uploads and outputs. Keeping it per instance means the two instances never share one SQLite
# file. Models are NOT in here; they come from the shared /opt/comfyui/models mount.
set -eu

DATA="${COMFYUI_DATA:-/data}"
# ComfyUI rejects a --user-directory that does not exist yet (is_valid_directory).
mkdir -p "$DATA/user" "$DATA/input" "$DATA/output" "$DATA/temp"

# The container runs on the host network (so the hold can reach llama-swap at 127.0.0.1:9292),
# on the port llama-swap assigned (COMFYUI_PORT=${PORT}). Loopback only: nothing on the LAN
# reaches ComfyUI except through llama-swap.
exec python /opt/comfyui/main.py \
    --listen "${COMFYUI_LISTEN:-127.0.0.1}" --port "${COMFYUI_PORT:-8188}" \
    --user-directory "$DATA/user" \
    --input-directory "$DATA/input" \
    --output-directory "$DATA/output" \
    --temp-directory "$DATA/temp" \
    "$@"
