# Architecture

## Goal

Serve a **small, fixed set of LLMs** on the `inference` server, with:

- **Concurrent** serving of more than one model.
- A **big model split across both 3090s** (tensor parallel) when it doesn't fit one card
  — e.g. `Qwen3.6-35B-A3B` at ~14 GB on **each** 3090.
- A **smaller model (~7 GB) co-resident on one of those two 3090s**.
- **Reliability first.** A predictable, static layout beats clever dynamic packing.
- **Mixed backends:** vLLM for AWQ/safetensors models; llama.cpp for GGUF-only models.

## Decision: llama-swap (stock), not the bespoke gateway, not a fork

We replace the custom `vlmm-gateway` with **stock llama-swap** + a thin custom image.

### Why

| Dimension | `vlmm-gateway` (old) | llama-swap (chosen) |
|-----------|----------------------|---------------------|
| Core design | Dynamic **VRAM-budget placement engine** | Static **process swapper + proxy** |
| Model→GPU assignment | Automatic (measures footprints, packs, evicts) | **Manual, explicit** per model |
| Backends | vLLM only | **Any** OpenAI/Anthropic server (vLLM, llama.cpp, …) |
| Maturity | Bespoke; we own every bug | Battle-tested, thousands of users, active |
| Failure surface | Placement races, footprint accounting, GGUF auto-select, budget rejects | Mostly eliminated — layout is declared, not inferred |

The gateway's one real differentiator — **automatic VRAM packing** — is exactly the code
that produced nearly every incident during operation (startup memory-profiling race, TP
footprint recorded as a cross-card total, budget-mode over-rejection, GGUF file
mis-selection). For a **fixed handful of models on 2–3 GPUs**, that machinery is
over-engineering. A static layout expressed in config is simpler **and** more reliable,
and it unlocks llama.cpp backends (which the vLLM-only gateway could never serve).

**Don't fork llama-swap.** Forking is only justified to add automatic VRAM packing — which
we've explicitly decided we don't need. If a feature is missing, contribute upstream. The
only customization we carry is a 2-line Dockerfile that adds the `docker` CLI.

## Hardware

| Idx | GPU | VRAM | UUID | PCI |
|-----|-----|------|------|-----|
| 0 | RTX 3090 | 24 GB | `GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1` | `06:10` |
| 1 | RTX A2000 | 12 GB | `GPU-690062e6-be81-ab00-ebd3-7181cafcea4a` | `06:11` |
| 2 | RTX 3090 | 24 GB | `GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb` | `06:1B` |

### Interconnect — the dominant constraint

- **No NVLink** (`nvidia-smi nvlink -s` → all links inactive; A2000 has no NVLink).
- `nvidia-smi topo -m` → **`PIX`** between all three (single PCIe switch, no host-bridge
  hop) — the best PCIe topology, but **still PCIe**, not NVLink.

**Implication:** a tensor-parallel model does an all-reduce **every layer** over PCIe.
That's already a tax on a single TP=2 model. Running **two TP-active models at once** makes
them contend for the same PCIe link → catastrophic decode slowdown (observed ~21 tok/s →
~2 tok/s). This is **physics, not orchestration** — llama-swap cannot fix it. Mitigations:
one model per interconnect-domain when hot, or add an **NVLink bridge** to the 3090s.

## Deployment topology

```
Portainer stack
└── container: llama-swap   (custom image: unified-cuda + docker CLI)
      • network_mode: host            → binds :9292, reaches vLLM at 127.0.0.1:<PORT>
      • runtime: nvidia               → for llama.cpp child processes
      • mounts: /var/run/docker.sock  → spawn vLLM as SIBLING containers (DooD)
                /models/hf-cache       → model weights
                ./config.yaml          → model + swap definitions
      │
      ├── spawns (docker run) ──▶ vllm/vllm-openai container  (per vLLM model, on demand)
      │                              • --gpus '"device=<uuid>"' , -p <PORT>:8000
      └── runs (child process) ──▶ llama-server               (per GGUF model, bundled in image)
```

### Image delivery (CI → GHCR → Portainer)

Portainer **cannot build** an image from a Git-repository stack (the `build:` directive is
unsupported there). So `.github/workflows/build-and-push.yml` builds the `Dockerfile` on
GitHub Actions and pushes it to **GHCR**; the compose file references it via `image:`
(`LLAMA_SWAP_IMAGE`). `config.yaml` is **baked into the image** (`COPY` in the Dockerfile),
not bind-mounted: a single-file bind mount from a Portainer stack fails when the file isn't
present at container-create time (Docker auto-creates the source as a directory →
"not a directory" mount error). Model edits stay GitOps — a `config.yaml` change triggers
the CI rebuild, and Portainer re-pulls the new image.

### Why these choices

- **Custom image (unified-cuda + `docker` CLI).** The stock `unified-cuda` image is
  `nvidia/cuda:runtime` + llama.cpp/whisper/sd binaries — it has **no `docker` CLI**.
  vLLM is run by having llama-swap execute `docker run …`, which needs the CLI **inside**
  the container. So: `FROM …:unified-cuda` + `apt-get install docker.io`.
- **Docker-out-of-Docker (socket mount).** vLLM is a heavy Python runtime; we don't embed
  it. llama-swap launches `vllm/vllm-openai` as **sibling** containers via the host Docker
  socket. (The old gateway used the same pattern — proven on this host.)
- **`network_mode: host`.** With sibling vLLM containers publishing `-p <PORT>:8000`,
  llama-swap reaches them at `127.0.0.1:<PORT>` (its default `proxy` form), and itself
  serves on host `:9292`. Simplest reliable wiring.
- **GPU pinning by UUID.** Indices can reorder across reboots; UUIDs are stable.

## Concurrency model (llama-swap `groups`)

llama-swap decides what runs together via **groups** (default engine) or the newer
**matrix** DSL. We use groups:

- `swap: false` → members of the group stay loaded **together**.
- `exclusive: false` → loading a member does **not** unload other groups.
- `persistent: true` → other groups can **never** evict this group.

The co-load set (`35B` + a small model) is one such group. Everything else can be an
on-demand model that swaps normally, or its own group.

## GPU / VRAM placement strategy (manual, explicit)

llama-swap does **not** account VRAM (it only reads `nvidia-smi` for metrics/UI). We size
by hand — once, in config — instead of the gateway guessing every launch.

Target layout for the primary co-load pair:

| Card | Resident | `--gpu-memory-utilization` | ≈ VRAM |
|------|----------|----------------------------|--------|
| 3090 #2 (`094f1ca3`) | `Qwen3.6-35B-A3B` rank 1 (TP) | 0.58 | ~14 GB |
| 3090 #0 (`a8c640ca`) | `Qwen3.6-35B-A3B` rank 0 (TP) **+** small model | 0.58 + 0.30 | ~14 + ~7 = 21 GB |
| A2000 (`690062e6`) | (free / GGUF or a tiny model) | — | — |

Rule: on any **shared** card, the sum of the co-resident models'
`--gpu-memory-utilization` must stay **< 1.0** (leave headroom, e.g. ≤ 0.90 total). vLLM
grabs `util × total` up front, so correct sizing prevents the "No available memory for the
cache blocks" failure.

> Alternative layout to consider (see TODO): put the small/GGUF model on the **A2000** so
> the 35B owns both 3090s exclusively — trades the small model's speed for the big model's,
> and removes shared-card contention.

## Backend matrix

| Model class | Backend | How llama-swap runs it |
|-------------|---------|------------------------|
| AWQ / safetensors (most models) | **vLLM** `v0.26.0` | `cmd: docker run … vllm/vllm-openai …` (DooD) |
| GGUF, mainstream arch | llama.cpp | bundled `llama-server` child process |
| GGUF, exotic (e.g. `Ternary-Bonsai-27B`) | **PrismML llama.cpp fork** | `cmd: docker run …` of a fork image (custom kernels) |
| GGUF, newer arch (e.g. `Muse-Glimmer-30B`) | **mainline llama.cpp, pinned build** | `cmd: docker run …` of `Dockerfile.llamacpp` |

`Ternary-Bonsai-27B` is a hybrid-attention, multimodal, ternary-quantized model built for
a **PrismML fork of llama.cpp** — vLLM 0.25.1 cannot serve it. This is a concrete reason
the backend-agnostic design matters.

### Why there are TWO llama.cpp images

This is the non-obvious bit. They are not redundant and neither can replace the other:

- `Dockerfile.bonsai` builds **PrismML's `prism` fork**, which carries the Q2_0_g128 ternary
  and hybrid-attention CUDA kernels `Ternary-Bonsai-27B` needs. Mainline does not have them.
- `Dockerfile.llamacpp` builds **mainline at a pinned build tag**. The fork's branch head is
  2026-07-31, so it predates any architecture merged after that — `Muse-Glimmer-30B` landed
  in mainline on 2026-08-10 (`ggml-org/llama.cpp#26841`, build `b10353`) and fails on the
  fork with an unknown-architecture error.

Both images share `docker/gguf-serve.sh` as their entrypoint, so moving a model between them
means changing only `image:` in the generated config. Pin the mainline tag rather than
tracking a rolling one, for the same reason the vLLM image is pinned to `v0.26.0`. When
adding a GGUF model, the question to answer first is *which image can actually load it* —
check when its architecture was merged against the fork's branch date.

## Request lifecycle

1. Client → `POST /v1/chat/completions` to llama-swap `:9292` with `"model": "<id>"`.
2. llama-swap checks if `<id>`'s upstream is running; if not (and the group allows), it
   runs the model's `cmd`, waits for `checkEndpoint` (`/health`) to return 200.
3. Request is proxied to `proxy` (`http://127.0.0.1:<PORT>`), response streamed back.
4. `ttl` unloads idle models — seconds since the last request finished, per model:
   5 h for pool/pair members, 10 h for the TP=2 solos (`TTL`/`TTL_SOLO` in
   `gen_pairs_config.py`). Long on purpose: cold starts run minutes, and because every
   group is `exclusive`, requesting another pair evicts the resident one immediately
   regardless of TTL. `cmdStop` (`docker stop ${MODEL_ID}`) tears down cleanly.

## Known issues llama-swap does NOT fix (set expectations)

- **PCIe contention** on concurrent TP (no NVLink) — hardware. Fix = NVLink bridge or
  don't run two TP-heavy models hot together.
- **`Qwen3.6-35B-A3B` Xid 31 (MMU fault) mid-inference** — a vLLM/AWQ-MoE kernel issue,
  not orchestration. Mitigate with **`--enforce-eager`** (disables CUDA graphs, the most
  likely trigger); if it persists, try a newer vLLM image or a different quant.

## Security note

Mounting `/var/run/docker.sock` grants the llama-swap container root-equivalent control of
the host Docker. Acceptable on this single-tenant box; do not expose `:9292` to untrusted
networks without an auth layer (llama-swap supports API keys).

**Resolved 2026-09-05 — HF token disclosure.** `GET /running` returns each model's fully
expanded `docker run` line. While the config passed `-e HF_TOKEN=${env.HF_TOKEN}`, that
echoed the token in plaintext to any unauthenticated caller on the LAN. The token env
lines were removed from `gen_pairs_config.py` (130 lines out of the generated config);
auth now comes from `/models/hf-cache/token`, inside the volume every model already
mounts. See README → *HuggingFace auth*. `/running` was the only leaking route — `/logs`,
`/v1/models`, `/health`, `/api/events` and `/ui` were checked and are clean.

Note this closed the *disclosure*, not the exposure: `:9292` is still unauthenticated on
the LAN, and the socket mount still makes it root-equivalent. Enabling llama-swap's API
key remains worthwhile — it needs an `Authorization` header added to the three `curl`
calls in `scripts/oncall-wakeup.sh` (lines 39, 79, 85) and to any client config.
