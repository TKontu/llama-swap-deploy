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

| Idx | Name | VRAM | UUID | PCI | Config label |
|-----|------|------|------|-----|--------------|
| 0 | RTX 3090 | 24 GB | `GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb` | `06:10` | `c2` (`CARD2`) |
| 1 | RTX 3090 | 24 GB | `GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1` | `06:11` | `c0` (`CARD0`) |
| 2 | RTX A2000 | 12 GB | `GPU-689f1c3c-d1f7-f348-29d3-90c12a0b5d43` | `06:1B` | **off-limits** |
| 3 | RTX A2000 | 12 GB | `GPU-690062e6-be81-ab00-ebd3-7181cafcea4a` | `06:1C` | **off-limits** |
| 4 | RTX A2000 | 12 GB | `GPU-037627b2-a49d-77c6-4b97-dc914ce581e9` | `08:0D` | **off-limits** |

**The three A2000s are dedicated to other, non-LLM workloads. This deployment must never use
them** — only the two 3090s. (That is also what the ~4.8 GiB resident on GPU 3 is.)

As of 2026-09-15 (driver 595.84, CUDA 13.2): two more A2000s, and the cards reordered. The
`c0`/`c2` labels are **card identities bound to UUIDs**, named after the indices the 3090s had
at migration. They no longer match `nvidia-smi` indices, and they don't need to — that is
the whole point of pinning by UUID.

The config uses only the two 3090s, by UUID. The A2000s belong to other workloads.

### Interconnect — the dominant constraint

- **No NVLink** (`nvidia-smi nvlink -s` → all links inactive; A2000 has no NVLink).
- `nvidia-smi topo -m` → **`PIX`** between all five, one NUMA node, CPU affinity 0-15
  (single PCIe switch, no host-bridge hop) — the best PCIe topology, but **still PCIe**, not
  NVLink. This is measured **inside the VM**: passthrough topology can be virtualized, and
  GPU 4 sits on a different bus (`08:`). Treat the physical path as unverified until checked
  on the Proxmox host.

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

## Concurrency model (llama-swap `matrix` router)

llama-swap decides what runs together via **groups** (default engine) or the **matrix**
router. We use the matrix router, because the layout is "each 3090 is an independent slot",
and groups cannot express that without enumerating every pair of models.

- Every single-card model is defined **once per 3090**: `c0.<model>` and `c2.<model>`.
- One matrix set, `(any c0 model) & (any c2 model)`, allows every combination, including
  the same model on both cards. Subsets are implied, so one card alone is also fine.
- When a model is requested, the solver evicts the running models outside the cheapest set
  that contains it. With disjoint card slots that is exactly the other model on the same
  card, so **a request never disturbs the other card**.
- Models that need both cards (TP=2 vLLM, `-sm layer` GGUF splits) are in **no set**, which
  llama-swap defines as "can only run alone".

History: this replaced ~35 generated `pairNN` groups (`swap: false, exclusive: true`), which
could only co-load pre-enumerated pairs in a fixed card orientation and evicted both cards on
every switch. The old callsigns remain as aliases for now (README → "Legacy callsigns").

## GPU / VRAM placement strategy (manual, explicit)

llama-swap does **not** account VRAM (it only reads `nvidia-smi` for metrics/UI). We size
by hand — once, in config — instead of the gateway guessing every launch.

Target layout for the primary co-load pair:

| Card | Resident | `--gpu-memory-utilization` | ≈ VRAM |
|------|----------|----------------------------|--------|
| 3090 #2 (`094f1ca3`) | `Qwen3.6-35B-A3B` rank 1 (TP) | 0.58 | ~14 GB |
| 3090 #0 (`a8c640ca`) | `Qwen3.6-35B-A3B` rank 0 (TP) **+** small model | 0.58 + 0.30 | ~14 + ~7 = 21 GB |
| A2000s | (not available — other workloads) | — | — |

Rule: on any **shared** card, the sum of the co-resident models'
`--gpu-memory-utilization` must stay **< 1.0** (leave headroom, e.g. ≤ 0.90 total). vLLM
grabs `util × total` up front, so correct sizing prevents the "No available memory for the
cache blocks" failure.

> (Historical: an earlier note suggested moving small models onto the A2000. That is no longer
> possible — the A2000s are dedicated to non-LLM workloads.)

## Backend matrix

| Model class | Backend | How llama-swap runs it |
|-------------|---------|------------------------|
| AWQ / safetensors (most models) | **vLLM** `v0.26.0` | `cmd: docker run … vllm/vllm-openai …` (DooD) |
| GGUF, mainstream arch | llama.cpp | bundled `llama-server` child process |
| GGUF, exotic (e.g. `Ternary-Bonsai-27B`) | **PrismML llama.cpp fork** | `cmd: docker run …` of a fork image (custom kernels) |
| GGUF, newer arch (e.g. `Muse-Glimmer-30B`) | **mainline llama.cpp, pinned build** | `cmd: docker run …` of `Dockerfile.llamacpp` |
| GGUF, weights larger than VRAM (`DeepSeek-V4-Flash`) | **mainline llama.cpp, newer pin**, experts in host RAM | same Dockerfile, `llamacpp-v4` image |

`Ternary-Bonsai-27B` is a hybrid-attention, multimodal, ternary-quantized model built for
a **PrismML fork of llama.cpp** — vLLM 0.25.1 cannot serve it. This is a concrete reason
the backend-agnostic design matters.

### Why there are THREE llama.cpp images (from two Dockerfiles)

This is the non-obvious bit. They are not redundant and none can replace another:

- `Dockerfile.bonsai` builds **PrismML's `prism` fork**, which carries the Q2_0_g128 ternary
  and hybrid-attention CUDA kernels `Ternary-Bonsai-27B` needs. Mainline does not have them.
- `Dockerfile.llamacpp` builds **mainline at a pinned build tag**. The fork's branch head is
  2026-07-31, so it predates any architecture merged after that — `Muse-Glimmer-30B` landed
  in mainline on 2026-08-10 (`ggml-org/llama.cpp#26841`, build `b10353`) and fails on the
  fork with an unknown-architecture error.
- The same `Dockerfile.llamacpp` is built a **second time at a newer pin** (`v0.4.0`) as
  `llamacpp-v4`, for DeepSeek-V4-Flash. `b10362` can already load `deepseek4` and DSpark, but
  `v0.4.0` adds CUDA sparse flash-attention for DSV4 (`#27970`). A separate pin rather than a
  bump, because `b10362` is what Muse-Glimmer's ATEM tool calling and Qwen3.8 were validated
  on — one new model should not re-open two working ones. The pins are rows in the
  `llamacpp-image` workflow matrix; a further pin is another row, not another Dockerfile.

All images share `docker/gguf-serve.sh` as their entrypoint, so moving a model between them
means changing only `image:` in the generated config. Pin the mainline tag rather than
tracking a rolling one, for the same reason the vLLM image is pinned to `v0.26.0`. When
adding a GGUF model, the question to answer first is *which image can actually load it* —
check when its architecture was merged against the fork's branch date.

## Request lifecycle

1. Client → `POST /v1/chat/completions` to llama-swap `:9292` with `"model": "<id>"`.
2. llama-swap resolves an alias to its real model ID, and checks if that upstream is running;
   if not, the matrix solver evicts whatever conflicts, then it runs the model's `cmd` and
   waits for `checkEndpoint` (`/health`) to return 200.
3. Request is proxied to `proxy` (`http://127.0.0.1:<PORT>`), response streamed back.
4. `ttl` unloads idle models — seconds since the last request finished, per model:
   5 h for per-card pool entries, 10 h for the TP=2 solos (`TTL`/`TTL_SOLO` in
   `gen_config.py`). Long on purpose: cold starts run minutes, and a request for another
   model on the same card evicts the resident one immediately regardless of TTL.
   `cmdStop` (`docker stop ${MODEL_ID}`) tears down cleanly.

## Known issues llama-swap does NOT fix (set expectations)

- **PCIe contention** on concurrent TP (no NVLink) — hardware. Fix = NVLink bridge or
  don't run two TP-heavy models hot together.
- **`Qwen3.6-35B-A3B` Xid 31 (MMU fault) mid-inference** — a vLLM/AWQ-MoE kernel issue,
  not orchestration. Mitigate with **`--enforce-eager`** (disables CUDA graphs, the most
  likely trigger); if it persists, try a newer vLLM image or a different quant.

## Security note

Mounting `/var/run/docker.sock` grants the llama-swap container root-equivalent control of
the host Docker. It is needed to spawn model containers: all 24 models launch via
`docker run` (10 vLLM, 14 llama.cpp).

### What an API caller can and cannot do

**Cannot reach the socket.** It is mounted into the container and never proxied over HTTP.
Over `:9292` a caller names a model ID and llama-swap runs the `cmd` string already baked
into the image, so a caller selects *which* of 85 predefined commands runs, never *what* it
runs. No value from an HTTP request reaches a docker command line.

**The residual risk is indirect** -- a bug in llama-swap (injection via model name, path
traversal, an admin route) would turn HTTP reach into host root; likewise the GHCR image
supply chain, or uncommenting the `config.yaml` bind-mount in `docker-compose.yml`, which
would make the spawn commands host-editable.

### Accepted risk: `:9292` is unauthenticated

Deliberate. Single-tenant box on a trusted LAN, and llama-swap's API key would need an
`Authorization` header added to the three `curl` calls in `scripts/oncall-wakeup.sh`
(lines 51, 92, 115) and to every client. **Network reachability is the compensating
control**, so it has to be an actual firewall rule rather than a convention.

### Hardening, ranked by value-per-effort

1. **Firewall `:9292` to the consuming host's IP.** The compensating control for
   accepting no auth; the one item that should not be skipped.
2. **Expose only `/v1/*`** behind a reverse proxy -- block `/running`, `/ui`, `/logs`,
   `/api/*`, `/metrics`. An OpenAI-compatible client needs only `/v1/models` and
   `/v1/chat/completions`. Removes the information-disclosure surface, including the
   route that leaked the HF token (see PR #20).
3. **Rootless Docker** on this host -- the only option that actually removes
   root-equivalence, since socket access would then yield an unprivileged user. Cost:
   `nvidia-container-toolkit` with CDI; GPU passthrough is the fiddly part.
4. **Socket-free instance for single-model consumers.** llama-swap's base image runs
   `llama-server` as a child process with no socket at all. Baking the
   `llamacpp-mainline` build into the image would let a second, minimal instance serve
   one GGUF model on its own port with zero docker access, keeping this DooD instance
   internal.
5. ~~Docker socket proxy~~ -- **not a fix.** llama-swap needs `containers/create`, and
   create-with-bind-mounts is itself root-equivalent (mount `/` into a container). It
   blocks `exec` and image tampering only. Not worth doing except alongside rootless
   Docker.

**Resolved 2026-09-05 — HF token disclosure.** `GET /running` returns each model's fully
expanded `docker run` line. While the config passed `-e HF_TOKEN=${env.HF_TOKEN}`, that
echoed the token in plaintext to any unauthenticated caller on the LAN. The token env
lines were removed from `gen_config.py` (130 lines out of the generated config);
auth now comes from `/models/hf-cache/token`, inside the volume every model already
mounts. See README → *HuggingFace auth*. `/running` was the only leaking route — `/logs`,
`/v1/models`, `/health`, `/api/events` and `/ui` were checked and are clean.

Note this closed the *disclosure*, not the exposure: `:9292` is still unauthenticated on
the LAN, and the socket mount still makes it root-equivalent. Enabling llama-swap's API
key remains worthwhile — it needs an `Authorization` header added to the three `curl`
calls in `scripts/oncall-wakeup.sh` (lines 51, 92, 115) and to any client config.
