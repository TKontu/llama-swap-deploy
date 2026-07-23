# llama-swap deployment (inference server)

OpenAI/Anthropic-compatible model gateway for the `inference` box, built on
[**llama-swap**](https://github.com/mostlygeek/llama-swap) instead of the bespoke
`vlmm-gateway`. llama-swap is a mature, backend-agnostic **process swapper + proxy**:
it launches inference servers (vLLM, llama.cpp, …) on demand and routes OpenAI/Anthropic
API calls to the right one. We use it to run a **fixed set of models with an explicit,
static GPU layout** — which is more reliable than dynamic VRAM packing for this hardware.

- **Why we switched:** see [`ARCHITECTURE.md`](./ARCHITECTURE.md).
- **Migration plan / open tasks:** see [`TODO.md`](./TODO.md).

---

## What's in this folder

| File | Purpose |
|------|---------|
| `README.md` | This file — quick start + operations |
| `ARCHITECTURE.md` | Design, hardware constraints, why llama-swap, topology |
| `TODO.md` | Migration checklist + open questions |
| `Dockerfile` | Custom image = `llama-swap:unified-cuda` + `docker` CLI (needed to spawn vLLM containers) |
| `docker-compose.yml` | Portainer stack definition |
| `config.yaml` | llama-swap model definitions (template — fill in all models) |
| `.env.example` | Secrets/vars (copy to `.env`) |

---

## Prerequisites (on the `inference` host)

- Docker + NVIDIA Container Toolkit (`--runtime nvidia` works).
- The vLLM image already in use: `vllm/vllm-openai:v0.25.1`.
- Model cache dir on host: `/models/hf-cache` (mounted into every container at
  `/root/.cache/huggingface`).
- Portainer installed and pointed at this host's Docker.

### GPU inventory (as of migration)

| Idx | Name | VRAM | UUID | PCI |
|-----|------|------|------|-----|
| 0 | RTX 3090 | 24 GB | `GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1` | `06:10` |
| 1 | RTX A2000 | 12 GB | `GPU-690062e6-be81-ab00-ebd3-7181cafcea4a` | `06:11` |
| 2 | RTX 3090 | 24 GB | `GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb` | `06:1B` |

> **No NVLink.** TP=2 across the two 3090s runs over **PCIe** — see ARCHITECTURE for the
> performance implications. Pin GPUs by **UUID** (indices can reorder across reboots).

---

## Deploy with Portainer

The upstream llama-swap repo ships **no compose file**, so you cannot point Portainer at
it directly. Deploy one of these two ways instead:

### Option A — Repository stack (recommended, GitOps)

1. Push this folder to its own git repo.
2. Portainer → **Stacks → Add stack → Repository**.
3. Repository URL = your repo; **Compose path** = `docker-compose.yml`.
4. Add env vars (or reference `.env`): `HF_TOKEN=…`.
5. Enable **automatic updates** (poll or webhook) if you want push-to-deploy.
6. Deploy. Portainer builds the `Dockerfile` and starts the stack.

### Option B — Web editor (quick start)

1. Portainer → **Stacks → Add stack → Web editor**.
2. Paste `docker-compose.yml`. Because it uses `build:`, either pre-build/push the image
   to a registry and switch to `image:`, or use the Repository method (Option A) which
   builds for you.
3. Set env vars, deploy.

After deploy, llama-swap listens on **`http://<host>:9292`** (OpenAI-compatible), with a
web UI at `http://<host>:9292/ui`.

---

## Verify

```bash
curl http://<host>:9292/v1/models                 # list configured models
curl http://<host>:9292/running                   # currently loaded upstreams
curl -Ns http://<host>:9292/logs/stream           # live logs

# fire a request (loads the model on demand)
curl http://<host>:9292/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.6-35B-A3B",
  "messages": [{"role":"user","content":"hi"}]
}'
```

---

## Adding / editing a model

Edit `config.yaml`. Each vLLM model is a `docker run` command that llama-swap executes
via the mounted Docker socket:

```yaml
models:
  "MyModel":
    cmd: |
      docker run --rm --name ${MODEL_ID}
      --gpus '"device=GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb"'
      -e HUGGING_FACE_HUB_TOKEN=${env.HF_TOKEN}
      -v /models/hf-cache:/root/.cache/huggingface
      -p ${PORT}:8000
      vllm/vllm-openai:v0.25.1
      --model org/My-Model-AWQ --quantization awq
      --tensor-parallel-size 1 --gpu-memory-utilization 0.90
      --max-model-len 16384 --port 8000
    cmdStop: docker stop ${MODEL_ID}
    proxy: http://127.0.0.1:${PORT}
    ttl: 600
```

Key rules:
- **Pin GPUs by UUID** in `--gpus '"device=<uuid>"'`.
- Publish the port (`-p ${PORT}:8000`) and set `proxy: http://127.0.0.1:${PORT}` — works
  because llama-swap runs with `network_mode: host`.
- Size `--gpu-memory-utilization` **by hand** so co-resident models fit the card
  (see ARCHITECTURE → "Manual VRAM sizing").
- Put models that must run **at the same time** in the same `group` (see below).

Redeploy the stack (or `POST /api/models/unload/:id`) to pick up changes.

---

## Concurrency (which models run together)

Defined under `swap.group.groups` in `config.yaml`. To keep the big model and a small
model **both loaded at once**:

```yaml
swap:
  use: group
  group:
    groups:
      coload:
        swap: false        # members stay loaded together (no swap among them)
        exclusive: false    # loading a member does not unload other groups
        persistent: true    # other groups can never evict this one
        members: ["Qwen3.6-35B-A3B", "gemma-4-E4B"]
```

---

## Operational notes

- **vLLM crash mitigation:** keep `--enforce-eager` on `Qwen3.6-35B-A3B` until the Xid 31
  crash is confirmed resolved (see ARCHITECTURE → "Known issues llama-swap does NOT fix").
- **Concurrent TP throughput is PCIe-bound** on this box (no NVLink). Don't expect two
  TP-heavy models to run fast simultaneously — that's hardware, not llama-swap.
- **Docker socket** is mounted into the llama-swap container so it can spawn vLLM
  siblings. This is privileged; the box is single-tenant, but be aware.
