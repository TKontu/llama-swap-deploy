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
| `.github/workflows/build-and-push.yml` | CI: builds the Dockerfile and pushes the image to GHCR (Portainer can't build from a repo) |
| `docker-compose.yml` | Portainer stack definition (pulls the pre-built image) |
| `config.yaml` | llama-swap model definitions — all 15 models, ported from the old gateway |
| `.env.example` | Secrets/vars (copy to `.env`, or set as Portainer stack env vars) |
| `stack.env` | Placeholder env file for the Portainer stack (set real values in the Portainer UI) |

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

## Deploy with Portainer (GitOps)

> **Important:** Portainer **cannot build an image** from a Git-repository stack — the
> `build:` directive is not supported there
> ([Portainer docs](https://docs.portainer.io/faqs/troubleshooting/stacks-deployments-and-updates/can-i-build-an-image-while-deploying-a-stack-application-from-git)).
> So this repo builds the image in **GitHub Actions** and pushes it to **GHCR**; Portainer
> just pulls it. The same GitHub repo is both the image source and the stack source.

### One-time setup

1. **Push this folder to a GitHub repo** and let the workflow run (Actions tab →
   `build-and-push`; it also runs on `Dockerfile` changes and via *Run workflow*).
2. **Make the GHCR package public** (repo → *Packages* → the image → *Package settings* →
   change visibility → *Public*). The image contains no secrets — only llama-swap + the
   `docker` CLI. (Keep it private instead if you prefer; then add GHCR registry credentials
   in Portainer → *Registries*.)
3. Grab the image reference from the workflow's run summary, e.g.
   `ghcr.io/<owner>/llama-swap-deploy:latest` (all lowercase).

### Create the stack

1. Portainer → **Stacks → Add stack → Repository**.
2. Repository URL = your repo; **Compose path** = `docker-compose.yml`.
3. Add these **environment variables**:
   - `LLAMA_SWAP_IMAGE=ghcr.io/<owner>/llama-swap-deploy:latest`
   - `HF_TOKEN=hf_…`
4. Enable **automatic updates** (poll or webhook) for push-to-deploy. Because `config.yaml`
   is bind-mounted from the cloned repo, **editing models is just a git push** — Portainer
   re-pulls and restarts; no image rebuild is needed (only `Dockerfile` changes rebuild).
5. Deploy.

After deploy, llama-swap listens on **`http://<host>:9292`** (OpenAI-compatible), with a
web UI at `http://<host>:9292/ui`.

### Local development

The compose file keeps a commented `build: .` for local use only (Portainer ignores it):

```bash
cp .env.example .env      # fill in HF_TOKEN; LLAMA_SWAP_IMAGE unused locally
docker compose build && docker compose up
```

---

## Verify

```bash
curl http://<host>:9292/v1/models                 # list configured models
curl http://<host>:9292/running                   # currently loaded upstreams
curl -Ns http://<host>:9292/logs/stream           # live logs

# fire a request (loads the model on demand)
curl http://<host>:9292/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.6-27B-AWQ-INT4",
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

Defined under the **top-level** `groups` key in `config.yaml` (there is no `swap:`
wrapper). Any model not in a group swaps on demand, one at a time. To keep the big model
and a small model **both loaded at once**:

```yaml
groups:
  coload:
    swap: false        # members stay loaded together (no swap among them)
    exclusive: true    # loading this pair unloads other models (one 24 GB pair at a time)
    persistent: false  # keep false — `true` would let this pair block every other model
    members: ["Qwen3.6-35B-A3B-AWQ-4bit_16k_8seqs", "gemma-4-E4B-it-qat-AWQ-INT4-shortkv"]
```

> ⚠️ Don't set `persistent: true` on a group that occupies both 3090s — the other 13
> models could then never load. See `config.yaml` for the sized co-load pair.

---

## Operational notes

- **vLLM crash mitigation:** keep `--enforce-eager` on `Qwen3.6-35B-A3B` until the Xid 31
  crash is confirmed resolved (see ARCHITECTURE → "Known issues llama-swap does NOT fix").
- **Concurrent TP throughput is PCIe-bound** on this box (no NVLink). Don't expect two
  TP-heavy models to run fast simultaneously — that's hardware, not llama-swap.
- **Docker socket** is mounted into the llama-swap container so it can spawn vLLM
  siblings. This is privileged; the box is single-tenant, but be aware.
