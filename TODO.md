# TODO — migrate `inference` to llama-swap

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Current status (2026-07-23) — deployable from GitHub via Portainer

**Done (this repo is ready to deploy):**

- [x] Custom image + CI: `.github/workflows/build-and-push.yml` builds the `Dockerfile` and
  pushes to GHCR (Portainer can't build from a repo). Compose references it via `image:`.
- [x] `config.yaml`: all 15 models ported from the old gateway (`vllm_refs/models.yaml`),
  with VRAM sizing anchored to `vllm_refs/memory_footprints.json`.
- [x] Co-load group (`groups.coload`) defined and hand-sized (35B-16k + Gemma short-KV).
- [x] `docker-compose.yml` / `.env.example` wired for Portainer stack env vars
  (`LLAMA_SWAP_IMAGE`, `HF_TOKEN`).

**Remaining — needs the host (can't be done from the repo):**

- [ ] **Rotate the HF token** that was pasted into `.env` (it's git-ignored, but was exposed).
- [ ] Make the GHCR package public (or add registry creds in Portainer).
- [ ] Cold-start validation, especially the tight-fit contexts flagged `VALIDATE` in
  `config.yaml`: `Qwen3.6-27B-AWQ-INT4` (262k), `Qwythos-…-1M-AWQ` (1M — likely needs TP2 or
  less context), `Qwythos-…-256k` (bf16).
- [~] `Ternary-Bonsai-27B`: PrismML fork image is now CI-built (`Dockerfile.bonsai` →
  `ghcr.io/tkontu/bonsai-llama`) and the model is wired on a 3090. Remaining: make that GHCR
  package public, download the GGUF weights (`prism-ml/Ternary-Bonsai-27B-gguf`: Q2_0 + mmproj
  + dspark-Q4_1) into `/models/hf-cache`, then cold-start to validate the fork flags.

## 0. Decisions to lock first

- [ ] **GPU layout for the co-load pair.** Pick one:
  - (A) `35B` TP=2 on both 3090s **+** small model shares one 3090. (Matches original goal;
    shared-card + PCIe contention when both hot.)
  - (B) `35B` TP=2 owns both 3090s exclusively; small/GGUF models go on the **A2000**.
    (Best `35B` speed; small model slower on A2000.)
  - _Recommendation: start with (B) for the heavy model's sake; fall back to (A) if the
    A2000 can't hold the models you want co-resident._
- [ ] **NVLink bridge?** If concurrent TP throughput matters long-term, price a 3/4-slot
  NVLink bridge for the two 3090s. Single highest-impact upgrade; llama-swap can't fix PCIe.
- [ ] Confirm cache dir (`/models/hf-cache`) and vLLM image tag (`vllm/vllm-openai:v0.25.1`).

## 1. Build the custom image

- [ ] `Dockerfile` = `FROM ghcr.io/mostlygeek/llama-swap:unified-cuda` + `docker.io`.
- [ ] Decide build strategy: Portainer builds from repo (`build: .`) **or** pre-build and
  push to a registry, then reference `image:` in compose.
- [ ] Verify `docker` CLI works inside the container against the mounted socket:
  `docker run --rm hello-world` from within.

## 2. Author `config.yaml` (all 15 current models)

Translate each entry from the old `config/models.yaml`. Source list:

- [ ] `Qwen3.6-35B-A3B-AWQ-4bit`  (TP=2; `--enforce-eager`; repo `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`)
- [ ] `Qwen3.6-35B-A3B-AWQ-4bit_16k_8seqs`  (profile variant of the above)
- [ ] `Qwen3.6-27B-AWQ-INT4`
- [ ] `gemma-4-E4B-it-qat-AWQ-INT4`  (small; TP=1; repo `cyankiwi/gemma-4-E4B-it-qat-AWQ-INT4`)
- [ ] `gemma-4-E4B-it-qat-AWQ-INT4-shortkv`  (profile variant)
- [ ] `gemma-4-12B-it-qat-AWQ-INT4`
- [ ] `gemma-4-26B-A4B-it-qat-AWQ-INT4`
- [ ] `gemma4-26B-A4B-it-INT4-max`
- [ ] `qwen3.5-9b`
- [ ] `Qwen3.5-4B-AWQ-4bit-shortkv`
- [ ] `phi-4-AWQ`
- [ ] `Mellum2-12B-A2.5B-Instruct-AWQ-INT4`
- [ ] `Qwythos-9B-Claude-Mythos-5-1M`
- [ ] `Qwythos-9B-Claude-Mythos-5-1M-AWQ`
- [ ] `Ternary-Bonsai-27B`  → **llama.cpp / PrismML fork**, NOT vLLM (see below)

For each vLLM model, carry over from the old config:

- [x] repo id + `--quantization`
- [ ] `--max-model-len`, `--max-num-seqs`
- [ ] `--gpu-memory-utilization` (compute by hand per card — see ARCHITECTURE)
- [ ] `--tensor-parallel-size` (2 only if it doesn't fit one card)
- [ ] GPU pin(s) by **UUID**
- [ ] `ttl` (idle unload), `cmdStop: docker stop ${MODEL_ID}`, `proxy: http://127.0.0.1:${PORT}`

## 3. Concurrency groups

- [ ] Define the co-load group (`swap:false, exclusive:false, persistent:true`) for the
  models that must be resident together.
- [ ] Leave the rest as on-demand (they swap normally) or their own groups.
- [ ] Sanity-check: for every group that can be co-resident, the per-card
  `gpu-memory-utilization` sums stay < ~0.90 on each shared card.

## 4. Ternary-Bonsai-27B (special case)

- [ ] Obtain / build the **PrismML llama.cpp fork** container image (custom `Q2_0_g128`
  hybrid-attention kernels).
- [ ] Add a `cmd: docker run … <prismml-image> -m /models/…gguf --host 0.0.0.0 --port 8080
  -ngl 99 …` entry (or run bundled `llama-server` if the mainline build ever supports it).
- [ ] File `Ternary-Bonsai-27B-dspark-bf16.gguf` (weights) + optional `mmproj` for vision.
- [ ] Point tokenizer/chat template appropriately (base model `Qwen/Qwen3.6-27B`).
- [ ] Decide GPU: A2000 (12 GB) likely fits the ternary bf16 (~7 GB).

## 5. Portainer stack

- [ ] Push this folder to its own git repo.
- [ ] Create stack via **Repository** method → `docker-compose.yml`.
- [ ] Add `HF_TOKEN` (and any other secrets) as stack env / `.env`.
- [ ] Enable auto-update (webhook or poll) if desired.
- [ ] Confirm llama-swap comes up on `:9292`, `/ui` loads.

## 6. Validation

- [ ] `GET /v1/models` lists all configured models.
- [ ] Cold-start each model once; confirm it reaches `/health` and answers.
- [ ] Load the co-load group; confirm both stay resident (`GET /running`).
- [ ] Concurrent load test on `35B` + small model; record tok/s to set realistic
  expectations (PCIe-bound — see ARCHITECTURE).
- [ ] Confirm `35B` no longer crashes with `--enforce-eager` (watch for Xid 31 in
  `sudo dmesg -T | grep -iE 'xid|nvrm'`). If it still crashes, escalate the vLLM/kernel path.
- [ ] Confirm idle `ttl` unloads and `cmdStop` cleanly removes containers (no orphans).

## 7. Cutover & decommission

- [ ] Run llama-swap alongside the old gateway on a different port; A/B a few models.
- [ ] Repoint clients from the old gateway to `:9292`.
- [ ] Stop/disable the old `vlmm-gateway` stack.
- [ ] Keep the old repo as a **reference** for vLLM sizing knowledge (kv reservation,
  budget util math, TP footprint), not as a running service.

## Open questions

- [ ] Do any clients depend on the old gateway's per-model `request_defaults` injection?
  If so, replicate via llama-swap `filters` (`setParams` / `setParamsByID`).
- [ ] Is API-key auth needed on `:9292` (who can reach it)?
- [ ] Long-context profiles (`_16k_8seqs`, `-shortkv`, `-max`): keep as separate model IDs
  (separate `cmd`s / ports) — llama-swap has no notion of "profiles of one repo", each is
  just its own model entry.
