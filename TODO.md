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
- [x] GHCR package visibility — no action needed. The repo is public and the images carry
  `org.opencontainers.image.source`, so packages inherit public visibility on first push
  (verified 2026-08-11: all three pull anonymously).
- [ ] Cold-start validation, especially the tight-fit contexts flagged `VALIDATE` in
  `config.yaml`: `Qwen3.6-27B-AWQ-INT4` (262k), `Qwythos-…-1M-AWQ` (1M — likely needs TP2 or
  less context), `Qwythos-…-256k` (bf16).
- [~] `Ternary-Bonsai-27B`: PrismML fork image is now CI-built (`Dockerfile.bonsai` →
  `ghcr.io/tkontu/bonsai-llama`) and the model is wired on a 3090. Remaining: download the
  GGUF weights (`prism-ml/Ternary-Bonsai-27B-gguf`: Q2_0 + mmproj + dspark-Q4_1) into
  `/models/hf-cache`, then cold-start to validate the fork flags.

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
- [ ] Confirm cache dir (`/models/hf-cache`) and vLLM image tag (`vllm/vllm-openai:v0.26.0`).

## Post-deploy verification (v0.26.0 + Qwen3.5 APC + gemma 64k branch)

- [ ] `qwen3.5-9b` (now `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT4`): run a coherence prompt on
  first load — the family's silent-failure mode (ignore-list vs shard tensor-name
  mismatch) produces incoherent output with a CLEAN startup log. Shard naming was
  verified compatible 2026-08-04, but trust the output, not the boot.
- [ ] APC on the Qwen3.5 hybrids: confirm `cache_config_info` shows
  `enable_prefix_caching=True` via `/upstream/<model>/metrics`, look for the
  experimental align-mode warning in startup logs, then watch
  `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` move under the
  repeated-instruction workload. Hits need a token-identical prefix ≥ 528 tokens.
- [ ] `gemma-26b` @ 65536: needle-in-haystack near the limit (×3), an over-limit probe
  (expect clean 400), and a co-load stress round with its pair partner busy — watch
  for prefill-activation OOM at util 0.95. Fallback ladder: 49152 → fp8 KV cache →
  cap `--max-num-batched-tokens`.
- [ ] MTP on `qwen3.5-9b` (`--speculative-config qwen3_next_mtp`, `mtp.fc` was left
  unquantized for this): test SEPARATELY from prefix caching first — the combo has
  crashed during cudagraph profiling on hybrid Mamba models — then together.

## Post-deploy verification (Muse-Glimmer-30B + on-call standby)

Config-side checks already pass locally: 106 models / 45 groups, with all 104 pre-existing
entries byte-identical and the two Muse-Glimmer entries purely additive. The rest needs the host.

- [x] Images built and pushed (2026-08-11): `llamacpp-mainline` (new), `bonsai-llama`
  (rebuilt via the fixed trigger, since `docker/gguf-serve.sh` changed), and the main image.
  All three pull anonymously — no visibility step needed.
- [ ] Pre-download the ~38 GB of GGUFs (README → Muse-Glimmer) so the first cold start
  isn't a multi-GB stall.
- [ ] **Restart llama-swap** — config is read at startup only. `/v1/models` going 104 → 106
  confirms the new config was actually picked up.
- [ ] `muse-glimmer` cold start: watch `nvidia-smi` against the budget (16.76 weights +
  1.40 mmproj + ~1.2 compute + ~1.82 KV ≈ 21.2 GiB, **~2.1 GiB spare**). This is the
  thinnest number in the whole change. It assumes llama.cpp allocates SWA layers windowed,
  not full — if it allocates full, KV jumps to ~6.5 GiB and it OOMs. Fallback ladder:
  drop `ctx` 131072 → 65536, then `cache_type="q8_0"`.
- [ ] **Coherence prompt** on first load — per the `qwen3.5-9b` precedent above, trust the
  output, not a clean startup log.
- [ ] **Vision probe**: send an image part and confirm a grounded description, proving
  `--mmproj` actually attached rather than being silently ignored.
- [ ] Context probe near 131k, plus an over-limit request (expect a clean 400).
- [ ] **Stop tokens**: the model card warns never to stop on `<|eom|>` (only
  `<|end_of_text|>` / `<|eot|>`). Confirm generations end cleanly and aren't truncated
  mid-reasoning; if they are, add explicit EOG handling in `gguf-serve.sh`.
- [ ] Confirm `-np 1` behaviour under load: a second concurrent request should QUEUE behind
  the first (llama-server has one slot; `concurrencyLimit: 4` lets llama-swap admit 4), not
  429 or share context. If queueing hurts in practice, that is the argument for raising
  `par` — at the cost of KV, since each extra slot adds its own 2048 sliding window.
- [ ] `Muse-Glimmer-30B-split`: confirm `-sm layer` spreads across both 3090s, and measure
  decode speed — expect ~single-card, since layer-split is pipeline, not tensor, parallel.
  If it's *slower* than the single-card entry, the split entry isn't earning its disk.
- [ ] On-call: set `IDLE_SECONDS=120` on the `oncall-wakeup` service, confirm exactly one
  wakeup fires and `/running` then shows `muse-glimmer`. Restore `3600`.
- [ ] On-call **yield test**: with `muse-glimmer` resident, request `gemma-26b` and confirm
  a clean swap. This is what `persistent: true` would have broken.
- [ ] On-call **no-interrupt test**: start a long generation on another model, confirm the
  poller doesn't fire mid-stream (GPU util stays above `IDLE_PCT`, resetting the counter).
- [ ] Decide whether the ~1 h replacement delay is right in practice — it now supersedes
  the 5 h TTL for *replacement* (TTLs still govern unloading).

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
