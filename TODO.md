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
- [x] `Ternary-Bonsai-27B`: PrismML fork image CI-built (`Dockerfile.bonsai` →
  `ghcr.io/tkontu/bonsai-llama`), weights downloaded, and the model **run successfully on the
  host** (confirmed 2026-08-14). It stays in `POOL` as an `anchor`, and the bonsai image stays
  with it — it is the only model needing the fork's ternary kernels.

## Hardware change: 2× 3090 + 3× A2000 (2026-09-15)

Both 3090 UUIDs unchanged → config and poller unaffected; `nvidia-smi` indices reordered (see
README GPU inventory). Plan impact in `SPEC-bigmoe.md` §12.

- [ ] **Decide the A2000 role** before any §11 work: small-model slots, extra VRAM for split
  models, or both via GPU-set-derived matrix sets (§12 proposal).
- [ ] Re-size the §11 candidates against ~80.5 GiB of VRAM. Qwen3-Coder-Next, gpt-oss-120b and
  Mistral Small 4 likely fit fully on GPU, which removes their RAM offload.
- [ ] Re-evaluate DeepSeek-V4-Flash with the A2000s: ~80 GiB left in RAM may make P2 (200 GiB
  VM) unnecessary. Measure before resizing the VM.
- [ ] Check the physical PCIe topology on the Proxmox host (VM shows `PIX`; GPU 4 is on `08:`).
- [ ] Measure A2000 usable VRAM and decode speed for a layer split, to replace §12's estimates.

## Candidate whole-box models — planned, not implemented (2026-09-14)

Plan and sizing in `SPEC-bigmoe.md` §11 (**sized for 2× 3090 — see §12 for the A2000
re-evaluation**). In recommended order:

- [ ] **Qwen3-Coder-Next** (80B/3B, `llamacpp-mainline`, no §2 prerequisites). Decide between
  `UD-Q4_K_S` (42.9 GiB, fully on GPU, thin margin) and `Q4_K_M` (45.2 GiB) with `n_cpu_moe` of
  ~2–4, by measured tok/s. `Q4_K_M` does NOT fit fully on the GPUs.
- [ ] **gpt-oss-120b** (117B/5.1B, MXFP4 59.0 GiB, ~15–18 GiB in RAM, `llamacpp-mainline`).
  A/B the EAGLE3 drafter. `bigmoe=True` unless measured decode GPU util clears `IDLE_PCT`.
- [ ] **Mistral Small 4** (119B/6.5B, UD-Q4_K_M 68.7 GiB + mmproj, ~23–25 GiB in RAM,
  `llamacpp-mainline`). Verify the real context limit: the GGUF says 1M, the model is described as 256K.
- [ ] **GLM-5.3-Flash** — blocked: `glm5next` is not in mainline llama.cpp (PRs #27752 / #27754
  / #27773 / #27917 open). Revisit on merge, and re-download the GGUF from after the merge.
  186.0 GiB at UD-Q4_K_XL: larger than P5's disk figure, and tight in a 200 GiB VM.

## DeepSeek-V4-Flash / bigmoe (2026-09-14)

Implements `SPEC-bigmoe.md` (see its §10 for deviations from the draft). One whole-box entry,
`deepseek-v4-flash` (UD-Q4_K_XL), on a new `llamacpp-v4` image (same Dockerfile at `v0.4.0`).
**22 -> 23 models.** Verified locally:

- [x] `llama-swap -validate` (v255) passes. The routing test with dummy upstreams passes:
  a bigmoe request clears both cards; a card request evicts it.
- [x] Config diff vs. the matrix-refactor output is purely additive (one entry). The
  `GGUF_ENV` refactor of `gen_config.py` leaves every existing entry byte-identical.
- [x] `gguf-serve.sh` with stubbed `hf`/`llama-server`, under both bash-sh and **dash**:
  - a first-shard name fetches all N shards (including 00012, which a shell would misread as
    octal); a cached re-run fetches nothing
  - `GGUF_N_CPU_MOE` + `GGUF_OT` exits 1; a glob-laden `-ot` regex passes through literally
  - a plain single-file entry produces the same argv as before
- [x] `oncall-wakeup.sh` under dash against a fake llama-swap, all 7 cases pass:
  - no wake while either bigmoe ID is resident, or while `/running` is down
  - still wakes when only card models are loaded; "already resident" still works
  - `c0Xmuse-glimmer` does not match `c0.muse-glimmer`

Needs CI / the host:

- [ ] **CI builds `llamacpp-v4` at `v0.4.0`.** The CMake layout matches `b10362` (same
  `llama-server` target, `build/bin` output, UI now OFF by default), but it has not been
  compiled. This PR changes `gguf-serve.sh`, so it also rebuilds `llamacpp-mainline` and
  `bonsai-llama` — same binaries, new entrypoint.
- [ ] **Spec prerequisites P2–P5**: VM RAM 200 GiB fixed / ballooning off, BIOS NPS1,
  `kernel.numa_balancing=0`, >=180 GiB free on `/models`. P3/P4 may be no-ops on one NUMA node
  (SPEC §10) — measure rather than assume.
- [ ] **Pre-download** `UD-Q4_K_XL/*` + the dspark GGUF (README → DeepSeek-V4-Flash).
- [ ] Cold start `deepseek-v4-flash`. Confirm the log shows `deepseek4`, the DSpark block size
  of 5, **sparse FA** enabled, and experts on CPU. Record VRAM per card and container RSS.
- [ ] **Walk `n_cpu_moe` down from 43** until ~44 GiB VRAM total; rebalance `tensor_split`,
  because the GPU expert layers land on card 2 first.
- [ ] SPEC §8 baselines on a cold box: decode (>=8 tok/s), decode with DSpark (>=1.4x, else drop
  the drafter and reclaim 10 GiB), 8k prefill (>=100 tok/s), peak VRAM/RSS, warm/cold load,
  and evict -> `c0.muse-glimmer` ready (<=60 s).
- [ ] On-call: with `deepseek-v4-flash` resident and idle GPUs, confirm the poller logs
  "is resident (BIGMOE_MODELS) — not evicting it" and fires no request.
- [ ] Probe tool calling (DSML) and thinking control (`enable_thinking:false`,
  `reasoning_effort:max`).
- [ ] Only if P2 slips: add a `UD-Q3_K_M` entry (119.3 GiB) as a capacity fallback.

## Matrix routing refactor (2026-09-14)

`pairNN` groups replaced by per-card entries (`c0.<model>`, `c2.<model>`) and one matrix set.
**85 -> 22 models, 36 groups -> 1 set.** Verified locally before merge:

- [x] `llama-swap -validate` (v255) accepts the config; a set naming an unknown model is
  rejected (negative control).
- [x] Every one of the 85 old IDs resolves (model or alias) to an entry whose `cmd`, `ttl`,
  `filters`, `proxy` etc. are identical — except the five card-0 `pairNN.muse-glimmer` aliases,
  which now share `c0.muse-glimmer`'s `ttl: 0` (intended: one on-call entry per card).
- [x] End-to-end on v255 with dummy upstreams: an alias rewrites the upstream `model` to the
  vLLM served name; a card request leaves the other card loaded; the same model runs on both
  cards; a whole-box model clears both and is evicted by a card request; `/v1/models` lists
  the aliases; `/running` reports real IDs (hence `ONCALL_MODEL=c0.muse-glimmer`).

Needs the host:

- [ ] **Check the deployed llama-swap version is >= v243** (full model IDs in matrix sets).
  The image is `unified-cuda` rolling; TODO above records v249, so re-pulling is enough.
- [ ] **Restart llama-swap** and confirm `/v1/models` shows 22 models + 81 aliases (103 ids).
- [ ] **`c2.gemma-26b` is new** — gemma-26b was always an anchor, so it never ran on 3090 #2.
  The cards are identical, but cold-start it once and check the ~21.9 GiB fit.
- [ ] Co-load two heavy models across cards (e.g. `c0.muse-glimmer` + `c2.qwen3.8-27b`), then
  request `c2.gemma-26b` and confirm card 0 stays loaded and only card 2 swaps.
- [ ] Same model on both cards: `c0.qwen3.5-4b` + `c2.qwen3.5-4b` — replaces `x2extract`.
- [ ] Whole-box: with both cards loaded, request `Qwen3.8-27B-split`; confirm both `docker stop`
  before it starts (no OOM from a straggler), then request a `c0.` model and confirm it evicts.
- [ ] On-call: `IDLE_SECONDS=120`, confirm the wakeup loads `c0.muse-glimmer`, a card-2 model
  stays loaded, and the next poll logs "already resident" (proves the real-ID grep matches).
- [ ] **Consumers**: move from `pairNN.<model>` / `x2extract.*` / bare names to `c0.`/`c2.` IDs.
  Anything that split callsigns on the first `.` to get a pair id now gets a card label.
  Then delete `LEGACY_PAIRS`, `LEGACY_X2EXTRACT` and `includeAliasesInList` from `gen_config.py`.

## Sampling + reasoning defaults on the llama.cpp entries (added 2026-08-14)

- [ ] **Muse-Glimmer has been serving on the wrong sampling since 2026-08-12.** llama.cpp does
  not read `general.sampling.*` from the GGUF (and Muse-Glimmer's carries none anyway), so it
  ran on llama.cpp's `common.h` defaults — `temp 0.80 / top_k 40 / min_p 0.05` against a card
  asking for `temp 1.0 / top_p 0.95 / top_k 64`. Now passed as explicit flags. **Re-run the
  quality-sensitive checks after the restart** — the 1.81x drafter A/B and the vision probe
  were both measured under the old sampling. The throughput numbers should be unaffected
  (sampling doesn't change acceptance much at these settings) but the outputs will differ.
- [ ] Confirm the flags actually landed: `--temp/--top-p/--top-k/--min-p` are appended after
  `--alias` and reach `llama-server` via `gguf-serve.sh`'s `"$@"`. The startup log echoes the
  sampler chain — check it rather than assuming.
- [ ] `reasoning_strength=low` on both Muse-Glimmer entries: confirm the rendered prompt shows
  `Reasoning strength: low.` Note the template **suppresses** its own injection when the system
  prompt already contains "reasoning strength" (and rewrites "reasoning effort" to match), so
  test with and without a system prompt that mentions it.
- [ ] Decide whether `low` is right for the **on-call** entry specifically. It is the standby
  model, so faster wake-up responses are probably what's wanted, but this is a behaviour change
  to whatever consumes that path.

## Pool retirement + role-based pairing (2026-08-14)

Retired: `phi-4`, `gemma-12b`, `mellum2-12b` (POOL) and the `Qwen3.6-27B-AWQ-INT4` /
`gemma4-26B-A4B-it-INT4-max` SOLO entries. Pairing switched from all-C(n,2) to
"pair unless same family AND same role". **128 -> 67 models, 55 -> 27 groups.**

- [ ] **`pairNN` labels renumbered again.** Retiring a POOL member reshuffles them (only
  APPENDING is additive). Anything pinning a literal pair id needs updating — read
  `/v1/models` instead.
- [ ] Confirm nothing downstream still requests a retired id: `phi-4`, `gemma-12b`,
  `mellum2-12b`, `Qwen3.6-27B-AWQ-INT4`, `gemma4-26B-A4B-it-INT4-max`, or any `pairNN.` name
  containing them. They will now 404 rather than swap in.
- [ ] `Qwen3.6-27B-AWQ-INT4` was retired as superseded by `qwen3.8-27b` (same class, newer gen,
  adds vision + 262k + thinking control). Sanity-check that on the host before deleting the
  cached weights — it is the easiest retirement to reverse while the GGUFs are still on disk.
- [ ] `gemma4-26B-A4B-it-INT4-max` was the SAME repo as pooled `gemma-26b`, differing only in
  `mml` (131072 vs 65536). If the 131k profile is actually wanted, it is cheaper to raise
  `gemma-26b`'s `mml` than to keep a second entry — but see the KV-ceiling note above, 65536
  is already near the fp16 limit on one card.
- [ ] Only one pair was dropped by the predicate itself (`qwen3.5-9b` + `qwen3.5-4b`); the rest
  of the reduction came from retirement. If more pruning is wanted, retiring members is the
  lever, not the rule.

## muse-glimmer as a pooled anchor (2026-08-14)

Moved from `UNGROUPED_GGUF` into `POOL` with `role="anchor"`, so it can be benchmarked
head-to-head. **67 -> 83 models, 27 -> 35 groups** (8 new pairs, appended as pair28-pair35).

- [ ] **The three strong-vs-strong pairs are the point of this** — `pair28` (gemma-26b),
  `pair32` (ternary), `pair35` (qwen3.8-27b). Run those first; the five anchor+fast pairs are
  ordinary co-load.
- [ ] **Verify the on-call path still works after the change.** The standalone `muse-glimmer`
  must still show `ttl: 0` and must still be the name `scripts/oncall-wakeup.sh` requests
  (`ONCALL_MODEL` defaults to it). Config-side both hold, but confirm a real wakeup fires and
  `/running` shows it.
- [ ] Confirm a `pairNN.muse-glimmer` member does NOT inherit `ttl: 0` — it should be 18000.
  A pooled member that never unloads would pin a card and quietly break the exclusive-swap
  assumption the whole on-call design rests on. (Generated config is correct; verify live.)
- [ ] VRAM on the strong-vs-strong pairs: muse-glimmer measured **20.82 GiB of ~23.3** on one
  card, and `gemma-26b` ~21.9, `qwen3.8-27b` ~21.6 on the other. Each owns its own 3090 so
  they should not interact, but these are the two fullest cards in the config running at once —
  watch for prefill spikes on the first co-load.
- [ ] Existing `pair01`-`pair27` labels did NOT move this time (appending to POOL is additive
  under the current ordering). Worth confirming against `/v1/models` anyway.

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
- [x] `muse-glimmer` cold start **verified 2026-08-12**: loaded on 3090 #0, `n_ctx=131072`,
  `vision=True`, build `b1-4801e3c` (= the pinned `b10362`). Used **18.79 GiB of 24**, i.e.
  15.61 weights + 1.30 mmproj + 1.88 KV+compute — the sliding-window KV analysis held (the
  full 131k context really does cost <2 GiB), so the OOM risk flagged pre-merge did not
  materialise. That estimate ("~2.1 GiB spare") was wrong for a different reason: it treated
  HF's DECIMAL GB file sizes as GiB, overstating weights ~7%. Actual free: **5.21 GiB**.
- [x] **Drafter needs `--spec-type draft-dflash`** (found 2026-08-12). Enabling `GGUF_DRAFT`
  alone made BOTH Muse-Glimmer entries die in ~2.8 s with "upstream command exited
  prematurely". Cause: llama.cpp defaults to `draft-simple`, which loads the drafter as a
  STANDALONE model. `dflash-kquant.gguf` is a coupled drafter — its GGUF declares
  `general.architecture=dflash` and `dflash.target_layers=[2,14,26,38,50]`, i.e. it hooks
  into the target's layers (`llama_set_embeddings_layer_inp`) rather than running on its own,
  so the standalone load fails and `server-context.cpp:1241` returns false → process exits.
  Not a file or flag-syntax problem: all four GGUFs verified byte-exact, and `-md`/`-ngld`
  are valid in b10362. No `--draft-max` needed — DFlash reads `dflash.block_size` (16) from
  metadata and clamps n_max to block_size-1 with a warning (`spec.cpp:979`), unlike the
  DSpark path in `bonsai-serve.sh`, which asserts and must be told explicitly.
- [x] **Images were built for the CI runner's CPU** (found 2026-08-12). Even with
  `--spec-type draft-dflash`, the drafter died with **exit 132 = SIGILL**. `GGML_NATIVE`
  defaults to ON, so llama.cpp compiled for the GitHub Actions runner's AVX-512/AMX cores
  (hence the `AMX is not ready to be used!` line) and the binary hits an illegal instruction
  on this box. Nasty because it was PARTIAL: plain generation ran for days and only the
  DFlash path crashed, which looked like a model/flag bug. Fixed by `-DGGML_NATIVE=OFF` on
  BOTH Dockerfiles — bonsai had the same latent hazard, surviving only because the ternary
  path never reached those instructions.
- [x] **VERIFIED END-TO-END 2026-08-12** (llama-swap v249, 106 models). `muse-glimmer`
  loads in ~35 s cold / ~14 s warm and uses **20.82 GiB** on 3090 #0 (predicted 20.31), so
  the drafter is genuinely resident — it was 18.79 GiB without it. GPU #2 stays free.
- [x] **Drafter measured at 1.81x** (A/B on the same host, 2026-08-12): **44.4 tok/s without**
  vs **80.3 tok/s with**, -np 1 greedy, 300 tokens. `draft_n` absent from the no-drafter
  response confirms speculation was really off. Acceptance is **85.7%** (draft_n=251,
  accepted=215) = 3.53 tokens per target forward pass; wall-clock gain is lower than that
  ratio because verification batches and drafting both cost time.
  Cost: **+2.03 GiB**, more than the drafter's 1.52 GiB file — the extra ~0.5 GiB is draft
  context plus embedding buffers for its five target-layer hooks. 3.18 GiB still free.
  Verdict: keep it. +81% throughput for 8% of the card.
  (The model card's 3.1x is a 5090 figure; 1.81x is what a 3090 gives.)
  NOTE: speculation cannot be toggled per request — `speculative.types` is server-level, so
  an A/B needs a second container, not a request flag.
- [x] **Vision + speculative work TOGETHER** (the actual goal): a generated PNG of red/green/
  blue horizontal stripes came back as "red, green, blue" with the drafter enabled. Stripe
  order was arbitrary, so this is grounded, not a guess.
- [x] **Reasoning is separated, not lost.** First probe returned empty `content` with
  `finish_reason: length` — the model spent all 80 tokens reasoning. llama.cpp puts it in
  `message.reasoning_content`; `content` is correct once the budget is adequate. Same class
  of trap as `qwen3.5-9b` above. Consumers must budget for it, or set
  `params=dict(reasoning_strength="low")` on the entry (mechanism already wired).
- [x] **Split entry works but earns nothing on speed**: 13.00 + 12.16 GiB across both 3090s
  (-sm layer distributes evenly), and **78.2 tok/s vs 78.8 for the single-card entry** —
  identical within noise, exactly as predicted for pipeline parallelism. Its only benefit is
  the higher-quality `dynamic` quant, paid for by owning BOTH cards and ~18 GiB of disk.
  DECIDE whether that trade is worth keeping.
- [x] **Yield/swap verified both directions**: requesting the split evicted `muse-glimmer`;
  requesting `muse-glimmer` back reloaded it in 13.5 s and freed GPU #2. `ttl: 0` confirmed
  on the resident entry.
- [ ] Confirm **vision + speculative work together** — the drafter loads before the mmproj
  (`server-context.cpp:1225-1275`), so a failure in either aborts the whole load. Run the
  vision probe with the drafter enabled, not just a text prompt.
- [ ] Benchmark decode with vs without the drafter at `-np 1` greedy — the card claims ~3.1x
  on a 5090. If it is not materially faster on a 3090, that 1.52 GiB is better spent moving
  this entry to the `dynamic` quant (21.49 GiB used / 2.51 GiB free) instead.
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

## Post-deploy verification (Qwen3.8-27B — added 2026-08-14)

Two entries: `qwen3.8-27b` (UD-Q4_K_XL + vision, one 3090, 16384 × 4 slots, **pooled** so it
co-loads) and `Qwen3.8-27B-split` (UD-Q6_K_XL + vision, both 3090s, full 262144 in one slot).

Config-side checks pass locally. NOTE: the counts below moved after the pool retirement in the
section above — the config is now **67 models / 27 groups**, not the 128/55 this section was
written against.

- [ ] **Pair labels were renumbered ONCE.** `gen_pairs_config.py` now orders pairs by
  `(higher index, lower index)` so appending a POOL member is purely additive. The pair *set*
  is unchanged and no pair was lost, but pair01–pair45 no longer point at the same partners
  they did before. **Check whether anything downstream pinned a `pairNN` id** rather than
  reading `/v1/models`. From here the numbering is stable across future additions.
- [ ] Pre-download the ~44 GB of GGUFs (README → Qwen3.8-27B) before the first cold start.
- [ ] **Restart llama-swap** — config is read at startup only, and a model TTL reload proves
  nothing. `/v1/models` going 106 → **83** is what confirms the new config was picked up.
- [ ] **Cold start `qwen3.8-27b`** and check the reported footprint against the prediction:
  16.69 (weights) + 0.86 (mmproj) + 4.00 (65536 tok × 64 KiB f16 KV) = **21.55 GiB** of the
  ~23.3 GiB budget, i.e. ~1.75 GiB for DeltaNet state (~0.3 GiB at `-np 4`) and prefill.
  That is the thinnest margin in the pool — if it OOMs or spikes, drop to plain
  `Qwen3.8-27B-Q4_K_M.gguf` (+0.76 GiB) *before* cutting `ctx`/`par`.
- [ ] Confirm `n_ctx` reports **65536** (llama.cpp is given `ctx × par`), not 16384, and that
  `vision = true`. Both are silent-failure surfaces.
- [ ] **Architecture claim is inferred, not observed.** It is read off the GGUF headers
  (`general.architecture=qwen35`, `clip.projector_type=qwen3vl_merger`) as already supported at
  the pinned `b10362` — no image rebuild. If either load path rejects it, that inference is
  what was wrong, and the fix is a `LLAMACPP_TAG` bump, **not** a flag change. Bumping the tag
  re-tests Muse-Glimmer, so re-run its checks above if it comes to that.
- [ ] **Coherence prompt** on first load — per the `qwen3.5-9b` and `muse-glimmer` precedents,
  trust the output, not a clean startup log. Doubly so here: `qwen35` is the same hybrid
  family whose documented failure mode is incoherent output from a clean boot.
- [ ] **Reasoning budget**: thinking is ON by default and llama.cpp puts it in
  `message.reasoning_content`. Both entries now ship `reasoning_effort=low` as a server-side
  default (the template's own default is `xhigh`), so confirm a short-`max_tokens` request
  actually reaches an answer instead of returning empty `content` with `finish_reason: length`.
- [ ] **Confirm the env var survived llama-swap's cmd tokenizer.** The default is passed as
  `-e 'LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_effort":"low","preserve_thinking":true}'` —
  single-quoted JSON containing double quotes. Same quoting shape as the existing
  `--gpus '"device=..."'`, so it should hold, but a mis-split would set a truncated env var and
  llama-server would fail to parse the JSON. `docker inspect` the running container, or just
  watch the startup log for a chat-template-kwargs parse error.
- [ ] **Verify it is a DEFAULT, not a forced override**: send `{"chat_template_kwargs":
  {"reasoning_effort": "xhigh"}}` and confirm the response reasons noticeably longer than the
  same prompt without it. The merge order in `server-common.cpp` says request beats env; this
  confirms it end-to-end.
- [ ] Note for consumers: only `xhigh|medium|low` are valid — the template `raise_exception()`s
  on anything else, so an OpenAI-shaped client sending `"high"` gets a hard error. And a
  TOP-LEVEL `reasoning_effort` is ignored except for the value `"none"`; it must be nested
  under `chat_template_kwargs`. Worth a probe of each so the failure mode is documented.
- [ ] **Vision probe** with a generated image whose content can't be guessed (the red/green/
  blue stripe trick), proving `--mmproj` attached rather than being silently ignored.
- [ ] **Co-load test** — the whole reason it's pooled. Request e.g. `pair46` (gemma-26b #0 +
  qwen3.8-27b #2) and confirm both stay resident and serve concurrently. gemma-26b is the
  heaviest partner (~21.9 GiB on #0) against qwen3.8-27b's ~21.6 GiB on #2; they are on
  separate cards, so this should hold, but it is the tightest pair in the set.
- [ ] `Qwen3.8-27B-split`: confirm `-sm layer` spreads across both 3090s and that 262144
  actually allocates (predicted 24.14 + 0.86 + 16.00 = **41.00 GiB** of ~46.6, ~5.6 GiB slack).
  Then an over-limit probe (expect a clean 400).
- [ ] Measure decode on the split entry. Expect **~single-card** speed — layer split is
  pipeline, not tensor, parallel, and Muse-Glimmer measured 78.2 vs 78.8 tok/s. If it is
  *slower* than `qwen3.8-27b`, the split entry isn't earning its disk.
- [ ] Revisit if an MTP GGUF appears: the weights carry a packed MTP layer
  (`nextn_predict_layers=1`) that mainline does not currently self-speculate off. That is the
  only route to a Muse-Glimmer-style 1.81x here.

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
