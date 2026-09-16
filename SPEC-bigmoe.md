# SPEC: hybrid CPU/GPU MoE backend (`bigmoe`)

Status: DEPLOYED and measured on the host (2026-09-16) — §8 acceptance met, see §15.
See §10 for where the implementation deviates from this draft, §11 for planned candidates,
§12 for why the three A2000s are not available to this deployment, §13 for storage
and the confirmed VM RAM (216 GiB).
Target repo: `TKontu/llama-swap-deploy`
First model: DeepSeek-V4-Flash (284B total / 13B active, native MXFP4 experts)

---

## 1. Purpose

Add a model class the current deployment has never had: a model whose **weights do not
fit in VRAM at all**, and which runs with routed experts resident in host RAM while
attention, shared experts and KV stay on the 3090s.

Every existing entry — `qwen3.8-27b`, `muse-glimmer`, the `pairNN.` co-loads, the vLLM
models — is sized against a 24 GiB or 48 GiB *VRAM* budget. This one is sized against a
~200 GiB *host RAM* budget. That is a second, orthogonal resource that llama-swap does
not model, and most of this spec is about making that safe rather than about the model.

## 2. Prerequisites

| # | Item | Owner | Blocking? |
|---|------|-------|-----------|
| P1 | TrueNAS stays on the Dell; not migrated to this host | done | — |
| P2 | Inference VM RAM raised 128 GiB → 200 GiB, fixed, ballooning off — **RAM done: 216 GiB** (§13); fixed/ballooning-off unverified | Proxmox | verify only |
| P3 | BIOS memory interleave set to **NPS1** | BMC | yes |
| P4 | `kernel.numa_balancing=0` on the inference VM | host | yes |
| P5 | ≥180 GiB free on `/models` (NVMe) for weights — **moved to `/fast`, satisfied** (§13) | host | done |
| P6 | llama.cpp build with DeepSeek-V4 support (see §3) | CI | yes |

P3/P4 are not optional polish. With ~150 GiB of expert tensors spread across all eight
channels, the kernel migrating pages mid-decode is a measurable and repeatable loss.

## 3. Backend image

DeepSeek-V4 support landed after the pinned `LLAMACPP_TAG=b10362`. Community quants state
they require the `wip/deepseek-v4-support` branch (PR #22378) or later.

**Decision: build a third image, `llamacpp-v4`. Do not bump `Dockerfile.llamacpp`.**

Rationale follows the `bonsai` precedent already in the repo: `b10362` is load-bearing for
`muse-glimmer` (its dedicated ATEM chat format lives in that build) and for `qwen3.8-27b`.
Bumping the shared tag to get one new model puts two working, tool-calling models at risk
of a silent template or sampler regression. A separate pin costs one workflow file and
~10 minutes of CUDA compile.

```
Dockerfile.v4                      # ARG LLAMACPP_V4_TAG, sm_86, ships docker/gguf-serve.sh
.github/workflows/v4-image.yml     # → ghcr.io/<owner>/llamacpp-v4:latest
```

Reuses `docker/gguf-serve.sh` unchanged except for §4. Three images is the ceiling — if a
fourth is ever needed, that is the signal to make the tag a build matrix instead.

## 4. `gguf-serve.sh` additions

New env vars, all optional and inert when unset, so the shared entrypoint stays valid for
the bonsai and mainline images:

```sh
# MoE CPU offload
GGUF_N_CPU_MOE       # int  → --n-cpu-moe N
GGUF_OT              # str  → -ot <regex>   (escape hatch; mutually exclusive with above)
GGUF_NUMA            # str  → --numa distribute
GGUF_THREADS         # int  → --threads / --threads-batch
GGUF_BATCH           # int  → -b
GGUF_UBATCH          # int  → -ub
```

Rules:

- `--mlock` is **never** set. The GGUF is mmap'd; page cache must stay reclaimable or an
  eviction under memory pressure becomes an OOM kill instead of a slow reload.
- If both `GGUF_N_CPU_MOE` and `GGUF_OT` are set, fail fast at entrypoint rather than
  letting llama-server pick. Silent precedence is how the KV-sizing bugs happened before.

## 5. Model entries

Both entries own **both 3090s and the RAM reservation**. They therefore live in
`UNGROUPED_GGUF`, not `POOL` — same reasoning as `Muse-Glimmer-30B-split`: a model that
owns both cards cannot be half of a co-load pair by construction.

| Model ID | Quant | On disk | Context | Notes |
|----------|-------|---------|---------|-------|
| `deepseek-v4-flash` | `UD-Q4_K_XL` | ~161 GB | 65536, 1 slot | default |
| `deepseek-v4-flash-q3` | `Q3_K_M` | ~125 GB | 65536, 1 slot | fallback if P2 slips |

Quant rationale: the routed experts are ~96% of the model and ship **natively MXFP4**.
A Q8 build (~162 GB lossless) buys essentially nothing over 4-bit — the experts were never
more than ~4 bits — while doubling the per-token read that decode speed is bound by. Do
not go above Q4-class here. Q3 and below are genuinely lossy for this model; treat the Q3
entry as a capacity fallback, not a quality tier.

Context (**superseded 2026-09-15: native 1M, see §14**): V4-Flash's CSA+HCA stack costs roughly a tenth of V3.2's KV at 1M. 64k in one
slot is the starting point; raise it only after §8 baselines exist, since the constraint
here is weights, not context.

DSpark speculative decoding is enabled for the GGUFs and is reported at 1.5–1.9×. Wire it
through the existing `GGUF_DRAFT` path (the bonsai entry already does exactly this with
its `*dspark-Q4_1*` drafter) and A/B it in §8 rather than assuming it.

### Config sketch

```yaml
  "deepseek-v4-flash":
    cmd: |
      docker run --rm --name ${MODEL_ID}
      --gpus '"device=GPU-a8c640ca-...,GPU-094f1ca3-..."'
      -v /models/hf-cache:/root/.cache/huggingface
      -e GGUF_N_CPU_MOE=48
      -e GGUF_NUMA=distribute
      -e GGUF_THREADS=12
      -e GGUF_BATCH=4096 -e GGUF_UBATCH=1024
      -p ${PORT}:8000
      ghcr.io/<owner>/llamacpp-v4:latest
    cmdStop: docker stop ${MODEL_ID}
    proxy: http://127.0.0.1:${PORT}
    ttl: 28800
```

`-b 4096 / -ub 1024` rather than the 2048/512 defaults: those defaults are tuned for pure
GPU inference, and below the op-offload threshold llama.cpp does prefill for the
CPU-assigned weights **on the CPU** — 12 Zen 2 cores, which is the worst path available
here. Bigger batches keep prefill on the 3090s.

`--n-cpu-moe` is the tuning knob, not `-ot`. Start at "all experts on CPU", then walk N
down until VRAM lands at ~44/48 GiB across the pair.

`ttl: 28800` (8 h). Cold load is disk-bound at ~160 GB off NVMe; this should not be
casually evicted.

## 6. Eviction and the on-call poller — **required change**

This is the part that breaks if it ships as-is.

`scripts/oncall-wakeup.sh` polls `/metrics`, and when **both 3090s** sit below `IDLE_PCT`
for `IDLE_SECONDS` it fires a 1-token request at `muse-glimmer`, evicting whatever is
loaded.

Two failure modes against this model:

1. **GPU utilisation is not an idle signal for a CPU-offload model.** During decode, the
   3090s hold only attention, shared experts and KV — they are mostly *waiting on RAM*.
   The cards will read as idle while the model is actively generating. The poller will
   evict a running inference.
2. Even when genuinely idle, the trade is bad: it discards a ~160 GB load that took a
   minute to fault in, to restore a 17 GB standby.

**Fix (do this in the same PR as the model entry, not after):** the poller must consult
`/running` before firing, and skip the wakeup entirely if the loaded model is in a
`BIGMOE_MODELS` set. Add the set as a service env var alongside `ONCALL_MODEL`.

Keep the group `exclusive: true` and **not** `persistent` — the existing warning applies
unchanged, and more sharply, since this one holds both cards *and* the RAM.

## 7. Things this spec explicitly does not do

- **No vLLM path.** V4-Flash needs vLLM 0.20.0+ and ~284 GB for an FP8-only conversion.
  Not reachable on this box. llama.cpp is the only backend.
- **No `POOL` membership, no `pairNN.` pairs.** Nothing co-loads with this.
- **No on-call role.** `muse-glimmer` stays the standby.
- **No RAM admission control in llama-swap.** Out of scope; §6 exclusivity plus one big
  model at a time is the mitigation. Revisit only if a second RAM-resident model appears.

## 8. Acceptance criteria

Baselines measured on a cold box, recorded in `TODO.md` next to the existing numbers:

| Metric | Target | Notes |
|--------|--------|-------|
| Decode, single stream | ≥ 8 tok/s | ~7 GB read/token at 4-bit vs ~100–130 GB/s real DDR4 bandwidth puts the ceiling near 14–18 |
| Decode with DSpark | ≥ 1.4× the above | else drop the drafter and reclaim its VRAM |
| Prefill, 8k prompt | ≥ 100 tok/s | if far below, `-b`/`-ub` are too small and prefill fell to the CPU |
| Peak VRAM | ≤ 44 GiB across both cards | tuned via `--n-cpu-moe` |
| Peak container RSS | ≤ 180 GiB | leaves headroom in the 200 GiB VM |
| Warm load | ≤ 90 s | cold load will be worse; record both |
| Evict → `muse-glimmer` ready | ≤ 60 s | confirms §6 didn't regress on-call |

Note the measured bandwidth assumption: the 3945WX has four CCDs, so expect ~100–130 GB/s
in practice, not the ~204 GB/s the eight channels imply. If STREAM comes back materially
below 100 GB/s, P3/P4 were not applied — fix that before tuning anything else.

## 9. Rollback

Each piece reverts independently, which is the point of the three-image split:

1. Remove the two entries from `UNGROUPED_GGUF`, regenerate `config.yaml`, push.
2. Poller change is additive and safe to leave in place (no-op when the set is empty).
3. `llamacpp-v4` image and workflow can sit unused; nothing else references them.
4. Weights stay on `/models`; deleting them is a separate, deliberate step.

No existing model's image tag, config, or VRAM budget is touched by this change.

## 10. Implementation notes (2026-09-14)

What shipped, and where it differs from the draft above. The draft text is kept unchanged so
the original reasoning stays readable.

| § | Draft | Implemented | Why |
|---|-------|-------------|-----|
| 3 | Needs `wip/deepseek-v4-support` (PR #22378); new `Dockerfile.v4` + `v4-image.yml` | `llamacpp-v4` image = the **same** `Dockerfile.llamacpp` at `LLAMACPP_TAG=v0.4.0`, a second row in the `llamacpp-image` workflow matrix | #22378 closed unmerged; V4 landed in mainline as #24162 (2026-06-29) and V4 DSpark as #25784 (2026-08-02). `b10362` already loads both, but `v0.4.0` adds CUDA sparse FA for DSV4 (#27970). The separate pin keeps the draft's point (don't move Muse/Qwen3.8 off `b10362`) without a copied Dockerfile. |
| 4 | Six `GGUF_*` env vars | As drafted, plus **split-shard download**: `GGUF_FILE` names the first shard and every shard is fetched | The quants are multi-file (`UD-Q4_K_XL` 5 shards) in a repo subdirectory. Fetching only the named file leaves shard 1 of N. |
| 4 | — | `GGUF_DRAFT_MAX` now maps to `--spec-draft-n-max` | `v0.4.0` turned `--draft-max` into a hard "argument has been removed" error. `b10362` already accepts the new name. |
| 5 | `unsloth/DeepSeek-V4-Flash-GGUF`, `UD-Q4_K_XL` ~161 GB + `Q3_K_M` ~125 GB fallback | `unsloth/DeepSeek-V4-Flash-0731-GGUF`, `UD-Q4_K_XL` only (144.4 GiB). **No Q3 entry.** | The original repo ships **no** DSpark drafter; the 0731 checkpoint does (`dspark-…-Q8_0.gguf`, 10.1 GiB). The Q3 entry only pays off if P2 slips, and Q3 is lossy for this model; add `UD-Q3_K_M` (119.3 GiB) then, not preemptively. |
| 5 | DSpark via `GGUF_DRAFT` "like bonsai" | `spec_type=draft-dspark`, `draft_max=3`, drafter on the GPUs (`-ngld 99`) | Bonsai uses its own entrypoint. Drafter metadata read from the GGUF header: `general.architecture=dflash`, `dflash.block_size=5`, `target_layers=[41,42,43]`. `v0.4.0` clamps an oversize draft instead of asserting. The drafter's 10.1 GiB counts against the §8 VRAM budget. |
| 5 | Sketch: `-p ${PORT}:8000`, `--n-cpu-moe 48` | Generated by `gen_config.py` (`UNGROUPED_GGUF`, `bigmoe=True`), port 8080, `n_cpu_moe=43` | gguf-serve listens on 8080. 43 = `deepseek4.block_count`, i.e. all experts in RAM — the "start at all experts on CPU" the draft asks for. A hand-written YAML entry would be deleted on the next regeneration. |
| 5 | — | Sampling `--temp 1.0 --top-p 1.0 --min-p 0.01` | Model card + unsloth's llama.cpp commands. Thinking left at the template default (High). |
| 6 | `BIGMOE_MODELS` skip | As drafted, plus: skip when `/running` is unreadable (fail closed), fixed-string ID match | IDs contain dots. `gen_config.py` refuses to generate if `BIGMOE_MODELS` in `docker-compose.yml` differs from the `bigmoe=True` entries. |
| 5/7 | Groups `exclusive: true` | Matrix router: the entries are in no set, so they run alone | The config moved to the matrix router (PR #22). Same eviction semantics. |

Open questions carried to TODO.md, not resolved by the implementation:

- **P3/P4 may be no-ops.** A single-socket 3945WX at NPS1 is one NUMA node, and page migration
  happens *between* nodes, so `--numa distribute` and `numa_balancing=0` should change little.
  Measure before treating them as the explanation for a low STREAM result.
- **CCD count.** The 3945WX (12 cores) is likely two CCDs, not the four §8 assumes; if so, the §8 bandwidth
  expectation (~100–130 GB/s) is optimistic. Check STREAM first.
- **Tool calling.** DSML parsing for DeepSeek has open upstream fixes (#28612, V3.2). Add a
  tool-call probe to §8.

## 11. Candidate models — planned, NOT implemented (2026-09-14)

Four more models for the two-card slot, grouped by how much of each lives in host RAM. Sizes,
architecture and layer/expert counts below were read from the HF repos and GGUF headers.
**Benchmark and throughput claims are as reported, not verified here.**

The budget throughout is the two-card figure used by the existing `-split` entries: ~46.6 GiB
of usable VRAM across both 3090s. Every model here needs both cards, so each is a whole-box
`UNGROUPED_GGUF` entry. It runs alone, and any card request evicts it (§5, §7).

### Summary

| Model | Total / active | Quant (file) | On disk | RAM spill | llama.cpp support | Image | Needs §2 |
|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next | 80B / 3B | `UD-Q4_K_S` or `Q4_K_M` | 42.9 / 45.2 GiB | none – ~3 GiB | `qwen3next` in `b10362` | `llamacpp-mainline` | nothing |
| gpt-oss-120b | 117B / 5.1B | `MXFP4` | 59.0 GiB | ~15–18 GiB | `gpt-oss` in `b10362` | `llamacpp-mainline` | P5 only |
| Mistral Small 4 | 119B / 6.5B | `UD-Q4_K_M` (3 shards) + mmproj | 68.7 + 0.8 GiB | ~23–25 GiB | `mistral4` in `b10362` | `llamacpp-mainline` | P5 only |
| DeepSeek-V4-Flash | 284B / 13B | `UD-Q4_K_XL` (5 shards) | 144.4 GiB | ~110 GiB | `deepseek4` in `b10362` | `llamacpp-v4` | P2–P5 — **implemented** |
| GLM-5.3-Flash | 321B / 18B | `UD-Q4_K_XL` (6 shards) + mmproj | 186.0 + 1.1 GiB | ~145 GiB | **not in mainline** | a PR build | P2–P5, more disk |

"RAM spill" = weights + KV + compute minus the VRAM budget; it becomes `n_cpu_moe`. The
estimates assume 64k context and are to be replaced by measurement, the way §5 starts
DeepSeek at all-experts-on-CPU and walks down.

**Recommended order:** Qwen3-Coder-Next → gpt-oss-120b → Mistral Small 4 → (DeepSeek-V4-Flash,
once P2–P5 land) → GLM-5.3-Flash once llama.cpp merges support. The first three need no new
image and no VM change, and at most ~25 GiB of RAM — they fit even a 128 GiB VM (the VM has 216 GiB, §13). So they can
ship before any §2 prerequisite, and they exercise the RAM-offload path (`GGUF_N_CPU_MOE`, the
poller skip) at low stakes before DeepSeek does.

### 11.1 Qwen3-Coder-Next — GPU-only (or nearly)

- `unsloth/Qwen3-Coder-Next-GGUF`, Apache-2.0. `qwen3next`: 48 layers, 512 experts top-10,
  262144 native context.
- **KV is cheap.** `full_attention_interval=4`, so 12 of 48 layers cache; `head_count_kv=2`,
  `key_length=256` → ~24 KiB/token at f16. 65536 tokens ≈ 1.5 GiB; the full 262144 ≈ 5.9 GiB.
  The Gated DeltaNet layers hold a constant per-slot state.
- **"Fits entirely in 48 GB" is true only below Q4_K_M.** `Q4_K_M` is 45.2 GiB of weights
  against ~46.6 GiB, before KV and compute buffers. Two options:
  - `UD-Q4_K_S` (42.9 GiB) fully on GPU at ~64k context — a thin margin; measure it.
  - `Q4_K_M` / `UD-Q4_K_XL` (45.2 / 46.2 GiB) with `n_cpu_moe` of ~2–4 layers (~0.9 GiB of
    experts each). That is a few percent of the model in RAM, so decode should stay GPU-bound.
  - Pick by measured tok/s, not by quant name.
- Reported: 70.6% SWE-bench Verified; 37–40 tok/s on four MI50s in a fork benchmark.
- Poller: GPU-bound, so **not** `bigmoe` — the idle signal works. If it ends up with more
  than a handful of CPU expert layers, re-check GPU utilisation during decode.
- No new image and no §2 prerequisites. The first model to try.

### 11.2 gpt-oss-120b — slight spill

- `ggml-org/gpt-oss-120b-GGUF`, Apache-2.0. `gpt-oss`: 36 layers, 128 experts top-4,
  131072 context, alternating sliding-window (128) and full attention.
- 59.0 GiB native MXFP4, one file (the "~63 GB" figure is decimal GB). KV: 18 full layers ×
  8 KV heads × 64 dims → ~36 KiB/token, so 131072 ≈ 4.7 GiB.
- Spill ≈ 15–18 GiB → roughly `n_cpu_moe` 10–12 of 36 (~1.6 GiB of experts per layer).
  Start higher and walk down (§5 method).
- The same repo ships an **EAGLE3 drafter** (`eagle3-gpt-oss-120b-Q8_0.gguf`, 0.8 GiB). A/B it
  like DSpark.
- Reasoning/tool format is Harmony. llama.cpp handles it with `--jinja`, but probe tool calls.
- Poller: mark `bigmoe=True` by default. With a third of the experts in RAM, decode GPU
  utilisation may still clear `IDLE_PCT`; drop the flag only if measurement shows it does.

### 11.3 Mistral Small 4 — modest offload

- `unsloth/Mistral-Small-4-119B-2603-GGUF`, Apache-2.0. `mistral4`: 36 layers, 128 experts
  top-4, **multimodal** (mmproj 0.8 GiB), MLA attention (`head_count_kv=1`) → very small KV.
- `UD-Q4_K_M` 68.7 GiB (3 shards; the shard download from §4 applies). Alternatives:
  `UD-Q4_K_S` 64.7 GiB, `MXFP4_MOE` 66.9 GiB. Spill ≈ 23–25 GiB.
- **Context: verify.** The GGUF declares `context_length=1048576`, but the model is described
  as 256K. Take the card's figure as the supported limit, and start at 64k anyway (§5).
- Reported: ~4 GB read per token → 25+ tok/s. That is plausible at 6.5B active with most
  experts on GPU, but unmeasured. Of the RAM-offload tier, the likely daily driver.
- Poller: `bigmoe=True`, same reasoning as 11.2.

### 11.4 DeepSeek-V4-Flash — implemented

See §5 and §10. The "~161 GB" in the draft is 144.4 GiB for the 0731 `UD-Q4_K_XL`.

### 11.5 GLM-5.3-Flash — blocked on llama.cpp

- `zai-org/GLM-5.3-Flash`, MIT, released 2026-08-25. GGUF architecture `glm5next`: 46 blocks,
  288 experts top-8, multimodal (mmproj 1.1 GiB), MLA-style attention.
- **Not loadable by mainline llama.cpp.** `glm5next` is absent from `b10362`, `v0.4.0` and master
  as of 2026-09-14. Four PRs are open: #27752, #27754, #27773, and #27917 (MTP). The unsloth
  GGUFs were therefore converted against an unmerged implementation, so their tensor layout
  may not match whatever merges.
  - Plan: wait for a merge, add a pin that includes it, and **re-download the GGUF from
    after the merge**.
  - Building from a PR head is possible but needs a clonable ref: `Dockerfile.llamacpp` does
    `git clone -b <tag|branch>`.
- **Size is tighter than "~180 GB".** `UD-Q4_K_XL` is 186.0 GiB (~200 GB) across 6 shards.
  - With ~40 GiB on the GPUs, ~145 GiB of experts stay in RAM. That fits a 200 GiB VM, but with
    little room for page cache or anything else.
  - It also exceeds P5's 180 GiB of free disk.
  - If this is the target, size P2 and P5 for it rather than for DeepSeek.
- 18B active vs DeepSeek's 13B → expect ~40% slower decode, since per-token RAM reads scale
  with active expert size.
- Reported: Artificial Analysis index 42 (GLM-5.3 at 45), ahead of DeepSeek V4 Pro. It is
  DeepSeek-V4-Flash's real competitor, pending support.
- Poller: `bigmoe=True`.

### Common implementation notes for §11

- All are `UNGROUPED_GGUF` entries through `gguf-serve.sh`; no new env vars are needed. The
  RAM-offload ones use `n_cpu_moe`, `threads`, `batch`/`ubatch` as in §5.
- Every `bigmoe=True` entry must be added to `BIGMOE_MODELS` in `docker-compose.yml`, or
  `gen_config.py` refuses to generate.
- Pre-download the weights (`hf download … --local-dir /models/hf-cache/gguf/<org>_<repo>`),
  as for DeepSeek. P5 disk now has to cover every model kept on disk at once.
- Per model, record in TODO.md: VRAM per card, container RSS, decode tok/s, 8k prefill tok/s,
  cold/warm load, and a coherence + tool-call probe.

## 12. Hardware change: three A2000s (2026-09-15) — NOT available to this deployment

The box now has 2× RTX 3090 + 3× RTX A2000 12GB (was 2× 3090 + 1× A2000). Both 3090 UUIDs are
unchanged, so the config and poller were unaffected; only `nvidia-smi` indices moved.

**The A2000s are dedicated to other, non-LLM workloads and must not be used here.** One of them
already carries ~4.8 GiB of someone else's allocation. Every model in this repo pins the two
3090s by UUID, which is what keeps that separation honest.

Consequences for the rest of this spec:

- The VRAM budget stays **~46.6 GiB** (two 3090s). §11's sizing already assumes that.
- Models cannot be "rescued" onto an A2000: not the DSpark drafter (§15, also blocked upstream),
  and not the §11 candidates' RAM spill.
- Faster decode for RAM-offload models has to come from host RAM bandwidth, CPU threads, or a
  smaller quant — not from more GPUs.

## 13. Storage (`/fast`) and the VM RAM figure (2026-09-15)

### Storage — implemented

`df -h` on the inference VM:

| Mount | Size | Used | Free | Role |
|---|---|---|---|---|
| `/models` (`/dev/sdb`) | 787 G | 660 G | **88 G** | HF cache: vLLM weights, existing GGUFs, HF token |
| `/fast` (`/dev/sdc`, mirrored NVMe) | 738 G | — | 730 G | large GGUFs (`/fast/gguf`) |

- **P5 failed on `/models`.** 88 G free cannot hold DeepSeek-V4-Flash (144.4 GiB), never mind
  the §11 candidates (~505 GiB together). `/fast` holds all of them with ~225 G to spare.
- **Mechanism:** a spec-level `storage="fast"` in `gen_config.py`.
  - It mounts `/fast/gguf` at `/fast-gguf` and sets `GGUF_DIR`, which `gguf-serve.sh` already honoured.
  - `/models/hf-cache` stays mounted, so the HF token file still resolves.
  - An unknown storage name fails generation.
- `deepseek-v4-flash` uses it. Every §11 candidate should too.
- Disk speed matters for **cold load**: the weights are mmap'd and faulted in from disk.
  - After load, decode reads the page cache; `--mlock` stays off, so the page
    cache remains reclaimable.
  - If a model's CPU-resident part exceeds free RAM, pages are evicted and re-read from
    disk every token. That works on NVMe but is roughly an order of magnitude slower
    (~3–7 GB/s against DDR4's ~100 GB/s): a fallback, not a plan.

Checks before relying on it (TODO.md):

- `lsblk -o NAME,SIZE,ROTA,MODEL,TRAN`, and a sequential read of a large GGUF from each mount.
- Both mounts are virtual disks. If `/fast` is backed by ZFS on the Proxmox host, the host's
  ARC caches the same pages the VM caches — set `primarycache=metadata` on that dataset/zvol,
  or budget host RAM for the double cache.

### RAM — 216 GiB (confirmed 2026-09-15)

`free -g`: 216 GiB total, 212 GiB available, 7 GiB swap. **P2's 200 GiB is met.** An earlier
estimate of ~92 GiB, inferred from tmpfs sizes, was wrong. Those sizes (`/dev/shm` and `/tmp`
46 G) are fixed when the filesystem is mounted, so they reflect RAM at boot rather than now.

That mismatch is itself worth checking. A VM that booted with ~92 GiB and now reports 216 GiB
suggests **memory hotplug or a balloon** on the Proxmox side. P2 asks for fixed memory with
ballooning off, because a balloon reclaiming guest memory under a 145 GiB mmap'd model turns
into page-cache eviction and disk re-reads mid-decode. Check `balloon: 0` and the `memory:` /
`hotplug:` lines in the VM config.

For **2× 3090 + DDR4 only**, at 128k context:

| Model | RAM for offload | + ~10 GiB overhead | Fits 216 GiB? |
|---|---|---|---|
| Qwen3-Coder-Next | ~2–5 GiB | ~15 GiB | yes |
| gpt-oss-120b | ~17–20 GiB | ~30 GiB | yes |
| Mistral Small 4 | ~25–27 GiB | ~37 GiB | yes |
| DeepSeek-V4-Flash | ~116 GiB | ~126 GiB | **yes**, ~90 GiB spare |
| GLM-5.3-Flash | ~146 GiB | ~156 GiB | **yes**, ~60 GiB spare; still blocked on llama.cpp |

DeepSeek-V4-Flash now waits only on the image build (CI) and the remaining host checks: P3/P4,
fixed memory, and the `/fast` download.

## 14. Context: native maximum for the big models (2026-09-15)

Decision: every big model runs at its **native maximum context**, one slot. This replaces §5's
"start at 64k and raise after baselines".

| Model | Context | KV at that context (f16) | RAM @ max, 2× 3090 + DDR4 | Status |
|---|---|---|---|---|
| DeepSeek-V4-Flash | **1,048,576** | ~7 GiB (~13 if f32) | ~122–128 GiB (+10 overhead) | **implemented** |
| Qwen3-Coder-Next | 262,144 | 5.9 GiB | ~5–8 GiB | planned |
| gpt-oss-120b | 131,072 | 4.5 GiB | ~17–20 GiB | planned |
| Mistral Small 4 | 256K per the card (GGUF says 1M) | ~5.5 GiB | ~27–29 GiB | planned; verify the limit |
| GLM-5.3-Flash | 1,048,576 | ~12 GiB (estimate) | ~156 GiB | planned; blocked on llama.cpp |

All fit the 216 GiB VM.

### DeepSeek-V4-Flash KV, from the v0.4.0 source

Read from `src/models/deepseek4.cpp` and `src/llama-kv-cache-dsv4.cpp`:

- `set_swa_pattern(0)` makes **every** layer's raw KV a sliding window of `n_swa=128` tokens.
  - The `compress_ratio=0` layers (2 in the main model, all 3 in the DSpark drafter) are
    therefore constant-size.
  - This rules out the worry that uncompressed layers would scale with context.
- The compressed caches scale with context:
  - `ceil(ctx/ratio)` rows × `n_embd_head=512`, one shared K/V vector per row, for 21 CSA
    layers (ratio 4, plus 128-dim indexer keys) and 20 HCA layers (ratio 128).
  - That is ~6.9 KB/token at f16 → **~6.7 GiB at 1M**, or ~13 GiB if llama.cpp keeps these
    rows at f32.
  - The startup log prints the buffer sizes; record them in TODO.md.
- Compute buffers do not grow with context. The attention mask is
  `min(raw window, n_swa) + top_k` wide.

Costs and risks:

- ~7–13 GiB of VRAM → roughly 2–4 fewer expert layers on GPU than at 64k → somewhat slower decode.
- **Prefill is the practical limit.** §8 targets ≥100 tok/s at 8k. At that rate a
  500k-token prompt takes over an hour of prefill.
  - Capacity for the whole context, not an expectation of filling it interactively.
- If the first load OOMs anyway, fall back in this order:
  1. `cache_type=q8_0` (halves KV)
  2. 393216 (384K, DeepSeek's minimum for "Think Max")
  3. dropping the drafter

## 15. Measured on the host (2026-09-16)

Deployed via PR #23/#24, llama-swap v255+ with the matrix router, `llamacpp-v4` (llama.cpp
v0.4.0, build `b1-5266f24`), weights on `/fast`.

### §8 acceptance criteria

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Decode, single stream | ≥ 8 tok/s | **12.5** (greedy, 300 tok) | pass |
| Decode with DSpark | ≥ 1.4× the above | **0.91×** (11.35 with drafter) | **fail — drafter dropped** |
| Prefill, 8k prompt | ≥ 100 tok/s | **304** (6k prompt) | pass |
| Peak VRAM | ≤ 44 GiB | 21.9 + 22.2 = **44.1 GiB** | at target |
| Peak container RSS | ≤ 180 GiB | ~155 GiB, as reclaimable page cache | pass |
| Warm load | ≤ 90 s | **~7 s** | pass |
| Evict → standby ready | ≤ 60 s | not yet measured | open |

Cold load (first read of 144 GiB off `/fast`) was not timed separately; warm reload is ~7 s.

### Tuning runs (greedy, `temperature 0`, 300 tokens, "Write 300 words about PCIe.")

| Config | tok/s | VRAM c0 / c2 | Note |
|---|---|---|---|
| n_cpu_moe 43, drafter, ts 1,1 | 11.35 | 17.3 / 15.8 | the shipped starting point |
| n_cpu_moe 43, drafter, `--load-mode none` | 10.94 | — | slower; 3 min load; 138 GiB unreclaimable |
| n_cpu_moe 41, drafter, ts 1,1 | 10.57 | — | acceptance fell to 43% |
| n_cpu_moe 39, no drafter, ts 1.3,1 | 12.34 | 11.4 / 22.9 | lopsided |
| n_cpu_moe 39, no drafter, ts 13,1 | 11.76 | 23.0 / 10.7 | lopsided the other way |
| n_cpu_moe 37, no drafter, ts 8,1 | OOM | — | compute buffers, card 0 |
| **n_cpu_moe 36, no drafter, ts 6,1** | **12.50** | **21.9 / 22.2** | **shipped** |

What the runs establish:

1. **The DSpark drafter does not pay for itself.** 10.1 GiB of VRAM spent on it yields 11.35
   tok/s; the same VRAM as expert layers yields 12.5. Acceptance was 43–51% with a mean draft
   length of 2.3–2.5, well below the 1.5–1.9× the model card advertises.
2. **`--load-mode none` is wrong for this box**, despite llama.cpp's own warning. The page cache
   is already warm (216 GiB RAM), so mmap wins; and the flag moved 138 GiB into shared memory.
3. **`tensor_split` is not optional** once `n_cpu_moe` drops. Layers are assigned in order, so
   GPU-resident expert layers land on the last card. `6,1` balances 36/43.
4. **Shaving more CPU layers stops helping.** 36 and 39 measure the same, so expert compute on
   12 threads is a co-bottleneck with memory bandwidth. Faster decode needs more GPU-resident
   *experts* (impossible — only two 3090s are available, §12) or more CPU threads, not a lower
   `n_cpu_moe`.
5. **Speculative-decoding A/Bs need `temperature 0`.** At temp 1.0 the drafted tokens differ
   run to run and swing throughput by ±7%, more than the effects being measured.

### Rejected: the drafter on an A2000 (blocked upstream)

The obvious rescue for the drafter — park its 10.1 GiB on an idle A2000 and keep the 7 expert
layers on the 3090s — does not work on llama.cpp v0.4.0:

- `--spec-draft-device CUDA2` (drafter alone on the A2000) aborts in `graph_reserve`:
  `pre-allocated tensor (output.weight) in a buffer (CUDA1) that cannot run the operation`.
  A coupled drafter (DSpark/DFlash hooks into the target's layers) cannot own a device.
- That is **ggml-org/llama.cpp#26475**, open since 2026-08-02 and reported against this exact
  model. The thread's workaround (give the drafter's device a slice of the target) either
  reintroduces the crash or splits the target across a slow device.
- `--device` order and `--tensor-split` order are NOT the same thing: `-ts` is indexed by
  absolute device number, so reordering `--device` to put the A2000 first does not move the
  split weights with it. That attempt OOMed CUDA1 at 17.6 GiB.

Revisit only if #26475 closes. The upside is bounded: the drafter was worth ≤10% even when its
VRAM was free, against 12.5 tok/s without it. And the A2000s are not ours to use (§12).

### Still open

- Cold-load time from `/fast`, and the evict → `c0.muse-glimmer` path (§8's last row).
- The poller's BIGMOE skip, observed live with DeepSeek resident and the GPUs idle.
- Long-context probe (the 1M context is allocated but only ~6k tokens have been pushed through).
- Tool calling (DSML) and the thinking controls.
