#!/usr/bin/env python3
"""Generate a llama-swap config of co-load PAIRS for benchmarking.

Each single-card model is paired with every other (all C(n,2) unique pairs). A pair
is a `swap:false, exclusive:true` group with one model pinned to 3090 #0 and the other
to 3090 #2 (each owns its card: vLLM TP=1 @ util 0.90; Ternary via the llama.cpp fork
image). No shared-card contention. Callsigns are `pairNN.<model>`; roles
(extractor/judge/...) are assigned by the consuming system, not baked in. Big TP=2
models that need both cards are emitted as ungrouped solo entries.
Regenerate:  python3 gen_pairs_config.py > config.pairs.yaml
"""
import itertools
import json

CARD0 = "GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1"   # 3090 #0
CARD2 = "GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb"   # 3090 #2
IMAGE = "vllm/vllm-openai:v0.26.0"
BONSAI = "ghcr.io/tkontu/bonsai-llama:latest"
# Mainline llama.cpp at a pinned build (Dockerfile.llamacpp). Separate from BONSAI because
# the PrismML fork carries ternary kernels mainline lacks, but its branch head (2026-07-31)
# predates newer architectures — Muse-Glimmer needs b10353+. Neither image serves both.
LLAMACPP = "ghcr.io/tkontu/llamacpp-mainline:latest"

# Uniform concurrency across the whole pool: a pair is only as fast as its slower
# member, so per-model admission limits just create bottlenecks. For vLLM this is
# FREE — the KV pool is preallocated by --gpu-memory-utilization and max-num-seqs
# only governs how much of that (already-bought) pool may be used at once.
CONCURRENCY = 64

# Cards are single-tenant (each model is pinned to one GPU by UUID), so there is no
# competing process to leave room for. The extra 0.05 is ~1.2 GiB that lands entirely
# in the KV pool — a large relative gain on weight-heavy models like gemma-26b, whose
# 16.63 GiB of INT4 weights leave only ~5 GiB of the util-0.90 budget for KV.
UTIL = 0.95

# llama-swap's OWN per-model request cap, in front of the backend. Omitting it (or
# setting 0) uses its internal default of 10, which silently ceilings every model at
# 10 concurrent requests and 429s the rest instantly — regardless of how many slots
# vLLM or llama-server actually has. Set it well above the backend's capacity so the
# backend's scheduler does admission (vLLM queues as 'Waiting'; llama-server queues
# past its -np slots) instead of the proxy rejecting at the door.
REQUEST_LIMIT = 256        # vLLM: 4x max-num-seqs
GGUF_LIMIT_MULT = 4        # llama.cpp: 4x its -np slots
FORK_LIMIT = 8             # ternary is -np 1 (DSpark); keep the queue shallow

# llama.cpp is NOT free: -c is a flat preallocated KV cache split evenly across
# --parallel slots, with no paging, prefix sharing, or preemption. Matching 32
# would mean either 32x the KV VRAM or 1k of context per slot, so the GGUF members
# carry their own (lower) parallelism. See gguf_entry(); GGUF_CTX is PER SLOT.
GGUF_PARALLEL = 8

# Idle unload timeout, in seconds since a model's last request finished. 10x the old
# 1800/3600: a cold start is expensive (gemma-26b @ 65k took ~7 min to ready on
# 2026-08-04) and holding idle weights costs almost nothing here, because every group
# is exclusive — requesting any other pair/solo evicts the resident one immediately
# regardless of TTL. So TTL only decides how long a card stays occupied when NOTHING
# is being served, and the cards are single-tenant.
# The tradeoff it does buy: TTL expiry is the de-facto recycle for a wedged backend
# (see the Xid 31 note in TODO.md), and that now takes 5h instead of 30 min — unload
# by hand (POST /api/models/unload) if a model misbehaves.
TTL = 18000        # 5 h  — pool + pair members
TTL_SOLO = 36000   # 10 h — TP=2 solo models (slowest to reload, own both cards)

# Single-card pool. Each entry is a dict keyed by "backend":
#   vllm: repo, mml, eager, think_off              (vLLM container, TP=1 @ util 0.90;
#                                                   think_off=True emits the
#                                                   enable_thinking:false filter)
#   fork: (none)                                   (Ternary via the PrismML bonsai image entrypoint)
#   gguf: repo, hf_file, ctx, par                  (standard GGUF via the bonsai image's llama-server)
# Concurrency is NOT per-model: vLLM members all use CONCURRENCY, GGUF members all
# use GGUF_PARALLEL. Only override `par` when a model genuinely can't hold the KV.
# Prefix caching on the Qwen3.5 GDN hybrids: vLLM auto-disables APC for hybrid
# attention+Mamba models unless asked (verified live: cache_config_info showed
# enable_prefix_caching=False, prefix_cache_queries_total stuck at 0). align is the
# only mamba cache mode the Qwen3.5 family supports, and it is EXPERIMENTAL:
#  - hits are per completed block and the attention block is padded to 528 tokens,
#    so shared prefixes under 528 tokens hit 0% — short system prompts gain nothing;
#  - align keeps only the Mamba checkpoint at the last block boundary; if that lands
#    in request-unique tokens the per-group intersection zeroes ALL reuse;
#  - do NOT combine with MTP (--speculative-config) until tested separately: the
#    combo has crashed during cudagraph profiling on hybrid Mamba models.
# Verify after each deploy: cache_config shows enable_prefix_caching=True and
# vllm:prefix_cache_hits_total moves under a repeated-prefix workload.
APC_ALIGN = ("--enable-prefix-caching", "--mamba-cache-mode align")

POOL = [
    # 65536: measured 19882 MiB @ 16800 and 20552 MiB @ 32768 (TP=1, kv_seqs 1,
    # vllm_refs/memory_footprints.json) → ~43 KiB/token, so 65536 extrapolates to
    # ~21.9 GiB against the util-0.95 budget of ~23.3 GiB on a 3090 (~1.4 GiB slack).
    # ~98k is the theoretical fp16-KV ceiling — do not raise further without fp8 KV.
    dict(tok="gemma-26b",   backend="vllm", repo="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",      mml=65536, think_off=True),
    dict(tok="phi-4",       backend="vllm", repo="stelterlab/phi-4-AWQ",                          mml=16384),
    dict(tok="gemma-12b",   backend="vllm", repo="cyankiwi/gemma-4-12B-it-qat-AWQ-INT4",          mml=32000),
    dict(tok="gemma-e4b",   backend="vllm", repo="cyankiwi/gemma-4-E4B-it-qat-AWQ-INT4",          mml=128000),
    # BF16-INT4 replaces AWQ-4bit: linear_attn (GDN) layers stay unquantized BF16 —
    # safer for this family. Shard naming verified 2026-08-04 against the known
    # silent-failure mode (ignore list vs shards BOTH use the split in_proj_qkv/z/b/a
    # names, and no linear_attn.*weight_scale exists → the ignore list matches; a
    # mismatch would make vLLM skip-load those layers and serve incoherent output).
    # Still run a coherence prompt on first load rather than trusting a clean start.
    # think=True: without the filter the 9B burns hundreds of output tokens in its
    # thinking phase even at temperature 0 (verified on first load, 2026-08-04) —
    # short-max_tokens requests never reach an answer.
    dict(tok="qwen3.5-9b",  backend="vllm", repo="cyankiwi/Qwen3.5-9B-AWQ-BF16-INT4",             mml=16384, think_off=True, extra=APC_ALIGN),
    # 32k: measured at 8156 MiB for weights+KV @ 16384x2 (vllm_refs/memory_footprints.json),
    # i.e. ~150 KiB/token, so the util-0.90 pool (~18 GiB after weights) holds ~120k tokens
    # — far more than one 32768-token sequence. Raising mml costs no VRAM, same as seqs.
    # No --enforce-eager: it was inherited from the old Qwen3.5-4B-AWQ-4bit-shortkv entry,
    # where disabling CUDA graphs reclaimed their VRAM reserve. That no longer applies at
    # util 0.95, and eager costs the most on small models (launch overhead dominates decode).
    # The documented Xid 31 / AWQ-MoE eager mitigation is for Qwen3.6-35B-A3B, not this model.
    dict(tok="qwen3.5-4b",  backend="vllm", repo="cyankiwi/Qwen3.5-4B-AWQ-4bit",                  mml=32768, think_off=True, extra=APC_ALIGN),
    dict(tok="mellum2-12b", backend="vllm", repo="cyankiwi/Mellum2-12B-A2.5B-Instruct-AWQ-INT4",  mml=128000),
    dict(tok="ternary",     backend="fork"),
    dict(tok="qwythos-v2",  backend="gguf", repo="empero-ai/Qwythos-9B-v2-GGUF", hf_file="Qwythos-9B-v2-Q4_K_M.gguf", ctx=8192),
    # Xet-backed repo (~11.3 GB Q6_K). llama-server -hf downloads via HTTP; if Xet blocks
    # that, we pre-download with the `hf` CLI (+hf_xet) instead. See README.
    # Q6_K weights are ~11.3 GB of the 24 GB card, so it gets fewer slots than qwythos.
    dict(tok="fablevibes",  backend="gguf", repo="tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF", hf_file="Qwen3.6-14B-A3B-FableVibes-Q6_K.gguf", ctx=8192, par=4),
    # MTP variant (self-speculative) — uncomment to add as its own pool member (needs a load test):
    # dict(tok="qwythos-v2-mtp", backend="gguf", repo="empero-ai/Qwythos-9B-v2-GGUF", hf_file="Qwythos-9B-v2-MTP-Q4_K_M.gguf", ctx=32768),
]

# Solo big models (need both 3090s → TP=2 → no partner).
# (id, repo, mml, seqs, util, think_off, eager)
# eager=True emits --enforce-eager. Only 35B-A3B needs it: vLLM's AWQ-MoE kernels
# fault with Xid 31 mid-inference and CUDA graphs are the likely trigger — see
# ARCHITECTURE.md "Known issues" and README.md "Operational notes". Keep it until
# TODO.md's dmesg check confirms the crash is resolved.
SOLO = [
    ("Qwen3.6-35B-A3B-AWQ-4bit",   "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",         131072, 1, 0.90, True,  True),
    ("Qwen3.6-27B-AWQ-INT4",       "cyankiwi/Qwen3.6-27B-AWQ-INT4",             262144, 1, 0.92, False, False),
    ("gemma4-26B-A4B-it-INT4-max", "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",  131072, 1, 0.90, False, False),
    ("Qwythos-9B-Claude-Mythos-5-1M", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M", 256000, 1, 0.90, False, False),
]

# Ungrouped GGUF entries — NOT in POOL, so they get no pairNN membership. Muse-Glimmer is
# excluded from POOL deliberately: it is on-call standby, not a co-load partner, and adding
# it there would have generated 10 extra pairs (45 -> 55) that nothing would ever request.
#
# Both run -np 1 (no parallelism) with the full native 131072 context in a single slot, and
# f16 KV (no cache quant). That is affordable because KV is unusually cheap on this model:
# 52 layers, num_key_value_heads=2, head_dim=128, and a 3:1 sliding/full split (39 sliding
# layers windowed at 2048, 13 full). The full layers cost 13 KiB/token -> 1.74 GiB at
# 131072; the sliding layers a flat 78 MiB at -np 1. ~1.82 GiB for the whole context.
# (-np 1 is also marginally cheaper than -np 4: llama.cpp gives each slot its own sliding
# window, so parallelism multiplies that 78 MiB while leaving the full-layer cost fixed.)
#
# `cards`: one 3090 for the standby entry, both for the split one. -sm layer is PIPELINE
# parallel — activations cross PCIe once per layer boundary, so the no-NVLink constraint
# that hurts vLLM TP=2 does not bite. It buys CAPACITY, not speed: only one card computes
# at a time, so decode is ~single-card. The point is to afford the 19.65 GB dynamic quant
# plus vision plus the drafter, which will not fit on one 3090 (22.68 GB of weights against
# a ~23.3 GB budget).
UNGROUPED_GGUF = [
    # On-call standby (see scripts/oncall-wakeup.sh). ttl 0 = never idle-unload; it is still
    # evicted by any exclusive group, which is exactly what we want — hence NOT persistent.
    #
    # MEASURED on the host 2026-08-12 (do not re-derive from HF's file sizes: those are
    # DECIMAL GB, and treating them as GiB overstates the weights by ~7%). With weights +
    # mmproj only, llama-server reported n_ctx=131072, vision=True, and used 18.79 GiB of
    # the 24 GiB card — i.e. 15.61 (weights) + 1.30 (mmproj) + 1.88 (KV + compute).
    # The 1.88 confirms the sliding-window KV analysis above: the full 131072 context really
    # does cost under ~2 GiB.
    #
    # That leaves 5.21 GiB free, so the dflash drafter (1.52 GiB) fits with ~3.7 GiB spare.
    # It is ON because the ~3.1x decode speedup is the reason to run llama.cpp here at all,
    # and the model card measured it at batch size 1 greedy — exactly this -np 1 setup.
    # Standard flags (-md + -ngld 99), unlike Ternary-Bonsai's DSpark --spec-type.
    # A "[spec] failed to measure draft model memory" warning at startup is documented as
    # harmless. The dynamic quant is NOT used here: dynamic+mmproj+dflash needs 23.01 GiB,
    # leaving ~1 GiB — too thin for prefill spikes at 131k. That is what the split entry is for.
    dict(tok="muse-glimmer", image=LLAMACPP, cards=[CARD0], ttl=0, oncall=True,
         repo="meta-models/Muse-Glimmer-30B-GGUF",
         hf_file="muse-glimmer-30B-kquant-17gb.gguf",
         mmproj="mmproj-kquant.gguf", draft="dflash-kquant.gguf",
         spec_type="draft-dflash", ctx=131072, par=1),
    dict(tok="Muse-Glimmer-30B-split", image=LLAMACPP, cards=[CARD0, CARD2], ttl=TTL_SOLO,
         repo="meta-models/Muse-Glimmer-30B-GGUF",
         hf_file="muse-glimmer-30B-kquant-dynamic.gguf",
         mmproj="mmproj-kquant.gguf", draft="dflash-kquant.gguf",
         spec_type="draft-dflash", ctx=131072, par=1, split_mode="layer", tensor_split="1,1"),
]

def setparams_filter(**kwargs):
    """Render a llama-swap `filters.setParams.chat_template_kwargs` block.

    Backend-agnostic — it rewrites the request before the proxy forwards it, so it works
    for llama.cpp (--jinja applies the template) as well as vLLM. Used for
    enable_thinking:false on the Qwen/gemma entries, and available for Muse-Glimmer's
    reasoning_strength (low|medium|high|xhigh).
    """
    if not kwargs:
        return ""
    out = "    filters:\n      setParams:\n        chat_template_kwargs:\n"
    return out + "".join(f"          {k}: {json.dumps(v)}\n" for k, v in kwargs.items())


THINK_FILTER = setparams_filter(enable_thinking=False)


def vllm_entry(model_id, repo, gpus, mml, seqs, eager, think_off, tp=1, util=UTIL, ttl=TTL,
               climit=REQUEST_LIMIT, extra=()):
    # NOTE: seqs (--max-num-seqs) costs no VRAM. The KV pool is sized once at startup
    # from util; this only caps how many sequences may share it. Oversubscribing
    # degrades via preemption/recompute, never OOM.
    eager_line = "      --enforce-eager\n" if eager else ""
    extra_lines = "".join(f"      {flag}\n" for flag in extra)
    e = (
        f'  "{model_id}":\n'
        f"    cmd: |\n"
        f"      docker run --rm --name ${{MODEL_ID}}\n"
        f"      -e HUGGING_FACE_HUB_TOKEN=${{env.HF_TOKEN}}\n"
        f"      -v /models/hf-cache:/root/.cache/huggingface\n"
        f"      -p ${{PORT}}:8000\n"
        f"      --gpus '\"device={gpus}\"'\n"
        f"      {IMAGE}\n"
        f"      --model {repo}\n"
        f"      --served-model-name ${{MODEL_ID}}\n"
        f"      --tensor-parallel-size {tp} --gpu-memory-utilization {util}\n"
        f"      --max-model-len {mml} --max-num-seqs {seqs}\n"
        f"{eager_line}"
        f"{extra_lines}"
        f"      --port 8000\n"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    ttl: {ttl}\n"
        f"    concurrencyLimit: {climit}\n"
    )
    if think_off:
        e += THINK_FILTER
    return e


def fork_entry(model_id, gpus, ttl=TTL):
    # Ternary-Bonsai via the PrismML llama.cpp fork image (see Dockerfile.bonsai).
    # The entrypoint discovers weights + applies vision/DSpark/tool flags; listens on 8080.
    return (
        f'  "{model_id}":\n'
        f"    cmd: |\n"
        f"      docker run --rm --name ${{MODEL_ID}}\n"
        f"      --pull=always\n"
        f"      --gpus '\"device={gpus}\"'\n"
        f"      -e HF_TOKEN=${{env.HF_TOKEN}}\n"
        f"      -e HUGGING_FACE_HUB_TOKEN=${{env.HF_TOKEN}}\n"
        f"      -v /models/hf-cache:/root/.cache/huggingface\n"
        f"      -p ${{PORT}}:8080\n"
        f"      {BONSAI}\n"
        f"      --alias ${{MODEL_ID}}\n"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    checkEndpoint: /health\n"
        f"    ttl: {ttl}\n"
        f"    concurrencyLimit: {FORK_LIMIT}\n"
    )


def gguf_entry(model_id, gpus, repo, hf_file, ctx, par, ttl=TTL, image=BONSAI,
               mmproj=None, draft=None, spec_type=None, draft_max=None, split_mode=None,
               tensor_split=None, cache_type=None, params=None):
    # Standard GGUF via the gguf-serve.sh entrypoint (present in BOTH images): it
    # downloads the file(s) with the `hf` CLI (HTTPS + gated + Xet) into the mounted
    # cache, then serves the local file with llama-server (this build's llama-server has
    # no HTTPS itself). GGUF_CTX is PER SLOT; gguf-serve.sh multiplies it by
    # GGUF_PARALLEL for -c, so KV VRAM here scales linearly with parallelism (unlike vLLM).
    # The optional args map 1:1 onto gguf-serve.sh's GGUF_* env vars; omitting them all
    # reproduces the original single-file, single-GPU command exactly.
    opt = ""
    if mmproj:
        opt += f"      -e GGUF_MMPROJ={mmproj}\n"
    if draft:
        opt += f"      -e GGUF_DRAFT={draft}\n"
    if spec_type:
        opt += f"      -e GGUF_SPEC_TYPE={spec_type}\n"
    if draft_max:
        opt += f"      -e GGUF_DRAFT_MAX={draft_max}\n"
    if split_mode:
        opt += f"      -e GGUF_SPLIT_MODE={split_mode}\n"
    if tensor_split:
        opt += f"      -e GGUF_TENSOR_SPLIT={tensor_split}\n"
    if cache_type:
        opt += f"      -e GGUF_CACHE_TYPE={cache_type}\n"
    e = (
        f'  "{model_id}":\n'
        f"    cmd: |\n"
        f"      docker run --rm --name ${{MODEL_ID}}\n"
        f"      --pull=always\n"
        f"      --entrypoint gguf-serve.sh\n"
        f"      --gpus '\"device={gpus}\"'\n"
        f"      -e HF_TOKEN=${{env.HF_TOKEN}}\n"
        f"      -e HUGGING_FACE_HUB_TOKEN=${{env.HF_TOKEN}}\n"
        f"      -e GGUF_REPO={repo}\n"
        f"      -e GGUF_FILE={hf_file}\n"
        f"      -e GGUF_CTX={ctx}\n"
        f"      -e GGUF_PARALLEL={par}\n"
        f"{opt}"
        f"      -v /models/hf-cache:/root/.cache/huggingface\n"
        f"      -p ${{PORT}}:8080\n"
        f"      {image}\n"
        f"      --alias ${{MODEL_ID}}\n"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    checkEndpoint: /health\n"
        f"    ttl: {ttl}\n"
        f"    concurrencyLimit: {par * GGUF_LIMIT_MULT}\n"
    )
    if params:
        e += setparams_filter(**params)
    return e


def member_entry(spec, model_id, card):
    b = spec["backend"]
    if b == "fork":
        return fork_entry(model_id, card)
    if b == "gguf":
        return gguf_entry(model_id, card, spec["repo"], spec["hf_file"], spec["ctx"],
                          spec.get("par", GGUF_PARALLEL))
    return vllm_entry(model_id, spec["repo"], card, spec["mml"], CONCURRENCY,
                      spec.get("eager", False), spec.get("think_off", False),
                      extra=spec.get("extra", ()))


def ungrouped_gguf_entry(spec):
    return gguf_entry(spec["tok"], ",".join(spec["cards"]), spec["repo"], spec["hf_file"],
                      spec["ctx"], spec.get("par", 1), ttl=spec.get("ttl", TTL),
                      image=spec.get("image", BONSAI),
                      mmproj=spec.get("mmproj"), draft=spec.get("draft"),
                      spec_type=spec.get("spec_type"), draft_max=spec.get("draft_max"),
                      split_mode=spec.get("split_mode"),
                      tensor_split=spec.get("tensor_split"),
                      cache_type=spec.get("cache_type"),
                      params=spec.get("params"))


def main():
    out = []
    out.append("# llama-swap PAIRS config (GENERATED by gen_pairs_config.py — do not hand-edit).")
    out.append("# Each pairNN is a co-load group: two single-card models, one per 3090, serving")
    out.append("# concurrently. Callsign = pairNN.<model> (roles are assigned by the consuming")
    out.append("# system; split on the FIRST '.' to get the pair id). Only one pair (or one solo")
    out.append("# model) is resident at a time; requesting another swaps it in.")
    out.append("# Regenerate: python3 gen_pairs_config.py > config.pairs.yaml")
    out.append("")
    out.append("healthCheckTimeout: 900")
    out.append("logLevel: info")
    out.append("")
    out.append("models:")
    out.append("")

    groups = []
    pairs = list(itertools.combinations(range(len(POOL)), 2))
    for k, (i, j) in enumerate(pairs, start=1):
        pair = f"pair{k:02d}"
        a, b = POOL[i], POOL[j]
        id_a = f"{pair}.{a['tok']}"      # -> 3090 #0
        id_b = f"{pair}.{b['tok']}"      # -> 3090 #2
        out.append(f"  # ===== {pair}: {a['tok']} (#0)  +  {b['tok']} (#2) =====")
        out.append(member_entry(a, id_a, CARD0))
        out.append(member_entry(b, id_b, CARD2))
        groups.append((pair, id_a, id_b))

    out.append("  # ===== Standalone entries (ungrouped): load one pooled model solo =====")
    out.append("  # Callsign = the base token (no pairNN prefix). Requesting one loads it alone")
    out.append("  # (exclusive swap unloads whatever else is resident). TP=1 on 3090 #0.")
    for spec in POOL:
        out.append(member_entry(spec, spec["tok"], CARD0))

    out.append("  # ===== Solo big models (TP=2, own both 3090s — no partner possible) =====")
    for (mid, repo, mml, seqs, util, think_off, eager) in SOLO:
        out.append(vllm_entry(mid, repo, f"{CARD0},{CARD2}", mml, seqs, eager, think_off, tp=2, util=util, ttl=TTL_SOLO))

    out.append("  # ===== Ungrouped GGUF entries (no pairNN membership; see UNGROUPED_GGUF) =====")
    for spec in UNGROUPED_GGUF:
        out.append(ungrouped_gguf_entry(spec))

    out.append("")
    out.append("# Each pair is its own group: members co-load and stay together (swap:false);")
    out.append("# loading a pair/solo unloads the others (exclusive).")
    out.append("groups:")
    for (pair, id_a, id_b) in groups:
        out.append(f"  {pair}:")
        out.append(f"    swap: false")
        out.append(f"    exclusive: true")
        out.append(f"    persistent: false")
        out.append(f"    members:")
        out.append(f'      - "{id_a}"')
        out.append(f'      - "{id_b}"')

    print("\n".join(out))


if __name__ == "__main__":
    main()
