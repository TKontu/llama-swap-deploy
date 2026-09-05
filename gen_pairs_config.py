#!/usr/bin/env python3
"""Generate a llama-swap config of co-load PAIRS for benchmarking.

Members are paired unless they are near-duplicates — the rule is "pair everything EXCEPT
same family AND same role" (see worth_pairing() in main()). Each POOL entry carries a
`family` and a `role` (anchor = heavy/high-KV, fast = small/high-concurrency). A pair
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

# Chat-template DEFAULTS shared by both Qwen3.8-27B entries. Server-side only: a request that
# sends its own chat_template_kwargs still wins (see the merge note in gguf_entry). Both keys
# were read off Qwen3.8-27B's chat_template.jinja, not the model card:
#
#   reasoning_effort   the template does `reasoning_effort|default('xhigh')`, so leaving it
#                      unset means EVERY request runs in xhigh — the most expensive reasoning
#                      mode. That is the qwen3.5-9b / muse-glimmer trap a third time: a
#                      short-max_tokens request spends the whole budget in reasoning_content
#                      and returns empty content with finish_reason=length. `low` is the right
#                      floor; callers who want depth can ask per request.
#                      VALID VALUES ARE ONLY xhigh|medium|low — the template calls
#                      raise_exception() on anything else, so a client sending the ordinary
#                      OpenAI "high" gets a hard template error rather than a graceful
#                      fallback. Worth knowing before pointing an OpenAI-shaped client at it.
#   preserve_thinking  already defaults to true in the template
#                      (`preserve_thinking is undefined or preserve_thinking is true`), so
#                      this is a NO-OP today. Pinned explicitly because the embedded template
#                      travels with the GGUF and moves whenever the repo is requantized.
QWEN38_TEMPLATE_KWARGS = dict(reasoning_effort="low", preserve_thinking=True)

# Same idea for Muse-Glimmer, but the knob has a DIFFERENT NAME and different semantics — read
# off the template embedded in muse-glimmer-30B-kquant-17gb.gguf, which is the one llama.cpp
# actually applies (not the HF repo's, and not the model card's example):
#
#   {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength
#                else 'high' -%}
#
#   * the key is `reasoning_strength`, NOT `reasoning_effort` (Qwen3.8's key). Passing the
#     wrong one is silently ignored — it just falls through to the default.
#   * the default is 'high', not Qwen3.8's 'xhigh'.
#   * valid values are xhigh|high|medium|low, and unlike Qwen3.8 there is NO raise_exception:
#     the value is interpolated straight into a "Reasoning strength: X." line, so a typo
#     degrades the prompt quietly rather than erroring.
#
# One caveat this default cannot beat: the template only injects that line
# `{%- if 'reasoning strength' not in (sys_text | lower) -%}`, and it first rewrites any
# "reasoning effort" in the system text to "reasoning strength". So a system prompt that
# mentions either phrase suppresses this default entirely and wins. That is a feature (callers
# can steer it inline) but it means the default is not a guarantee.
MUSE_TEMPLATE_KWARGS = dict(reasoning_strength="low")


def sampling_args(**kw):
    """Render model-card sampling defaults as llama-server CLI flags (forwarded via "$@").

    These MUST be passed explicitly. llama.cpp does NOT read the `general.sampling.*` keys
    some GGUFs carry (Qwen3.8's has them; grep b10362 for the key — llama-model-loader.cpp
    and common.cpp never look at it), so without these flags every model silently runs on
    llama.cpp's own defaults from common.h:

        top_k = 40    top_p = 0.95    min_p = 0.05    temp = 0.80

    which match neither model card. min_p in particular is the quiet one: 0.05 is a llama.cpp
    invention that truncates the tail, and Qwen3.8 explicitly asks for 0.0.

    Server-level defaults only — a request's own temperature/top_p/top_k still wins, same as
    the chat-template kwargs above.
    """
    flag = {"temp": "--temp", "top_p": "--top-p", "top_k": "--top-k",
            "min_p": "--min-p", "presence_penalty": "--presence-penalty"}
    return "".join(f"      {flag[k]} {v}\n" for k, v in kw.items())


# Muse-Glimmer card: temperature 1.0, top_p 0.95, top_k 64. Its GGUF carries NO general.sampling.*
# keys at all, so before this it ran at temp 0.80 / top_k 40 — wrong on two of the three.
MUSE_SAMPLING = sampling_args(temp=1.0, top_p=0.95, top_k=64, min_p=0.0)

# Qwen3.8 card, THINKING mode (which is the default here): temperature 1.0, top_p 0.95,
# top_k 20, min_p 0.0, presence_penalty 0.0. Callers who disable thinking should override to
# the card's instruct values (temp 0.7, top_p 0.80, presence_penalty 1.5) per request.
QWEN38_SAMPLING = sampling_args(temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)

POOL = [
    # 65536: measured 19882 MiB @ 16800 and 20552 MiB @ 32768 (TP=1, kv_seqs 1,
    # vllm_refs/memory_footprints.json) → ~43 KiB/token, so 65536 extrapolates to
    # ~21.9 GiB against the util-0.95 budget of ~23.3 GiB on a 3090 (~1.4 GiB slack).
    # ~98k is the theoretical fp16-KV ceiling — do not raise further without fp8 KV.
    dict(tok="gemma-26b",   family="gemma", role="anchor", backend="vllm", repo="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",      mml=65536, think_off=True),
    dict(tok="gemma-e4b",   family="gemma", role="fast", backend="vllm", repo="cyankiwi/gemma-4-E4B-it-qat-AWQ-INT4",          mml=128000),
    # BF16-INT4 replaces AWQ-4bit: linear_attn (GDN) layers stay unquantized BF16 —
    # safer for this family. Shard naming verified 2026-08-04 against the known
    # silent-failure mode (ignore list vs shards BOTH use the split in_proj_qkv/z/b/a
    # names, and no linear_attn.*weight_scale exists → the ignore list matches; a
    # mismatch would make vLLM skip-load those layers and serve incoherent output).
    # Still run a coherence prompt on first load rather than trusting a clean start.
    # think=True: without the filter the 9B burns hundreds of output tokens in its
    # thinking phase even at temperature 0 (verified on first load, 2026-08-04) —
    # short-max_tokens requests never reach an answer.
    dict(tok="qwen3.5-9b",  family="qwen3.5", role="fast", backend="vllm", repo="cyankiwi/Qwen3.5-9B-AWQ-BF16-INT4",             mml=16384, think_off=True, extra=APC_ALIGN),
    # 32k: measured at 8156 MiB for weights+KV @ 16384x2 (vllm_refs/memory_footprints.json),
    # i.e. ~150 KiB/token, so the util-0.90 pool (~18 GiB after weights) holds ~120k tokens
    # — far more than one 32768-token sequence. Raising mml costs no VRAM, same as seqs.
    # No --enforce-eager: it was inherited from the old Qwen3.5-4B-AWQ-4bit-shortkv entry,
    # where disabling CUDA graphs reclaimed their VRAM reserve. That no longer applies at
    # util 0.95, and eager costs the most on small models (launch overhead dominates decode).
    # The documented Xid 31 / AWQ-MoE eager mitigation is for Qwen3.6-35B-A3B, not this model.
    dict(tok="qwen3.5-4b",  family="qwen3.5", role="fast", backend="vllm", repo="cyankiwi/Qwen3.5-4B-AWQ-4bit",                  mml=32768, think_off=True, extra=APC_ALIGN),
    dict(tok="ternary",     family="qwen3.6", role="anchor", backend="fork"),
    dict(tok="qwythos-v2",  family="qwythos", role="fast", backend="gguf", repo="empero-ai/Qwythos-9B-v2-GGUF", hf_file="Qwythos-9B-v2-Q4_K_M.gguf", ctx=8192),
    # Xet-backed repo (~11.3 GB Q6_K). llama-server -hf downloads via HTTP; if Xet blocks
    # that, we pre-download with the `hf` CLI (+hf_xet) instead. See README.
    # Q6_K weights are ~11.3 GB of the 24 GB card, so it gets fewer slots than qwythos.
    dict(tok="fablevibes",  family="qwen3.6", role="fast", backend="gguf", repo="tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF", hf_file="Qwen3.6-14B-A3B-FableVibes-Q6_K.gguf", ctx=8192, par=4),
    # Qwen3.8-27B — dense 27B, hybrid Gated DeltaNet, native vision. First POOL member on
    # the MAINLINE image (the rest of the GGUF pool runs the bonsai build): its GGUF declares
    # general.architecture=qwen35 and the mmproj clip.projector_type=qwen3vl_merger, BOTH of
    # which the pinned b10362 already supports (Qwen3.5 + Qwen3-VL predate the b10353 Muse
    # Glimmer cut), so this needs no image bump and no rebuild. Verified by reading the GGUF
    # headers directly, not the model card.
    #
    # KV is unusually cheap for a 27B, which is what makes it a viable co-load member:
    # qwen35.full_attention_interval=4, so only 16 of the 64 layers cache anything; the other
    # 48 are Gated DeltaNet with a constant-size recurrent state (ssm.state_size=128,
    # inner_size=6144 -> ~75 MiB per SLOT, not per token). At head_count_kv=4 and
    # key_length=value_length=256 that is 4*256*2B*2 = 4 KiB/layer/token * 16 = 64 KiB/token
    # f16 -> 16384*4 = 65536 tokens costs exactly 4.00 GiB.
    # Against the 23.3 GiB card budget: 16.69 (UD-Q4_K_XL) + 0.86 (mmproj-F16) + 4.00 (KV)
    # = 21.55, leaving ~1.75 GiB for the DeltaNet states (~0.3 GiB at par=4) and prefill.
    # UD (Unsloth Dynamic v3.0) over plain Q4_K_M (15.93 GiB): +0.76 GiB buys the
    # keep-sensitive-tensors-wider treatment, which pays off most at 4-bit. If the fit turns
    # out tight on the host, drop to Qwen3.8-27B-Q4_K_M.gguf for +0.76 GiB of slack before
    # touching ctx/par.
    # Thinking is ON by default in this family and is NOT filtered off here: the model card's
    # reasoning_effort/preserve_thinking controls are per-request, and llama.cpp returns the
    # trace in message.reasoning_content (see the muse-glimmer note in TODO.md) rather than
    # burning the content budget the way qwen3.5-9b does.
    dict(tok="qwen3.8-27b", family="qwen3.8", role="anchor", backend="gguf", image=LLAMACPP,
         repo="unsloth/Qwen3.8-27B-GGUF", hf_file="Qwen3.8-27B-UD-Q4_K_XL.gguf",
         mmproj="mmproj-F16.gguf", ctx=16384, par=4,
         template_kwargs=QWEN38_TEMPLATE_KWARGS, sampling=QWEN38_SAMPLING),
    # Muse-Glimmer-30B — the on-call standby, and now also a pooled ANCHOR so it can be
    # benchmarked head-to-head against the other strong models. It was deliberately kept out of
    # POOL before ("on-call standby, not a co-load partner"); that reasoning was about avoiding
    # 10 pairs nobody would request, and it no longer holds now that pairs are filtered by
    # family+role instead of enumerated exhaustively.
    #
    # `standalone_ttl=0` preserves the on-call contract: the bare `muse-glimmer` entry never
    # idle-unloads (scripts/oncall-wakeup.sh wakes it by that exact name), while its pairNN
    # members age out on the normal TTL. That is what the README's on-call section already
    # described — "its pairNN. members keep the normal 5 h TTL" — which was aspirational until
    # now, since it had no pair members at all.
    #
    # Still NOT persistent: every pair/solo group is exclusive, so any other request evicts it.
    # That is the design; persistent:true would pin the cards and starve everything else.
    #
    # Fits one card with room to spare — MEASURED 20.82 GiB of ~23.3 on 3090 #0 with weights +
    # mmproj + the dflash drafter at the full 131072 context (-np 1). Its pair partner sits on
    # the other 3090, so pairing costs it nothing.
    dict(tok="muse-glimmer", family="muse", role="anchor", backend="gguf", image=LLAMACPP,
         repo="meta-models/Muse-Glimmer-30B-GGUF",
         hf_file="muse-glimmer-30B-kquant-17gb.gguf",
         mmproj="mmproj-kquant.gguf", draft="dflash-kquant.gguf",
         spec_type="draft-dflash", ctx=131072, par=1, standalone_ttl=0,
         template_kwargs=MUSE_TEMPLATE_KWARGS, sampling=MUSE_SAMPLING),
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
    dict(tok="Muse-Glimmer-30B-split", image=LLAMACPP, cards=[CARD0, CARD2], ttl=TTL_SOLO,
         repo="meta-models/Muse-Glimmer-30B-GGUF",
         hf_file="muse-glimmer-30B-kquant-dynamic.gguf",
         mmproj="mmproj-kquant.gguf", draft="dflash-kquant.gguf",
         spec_type="draft-dflash", ctx=131072, par=1, split_mode="layer", tensor_split="1,1",
         template_kwargs=MUSE_TEMPLATE_KWARGS, sampling=MUSE_SAMPLING),
    # Qwen3.8-27B at the full native 262144 context, across BOTH 3090s. Same shape and same
    # bargain as Muse-Glimmer-30B-split: -sm layer is PIPELINE parallel, so it buys CAPACITY,
    # not speed (measured 78.2 vs 78.8 tok/s on Muse-Glimmer — identical within noise). It
    # exists because 262k of context at a >4-bit quant does not fit one card, full stop.
    # 24.14 (UD-Q6_K_XL) + 0.86 (mmproj-F16) + 16.00 (262144 tok * 64 KiB f16 KV) = 41.00 GiB
    # of the ~46.6 GiB two-card budget -> ~5.6 GiB slack.
    #
    # Q6 over Q8_0 (27.05 GiB): both quants own both cards, so Q8 buys no deployment
    # flexibility over Q6 — only quality — and it is on the flattest part of that curve while
    # cutting the slack to ~2.7 GiB at 262k. UD-Q6_K_XL already keeps the sensitive tensors at
    # 8-bit, so it lands at ~Q8 quality for 2.9 GiB less. Q8_0 does NOT fit a single 3090
    # either way (27.05 GiB of weights against a ~23.3 GiB budget), so there is no third
    # option where it earns its size.
    #
    # No drafter, unlike muse-glimmer: unsloth ships no standalone DFlash/MTP GGUF here, and
    # while the weights carry a packed MTP layer (qwen35.nextn_predict_layers=1) mainline does
    # not self-speculate off it. So there is no 1.81x to be had — this entry is quality and
    # context only. Revisit if an MTP GGUF appears (cf. the commented qwythos-v2-mtp in POOL).
    dict(tok="Qwen3.8-27B-split", image=LLAMACPP, cards=[CARD0, CARD2], ttl=TTL_SOLO,
         repo="unsloth/Qwen3.8-27B-GGUF",
         hf_file="Qwen3.8-27B-UD-Q6_K_XL.gguf",
         mmproj="mmproj-F16.gguf",
         ctx=262144, par=1, split_mode="layer", tensor_split="1,1",
         template_kwargs=QWEN38_TEMPLATE_KWARGS, sampling=QWEN38_SAMPLING),
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
               tensor_split=None, cache_type=None, params=None, template_kwargs=None,
               sampling=None):
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
    if template_kwargs:
        # SERVER-SIDE DEFAULTS for the jinja chat template, NOT a forced override. This is
        # llama-server's own --chat-template-kwargs, reached via its LLAMA_ARG_* env alias so
        # the JSON never has to survive llama-swap's cmd tokenizer. Merge order is explicit in
        # tools/server/server-common.cpp: the CLI/env values seed inputs.chat_template_kwargs
        # and any per-request `chat_template_kwargs` object is then written OVER them, so a
        # client can still override per request. Contrast with `params=` below, which routes
        # through llama-swap's filters.setParams and REWRITES the request — that one forces.
        #
        # Note this is the only channel that reaches the template: llama.cpp reads a TOP-LEVEL
        # OpenAI `reasoning_effort` only to catch the value "none" (-> enable_thinking=false)
        # and explicitly leaves everything else "model-specific and not yet handled". So a
        # client following the model card and sending reasoning_effort at the top level is
        # silently ignored; it has to be nested under chat_template_kwargs.
        opt += f"      -e 'LLAMA_ARG_CHAT_TEMPLATE_KWARGS={json.dumps(template_kwargs, separators=(',', ':'))}'\n"
    e = (
        f'  "{model_id}":\n'
        f"    cmd: |\n"
        f"      docker run --rm --name ${{MODEL_ID}}\n"
        f"      --pull=always\n"
        f"      --entrypoint gguf-serve.sh\n"
        f"      --gpus '\"device={gpus}\"'\n"
        f"      -e GGUF_REPO={repo}\n"
        f"      -e GGUF_FILE={hf_file}\n"
        f"      -e GGUF_CTX={ctx}\n"
        f"      -e GGUF_PARALLEL={par}\n"
        f"{opt}"
        f"      -v /models/hf-cache:/root/.cache/huggingface\n"
        f"      -p ${{PORT}}:8080\n"
        f"      {image}\n"
        f"      --alias ${{MODEL_ID}}\n"
        f"{sampling or ''}"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    checkEndpoint: /health\n"
        f"    ttl: {ttl}\n"
        f"    concurrencyLimit: {par * GGUF_LIMIT_MULT}\n"
    )
    if params:
        e += setparams_filter(**params)
    return e


def member_entry(spec, model_id, card, ttl=TTL):
    b = spec["backend"]
    if b == "fork":
        return fork_entry(model_id, card, ttl=ttl)
    if b == "gguf":
        # Forward the optional GGUF knobs the same way ungrouped_gguf_entry() does. Until
        # qwen3.8-27b joined POOL every pooled GGUF was a plain single-file model on the
        # bonsai image, so these were silently dropped — a pooled member with an mmproj or a
        # non-default image would have been generated WITHOUT them and served text-only (or
        # failed to load) with nothing in the config to show why.
        return gguf_entry(model_id, card, spec["repo"], spec["hf_file"], spec["ctx"],
                          spec.get("par", GGUF_PARALLEL), ttl=ttl,
                          image=spec.get("image", BONSAI),
                          mmproj=spec.get("mmproj"), draft=spec.get("draft"),
                          spec_type=spec.get("spec_type"), draft_max=spec.get("draft_max"),
                          cache_type=spec.get("cache_type"), params=spec.get("params"),
                          template_kwargs=spec.get("template_kwargs"),
                          sampling=spec.get("sampling"))
    return vllm_entry(model_id, spec["repo"], card, spec["mml"], CONCURRENCY,
                      spec.get("eager", False), spec.get("think_off", False),
                      ttl=ttl, extra=spec.get("extra", ()))


def ungrouped_gguf_entry(spec):
    return gguf_entry(spec["tok"], ",".join(spec["cards"]), spec["repo"], spec["hf_file"],
                      spec["ctx"], spec.get("par", 1), ttl=spec.get("ttl", TTL),
                      image=spec.get("image", BONSAI),
                      mmproj=spec.get("mmproj"), draft=spec.get("draft"),
                      spec_type=spec.get("spec_type"), draft_max=spec.get("draft_max"),
                      split_mode=spec.get("split_mode"),
                      tensor_split=spec.get("tensor_split"),
                      cache_type=spec.get("cache_type"),
                      params=spec.get("params"),
                      template_kwargs=spec.get("template_kwargs"),
                      sampling=spec.get("sampling"))


def main():
    # A pair is only worth generating if the two members are worth running SIDE BY SIDE. One
    # predicate covers that: pair everything EXCEPT same family AND same role.
    #
    #   same family + same role       -> DROP. Near-duplicates; nothing is learned by co-loading
    #                                    qwen3.5-4b next to qwen3.5-9b, and for any given
    #                                    request one of them is simply the better choice.
    #   same family + different role  -> KEEP. gemma-26b (anchor) + gemma-e4b (fast) is the
    #                                    complementary case: a heavy/high-KV model sharing the
    #                                    box with a small high-concurrency one.
    #   different family + same role  -> KEEP. This is the head-to-head the whole pairs config
    #                                    exists for: strong-vs-strong (gemma-26b + qwen3.8-27b)
    #                                    and fast-vs-fast (gemma-e4b + qwen3.5-4b) ACROSS
    #                                    families.
    #
    # Replaces the previous all-C(n,2) enumeration, which also emitted every same-family
    # near-duplicate. Ordering stays (later member, earlier member) so APPENDING a POOL member
    # is additive to the numbering; RETIRING one still renumbers, which this change does once.
    def worth_pairing(a, b):
        return not (a["family"] == b["family"] and a["role"] == b["role"])

    combos = sorted(itertools.combinations(range(len(POOL)), 2), key=lambda p: (p[1], p[0]))
    pairs = [(i, j) for (i, j) in combos if worth_pairing(POOL[i], POOL[j])]
    dropped = [(POOL[i]["tok"], POOL[j]["tok"])
               for (i, j) in combos if not worth_pairing(POOL[i], POOL[j])]

    out = []
    out.append("# llama-swap PAIRS config (GENERATED by gen_pairs_config.py — do not hand-edit).")
    out.append("# Each pairNN is a co-load group: two single-card models, one per 3090, serving")
    out.append("# concurrently. Callsign = pairNN.<model> (roles are assigned by the consuming")
    out.append("# system; split on the FIRST '.' to get the pair id). Only one pair (or one solo")
    out.append("# model) is resident at a time; requesting another swaps it in.")
    out.append("# Regenerate: python3 gen_pairs_config.py > config.pairs.yaml")
    out.append("#")
    out.append("# HF AUTH: deliberately NOT passed as -e HF_TOKEN/-e HUGGING_FACE_HUB_TOKEN.")
    out.append("# llama-swap expands ${env.*} at spawn time and echoes the fully expanded")
    out.append("# command back from GET /running, which leaked the token in plaintext to")
    out.append("# anyone who could reach :9292. Instead the token is read from the HF cache")
    out.append("# that every model already mounts: write it to /models/hf-cache/token")
    out.append("# (chmod 600) on the host; huggingface_hub resolves $HF_HOME/token when no")
    out.append("# env var is set. Do not re-add the -e lines.")
    out.append("")
    out.append("healthCheckTimeout: 900")
    out.append("logLevel: info")
    out.append("")
    if dropped:
        out.append("# Pairs deliberately NOT generated (same family AND same role — near-duplicates):")
        for a, b in dropped:
            out.append(f"#   {a} + {b}")
        out.append("")
    out.append("models:")
    out.append("")

    groups = []
    for k, (i, j) in enumerate(pairs, start=1):
        pair = f"pair{k:02d}"
        # The anchor takes 3090 #0 when the roles differ, so the heavy member's card is
        # predictable across pairs; equal-role pairs fall back to POOL order.
        a, b = POOL[i], POOL[j]
        if a["role"] == "fast" and b["role"] == "anchor":
            a, b = b, a
        id_a = f"{pair}.{a['tok']}"      # -> 3090 #0
        id_b = f"{pair}.{b['tok']}"      # -> 3090 #2
        out.append(f"  # ===== {pair}: {a['tok']} ({a['role']}, #0)  +  {b['tok']} ({b['role']}, #2) =====")
        out.append(member_entry(a, id_a, CARD0))
        out.append(member_entry(b, id_b, CARD2))
        groups.append((pair, id_a, id_b))

    out.append("  # ===== Standalone entries (ungrouped): load one pooled model solo =====")
    out.append("  # Callsign = the base token (no pairNN prefix). Requesting one loads it alone")
    out.append("  # (exclusive swap unloads whatever else is resident). TP=1 on 3090 #0.")
    for spec in POOL:
        # `standalone_ttl` applies to the bare `<tok>` entry ONLY, not to its pairNN members.
        # muse-glimmer needs it: as the on-call standby its standalone entry must never
        # idle-unload (ttl 0), while its pair members should age out on the normal TTL.
        out.append(member_entry(spec, spec["tok"], CARD0,
                                ttl=spec.get("standalone_ttl", TTL)))

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
