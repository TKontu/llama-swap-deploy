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

CARD0 = "GPU-a8c640ca-4d44-440b-5caf-28eca88ea7c1"   # 3090 #0
CARD2 = "GPU-094f1ca3-2155-7b04-b5aa-4abae3b5ffeb"   # 3090 #2
IMAGE = "vllm/vllm-openai:v0.25.1"
BONSAI = "ghcr.io/tkontu/bonsai-llama:latest"

# Single-card pool. Each entry is a dict keyed by "backend":
#   vllm: repo, mml, seqs, eager, think            (vLLM container, TP=1 @ util 0.90)
#   fork: (none)                                   (Ternary via the PrismML bonsai image entrypoint)
#   gguf: repo, hf_file, ctx                       (standard GGUF via the bonsai image's llama-server)
POOL = [
    dict(tok="gemma-26b",   backend="vllm", repo="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",      mml=16800,  seqs=1),
    dict(tok="phi-4",       backend="vllm", repo="stelterlab/phi-4-AWQ",                          mml=16384,  seqs=4),
    dict(tok="gemma-12b",   backend="vllm", repo="cyankiwi/gemma-4-12B-it-qat-AWQ-INT4",          mml=32000,  seqs=4),
    dict(tok="gemma-e4b",   backend="vllm", repo="cyankiwi/gemma-4-E4B-it-qat-AWQ-INT4",          mml=128000, seqs=8),
    dict(tok="qwen3.5-9b",  backend="vllm", repo="cyankiwi/Qwen3.5-9B-AWQ-4bit",                  mml=16384,  seqs=2),
    dict(tok="qwen3.5-4b",  backend="vllm", repo="cyankiwi/Qwen3.5-4B-AWQ-4bit",                  mml=16384,  seqs=32, eager=True, think=True),
    dict(tok="mellum2-12b", backend="vllm", repo="cyankiwi/Mellum2-12B-A2.5B-Instruct-AWQ-INT4",  mml=128000, seqs=12),
    dict(tok="ternary",     backend="fork"),
    dict(tok="qwythos-v2",  backend="gguf", repo="empero-ai/Qwythos-9B-v2-GGUF", hf_file="Qwythos-9B-v2-Q4_K_M.gguf", ctx=32768),
    # Xet-backed repo (~11.3 GB Q6_K). llama-server -hf downloads via HTTP; if Xet blocks
    # that, we pre-download with the `hf` CLI (+hf_xet) instead. See README.
    dict(tok="fablevibes",  backend="gguf", repo="tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF", hf_file="Qwen3.6-14B-A3B-FableVibes-Q6_K.gguf", ctx=32768),
    # MTP variant (self-speculative) — uncomment to add as its own pool member (needs a load test):
    # dict(tok="qwythos-v2-mtp", backend="gguf", repo="empero-ai/Qwythos-9B-v2-GGUF", hf_file="Qwythos-9B-v2-MTP-Q4_K_M.gguf", ctx=32768),
]

# Solo big models (need both 3090s → TP=2 → no partner). (id, repo, mml, seqs, util, think_off)
SOLO = [
    ("Qwen3.6-35B-A3B-AWQ-4bit",   "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",         131072, 1, 0.90, True),
    ("Qwen3.6-27B-AWQ-INT4",       "cyankiwi/Qwen3.6-27B-AWQ-INT4",             262144, 1, 0.92, False),
    ("gemma4-26B-A4B-it-INT4-max", "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",  131072, 1, 0.90, False),
    ("Qwythos-9B-Claude-Mythos-5-1M", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M", 256000, 1, 0.90, False),
]

THINK_FILTER = (
    "    filters:\n"
    "      setParams:\n"
    "        chat_template_kwargs:\n"
    "          enable_thinking: false\n"
)


def vllm_entry(model_id, repo, gpus, mml, seqs, eager, think_off, tp=1, util=0.90, ttl=1800):
    eager_line = "      --enforce-eager\n" if eager else ""
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
        f"      --port 8000\n"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    ttl: {ttl}\n"
    )
    if think_off:
        e += THINK_FILTER
    return e


def fork_entry(model_id, gpus, ttl=1800):
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
    )


def gguf_entry(model_id, gpus, repo, hf_file, ctx, ttl=1800):
    # Standard GGUF via the bonsai image's gguf-serve.sh entrypoint: it downloads the
    # file with the `hf` CLI (HTTPS + gated + Xet) into the mounted cache, then serves
    # the local file with llama-server (this build's llama-server has no HTTPS itself).
    return (
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
        f"      -v /models/hf-cache:/root/.cache/huggingface\n"
        f"      -p ${{PORT}}:8080\n"
        f"      {BONSAI}\n"
        f"      --alias ${{MODEL_ID}}\n"
        f"    cmdStop: docker stop ${{MODEL_ID}}\n"
        f"    proxy: http://127.0.0.1:${{PORT}}\n"
        f"    checkEndpoint: /health\n"
        f"    ttl: {ttl}\n"
    )


def member_entry(spec, model_id, card):
    b = spec["backend"]
    if b == "fork":
        return fork_entry(model_id, card)
    if b == "gguf":
        return gguf_entry(model_id, card, spec["repo"], spec["hf_file"], spec["ctx"])
    return vllm_entry(model_id, spec["repo"], card, spec["mml"], spec["seqs"],
                      spec.get("eager", False), spec.get("think", False))


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

    out.append("  # ===== Solo big models (TP=2, own both 3090s — no partner possible) =====")
    for (mid, repo, mml, seqs, util, think_off) in SOLO:
        out.append(vllm_entry(mid, repo, f"{CARD0},{CARD2}", mml, seqs, False, think_off, tp=2, util=util, ttl=3600))

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
