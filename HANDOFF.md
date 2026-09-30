# Handoff: llama-swap stack with ComfyUI, edge and deploy gate

Status as of **2026-09-28**. `main` is at `46e04d3` (PR #30 merged) and **deployed** on the
inference host (`192.168.0.94`), via a manual Portainer redeploy.

## What's running

```
LAN ─► edge (Caddy, :9292) ─► llama-swap 127.0.0.1:9293 (v256, pinned) ─► models 127.0.0.1:<port>
         │ 403 for /upstream/*comfyui*, /comfyui from non-loopback
         │ 503 + Retry-After for new work while a deploy drains
host ─► deploy-gate (watches GHCR, drains, calls Portainer webhook: webhook NOT SET yet)
        oncall-wakeup (via the edge)
```

| Container | Image | Role |
|---|---|---|
| `llama-swap` | `ghcr.io/tkontu/llama-swap-deploy:latest` | Scheduler for all GPUs; config baked in (`gen_config.py` → `config.yaml`) |
| `llama-swap-edge` | `ghcr.io/tkontu/llama-swap-edge:latest` | LAN entry point on `:9292` (`docker/edge/Caddyfile`) |
| `deploy-gate` | same as llama-swap | Push-to-deploy that waits for idle (`scripts/deploy-gate.py`) |
| `oncall-wakeup` | same as llama-swap | Keeps `c0.muse-glimmer` warm when the box is idle |

GPU layout:

| GPU | Card | Used by |
|---|---|---|
| 0 | 3090 `c2` | LLMs; `c2.comfyui` only for models over 12 GB |
| 1 | 3090 `c0` | LLMs (on-call `c0.muse-glimmer`) |
| 2 | A2000 | MinerU (outside llama-swap) |
| 3 | A2000 | TEI + Infinity (outside llama-swap) |
| 4 | A2000 `a4` | `a4.comfyui` (the default ComfyUI instance) |

## Policies (decided; don't change without the owner)

- **c2 belongs to the LLMs.** ComfyUI runs on the A2000 when the model fits 12 GB, on c2 only
  when it needs more. Both 3090s later, if a model needs them.
- **ComfyUI is always cold-loaded** and unloads 5 min after its last job (`ttl: 300`).
- **Work in progress must never be interrupted:** not by swaps, idle timeouts or deploys.
  Waiting (an LLM behind a render, a deploy behind both) is the accepted price.
- **MinerU, TEI and Infinity stay outside llama-swap** (llama-swap's 30 s restart drain would cut
  them). A possible consolidation onto one A2000 is open (see "Later").
- **Clients reach ComfyUI only through media-gateway** (planned, not built), never `/upstream/`.
- **llama-swap's unload/cancel APIs stay open to the LAN** (owner's choice). Operating rule:
  don't use them while anything runs.

## How "never interrupted" is achieved

| Threat | Protection | Where |
|---|---|---|
| A swap evicts a busy model | llama-swap waits for the evicted model's in-flight requests | llama-swap v256 (pinned; `Dockerfile`) |
| Idle TTL unloads a busy model | The TTL skips while requests are in flight | same |
| ComfyUI renders are invisible (`/prompt` returns at once) | **The hold**: ComfyUI keeps a request to itself open through llama-swap while it has work; `X-Hold-Ack` keeps it until the client has collected results; a watchdog releases a stuck job's hold after 30 min without progress / 4 h total | `docker/comfyui_hold/` |
| A deploy cuts in-flight work (llama-swap drains only 30 s) | **deploy-gate** drains via the edge and redeploys only when nothing is in flight; never forces | `scripts/deploy-gate.py`, `docker/edge/Caddyfile` |
| Unplanned stop | `stop_grace_period: 60s` gives the full 30 s drain | `docker-compose.yml` |

Not covered: a manual redeploy or host reboot, `POST /api/models/unload`, a ComfyUI crash.

All of the above except the host itself is tested by **`bash tests/sim/run.sh`**: 14 cases
against the pinned llama-swap and Caddy binaries, with fake model servers running the real hold
code. CI runs it as `sim-tests`.

## Verified on the host (2026-09-28)

- [x] All four CI workflows green on `46e04d3` (`comfyui-image` incl. import test, `edge-image`,
  `build-and-push`, `sim-tests`).
- [x] Stack redeployed; `deploy-gate` runs (so the new image was pulled).
- [x] Edge `:9292` healthy; llama-swap reports v256 (6701d0d).
- [x] From the LAN: `/upstream/c2.comfyui/…` and `/comfyui/` → **403**; `:9293` unreachable.
- [ ] Model backends unreachable on their own ports from the LAN. Check next time a model is
  loaded: take the port from `/running` → `proxy`, then `curl 192.168.0.94:<port>/health` must fail.

## Known gap: the gate doesn't work on Portainer CE yet

Portainer here is **Community Edition**, where **"Re-pull image" and "Force redeployment" are
Business features**. The webhook redeploys from git but keeps the old `:latest` images, so the
gate as merged would drain, call the webhook, see nothing change, and drain again every
~15 min.

**Mitigation in place:** `PORTAINER_WEBHOOK_URL` is **left empty**, so the gate only logs new
images and never drains. Deploys are manual until the fix below lands.

## Next steps (in order)

1. **Certificate check** on the inference host (decides whether the gate needs a CA option):
   ```
   curl -sS -o /dev/null -w '%{http_code}\n' https://portainer.lan.<rest>/api/system/status
   ```
2. **Gate fix for CE** (branch `fix/gate-ce-pull`):
   - Before calling the webhook, `docker pull` both images (`llama-swap-deploy`,
     `llama-swap-edge`).
   - After a deploy that didn't take effect, wait `RETRY_AFTER_ABORT_S` before draining again
     (no drain loop).
   - Optional `PORTAINER_CA_FILE`, if step 1 shows a certificate error.
   - New sim case "deploy that doesn't take effect → backs off"; all 14 existing cases must
     still pass. PR, CI, merge.
3. **Last manual deploy**, when idle: on the host run
   `docker pull ghcr.io/tkontu/llama-swap-deploy:latest`, then update the stack in Portainer.
4. **Enable the gate:** Portainer → stack → GitOps updates: **Webhook**, repository reference
   `refs/heads/main`. Copy the webhook URL into the stack variable `PORTAINER_WEBHOOK_URL`
   (it's a secret; never commit it). Update the stack.
5. **Test the gate once:** push a trivial change (e.g. a comment in `gen_config.py`, regenerate
   `config.yaml`) while a long LLM request runs. `docker logs -f deploy-gate` should show
   "draining" → "waiting for 1 item(s)" → "calling the Portainer webhook" → new image
   running.
6. **ComfyUI bring-up** (TODO Step 6):
   - Download the weights to `/fast/comfyui/models` (commands in README → "ComfyUI").
   - Pick the emoji LoRA on Civitai (check its licence, note the trigger word).
   - Open ComfyUI via `ssh -L 9292:127.0.0.1:9292 inference` →
     `http://localhost:9292/upstream/a4.comfyui/`, and check the UI works under that subpath.
   - Measure peak VRAM and seconds per image for SDXL + LoRA + IP-Adapter + BiRefNet on the
     A2000, and record them in the README.
   - Verify the hold on real hardware: a render longer than 5 min is not unloaded; a c2 LLM
     request waits for a `c2.comfyui` render.
7. **media-gateway** (`/work/media-gateway-deploy`, docs only; not a git repo yet): build it per its
   `todo.md`. Contract with this stack:
   - Call `127.0.0.1:9292/upstream/<instance>/` with `X-Hold-Ack: 1` on `/prompt`, then POST
     `/comfyui-hold/ack` after collecting `/history` and the outputs.
   - Run one job at a time per instance, and pause ≥ 1 s between an ack and the next job.
   - Treat the edge's `503` + `Retry-After` as "wait", not a failure.

## Later / open

- **Consolidate MinerU + TEI + Infinity onto one A2000**, which would free a card for llama-swap:
  - Cap MinerU's vLLM memory reservation (it defaults to `hybrid-auto-engine` and holds ~10 GB).
  - Drop Infinity's duplicate bge-m3 if nothing uses it (Iknos embeds via TEI).
  - Measure under a real Iknos ingest; the combined load is ~15 GB today, on a 12 GB card.
- Propose a queue-aware "busy" check for ComfyUI to llama-swap upstream; if it lands, drop the
  hold.
- A ComfyUI entry across both 3090s (needs a multi-GPU node pack), only when a model needs more
  than 24 GB.
- Hardening (`ARCHITECTURE.md` → Security): firewall `:9292` to the actual clients; optionally
  expose only `/v1/*` at the edge.
- **Rotate the HF token** that was pasted into the local `.env` (never committed, but the TODO
  item is still open).
- media-gateway open decisions: STT/TTS placement (D6), Hermes tool binding (D7), and its docs
  say the host is `192.168.0.247`, while the GPU host answers on `.94`.

## Where things are

| Topic | File |
|---|---|
| Model definitions (edit here, then `python3 gen_config.py > config.yaml`) | `gen_config.py` |
| Operations, deploy, edge, gate, ComfyUI setup | `README.md` |
| Design, topology, security | `ARCHITECTURE.md` |
| Detailed task list (ComfyUI section = Steps 2–6) | `TODO.md` |
| ComfyUI image / hold | `Dockerfile.comfyui`, `docker/comfyui-serve.sh`, `docker/comfyui_hold/` |
| Edge | `Dockerfile.edge`, `docker/edge/Caddyfile` |
| Deploy gate | `scripts/deploy-gate.py` |
| Tests | `tests/sim/` (`bash tests/sim/run.sh`; binaries are cached in `~/.cache/llama-swap-sim`) |
| Pins (bump together with a green sim run) | llama-swap: `Dockerfile` `LLAMA_SWAP_VERSION`/`SHA256`; Caddy: `Dockerfile.edge` + `tests/sim/run.sh`; ComfyUI + nodes: `Dockerfile.comfyui` |

Useful commands:
```
docker logs -f deploy-gate                                          # what the gate is waiting for
curl -s http://127.0.0.1:9293/running                               # loaded models (on the host)
curl -s http://127.0.0.1:9292/upstream/a4.comfyui/comfyui-hold/status   # hold state (running instance only)
```
