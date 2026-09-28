#!/usr/bin/env python3
"""Deploy gate: push-to-deploy that never cuts work in progress.

Runs as the `deploy-gate` service (docker-compose.yml). It polls GHCR for a new digest of the
llama-swap image. When one appears it:
  1. sets the edge's drain flag (/state/drain). The edge then refuses new work with 503 +
     Retry-After; requests already in flight continue;
  2. waits until nothing is in flight for STABLE_S seconds. That means: no LLM request in
     llama-swap's in-flight list (/api/events), no model starting, and every running ComfyUI
     instance idle by its own hold status. The in-flight list does not see ComfyUI holds;
  3. calls the Portainer webhook, which re-pulls and redeploys the stack (this gate included).
If work is still in flight after DRAIN_MAX_S it clears the flag, logs what is still running and
tries again after RETRY_AFTER_ABORT_S. It never forces a deploy.

On startup it clears any drain flag: a fresh stack is open, and a drain that died with the
previous gate did not deploy.

Stdlib only (the llama-swap image has python3-minimal). Env:
  PORTAINER_WEBHOOK_URL   GitOps webhook of this stack (required to deploy; a secret)
  IMAGE                   image ref to watch, e.g. ghcr.io/tkontu/llama-swap-deploy:latest
  CONTAINER               container running IMAGE (default llama-swap)
  LLAMASWAP_URL           llama-swap itself, not the edge (default http://127.0.0.1:9293)
  DRAIN_FLAG              default /state/drain
  POLL_S STABLE_S DRAIN_MAX_S RETRY_AFTER_ABORT_S DEPLOY_WAIT_S
  LATEST_DIGEST_CMD / RUNNING_DIGEST_CMD   shell commands replacing the GHCR / docker
                          lookups (tests/sim only)
"""
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

ENV = os.environ
WEBHOOK = ENV.get("PORTAINER_WEBHOOK_URL", "").strip()
IMAGE = ENV.get("IMAGE", "ghcr.io/tkontu/llama-swap-deploy:latest")
CONTAINER = ENV.get("CONTAINER", "llama-swap")
LLAMASWAP = ENV.get("LLAMASWAP_URL", "http://127.0.0.1:9293").rstrip("/")
FLAG = ENV.get("DRAIN_FLAG", "/state/drain")
POLL_S = float(ENV.get("POLL_S", 120))
STABLE_S = float(ENV.get("STABLE_S", 15))
DRAIN_MAX_S = float(ENV.get("DRAIN_MAX_S", 21600))
RETRY_AFTER_ABORT_S = float(ENV.get("RETRY_AFTER_ABORT_S", 3600))
DEPLOY_WAIT_S = float(ENV.get("DEPLOY_WAIT_S", 900))

MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} deploy-gate: {msg}", flush=True)


def http_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


# --- digests ------------------------------------------------------------------------------------

def _shell(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout.strip()


def split_image(ref):
    """ghcr.io/owner/name:tag -> (registry, repo, tag)."""
    registry, _, rest = ref.partition("/")
    repo, _, tag = rest.partition(":")
    return registry, repo, tag or "latest"


def latest_digest():
    if ENV.get("LATEST_DIGEST_CMD"):
        return _shell(ENV["LATEST_DIGEST_CMD"])
    registry, repo, tag = split_image(IMAGE)
    token = http_json(f"https://{registry}/token?scope=repository:{repo}:pull")["token"]
    req = urllib.request.Request(f"https://{registry}/v2/{repo}/manifests/{tag}", method="HEAD",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Accept": MANIFEST_TYPES})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.headers["Docker-Content-Digest"]


def running_digest():
    if ENV.get("RUNNING_DIGEST_CMD"):
        return _shell(ENV["RUNNING_DIGEST_CMD"])
    image_id = _shell(f"docker inspect --format '{{{{.Image}}}}' {CONTAINER}")
    digests = json.loads(_shell(f"docker image inspect --format '{{{{json .RepoDigests}}}}' {image_id}"))
    _, repo, _ = split_image(IMAGE)
    for d in digests or []:
        name, _, digest = d.partition("@")
        if name.endswith(repo):
            return digest
    raise RuntimeError(f"no RepoDigest for {IMAGE} on container {CONTAINER}: {digests}")


# --- drain flag ---------------------------------------------------------------------------------

def set_flag():
    os.makedirs(os.path.dirname(FLAG), exist_ok=True)
    with open(FLAG, "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S\n"))


def clear_flag():
    try:
        os.remove(FLAG)
        return True
    except FileNotFoundError:
        return False


# --- in-flight tracking ---------------------------------------------------------------------------

class InflightWatcher(threading.Thread):
    """Mirrors llama-swap's in-flight list from one long-lived /api/events stream."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.requests = {}
        self.known = False          # a snapshot has arrived on the current connection

    def run(self):
        while True:
            try:
                with urllib.request.urlopen(f"{LLAMASWAP}/api/events", timeout=None) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").rstrip("\n")
                        if line.startswith("data:"):
                            self._message(json.loads(line[5:]))
            except Exception as exc:  # llama-swap restarting, connection dropped, ...
                log(f"/api/events stream lost ({exc!r}); reconnecting")
            with self.lock:
                self.known = False
                self.requests = {}
            time.sleep(2)

    def _message(self, envelope):
        if envelope.get("type") != "inflight":
            return
        update = json.loads(envelope["data"])
        op = update.get("operation")
        with self.lock:
            if op == "snapshot":
                self.requests = {r["id"]: r for r in update.get("requests") or []}
                self.known = True
            elif op == "upsert" and update.get("request"):
                self.requests[update["request"]["id"]] = update["request"]
            elif op == "remove":
                self.requests.pop(update.get("id"), None)

    def busy(self):
        with self.lock:
            if not self.known:
                return ["llama-swap in-flight list not known yet"]
            return [f"{r.get('model')} {r.get('method')} {r.get('req_path')} "
                    f"({r.get('elapsed_ms', 0) // 1000}s)" for r in self.requests.values()]


def busy_reasons(watcher):
    reasons = list(watcher.busy())
    try:
        running = http_json(f"{LLAMASWAP}/running")["running"]
    except Exception as exc:
        return reasons + [f"/running unavailable: {exc!r}"]
    for m in running:
        model, state = m.get("model", ""), m.get("state", "")
        if state not in ("ready", "stopped"):
            reasons.append(f"{model} is {state}")
            continue
        if model.endswith(".comfyui") and state == "ready":
            # Only ready instances: a GET through /upstream would start a stopped one.
            try:
                st = http_json(f"{LLAMASWAP}/upstream/{model}/comfyui-hold/status")
            except Exception as exc:
                reasons.append(f"{model} hold status unavailable: {exc!r}")
                continue
            if st.get("open") or st.get("awaiting_ack") or st.get("tasks_remaining"):
                reasons.append(f"{model} busy (tasks {st.get('tasks_remaining')}, "
                               f"awaiting ack {st.get('awaiting_ack')}, hold open {st.get('open')})")
    return reasons


# --- the gate -------------------------------------------------------------------------------------

def drain_and_deploy(watcher, latest):
    set_flag()
    log(f"new image {latest}: draining (edge refuses new work)")
    start, idle_since, last_report = time.monotonic(), None, 0.0
    while True:
        now = time.monotonic()
        reasons = busy_reasons(watcher)
        if not reasons:
            idle_since = idle_since or now
            if now - idle_since >= STABLE_S:
                break
        else:
            idle_since = None
            if now - last_report >= 60:
                log(f"waiting for {len(reasons)} item(s): " + "; ".join(reasons))
                last_report = now
        if now - start >= DRAIN_MAX_S:
            clear_flag()
            log(f"deploy postponed: work still in flight after {DRAIN_MAX_S:.0f}s ("
                + "; ".join(reasons) + f"); retrying in {RETRY_AFTER_ABORT_S:.0f}s")
            time.sleep(RETRY_AFTER_ABORT_S)
            return
        time.sleep(2)

    if not WEBHOOK:
        clear_flag()
        log("PORTAINER_WEBHOOK_URL is not set: cannot deploy; drain cancelled")
        time.sleep(RETRY_AFTER_ABORT_S)
        return
    log(f"idle for {STABLE_S:.0f}s: calling the Portainer webhook")
    try:
        with urllib.request.urlopen(urllib.request.Request(WEBHOOK, data=b"", method="POST"),
                                    timeout=60) as r:
            log(f"webhook answered HTTP {r.status}")
    except Exception as exc:
        clear_flag()
        log(f"webhook failed ({exc!r}); drain cancelled, retrying in {RETRY_AFTER_ABORT_S:.0f}s")
        time.sleep(RETRY_AFTER_ABORT_S)
        return

    # The redeploy normally replaces this container (its image changed too), and the new gate
    # clears the flag on startup. If we are still here, clear it ourselves.
    deadline = time.monotonic() + DEPLOY_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            if running_digest() == latest:
                clear_flag()
                log("llama-swap now runs the new image; drain flag cleared")
                return
        except Exception:
            pass  # llama-swap container being recreated
    clear_flag()
    log(f"no redeploy observed within {DEPLOY_WAIT_S:.0f}s of the webhook; drain flag cleared")


def wait_for_llamaswap():
    while True:
        try:
            with urllib.request.urlopen(f"{LLAMASWAP}/health", timeout=5):
                return
        except Exception:
            time.sleep(5)


def main():
    log(f"watching {IMAGE} (container {CONTAINER}); llama-swap at {LLAMASWAP}")
    wait_for_llamaswap()
    if clear_flag():
        log("cleared a drain flag left by a previous gate")
    if not WEBHOOK:
        log("WARNING: PORTAINER_WEBHOOK_URL is not set; new images will be detected but not deployed")
    watcher = InflightWatcher()
    watcher.start()
    while True:
        try:
            latest, running = latest_digest(), running_digest()
        except Exception as exc:
            log(f"digest check failed: {exc!r}")
            time.sleep(POLL_S)
            continue
        if latest != running:
            drain_and_deploy(watcher, latest)
        else:
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
