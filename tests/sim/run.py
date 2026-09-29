"""Scheduler, edge and deploy-gate tests against real llama-swap and Caddy binaries (see run.sh).

Each case starts a fresh llama-swap (config.yaml: fake upstreams, the production slot layout) on
127.0.0.1:19293 behind the real edge (docker/edge/Caddyfile) on :19292, drives it over HTTP, and
checks the upstream event log. The guarantees under test: work in progress is never interrupted
(not by swaps, TTLs or deploys), waiting LLM requests still get the card between jobs, raw
ComfyUI stays off the LAN, and a deploy happens only once nothing is in flight.

Usage: python3 run.py <llama-swap binary> <caddy binary> [case ...]
"""
import http.client as httpclient
import http.server as httpserver
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SIM = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SIM))
EDGE_PORT, LS_PORT, HOOK_PORT = 19292, 19293, 19400
BASE = f"http://127.0.0.1:{EDGE_PORT}"          # clients go through the edge
DIRECT = f"http://127.0.0.1:{LS_PORT}"          # llama-swap itself
EVENTS = os.path.join(SIM, "events.log")
STATE = os.path.join(SIM, "state")              # the edge's /state (drain flag)
FLAG = os.path.join(STATE, "drain")


# --- plumbing ---------------------------------------------------------------------------------

def http(method, path, body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_from(src_ip, method, path, body=None):
    """Like http() but from another local source address, to play a LAN client for the edge
    (127.0.0.2 is not in the edge's 127.0.0.1/32 loopback allowance)."""
    conn = httpclient.HTTPConnection("127.0.0.1", EDGE_PORT, timeout=30,
                                      source_address=(src_ip, 0))
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, r.read()


class Webhook(threading.Thread):
    """Fake Portainer webhook: records call times; optionally 'deploys' by rewriting the
    running-digest file, as a real redeploy would change the running image."""

    def __init__(self, on_call=None):
        super().__init__(daemon=True)
        self.calls, hook = [], self

        class H(httpserver.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                hook.calls.append(time.time())
                if on_call:
                    on_call()
                self.send_response(204)
                self.end_headers()

        self.server = httpserver.ThreadingHTTPServer(("127.0.0.1", HOOK_PORT), H)

    def run(self):
        self.server.serve_forever()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Sim:
    def __init__(self, binary, caddy):
        self.binary, self.caddy = binary, caddy
        self.proc = self.edge = self.gate = None
        self.results = {}

    def start(self):
        if os.path.exists(EVENTS):
            os.remove(EVENTS)
        subprocess.run(["rm", "-rf", STATE], check=True)
        os.makedirs(STATE)
        env = dict(os.environ, SIM_EVENTS=EVENTS)
        # Own session: llama-swap signals its whole process group on shutdown.
        self.proc = subprocess.Popen(
            [self.binary, "--config", "config.yaml", "--listen", f"127.0.0.1:{LS_PORT}"],
            cwd=SIM, env=env, start_new_session=True,
            stdout=open(os.path.join(SIM, "llama-swap.log"), "w"), stderr=subprocess.STDOUT)
        edge_env = dict(os.environ, EDGE_LISTEN=f":{EDGE_PORT}",
                        EDGE_UPSTREAM=f"127.0.0.1:{LS_PORT}", EDGE_STATE=STATE)
        self.edge = subprocess.Popen(
            [self.caddy, "run", "--adapter", "caddyfile",
             "--config", os.path.join(REPO, "docker", "edge", "Caddyfile")],
            cwd=SIM, env=edge_env, start_new_session=True,
            stdout=open(os.path.join(SIM, "edge.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                if http("GET", "/health", timeout=1)[0] == 200:
                    self.t0 = time.time()
                    return
            except OSError:
                pass
            time.sleep(0.1)
        raise RuntimeError("llama-swap or the edge did not start")

    def start_gate(self, latest, running, **env):
        """Run the real scripts/deploy-gate.py with digests from files and a fake webhook."""
        self.latest_file = os.path.join(STATE, "latest")
        self.running_file = os.path.join(STATE, "running")
        self.pull_file = os.path.join(STATE, "pulls")
        for path, value in ((self.latest_file, latest), (self.running_file, running)):
            with open(path, "w") as f:
                f.write(value)
        gate_env = dict(os.environ, LLAMASWAP_URL=DIRECT, DRAIN_FLAG=FLAG,
                        LATEST_DIGEST_CMD=f"cat {self.latest_file}",
                        RUNNING_DIGEST_CMD=f"cat {self.running_file}",
                        PULL_IMAGES="img/llama-swap:latest,img/edge:latest",
                        PULL_CMD=f"date +%s.%N >> {self.pull_file}; echo pulled",
                        PORTAINER_WEBHOOK_URL=f"http://127.0.0.1:{HOOK_PORT}/hook",
                        POLL_S="1", STABLE_S="2", DRAIN_MAX_S="600",
                        RETRY_AFTER_ABORT_S="600", DEPLOY_WAIT_S="5")
        gate_env.update({k: str(v) for k, v in env.items()})
        self.gate_log = os.path.join(SIM, "gate.log")
        self.gate = subprocess.Popen(
            [sys.executable, os.path.join(REPO, "scripts", "deploy-gate.py")],
            env=gate_env, stdout=open(self.gate_log, "w"), stderr=subprocess.STDOUT)

    def gate_output(self):
        with open(self.gate_log) as f:
            return f.read()

    def pulls(self):
        """Timestamps of the gate's image pulls (PULL_CMD appends one line each)."""
        try:
            with open(self.pull_file) as f:
                return [float(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def stop(self):
        for p in (self.gate, self.edge):
            if p and p.poll() is None:
                p.terminate()
                p.wait(10)
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(40)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        subprocess.run(["pkill", "-f", "[f]ake_(llm|comfy)[.]py"], check=False)
        time.sleep(0.3)

    # requests, each recorded as results[tag] = (start, end, status)
    def llm(self, model, tag, secs):
        def go():
            s = time.time()
            st, _ = http("POST", "/v1/chat/completions",
                         {"model": model, "tag": tag, "sleep": secs, "messages": []})
            self.results[tag] = (s, time.time(), st)
        t = threading.Thread(target=go)
        t.start()
        return t

    def prompt(self, inst, tag, secs, stuck=False, ack=False):
        q = f"/upstream/{inst}/prompt?tag={tag}&secs={secs}" + ("&stuck=1" if stuck else "")
        st, body = http("POST", q, headers={"X-Hold-Ack": "1"} if ack else None)
        self.results[tag] = (time.time(), None, st)
        return st, body

    def wait_job(self, inst, tag, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            st, body = http("GET", f"/upstream/{inst}/job/{tag}")
            if st == 200 and json.loads(body)["state"] == "done":
                return
            time.sleep(0.5)
        raise AssertionError(f"job {tag} on {inst} did not finish")

    def ack(self, inst):
        return http("POST", f"/upstream/{inst}/comfyui-hold/ack")[0]

    # event log
    def events(self):
        if not os.path.exists(EVENTS):
            return []
        out = []
        with open(EVENTS) as f:
            for line in f:
                # The fakes append while we read, so the last line can be half-written.
                if line.endswith("\n"):
                    out.append(json.loads(line))
        return out

    def at(self, model, event, tag=""):
        """Time of the first matching event, or None."""
        for e in self.events():
            if e["model"] == model and e["event"] == event and (not tag or e["tag"] == tag):
                return e["t"]
        return None

    def dump(self):
        for e in self.events():
            print(f"      {e['t'] - self.t0:6.2f} {e['model']:<11} {e['event']:<9} {e['tag']}")
        for tag, (s, e, st) in sorted(self.results.items(), key=lambda kv: kv[1][0]):
            end = f"{e - self.t0:6.2f}" if e else "     -"
            print(f"      req {tag:<8} {s - self.t0:6.2f} -> {end} http={st}")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def before(a, b):
    return a is not None and (b is None or a <= b)


# --- cases ------------------------------------------------------------------------------------

def case_render_outlives_ttl(sim):
    """A render longer than the TTL is not unloaded; it unloads TTL after the queue empties."""
    check(sim.prompt("a4.comfyui", "J", 6)[0] == 200, "/prompt failed")
    time.sleep(11)
    end, stop = sim.at("a4.comfyui", "job_end", "J"), sim.at("a4.comfyui", "stopped")
    check(end is not None, "job never finished")
    check(before(end, stop), "a4.comfyui was stopped before the job finished")
    check(stop is not None and stop - end >= 2.5, "not unloaded ~TTL after the job")


def case_no_gap_at_submit(sim):
    """An LLM request 50 ms after /prompt returns still waits for the render."""
    check(sim.prompt("c2.comfyui", "J", 4)[0] == 200, "/prompt failed")
    time.sleep(0.05)
    sim.llm("c2.llm", "L", 1).join(30)
    end = sim.at("c2.comfyui", "job_end", "J")
    check(before(end, sim.at("c2.comfyui", "stopped")), "c2.comfyui stopped mid-render")
    check(before(end, sim.at("c2.llm", "begin", "L")), "LLM started before the render ended")
    check(sim.results["L"][2] == 200, "LLM request failed")


HANDOFF_S = 1.0  # media-gateway's pause between ack and its next job on the same instance


def case_gateway_sequence(sim):
    """Gateway pattern (ack after collecting, then a hand-off pause): a waiting LLM gets c2
    between two jobs. The pause is required: llama-swap serves a request for the loaded model
    ahead of a pending swap, so a next /prompt sent right after the ack would win the race."""
    def gateway():
        for tag in ("J1", "J2"):
            sim.prompt("c2.comfyui", tag, 3, ack=True)
            sim.wait_job("c2.comfyui", tag)
            http("GET", f"/upstream/c2.comfyui/job/{tag}")  # "fetch results" while held
            sim.ack("c2.comfyui")
            time.sleep(HANDOFF_S)
    g = threading.Thread(target=gateway)
    g.start()
    time.sleep(1)
    sim.llm("c2.llm", "L", 1).join(60)
    g.join(60)
    j1, j2 = sim.at("c2.comfyui", "job_end", "J1"), sim.at("c2.comfyui", "job_begin", "J2")
    llm = sim.at("c2.llm", "begin", "L")
    check(before(j1, llm) and before(llm, j2), "LLM did not run between J1 and J2")
    check(sim.at("c2.comfyui", "job_end", "J2") is not None, "J2 did not finish")


def case_results_kept_until_ack(sim):
    """With X-Hold-Ack, a finished job keeps ComfyUI loaded until acked (results not lost)."""
    sim.prompt("c2.comfyui", "J", 2, ack=True)
    time.sleep(0.5)
    sim.llm("c2.llm", "L", 1)
    sim.wait_job("c2.comfyui", "J")
    time.sleep(1.5)  # job done, not acked yet: the LLM must still be waiting
    check(sim.at("c2.llm", "begin", "L") is None, "LLM evicted ComfyUI before the ack")
    st, body = http("GET", "/upstream/c2.comfyui/job/J")
    check(st == 200 and json.loads(body)["state"] == "done", "results not reachable before ack")
    check(sim.ack("c2.comfyui") == 200, "ack failed")
    time.sleep(3)
    check(sim.at("c2.llm", "begin", "L") is not None, "LLM did not run after the ack")


def case_ack_timeout(sim):
    """A client that never acks can't hold the card: released after ack_timeout (3 s)."""
    sim.prompt("c2.comfyui", "J", 1, ack=True)
    time.sleep(0.3)
    sim.llm("c2.llm", "L", 1).join(30)
    end, llm = sim.at("c2.comfyui", "job_end", "J"), sim.at("c2.llm", "begin", "L")
    check(llm is not None and llm - end >= 2.5, "released before the ack timeout")


def case_wholebox_waits(sim):
    """A whole-box request waits for the c2 render; the a4 render continues untouched."""
    sim.prompt("c2.comfyui", "C", 4)
    sim.prompt("a4.comfyui", "A", 7)
    time.sleep(1)
    sim.llm("wholebox", "W", 1).join(30)
    time.sleep(4)
    check(before(sim.at("c2.comfyui", "job_end", "C"), sim.at("wholebox", "begin", "W")),
          "whole-box started before the c2 render ended")
    check(before(sim.at("a4.comfyui", "job_end", "A"), sim.at("a4.comfyui", "stopped")),
          "a4 render was interrupted")


def case_watchdog(sim):
    """A stuck job (no progress) releases the hold after stall_s (3 s); the LLM then runs."""
    sim.prompt("c2.comfyui", "S", 30, stuck=True)
    time.sleep(0.5)
    sim.llm("c2.llm", "L", 1).join(30)
    llm = sim.at("c2.llm", "begin", "L")
    check(llm is not None and 2.5 <= llm - sim.results["S"][0] <= 10,
          "watchdog did not release the stuck job's hold in time")


def case_llamaswap_unreachable(sim):
    """If the hold can't open, /prompt answers 503 and nothing is queued."""
    port = 19399
    env = dict(os.environ, SIM_EVENTS=EVENTS, HOLD_MODEL_ID="orphan",
               LLAMASWAP_URL="http://127.0.0.1:1", HOLD_OPEN_TIMEOUT_S="2")
    p = subprocess.Popen([sys.executable, "fake_comfy.py", str(port)], cwd=SIM, env=env)
    try:
        time.sleep(1.5)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt?tag=X&secs=1", method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        check(code == 503, f"expected 503, got {code}")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/comfyui-hold/status") as r:
            status = json.load(r)
        check(status["tasks_remaining"] == 0, "job was queued without a hold")
        check(status["errors"] >= 1, "hold failure not recorded")
    finally:
        p.terminate()


def case_prompt_during_pending_swap(sim):
    """A second job arriving while an LLM waits: no deadlock, everything finishes."""
    sim.prompt("c2.comfyui", "A", 3)
    time.sleep(0.5)
    t = sim.llm("c2.llm", "L", 1)
    time.sleep(1)
    check(sim.prompt("c2.comfyui", "B", 2)[0] == 200, "second /prompt failed")
    t.join(30)
    b = sim.at("c2.comfyui", "job_end", "B")
    check(b is not None, "B did not finish")
    check(before(b, sim.at("c2.llm", "begin", "L")), "LLM interrupted the queued work")
    check(sim.results["L"][2] == 200, "LLM request failed")


def case_edge_lan_block(sim):
    """The edge: ComfyUI is loopback-only, LLMs are open to the LAN."""
    st, _ = http_from("127.0.0.2", "GET", "/upstream/c2.comfyui/system_stats")
    check(st == 403, f"LAN request to ComfyUI got {st}, expected 403")
    st, _ = http_from("127.0.0.2", "GET", "/comfyui/")
    check(st == 403, f"LAN request to /comfyui/ got {st}, expected 403")
    check(sim.at("c2.comfyui", "started") is None, "a blocked request still started ComfyUI")
    st, _ = http_from("127.0.0.2", "POST", "/v1/chat/completions",
                      {"model": "c0.llm", "tag": "L", "sleep": 0, "messages": []})
    check(st == 200, f"LAN LLM request got {st}")
    st, _ = http("GET", "/upstream/a4.comfyui/system_stats")
    check(st == 200, f"loopback request to ComfyUI got {st}")


def case_edge_drain(sim):
    """Drain flag: new work gets 503 + Retry-After; reads, /api and acks pass; in-flight finishes."""
    sim.prompt("a4.comfyui", "J", 1, ack=True)
    long_llm = sim.llm("c0.llm", "L1", 4)
    time.sleep(1)
    open(FLAG, "w").close()
    req = urllib.request.Request(BASE + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "c2.llm", "messages": []}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        check(False, "new work was accepted during a drain")
    except urllib.error.HTTPError as e:
        check(e.code == 503 and e.headers.get("Retry-After") == "60",
              f"expected 503 + Retry-After 60, got {e.code} {e.headers.get('Retry-After')}")
    check(http("GET", "/running")[0] == 200, "GET /running refused during a drain")
    check(sim.prompt("a4.comfyui", "K", 1)[0] == 503, "/prompt accepted during a drain")
    sim.wait_job("a4.comfyui", "J")
    check(sim.ack("a4.comfyui") == 200, "ack refused during a drain")
    long_llm.join(30)
    check(sim.results["L1"][2] == 200, "the in-flight LLM request was cut by the drain")


def case_gate_waits_for_work(sim):
    """A new image during a render + LLM request: drain at once, deploy only after all is done."""
    hook = Webhook(on_call=lambda: open(sim.running_file, "w").write("sha256:new"))
    hook.start()
    try:
        sim.start_gate(latest="sha256:old", running="sha256:old")
        time.sleep(2)
        sim.prompt("c2.comfyui", "J", 4, ack=True)
        sim.llm("c0.llm", "L", 6)
        time.sleep(0.5)
        with open(sim.latest_file, "w") as f:
            f.write("sha256:new")
        t_new = time.time()
        for _ in range(30):
            if os.path.exists(FLAG):
                break
            time.sleep(0.2)
        check(os.path.exists(FLAG) and time.time() - t_new < 4, "drain flag not set promptly")
        sim.wait_job("c2.comfyui", "J")
        time.sleep(1)
        check(not hook.calls, "deployed while a job waited for its ack")
        t_ack = time.time()
        sim.ack("c2.comfyui")
        for _ in range(60):
            if hook.calls:
                break
            time.sleep(0.5)
        check(len(hook.calls) == 1, f"webhook called {len(hook.calls)} times, expected 1")
        pulls = sim.pulls()
        check(len(pulls) == 2, f"pulled {len(pulls)} image(s), expected 2 (Portainer CE)")
        check(max(pulls) < hook.calls[0], "pulled after the webhook, so it would redeploy the old image")
        llm_end = sim.results["L"][1]
        check(sim.results["L"][2] == 200, "the LLM request was cut")
        check(hook.calls[0] >= max(t_ack, llm_end) + 2, "deployed before the stable-idle window")
        for _ in range(40):
            if not os.path.exists(FLAG):
                break
            time.sleep(0.5)
        check(not os.path.exists(FLAG), "drain flag not cleared after the deploy")
    finally:
        hook.close()


def case_gate_abort(sim):
    """Work that outlasts DRAIN_MAX_S: no deploy, flag cleared, postponement logged."""
    hook = Webhook()
    hook.start()
    try:
        sim.start_gate(latest="sha256:new", running="sha256:old", DRAIN_MAX_S=4)
        sim.prompt("a4.comfyui", "LONG", 15)
        time.sleep(9)
        check(not hook.calls, "deployed while work was in flight")
        check(not os.path.exists(FLAG), "drain flag left set after giving up")
        check("deploy postponed" in sim.gate_output(), "postponement not logged")
        check(sim.at("a4.comfyui", "stopped") is None, "the long job was interrupted")
    finally:
        hook.close()


def case_gate_startup_clears_flag(sim):
    """A drain flag left by a previous gate is cleared when the gate starts."""
    open(FLAG, "w").close()
    sim.start_gate(latest="sha256:same", running="sha256:same")
    for _ in range(30):
        if not os.path.exists(FLAG):
            break
        time.sleep(0.2)
    check(not os.path.exists(FLAG), "stale drain flag not cleared on startup")


def case_gate_deploy_no_effect(sim):
    """A deploy that doesn't take effect (Portainer CE reusing a cached image): the gate backs
    off instead of draining again on the next poll."""
    hook = Webhook()          # answers 204 but never changes the running digest
    hook.start()
    try:
        sim.start_gate(latest="sha256:new", running="sha256:old",
                       DEPLOY_WAIT_S=4, RETRY_AFTER_ABORT_S=30)
        for _ in range(60):
            if hook.calls:
                break
            time.sleep(0.5)
        check(len(hook.calls) == 1, f"webhook called {len(hook.calls)} times, expected 1")
        for _ in range(40):
            if "no redeploy observed" in sim.gate_output():
                break
            time.sleep(0.5)
        check("no redeploy observed" in sim.gate_output(), "an ineffective deploy was not logged")
        check(not os.path.exists(FLAG), "drain flag left set after an ineffective deploy")
        # The backoff: without it the main loop would see the digests still differ and drain again.
        time.sleep(8)
        check(len(hook.calls) == 1, f"redeployed {len(hook.calls)} times: the gate is looping")
        check(not os.path.exists(FLAG), "the gate drained again instead of backing off")
    finally:
        hook.close()


def case_gate_no_webhook_no_drain(sim):
    """Without PORTAINER_WEBHOOK_URL a new image must not drain: a drain 503s the whole box and
    could never end in a deploy."""
    sim.start_gate(latest="sha256:new", running="sha256:old", PORTAINER_WEBHOOK_URL="",
                   RETRY_AFTER_ABORT_S=30)
    for _ in range(40):
        if "not deploying" in sim.gate_output():
            break
        time.sleep(0.5)
    check("not deploying" in sim.gate_output(), "a new image without a webhook was not logged")
    check(not os.path.exists(FLAG), "drained although no deploy was possible")
    check(not sim.pulls(), "pulled images although no deploy was possible")
    st, _ = http("POST", "/v1/chat/completions",
                 {"model": "c0.llm", "tag": "N", "sleep": 1, "messages": []})
    check(st == 200, f"new work got {st}: the edge is draining although no deploy was possible")


CASES = [case_render_outlives_ttl, case_no_gap_at_submit, case_gateway_sequence,
         case_results_kept_until_ack, case_ack_timeout, case_wholebox_waits, case_watchdog,
         case_llamaswap_unreachable, case_prompt_during_pending_swap,
         case_edge_lan_block, case_edge_drain, case_gate_waits_for_work, case_gate_abort,
         case_gate_startup_clears_flag, case_gate_deploy_no_effect,
         case_gate_no_webhook_no_drain]


def main():
    binary, caddy = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    wanted = set(sys.argv[3:])
    failed = 0
    for case in CASES:
        if wanted and case.__name__ not in wanted:
            continue
        sim = Sim(binary, caddy)
        try:
            sim.start()
            case(sim)
            print(f"PASS {case.__name__}")
        except Exception as exc:  # AssertionError or plumbing failure
            failed += 1
            print(f"FAIL {case.__name__}: {exc}")
            sim.dump()
        finally:
            sim.stop()
    print(f"\n{'all passed' if not failed else f'{failed} failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
