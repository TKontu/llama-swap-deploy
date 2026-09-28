"""Fake ComfyUI for tests/sim, running the REAL hold (docker/comfyui_hold/hold.py).

It mimics what the hold depends on in ComfyUI: POST /prompt queues a job and returns at once,
one worker runs jobs in order, and every second of a job emits a progress event (unless the
job is `stuck`). Routes:
  POST /prompt?secs=N&tag=x[&stuck=1]  queue a job (behind the hold middleware)
  GET  /job/<tag>                       {"state": "queued"|"running"|"done"}
  GET  /system_stats                    readiness
  GET  /comfyui-hold/hold, /comfyui-hold/status, POST /comfyui-hold/ack   from HoldManager
Env: SIM_EVENTS, HOLD_MODEL_ID, LLAMASWAP_URL, HOLD_STALL_S, HOLD_MAX_S, HOLD_OPEN_TIMEOUT_S,
HOLD_ACK_TIMEOUT_S.
"""
import asyncio
import importlib.util
import json
import os
import signal
import sys
import time

from aiohttp import web

# Load hold.py by path: importing the comfyui_hold package would run its __init__.py, the
# ComfyUI adapter, which needs ComfyUI's `server` module.
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "docker", "comfyui_hold", "hold.py")
_spec = importlib.util.spec_from_file_location("comfyui_hold_hold", _path)
hold_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hold_mod)
ACK_PATH, HOLD_PATH, STATUS_PATH, HoldManager = (
    hold_mod.ACK_PATH, hold_mod.HOLD_PATH, hold_mod.STATUS_PATH, hold_mod.HoldManager)

port = int(sys.argv[1])
name = os.environ.get("HOLD_MODEL_ID", "fake-comfy")
EVENTS = os.environ["SIM_EVENTS"]


def event(what, tag=""):
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"t": time.time(), "model": name, "event": what, "tag": tag}) + "\n")


queue, running, jobs, progress = [], [], {}, [0]


async def worker():
    while True:
        if not queue:
            await asyncio.sleep(0.05)
            continue
        job = queue[0]
        running.append(job)
        queue.pop(0)
        jobs[job["tag"]] = "running"
        event("job_begin", job["tag"])
        for _ in range(int(job["secs"])):
            await asyncio.sleep(1)
            if not job["stuck"]:
                progress[0] += 1
        running.remove(job)
        jobs[job["tag"]] = "done"
        event("job_end", job["tag"])


async def prompt(request):
    q = request.query
    job = {"tag": q["tag"], "secs": float(q.get("secs", 1)), "stuck": q.get("stuck") == "1"}
    jobs[job["tag"]] = "queued"
    queue.append(job)
    return web.json_response({"prompt_id": job["tag"]})


async def job_state(request):
    return web.json_response({"state": jobs.get(request.match_info["tag"], "unknown")})


async def ok(request):
    return web.json_response({"ok": True})


def main():
    hold = HoldManager(
        tasks_remaining=lambda: len(queue) + len(running),
        progress_marker=lambda: progress[0],
        llamaswap_url=os.environ.get("LLAMASWAP_URL", "http://127.0.0.1:9292"),
        model_id=name,
        stall_s=float(os.environ.get("HOLD_STALL_S", 1800)),
        max_s=float(os.environ.get("HOLD_MAX_S", 14400)),
        open_timeout_s=float(os.environ.get("HOLD_OPEN_TIMEOUT_S", 10)),
        ack_timeout_s=float(os.environ.get("HOLD_ACK_TIMEOUT_S", 120)),
        heartbeat_s=1,
    )
    app = web.Application(middlewares=[hold.middleware()])
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/job/{tag}", job_state)
    app.router.add_get("/system_stats", ok)
    app.router.add_get(HOLD_PATH, hold.hold_handler)
    app.router.add_get(STATUS_PATH, hold.status_handler)
    app.router.add_post(ACK_PATH, hold.ack_handler)

    async def start_worker(app):
        app["worker"] = asyncio.ensure_future(worker())

    app.on_startup.append(start_worker)
    signal.signal(signal.SIGTERM, lambda *a: (event("stopped"), os._exit(0)))
    event("started")
    web.run_app(app, host="127.0.0.1", port=port, print=None, handle_signals=False)


main()
