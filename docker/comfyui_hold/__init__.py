"""ComfyUI adapter for the hold (see hold.py). Baked into the image by Dockerfile.comfyui.

Registers the hold and status routes, and a middleware that makes POST /prompt (and /api/prompt)
open the hold before the job is queued. Configured through env vars that llama-swap's
`docker run` sets (gen_config.py → comfyui_entry):
  HOLD_MODEL_ID        this instance's llama-swap model ID (required; without it the hold is
                       disabled and a warning is logged, e.g. in CI's import test)
  LLAMASWAP_URL        default http://127.0.0.1:9292
  HOLD_STALL_S         release after this long with no progress event (default 1800)
  HOLD_MAX_S           release after this long in total (default 14400)
  HOLD_OPEN_TIMEOUT_S  how long /prompt waits for the hold to open (default 10)
  HOLD_ACK_TIMEOUT_S   how long a finished X-Hold-Ack job keeps the hold waiting for its
                       client's ack (default 120)
"""
import logging
import os

from server import PromptServer

from .hold import ACK_PATH, HOLD_PATH, STATUS_PATH, HoldManager

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

log = logging.getLogger("comfyui_hold")


def _install():
    ps = PromptServer.instance
    model_id = os.environ.get("HOLD_MODEL_ID", "").strip()
    if not model_id:
        log.warning("comfyui_hold: HOLD_MODEL_ID is not set; the hold is DISABLED, so llama-swap "
                    "cannot see renders in progress and may unload this instance mid-render")
        return

    # Progress: every event ComfyUI sends to its clients (progress per sampler step,
    # executing, executed, ...) goes through send_sync from the worker thread. Counting them
    # gives a marker that moves even inside one long node, which last_node_id alone would not.
    events = [0]
    send_sync = ps.send_sync

    def counting_send_sync(event, data, sid=None):
        events[0] += 1
        return send_sync(event, data, sid)

    ps.send_sync = counting_send_sync

    manager = HoldManager(
        tasks_remaining=ps.prompt_queue.get_tasks_remaining,
        progress_marker=lambda: events[0],
        llamaswap_url=os.environ.get("LLAMASWAP_URL", "http://127.0.0.1:9292"),
        model_id=model_id,
        stall_s=float(os.environ.get("HOLD_STALL_S", 1800)),
        max_s=float(os.environ.get("HOLD_MAX_S", 14400)),
        open_timeout_s=float(os.environ.get("HOLD_OPEN_TIMEOUT_S", 10)),
        ack_timeout_s=float(os.environ.get("HOLD_ACK_TIMEOUT_S", 120)),
    )
    # ComfyUI mirrors every entry of ps.routes under /api when it adds its routes, after the
    # custom nodes have loaded. The app is not frozen yet, so the middleware list is still open.
    ps.routes.get(HOLD_PATH)(manager.hold_handler)
    ps.routes.get(STATUS_PATH)(manager.status_handler)
    ps.routes.post(ACK_PATH)(manager.ack_handler)
    ps.app.middlewares.append(manager.middleware())
    log.info("comfyui_hold: enabled for %s via %s", model_id, manager.url)


_install()
