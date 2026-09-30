"""Keep ComfyUI visible to llama-swap while it has work (the "hold").

llama-swap never evicts or idle-unloads a model while a request to it is still open. ComfyUI's
POST /prompt returns as soon as the job is queued, though, so a render in progress looks idle, and
a c2 LLM request or the TTL could stop the container mid-render. The hold closes that gap: while
ComfyUI has work queued or running, it keeps one request to itself open *through llama-swap*
(GET <llama-swap>/upstream/<model id>/comfyui-hold/hold). To llama-swap that is an ordinary
request in flight, so the model stays loaded until the work is done.

Guarantees:
  * No gap at submit. The middleware opens (and confirms) the hold before the /prompt handler
    runs, so the job is never queued without one.
  * Results are collected before release. A client that sends `X-Hold-Ack: 1` with its
    POST /prompt (media-gateway does) keeps the hold open after the job finishes, until it
    POSTs /comfyui-hold/ack once it has fetched /history and the outputs. ComfyUI's history is
    in memory, so releasing earlier could let a waiting LLM evict ComfyUI and lose it. Capped by
    `ack_timeout_s`, so a vanished client can't hold the card. Jobs without the header (the
    browser UI) release as soon as the queue is empty.
  * Fair on a shared card. Beyond that there is no grace period. A queued LLM request gets the
    card between two jobs (the next job cold-loads again).
  * Fails loudly. If the hold can't be opened, /prompt answers 503 and nothing is queued.
  * Bounded. A watchdog releases the hold when no progress event has been seen for `stall_s`,
    or after `max_s`, so a stuck job can't keep a shared card forever.

This module has no ComfyUI imports: the adapter (__init__.py) passes in callables, and the
simulator in tests/sim runs this same code against a real llama-swap.
"""
import asyncio
import logging
import secrets
import time

import aiohttp
from aiohttp import web

log = logging.getLogger("comfyui_hold")

HOLD_PATH = "/comfyui-hold/hold"
STATUS_PATH = "/comfyui-hold/status"
ACK_PATH = "/comfyui-hold/ack"
ACK_HEADER = "X-Hold-Ack"


class HoldError(RuntimeError):
    pass


class HoldManager:
    def __init__(self, tasks_remaining, progress_marker, llamaswap_url, model_id,
                 stall_s=1800, max_s=14400, open_timeout_s=10, ack_timeout_s=120,
                 heartbeat_s=15, poll_s=0.25, api_key=""):
        self.tasks_remaining = tasks_remaining      # () -> int: queued + running jobs
        self.progress_marker = progress_marker      # () -> hashable, changes on any progress
        self.url = f"{llamaswap_url.rstrip('/')}/upstream/{model_id}{HOLD_PATH}"
        # llama-swap's apiKeys cover /upstream/* too, so the hold has to authenticate to open
        # itself. Empty means llama-swap has no keys configured; send no header at all then.
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.stall_s = stall_s
        self.max_s = max_s
        self.open_timeout_s = open_timeout_s
        self.ack_timeout_s = ack_timeout_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s

        self._lock = asyncio.Lock()
        self._token = None           # token of the hold currently wanted
        self._confirmed = None       # asyncio.Event, set by the handler for that token
        self._client_task = None
        self._open = False           # a handler for the current token is streaming
        self._since = None
        self._submitting = 0         # /prompt requests between "hold ensured" and "queued"
        self._awaiting_ack = 0       # finished-or-queued jobs whose client hasn't acked yet
        self._last_reason = None
        self._errors = 0
        self._last_error = None

    # --- client side: open the hold through llama-swap ------------------------------------

    async def ensure_open(self):
        """Return once a hold is confirmed open; raise HoldError otherwise."""
        async with self._lock:
            if self._open:
                return
            if self._client_task is not None:
                self._client_task.cancel()
            token = secrets.token_hex(16)
            self._token = token
            self._confirmed = asyncio.Event()
            self._client_task = asyncio.ensure_future(self._client(token))
            waiter = asyncio.ensure_future(self._confirmed.wait())
            done, _ = await asyncio.wait({waiter, self._client_task},
                                         timeout=self.open_timeout_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                return
            waiter.cancel()
            if self._client_task in done:
                exc = self._client_task.exception()
                msg = f"hold request failed before it was confirmed: {exc!r}"
            else:
                self._client_task.cancel()
                msg = f"hold not confirmed within {self.open_timeout_s}s"
            self._token = None
            self._error(msg)
            raise HoldError(msg)

    async def _client(self, token):
        """Hold one request open; reopen if it ends while work remains."""
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        first = True
        while True:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(self.url, params={"token": token},
                                           headers=self.headers) as resp:
                        if resp.status != 200:
                            raise HoldError(f"hold request returned HTTP {resp.status}: "
                                            f"{(await resp.text())[:200]}")
                        async for _ in resp.content.iter_any():
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # HTTP error, connection lost, llama-swap restarting, ...
                if first and not self._confirmed.is_set():
                    raise  # ensure_open reports it; /prompt answers 503
                self._error(f"hold request failed: {exc!r}")
            first = False
            if self._token != token:
                return
            # The handler records why it ended. Queue empty or a watchdog release is expected;
            # anything else (proxy dropped it, llama-swap restarted) while work remains is not.
            if self._last_reason is not None and self._last_reason.startswith(("queue empty", "watchdog")):
                return
            if self.tasks_remaining() == 0 and self._submitting == 0 and self._awaiting_ack == 0:
                return
            self._open = False
            self._error(f"hold ended unexpectedly ({self._last_reason}) with "
                        f"{self.tasks_remaining()} job(s) left; reopening")
            await asyncio.sleep(1)

    # --- server side: the held request -----------------------------------------------------

    async def hold_handler(self, request):
        token = request.query.get("token", "")
        if not token or token != self._token:
            return web.Response(status=409, text="unknown or stale hold token\n")
        resp = web.StreamResponse(headers={"Content-Type": "text/plain", "Cache-Control": "no-store"})
        await resp.prepare(request)
        await resp.write(b"hold\n")
        # By the time this handler runs, llama-swap has already counted the request as in flight.
        self._open = True
        self._since = time.time()
        self._last_reason = None
        self._confirmed.set()
        log.info("hold opened")
        try:
            reason = await self._watch(resp, token)
        except (ConnectionResetError, asyncio.CancelledError):
            reason = "connection closed"
            if self._token == token:
                self._open = False
            self._last_reason = reason
            raise
        self._last_reason = reason
        log.info("hold released: %s", reason)
        try:
            await resp.write_eof()
        except ConnectionResetError:
            pass
        return resp

    async def _watch(self, resp, token):
        start = last_beat = last_progress = time.monotonic()
        idle_since = None
        marker = self.progress_marker()
        while True:
            await asyncio.sleep(self.poll_s)
            if self._token != token:
                return "superseded"
            now = time.monotonic()
            # No await between these checks and clearing _open: the middleware either sees the
            # hold still open (and bumps _submitting first, keeping it open) or sees it closed
            # and opens a new one. It can't slip in between.
            if self.tasks_remaining() == 0 and self._submitting == 0:
                if self._awaiting_ack == 0:
                    self._open = False
                    return "queue empty"
                idle_since = idle_since or now
                if now - idle_since >= self.ack_timeout_s:
                    self._error(f"no ack for {self._awaiting_ack} job(s) within "
                                f"{self.ack_timeout_s}s; releasing the hold")
                    self._awaiting_ack = 0
                    self._open = False
                    return f"queue empty (ack timeout {self.ack_timeout_s}s)"
                if now - last_beat >= self.heartbeat_s:
                    await resp.write(b".\n")
                    last_beat = now
                continue
            idle_since = None
            m = self.progress_marker()
            if m != marker:
                marker, last_progress = m, now
            elif now - last_progress >= self.stall_s:
                self._open = False
                self._error(f"watchdog: no progress for {self.stall_s}s; releasing the hold")
                return f"watchdog: no progress for {self.stall_s}s"
            if now - start >= self.max_s:
                self._open = False
                self._error(f"watchdog: hold reached its {self.max_s}s maximum; releasing it")
                return f"watchdog: maximum {self.max_s}s reached"
            if now - last_beat >= self.heartbeat_s:
                await resp.write(b".\n")
                last_beat = now

    # --- /prompt middleware + status -----------------------------------------------------

    def middleware(self, paths=("/prompt", "/api/prompt")):
        @web.middleware
        async def hold_middleware(request, handler):
            if request.method != "POST" or request.path not in paths:
                return await handler(request)
            self._submitting += 1
            try:
                try:
                    await self.ensure_open()
                except HoldError as exc:
                    return web.json_response(
                        {"error": {"type": "hold_unavailable", "message": str(exc),
                                   "details": "ComfyUI could not register this job with "
                                              "llama-swap, so it was not queued."},
                         "node_errors": {}},
                        status=503)
                response = await handler(request)
                if request.headers.get(ACK_HEADER) == "1" and response.status == 200:
                    self._awaiting_ack += 1
                return response
            finally:
                self._submitting -= 1
        return hold_middleware

    async def ack_handler(self, request):
        """The client has collected a job's results (POST, once per X-Hold-Ack job)."""
        self._awaiting_ack = max(0, self._awaiting_ack - 1)
        return web.json_response({"awaiting_ack": self._awaiting_ack})

    def status(self):
        return {"open": self._open, "since": self._since, "url": self.url,
                "tasks_remaining": self.tasks_remaining(), "awaiting_ack": self._awaiting_ack,
                "last_release": self._last_reason,
                "errors": self._errors, "last_error": self._last_error}

    async def status_handler(self, request):
        return web.json_response(self.status())

    def _error(self, msg):
        self._errors += 1
        self._last_error = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        log.error(msg)
