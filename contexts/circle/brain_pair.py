"""The shared brain as two friends: one 122B model split across the two Strix Halo pods, live on the Circle screen (G6).

    pair = build_brain_pair()                        # never raises; without the pods' keys it answers configured false
    app.include_router(shared_brain_router(pair))    # GET /api/circle/shared-brain

The brain is Qwen3.5-122B-A10B (int4) served as ONE vLLM server with pipeline parallelism 2: 24 layers on each pod.
The head (the pod in the pod-a role of LAB_POD_ROLES, pod C on SJC) holds the first half and serves the OpenAI API on
port 8000 under the names "teacher" and "group-brain"; the other half (pod-b, pod D) runs headless. The two are joined
by a tunnel (lab/net/, PR 418). Each pod also runs its power sampler (lab/pod_power.py serve, port 8790). This box
reaches all of it only through the hub's Jupyter proxy, https://<hub>/user/<user>/proxy/<port>, with each account's
hub token, which stays here: the phone reads this route and never sees a token.

GET /api/circle/shared-brain
    {"configured": bool,
     "model": {"name": "Qwen3.5-122B-A10B", "params_b": 122, "active_b": 10, "quant": "int4", "served": [str]},
     "members": [{"name": "pod C", "role": "head", "node": "pod-a", "layers": 24, "online", "watts_now",
                  "watts_idle", "vram_gb"},
                 {"name": "pod D", "role": "half", "node": "pod-b", "layers": 24, ...}],
     "link": {"kind": "tunnel", "rtt_ms": 0.36, "mbps": 100},
     "serving": {"tok_s", "ttft_s", "requests_running"},
     "updated": ISO time, "reason": str | null}

Where each field comes from:
- model.served: the head's /v1/models (the ids it serves). The name, size and quantisation are the deployment's.
- online: the pod answered this refresh. The head is online when /v1/models or its sampler answers; the other half
  when its sampler answers, or when the head's /v1/models answers (the server only comes up with both halves).
- watts_now: the newest sample of that pod's sampler (socket_w: the whole APU socket, in watts), null when the sampler
  does not answer or its newest sample is more than STALE_S older than the pod's own clock says it is now.
- watts_idle: the pod's calibrated idle (config/telemetry_idle.yaml, ATLAS_TELEMETRY_IDLE_W; 10.27 W, T10).
- vram_gb: what each half of the model holds on its pod, measured at load on 2026-10-11 (34 to 35 GiB each). It is
  not read live; it is shown only while the pod is online.
- link: the tunnel as measured on 2026-10-11 (0.36 ms per message). Not read live either.
- serving: the head's /metrics (vLLM's Prometheus text). requests_running is vllm:num_requests_running; ttft_s is
  the mean of vllm:time_to_first_token_seconds and tok_s is one over the mean of vllm:inter_token_latency_seconds
  (vllm:time_per_output_token_seconds on an older server), both over the requests that finished since the previous
  refresh, else the last such figure this process saw, else the whole histogram since the server started.

Every fetch has a TIMEOUT_S timeout and the whole answer is cached for CACHE_S, so phones polling every 2 s cost at
most one round of fetches per CACHE_S. A source that does not answer gives null fields and online false; the route
answers 200 in every state, never an error page. Without the pods' keys (`reason` names the missing key NAMES, never
a value) it answers configured false and fetches nothing.

Settings, read the way lab/hub.py reads them (the process environment, then the .env files): LAB_POD_ROLES
(C:pod-a,D:pod-b on SJC; unset it means A:pod-a,B:pod-b), and per account letter AUP_HUB_USER_<X>, AUP_HUB_TOKEN_<X>
and AUP_HUB_URL_<X> (or LAB_HUB_URL). The head's account is required for configured true; the other half's is
optional (its member is then offline with null fields).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter

from contexts.telemetry.domain.energy import clock_offset, integrate
from lab import hub as lab_hub

log = logging.getLogger("atlas.circle.brain_pair")

HEAD_ROLE = "pod-a"
HALF_ROLE = "pod-b"
MODEL_PORT = 8000  # the head's vLLM server (lab/pod_a_teacher.sh)
SAMPLER_PORT = 8790  # lab/pod_power.py DEFAULT_PORT
TIMEOUT_S = 3.0
CACHE_S = 2.0
CHAT_TIMEOUT_S = 60.0  # one chat completion from the head (the Ask panel's comparison)
STALE_S = 15.0  # a sampler reads once a second: a newest sample older than this is not "now"
MODEL = {"name": "Qwen3.5-122B-A10B", "params_b": 122, "active_b": 10, "quant": "int4"}
LAYERS = 24  # pipeline parallel 2: 48 layers, 24 on each pod
VRAM_GB = 34.5  # GiB each half holds on its pod, measured at load on 2026-10-11 (34 to 35)
LINK = {"kind": "tunnel", "rtt_ms": 0.36, "mbps": 100}  # lab/net (PR 418), measured on 2026-10-11
TTFT = "vllm:time_to_first_token_seconds"
ITL = ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds")
RUNNING = "vllm:num_requests_running"
_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def parse_metrics(text: str, names: Sequence[str]) -> dict[str, float]:
    """The named series of a Prometheus text page, each summed over its label sets (engines, model names)."""
    wanted = set(names)
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE.match(line)
        if not m or m.group(1) not in wanted:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if value == value and value not in (float("inf"), float("-inf")):  # NaN and infinities are not a reading
            out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


class _Mean:
    """A histogram's mean over the requests counted since the previous read, falling back as the module says."""

    def __init__(self) -> None:
        self.prev: tuple[float, float] | None = None
        self.recent: float | None = None

    def update(self, total: float | None, count: float | None) -> float | None:
        if total is None or count is None or count <= 0:
            self.prev = None
            return self.recent
        if self.prev is not None:
            p_total, p_count = self.prev
            if count > p_count and total >= p_total:
                self.recent = (total - p_total) / (count - p_count)
            elif count < p_count:  # the server restarted: its histogram started again
                self.recent = None
        self.prev = (total, count)
        return self.recent if self.recent is not None else total / count


@dataclass(frozen=True)
class Pod:
    """One half of the brain: who it is on the card, and how to reach it (base is None without its account)."""

    name: str
    role: str  # "head" or "half"
    node: str  # pod-a or pod-b: the telemetry node, for its idle figure
    base: str | None = None  # https://<hub>/user/<user>/proxy
    token: str | None = field(default=None, repr=False)


class BrainPair:
    """Reads both pods through the hub proxy and folds them into the card's shape (see the module docstring)."""

    def __init__(self, head: Pod, half: Pod, *, idle: Mapping[str, float] | None = None, reason: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, user_agent: str = lab_hub.USER_AGENT,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic,
                 timeout_s: float = TIMEOUT_S, cache_s: float = CACHE_S) -> None:
        self.head = head
        self.half = half
        self.idle = dict(idle or {})
        self.reason = reason
        self.transport = transport
        self.user_agent = user_agent
        self.clock = clock
        self.monotonic = monotonic
        self.timeout_s = timeout_s
        self.cache_s = cache_s
        self._client: httpx.AsyncClient | None = None
        self._lock: asyncio.Lock | None = None
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0
        self._last_seq: dict[str, int] = {}
        self._ttft = _Mean()
        self._itl = _Mean()
        self._chat_client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return self.head.base is not None

    # --- reading the pods ---

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s, follow_redirects=False,
                                             headers={"User-Agent": self.user_agent})
        return self._client

    async def _get(self, pod: Pod, port: int, path: str, params: Mapping[str, Any] | None = None
                   ) -> httpx.Response | None:
        """One GET through the proxy, or None for anything but a 200 in time (no answer, a login page, a 502)."""
        if pod.base is None:
            return None
        try:
            reply = await asyncio.wait_for(
                self._http().get(f"{pod.base}/{port}{path}", params=params,
                                 headers={"Authorization": f"token {pod.token}"}), self.timeout_s)
        except (httpx.HTTPError, asyncio.TimeoutError, TimeoutError, OSError):
            return None
        return reply if reply.status_code == 200 else None

    @staticmethod
    def _json(reply: httpx.Response | None) -> dict[str, Any] | None:
        if reply is None:
            return None
        try:
            body = reply.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    async def _models(self) -> list[str] | None:
        body = self._json(await self._get(self.head, MODEL_PORT, "/v1/models"))
        if body is None or not isinstance(body.get("data"), list):
            return None
        return [str(m["id"]) for m in body["data"] if isinstance(m, dict) and m.get("id")]

    async def _serving(self) -> dict[str, Any]:
        reply = await self._get(self.head, MODEL_PORT, "/metrics")
        if reply is None:
            return {"tok_s": None, "ttft_s": None, "requests_running": None}
        itl_name = next((n for n in ITL if f"{n}_count" in reply.text), ITL[0])
        found = parse_metrics(reply.text, [RUNNING, f"{TTFT}_sum", f"{TTFT}_count", f"{itl_name}_sum",
                                           f"{itl_name}_count"])
        ttft = self._ttft.update(found.get(f"{TTFT}_sum"), found.get(f"{TTFT}_count"))
        itl = self._itl.update(found.get(f"{itl_name}_sum"), found.get(f"{itl_name}_count"))
        running = found.get(RUNNING)
        return {"tok_s": round(1.0 / itl, 1) if itl else None, "ttft_s": round(ttft, 2) if ttft is not None else None,
                "requests_running": int(running) if running is not None else None}

    async def _samples(self, pod: Pod, after: int) -> dict[str, Any] | None:
        return self._json(await self._get(pod, SAMPLER_PORT, "/samples", {"after": max(0, after)}))

    async def _watts(self, pod: Pod) -> tuple[bool, float | None]:
        """(the sampler answered, its newest watts or None). One fetch once the sampler's sequence is known."""
        known = self._last_seq.get(pod.name)
        if known is None:
            status = self._json(await self._get(pod, SAMPLER_PORT, "/status"))
            if status is None or not isinstance(status.get("last_seq"), int):
                return False, None
            known = status["last_seq"]
        body = await self._samples(pod, known - 2)
        if body is None:
            return False, None
        last = body.get("last_seq")
        samples = [s for s in body.get("samples") or [] if isinstance(s, dict) and isinstance(s.get("seq"), int)]
        newest = samples[-1]["seq"] if samples else None
        if isinstance(last, int) and last > 0 and newest != last:
            # the sampler restarted (its count began again) or a long pause left more than one page: read the end
            body = await self._samples(pod, last - 2) or body
            samples = [s for s in body.get("samples") or [] if isinstance(s, dict) and isinstance(s.get("seq"), int)]
        if isinstance(body.get("last_seq"), int):
            self._last_seq[pod.name] = body["last_seq"]
        for s in reversed(samples):
            event = s.get("event") or {}
            w = event.get("socket_w")
            if not isinstance(w, int | float) or isinstance(w, bool):
                continue
            try:
                age = _epoch(str(body.get("pod_sent") or event["ts"])) - _epoch(str(event["ts"]))
            except (KeyError, ValueError):
                return True, None
            return True, (round(float(w), 1) if age <= STALE_S else None)
        return True, None

    # --- the answer ---

    def _member(self, pod: Pod, online: bool, watts: float | None) -> dict[str, Any]:
        return {"name": pod.name, "role": pod.role, "node": pod.node, "layers": LAYERS, "online": online,
                "watts_now": watts if online else None, "watts_idle": self.idle.get(pod.node),
                "vram_gb": VRAM_GB if online else None}

    def offline(self) -> dict[str, Any]:
        """The shape with every live field null: no keys, or nothing could be read."""
        return {"configured": self.configured, "model": {**MODEL, "served": []},
                "members": [self._member(self.head, False, None), self._member(self.half, False, None)],
                "link": dict(LINK), "serving": {"tok_s": None, "ttft_s": None, "requests_running": None},
                "updated": _iso(self.clock()), "reason": self.reason}

    async def _build(self) -> dict[str, Any]:
        if not self.configured:
            return self.offline()
        models, serving, (head_up, head_w), (half_up, half_w) = await asyncio.gather(
            self._models(), self._serving(), self._watts(self.head), self._watts(self.half))
        serving_up = bool(models)
        return {"configured": True, "model": {**MODEL, "served": models or []},
                "members": [self._member(self.head, serving_up or head_up, head_w),
                            self._member(self.half, serving_up or half_up, half_w)],
                "link": dict(LINK), "serving": serving, "updated": _iso(self.clock()), "reason": self.reason}

    async def snapshot(self) -> dict[str, Any]:
        """The card's shape, fetched at most once per cache_s however many phones ask."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._cached is not None and self.monotonic() - self._cached_at < self.cache_s:
                return self._cached
            self._cached = await self._build()
            self._cached_at = self.monotonic()
            return self._cached

    # --- asking the brain, and what an answer cost on both pods (the Ask panel's comparison) ---

    async def chat(self, body: Mapping[str, Any], timeout_s: float = CHAT_TIMEOUT_S) -> dict[str, Any]:
        """One OpenAI chat completion from the head (port 8000) through the hub proxy, with the head's token.
        Raises RuntimeError for anything but a 200 with JSON, httpx errors for no answer."""
        if self.head.base is None:
            raise RuntimeError(self.reason or "this box has no keys for the head pod")
        if self._chat_client is None:
            self._chat_client = httpx.AsyncClient(transport=self.transport, timeout=timeout_s, follow_redirects=False,
                                                  headers={"User-Agent": self.user_agent})
        reply = await self._chat_client.post(f"{self.head.base}/{MODEL_PORT}/v1/chat/completions", json=dict(body),
                                             headers={"Authorization": f"token {self.head.token}"})
        if reply.status_code != 200:
            raise RuntimeError(f"the head pod answered HTTP {reply.status_code}")
        data = reply.json()
        if not isinstance(data, dict):
            raise RuntimeError("the head pod's answer is not a chat completion")
        return data

    async def _pod_window(self, pod: Pod, start: float, end: float) -> dict[str, Any] | None:
        """That pod's gross joules over [start, end] on this box's clock, moved onto the pod's clock by the offset
        measured from this very read (four timestamps). None when its sampler does not cover the window."""
        if pod.base is None:
            return None
        if pod.name not in self._last_seq:
            await self._watts(pod)  # learns where the sampler's count is
        known = self._last_seq.get(pod.name)
        if known is None:
            return None
        span = int(self.clock() - start) + 20
        t1 = self.clock()
        body = self._json(await self._get(pod, SAMPLER_PORT, "/samples", {"after": max(0, known - span),
                                                                            "limit": span + 600}))
        t4 = self.clock()
        if body is None:
            return None
        if isinstance(body.get("last_seq"), int):
            self._last_seq[pod.name] = body["last_seq"]
        try:
            off = clock_offset(t1, _epoch(body["pod_received"]), _epoch(body["pod_sent"]), t4)[0]
        except (KeyError, TypeError, ValueError):
            off = 0.0
        trace = []
        for sample in body.get("samples") or []:
            event = (sample or {}).get("event") or {}
            w = event.get("socket_w")
            if isinstance(w, int | float) and not isinstance(w, bool) and event.get("ts"):
                trace.append((_epoch(str(event["ts"])), float(w)))
        trace.sort()
        joules = integrate(trace, start + off, end + off) if trace else None
        if joules is None:
            return None
        seconds = max(end - start, 1e-9)
        return {"joules": round(joules, 1), "watts_avg": round(joules / seconds, 1),
                "idle_w": self.idle.get(pod.node),
                "samples": sum(1 for t, _ in trace if start + off <= t <= end + off)}

    async def window_energy(self, start: float, end: float, *, wait_s: float = 2.5,
                            sleep: Callable[[float], Any] = asyncio.sleep) -> dict[str, Any] | None:
        """Both pods' gross joules over a request window (epoch seconds on this box's clock), the T4 rule: their
        power integrated over the window, nothing subtracted, each pod's idle beside it. Waits up to wait_s for each
        sampler's first sample after the end (they read once a second). None when neither pod's sampler covers it.

            {"joules", "watts_avg", "seconds", "pods": {"C": {"joules", "watts_avg", "idle_w", "samples"}, "D": ...}}"""
        pods: dict[str, Any] = {}
        waited = 0.0
        for pod in (self.head, self.half):
            letter = pod.name.split()[-1]
            got = await self._pod_window(pod, start, end)
            while got is None and pod.base is not None and waited < wait_s:
                await sleep(1.0)
                waited += 1.0
                got = await self._pod_window(pod, start, end)
            pods[letter] = got
        measured = [p["joules"] for p in pods.values() if p]
        if not measured:
            return None
        seconds = max(end - start, 1e-9)
        total = sum(measured)
        return {"joules": round(total, 1), "watts_avg": round(total / seconds, 1), "seconds": round(end - start, 2),
                "pods": pods}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._chat_client is not None:
            await self._chat_client.aclose()
            self._chat_client = None


def _pod(role: str, kind: str, roles: Mapping[str, str], environ: Mapping[str, str] | None,
         env_files: Sequence[Path] | None) -> tuple[Pod, list[str]]:
    """The pod in `role`, and the key names it is missing (none when its account loaded)."""
    letters = sorted(letter for letter, r in roles.items() if r == role)
    if len(letters) != 1:
        return Pod(f"the {kind} pod", kind, role), [f"{lab_hub.ENV_POD_ROLES} ({role})"]
    letter = letters[0]
    try:
        account = lab_hub.load_accounts([letter], environ=environ, env_files=env_files)[letter]
    except lab_hub.MissingCredentials as exc:
        return Pod(f"pod {letter}", kind, role), list(exc.missing)
    hub = (account.url or lab_hub.DEFAULT_URL).rstrip("/")
    return Pod(f"pod {letter}", kind, role, f"{hub}/user/{quote(account.user, safe='')}/proxy", account.token), []


def _idle(environ: Mapping[str, str]) -> dict[str, float]:
    from contexts.telemetry.application.idle import idle_figures

    try:
        return {node: fig.watts for node, fig in idle_figures(environ).items()}
    except ValueError as exc:
        log.warning("no idle figures for the shared brain's pods: %s", exc)
        return {}


def build_brain_pair(environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None,
                     **kwargs: Any) -> BrainPair:
    """The pair from the settings in the module docstring. Never raises: what is missing goes in `reason`."""
    try:
        roles = lab_hub.pod_roles(environ, env_files)
    except lab_hub.PodRoleError as exc:
        reason = f"the shared brain is not set up: {exc}"
        log.warning("%s", reason)
        return BrainPair(Pod("the head pod", "head", HEAD_ROLE), Pod("the half pod", "half", HALF_ROLE),
                         reason=reason, **kwargs)
    head, head_missing = _pod(HEAD_ROLE, "head", roles, environ, env_files)
    half, half_missing = _pod(HALF_ROLE, "half", roles, environ, env_files)
    reason = None
    if head_missing or half_missing:
        reason = "missing keys: " + ", ".join(head_missing + half_missing)
        log.warning("shared brain: %s", reason)
    kwargs.setdefault("idle", _idle(os.environ if environ is None else environ))
    return BrainPair(head, half, reason=reason, **kwargs)


def shared_brain_router(pair: BrainPair | None) -> APIRouter:
    router = APIRouter()

    @router.get("/api/circle/shared-brain")
    async def shared_brain() -> dict[str, Any]:
        if pair is None:
            return BrainPair(Pod("the head pod", "head", HEAD_ROLE), Pod("the half pod", "half", HALF_ROLE),
                             reason="the shared brain is not wired on this box").offline()
        try:
            return await pair.snapshot()
        except Exception:  # the card answers in every state: a bug here must not become an error page
            log.exception("reading the shared brain failed")
            return pair.offline()

    return router
