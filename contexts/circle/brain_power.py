"""The Circle's shared brain on the Circle screen: where it runs, its live watts, and what the last answer cost.

    shared = build_shared_brain(circle_cfg, bus)          # None without a Circle; brain or power may each be None
    app.include_router(group_brain_router(shared.brain, guard=authorized))   # contexts/circle/brain_api.py (G4)
    app.include_router(brain_power_router(shared.power))  # GET /api/circle/brain/power

The group brain (G4) is the teacher on the pod in the pod-a role (LAB_POD_ROLES; account C on SJC), reached through
the hub proxy. That pod runs lab/pod_power.py, which serves its power samples on port 8790 behind the same proxy.
BrainPower relays a window of those samples to the phone with the hub token added here, on the box, so the browser
never sees a token, and works out the energy of the last group-brain answer from them.

GET /api/circle/brain/power?after=<seq>
    {"configured", "node", "host", "where", "idle_w", "samples": [{"seq", "ts", "w"}], "last_seq",
     "offset_s", "last_answer": {...} | null, "error"}
    `samples` are the pod's own readings (APU socket watts, its clock) after `after`; with no `after`, the last two
    minutes. `offset_s` is the pod's clock minus this box's, measured from the sampler's own timestamps (four-timestamp
    exchange, the smallest round trip of the last ten), so a page can draw both on one time axis. 200 in every state:
    with no Circle or no pod account `configured` is false and `error` says why; a pod that does not answer gives
    `error` and no samples.

What an answer cost. The window is the group brain's own: circle.brain.asked to circle.brain.answered (ok) in this
box's log, moved onto the pod's clock by the measured offset. Joules are the pod's power integrated over that window
(contexts/telemetry/domain/energy.integrate: bracketing samples, linear at both edges), gross, as Context: Telemetry
defines joules per answer; the pod's calibrated idle (10.27 W, config/telemetry_idle.yaml, T10) is stored beside it,
and `above_idle_j` is the gross figure minus idle over the window. `low_confidence` names its reasons: fewer than 10
samples (the sampler reads once a second), or the pod already above idle + 15 W before the question (other work on
the teacher, which the figure then includes). Nothing here is sent anywhere: it reads the sampler and the log.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Query

from bus import EventBus
from contexts.telemetry.domain.energy import clock_offset, integrate
from contracts.circle import BrainAnswered, BrainAsked
from contracts.events import Event
from lab import hub as lab_hub

log = logging.getLogger("atlas.circle.brain_power")

SAMPLER_PORT = 8790  # lab/pod_power.py DEFAULT_PORT
IDLE_W = 10.27  # config/telemetry_idle.yaml, pod-a (T10: deep idle with the teacher loaded)
SPAN_S = 120  # the first page: the last two minutes
MIN_SAMPLES = 10
BUSY_MARGIN_W = 15.0
KEEP = 900  # samples kept for the last answer: fifteen minutes at one a second
WHERE = "the teacher on the lab's AMD Strix Halo pod (Radeon 8060S), through the hub"


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _Ask:
    ask_id: str
    asked: float  # this box's clock, epoch seconds
    answered: float | None = None
    ok: bool = False
    result: dict[str, Any] | None = None


class BrainPower:
    """Reads the teacher pod's power sampler through the hub proxy and keeps the last group-brain answer's window."""

    def __init__(self, base_url: str, token: str, *, host: str, node: str = "pod-a", idle_w: float = IDLE_W,
                 user_agent: str = lab_hub.USER_AGENT, transport: httpx.AsyncBaseTransport | None = None,
                 clock=time.time) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.host = host
        self.node = node
        self.idle_w = idle_w
        self.clock = clock
        self._client = httpx.AsyncClient(transport=transport, timeout=10.0, follow_redirects=False,
                                         headers={"User-Agent": user_agent, "Accept": "application/json"})
        self._samples: deque[tuple[int, float, float]] = deque(maxlen=KEEP)  # (seq, pod epoch s, watts)
        self._exchanges: deque[tuple[float, float]] = deque(maxlen=10)  # (offset, round trip)
        self._last_seq: int | None = None
        self._asks: dict[str, _Ask] = {}
        self._last: _Ask | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None,
                 **kwargs: Any) -> "BrainPower":
        """The pod in the teacher's pod-a role (LAB_POD_ROLES), its keys and hub as lab/hub.py reads them."""
        from contexts.circle.infrastructure.teacher import TEACHER_ROLE, teacher_letter

        roles = lab_hub.pod_roles(environ, env_files)
        letter = teacher_letter(roles)
        account = lab_hub.load_accounts([letter], environ=environ, env_files=env_files)[letter]
        hub = (account.url or lab_hub.DEFAULT_URL).rstrip("/")
        base = f"{hub}/user/{quote(account.user, safe='')}/proxy/{SAMPLER_PORT}"
        return cls(base, account.token, host=lab_hub.hub_of(hub), node=TEACHER_ROLE, **kwargs)

    # --- the log: the group brain's own windows ---

    def on_event(self, e: Event) -> None:
        if isinstance(e, BrainAsked):
            self._asks[e.ask_id] = _Ask(e.ask_id, e.ts.timestamp())
            while len(self._asks) > 50:
                self._asks.pop(next(iter(self._asks)))
        elif isinstance(e, BrainAnswered) and (a := self._asks.get(e.ask_id)):
            a.answered, a.ok = e.ts.timestamp(), e.ok
            if e.ok:
                self._last = a

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("circle.brain.", self._on, replay=True)  # state only: the windows

    async def _on(self, e: Event) -> None:
        self.on_event(e)

    # --- the pod ---

    @property
    def offset(self) -> float | None:
        """The pod's clock minus this box's, in seconds, from the exchange with the smallest round trip."""
        return min(self._exchanges, key=lambda x: x[1])[0] if self._exchanges else None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        t1 = self.clock()
        reply = await self._client.get(f"{self.base_url}{path}", params=params,
                                       headers={"Authorization": f"token {self._token}"})
        t4 = self.clock()
        if reply.status_code != 200:
            raise RuntimeError(f"the pod's sampler answered HTTP {reply.status_code}")
        body = reply.json()
        try:
            off, rtt, _ = clock_offset(t1, _epoch(body["pod_received"]), _epoch(body["pod_sent"]), t4)
            self._exchanges.append((off, rtt))
        except (KeyError, TypeError, ValueError):
            pass
        return body

    async def _fetch(self, after: int) -> list[tuple[int, float, float]]:
        body = await self._get("/samples", {"after": max(0, after)})
        out = []
        for s in body.get("samples") or []:
            ev = s.get("event") or {}
            w = ev.get("socket_w")
            if isinstance(s.get("seq"), int) and isinstance(w, int | float) and ev.get("ts"):
                out.append((s["seq"], _epoch(ev["ts"]), float(w)))
        if out:
            newest = self._samples[-1][0] if self._samples else -1
            self._samples.extend(x for x in out if x[0] > newest)
            self._last_seq = max(self._last_seq or 0, out[-1][0])
        elif isinstance(body.get("last_seq"), int):
            self._last_seq = max(self._last_seq or 0, body["last_seq"])
        return out

    async def window(self, after: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"configured": True, "node": self.node, "host": self.host, "where": WHERE,
                               "idle_w": self.idle_w, "samples": [], "last_seq": self._last_seq, "error": None}
        async with self._lock:
            try:
                if after is None:
                    if self._last_seq is None:
                        status = await self._get("/status")
                        self._last_seq = int(status.get("last_seq") or 0)
                    after = self._last_seq - SPAN_S
                page = await self._fetch(after)
                out["samples"] = [{"seq": q, "ts": _iso(t), "w": w} for q, t, w in page]
                out["last_seq"] = self._last_seq
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                out["error"] = f"the pod's power sampler did not answer ({type(exc).__name__}: {str(exc)[:80]})"
            out["offset_s"] = self.offset
            out["last_answer"] = await self._answer()
        return out

    async def _answer(self) -> dict[str, Any] | None:
        a = self._last
        if a is None or a.answered is None:
            return None
        if a.result is not None:
            return a.result
        off = self.offset
        if off is None:
            return None
        start, end = a.asked + off, a.answered + off
        if not self._samples or self._samples[0][1] > start - 10:  # the window is older than what is kept: read back
            span = int(self.clock() + off - start) + 30
            await self._fetch((self._last_seq or 0) - span)
            self._samples = deque(sorted(self._samples), maxlen=KEEP)
        if not self._samples or self._samples[-1][1] < end:
            return {"ask_id": a.ask_id, "pending": True}
        trace = [(t, w) for _, t, w in self._samples]
        joules = integrate(trace, start, end)
        if joules is None:
            return {"ask_id": a.ask_id, "pending": False, "joules": None,
                    "reason": "the pod's samples do not cover the answer's window"}
        secs = end - start
        inside = [w for _, t, w in self._samples if start <= t <= end]
        before = [w for _, t, w in self._samples if start - 10 <= t < start]
        pre = sum(before) / len(before) if before else None
        reasons = []
        if len(inside) < MIN_SAMPLES:
            reasons.append(f"{len(inside)} samples in the window (the sampler reads once a second)")
        if pre is not None and pre > self.idle_w + BUSY_MARGIN_W:
            reasons.append(f"the pod was already at {pre:.0f} W before the question (other work on the teacher)")
        a.result = {"ask_id": a.ask_id, "pending": False, "asked": _iso(a.asked), "answered": _iso(a.answered),
                    "seconds": round(secs, 2), "joules": round(joules, 1),
                    "above_idle_j": round(joules - self.idle_w * secs, 1),
                    "mean_w": round(joules / secs, 1) if secs > 0 else None, "samples": len(inside),
                    "before_w": round(pre, 1) if pre is not None else None, "idle_w": self.idle_w,
                    "offset_s": round(off, 3), "low_confidence": bool(reasons), "reasons": reasons}
        return a.result


@dataclass
class SharedBrain:
    brain: Any | None  # contexts.circle.application.group_brain.GroupBrain
    power: BrainPower | None
    reason: str | None = None


def build_shared_brain(circle_cfg: Mapping[str, Any], bus: EventBus, environ: Mapping[str, str] | None = None
                       ) -> SharedBrain | None:
    """The group brain and its power reading, for a box in a Circle with the teacher pod's account in its .env.

    Each part is None, with the reason logged once, when its keys are missing: the rest of the gateway is unchanged."""
    if not (circle_cfg.get("member_id") and circle_cfg.get("circle_id")):
        return None
    from contexts.circle.application.group_brain import GroupBrain
    from contexts.circle.infrastructure.teacher import TeacherLink

    brain = power = None
    reason = None
    try:
        brain = GroupBrain(circle_cfg["circle_id"], circle_cfg["member_id"], TeacherLink.from_env(environ=environ),
                           bus.publish)
        bus.subscribe("circle.brain.consent", brain.on_consent, replay=True)  # state only
    except (lab_hub.MissingCredentials, lab_hub.PodRoleError, ValueError) as exc:
        reason = f"no group brain: {exc}"
        log.warning("%s", reason)
    try:
        power = BrainPower.from_env(environ)
        power.subscribe(bus)
    except (lab_hub.MissingCredentials, lab_hub.PodRoleError, ValueError) as exc:
        reason = reason or f"no brain power reading: {exc}"
        log.warning("no brain power reading: %s", exc)
    return SharedBrain(brain, power, reason)


def brain_power_router(power: BrainPower | None, reason: str | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/api/circle/brain/power")
    async def brain_power(after: int | None = Query(default=None, ge=0)) -> dict[str, Any]:
        if power is None:
            return {"configured": False, "node": None, "host": None, "where": WHERE, "idle_w": IDLE_W, "samples": [],
                    "last_seq": None, "offset_s": None, "last_answer": None,
                    "error": reason or "this box has no Circle, or no account for the teacher's pod in its .env"}
        return await power.window(after)

    return router
