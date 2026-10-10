"""Reading the pod power samplers from the Jetson, through the hub proxy, as telemetry events.

    reader = PodReader(pod_endpoints(os.environ))        # gpu-a and gpu-b, from the AUP_HUB_* keys of their accounts
    async for event in reader.samples(): ...             # telemetry.gpu.power, received_ts on this machine's clock

Each pod runs lab/pod_power.py, which keeps its samples and serves them at GET /samples?after=<seq> on a local port
that the hub proxy exposes as https://<hub>/user/<account>/proxy/<port>/ (with the account's hub token).
Every `poll_s` the reader asks each pod for what came after the last sample it has, so a proxy gap delays samples
but loses none the pod still holds, and a sampler restart (a new sampler_started, its seq counting from 1 again) is
followed from its first sample. Each sample is parsed against the contract and must be a telemetry.gpu.power for
the pod it came from; it then gets `received_ts`, the moment this reader got it on the Jetson's clock, so `ts`
(the pod's clock) and `received_ts` together show how late it arrived.

A sample at or before the newest one already held for that node (`skip_until`, the projection the log rebuilt) is
not published again, so restarting the app does not copy the pod's buffer into the log a second time. Samples the
pod dropped before they were fetched (its buffer is bounded) are counted as lost.

A pod that does not answer is logged once per change of reason and asked again at the next poll; the other pod is
not held up. The token is sent only in the Authorization header and appears in no log line or status.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from contexts.telemetry.domain.nodes import GPU_A, GPU_B, check_node
from contracts import parse_event
from contracts.events import Event
from contracts.telemetry import GpuPower

log = logging.getLogger("atlas.telemetry.pods")

HUB_URL = "https://dub.aupcloud.io"
DEFAULT_PORT = 8790  # lab/pod_power.py DEFAULT_PORT
POLL_S = 2.0
LIMIT = 1000
ACCOUNTS = "gpu-a=A,gpu-b=B"  # which hub account runs which pod, unless ATLAS_POD_ACCOUNTS says otherwise


@dataclass(frozen=True)
class PodEndpoint:
    node: str
    base_url: str  # https://<hub>/user/<account>/proxy/<port>, no trailing slash
    token: str = field(repr=False)


def pod_accounts(text: str) -> list[tuple[str, str]]:
    """(node, account letter) pairs from text such as "gpu-a=C,gpu-b=D"."""
    pairs = []
    for part in text.split(","):
        node, sep, letter = part.strip().partition("=")
        node, letter = node.strip(), letter.strip().upper()
        if not sep or node not in (GPU_A, GPU_B) or len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"ATLAS_POD_ACCOUNTS takes pairs such as gpu-a=A,gpu-b=B; got {part.strip()!r}")
        pairs.append((node, letter))
    if len({node for node, _ in pairs}) != len(pairs) or len({letter for _, letter in pairs}) != len(pairs):
        raise ValueError("ATLAS_POD_ACCOUNTS names a pod or an account twice")
    return pairs


def pod_endpoints(environ: Mapping[str, str], *, hub_url: str | None = None, port: int | None = None) -> list[PodEndpoint]:
    """The pods this machine can read: each account with both AUP_HUB_USER_<X> and AUP_HUB_TOKEN_<X> set.

    ATLAS_POD_ACCOUNTS says which account runs which pod (default gpu-a=A,gpu-b=B). Each account's hub is
    AUP_HUB_URL_<X> when set, else ATLAS_HUB_URL, else the DUB hub; ATLAS_POD_POWER_PORT moves the sampler's port.
    A half-set pair is an error naming the missing key (never its value); no pair at all is an error too."""
    default_hub = hub_url or environ.get("ATLAS_HUB_URL") or HUB_URL
    port = port or int(environ.get("ATLAS_POD_POWER_PORT") or DEFAULT_PORT)
    accounts = pod_accounts(environ.get("ATLAS_POD_ACCOUNTS") or ACCOUNTS)
    pods, missing = [], []
    for node, letter in accounts:
        user_key, token_key = f"AUP_HUB_USER_{letter}", f"AUP_HUB_TOKEN_{letter}"
        user, token = (environ.get(user_key) or "").strip(), (environ.get(token_key) or "").strip()
        hub = ((environ.get(f"AUP_HUB_URL_{letter}") or "").strip() or default_hub).rstrip("/")
        if user and token:
            pods.append(PodEndpoint(node, f"{hub}/user/{quote(user, safe='')}/proxy/{port}", token))
        elif user or token:
            missing.append(token_key if user else user_key)
    if missing:
        raise ValueError(f"set {' and '.join(missing)} too, or neither key of that account")
    if not pods:
        names = " or ".join(f"AUP_HUB_USER_{letter} and AUP_HUB_TOKEN_{letter}" for _, letter in accounts)
        raise ValueError(f"the pods source needs {names} in the environment")
    return pods


class PodReader:
    """A sampler (anything with samples()) for the SamplerRunner: the pods' telemetry.gpu.power samples."""

    def __init__(self, pods: Sequence[PodEndpoint], *, poll_s: float = POLL_S, limit: int = LIMIT,
                 transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 15.0,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 skip_until: Callable[[str], datetime | None] | None = None, rounds: int | None = None) -> None:
        if not pods:
            raise ValueError("a pod reader needs at least one pod")
        for pod in pods:
            if check_node(pod.node) not in (GPU_A, GPU_B):
                raise ValueError(f"{pod.node} is not a pod")
        if len({pod.node for pod in pods}) != len(pods):
            raise ValueError("each pod may appear once")
        self.pods = list(pods)
        self.poll_s = poll_s
        self.limit = limit
        self.transport = transport
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.skip_until = skip_until
        self.rounds = rounds  # stop after this many polls of every pod (tests); None polls until cancelled
        self.state: dict[str, dict[str, Any]] = {
            pod.node: {"after": 0, "sampler_started": None, "newest": None, "seeded": False, "fetched": 0, "lost": 0,
                       "rejected": 0, "repeats": 0, "restarts": 0, "error": None, "last_ok": None} for pod in pods}

    async def samples(self) -> AsyncIterator[Event]:
        done = 0
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s,
                                     follow_redirects=False) as client:
            while True:
                for pod in self.pods:
                    for event in await self.fetch(client, pod):
                        yield event
                done += 1
                if self.rounds is not None and done >= self.rounds:
                    return
                await self.sleep(self.poll_s)

    async def fetch(self, client: httpx.AsyncClient, pod: PodEndpoint) -> list[GpuPower]:
        """One poll of one pod: the samples it has after the last one fetched, stamped with their receive time."""
        state = self.state[pod.node]
        if not state["seeded"]:
            state["newest"] = self.skip_until(pod.node) if self.skip_until else None
            state["seeded"] = True
        body = await self._get(client, pod, state["after"])
        if body is None:
            return []
        started = body.get("sampler_started")
        if state["sampler_started"] is not None and started != state["sampler_started"]:
            state["restarts"] += 1  # a new sampler process numbers its samples from 1 again
            state["after"] = 0
            log.info("the %s sampler restarted; reading it from its first sample", pod.node)
            body = await self._get(client, pod, 0)
            if body is None:
                return []
            started = body.get("sampler_started")
        state["sampler_started"] = started
        received = self.clock()
        first_seq = body.get("first_seq")
        if state["after"] and isinstance(first_seq, int) and first_seq > state["after"] + 1:
            state["lost"] += first_seq - state["after"] - 1
            log.warning("%d %s samples were dropped on the pod before they could be read",
                        first_seq - state["after"] - 1, pod.node)
        out: list[GpuPower] = []
        for item in body.get("samples") or []:
            seq = item.get("seq") if isinstance(item, Mapping) else None
            if not isinstance(seq, int) or isinstance(seq, bool):
                state["rejected"] += 1
                continue
            state["after"] = max(state["after"], seq)
            try:
                event = parse_event(item.get("event"))
            except (ValueError, TypeError):
                state["rejected"] += 1
                continue
            if not isinstance(event, GpuPower) or event.node != pod.node:
                state["rejected"] += 1  # only this pod's power samples come through this pod's address
                continue
            if state["newest"] is not None and event.ts <= state["newest"]:
                state["repeats"] += 1  # already in the log
                continue
            state["newest"] = event.ts
            out.append(event.model_copy(update={"received_ts": received}))
        state["fetched"] += len(out)
        state["last_ok"] = received
        return out

    async def _get(self, client: httpx.AsyncClient, pod: PodEndpoint, after: int) -> dict[str, Any] | None:
        state = self.state[pod.node]
        try:
            reply = await client.get(f"{pod.base_url}/samples", params={"after": after, "limit": self.limit},
                                     headers={"Authorization": f"token {pod.token}", "Accept": "application/json"})
        except httpx.HTTPError as exc:
            return self._trouble(pod, f"no answer ({type(exc).__name__})")
        if reply.status_code != 200:
            return self._trouble(pod, f"HTTP {reply.status_code}")
        try:
            body = reply.json()
        except ValueError:
            return self._trouble(pod, "an answer that is not JSON")
        if not isinstance(body, dict) or not isinstance(body.get("samples"), list):
            return self._trouble(pod, "an answer without samples")
        if state["error"] is not None:
            log.info("the %s sampler answers again", pod.node)
            state["error"] = None
        return body

    def _trouble(self, pod: PodEndpoint, reason: str) -> None:
        state = self.state[pod.node]
        if state["error"] != reason:
            log.warning("the %s sampler gave %s; asking again every %.0f s", pod.node, reason, self.poll_s)
        state["error"] = reason
        return None

    def status(self) -> dict[str, Any]:
        return {node: {k: v for k, v in s.items() if k != "seeded"} for node, s in self.state.items()}
