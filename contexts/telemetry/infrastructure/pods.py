"""Reading the pod power samplers from the Jetson, through the hub proxy, as telemetry events.

    reader = PodReader(pod_endpoints(os.environ),        # pod-a and pod-b, from the AUP_HUB_* keys of their accounts
                       user_agent=hub_user_agent(os.environ))
    async for event in reader.samples(): ...             # telemetry.gpu.power, received_ts on this machine's clock

Each pod runs lab/pod_power.py, which keeps its samples and serves them at GET /samples?after=<seq> on a local port
that the hub proxy exposes as https://<hub>/user/<account>/proxy/<port>/ (with the account's hub token).
Each pod is polled by its own task, every `poll_s`, for what came after its cursor, so a proxy gap delays samples
but loses none the pod still holds, and a sampler restart (a new sampler_started, its seq counting from 1 again) is
followed from its first sample. Each sample is parsed against the contract and must be a telemetry.gpu.power for
the pod it came from; it then gets `received_ts`, the moment this reader got it on the Jetson's clock, so `ts`
(the pod's clock) and `received_ts` together show how late it arrived.

The cursor (`after`, and `newest`, the newest sample's ts) moves one sample at a time, and only once samples() has
handed that sample on and been asked for the next one, that is once the runner has published it. A pod is polled
again only after its previous page was taken, so its next after= is that cursor. When a publish fails, the runner
abandons samples() at that sample and calls it again: the new call stops the old polls and reads every pod again
from its cursor, so the failed sample and the rest of its page come back (should the failed write have landed
after all, the bus drops the repeat by its event_id).

A sample at or before the newest one already held for that node (`skip_until`, the projection the log rebuilt) is
not published again, so restarting the app does not copy the pod's buffer into the log a second time. Samples the
pod dropped before they were fetched (its buffer is bounded) are counted as lost, once.
A sample that is not this pod's telemetry.gpu.power (another node's name, a broken event) is counted as rejected
and logged once per page and reason, with the pod and the count, so a pod whose every sample is rejected shows in
the log and not only in the counters.

A pod that does not answer, or whose proxy holds the request for up to `timeout_s`, delays only its own samples:
the other pod's are read and published meanwhile. A pod that does not answer is logged once per change of reason
and asked again at its next poll. The token is sent only in the Authorization header and appears in no log line or
status.

The pod's clock. Every answer carries pod_received and pod_sent (lab/pod_power.py), so each poll is one
four-timestamp exchange: t1 and t4 on this reader's clock, around the request, and the pod's two. The reader keeps
the last OFFSET_KEEP exchanges per pod and, on the first poll and every `offset_every` polls after, hands on a
telemetry.clock.offset from the one with the smallest round trip (infrastructure/clock.py) ahead of that poll's
samples. A sampler started under the pod's old name (--node gpu-a) is read as pod-a.

The hub is spoken to by the rules of lab/hub.py, the lab's one hub client, whose module-level pieces are reused here
rather than copied (its Hub class is a blocking httpx.Client with blocking retries, so the reader keeps its own async
client): every request carries the browser-like lab.hub.USER_AGENT, or $LAB_HUB_USER_AGENT (lab.hub.ENV_USER_AGENT),
because the hub's front end answers 403 to a script's own agent; the account keys are lab.hub.key_names; and the hub
of an account is its own AUP_HUB_URL_<X> (lab.hub.url_key), else $LAB_HUB_URL (lab.hub.ENV_URL, the one setting the
lab and jetson/egress.py read too), else lab.hub.DEFAULT_URL.
"""

import asyncio
import logging
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from contexts.telemetry.domain.nodes import POD_A, POD_B, canonical, check_node
from contexts.telemetry.infrastructure.clock import Exchange, best_offset, exchange_of
from contracts import parse_event
from contracts.events import Event
from contracts.telemetry import ClockOffset, GpuPower
from lab import hub as lab_hub

log = logging.getLogger("atlas.telemetry.pods")

HUB_URL = lab_hub.DEFAULT_URL  # the DUB hub, unless $LAB_HUB_URL (lab_hub.ENV_URL) or AUP_HUB_URL_<X> moves it
DEFAULT_PORT = 8790  # lab/pod_power.py DEFAULT_PORT
POLL_S = 2.0
LIMIT = 1000
OFFSET_EVERY = 30  # polls between two telemetry.clock.offset events of one pod: about a minute at the 2 s poll
OFFSET_KEEP = 10  # exchanges an offset is chosen from, by the smallest round trip
ACCOUNTS = "pod-a=A,pod-b=B"  # which hub account runs which pod, unless ATLAS_POD_ACCOUNTS says otherwise


@dataclass(frozen=True)
class PodEndpoint:
    node: str
    base_url: str  # https://<hub>/user/<account>/proxy/<port>, no trailing slash
    token: str = field(repr=False)


def pod_accounts(text: str) -> list[tuple[str, str]]:
    """(node, account letter) pairs from text such as "pod-a=C,pod-b=D"."""
    pairs = []
    for part in text.split(","):
        node, sep, letter = part.strip().partition("=")
        node, letter = canonical(node.strip()), letter.strip().upper()  # the old names gpu-a, gpu-b still read
        if not sep or node not in (POD_A, POD_B) or len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"ATLAS_POD_ACCOUNTS takes pairs such as pod-a=A,pod-b=B; got {part.strip()!r}")
        pairs.append((node, letter))
    if len({node for node, _ in pairs}) != len(pairs) or len({letter for _, letter in pairs}) != len(pairs):
        raise ValueError("ATLAS_POD_ACCOUNTS names a pod or an account twice")
    return pairs


def pod_endpoints(environ: Mapping[str, str], *, hub_url: str | None = None, port: int | None = None) -> list[PodEndpoint]:
    """The pods this machine can read: each account with both AUP_HUB_USER_<X> and AUP_HUB_TOKEN_<X> set.

    ATLAS_POD_ACCOUNTS says which account runs which pod (default pod-a=A,pod-b=B). Each account's hub is
    AUP_HUB_URL_<X> when set, else $LAB_HUB_URL (the setting lab/hub.py reads), else the DUB hub;
    ATLAS_POD_POWER_PORT moves the sampler's port.
    A half-set pair is an error naming the missing key (never its value); no pair at all is an error too."""
    default_hub = hub_url or (environ.get(lab_hub.ENV_URL) or "").strip() or HUB_URL
    port = port or int(environ.get("ATLAS_POD_POWER_PORT") or DEFAULT_PORT)
    accounts = pod_accounts(environ.get("ATLAS_POD_ACCOUNTS") or ACCOUNTS)
    pods, missing = [], []
    for node, letter in accounts:
        user_key, token_key = lab_hub.key_names(letter)
        user, token = (environ.get(user_key) or "").strip(), (environ.get(token_key) or "").strip()
        hub = ((environ.get(lab_hub.url_key(letter)) or "").strip() or default_hub).rstrip("/")
        if user and token:
            pods.append(PodEndpoint(node, f"{hub}/user/{quote(user, safe='')}/proxy/{port}", token))
        elif user or token:
            missing.append(token_key if user else user_key)
    if missing:
        raise ValueError(f"set {' and '.join(missing)} too, or neither key of that account")
    if not pods:
        names = " or ".join(" and ".join(lab_hub.key_names(letter)) for _, letter in accounts)
        raise ValueError(f"the pods source needs {names} in the environment")
    return pods


def _first_error(exc: Exception) -> str:
    """The first line of a parse error, cut short: enough to name the field, never the whole sample."""
    lines = str(exc).strip().splitlines()
    return " ".join(lines[:2])[:200] if lines else type(exc).__name__


def hub_user_agent(environ: Mapping[str, str]) -> str:
    """The User-Agent every request to the hub carries: $LAB_HUB_USER_AGENT when set, else lab/hub.py's browser-like
    agent, by the same rule lab.hub.Hub follows (the hub's front end answers 403 to a script's own agent)."""
    return (environ.get(lab_hub.ENV_USER_AGENT) or "").strip() or lab_hub.USER_AGENT


@dataclass(frozen=True)
class PodPage:
    """One answer of one pod, not yet taken: the reader's cursor has not moved for any of its samples."""
    first_seq: int | None  # the oldest sample the pod still held when it answered
    received: datetime  # when the answer reached this reader, on the Jetson's clock
    samples: list[Any]  # the raw {"seq", "event"} items, oldest first
    offset: ClockOffset | None = None  # the pod's clock against this reader's, when one is due with this page


@dataclass(frozen=True)
class _Handoff:
    """What a pod's poll task hands samples(): a page and the flag that says it was taken, or the poll's end."""
    pod: PodEndpoint
    page: PodPage | None = None
    taken: asyncio.Event | None = None
    error: Exception | None = None  # the poll task ended on this error rather than after its rounds


class PodReader:
    """A sampler (anything with samples()) for the SamplerRunner: the pods' telemetry.gpu.power samples."""

    def __init__(self, pods: Sequence[PodEndpoint], *, poll_s: float = POLL_S, limit: int = LIMIT,
                 transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 15.0,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 skip_until: Callable[[str], datetime | None] | None = None, rounds: int | None = None,
                 user_agent: str = lab_hub.USER_AGENT, offset_every: int = OFFSET_EVERY) -> None:
        if not pods:
            raise ValueError("a pod reader needs at least one pod")
        for pod in pods:
            if check_node(pod.node) not in (POD_A, POD_B):
                raise ValueError(f"{pod.node} is not a pod")
        if len({pod.node for pod in pods}) != len(pods):
            raise ValueError("each pod may appear once")
        if not user_agent.strip():
            raise ValueError("a pod reader needs a User-Agent: the hub's front end answers 403 to a request without "
                             "a browser-like one")
        self.pods = list(pods)
        self.poll_s = poll_s
        self.limit = limit
        self.transport = transport
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.skip_until = skip_until
        self.rounds = rounds  # stop after this many polls of every pod (tests); None polls until cancelled
        self.user_agent = user_agent  # sent on every request through the client's default headers, as lab/hub.py does
        self.state: dict[str, dict[str, Any]] = {
            pod.node: {"after": 0, "sampler_started": None, "newest": None, "seeded": False, "fetched": 0, "lost": 0,
                       "rejected": 0, "repeats": 0, "restarts": 0, "error": None, "last_ok": None} for pod in pods}
        self._polls: list[asyncio.Task[None]] = []  # the poll tasks of the samples() call now running
        self.offset_every = offset_every
        self._exchanges: dict[str, deque[Exchange]] = {pod.node: deque(maxlen=OFFSET_KEEP) for pod in pods}
        self._since_offset: dict[str, int | None] = {pod.node: None for pod in pods}  # None: none reported yet
        self.offsets: dict[str, ClockOffset] = {}  # the latest reported per pod

    async def samples(self) -> AsyncIterator[Event]:
        """Every pod's new samples as they arrive, each pod polled by its own task; see the module docstring.

        A sample's cursor is committed when the caller asks for the next event after it, never before, so a
        caller that fails on a sample and calls samples() again gets that sample and the rest of its page again."""
        # A call abandoned at a failed publish may not have been closed yet; its polls must not outlive it.
        await self._stop_polls(self._polls)
        pages: asyncio.Queue[_Handoff] = asyncio.Queue()
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s, follow_redirects=False,
                                     headers=self.hub_headers()) as client:
            polls = [asyncio.create_task(self._poll_pod(client, pod, pages), name=f"pod-reader-{pod.node}")
                     for pod in self.pods]
            self._polls = polls
            try:
                running = len(polls)
                while running:
                    handoff = await pages.get()
                    if handoff.page is None or handoff.taken is None:
                        running -= 1
                        if handoff.error is not None:
                            raise RuntimeError(f"polling the {handoff.pod.node} sampler failed") from handoff.error
                        continue
                    for event in self._take(handoff.pod, handoff.page):
                        yield event
                    handoff.taken.set()
            finally:
                await self._stop_polls(polls)
                if self._polls is polls:
                    self._polls = []

    async def _poll_pod(self, client: httpx.AsyncClient, pod: PodEndpoint, pages: asyncio.Queue[_Handoff]) -> None:
        """One pod's polls: a page at a time, the next poll only after samples() took the last page."""
        polls = 0
        try:
            while True:
                page = await self._poll(client, pod)
                polls += 1
                if page is not None:
                    taken = asyncio.Event()
                    pages.put_nowait(_Handoff(pod, page, taken))
                    await taken.wait()
                if self.rounds is not None and polls >= self.rounds:
                    break
                await self.sleep(self.poll_s)
        except Exception as exc:  # handed to samples(), which raises it for the runner to log and restart
            pages.put_nowait(_Handoff(pod, error=exc))
            return
        pages.put_nowait(_Handoff(pod))

    def hub_headers(self) -> dict[str, str]:
        """The headers every request to the hub carries, the token aside (that one goes per request, to its pod)."""
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    async def _stop_polls(self, polls: Sequence[asyncio.Task[None]]) -> None:
        for task in polls:
            task.cancel()
        results = await asyncio.gather(*polls, return_exceptions=True)
        for task, result in zip(polls, results):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                log.error("the %s poll ended on %s: %s", task.get_name(), type(result).__name__, result)

    async def fetch(self, client: httpx.AsyncClient, pod: PodEndpoint) -> list[GpuPower | ClockOffset]:
        """One poll of one pod, taken whole: the samples it has after the cursor, stamped with their receive time.

        The cursor moves past every one of them, as for a caller that keeps all it is handed; samples() takes a
        page one sample at a time instead."""
        page = await self._poll(client, pod)
        return [] if page is None else list(self._take(pod, page))

    async def _poll(self, client: httpx.AsyncClient, pod: PodEndpoint) -> PodPage | None:
        """Ask one pod for what came after its cursor; None when it gave no usable answer (logged by _trouble)."""
        state = self.state[pod.node]
        if not state["seeded"]:
            state["newest"] = self.skip_until(pod.node) if self.skip_until else None
            state["seeded"] = True
        body = await self._get(client, pod, state["after"])
        if body is None:
            return None
        started = body.get("sampler_started")
        if state["sampler_started"] is not None and started != state["sampler_started"]:
            state["restarts"] += 1  # a new sampler process numbers its samples from 1 again
            state["after"] = 0
            log.info("the %s sampler restarted; reading it from its first sample", pod.node)
            body = await self._get(client, pod, 0)
            if body is None:
                return None
            started = body.get("sampler_started")
        state["sampler_started"] = started
        received = self.clock()
        state["last_ok"] = received
        first_seq = body.get("first_seq")
        return PodPage(first_seq if isinstance(first_seq, int) and not isinstance(first_seq, bool) else None,
                       received, list(body.get("samples") or []), self._offset_due(pod))

    def _offset_due(self, pod: PodEndpoint) -> ClockOffset | None:
        """The pod's clock offset from its recent exchanges, when the first or the next one is due."""
        exchanges = self._exchanges[pod.node]
        since = self._since_offset[pod.node]
        if not exchanges or (since is not None and since + 1 < self.offset_every):
            if since is not None:
                self._since_offset[pod.node] = since + 1
            return None
        self._since_offset[pod.node] = 0
        return best_offset(pod.node, list(exchanges))

    def _take(self, pod: PodEndpoint, page: PodPage) -> Iterator[GpuPower | ClockOffset]:
        """The page's samples for this pod, committing the cursor for each only when the next one is asked for.

        Samples skipped here (rejected, or already in the log) move the cursor at once, since nothing is published
        for them. A caller that stops at a sample leaves the cursor before it, and its counts untouched."""
        state = self.state[pod.node]
        if state["after"] and page.first_seq is not None and page.first_seq > state["after"] + 1:
            dropped = page.first_seq - state["after"] - 1
            state["lost"] += dropped
            state["after"] = page.first_seq - 1  # they are gone; counted once even if this page is read again
            log.warning("%d %s samples were dropped on the pod before they could be read", dropped, pod.node)
        rejected: Counter[str] = Counter()  # reason -> samples of this page rejected for it, logged once per page
        if page.offset is not None and self.offsets.get(pod.node) is not page.offset:
            yield page.offset  # before the samples, so they can be placed on this reader's clock as they arrive
            self.offsets[pod.node] = page.offset
        try:
            for item in page.samples:
                seq = item.get("seq") if isinstance(item, Mapping) else None
                if not isinstance(seq, int) or isinstance(seq, bool):
                    state["rejected"] += 1
                    rejected["they carry no sequence number"] += 1
                    continue
                try:
                    event = parse_event(item.get("event"))
                except (ValueError, TypeError) as exc:
                    state["rejected"] += 1
                    rejected[f"they are not contract events ({_first_error(exc)})"] += 1
                    state["after"] = max(state["after"], seq)
                    continue
                if not isinstance(event, GpuPower) or canonical(event.node) != pod.node:
                    state["rejected"] += 1  # only this pod's power samples come through this pod's address
                    rejected[f"they are {event.type} for node {getattr(event, 'node', None)!r}, not {pod.node}'s "
                             "power samples"] += 1
                    state["after"] = max(state["after"], seq)
                    continue
                if state["newest"] is not None and event.ts <= state["newest"]:
                    state["repeats"] += 1  # already in the log
                    state["after"] = max(state["after"], seq)
                    continue
                # a sampler started under the pod's old name (--node gpu-a) is read under its name today
                yield event.model_copy(update={"received_ts": page.received, "node": pod.node})
                state["after"] = max(state["after"], seq)  # handed on: the runner published it
                state["newest"] = event.ts
                state["fetched"] += 1
        finally:  # also when the caller stops mid-page: what was rejected so far is said once
            for reason, count in rejected.items():
                log.warning("rejected %d %s sample%s: %s", count, pod.node, "" if count == 1 else "s", reason)

    async def _get(self, client: httpx.AsyncClient, pod: PodEndpoint, after: int) -> dict[str, Any] | None:
        state = self.state[pod.node]
        try:
            t1 = self.clock()
            reply = await client.get(f"{pod.base_url}/samples", params={"after": after, "limit": self.limit},
                                     headers={"Authorization": f"token {pod.token}"})
            t4 = self.clock()
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
        exchange = exchange_of(t1, body, t4)  # the pod's two timestamps in every answer: one clock exchange
        if exchange is not None:
            self._exchanges[pod.node].append(exchange)
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
        out = {node: {k: v for k, v in s.items() if k != "seeded"} for node, s in self.state.items()}
        for node, offset in self.offsets.items():
            out[node]["clock_offset"] = {"offset_s": offset.offset_s, "error_s": offset.error_s, "ts": offset.ts}
        return out
