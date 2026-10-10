"""In-process event bus + durable JSONL event log (F3).

    bus = EventBus()
    bus.subscribe("perception.zone", rules.on_zone_event)   # prefix match on event type
    await bus.publish(event)

- Every published event is appended to data/events/YYYY-MM-DD.jsonl (UTC day of
  `event.ts`) before any handler sees it. The log is the source of truth;
  projections are rebuilt with `await bus.replay_into()`.
- Events are deduped on `event_id`, so a retry or a replay never runs twice.
- Handlers may be sync or async and may return events (or a list of them),
  which are published in turn. A failing handler is logged and skipped; it
  never takes the bus down.
"""

import inspect
import logging
from datetime import UTC
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Iterator
from pathlib import Path

from contracts import Event, parse_event

log = logging.getLogger("atlas.bus")

Handler = Callable[[Event], "Event | Iterable[Event] | None | Awaitable[Event | Iterable[Event] | None]"]

# High-rate, in-process signals that are not worth keeping in the log.
NOT_LOGGED = frozenset({"capture.frames.captured"})
DEDUPE_WINDOW = 50_000


class EventBus:
    def __init__(self, log_dir: str | Path = "data/events", persist: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.persist = persist
        self._subs: list[tuple[str, Handler, bool]] = []
        self._seen: OrderedDict[str, None] = OrderedDict()

    def subscribe(self, prefix: str, handler: Handler, replay: bool = False) -> None:
        """`prefix` matches the start of the event type: "perception.zone", "experience.", or "" for all.

        Only projections (tables rebuilt from the log) should pass `replay=True`. Handlers with
        side effects (approval executors, connector writes, model calls) must not run on replay."""
        self._subs.append((prefix, handler, replay))

    async def publish(self, event: Event) -> None:
        await self._publish(event, log_it=True)

    async def publish_all(self, events: Iterable[Event]) -> None:
        for e in events:
            await self.publish(e)

    async def _publish(self, event: Event, log_it: bool) -> None:
        if event.event_id in self._seen:
            return
        self._seen[event.event_id] = None
        if len(self._seen) > DEDUPE_WINDOW:
            self._seen.popitem(last=False)
        if log_it and self.persist and event.type not in NOT_LOGGED:
            self._append(event)
        follow_ups: list[Event] = []
        for prefix, handler, on_replay in list(self._subs):
            if not event.type.startswith(prefix) or (not log_it and not on_replay):
                continue
            try:
                out = handler(event)
                if inspect.isawaitable(out):
                    out = await out
            except Exception:
                log.exception("handler %s failed on %s", getattr(handler, "__qualname__", handler), event.type)
                continue
            follow_ups += _as_events(out)
        if not log_it:
            return  # replay: follow-ups were logged the first time and replay from the log on their own
        # every subscriber sees this event before anyone sees what it caused
        for follow in follow_ups:
            await self._publish(follow, log_it=True)

    def _append(self, event: Event) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self._path_for(event).open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def _path_for(self, event: Event) -> Path:
        day = event.ts.astimezone(UTC).date().isoformat() if event.ts.tzinfo else event.ts.date().isoformat()
        return self.log_dir / f"{day}.jsonl"

    # --- replay ---

    def read_log(self, days: Iterable[str] | None = None) -> Iterator[Event]:
        """Events from the log in file order. `days` like ["2026-10-10"]; all days by default."""
        paths = sorted(self.log_dir.glob("*.jsonl")) if days is None else [self.log_dir / f"{d}.jsonl" for d in days]
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield parse_event(line)
                    except Exception:
                        log.warning("skipping bad line %s:%d", path.name, n)

    async def replay_into(self, days: Iterable[str] | None = None) -> int:
        """Feed the log through `replay=True` subscribers without writing it again. Returns events replayed.

        Follow-up events that handlers return during replay are dropped: the original run
        already logged them, and they are replayed from the log in their own turn."""
        count = 0
        for event in self.read_log(days):
            await self._publish(event, log_it=False)
            count += 1
        return count


def _as_events(out) -> list[Event]:
    if out is None:
        return []
    if isinstance(out, Event):
        return [out]
    return [e for e in out if isinstance(e, Event)]
