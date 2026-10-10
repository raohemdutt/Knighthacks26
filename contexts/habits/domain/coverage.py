"""When the desk camera was actually reporting: what a night or a day can be judged on.

    cover = Coverage("cam-desk", gap=timedelta(minutes=5), steady=timedelta(minutes=30))
    cover.beat(e.ts)                      # every perception.observation.recorded of that camera
    cover.policy(paused, e.ts)            # every vault.policy.changed
    cover.seen(start, end, known)         # the parts of [start, end] the camera was reporting

Time the camera was not reporting is neither sleep nor not-sleep. Three things say it was not:

- the camera was paused in the Vault policy (`vault.policy.changed`);
- a steady camera went silent for longer than `gap`: a camera that has once reported without a break for
  `steady` is one that reports every second, so from that run on its silences are gaps. This is still open
  when the last report is more than `gap` before the latest time the tracker knows (`known`);
- for a steady camera, anything before its first report.

A camera that has never reported steadily is not measured: a scripted log (the demo night, the tests) is a
few bursts of frames with nothing between them, and that silence is an idle room, not a camera that was off.
Until one of the above is known, `measured` is False and everything counts as seen, as before.

Event time only; the same log gives the same answer.
"""

from datetime import datetime, timedelta

Span = tuple[datetime, datetime]


class Coverage:
    def __init__(self, camera: str, gap: timedelta, steady: timedelta) -> None:
        self.camera, self.gap, self.steady = camera, gap, steady
        self.first: datetime | None = None  # the camera's first report
        self.last: datetime | None = None  # its latest
        self.run_start: datetime | None = None  # start of the unbroken run `last` belongs to
        self.steady_since: datetime | None = None  # start of the first run that lasted `steady`
        self.gaps: list[Span] = []  # silences of a steady camera
        self.pauses: list[Span] = []  # paused in the Vault policy
        self.paused_since: datetime | None = None

    def beat(self, ts: datetime) -> None:
        """The camera reported at `ts`."""
        if self.last is None:
            self.first = self.run_start = self.last = ts
        elif ts > self.last:
            if ts - self.last > self.gap:
                if self.steady_since is not None:
                    self.gaps.append((self.last, ts))
                self.run_start = ts
            self.last = ts
        if self.steady_since is None and self.last - self.run_start >= self.steady:
            self.steady_since = self.run_start

    def policy(self, paused: bool, ts: datetime) -> None:
        if paused and self.paused_since is None:
            self.paused_since = ts
        elif not paused and self.paused_since is not None:
            self.pauses.append((self.paused_since, ts))
            self.paused_since = None

    @property
    def measured(self) -> bool:
        return self.steady_since is not None or self.paused_since is not None or bool(self.pauses)

    def _unseen(self, known: datetime) -> list[Span]:
        """Every stretch since its first report the camera was not reporting. Open ends run to `known`."""
        out = list(self.pauses)
        if self.paused_since is not None and self.paused_since < known:
            out.append((self.paused_since, known))
        if self.steady_since is not None:
            out += self.gaps
            if known - self.last > self.gap:
                out.append((self.last, known))
        return out

    def seen(self, start: datetime, end: datetime, known: datetime) -> list[Span]:
        """The parts of [start, end] the camera was reporting, oldest first."""
        if self.steady_since is not None:
            start = max(start, self.first)
        out = [(start, end)] if end > start else []
        for a, b in self._unseen(max(known, end)) if self.measured else []:
            out = [piece for x, y in out for piece in ((x, min(y, a)), (max(x, b), y)) if piece[1] > piece[0]]
        return sorted(out)

    def unbroken(self, since: datetime, known: datetime) -> bool:
        """True when the camera has been reporting all the way from `since` to `known`."""
        return known <= since or self.seen(since, known, known) == [(since, known)]

    def paused_in(self, start: datetime, end: datetime, known: datetime) -> timedelta:
        """How much of [start, end] the camera was paused for."""
        spans = self.pauses + ([(self.paused_since, max(known, end))] if self.paused_since is not None else [])
        return sum((min(b, end) - max(a, start) for a, b in spans if min(b, end) > max(a, start)), timedelta(0))

    def forget(self, before: datetime) -> None:
        """Drop what ended before `before`: no open day can ask about it."""
        self.gaps = [g for g in self.gaps if g[1] >= before]
        self.pauses = [p for p in self.pauses if p[1] >= before]
