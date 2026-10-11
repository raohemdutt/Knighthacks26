"""The hackathon in numbers, from the box's own event log (D11).

    python -m contexts.demo totals [--data-dir data] [--since ISO] [--json]

Reads data/events/*.jsonl directly (no app, no model) and prints one row per number with its
source: the event type it was counted from, how many events, and the first and last event time.
Nothing is made up: a number whose events are not in the log prints as "none in the log", and a
number that is estimated rather than counted (frames seen, joules) says so in its source.
For the video's closing card (D10) and the Devpost (D4).
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GAP_S = 600.0  # a quiet stretch longer than this is the box being off, not running
SAMPLE_GAP_S = 120.0  # a camera-health or power sample covers at most this long after it


@dataclass
class Total:
    name: str
    value: str
    source: str
    count: int = 0
    first: datetime | None = None
    last: datetime | None = None

    def span(self) -> str:
        if self.first is None:
            return ""
        return f"{_short(self.first)} to {_short(self.last)}"


@dataclass
class _Tally:
    n: int = 0
    first: datetime | None = None
    last: datetime | None = None

    def add(self, ts: datetime) -> None:
        self.n += 1
        self.first = ts if self.first is None or ts < self.first else self.first
        self.last = ts if self.last is None or ts > self.last else self.last


@dataclass
class _State:
    tallies: dict[str, _Tally] = field(default_factory=lambda: defaultdict(_Tally))
    times: list[datetime] = field(default_factory=list)
    episode_s: float = 0.0
    keyframes: int = 0
    captions: int = 0
    decisions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    completed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    frames: float = 0.0
    health_last: dict[str, tuple[datetime, float]] = field(default_factory=dict)
    jetson_j: float = 0.0
    jetson_last: tuple[datetime, float] | None = None
    gpu_j: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gpu_last: dict[str, tuple[datetime, float]] = field(default_factory=dict)


def read_events(events_dir: Path) -> Iterator[dict]:
    """Every parseable line of the log, in file order. A bad line is skipped, not fatal."""
    for path in sorted(events_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _short(ts: datetime | None) -> str:
    return ts.strftime("%a %H:%M UTC") if ts else ""


def _integrate(last: tuple[datetime, float] | None, ts: datetime, watts: float | None) -> tuple[float, tuple | None]:
    """Joules since the previous sample (its watts held until this one, capped at SAMPLE_GAP_S)."""
    joules = 0.0
    if last is not None:
        dt = (ts - last[0]).total_seconds()
        if 0 < dt <= SAMPLE_GAP_S:
            joules = last[1] * dt
    return joules, ((ts, watts) if watts is not None else None)


def _fold(state: _State, e: dict) -> None:
    kind, ts = e.get("type", ""), _ts(e["ts"])
    state.tallies[kind].add(ts)
    state.times.append(ts)
    if kind == "memory.episode.closed":
        state.episode_s += max(0.0, (_ts(e["end"]) - _ts(e["start"])).total_seconds())
        state.keyframes += len(e.get("keyframe_refs") or [])
        state.captions += len(e.get("captions") or [])
    elif kind == "experience.approval.decided":
        state.decisions[e.get("decision", "?")] += 1
    elif kind == "workspace.action.completed":
        state.completed[e.get("status") or "?"] += 1
    elif kind == "capture.camera.health":
        cam, fps = e.get("camera_id", "?"), float(e.get("fps_actual") or 0.0)
        prev = state.health_last.get(cam)
        if prev is not None:
            dt = (ts - prev[0]).total_seconds()
            if 0 < dt <= SAMPLE_GAP_S:
                state.frames += prev[1] * dt
        state.health_last[cam] = (ts, fps if e.get("status") in ("ok", "degraded") else 0.0)
    elif kind == "telemetry.jetson.power":
        mw = (e.get("rails") or {}).get("VDD_IN")
        j, state.jetson_last = _integrate(state.jetson_last, ts, mw / 1000 if mw is not None else None)
        state.jetson_j += j
    elif kind == "telemetry.gpu.power":
        node = e.get("node", "gpu")
        j, last = _integrate(state.gpu_last.get(node), ts, e.get("socket_w"))
        state.gpu_j[node] += j
        if last is None:
            state.gpu_last.pop(node, None)
        else:
            state.gpu_last[node] = last


def _running_hours(times: list[datetime]) -> float:
    times = sorted(times)
    return sum(d for a, b in zip(times, times[1:]) if (d := (b - a).total_seconds()) <= GAP_S) / 3600


def totals(events: Iterable[dict], since: datetime | None = None) -> list[Total]:
    state = _State()
    for e in events:
        try:
            if since is not None and _ts(e["ts"]) < since:
                continue
            _fold(state, e)
        except (KeyError, TypeError, ValueError):
            continue
    t = state.tallies

    def counted(name: str, kind: str, value: str | None = None, note: str = "") -> Total:
        tally = t.get(kind) or _Tally()
        shown = value if value is not None else (f"{tally.n:,}" if tally.n else "none in the log")
        return Total(name, shown, f"`{kind}` x {tally.n}" + (f"; {note}" if note else ""), tally.n, tally.first, tally.last)

    every = _Tally()
    for ts in state.times:
        every.add(ts)
    rows = [
        Total("Hours Atlas was running", f"{_running_hours(state.times):.1f} h" if state.times else "none in the log",
              f"all {len(state.times):,} logged events; gaps over {GAP_S / 60:.0f} min count as off", every.n,
              every.first, every.last),
    ]
    health = t.get("capture.camera.health") or _Tally()
    rows.append(Total("Frames seen by the cameras", f"~{state.frames:,.0f}" if health.n > 1 else "none in the log",
                      f"estimated: fps from `capture.camera.health` x {health.n} held until the next sample",
                      health.n, health.first, health.last))
    rows += [
        counted("Episodes remembered", "memory.episode.closed",
                note=f"{state.episode_s / 3600:.1f} h of activity" if state.episode_s else ""),
        counted("Keyframes kept (faces blurred)", "memory.episode.closed",
                f"{state.keyframes:,}" if state.keyframes else None, "keyframe_refs summed"),
        counted("Scene captions written on the box", "memory.episode.closed",
                f"{state.captions:,}" if state.captions else None, "captions summed"),
        counted("Objects and people observed", "perception.observation.recorded"),
        counted("Questions asked", "experience.question.asked"),
        counted("Answers given, with citations", "agent.answer.generated"),
        counted("Alerts raised", "agent.alert.raised"),
        counted("Actions proposed for approval", "agent.action.proposed"),
    ]
    decided = t.get("experience.approval.decided") or _Tally()
    rows.append(Total("Approval decisions (approve / deny)",
                      f"{state.decisions['approve']} / {state.decisions['deny']}" if decided.n else "none in the log",
                      f"`experience.approval.decided` x {decided.n}; includes tests and pre-flight resets",
                      decided.n, decided.first, decided.last))
    done = t.get("workspace.action.completed") or _Tally()
    others = ", ".join(f"{n} {s}" for s, n in sorted(state.completed.items()) if s != "done")
    rows.append(Total("Outside actions carried out, each after an approval",
                      f"{state.completed['done']:,}" if state.completed.get("done") else "none in the log",
                      f"`workspace.action.completed` with status done, of {done.n}" + (f" ({others})" if others else ""),
                      done.n, done.first, done.last))
    rows.append(counted("Connector items read", "connector.item.synced"))
    jetson = t.get("telemetry.jetson.power") or _Tally()
    rows.append(Total("Energy used by the Jetson", f"{state.jetson_j / 3600:,.1f} Wh" if state.jetson_j else "none in the log",
                      f"estimated: VDD_IN from `telemetry.jetson.power` x {jetson.n} integrated over time",
                      jetson.n, jetson.first, jetson.last))
    gpu = t.get("telemetry.gpu.power") or _Tally()
    gpu_wh = sum(state.gpu_j.values()) / 3600
    rows.append(Total("Energy used by the lab GPUs", f"{gpu_wh:,.1f} Wh" if gpu_wh else "none in the log",
                      f"estimated: socket_w from `telemetry.gpu.power` x {gpu.n} integrated over time"
                      + (f" ({', '.join(sorted(state.gpu_j))})" if state.gpu_j else ""),
                      gpu.n, gpu.first, gpu.last))
    return rows


def table(rows: list[Total]) -> str:
    out = ["| Number | Value | Source | Events from |", "| --- | --- | --- | --- |"]
    out += [f"| {r.name} | {r.value} | {r.source} | {r.span()} |" for r in rows]
    return "\n".join(out)


def as_json(rows: list[Total]) -> str:
    return json.dumps([{"name": r.name, "value": r.value, "source": r.source, "events": r.count,
                        "first": r.first.isoformat() if r.first else None,
                        "last": r.last.isoformat() if r.last else None} for r in rows], indent=2)
