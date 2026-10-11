"""Seed a data directory with the demo night, through the real app.

    seeded = await seed(Path("data"))          # or: python -m contexts.demo seed

Builds the app the gateway builds (fake runtime, connectors off, no memory fixture, no phone
pushes), publishes the story's source events through its bus so every handler runs and the log
is written, then runs the ticker's flushes once at the story's `now` so episodes and days close.

Time is the story's: the agent's clock follows the event being published, and nothing here reads
the wall clock. Handlers that run in the background are awaited after every event, so the log
comes out in the same order every run.
"""

import asyncio
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from contexts.demo import history as past_days
from contexts.demo.nearby import nearby_events
from contexts.demo.story import ROOT, load, story
from contexts.gateway.bootstrap import Atlas, Settings, build
from contexts.play.league.demo import friend_summaries
from contracts.events import Event

PACKAGE_RULE = "package-at-door"


class SeedRefused(RuntimeError):
    """The data directory already holds something; nothing was written."""


@dataclass
class Seeded:
    app: Atlas
    now: datetime
    source_events: int  # what the story published
    seconds: float
    written: Counter = field(default_factory=Counter)  # event type -> events that reached the log, derived ones included
    alerts: list[Event] = field(default_factory=list)
    nudges: list[Event] = field(default_factory=list)
    plans: list[Event] = field(default_factory=list)
    rounds: list[Event] = field(default_factory=list)  # Sort It rounds scored
    people: list[str] = field(default_factory=list)  # demo people written to the identity store now (contexts/demo/people.py)
    looks: int = 0  # ... and the owner's expression stretches handed to the bus

    @property
    def last_full_day(self):
        return self.app.habits.sched.local_day(self.now) - timedelta(days=1)


def settings(data_dir: Path, config_dir: Path | None = None) -> Settings:
    """The gateway's settings with everything that reaches outside switched off."""
    s = Settings(data_dir=data_dir, llm_url=None, ntfy_url=None, token=None, connectors=False, replay=False,
                 memory_fixture="")
    if config_dir is not None:  # the Circle demo's own copy (contexts/demo/circle.py)
        s.config_dir = Path(config_dir)
    elif not s.config_dir.exists():  # run from somewhere else than the repo root
        s.config_dir = ROOT / "config"
    return s


def build_app(data_dir: Path, config_dir: Path | None = None) -> Atlas:
    os.environ.pop("ATLAS_LLM_URL", None)  # runtime.registry reads it directly: the seed never calls a real model
    return build(settings(data_dir, config_dir))


async def settle() -> None:
    """Wait for the handlers the bus started in the background, and for whatever they published."""
    me = asyncio.current_task()
    while pending := [t for t in asyncio.all_tasks() if t is not me and not t.done()]:
        await asyncio.gather(*pending, return_exceptions=True)


async def expire(app: Atlas, now: datetime) -> None:
    """The ticker's approvals job, in the story's time: a card nobody answered expires five minutes after it was
    raised, as it does on a running box. Without it every card of the story stayed "waiting" in the log, the Day loop
    held back its later re-plans behind the first one, and the gateway that then started on the seeded directory
    expired them all at once and raised the held-back re-plan as a new card: one stray pending card on every fresh
    seed (it failed the pre-flight's "no stale approval cards")."""
    done = app.approvals.expire(now)
    if done:
        await app.bus.publish_all(done)
        await settle()


async def seed(data_dir: str | Path, now: datetime | None = None, force: bool = False, config_dir: Path | None = None,
               extra: Callable[[datetime], list[Event]] | None = None, history: bool = False, people: bool = False) -> Seeded:
    """Publish the night into `data_dir`. `now` moves the story so it ends then (default: the script's own time).
    `config_dir` builds the app on another config directory; `extra(now)` adds more source events to the story.
    `history` first lives the twelve days before the story through the Day loop (contexts/demo/history.py), so the
    date strip on Today's Plan has earlier days to open.
    `people` adds the demo people and a week of the owner's looks (contexts/demo/people.py: made-up vectors marked as
    demo data), after the story and its closing jobs, so nothing of the story changes.

    Refuses a directory that already has an event log or a memory file unless `force`; never deletes.
    With `force` the existing log is replayed first, so seeding the same story twice adds nothing."""
    data_dir = Path(data_dir)
    log_files = sorted((data_dir / "events").glob("*.jsonl"))
    memory_db = data_dir / "memory.sqlite"
    if (log_files or memory_db.exists()) and not force:
        found = f"{len(log_files)} event log file(s)" if log_files else memory_db.name
        raise SeedRefused(f"{data_dir} already has {found}. Seed an empty directory, or pass --force to add to this one.")

    started = time.perf_counter()
    now = now or load().now
    events = story(now)
    app = build_app(data_dir, config_dir)
    if log_files:
        await app.bus.replay_into()

    clock = {"now": events[0].ts}
    app.qa.clock = lambda: clock["now"]  # event time for the planner, not the wall clock
    out = Seeded(app=app, now=now, source_events=len(events), seconds=0.0)

    def note(e: Event) -> None:
        out.written[e.type] += 1
        for prefix, kept in (("agent.alert", out.alerts), ("habits.nudge", out.nudges), ("agent.plan", out.plans),
                             ("play.round.scored", out.rounds)):
            if e.type.startswith(prefix):
                kept.append(e)

    if app.league:  # the League's second member: a friend's shared summaries, demo fixture only (fixtures/demo/league.yaml)
        events = sorted(events + friend_summaries(now, circle_id=app.league.circle_id), key=lambda e: (e.ts, e.event_id))

    events = sorted(events + nearby_events(now), key=lambda e: e.ts)  # "Near you": optional events (fixtures/demo/nearby.yaml)

    if extra:
        events = sorted(events + extra(now), key=lambda e: e.ts)  # stable: the story keeps its own order

    app.bus.subscribe("", note)  # live only: sees exactly what is appended to the log
    if history:
        out.source_events += await past_days.publish(app, now, clock, expire, settle)
    for e in events:
        clock["now"] = e.ts
        await expire(app, e.ts)
        await app.bus.publish(e)
        await settle()

    # The ticker's event-time jobs, once, at the end of the story (contexts.gateway.bootstrap.Atlas._ticks).
    clock["now"] = now
    for job in (lambda: app.approvals.expire(now), lambda: app.zone_engine.flush(now), lambda: app.projection.flush(now),
                lambda: app.habits.flush(now) + app.nudger.check(now),
                lambda: app.day.tick(now) if app.day else []):  # the Day loop's status at the end of the story
        await app.bus.publish_all(job())
        await settle()
    if people:
        from contexts.demo import people as demo_people  # here, not at the top: it reads the check-in story, which imports this module

        out.people, out.looks = await demo_people.publish(app, now, story(now), settle)
        out.source_events += out.looks
    out.seconds = time.perf_counter() - started
    return out


def summary(s: Seeded) -> str:
    """What the seed printed: the log by type, then each projection as the app now sees it."""
    app, tz = s.app, s.app.habits.sched.tz
    lines = [f"Seeded {app.settings.data_dir}: {sum(s.written.values())} events in the log "
             f"({s.source_events} published, the rest derived by the app) in {s.seconds:.1f} s",
             f"Story ends at {s.now.astimezone(tz):%a %Y-%m-%d %H:%M %Z}", "", "Events written, by type:"]
    lines += [f"  {n:5d}  {t}" for t, n in sorted(s.written.items())]

    stats = app.memory.stats()
    keys = app.memory.last_seen("keys")
    lines += ["", f"Memory: {stats.episodes} episodes, {stats.captions} captions, {stats.hours_recorded} hours recorded"]
    if keys:
        lines.append(f"  keys last seen: {keys.zone_id or keys.camera_id}, {keys.position}, at {keys.ts.astimezone(tz):%a %H:%M}")

    for title, digest in ((f"Habits, last full day ({s.last_full_day})", app.habits.digest(day=s.last_full_day, now=s.now)),
                          (f"Habits, final day so far ({app.habits.sched.local_day(s.now)})", app.habits.digest(now=s.now))):
        lines += ["", title + ":"]
        for r in digest["habits"]:
            state = "met" if r["met"] else "not met"
            lines.append(f"  {r['name']:14s} {r['value']:>6} {r['unit']:6s} (target {r['direction'].replace('_', ' ')} "
                         f"{r['target']:g}, {state}{'' if r['applies'] else ', nothing to judge'})")

    lines += ["", "Waste Watch: " + ", ".join(f"{k} {v}" for k, v in app.waste.metrics().items())]
    if app.league:
        view = app.league.view(s.now)
        lines += ["", f"League, week of {view['week']['start']} ({'this box alone' if view['local'] else view['circle_id']}):"]
        for r in view["standings"]:
            score = "  n/a " if r["score"] is None else f"{r['score']:5.1f}%"
            lines.append(f"  {r['rank']}. {r['display_name']:10s} {score} {r['met']}/{r['judged']} target-days; " + "; ".join(r["reasons"]))
        lines += [f"  {view['summary']}", f"  Stake: {view['stake']['text_for_week']}", f"  Waste Watch {view['team_waste']['line']}"]
    lines.append(f"Sort It: {len(s.rounds)} round(s) scored" + "".join(
        f"; {r.item} in the {'right' if r.correct else 'wrong'} bin in {r.seconds:.0f} s, {r.score:g} points" for r in s.rounds))
    if s.people or s.looks or any(m.get("demo") for m in (app.identity.members() if app.identity else [])):
        rows = app.identity.members()
        lines += ["", f"People (DEMO DATA: made-up vectors, never a face; {s.looks} stretches of looks for the owner):"]
        lines += [f"  {m['name']:10s} " + ("recognised through the Circle" if m["source"] == "circle" else "scanned" if m["scanned"]
                                           else "photos only: scan to finish") + (", reads expressions" if m["expressions"] else "")
                  for m in rows if m.get("demo")]
        if not app.identity.available:
            lines.append("  This machine has no face model, so nobody will be recognised live.")
    for title, events, text in ((f"Alerts raised ({len(s.alerts)})", s.alerts, lambda e: f"[{e.rule_id}] {e.message}"),
                                (f"Nudges raised ({len(s.nudges)})", s.nudges, lambda e: f"[{e.habit_id}] {e.message}"),
                                (f"Plans created ({len(s.plans)})", s.plans, lambda e: f"[{e.subject}] {e.text}")):
        lines += ["", title + ":"]
        lines += [f"  {e.ts.astimezone(tz):%a %H:%M}  {text(e)}" for e in events]
    return "\n".join(lines)
