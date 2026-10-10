"""The Day loop's API (D1) for the phone's Now screen.

    app.include_router(day_router(atlas.day, publish=atlas.bus.publish, guard=authorized, memory=atlas.memory))

- GET  /api/day/now            the live day: status, plan (and whether it is committed), a pending re-plan, the review,
                               friends' places, the cameras' last activity
- GET  /api/day/plan?day=      one day's plan, with `moves` in the shape Day Run's editor takes (POST /api/dayrun/simulate)
- POST /api/day/plan           {} make or remake today's plan now; {moves: [...]} replace it with the person's own edited
                               moves; {from_here: true} a re-plan from where the box thinks you are. Each publishes a new
                               plan event and one approval card; nothing is written until the card is approved.
- GET  /api/day/patterns       plain statements from the habit history ("on the N nights under 6 h, ...")
- GET  /api/day/coach?day=     the study coach: each block of the main activity against the desk camera (minutes at the
                               desk, breaks and their likely triggers with evidence), the day in words, one or two
                               suggestions, the running forecast calibration, and the notes sent when blocks ended.
                               Evidence is cited by id in the events; here each one also gets its words and, when
                               Memory has a picture of that moment, a `frame_ref` the phone can open.

Every answer is what the log says: the plan and status are projections of `day.*` events.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from contexts.day.application.loop import DayLoop
from contexts.day.domain.coach import FOOD_WORDS, PHONE_WORDS, mentions
from contexts.day.domain.patterns import patterns
from contexts.play.dayrun.models import Move
from contracts.day import DayPlanProposed, DayReplanProposed, MoveIn


class PlanBody(BaseModel):
    moves: list[MoveIn] | None = None  # the person's own moves (Day Run's shape); None: let Atlas choose
    from_here: bool = False  # a re-plan for the rest of the day from where the box thinks you are


def day_router(day: DayLoop, publish: Callable[[object], Awaitable] | None = None, guard: Callable | None = None,
               memory=None, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> APIRouter:
    router = APIRouter(prefix="/api/day", tags=["day"])
    writes = [Depends(guard)] if guard else []
    p, t = day.planner, day.tracker

    def event_now() -> datetime:
        """The latest moment the loop knows about: the wall clock, or the last status on a replayed log that ends earlier."""
        now = clock()
        latest = max((e.ts for e in (t.status, *p.current.values()) if e), default=None)
        return now if latest is None or now >= latest else latest

    def plan_for(which: date | None) -> tuple[date, DayPlanProposed | DayReplanProposed | None]:
        if which is None:
            which = day.clock.day_of(event_now())
            if which not in p.current and p.current:
                which = max(p.current)  # the last planned day, when the log ends before today
        return which, p.current.get(which)

    def cite(evidence: dict) -> dict:
        """An evidence row with what Memory knows about it: the message or caption, and a keyframe when there is one."""
        out = {**evidence, "title": None, "frame_ref": None}
        if memory is None:
            return out
        try:
            at = datetime.fromisoformat(evidence["at"])
            if evidence["kind"] == "chat":
                hit = next((i for i in memory.items(at - timedelta(minutes=1), at + timedelta(minutes=1), kind="message")
                            if f"{i.connector}:{i.external_id}" == evidence["id"]), None)
                out["title"] = hit.title if hit else None
            else:
                ep = next((e for e in memory.episodes(at - timedelta(hours=2), at + timedelta(seconds=1)) if e.id == evidence["id"]), None)
                if ep:
                    words = {"phone": PHONE_WORDS, "food": FOOD_WORDS, "door": ("door",)}[evidence["kind"]]
                    out["title"] = next((c for c in ep.captions if mentions(c, words)), None) or ep.summary or (ep.captions[-1] if ep.captions else None)
                    out["frame_ref"] = nearest(ep.keyframe_refs, at)
        except Exception:
            pass
        return out

    def nearest(refs: list[str], at: datetime) -> str | None:
        """The keyframe closest in time to `at` ("frames:<camera>:<ts_ms>"); the last one when the times cannot be read."""
        if not refs:
            return None
        try:
            return min(refs, key=lambda r: abs(int(r.rsplit(":", 1)[1]) / 1000 - at.timestamp()))
        except ValueError:
            return refs[-1]

    def cited(session: dict) -> dict:
        return {**session, "breaks": [{**b, "evidence": [cite(e) for e in b["evidence"]]} for b in session["breaks"]]}

    def noted(note) -> dict:
        body = note.model_dump(mode="json")
        return {**body, "session": cited(body["session"])}

    def coach_recent(now: datetime, limit: int = 4) -> list[dict]:
        """The blocks the coach spoke about in the last day, newest first."""
        return [noted(n) for n in day.coach.recent(now)[:limit]]

    def coach_latest(now: datetime) -> dict | None:
        recent = coach_recent(now, 1)
        return recent[0] if recent else None

    def last_seen() -> dict | None:
        if memory is None:
            return None
        try:
            now = event_now()
            eps = memory.episodes(now - timedelta(hours=36), now + timedelta(minutes=1))
        except Exception:
            return None
        if not eps:
            return None
        ep = eps[-1]
        return {"episode_id": ep.id, "camera_id": ep.camera_id, "start": ep.start, "end": ep.end, "labels": ep.labels,
                "caption": ep.summary or (ep.captions[-1] if ep.captions else None),
                "frame_ref": ep.keyframe_refs[-1] if ep.keyframe_refs else None}

    @router.get("/now")
    async def now() -> dict:
        which, plan = plan_for(None)
        review = p.reviews.get(which)
        friends = []
        if t.presence is not None:
            friends = [f.model_dump(mode="json") for f in t.presence.friends(event_now())]
        coach = coach_recent(event_now())  # the last block the coach spoke about, and the ones before it
        return {
            "day": which.isoformat(), "now": event_now().isoformat(),
            "plan": plan.model_dump(mode="json") if plan else None,
            "committed": bool(plan and plan.plan_id in p.committed),
            "status": t.status.model_dump(mode="json") if t.status and t.status.day == which else None,
            "replan": p.pending[which].model_dump(mode="json") if which in p.pending else None,
            "review": review.model_dump(mode="json") if review else None,
            "yesterday": p.reviews[which - timedelta(days=1)].model_dump(mode="json") if which - timedelta(days=1) in p.reviews else None,
            "friends": friends, "last_seen": last_seen(),
            "coach": coach[0] if coach else None, "coach_earlier": coach[1:],
        }

    @router.get("/coach")
    async def get_coach(day_: date | None = Query(None, alias="day")) -> dict:
        c = day.coach
        which, plan = plan_for(day_)
        now = event_now()
        body = c.day_review(which, now) if plan else {"sessions": [], "coach_text": "", "suggestions": [], "calibration": None, "block_minutes": None}
        sessions = [cited(s.model_dump(mode="json")) for s in body["sessions"]]
        notes = sorted(c.notes.get(which, {}).values(), key=lambda n: n.ts)
        return {
            "day": which.isoformat(), "now": now.isoformat(), "plan_id": plan.plan_id if plan else None,
            "words": {"activity": c.words.activity, "workplace": c.words.workplace},
            "sessions": sessions, "text": body["coach_text"], "suggestions": body["suggestions"],
            "calibration": body["calibration"].model_dump() if body["calibration"] else None, "block_minutes": body["block_minutes"],
            "desk_minutes_today": c.habits_desk_minutes(which, now),
            "notes": [noted(n) for n in notes], "latest": coach_latest(now), "limits": c.limits(),
        }

    @router.get("/plan")
    async def get_plan(day: date | None = None) -> dict:
        which, plan = plan_for(day)
        if plan is None:
            raise HTTPException(404, f"no plan for {which}")
        return {"day": which.isoformat(), "plan_id": plan.plan_id, "start": plan.start.isoformat(),
                "committed": plan.plan_id in p.committed, "moves": [m.model_dump(mode="json") for m in plan.dayrun_moves],
                "blocks": [m.model_dump(mode="json") for m in plan.moves], "outcome": plan.outcome.model_dump(),
                "plan": plan.model_dump(mode="json")}

    @router.post("/plan", dependencies=writes)
    async def make_plan(body: PlanBody | None = None) -> dict:
        body = body or PlanBody()
        now = clock()
        if body.from_here:
            seen = t.where(now)
            where = f"At {t.walks.name(seen[0])}" if seen else "From here"
            events = p.replan(now, seen[0] if seen else None, f"{where}, you asked for a new plan for the rest of the day.", asked=True)
            if not events:
                raise HTTPException(409, "there is no plan for today yet to re-plan from; make one first")
        elif body.moves is not None:
            events = p.edited(now, [Move(**m.model_dump()) for m in body.moves])
        else:
            events = p.plan(now, remake=day.clock.day_of(now) in p.plans)
        if publish:
            for e in events:
                await publish(e)
        head = events[0]
        return {"plan_id": head.plan_id, "type": head.type, "action_id": head.action_id, "plan": head.model_dump(mode="json")}

    @router.get("/patterns")
    async def get_patterns() -> dict:
        try:
            return {"patterns": patterns(t.habits)}
        except Exception as exc:
            return {"patterns": [], "note": f"habit history unavailable: {exc}"}

    return router
