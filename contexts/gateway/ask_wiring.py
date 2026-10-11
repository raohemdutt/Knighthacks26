"""Ask Atlas's tools wired to the contexts that own the data (composition root, called from bootstrap.py).

Each port is a small read of another context's public object, or the existing approval path: nothing here writes
except the two executors, and they run only after the phone approves (Workspace, W2):

- connector_write (wrapped): after the calendar connector writes an event that Ask Atlas proposed, the event is
  also stored on the box as a calendar item (connector.item.synced), so "what's on my calendar tomorrow?" names it at
  once. With no calendar connector on the box, approving keeps it on the box's calendar only (the card says so).
- device_control "cameras.pause": the same privacy-policy change the Privacy screen makes (vault.policy.changed with
  every camera paused, or none on "resume").

The router itself is off for the FakeRuntime (tests and a laptop with no model) unless ATLAS_ASK_TOOLS=1, and off
everywhere with ATLAS_ASK_TOOLS=0.
"""

import inspect
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from contexts.agent.application import schedule
from contexts.agent.application.ask_router import AskRouter, NetworkBrain
from contexts.agent.application.product_tools import CAMERA_PAUSE, CALENDAR_WRITE, Ports
from contexts.agent.application.qa import QA, Evidence
from contexts.day.domain import ear
from contracts.connectors import ItemSynced
from contracts.vault import PolicyChanged
from runtime import FakeRuntime

log = logging.getLogger("atlas.ask")

JETSON = "jetson"


def _say_calendar(qa: QA, day: str) -> str | None:
    """The owner's wording of a day's calendar (QA's fast lane), or None when the words are not a day it knows."""
    question = f"what's on my calendar {day}"
    if schedule.parse(question, qa.clock(), qa.sleep.tz) is None:
        return None
    ev = Evidence()
    qa.upcoming_items(question.lower(), ev)
    return ev.direct or None


def _last_seen(qa: QA, label: str) -> str | None:
    """Said like a person (ear.said_last_seen), from the same sighting QA's "where is" path reads."""
    label = label.strip().lower()
    head = label.split()[-1] if label.split() else label  # "water bottle": the detector's label is "bottle"
    s = None
    for name in dict.fromkeys((label, head)):
        s = qa.memory.last_seen(name) or (qa.memory.last_seen(name[:-1]) if name.endswith("s") else None)
        if s:
            break
    if not s:
        return None
    tz = qa.sleep.tz
    return ear.said_last_seen(label, s.ts.astimezone(tz), qa.clock().astimezone(tz), s.zone_id, s.camera_id, s.position)


def _search(qa: QA, q: str) -> list[str]:
    tz, now = qa.sleep.tz, qa.clock()
    hits = sorted(qa.memory.search(q), key=lambda h: h.ts, reverse=True)
    out = []
    for h in hits:
        text = (h.text or "").strip().rstrip(".")
        if text:
            out.append(f"{ear.ear_about(h.ts.astimezone(tz), now.astimezone(tz)).capitalize()}: {text[0].lower() + text[1:]}.")
    return out


def _recap(qa: QA, start: datetime, end: datetime) -> str | None:
    """What the cameras saw in a window, in two short sentences: how many moments, and the busiest hour's caption."""
    hours = [b for b in qa.memory.window(start, end).hours if b.captions]
    if not hours:
        return None
    busiest = max(hours, key=lambda b: len(b.captions))
    when = busiest.hour.astimezone(qa.sleep.tz).strftime("%I %p").lstrip("0")
    moments = sum(len(b.captions) for b in hours)
    caption = busiest.captions[0].rstrip(".")
    return f"The cameras saw {moments} moment{'s' if moments != 1 else ''} in {len(hours)} hour{'s' if len(hours) != 1 else ''}. " \
           f"The busiest was around {when}: {caption[0].lower() + caption[1:] if caption else 'activity at the desk'}."


def _energy(qa: QA, telemetry) -> dict | None:
    """The previous answer's energy on the box: the Jetson's module power over that answer's window, integrated."""
    window = qa.last_window
    if window is None or telemetry is None:
        return None
    start, end = window
    readings = [r for r in telemetry.projection.readings(JETSON, start - timedelta(seconds=1), end + timedelta(seconds=1))
                if r.watts is not None]
    if not readings:
        return None
    seconds = max((end - start).total_seconds(), 0.001)
    watts = sum(r.watts for r in readings) / len(readings)
    return {"joules": round(watts * seconds, 1), "seconds": round(seconds, 2), "watts": round(watts, 1),
            "readings": len(readings), "node": JETSON, "started": start.isoformat(), "ended": end.isoformat()}


def _room(room, config_dir: Path) -> dict | None:
    if room is None:
        return None
    from contexts.perception.domain.zones import load_zones

    plan = room.plan
    zones_file = config_dir / "zones.yaml"
    zones = load_zones(zones_file) if zones_file.is_file() else {}
    on_floor = plan.zones_on_floor(zones)
    names = list(dict.fromkeys(z for cam in on_floor.values() for z in cam))
    data = plan.as_json()
    return {"name": data.get("name"), "title": data.get("title"), "cameras": sorted(plan.calibration),
            "zones": names, "furniture": len(data.get("furniture") or [])}


def _people(room, identity, owner_ok: bool) -> list[dict]:
    tracks = []
    if room is not None:
        try:
            tracks = [t for t in room.state(datetime.now(UTC)).get("tracks") or [] if t.get("kind") == "person"]
        except Exception:
            tracks = []
    present = identity.present() if identity is not None else []
    names = {m["member_id"]: m.get("name") for m in identity.members()} if (identity is not None and owner_ok) else {}
    known = [{"name": names.get(p["member_id"]), "known": True} for p in present]
    unknown = max(0, len(tracks) - len(known))
    return known + [{"name": None, "known": False} for _ in range(unknown)]


def _cameras(capture, policy) -> dict:
    stats = capture.stats() if capture is not None else {}
    paused = sorted(policy.state.paused_cameras) if policy is not None else []
    cams = sorted(stats)
    return {"cameras": cams, "paused": bool(cams) and set(cams) <= set(paused), "paused_cameras": paused,
            "running": bool(stats) and any((v or {}).get("status") not in ("paused", "stopped", "error") for v in stats.values())}


def _box_model(qa: QA, data_dir: Path, llm_url: str | None) -> dict:
    info = qa.runtime.model_info()
    out: dict[str, Any] = {"version": info.version, "base": info.base, "source": "the runtime on this box"}
    try:
        from contexts.learning.lab_deploy import DeployRunner

        view = DeployRunner(data_dir / "models", data_dir / "lab_deploy", llm_url=llm_url).view()
        current = next((v for v in view.get("versions") or [] if v.get("current")), None)
        if current:
            out.update(deployed=current.get("version"), deployed_at=current.get("created"), task=current.get("task"))
    except Exception:
        log.debug("no deploy view for box.model", exc_info=True)
    return out


def _learning(lab_view) -> dict | None:
    if lab_view is None:
        return None
    try:
        from contexts.learning.lab_api import is_fixture_run
    except ImportError:
        is_fixture_run = None
    runs = lab_view.runs()
    if is_fixture_run is not None:
        runs = [r for r in runs if not _safe(is_fixture_run, r)] or runs
    if not runs:
        return None
    r = runs[0]
    scores = r.get("scores") or {}
    score = next(iter(scores.values()), None) if isinstance(scores, dict) and scores else None
    return {"run": r.get("run_id"), "task": r.get("task"), "status": r.get("run_status") or r.get("status"),
            "score": score, "scores": scores, "verdict": r.get("verdict"), "deployed": r.get("deployed")}


def _safe(fn, *a) -> bool:
    try:
        return bool(fn(*a))
    except Exception:
        return False


def _shared_brain(shared) -> dict | None:
    brain = getattr(shared, "brain", None)
    if brain is None:
        return None
    s = brain.status()
    return {"enabled": bool(s.get("opted_in")), "node": s.get("node"), "badge": s.get("badge"),
            "reason": getattr(shared, "reason", None)}


def _inspection(spot) -> dict | None:
    """Inspection's shift summary (contexts/spot, the words its first screen shows), cut down to what the tool says."""
    read = getattr(spot, "read", None)
    if read is None or not read.configured:
        return None
    from contexts.spot.application import plain

    view = plain.shift(read)
    return {"status": view["status"]["sentence"], "watch": view["status"]["watch"], "do_first_total": view["do_first_total"],
            "do_first": [{"headline": i["headline"], "sentence": i["sentence"]} for i in view["do_first"]]}


def enabled_for(runtime) -> bool:
    flag = os.environ.get("ATLAS_ASK_TOOLS", "").strip().lower()
    if flag in ("0", "off", "false", "no"):
        return False
    return not isinstance(runtime, FakeRuntime) or flag in ("1", "on", "true", "yes")


def wire_ask_tools(*, qa: QA, bus, approvals, hub, settings, room=None, identity=None, capture=None, policy=None,
                   telemetry=None, lab_view=None, shared_brain=None, spot=None) -> AskRouter | None:
    """Build the ports and the router, attach it to QA, and add the two executors. None when the router is off."""
    if not enabled_for(qa.runtime):
        return None
    owner_ok = bool(settings.token)  # every /api/ask holds the box's token: the owner's own request
    calendar_ok = hub is not None and hub.is_write(CALENDAR_WRITE)
    ports = Ports(
        tz=qa.sleep.tz,
        items=lambda start, end: qa.memory.items(start, end),
        say_calendar=lambda day: _say_calendar(qa, day),
        calendar_writes=lambda: calendar_ok,
        last_seen=lambda label: _last_seen(qa, label),
        search=lambda q: _search(qa, q),
        recap=lambda start, end: _recap(qa, start, end),
        room=(lambda: _room(room, settings.config_dir)) if room is not None else None,
        people=(lambda: _people(room, identity, owner_ok)) if (room is not None or identity is not None) else None,
        owner_signed_in=lambda: owner_ok,
        cameras=(lambda: _cameras(capture, policy)) if capture is not None else None,
        can_pause=lambda: capture is not None and policy is not None,
        box_model=lambda: _box_model(qa, settings.data_dir, settings.llm_url),
        energy=lambda: _energy(qa, telemetry),
        learning=(lambda: _learning(lab_view)) if lab_view is not None else None,
        shared_brain=(lambda: _shared_brain(shared_brain)) if shared_brain is not None else None,
        inspection=(lambda: _inspection(spot)) if spot is not None else None,
    )
    network = NetworkBrain.from_env()
    router = AskRouter(qa.runtime, ports, qa.clock, publish=bus.publish, network=network)
    qa.router = router
    log.info("Ask Atlas tools: %d tools, network brain %s", len(router.registry.tools), "on" if network else "off (no teacher pod keys)")

    # --- the calendar: an approved event Ask Atlas proposed lands on the box's calendar too ---
    proposed: set[str] = set()
    publish_proposal = router.publish

    async def remember_and_publish(event) -> None:
        proposed.add(event.action_id)
        await publish_proposal(event)

    router.publish = remember_and_publish
    write = approvals.executors.get("connector_write")

    async def connector_write(req) -> str:
        if req.id not in proposed or req.target != CALENDAR_WRITE:
            if write is None:
                raise RuntimeError("no executor for connector_write")
            out = write(req)
            return await out if inspect.isawaitable(out) else out
        result = "kept on the box's calendar (no calendar connector)"
        if write is not None and calendar_ok:
            out = write(req)
            result = await out if inspect.isawaitable(out) else out
        await bus.publish_all([ItemSynced(connector="calendar", kind="calendar_event",
                                          external_id=f"atlas:{req.id}:{i}", title=e["title"],
                                          when=datetime.fromisoformat(e["start"]),
                                          summary=f"until {datetime.fromisoformat(e['end']).astimezone(qa.sleep.tz):%I:%M %p}".replace(" 0", " "))
                               for i, e in enumerate(req.params.get("events") or [])])
        return result

    approvals.executors["connector_write"] = connector_write

    # --- the cameras: pause or resume through the privacy policy, after an approval ---
    if capture is not None and policy is not None:
        async def device_control(req) -> str:
            if req.target != CAMERA_PAUSE:
                raise ValueError(f"{req.target} is not a device control")
            now = policy.state
            cams = [] if req.params.get("resume") else sorted(set(capture.stats()) | set(now.paused_cameras))
            await bus.publish(PolicyChanged(blur_faces=now.blur_faces, paused_cameras=cams, offline=now.offline,
                                            live_faces=now.live_faces, train_faces=now.train_faces))
            return "cameras resumed" if not cams else f"paused {len(cams)} camera(s)"

        approvals.executors["device_control"] = device_control
    return router

