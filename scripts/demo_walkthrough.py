#!/usr/bin/env python3
"""Walk every demo use case against a running box and say how far each one works.

    python scripts/demo_walkthrough.py --gateway http://127.0.0.1:8000
        [--token $ATLAS_TOKEN]           the box's token, if it has one (never printed)
        [--proxy socks5h://127.0.0.1:1055]   reach the box through a SOCKS proxy (uses curl)
        [--read-only]                    only read: GETs and the Day Run simulations, which store nothing
        [--ask]                          with --read-only: also ask questions and request a day brief (normal use)
        [--scene]                        first push the fake camera scene, in real time (about a minute)
        [--approve-local]                approve the cards that stay on the box (keep a plan, a local work order)
        [--only day,spot]                run only these use cases (see USE_CASES)
        [--expect-profile inspection]    fail unless the box runs this profile
        [--json results.json]            also write the rows as JSON

It is the executable half of docs/demo-run-sheet.md: the same use cases in the same order. Every step that
needs no person is called on the real endpoints, timed, and printed as

    use case | step | PASS/FAIL/SKIP | seconds | note

A step that needs a person or hardware is printed as SKIP with what to do by hand, never left out.
The exit code is 1 when any step FAILs.

What it will never do: approve a card with an effect outside the box. `local_only_card` decides, from the
card's kind, target and parameters, and anything it does not recognise counts as outside. Cards this script
raised itself and may not approve are denied again, so the box is left without stale cards. With --read-only
nothing is raised, approved, denied, pushed or changed.

It does not repeat the pre-flight (`python -m contexts.demo preflight --gateway ...`, contexts/gateway/preflight.py):
it reads GET /api/preflight and reports the count, and the run sheet says when to run the real one. The shapes
of the answers are checked in depth by `node web/scripts/check-api.mjs`; here only what a presenter would see.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_EVIDENCE = "I didn't see that."  # contexts/agent/application/qa.py
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# (key, title) in demo order: the same order as docs/demo-run-sheet.md
USE_CASES = [
    ("ask", "1 Ask what the cameras saw"),
    ("talk", "2 Conversation"),
    ("day", "3 The day"),
    ("coach", "4 Study coach"),
    ("dayrun", "5 Day Run"),
    ("circle", "6 Circle"),
    ("sortit", "7 Sort It"),
    ("impact", "8 Waste Watch / Impact"),
    ("identity", "9 Identity"),
    ("privacy", "10 Privacy"),
    ("teach", "11 Teach / learning"),
    ("voice", "12 Voice"),
    ("spot", "13 Inspection (SPOT)"),
    ("profiles", "14 Profiles"),
    ("operator", "15 Operator, zones, Live"),
]
TITLES = dict(USE_CASES)


# ---------- the pure parts (unit-tested in tests/test_demo_walkthrough.py) ----------

@dataclass
class Row:
    use_case: str
    step: str
    status: str
    seconds: float | None = None
    note: str = ""


def one_line(text: object, limit: int = 110) -> str:
    """Any value as one short line for the table: whitespace folded, pipes replaced, cut with an ellipsis."""
    s = " ".join(str(text).split()).replace("|", "/")
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def format_table(rows: list[Row]) -> str:
    """`use case | step | PASS/FAIL/SKIP | seconds | note`, columns padded to the widest cell."""
    head = ("use case", "step", "result", "seconds", "note")
    body = [(r.use_case, r.step, r.status, "" if r.seconds is None else f"{r.seconds:.2f}", r.note) for r in rows]
    widths = [max(len(head[i]), *(len(b[i]) for b in body)) if body else len(head[i]) for i in range(4)]
    def line(cells: tuple) -> str:
        return " | ".join([*(c.ljust(widths[i]) for i, c in enumerate(cells[:4])), cells[4]]).rstrip()
    return "\n".join([line(head), "-+-".join("-" * w for w in [*widths, 4]), *(line(b) for b in body)])


def markdown_table(rows: list[Row]) -> str:
    """The same rows as a Markdown table, for pasting into the run sheet."""
    out = ["| Use case | Step | Result | Seconds | Note |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {r.use_case} | {r.step} | {r.status} | {'' if r.seconds is None else f'{r.seconds:.2f}'} | {r.note} |" for r in rows]
    return "\n".join(out)


def counts(rows: list[Row]) -> dict[str, int]:
    return {s: sum(r.status == s for r in rows) for s in (PASS, FAIL, SKIP)}


def by_use_case(rows: list[Row]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(r.use_case, {PASS: 0, FAIL: 0, SKIP: 0})[r.status] += 1
    return out


def exit_code(rows: list[Row]) -> int:
    return 1 if any(r.status == FAIL for r in rows) else 0


def night_window(now: datetime) -> tuple[datetime, datetime]:
    """Last night as the agent's default sleep schedule has it: 8 PM to 8 AM, the night that ended today
    (or is still running, before 8 AM). In the timezone of `now`."""
    end = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return end - timedelta(hours=12), end


# Approving runs the card. Only two families stay on the box; both are `connector_write` because
# contracts.agent.ActionKind is a closed list, so the kind alone says nothing and the target is checked too.
NO_CALENDAR = "No calendar connector on this box"  # contexts/day/application/planner.py: DayPlanner.card
NOTHING_WRITTEN = "nothing is written"
OUTSIDE_KINDS = ("send_message", "circle_share", "export_training_data", "browse")


def local_only_card(card: object) -> tuple[bool, str]:
    """May this approval card be approved by a script? (yes/no, why).

    Yes only for a card whose approval provably stays on the box:
    - an inspection work order: kind connector_write, target spot.reinspect or spot.work_order, and
      params.stays_on_this_box is true (contexts/spot/application/workorders.py writes one local JSON file);
    - "keep this as today's plan" on a box with no calendar connector: kind connector_write, target
      calendar.create_event, and the reason carries the planner's own two phrases. With a calendar connected
      the same target writes to a real calendar and the phrases are absent, so it is refused.
    Everything else is treated as having an outside effect: messages (GroupMe, ntfy), Circle shares,
    training-data exports, browsing, any other connector write, and anything malformed or unknown."""
    if not isinstance(card, dict):
        return False, "not a card"
    kind, target, reason = card.get("kind"), card.get("target"), card.get("reason")
    params = card.get("params") if isinstance(card.get("params"), dict) else {}
    if not isinstance(kind, str) or not isinstance(target, str):
        return False, "card without a kind or target"
    if kind in OUTSIDE_KINDS:
        return False, f"{kind} leaves the box"
    if kind != "connector_write":
        return False, f"unknown kind {kind!r}: treated as an outside effect"
    if target in ("spot.reinspect", "spot.work_order"):
        if params.get("stays_on_this_box") is True:
            return True, f"{target}: a work-order file on this box"
        return False, f"{target} without stays_on_this_box"
    if target == "calendar.create_event":
        if isinstance(reason, str) and NO_CALENDAR in reason and NOTHING_WRITTEN in reason:
            return True, "keeps today's plan; no calendar connector, nothing is written"
        return False, "calendar.create_event with a calendar connected: writes to a real calendar"
    return False, f"connector_write to {target}: an outside write"


# ---------- transport ----------

@dataclass
class Resp:
    status: int  # 0 when the request itself failed
    data: object = None
    text: str = ""
    seconds: float = 0.0
    content_type: str = ""

    @property
    def detail(self) -> str:
        if isinstance(self.data, dict) and "detail" in self.data:
            return one_line(self.data["detail"], 140)
        return one_line(self.text, 140)

    @property
    def is_json(self) -> bool:
        return "json" in self.content_type and self.data is not None


class Client:
    """GET/POST/DELETE against one gateway. urllib by default; curl when a proxy is given (the standard library
    has no SOCKS, and no new dependency is added for it). The token goes in a header and is never printed."""

    def __init__(self, base: str, token: str | None = None, proxy: str | None = None) -> None:
        self.base, self.token, self.proxy = base.rstrip("/"), token or None, proxy or None

    def call(self, method: str, path: str, body: object = None, timeout: float = 30) -> Resp:
        return self._curl(method, path, body, timeout) if self.proxy else self._urllib(method, path, body, timeout)

    def get(self, path: str, timeout: float = 30) -> Resp:
        return self.call("GET", path, None, timeout)

    def post(self, path: str, body: object = None, timeout: float = 30) -> Resp:
        return self.call("POST", path, {} if body is None else body, timeout)

    def _headers(self, body: object) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if body is not None:
            h["Content-Type"] = "application/json"
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _urllib(self, method: str, path: str, body: object, timeout: float) -> Resp:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers=self._headers(body), method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through an environment proxy
        start = time.monotonic()
        try:
            with opener.open(req, timeout=timeout) as r:
                raw, status, ctype = r.read(), r.status, r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            raw, status, ctype = e.read(), e.code, e.headers.get("content-type", "")
        except Exception as e:  # refused, timed out, DNS
            return Resp(0, text=f"{type(e).__name__}: {e}", seconds=time.monotonic() - start)
        return self._resp(status, raw, ctype, time.monotonic() - start)

    def _curl(self, method: str, path: str, body: object, timeout: float) -> Resp:
        # headers and body go through stdin as a curl config, so the token never shows in the process list
        cfg = [f'header = "{k}: {v}"' for k, v in self._headers(body).items()]
        if body is not None:
            cfg.append("data-raw = " + json.dumps(json.dumps(body)))
        cmd = ["curl", "-sS", "-X", method, "--proxy", self.proxy, "--max-time", str(int(timeout)), "-K", "-",
               "-w", "\n%{http_code}\t%{content_type}", self.base + path]
        start = time.monotonic()
        try:
            p = subprocess.run(cmd, input="\n".join(cfg).encode(), capture_output=True, timeout=timeout + 10)
        except (OSError, subprocess.TimeoutExpired) as e:
            return Resp(0, text=f"{type(e).__name__}: curl did not finish", seconds=time.monotonic() - start)
        took = time.monotonic() - start
        if p.returncode != 0:
            return Resp(0, text="curl: " + one_line(p.stderr.decode("utf-8", "replace"), 160), seconds=took)
        raw, _, tail = p.stdout.rpartition(b"\n")
        code, _, ctype = tail.decode("ascii", "replace").partition("\t")
        return self._resp(int(code or 0), raw, ctype, took)

    @staticmethod
    def _resp(status: int, raw: bytes, ctype: str, seconds: float) -> Resp:
        text, data = raw.decode("utf-8", "replace"), None
        if "json" in ctype:
            try:
                data = json.loads(text)
            except ValueError:
                pass
        return Resp(status, data, text if data is None else "", seconds, ctype)


# ---------- the run ----------

@dataclass
class Run:
    api: Client
    read_only: bool = False
    ask: bool = True  # questions and the day brief; off under --read-only unless --ask
    approve_local: bool = False
    rows: list[Row] = field(default_factory=list)
    case: str = ""
    model: str = "?"
    raised: list[str] = field(default_factory=list)  # approval cards this run created

    @property
    def writes(self) -> bool:
        return not self.read_only

    @property
    def fake_model(self) -> bool:
        return self.model.startswith("fake")

    def add(self, step: str, status: str, seconds: float | None = None, note: object = "") -> None:
        row = Row(TITLES.get(self.case, self.case), step, status, None if seconds is None else round(seconds, 2), one_line(note))
        self.rows.append(row)
        print(format_table([row]).splitlines()[-1], flush=True)

    def manual(self, step: str, what: str) -> None:
        """A step only a person with hardware can do: listed, never silently left out."""
        self.add(step, SKIP, None, "needs a person: " + what)

    def fetch(self, step: str, path: str, need: tuple[str, ...] = (), missing: str | None = None, note=None,
              check=None, timeout: float = 30):
        """GET a JSON endpoint. PASS when it answers 200 with the `need` fields and `check(data)` returns no
        complaint; SKIP on 404 when `missing` says what is not set up on this box; FAIL otherwise."""
        r = self.api.get(path, timeout)
        if (r.status == 404 or (r.status == 200 and not r.is_json)) and missing is not None:
            self.add(step, SKIP, r.seconds, f"{missing} (HTTP {r.status})")
            return None
        if r.status != 200 or not r.is_json:
            self.add(step, FAIL, r.seconds, f"GET {path} -> HTTP {r.status}: {r.detail}")
            return None
        gone = [k for k in need if not (isinstance(r.data, dict) and k in r.data)]
        complaint = f"missing field(s): {', '.join(gone)}" if gone else (check(r.data) if check else None)
        if complaint:
            self.add(step, FAIL, r.seconds, complaint)
            return None
        self.add(step, PASS, r.seconds, note(r.data) if callable(note) else (note or ""))
        return r.data

    def screen(self, path: str) -> None:
        r = self.api.get(path)
        ok = r.status == 200 and "html" in r.content_type
        self.add(f"screen {path} is served", PASS if ok else FAIL, r.seconds, "" if ok else f"HTTP {r.status} {r.content_type}")

    def question(self, step: str, text: str, evidence: bool | None = None, want: str | None = None,
                 cited: bool = False, needs_model: bool = False, context: str = ""):
        """Ask through POST /api/ask, as the phone does. `evidence` True: the box has seen this, so the
        no-evidence line is a FAIL. `want`: a substring the answer must contain. `needs_model`: the fake
        runtime cannot answer this kind, so it is a SKIP there, not a FAIL."""
        if not self.ask:
            self.add(step, SKIP, None, "read-only without --ask: no question asked")
            return None
        if needs_model and self.fake_model:
            self.add(step, SKIP, None, f"needs the on-device model; this box runs {self.model}")
            return None
        r = self.api.post("/api/ask", {"text": text}, timeout=100)
        if r.status != 200 or not isinstance(r.data, dict) or not isinstance(r.data.get("answer"), str):
            self.add(step, FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")
            return None
        answer, cites = r.data["answer"].strip(), r.data.get("citations") or []
        kinds = ["frame" if c.get("frame_ref") else "link" if c.get("url") else "source" for c in cites]
        note = f"{len(cites)} citation(s){' (' + ', '.join(sorted(set(kinds))) + ')' if kinds else ''}: {answer}"
        problem = None
        if not answer:
            problem = "empty answer"
        elif evidence and answer == NO_EVIDENCE:
            problem = f'answered "{NO_EVIDENCE}" although the box has evidence'
        elif cited and not cites:
            problem = "no citation on an answer that should cite its evidence"
        elif want and want.lower() not in answer.lower():
            problem = f"expected {want!r} in the answer"
        if context:
            note, problem = f"[{context}] {note}", f"[{context}] {problem}" if problem else None
        self.add(step, FAIL if problem else PASS, r.seconds, f"{problem}: {answer}" if problem else note)
        return r.data

    # --- approval cards ---

    def card(self, action_id: str) -> dict | None:
        r = self.api.get("/api/approvals")
        return next((c for c in r.data if c.get("id") == action_id), None) if isinstance(r.data, list) else None

    def settle_card(self, step: str, action_id: str, expect_local: bool) -> bool:
        """The card `action_id` this run raised: check it is listed and is the kind expected, then approve it
        (only when it stays on the box and --approve-local was given) or deny it. Returns True when approved."""
        self.raised.append(action_id)
        card = self.card(action_id)
        if card is None:
            self.add(step, FAIL, None, f"the card {action_id[:8]} is not in GET /api/approvals")
            return False
        local, why = local_only_card(card)
        label = f"{card.get('kind')} -> {card.get('target')}"
        if local != expect_local:
            self.add(step, FAIL, None, f"{label}: the guard says {'local' if local else 'outside'} ({why}), expected the opposite; denied")
            self.api.post(f"/api/approvals/{action_id}", {"decision": "deny"})
            return False
        if local and self.approve_local:
            r = self.api.post(f"/api/approvals/{action_id}", {"decision": "approve"})
            ok = r.status == 200 and isinstance(r.data, dict) and r.data.get("status") in ("approved", "done")
            self.add(step, PASS if ok else FAIL, r.seconds, f"{label}: approved ({why})" if ok else f"approve -> HTTP {r.status}: {r.detail}")
            return ok
        r = self.api.post(f"/api/approvals/{action_id}", {"decision": "deny"})
        ok = r.status == 200
        reason = why if not local else "local, but --approve-local was not given"
        self.add(step, PASS if ok else FAIL, r.seconds, f"{label}: card shown, then denied ({reason})" if ok else f"deny -> HTTP {r.status}: {r.detail}")
        return False


def push_scene(run: Run, gateway: str, token: str | None) -> None:
    """`python -m contexts.demo fake perception --gateway ... --realtime`: the scripted camera scene, live."""
    run.case = "scene"
    TITLES["scene"] = "0 Fake camera scene"
    if run.read_only or run.api.proxy:
        run.add("push the fake camera scene", SKIP, None, "never pushed with --read-only or through a proxy (someone else's box)")
        return
    import os

    env = dict(os.environ)
    if token:
        env["ATLAS_TOKEN"] = token  # read by contexts.demo from the environment: not on the command line
    cmd = [sys.executable, "-m", "contexts.demo", "fake", "perception", "--gateway", gateway, "--realtime"]
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        run.add("push the fake camera scene", FAIL, time.monotonic() - start, "did not finish in 300 s")
        return
    out = (p.stdout.strip().splitlines() or p.stderr.strip().splitlines() or [""])[-1]
    run.add("push the fake camera scene (--realtime)", PASS if p.returncode == 0 else FAIL, time.monotonic() - start, out)


def uc_ask(run: Run) -> None:
    h = run.fetch("the box answers /health", "/health", ("ok", "model"), note=lambda d: f"model {d['model']}")
    if h:
        run.model = str(h["model"])
    run.screen("/")
    stats = run.fetch("memory has something to cite", "/memory/stats", ("episodes",),
                      note=lambda d: f"{d['episodes']} episodes, {d.get('captions')} captions, {d.get('hours_recorded')} h recorded")
    episodes = bool(stats and stats["episodes"])
    keys = run.api.get("/memory/objects/keys/last-seen")
    seen_keys = keys.status == 200
    a = run.question('ask "Where did I leave my keys?"', "Where did I leave my keys?", evidence=seen_keys, cited=seen_keys)
    hits = run.api.get("/memory/search?q=package&limit=5")
    seen_package = isinstance(hits.data, list) and bool(hits.data)
    run.question('ask "Did a package arrive?"', "Did a package arrive?", evidence=seen_package, cited=seen_package)
    night = night_window(datetime.now().astimezone())
    q = urllib.parse.urlencode({"from": night[0].isoformat(), "to": night[1].isoformat()})
    seen = run.api.get(f"/memory/episodes?{q}")
    overnight = len(seen.data) if isinstance(seen.data, list) else 0
    run.question('ask "What happened at our table overnight?"', "What happened at our table overnight?", evidence=overnight > 0, cited=overnight > 0,
                 context=f"{overnight} episode(s) between {night[0]:%a %H:%M} and {night[1]:%a %H:%M}")
    run.question('ask "What happened while I was out?"', "What happened while I was out?", evidence=episodes)
    ref = next((c["frame_ref"] for c in (a or {}).get("citations", []) if c.get("frame_ref")), None)
    if ref is None and isinstance(keys.data, dict):
        ref = keys.data.get("frame_ref")
    if ref:
        r = run.api.get("/memory/keyframes/" + urllib.parse.quote(ref, safe=""))
        if r.status == 200 and r.content_type.startswith("image/"):
            run.add("the cited frame opens", PASS, r.seconds, r.content_type)
        elif r.status == 404:
            run.add("the cited frame opens", SKIP, r.seconds, "no picture stored for this frame (a seeded log has none): the app shows its placeholder")
        else:
            run.add("the cited frame opens", FAIL, r.seconds, f"HTTP {r.status} {r.content_type}")
    else:
        run.add("the cited frame opens", SKIP, None, "no answer cited a frame")


def uc_talk(run: Run) -> None:
    run.question('"How\'s your day?"', "How's your day?")
    run.question('"Tell me a joke"', "Tell me a joke")
    run.question("a personal question", "What's your favorite color?", evidence=True, needs_model=True)
    run.question("simple maths: 17 times 23", "What is 17 times 23?", want="391", needs_model=True)
    a = run.question("a web question (fallback)", "Who won the FIFA World Cup in 2022?", evidence=True, needs_model=True)
    if a is not None:
        links = [c for c in a.get("citations") or [] if c.get("url")]
        run.add("the web answer names its source", PASS if links else SKIP, None,
                f"{len(links)} link(s) cited" if links else "answered without a link: from the model, or the box has no web right now")


def uc_day(run: Run) -> None:
    run.screen("/now/")
    now = run.fetch("the day as the box sees it", "/api/day/now", ("day", "plan", "status"), missing="the Day loop is not on this box",
                    check=lambda d: None if d["plan"] else "no plan for today",
                    note=lambda d: f"{len(d['plan']['moves'])} blocks; next: {((d.get('status') or {}).get('next_block') or {}).get('title')}")
    if now is None:
        return
    run.fetch("today's plan", "/api/day/plan", ("moves", "blocks", "outcome"),
              note=lambda d: f"{d['outcome'].get('summary', '')} committed={d['committed']}")
    brief = run.fetch("the day in words", "/api/day/brief", ("mode", "engine", "latest"),
                      note=lambda d: f"{d['engine']} engine; said: {', '.join(d['latest']) or 'nothing yet'}")
    if brief:
        morning = brief["latest"].get("morning")
        run.add("morning plan narrated", PASS if morning else FAIL, None, morning["display"] if morning else "no morning narration in the log")
    if not run.ask:
        run.add('"Brief me"', SKIP, None, "read-only without --ask")
    else:
        for moment, step in (("now", '"Brief me" (the Now ribbon)'), ("morning", "the morning brief again")):
            r = run.api.post("/api/day/brief", {"moment": moment}, timeout=60)
            n = r.data.get("narration") if isinstance(r.data, dict) else None
            if r.status == 200 and n and n.get("display") and n.get("spoken"):
                run.add(step, PASS, r.seconds, f"{r.data.get('engine')}: {n['display']}")
            elif r.status in (404, 409):
                run.add(step, SKIP, r.seconds, r.detail)
            else:
                run.add(step, FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")

    nxt = (now.get("status") or {}).get("next_block") or {}
    here = (now.get("status") or {}).get("seen_at") or (now.get("status") or {}).get("should_be")
    if nxt.get("place") and here and here != nxt["place"]:
        q = urllib.parse.urlencode({"from": here, "to": nxt["place"], "arrive_by": nxt["start"]})
        run.fetch("leave-by and the campus route", f"/api/places/route?{q}", ("sentence", "depart", "minutes", "polyline"),
                  check=lambda d: None if d["polyline"] or d["minutes"] >= 0 else "no route",
                  note=lambda d: f"{d['sentence']} ({len(d['polyline'])} points drawn)")
    else:
        run.add("leave-by and the campus route", SKIP, None, "no next block at another place right now")

    review = now.get("review") or now.get("yesterday")
    evening = ((brief or {}).get("latest") or {}).get("evening")
    if review or evening:
        text = (evening or {}).get("display") or review.get("text", "")
        run.add("the evening review", PASS, None, text)
        waste = "Waste Watch" in text or "sorting" in ((evening or {}).get("spoken") or "")
        run.add("the review's Waste Watch clause", PASS if waste else SKIP, None,
                "present" if waste else "no Waste Watch clause in this review (nothing was thrown away that day, or Waste Watch is off)")
    else:
        run.add("the evening review", SKIP, None, "no closed day in this log yet: it is written when the day closes "
                "(a night seeded at its scripted time has one: python -m contexts.demo seed)")


def uc_day_replan(run: Run) -> None:
    """The re-plan and its approval card. Run after the coach's use case: approving a re-plan replaces the
    day's plan, and what the coach shows hangs off that plan (checked here)."""
    coach = run.api.get("/api/day/coach")
    before = len(coach.data.get("sessions") or []) if isinstance(coach.data, dict) else 0
    if run.writes:
        r = run.api.post("/api/day/plan", {"from_here": True}, timeout=60)
        if r.status == 200 and isinstance(r.data, dict) and r.data.get("action_id"):
            run.add("re-plan from here", PASS, r.seconds, f"{r.data.get('type')}: {r.data['plan'].get('reason') or r.data['plan'].get('text') or r.data['plan_id']}")
            card = run.card(r.data["action_id"])
            expect = bool(card) and local_only_card(card)[0]  # local only when the box has no calendar connector
            approved = run.settle_card("the re-plan's approval card", r.data["action_id"], expect_local=expect)
            if approved:
                time.sleep(1.0)
                run.fetch("the approved plan is today's plan", "/api/day/now", ("committed",),
                          check=lambda d: None if d["committed"] else "the plan is not marked committed after approval",
                          note="committed")
                after = run.api.get("/api/day/coach")
                kept = len(after.data.get("sessions") or []) if isinstance(after.data, dict) else 0
                if before:
                    run.add("the coach keeps its blocks after the re-plan", PASS if kept >= before else FAIL, after.seconds,
                            f"{before} focus block(s) before the re-plan, {kept} after" + ("" if kept >= before else
                            ": the blocks that already happened are gone from the coach, so \"Was that right?\" has nothing left to ask"))
        elif r.status == 200:
            run.add("re-plan from here", FAIL, r.seconds, "a new plan without an approval card")
        elif r.status == 409:
            run.add("re-plan from here", SKIP, r.seconds, r.detail)
        else:
            run.add("re-plan from here", FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")
    else:
        run.add("re-plan from here + its approval card", SKIP, None, "read-only: a re-plan raises a card")


def uc_coach(run: Run) -> None:
    run.screen("/coach/")
    coach = run.fetch("the coach's view of today", "/api/day/coach", ("sessions", "notes", "latest"), missing="the Day loop is not on this box",
                      note=lambda d: f"{len(d['sessions'])} focus block(s), {sum(len(s['breaks']) for s in d['sessions'])} break(s), {len(d['notes'])} note(s)")
    if coach is None:
        return
    latest = coach.get("latest")
    run.add("the coach note", PASS if latest else SKIP, None, (latest or {}).get("text") or (latest or {}).get("display") or
            ("a note exists" if latest else "no block has ended with a note yet"))
    fb = run.fetch('"Was that right?"', "/api/day/coach/feedback", ("sessions", "options", "answers", "threshold", "learning"),
                   missing="the feedback loop is not wired",
                   note=lambda d: f"{sum(len(s['breaks']) for s in d['sessions'])} break(s) to ask about, {d['answers']} answered; options: "
                                  + ", ".join(o["label"] for o in d["options"]))
    if fb is None:
        return
    target = next(((s, b) for s in fb["sessions"] for b in s["breaks"] if not b.get("correction")), None)
    if not run.writes:
        run.add("answer one break", SKIP, None, "read-only")
    elif target is None:
        run.add("answer one break", SKIP, None, "no unanswered break in the last three days")
    else:
        s, b = target
        body = {"day": s.get("day") or s["start"][:10], "block_id": s["block_id"], "break_index": b["index"], "right": False, "answer": "phone",
                "note": "demo walkthrough"}
        r = run.api.post("/api/day/coach/corrections", body)
        ok = r.status == 200 and isinstance(r.data, dict) and "correction" in r.data
        run.add("answer one break (it was the phone)", PASS if ok else FAIL, r.seconds,
                f"guess {r.data['correction'].get('guess')} -> answer {r.data['correction'].get('answer')}" if ok else f"HTTP {r.status}: {r.detail}")
        for action_id in ((r.data or {}).get("proposed") or {}).values() if ok else ():
            run.settle_card("the teach-task proposal (export card)", action_id, expect_local=False)
    th = fb.get("threshold") or {}
    if fb.get("learning"):
        run.add("the teach-task proposal", PASS, None, f"{len(fb['learning'])} task(s) proposed from corrections")
    else:
        run.add("the teach-task proposal", SKIP, None, f"needs {th.get('answers', 5)} answers on one kind of break with {th.get('wrong', 2)} wrong; "
                "this log has too few breaks (covered by contexts/day/tests/test_coach_feedback.py)")


def uc_dayrun(run: Run) -> None:
    run.screen("/dayrun/")
    script = run.fetch("the scripted day (Demo day)", "/api/dayrun/scenarios/demo", ("plans",), missing="Day Run is not configured",
                       note=lambda d: f"plans: {', '.join(d['plans'])}; starts {d.get('now')}")
    board = run.fetch("the board", "/api/dayrun/board?scenario=demo", ("slots", "blocks", "deadlines"), missing="Day Run is not configured",
                      note=lambda d: f"{len(d['slots'])} open slots, {len(d['blocks'])} fixed blocks, {len(d['deadlines'])} deadline(s)")
    if not script or not board:
        return
    plans = script["plans"]

    def sim(step: str, moves: list, check, note) -> dict | None:
        r = run.api.post("/api/dayrun/simulate", {"scenario": "demo", "start": board["start"], "moves": moves}, timeout=60)
        if r.status != 200 or not isinstance(r.data, dict):
            run.add(step, FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")
            return None
        complaint = check(r.data)
        run.add(step, FAIL if complaint else PASS, r.seconds, complaint or note(r.data))
        return r.data

    sim("Bad plan fails with reasons", plans.get("bad", []),
        lambda d: None if not d["outcome"]["won"] and d["stars"] == 0 and d["outcome"]["reasons"] else f"won={d['outcome']['won']} stars={d['stars']}",
        lambda d: f"{d['outcome']['summary']} Rejected: {'; '.join(x['reason'] for x in d['rejected']) or 'nothing'}")
    sim("Good plan wins three stars", plans.get("good", []),
        lambda d: None if d["outcome"]["won"] and d["stars"] == 3 else f"won={d['outcome']['won']} stars={d['stars']}",
        lambda d: f"{d['outcome']['summary']} {len(d['routes'])} routes, {d['totals']['walk_minutes']} min walking")
    body = {"scenario": "demo", "start": board["start"], "moves": plans.get("good", []), **({"disruptions": script["disruptions"]} if script.get("disruptions") else {"seed": 7, "count": 2})}
    r = run.api.post("/api/dayrun/hard", body, timeout=60)
    if r.status == 200 and isinstance(r.data, dict):
        kinds = sorted({c["kind"] for c in r.data["changes"]})
        ok = len(r.data["disruptions"]) >= 1 and bool(kinds)
        run.add("Hard mode re-routes the plan", PASS if ok else FAIL, r.seconds,
                f"{'; '.join(d['text'] for d in r.data['disruptions'])} -> {', '.join(kinds)}; after: {r.data['after']['outcome']['summary']}")
    else:
        run.add("Hard mode re-routes the plan", FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")
    run.fetch("Play mode: the real campus map", "/api/places/map/ucf", ("buildings", "walk", "places", "attribution"), missing="no drawn campus map on this box",
              note=lambda d: f"{len(d['buildings'])} buildings, {len(d['places'])} places; {d['attribution']}")
    run.fetch("Play mode: a walk on the footpaths", "/api/play/campus-route?from=ucf.eng2&to=ucf.library", ("path", "minutes"), missing="no place packs",
              note=lambda d: f"{d['minutes']} min, {d.get('meters', '?')} m, {len(d.get('polyline') or [])} points")
    run.screen("/dayrun/?play=1")


def uc_circle(run: Run) -> None:
    run.screen("/circle/")
    members = run.fetch("People", "/api/circle/members", ("configured", "members"),
                        note=lambda d: ", ".join(m["display_name"] + (" (" + m["place"]["place"] + ")" if m.get("place") else "") for m in d["members"]) or "nobody")
    if not members or not members["configured"]:
        run.add("a Circle on this box", SKIP, None, "the Circle is not configured here (seed one: python -m contexts.demo circle --data-dir <dir>)")
        return
    threads = run.fetch("plans the agents negotiate", "/api/circle/threads", ("threads",),
                        check=lambda d: None if d["threads"] else "no conversation in the Circle",
                        note=lambda d: "; ".join(f"{t['title']} [{t['state']}]" for t in d["threads"]))
    agreed: list[dict] = []
    if threads:
        long = max(threads["threads"], key=lambda t: t.get("messages_count", 0))
        q = urllib.parse.quote(long["thread_id"], safe="")
        run.fetch("the conversation view", f"/api/circle/threads/{q}", ("messages",),
                  check=lambda d: None if d["messages"] else "a thread without messages",
                  note=lambda d: f"{d['title']}: " + " / ".join(m["line"] for m in d["messages"]))
        agreed = [t for t in threads["threads"] if t["state"] == "agreed"]
    events = run.fetch("agreed plans on the shared calendar", "/api/circle/events", ("events", "source"),
                       note=lambda d: f"{len(d['events'])} upcoming from {d['source']}: " + ", ".join(e["title"] for e in d["events"]))
    now = run.api.get("/api/day/now")
    if isinstance(now.data, dict) and now.data.get("plan") and (agreed or (events and events["events"])):
        titles = {t["title"] for t in agreed} | {e["title"] for e in (events or {}).get("events", [])}
        landed = [m["title"] for m in now.data["plan"]["moves"] if any(t in m["title"] for t in titles)]
        horizon = now.data["plan"].get("end", "")
        soon = [t["title"] for t in agreed if t.get("starts_at") and t["starts_at"][:19] < horizon[:19]] if horizon else []
        if landed:
            run.add("an agreed plan lands in the day", PASS, None, f"in today's plan: {', '.join(landed)}")
        elif soon:
            run.add("an agreed plan lands in the day", FAIL, None, f"{', '.join(soon)} is agreed and starts before the plan ends ({horizon}), but no block carries it")
        else:
            run.add("an agreed plan lands in the day", SKIP, None, "no agreed plan starts inside the day's plan window")
    else:
        run.add("an agreed plan lands in the day", SKIP, None, "no plan for today or nothing agreed")
    run.fetch("League and stakes", "/api/league", ("standings", "stake", "summary"),
              check=lambda d: None if d.get("configured") else "the League is not configured",
              note=lambda d: f"{d['summary']} Stake: {(d['stake'] or {}).get('text_for_week')}")
    run.fetch("friends' places", "/api/presence", missing="the Circle is not configured",
              note=lambda d: ", ".join(f"{p['member_id']} at {p['place']}" for p in d) or "nobody is sharing")
    sharing = run.fetch("what leaves the box", "/api/circle/sharing", ("database", "share_presence", "kinds"),
                        note=lambda d: f"shared store: {d['database']}; places {'on' if d['share_presence'] else 'off'}; summaries {'on' if d['share_summaries'] else 'off'}")
    if not run.writes:
        run.add("share a place / start a plan", SKIP, None, "read-only")
        return
    if sharing and sharing["database"] == "not_configured":
        r = run.api.post("/api/presence", {"place": "Library", "hours": 1})
        ok = r.status == 200 and isinstance(r.data, dict) and r.data.get("place") == "Library"
        run.add("share a place (no shared store: it stays on this box)", PASS if ok else (SKIP if r.status == 422 else FAIL), r.seconds,
                f"Library until {r.data.get('expires_at')}" if ok else f"HTTP {r.status}: {r.detail}")
        if ok:
            r = run.api.call("DELETE", "/api/presence")
            run.add("stop sharing", PASS if r.status == 200 else FAIL, r.seconds, "" if r.status == 200 else f"HTTP {r.status}: {r.detail}")
    else:
        run.add("share a place", SKIP, None, "a shared store is connected: sharing would leave this box, so a person does it")
    start = (datetime.now(timezone.utc) + timedelta(days=3)).replace(minute=0, second=0, microsecond=0)
    body = {"title": "Walkthrough check (deny me)", "start": start.isoformat(), "end": (start + timedelta(hours=1)).isoformat(), "location": "the Library"}
    r = run.api.post("/api/circle/proposals", body)
    if r.status == 200 and isinstance(r.data, dict) and r.data.get("action_id"):
        run.settle_card("a new plan becomes an approval card (never sent by this script)", r.data["action_id"], expect_local=False)
    else:
        run.add("a new plan becomes an approval card", FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")


def uc_sortit(run: Run, scene_pushed: bool) -> None:
    run.screen("/play/")
    run.screen("/map/")
    room = run.fetch("the room twin", "/api/room", ("title", "furniture", "calibration", "uncalibrated"), missing="the room twin is not set up (no floor plan)",
                     note=lambda d: f"{d['title']}: {len(d['furniture'])} pieces of furniture; calibrated: {', '.join(d['calibration']) or 'no camera'}; "
                                    f"not calibrated: {', '.join(d['uncalibrated']) or 'none'}")
    live = run.fetch("the room, live", "/api/room/live", ("tracks", "seen"), missing="the room twin is not set up",
                     note=lambda d: f"{len(d['tracks'])} people now, {len(d['seen'])} things placed, round: {(d.get('round') or {}).get('item') or 'none'}, "
                                    f"last result: {_result(d.get('result'))}")
    acc = run.fetch("rounds played in this sitting", "/api/play/accuracy", ("rounds",), missing="Sort It is not configured",
                    note=lambda d: f"{d['rounds']} round(s)")
    if scene_pushed:
        played = bool(live and (live.get("result") or live.get("round"))) or bool(acc and acc["rounds"])
        run.add("a round started and was scored from the fake scene", PASS if played else FAIL, None,
                f"simulated: last result {_result((live or {}).get('result'))}" if played else "the scene was pushed but no round shows in /api/room/live or /api/play/accuracy")
    else:
        run.add("a round started from the fake scene", SKIP, None, "run with --scene (python -m contexts.demo fake perception --gateway ... --realtime)")
    run.manual("a face-started round", "hold an item up to the bin camera for two seconds with recognition on (table-day checklist)")
    if room:
        clear = run.fetch("route to the bin, against par", "/api/play/room-route?item=can", ("reachable", "points", "distance_m", "par_s"),
                          check=lambda d: None if d["reachable"] else "the bin is not reachable",
                          note=lambda d: f"can -> {d['bin']}: {d['distance_m']} m, par {d['par_s']} s, {len(d['points'])} points")
        tracks = (live or {}).get("tracks") or []
        if clear and tracks:
            run.add("route around a person", PASS, None, f"{len(tracks)} person(s) in the room; the route above was planned with them as obstacles")
        elif clear:
            run.add("route around a person", SKIP, None, "nobody is in the room right now: the detour needs a live person (or --scene, while it plays); "
                    "covered by contexts/play/tests/test_room_live.py")
    run.fetch('"which bin?" hint', "/api/play/route?item=cup", ("bin", "path"), missing="Sort It is not configured",
              note=lambda d: f"cup -> {d['bin']}, {len(d['path'])} cells on the bin camera's grid")
    run.fetch("practice set", "/api/impact/practice", ("size", "items"), missing="Waste Watch is not configured",
              note=lambda d: f"{d['size']} to practise: " + ", ".join(f"{i['item']} -> {i['belongs_in']}" for i in d["items"]))


def _result(result: dict | None) -> str:
    if not result:
        return "none"
    return f"{result.get('item')} {'right' if result.get('correct') else 'wrong'} bin, {result.get('seconds')} s, {result.get('score')} points"


def uc_impact(run: Run) -> None:
    run.screen("/impact/")
    run.fetch("measured counts", "/api/impact", ("total",), missing="Waste Watch is not configured",
              note=lambda d: f"{d['total']['recycled']} of {d['total']['recyclable_seen']} recyclables recycled, {d['total']['contamination']} wrong item(s)")
    story = run.fetch("the Impact story", "/api/impact/story", ("goal", "feed", "milestones", "week", "practice"), missing="Waste Watch is not configured",
                      note=lambda d: d.get("empty_text") if d.get("empty") else f"{len(d['feed'])} disposals in the feed, {d['clean_streak']} clean day(s)")
    if story is None:
        return
    goal = (story["goal"] or {}).get("diversion") or {}
    run.add("weekly goal", PASS if goal.get("text") else SKIP, None, goal.get("text") or "no goal yet: nothing counted this week")
    ruled = [f for f in story["feed"] if f.get("rule")]
    run.add("the feed shows the rule", PASS if ruled else SKIP, None,
            f"{ruled[0]['item']} -> {ruled[0]['bin']} ({'right' if ruled[0]['correct'] else 'wrong'}): {ruled[0]['rule']}" if ruled else "nothing in the feed yet")
    reached = (story["milestones"] or {}).get("reached") or []
    run.add("milestone", PASS if reached else SKIP, None, reached[-1]["text"] if reached else "no milestone reached this week yet")
    if not run.writes:
        run.add("add to practice (Teach me)", SKIP, None, "read-only")
    else:
        r = run.api.post("/api/impact/practice", {"label": "can"})
        ok = r.status == 200 and isinstance(r.data, dict) and r.data.get("added")
        run.add("add to practice (Teach me)", PASS if ok else FAIL, r.seconds,
                f"added {r.data['added']}; {r.data['practice']['size']} in the set" if ok else f"HTTP {r.status}: {r.detail}")


def uc_identity(run: Run) -> None:
    run.screen("/people/")
    s = run.fetch("recognition settings", "/identity/settings", ("identity_enabled", "available"), missing="owner recognition is not wired into this gateway",
                  note=lambda d: f"available={d['available']} on={d['identity_enabled']} everyone={d.get('everyone')} expressions={d.get('expressions_available')}"
                                 + (f"; {d['reason']}" if d.get("reason") else ""))
    if s is None:
        return
    run.fetch("who is enrolled", "/identity/members", missing="not wired", note=lambda d: f"{len(d)} enrolled")  # names are not printed
    run.fetch("enrolment state", "/identity/enroll/status", ("state",), missing="not wired", note=lambda d: f"{d['state']}, {d.get('frames_needed')} frames needed")
    run.fetch("who is in the room", "/identity/present", ("present",), missing="not wired", note=lambda d: f"{len(d['present'])} recognised now")
    run.fetch("expressions for the coach", "/identity/expressions", ("samples", "note"), missing="not wired", note=lambda d: f"{d['samples']} samples: {d['note']}")
    run.manual("enrol by phone scan / photos", "People -> Teach Atlas your face, with the model installed (table-day checklist)")
    run.manual("forget me", "People -> Enrolled -> Forget, then check the name is gone; this script never deletes a person")


def uc_privacy(run: Run) -> None:
    run.screen("/privacy/")
    run.fetch("the privacy policy", "/vault/policy", ("blur_faces", "offline"), missing="the vault is not on this box",
              check=lambda d: None if d["blur_faces"] else "face blur is OFF: stored pictures would show faces",
              note=lambda d: f"blur faces on; offline={d['offline']}; paused cameras: {', '.join(d['paused_cameras']) or 'none'}; keep {d.get('keep_days') or 'for ever'}")
    run.fetch("what left the box (egress proof)", "/vault/egress", ("allowed", "blocked", "enforced"), missing="the vault is not on this box",
              note=lambda d: f"firewall {'enforced' if d['enforced'] else 'NOT enforced on this box'}; {len(d['allowed'])} allowed destinations, {d['blocked']} blocked")
    run.manual("a stored frame shows blurred faces", "open a keyframe with a face in it on Timeline and look (table-day checklist)")
    run.manual("wipe scopes (Everything / Pictures only / Memory only)", "Privacy -> Delete…, on a throwaway box only; this script never wipes")


def uc_teach(run: Run) -> None:
    run.screen("/teach/")
    run.fetch("what the box has been taught", "/api/learning", ("tasks", "deployments"), missing="learning is not on this box",
              note=lambda d: f"{len(d['tasks'])} task(s), {len(d['deployments'])} deployment(s)")
    lab = run.fetch("Learning lab", "/api/learning/lab", ("runs", "sample", "real_runs"), missing="the Learning lab is not on this box",
                    note=lambda d: f"{d['real_runs']} real run(s); showing {len(d['runs'])} {'SAMPLE ' if d['sample'] else ''}run(s)")
    if lab is not None:
        run.screen("/lab/")
        sample = run.fetch("Learning lab: the labelled sample run", "/api/learning/lab?sample=1", ("runs", "sample"),
                           check=lambda d: None if d["sample"] and d["runs"] and all(r.get("sample") for r in d["runs"]) else "the sample run is missing or not labelled as a sample",
                           note=lambda d: f"simulated: run {d['runs'][0]['run_id']} ({d['runs'][0].get('status')}), labelled sample")
        if sample:
            rid = urllib.parse.quote(sample["runs"][0]["run_id"], safe="")
            run.fetch("Learning lab: the run's story", f"/api/learning/lab/{rid}?sample=1", ("training", "eval", "power", "dataset"),
                      note=lambda d: f"simulated: gate {((d.get('eval') or {}).get('verdict') or (d.get('eval') or {}).get('decision') or d.get('status'))}")
    if not run.writes:
        run.add("teach a task -> export approval card", SKIP, None, "read-only: teaching raises a card")
        return
    r = run.api.post("/api/teach", {"task": f"walkthrough-{int(time.time())}", "instruction": "demo walkthrough: safe to deny"})
    if r.status == 200 and isinstance(r.data, dict) and r.data.get("action_id"):
        run.add("teach a task", PASS, r.seconds, f"task {r.data.get('task')}")
        run.settle_card("the export approval card (never approved by this script)", r.data["action_id"], expect_local=False)
    else:
        run.add("teach a task", FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")


def uc_voice(run: Run) -> None:
    run.fetch("voice on the box", "/api/voice/status", ("state", "running"), missing="the voice proxy is not in this gateway",
              note=lambda d: f"{'running' if d['running'] else 'not running'}, state {d['state']}" + (f": {d['problem']}" if d.get("problem") else ""))
    run.manual("hands-free question", 'Ask screen -> "Hands-free off" to on, speak a question (needs a microphone, and https on a phone)')
    run.manual("spoken narration", 'Settings -> "Speak up on its own", then "Brief me" on Now and listen')
    run.manual("spoken approval ask", 'raise a card (Now -> "Re-plan from here") and listen; "yes" only opens Approvals')


def uc_spot(run: Run) -> None:
    run.screen("/spot/")
    ov = run.fetch("the kit is loaded", "/api/spot/overview", ("configured",), missing="the inspection context is not in this gateway")
    if ov is None:
        return
    if not ov["configured"]:
        run.add("an inspection kit on this box", SKIP, None, "no kit: the screen says how to add one (ATLAS_SPOT_KIT)")
        return
    triage = run.fetch("shift view: what needs attention", "/api/spot/triage", ("items", "uninspected"),
                       check=lambda d: None if d["items"] else "nothing ranked",
                       note=lambda d: f"{len(d['items'])} ranked; first {d['items'][0]['id']} (score {d['items'][0].get('score')}); "
                                      f"uninspected: {', '.join(str(u.get('asset_id')) for u in d['uninspected'])}")
    run.fetch("findings with intervals", "/api/spot/findings", ("findings",), note=lambda d: f"{len(d['findings'])} findings")
    run.fetch("the data check", "/api/spot/validation", ("rows", "issues"),
              note=lambda d: f"{d.get('agree')} of {d.get('compared')} numbers agree with the kit; {len(d['issues'])} thing(s) to know about the files")
    run.fetch("shift handoff", "/api/spot/handoff", ("lines", "text"), check=lambda d: None if d["lines"] else "an empty handoff",
              note=lambda d: f"{len(d['lines'])} lines; first: {d['lines'][0] if isinstance(d['lines'][0], str) else d['lines'][0].get('text')}")
    state = run.fetch("replay state", "/api/spot/replay/state", ("index", "total", "playing"), note=lambda d: f"event {d['index']} of {d['total']}, playing={d['playing']}")
    if run.ask:
        r = run.api.post("/api/spot/ask", {"text": "What should I check first this shift?"}, timeout=100)
        ok = r.status == 200 and isinstance(r.data, dict) and isinstance(r.data.get("text"), str) and len(r.data["text"]) > 20
        cited = ok and triage and triage["items"][0]["id"] in r.data["text"]
        run.add('ask "What should I check first this shift?"', PASS if cited else FAIL, r.seconds,
                r.data["text"] if ok else f"HTTP {r.status}: {r.detail}")
    else:
        run.add("ask the inspection assistant", SKIP, None, "read-only without --ask")
    if not run.writes:
        run.add("replay / work-order card", SKIP, None, "read-only: the replay is shared state and a work order raises a card")
        return
    if state:
        before = state["index"]
        r = run.api.post("/api/spot/replay/speed", {"speed": 1200})
        r = run.api.post("/api/spot/replay/start")
        time.sleep(3)
        r2 = run.api.post("/api/spot/replay/pause")
        moved = isinstance(r2.data, dict) and r2.data.get("index", 0) > before
        run.add("replay plays (simulated: the kit's own event stream)", PASS if r.status == 200 and moved else FAIL, r.seconds + r2.seconds,
                f"event {before} -> {r2.data.get('index') if isinstance(r2.data, dict) else '?'} of {state['total']} in 3 s at 1200x")
        run.api.post("/api/spot/replay/seek", {"index": before})
        run.api.post("/api/spot/replay/speed", {"speed": 60})
    if triage:
        asset = next((i["id"] for i in triage["items"] if i.get("kind", "asset") == "asset" and "-" in str(i["id"])), None)
        orders = run.api.get("/api/spot/work_orders")
        n_before = len(orders.data["work_orders"]) if isinstance(orders.data, dict) else 0
        r = run.api.post("/api/spot/actions/reinspect", {"asset_id": asset or ""})
        if r.status == 200 and isinstance(r.data, dict) and r.data.get("action_id"):
            run.add(f"Re-inspect {asset} raises a card", PASS, r.seconds, r.data.get("reason"))
            if run.settle_card("the work-order card", r.data["action_id"], expect_local=True):
                time.sleep(1.0)
                run.fetch("the work order is a file on this box", "/api/spot/work_orders", ("work_orders", "folder"),
                          check=lambda d: None if len(d["work_orders"]) == n_before + 1 else f"{len(d['work_orders'])} work orders, expected {n_before + 1}",
                          note=lambda d: f"{len(d['work_orders'])} work order(s) under {Path(d['folder']).name}/; nothing was sent anywhere")
        else:
            run.add(f"Re-inspect {asset} raises a card", FAIL, r.seconds, f"HTTP {r.status}: {r.detail}")


def uc_profiles(run: Run, expect: str | None) -> None:
    p = run.fetch("the profile's words", "/api/profile", ("name", "vocabulary"), missing="no profile on this box",
                  note=lambda d: f"{d['name']}: " + ", ".join(f"{k}={v}" for k, v in d["vocabulary"].items() if isinstance(v, str))
                                 + f"; meters {d.get('meters')}; scenario {d.get('scenario')}")
    if p is None:
        return
    if expect:
        run.add(f"the box runs the {expect} profile", PASS if p["name"] == expect else FAIL, None, f"it runs {p['name']}")
    else:
        run.add("switching profiles", SKIP, None, "one box runs one profile: start another with ATLAS_PROFILE=shift-worker|family-home|inspection "
                "and run this with --only profiles --expect-profile <name>")


def uc_operator(run: Run) -> None:
    for path in ("/operator/", "/zones/", "/live/"):
        run.screen(path)
    run.fetch("cameras", "/capture/status", missing="no cameras on this box (capture is off)",
              note=lambda d: "; ".join(f"{cam}: {st.get('status')} at {st.get('fps_actual', 0):.1f} fps" for cam, st in (d.get("cameras") or d).items()
                                       if isinstance(st, dict)) or one_line(d, 90))
    run.fetch("zones", "/api/zones", missing="this box does not serve its zones",
              note=lambda d: "; ".join(f"{cam}: {', '.join(z)}" for cam, z in d.items()))
    run.fetch("rules", "/api/rules", note=lambda d: ", ".join(f"{x['id']} ({'on' if x['enabled'] else 'off'})" for x in d))
    run.fetch("recent alerts", "/api/alerts", note=lambda d: f"{len(d)} alert(s); latest: {d[0]['message'] if d else 'none'}")
    run.fetch("runtime numbers", "/runtime/stats", missing="this build has no /runtime/stats (added in #180)",
              note=lambda d: ", ".join(f"{k}={v['value']}" for k, v in d.items() if isinstance(v, dict) and v.get("value") is not None) or "no number recorded yet")
    run.fetch("pre-flight (see: python -m contexts.demo preflight)", "/api/preflight", ("checks",), timeout=60,
              note=lambda d: f"{sum(c['ok'] for c in d['checks'])}/{len(d['checks'])} ready; not ready: "
                             + (", ".join(c["name"] for c in d["checks"] if not c["ok"]) or "nothing"))
    run.manual("draw a zone, watch Live", "Zones -> pick camera and zone, drag a box, Save zone; Live shows the camera (owner: teammates' screens)")


def walk(run: Run, only: set[str] | None, scene_pushed: bool, expect_profile: str | None) -> None:
    steps = {"ask": uc_ask, "talk": uc_talk, "day": uc_day, "coach": uc_coach, "dayrun": uc_dayrun, "circle": uc_circle,
             "sortit": lambda r: uc_sortit(r, scene_pushed), "impact": uc_impact, "identity": uc_identity, "privacy": uc_privacy,
             "teach": uc_teach, "voice": uc_voice, "spot": uc_spot, "profiles": lambda r: uc_profiles(r, expect_profile), "operator": uc_operator}
    if only is None or "ask" not in only:  # the model name decides which talk steps apply
        h = run.api.get("/health")
        if isinstance(h.data, dict):
            run.model = str(h.data.get("model", "?"))
    for key, _ in USE_CASES:
        if only and key not in only:
            continue
        todo = [(key, steps[key])]
        if key == "coach" and (only is None or "day" in only):
            todo.append(("day", uc_day_replan))  # use case 3's last step, after the coach has been looked at
        for case, step in todo:
            run.case = case
            try:
                step(run)
            except Exception as exc:  # an answer in a shape this script did not expect: one FAIL, and the walk goes on
                run.add("the walkthrough's own check", FAIL, None, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gateway", default="http://127.0.0.1:8000")
    p.add_argument("--token", default=None, help="the box's ATLAS_TOKEN (default: the ATLAS_TOKEN environment variable)")
    p.add_argument("--proxy", help="a SOCKS or HTTP proxy for curl, e.g. socks5h://127.0.0.1:1055")
    p.add_argument("--read-only", action="store_true", help="skip everything that writes, raises a card or changes shared state")
    p.add_argument("--ask", action="store_true", help="with --read-only: still ask questions and request a day brief")
    p.add_argument("--scene", action="store_true", help="first push the fake camera scene in real time")
    p.add_argument("--approve-local", action="store_true", help="approve cards that stay on the box (never any other)")
    p.add_argument("--only", help="comma-separated use cases: " + ",".join(k for k, _ in USE_CASES))
    p.add_argument("--expect-profile", help="fail unless GET /api/profile names this profile")
    p.add_argument("--json", type=Path, help="write the rows and counts to this file")
    args = p.parse_args(argv)

    import os

    token = args.token or os.environ.get("ATLAS_TOKEN") or None
    only = {k.strip() for k in args.only.split(",")} if args.only else None
    if only and (unknown := only - set(TITLES)):
        p.error(f"unknown use case(s): {', '.join(sorted(unknown))}")
    if args.read_only and args.approve_local:
        p.error("--approve-local does nothing with --read-only: no card is raised")
    run = Run(Client(args.gateway, token, args.proxy), read_only=args.read_only, ask=args.ask or not args.read_only,
              approve_local=args.approve_local)
    started = datetime.now().astimezone()
    mode = ("read-only" + (" + questions" if args.ask else "")) if args.read_only else ("writes" + (", approving local cards" if args.approve_local else ""))
    print(f"Atlas demo walkthrough against {args.gateway}{' via proxy' if args.proxy else ''} ({mode}), {started:%Y-%m-%d %H:%M %Z}\n")

    health = run.api.get("/health", timeout=20)
    if health.status != 200:
        print(f"The gateway did not answer /health (HTTP {health.status}: {health.detail}). Nothing was checked.")
        return 2
    scene_pushed = False
    if args.scene:
        push_scene(run, args.gateway, token)
        scene_pushed = run.rows[-1].status == PASS
    walk(run, only, scene_pushed, args.expect_profile)

    c = counts(run.rows)
    print("\n" + format_table(run.rows))
    print(f"\n{c[PASS]} PASS, {c[FAIL]} FAIL, {c[SKIP]} SKIP ({len(run.rows)} steps) against {args.gateway}, model {run.model}")
    for r in run.rows:
        if r.status == FAIL:
            print(f"  FAIL  {r.use_case}: {r.step}: {r.note}")
    if args.json:
        args.json.write_text(json.dumps({"gateway": args.gateway, "started": started.isoformat(), "mode": mode, "model": run.model,
                                         "counts": c, "by_use_case": by_use_case(run.rows), "rows": [asdict(r) for r in run.rows]}, indent=2) + "\n",
                             encoding="utf-8")
        print(f"wrote {args.json}")
    return exit_code(run.rows)


if __name__ == "__main__":
    sys.exit(main())
