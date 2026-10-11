"""What is where in the room right now, and Sort It played on the floor plan.

    perception.observation.recorded  -> every track's foot point through the camera's homography: people and things
                                        on the floor plan; where each thing was last seen (for the memory pins)
    play.round.started               -> a route per player from where the camera sees them to the right bin
                                        (play.route.planned), planned again when something moves into the way
    play.round.scored                -> the result against the route's par
    outside a round                  -> "which bin?": someone holding a sortable thing near the bins

State rebuilds from the log (observations replay); routes are only planned on live events. Event time throughout.

Every plan of a round is kept with how it was found (the cells the A* search expanded, what it went around, and on
a re-plan what moved into the way): `plans()`, for GET /api/play/route-search. The last round's stay until the next.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from contexts.play.domain.pathfinding import SortingRules
from contexts.play.room.grid import Disc, FloorGrid, Route, blocked_ahead, off_route, plan_route, search_json
from contexts.play.room.plan import Furniture, RoomPlan
from contracts.perception import ObservationRecorded, PersonIdentified, PersonLost
from contracts.play import FloorPoint, PlannedRoute, RoundScored, RoundStarted, RoutePlanned
from contracts.types import Track

PERSON = "person"
FRESH = timedelta(seconds=4)  # a camera's tracks count as "now" for this long after its last observation
HAND_M = 1.0  # a carried thing is drawn at hand height
PERSON_R = 0.25  # discs other walkers must keep out of, before the body radius is added
THING_R = 0.18
IN_BIN_M = 0.8  # a sortable thing seen lying this close to a bin's middle is in it
BEYOND_WALL_M = 0.15  # a foot point this far outside the walls is not on the floor: the thing is up in the air
ASSIST_BIN_M = 1.6  # "near the bins"
ASSIST_PERSON_M = 1.2  # "a person nearby"
REPLAN_EVERY = timedelta(seconds=1)
OFF_ROUTE_M = 0.8
ROUND_TIMEOUT = timedelta(seconds=90)  # an abandoned round has no event: forget it after this long
AROUND_M = 0.25  # a corner of the walk this close to something (past the body radius) went around it
PLANS_KEPT = 6  # of one round: the first plan and the newest re-plans


@dataclass(frozen=True)
class InTheWay:
    """Something a plan had to keep clear of, for this plan only."""

    disc: Disc
    kind: str  # "person", "thing", or "player" (the line of a route already given to another player)
    label: str
    key: str | None = None  # "<camera>:<track id>"

    def as_json(self) -> dict:
        return {"kind": self.kind, "label": self.label, "key": self.key, "x": round(self.disc[0], 3), "y": round(self.disc[1], 3),
                "r": round(self.disc[2], 3)}


@dataclass
class FloorTrack:
    key: str  # "<camera>:<track id>"
    camera: str
    track_id: int
    label: str
    x: float
    y: float
    z: float = 0.0  # metres above the floor: on a desk, in a hand, in a bin
    on: str | None = None  # the furniture it rests on or sits in
    held_by: str | None = None  # the key of the person carrying it
    in_bin: str | None = None
    seated: bool = False
    member_id: str | None = None

    @property
    def person(self) -> bool:
        return self.label == PERSON

    def as_json(self) -> dict:
        return {"key": self.key, "camera": self.camera, "track_id": self.track_id, "label": self.label,
                "kind": "person" if self.person else "object", "x": round(self.x, 3), "y": round(self.y, 3),
                "z": round(self.z, 3), "on": self.on, "held_by": self.held_by, "in_bin": self.in_bin,
                "seated": self.seated, "member_id": self.member_id}


def _held_by(thing: Track, people: list[tuple[Track, FloorTrack]]) -> FloorTrack | None:
    """The person carrying `thing`, judged in the picture: it overlaps them and is well above their feet."""
    cx, bottom = (thing.box.x1 + thing.box.x2) / 2, thing.box.y2
    for raw, placed in people:
        w, h = raw.box.x2 - raw.box.x1, raw.box.y2 - raw.box.y1
        if raw.box.x1 - 0.25 * w <= cx <= raw.box.x2 + 0.25 * w and raw.box.y1 - 0.05 * h <= bottom <= raw.box.y2 - 0.35 * h:
            return placed
    return None


def project_tracks(plan: RoomPlan, camera: str, tracks: list[Track], sortable: Callable[[str], bool] = lambda _: False) -> list[FloorTrack]:
    """One camera's tracks on the floor plan. A track stands where the bottom-centre of its box is.

    A person in a chair (or on the bed) is seated. A thing that lands on furniture rests on top of it; a thing that
    overlaps a person well above their feet is in their hand; a sortable thing at a bin is in the bin."""
    if camera not in plan.calibration:
        return []
    out: list[FloorTrack] = []
    people: list[tuple[Track, FloorTrack]] = []
    for t in tracks:
        if t.label != PERSON:
            continue
        p = plan.to_floor(camera, (t.box.x1 + t.box.x2) / 2, t.box.y2)
        if p is None:
            continue
        seat = next((f for f in plan.furniture if f.type in ("chair", "bed") and f.contains(*p, grow=0.12)), None)
        placed = FloorTrack(f"{camera}:{t.track_id}", camera, t.track_id, t.label, p[0], p[1], seated=seat is not None,
                            on=seat.id if seat else None)
        people.append((t, placed))
        out.append(placed)
    bins = list(plan.bins().values())
    for t in tracks:
        if t.label == PERSON:
            continue
        raw = plan.to_floor(camera, (t.box.x1 + t.box.x2) / 2, t.box.y2, clamp=False)
        if raw is None:
            continue
        p = plan.clamp(*raw)
        placed = FloorTrack(f"{camera}:{t.track_id}", camera, t.track_id, t.label, p[0], p[1])
        under = plan.furniture_at(*p)
        near_bin = min(bins, key=lambda b: math.hypot(b.at[0] - p[0], b.at[1] - p[1]), default=None)
        holder = None if under and under.type != "bin" else _held_by(t, people)
        if under and under.type != "bin" and under.height >= 0.3:
            placed.z, placed.on = under.height, under.id
        elif holder is not None:
            placed.x, placed.y, placed.z, placed.held_by = holder.x, holder.y, HAND_M, holder.key
        elif math.hypot(raw[0] - p[0], raw[1] - p[1]) > BEYOND_WALL_M:
            placed.z = HAND_M  # carried by someone the camera does not see
        elif near_bin and sortable(t.label) and math.hypot(near_bin.at[0] - p[0], near_bin.at[1] - p[1]) <= IN_BIN_M:
            placed.x, placed.y, placed.z = near_bin.at[0], near_bin.at[1], max(near_bin.height - 0.12, 0.0)
            placed.on, placed.in_bin = near_bin.id, near_bin.bin
        out.append(placed)
    return out


def side_of(person: tuple[float, float], facing: tuple[float, float], target: tuple[float, float]) -> str:
    """Is `target` on the left or the right of someone at `person` looking towards `facing`? (y runs south.)"""
    fx, fy = facing[0] - person[0], facing[1] - person[1]
    tx, ty = target[0] - person[0], target[1] - person[1]
    cross = fx * ty - fy * tx
    if abs(cross) < 0.12 * (math.hypot(fx, fy) * math.hypot(tx, ty) or 1.0):
        return "in front of you"
    return "on your right" if cross > 0 else "on your left"


@dataclass
class ActiveRound:
    round_id: str
    item: str
    target_bin: str  # zone id
    bin: str  # bin name
    started: datetime
    players: int
    routes: list[tuple[PlannedRoute, Route | None]]
    planned: datetime
    replans: int = 0


class RoomLive:
    def __init__(self, plan: RoomPlan, rules: SortingRules, active_round: Callable[[], str | None] | None = None) -> None:
        self.plan = plan
        self.rules = rules
        self.active_round = active_round  # the game's own round id, to notice a round it dropped without an event
        self.grid = FloorGrid.of(plan)
        self.by_camera: dict[str, tuple[datetime, list[FloorTrack]]] = {}
        self.members: dict[tuple[str, int], str] = {}
        self.seen: dict[str, dict] = {}  # label -> where it was last seen on the floor
        self.now: datetime | None = None  # the newest observation's time
        self.round: ActiveRound | None = None
        self._found: dict | None = None
        self.result: dict | None = None
        self.sessions: dict[str, list[int]] = {}  # session -> [right, rounds], for "3 of 4 right"
        self.assist: dict | None = None
        self.searched: dict | None = None  # the current or the last round's plans, each with how it was found

    def set_plan(self, plan: RoomPlan) -> None:
        """A new calibration: positions from now on use it."""
        self.plan = plan
        self.grid = FloorGrid.of(plan)

    # --- what is where ---

    def tracks(self, now: datetime | None = None) -> list[FloorTrack]:
        """Every track from a camera that reported within FRESH of `now` (default: the newest observation)."""
        now = now or self.now
        if now is None:
            return []
        return [t for ts, tracks in self.by_camera.values() if now - ts <= FRESH for t in tracks]

    def bin_of(self, label: str) -> Furniture | None:
        return self.plan.bins().get(self.rules.bin_for(label))

    def held_item(self) -> str | None:
        """The sortable thing a camera sees in play right now (in a hand first), as the game names it; else None.
        Sort It asks this when a round starts, so the round is about what the player is holding."""
        things = [t for t in self.tracks() if not t.person and not t.in_bin and self.rules.canonical(t.label)]
        things.sort(key=lambda t: (t.held_by is None, t.key))
        return self.rules.canonical(things[0].label) if things else None

    def on_identified(self, e: PersonIdentified) -> None:
        self.members[(e.camera_id, e.track_id)] = e.member_id
        self._tag(e.camera_id)

    def on_lost(self, e: PersonLost) -> None:
        for key in [k for k in self.members if k[0] == e.camera_id and e.track_id in (None, k[1])]:
            del self.members[key]
        self._tag(e.camera_id)

    def _tag(self, camera: str) -> None:
        for t in self.by_camera.get(camera, (None, []))[1]:
            t.member_id = self.members.get((camera, t.track_id))

    def on_observation(self, e: ObservationRecorded) -> list:
        placed = project_tracks(self.plan, e.camera_id, e.tracks, lambda label: self.rules.canonical(label) is not None)
        self.by_camera[e.camera_id] = (e.ts, placed)
        self._tag(e.camera_id)
        if self.now is None or e.ts > self.now:
            self.now = e.ts
        for t in placed:
            if not t.person:
                self.seen[t.label] = {"label": t.label, "x": round(t.x, 3), "y": round(t.y, 3), "z": round(t.z, 3),
                                      "on": t.on, "camera": t.camera, "ts": e.ts.isoformat()}
        self._drop_stale_round(e.ts)
        self.assist = None if self.round else self._assist(e.ts)
        return self._replan(e.ts) if self.round else []

    # --- Sort It on the floor ---

    def in_the_way(self, walker: FloorTrack | None, item: str, others: list[tuple[float, float]] = ()) -> list[InTheWay]:
        """Everyone else, and whatever is lying on the floor; never the walker or the thing being sorted."""
        out = [InTheWay((x, y, PERSON_R + self.plan.body_radius), "player", "the other player's line") for x, y in others]
        for t in self.tracks():
            if walker is not None and (t.key == walker.key or t.held_by == walker.key):
                continue
            if t.person:
                out.append(InTheWay((t.x, t.y, PERSON_R + self.plan.body_radius), "person", "person", t.key))
            elif t.z < 0.05 and not t.held_by and not t.in_bin and not self.rules.same_item(t.label, item):
                out.append(InTheWay((t.x, t.y, THING_R + self.plan.body_radius), "thing", t.label, t.key))
        return out

    def _obstacles(self, walker: FloorTrack | None, item: str, others: list[tuple[float, float]] = ()) -> list[Disc]:
        return [o.disc for o in self.in_the_way(walker, item, others)]

    def went_around(self, route: Route, things: list[InTheWay], target: Furniture | None = None) -> list[dict]:
        """What the walk bends around: at each corner of the pulled line, the nearest thing it had to keep clear of.

        A corner is only there because a straight walk past it was blocked, so the thing nearest to it (furniture,
        a person, something on the floor, else the wall) is what the route went around. In walking order, once each."""
        out: list[dict] = []
        reach = self.plan.body_radius + AROUND_M
        for x, y in route.points[1:-1]:
            best: tuple[float, dict] | None = None
            for f in self.plan.furniture:
                if f.blocks and (target is None or f.id != target.id) and (d := f.distance(x, y)) <= reach and (best is None or d < best[0]):
                    best = (d, {"id": f.id, "kind": "furniture", "label": f.label or (f"{f.bin} bin" if f.bin else f.type.replace("-", " ")),
                                "x": f.at[0], "y": f.at[1]})
            for o in things:
                d = math.hypot(o.disc[0] - x, o.disc[1] - y) - o.disc[2] + self.plan.body_radius
                if d <= reach and (best is None or d < best[0]):
                    best = (d, {"id": o.key or o.kind, "kind": o.kind, "label": o.label, "x": round(o.disc[0], 3), "y": round(o.disc[1], 3)})
            if best is None and min(x, y, self.plan.width - x, self.plan.depth - y) <= reach:
                best = (0.0, {"id": "wall", "kind": "wall", "label": "wall", "x": round(x, 3), "y": round(y, 3)})
            if best is not None and all(a["id"] != best[1]["id"] for a in out):
                out.append(best[1])
        return out

    def describe(self, route: Route | None, things: list[InTheWay], target: Furniture | None = None) -> dict:
        """How one route was found, for a screen: the search, what was in the way, and what the walk went around."""
        seen = [o.as_json() for o in things if o.kind != "player"]
        if route is None:
            return {"search": None, "obstacles": seen, "around": []}
        return {"search": search_json(route), "obstacles": seen, "around": self.went_around(route, things, target)}

    def players_for(self, item: str, count: int) -> list[FloorTrack]:
        """The people to route: whoever holds the item first, then whoever is nearest to it (or to the bin)."""
        people = [t for t in self.tracks() if t.person]
        things = [t for t in self.tracks() if not t.person and self.rules.same_item(t.label, item)]
        holder = next((t.held_by for t in things if t.held_by), None)
        target = self.bin_of(item)
        ref = (things[0].x, things[0].y) if things else (target.at if target else self.plan.start)
        people.sort(key=lambda p: (p.key != holder, math.hypot(p.x - ref[0], p.y - ref[1]), p.key))
        return people[:max(1, count)]

    def route_for(self, item: str, start: tuple[float, float], walker: FloorTrack | None = None,
                  keep_clear: list[tuple[float, float]] = ()) -> tuple[Furniture | None, Route | None]:
        """The right bin for `item` and the walk to it from `start` (None when there is no way or no such bin)."""
        target = self.bin_of(item)
        if target is None:
            return None, None
        grid = self.grid.with_obstacles(self._obstacles(walker, item, list(keep_clear)))
        radius = max(target.size) / 2
        route = plan_route(grid, self.plan.clamp(*start), target.at, radius, self.plan.walk_speed)
        if route is None and keep_clear:  # the other player's line leaves no way: share it
            route = plan_route(self.grid.with_obstacles(self._obstacles(walker, item)), self.plan.clamp(*start), target.at,
                               radius, self.plan.walk_speed)
        return target, route

    def _plan_round(self, item: str, players: int) -> list[tuple[PlannedRoute, Route | None]]:
        walkers: list[FloorTrack | None] = list(self.players_for(item, players)) or [None]
        thing = next((t for t in self.tracks() if not t.person and self.rules.same_item(t.label, item)), None)
        out: list[tuple[PlannedRoute, Route | None]] = []
        taken: list[tuple[float, float]] = []  # points along the routes already given out: the next player keeps clear
        self._found = None  # how the first player's route was found, for `_keep`
        for n, walker in enumerate(walkers, start=1):
            start = (walker.x, walker.y) if walker else ((thing.x, thing.y) if thing else self.plan.start)
            target, route = self.route_for(item, start, walker, keep_clear=taken)
            if n == 1:
                self._found = {"from": {"x": round(start[0], 3), "y": round(start[1], 3), "track": walker.key if walker else None},
                               "bin_id": target.id if target else None, "reachable": route is not None,
                               **(route.as_json() if route else {"points": [], "distance_m": 0.0, "par_s": 0.0}),
                               **self.describe(route, self.in_the_way(walker, item), target)}
            if route is None:
                out.append((PlannedRoute(player=n, track=walker.key if walker else None, reachable=False), None))
                continue
            out.append((PlannedRoute(player=n, track=walker.key if walker else None, distance_m=round(route.distance_m, 2),
                                     par_s=round(route.par_s, 1), points=[FloorPoint(x=round(x, 3), y=round(y, 3)) for x, y in route.points]),
                        route))
            # the middle of this walk is the next player's to avoid; both ends are shared (they start apart, end at the bin)
            cells = [self.grid.center(c) for c in route.cells]
            taken += [p for p in cells[3::3] if target and math.hypot(p[0] - target.at[0], p[1] - target.at[1]) > 1.0]
        return out

    def _event(self, r: ActiveRound, reason: str, ts: datetime) -> RoutePlanned:
        return RoutePlanned(ts=ts, round_id=r.round_id, item=r.item, target_bin=r.target_bin, room=self.plan.name,
                            reason=reason, routes=[planned for planned, _ in r.routes])

    def _keep(self, r: ActiveRound, reason: str, ts: datetime, rerouted: dict | None = None) -> None:
        """Remember this plan of the round with how it was found: the first one, and the newest re-plans."""
        if reason == "round_started" or self.searched is None or self.searched["round_id"] != r.round_id:
            self.searched = {"round_id": r.round_id, "item": r.item, "bin": r.bin, "players": r.players,
                             "started": r.started.isoformat(), "plans": []}
        plans = self.searched["plans"]
        plans.append({"seq": r.replans, "reason": reason, "ts": ts.isoformat(), "rerouted": rerouted, **(self._found or {})})
        if len(plans) > PLANS_KEPT:
            del plans[1:len(plans) - PLANS_KEPT + 1]

    def plans(self, now: datetime | None = None) -> dict:
        """GET /api/play/route-search: every plan of the round that is on, else of the last one, with its search.

        `live` follows the same rule as `state`: a round the game has dropped, or one left running past
        ROUND_TIMEOUT by the wall clock `now`, is not on any more; its plans are the last round's."""
        s = self.searched
        if s is None:
            return {"room": self.plan.name, "round_id": None, "live": False, "plans": []}
        r = self.round
        on = r is not None and r.round_id == s["round_id"]
        if on and ((self.active_round is not None and self.active_round() != r.round_id) or (now is not None and now - r.started > ROUND_TIMEOUT)):
            on = False
        return {"room": self.plan.name, **s, "live": on}

    def on_round_started(self, e: RoundStarted) -> list:
        self.result = None
        self.assist = None
        self.round = ActiveRound(e.round_id, e.item, e.target_bin, self.rules.bin_for(e.item), e.ts, e.players,
                                 self._plan_round(e.item, e.players), e.ts)
        self._keep(self.round, "round_started", e.ts)
        return [self._event(self.round, "round_started", e.ts)]

    def _moved_in(self, grid: FloorGrid, route: Route, at: tuple[float, float], things: list[InTheWay]) -> InTheWay | None:
        """The first thing standing on the part of `route` still ahead of the walker at `at`."""
        for cell in route.cells:
            x, y = grid.center(cell)
            if cell in grid.dynamic and math.hypot(x - at[0], y - at[1]) > 0.45:
                return min(things, key=lambda o: math.hypot(o.disc[0] - x, o.disc[1] - y) - o.disc[2], default=None)
        return None

    def _replan(self, now: datetime) -> list:
        """Plan again when something has moved onto a route, or its player has left it; say so only if it changed."""
        r = self.round
        if r is None or now - r.planned < REPLAN_EVERY:
            return []
        by_key = {t.key: t for t in self.tracks()}
        needed = False
        why: dict | None = None  # the first reason found, for the screen: what moved into the way, or that the player left the line
        for planned, route in r.routes:
            walker = by_key.get(planned.track) if planned.track else None
            if walker is None:
                continue
            if route is None:
                needed = True  # there was no way: there may be one now
                why = why or {"why": "opened", "kind": None, "label": None, "key": None}
                continue
            things = self.in_the_way(walker, r.item)
            grid = self.grid.with_obstacles([o.disc for o in things])
            if blocked_ahead(grid, route, (walker.x, walker.y)):
                needed = True
                o = self._moved_in(grid, route, (walker.x, walker.y), things)
                why = why or {"why": "blocked", "kind": o.kind if o else None, "label": o.label if o else None, "key": o.key if o else None}
            elif off_route(route, (walker.x, walker.y)) > OFF_ROUTE_M:
                needed = True
                why = why or {"why": "off_route", "kind": None, "label": None, "key": None}
        if not needed:
            return []
        r.planned = now
        fresh = self._plan_round(r.item, r.players)
        if [p for p, _ in fresh] == [p for p, _ in r.routes]:
            return []
        r.routes = fresh
        r.replans += 1
        self._keep(r, "replan", now, why)
        return [self._event(r, "replan", now)]

    def on_round_scored(self, e: RoundScored) -> None:
        r = self.round if self.round and self.round.round_id == e.round_id else None
        par = next((p.par_s for p, _ in r.routes if p.reachable), None) if r else None
        tally = self.sessions.setdefault(e.session_id, [0, 0])
        tally[0] += int(e.correct)
        tally[1] += 1
        self.result = {"round_id": e.round_id, "item": e.item, "correct": e.correct, "seconds": round(e.seconds, 1),
                       "score": e.score, "players": e.players, "par_s": par, "bin": self.rules.bin_for(e.item),
                       "ts": e.ts.isoformat(), "right": tally[0], "rounds": tally[1]}
        if r:
            self.round = None

    def _drop_stale_round(self, now: datetime) -> None:
        r = self.round
        if r is None:
            return
        dropped = self.active_round is not None and self.active_round() != r.round_id
        if dropped or now - r.started > ROUND_TIMEOUT:
            self.round = None

    def _assist(self, now: datetime) -> dict | None:
        """Someone is holding (or standing by) a sortable thing near the bins: which bin, and on which side."""
        tracks = self.tracks()
        people = [t for t in tracks if t.person]
        bins = list(self.plan.bins().values())
        if not bins or not people:
            return None
        mid = (sum(b.at[0] for b in bins) / len(bins), sum(b.at[1] for b in bins) / len(bins))
        for t in sorted((t for t in tracks if not t.person and not t.in_bin), key=lambda t: (t.held_by is None, t.key)):
            item = self.rules.canonical(t.label)
            target = self.bin_of(t.label) if item else None
            if target is None or min(math.hypot(b.at[0] - t.x, b.at[1] - t.y) for b in bins) > ASSIST_BIN_M:
                continue
            who = next((p for p in people if p.key == t.held_by), None) or min(people, key=lambda p: math.hypot(p.x - t.x, p.y - t.y))
            if math.hypot(who.x - t.x, who.y - t.y) > ASSIST_PERSON_M:
                continue
            side = side_of((who.x, who.y), mid, target.at) if len(bins) > 1 else "in front of you"
            name = "the trash" if target.bin == "trash" else target.bin
            colour = f"the {target.color} bin" if target.color else "the bin"
            return {"item": item, "label": t.label, "bin": target.bin, "bin_id": target.id, "color": target.color, "side": side,
                    "track": t.key, "person": who.key, "text": f"That {t.label} goes in {name}, {colour} {side}."}
        return None

    # --- for the phone ---

    def state(self, now: datetime | None = None) -> dict:
        """GET /api/room/live. `now` (wall clock) drops the tracks of a camera that stopped reporting."""
        newest = self.now
        live = now is None or newest is None or now - newest <= FRESH * 2
        r = self.round
        if r is not None and now is not None and now - r.started > ROUND_TIMEOUT:
            r = None  # the cameras went quiet in the middle of a round: do not leave its clock running on the phones
        return {
            "room": self.plan.name,
            "ts": newest.isoformat() if newest else None,
            "tracks": [t.as_json() for t in self.tracks()] if live else [],
            "seen": self.seen,
            "round": None if r is None else {
                "round_id": r.round_id, "item": r.item, "target_bin": r.target_bin, "bin": r.bin, "players": r.players,
                "started": r.started.isoformat(), "routes": [p.model_dump() for p, _ in r.routes], "replans": r.replans},
            "result": self.result,
            "assist": self.assist if live else None,
        }
