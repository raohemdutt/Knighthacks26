"""Place packs and walking routes for Day Run (S5).

A place pack is one small file describing a campus, a neighborhood, or a home:
named places (with coordinates when known, and a kind) plus edges in walking
minutes. Packs nest: a campus place can point at a home pack (`pack:`), whose
`entry` place is where you come in; a room can point at a zones file. The
same A* runs at every scale.

    packs = PlaceSet.load("config/places")
    route = packs.route("ucf.library", "apartment.desk")   # crosses into the dorm's apartment
    packs.feasible("ucf.union", "ucf.hec", depart, start_of_class)  # "you miss the 10:00 by 6 minutes"
"""

import heapq
import math
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class Place(BaseModel):
    name: str
    kind: str = "place"
    lat: float | None = None
    lon: float | None = None
    aliases: list[str] = []  # other names calendars use: "HEC", "ENG2", "SU"
    pack: str | None = None  # a nested pack this place leads into (a dorm -> its apartment)
    zones: str | None = None  # a room's zones file (config/zones.yaml)


class PlacePack(BaseModel):
    name: str
    kind: str = "campus"  # campus | neighborhood | home
    entry: str | None = None  # where a parent pack's place leads in (home packs)
    places: dict[str, Place]
    edges: list[tuple[str, str, float]] = Field(default_factory=list)  # [a, b, minutes], walkable both ways
    network: str | None = None  # a map file next to the pack with the real footpaths (ucf.map.json); routes follow it
    walk_speed_mps: float | None = None  # walking speed on that network; None is walknet's default (1.35 m/s)
    _net: object = PrivateAttr(default=None)  # the loaded WalkNetwork, set by PlaceSet.load
    _walked: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "PlacePack":
        problems = []
        for a, b, minutes in self.edges:
            for p in (a, b):
                if p not in self.places:
                    problems.append(f"edge {a}-{b} uses unknown place {p!r}")
            if minutes <= 0:
                problems.append(f"edge {a}-{b} has {minutes} minutes; must be positive")
        if self.entry and self.entry not in self.places:
            problems.append(f"entry {self.entry!r} is not a place")
        if not problems and self.places:
            unreachable = set(self.places) - _reachable(next(iter(self.places)), self._graph())
            if unreachable:
                problems.append(f"not reachable from the rest: {', '.join(sorted(unreachable))}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def _graph(self) -> dict[str, list[tuple[str, float]]]:
        g: dict[str, list[tuple[str, float]]] = {p: [] for p in self.places}
        for a, b, m in self.edges:
            g[a].append((b, m))
            g[b].append((a, m))
        return g

    def attach_network(self, net) -> None:
        """Use a walknet.WalkNetwork for this pack's routes (None goes back to the edges)."""
        self._net = net
        self._walked = {}

    @property
    def map_file(self) -> Path | None:
        """The map file the attached network was read from, for the phone to draw."""
        return getattr(self._net, "path", None)

    def walk(self, a: str, b: str):
        """The real walk between two places on the pack's network (a walknet.WalkRoute: points, meters, minutes),
        or None when the pack has no network or either place is not on it."""
        if self._net is None or a == b or a not in self.places or b not in self.places:
            return None
        if (a, b) not in self._walked:
            pa, pb = self.places[a], self.places[b]
            real = None
            if None not in (pa.lat, pa.lon, pb.lat, pb.lon):
                try:  # a building's doors when the map knows them, else the place's own point
                    real = self._net.route(self._net.doors.get(a) or (pa.lat, pa.lon), self._net.doors.get(b) or (pb.lat, pb.lon))
                    real = real.with_ends((pa.lat, pa.lon), (pb.lat, pb.lon))
                except ValueError:
                    real = None
            self._walked[a, b] = real
        return self._walked[a, b]

    def route(self, a: str, b: str) -> tuple[list[str], float]:
        """Shortest walk by minutes. A* with a straight-line heuristic scaled to the fastest edge, so it stays admissible.
        With a network attached, the walk is the real one along the footpaths, straight from a to b."""
        if (real := self.walk(a, b)) is not None:
            return [a, b], round(real.minutes, 1)
        g = self._graph()
        speed = self._max_speed_m_per_min()

        def h(p: str) -> float:
            pa, pb = self.places[p], self.places[b]
            if speed and None not in (pa.lat, pa.lon, pb.lat, pb.lon):
                return _meters(pa.lat, pa.lon, pb.lat, pb.lon) / speed
            return 0.0

        frontier = [(h(a), 0.0, a)]
        came: dict[str, str | None] = {a: None}
        cost = {a: 0.0}
        while frontier:
            _, c, p = heapq.heappop(frontier)
            if p == b:
                path = [p]
                while came[path[-1]] is not None:
                    path.append(came[path[-1]])
                return path[::-1], round(c, 1)
            if c > cost[p]:
                continue
            for q, m in g[p]:
                if c + m < cost.get(q, math.inf):
                    cost[q] = c + m
                    came[q] = p
                    heapq.heappush(frontier, (c + m + h(q), c + m, q))
        raise ValueError(f"no route from {a} to {b}")

    def _max_speed_m_per_min(self) -> float:
        speeds = []
        for a, b, m in self.edges:
            pa, pb = self.places[a], self.places[b]
            if None not in (pa.lat, pa.lon, pb.lat, pb.lon):
                speeds.append(_meters(pa.lat, pa.lon, pb.lat, pb.lon) / m)
        return max(speeds, default=0.0)

    def find(self, name: str) -> str | None:
        """A place id from an id or a (case-insensitive) display name, e.g. a friend's shared "Library"."""
        if name in self.places:
            return name
        low = name.strip().lower()
        for pid, p in self.places.items():
            if p.name.lower() == low or low in p.name.lower().split(" (")[0]:
                return pid
        return None


class PlaceSet:
    """All packs, addressed as "pack.place"; routes cross into nested packs through their entry."""

    def __init__(self, packs: dict[str, PlacePack]) -> None:
        self.packs = packs
        for pack in packs.values():
            for pid, place in pack.places.items():
                if place.pack and place.pack not in packs:
                    raise ValueError(f"{pack.name}.{pid} points at unknown pack {place.pack!r}")
                if place.pack and not packs[place.pack].entry:
                    raise ValueError(f"pack {place.pack!r} needs an `entry` to be nested")

    @classmethod
    def load(cls, *dirs: str | Path) -> "PlaceSet":
        packs = {}
        for d in dirs:
            for f in sorted(Path(d).glob("*.y*ml")) if Path(d).exists() else []:
                if f.name.startswith("."):  # macOS `._x.yaml` metadata copied along with a pack is not a pack
                    continue
                pack = parse_pack(f.read_text(encoding="utf-8"))
                packs[pack.name] = pack
                _attach(pack, [f.parent, *map(Path, dirs)])
        return cls(packs)

    def add(self, pack: PlacePack) -> None:
        candidate = dict(self.packs)
        candidate[pack.name] = pack
        PlaceSet(candidate)  # validates nesting before accepting
        self.packs = candidate

    def _split(self, ref: str) -> tuple[str, str]:
        if "." not in ref:
            raise ValueError(f"use pack.place, e.g. ucf.library (got {ref!r})")
        pack, place = ref.split(".", 1)
        if pack not in self.packs or self.packs[pack].find(place) is None:
            raise ValueError(f"unknown place {ref!r}")
        return pack, self.packs[pack].find(place)

    def _parent(self, pack: str) -> tuple[str, str] | None:
        for pname, p in self.packs.items():
            for pid, place in p.places.items():
                if place.pack == pack:
                    return pname, pid
        return None

    def route(self, src: str, dst: str) -> tuple[list[str], float]:
        """Route between any two places, up out of a home pack and down into another, in minutes."""
        sp, s = self._split(src)
        dp, d = self._split(dst)
        if sp == dp:
            path, minutes = self.packs[sp].route(s, d)
            return [f"{sp}.{p}" for p in path], minutes
        legs: list[tuple[list[str], float]] = []
        # leave a nested pack through its entry, onto the parent place
        while sp != dp and self._parent(sp) and not self._is_ancestor(sp, dp):
            entry = self.packs[sp].entry
            path, m = self.packs[sp].route(s, entry)
            legs.append(([f"{sp}.{p}" for p in path], m))
            sp, s = self._parent(sp)
        down: list[tuple[str, str]] = []
        cur = dp
        while cur != sp:
            parent = self._parent(cur)
            if parent is None:
                raise ValueError(f"no route from {src} to {dst}")
            down.append((cur, parent[1]))
            cur = parent[0]
        target = down[-1][1] if down else d
        path, m = self.packs[sp].route(s, target)
        legs.append(([f"{sp}.{p}" for p in path], m))
        for pack, _ in reversed(down):
            inner_dst = d if pack == dp else self._child_place(pack, dp)
            path, m = self.packs[pack].route(self.packs[pack].entry, inner_dst)
            legs.append(([f"{pack}.{p}" for p in path], m))
        full: list[str] = []
        for path, _ in legs:
            full += [p for p in path if not full or p != full[-1]]
        return full, round(sum(m for _, m in legs), 1)

    def _is_ancestor(self, pack: str, of: str) -> bool:
        cur = of
        while (parent := self._parent(cur)) is not None:
            if parent[0] == pack:
                return True
            cur = parent[0]
        return False

    def _child_place(self, pack: str, toward: str) -> str:
        cur = toward
        while (parent := self._parent(cur)) is not None:
            if parent[0] == pack:
                return parent[1]
            cur = parent[0]
        raise ValueError(f"{toward} is not inside {pack}")

    def feasible(self, src: str, dst: str, depart: datetime, due: datetime) -> dict:
        """Can you leave `src` at `depart` and be at `dst` by `due`? With the margin in minutes, either way."""
        path, minutes = self.route(src, dst)
        arrive = depart + timedelta(minutes=minutes)
        slack = round((due - arrive).total_seconds() / 60)
        at = due.strftime("%I:%M").lstrip("0")
        msg = f"you make the {at} with {slack} minutes to spare" if slack >= 0 else f"you miss the {at} by {-slack} minutes"
        out = {"ok": slack >= 0, "minutes": minutes, "slack_minutes": slack, "arrive": arrive.isoformat(), "path": path,
               "message": msg}
        line, meters = self.trace(path)
        if line:  # only packs with a network have the real line and its length
            out.update(polyline=line, meters=meters)
        return out

    def trace(self, path: list[str]) -> tuple[list[list[float]], float | None]:
        """The line walked along a route's places, as [[lat, lon], ...], and its metres; ([], None) without a network."""
        line: list[list[float]] = []
        meters = 0.0
        for x, y in zip(path, path[1:]):
            (px, a), (py, b) = x.split(".", 1), y.split(".", 1)
            real = self.packs[px].walk(a, b) if px == py else None
            if real is not None:
                line += [list(p) for p in real.points if not line or list(p) != line[-1]]
                meters += real.meters
        return line, (round(meters, 1) if line else None)

    def locate(self, text: str | None) -> str | None:
        """Find a place mentioned in free text ("until 11:15 AM - HEC 101") -> "pack.place". Longest match wins."""
        if not text:
            return None
        low = f" {text.lower()} "
        best: tuple[int, str] | None = None
        for pname, pack in self.packs.items():
            for pid, place in pack.places.items():
                for name in [place.name.split(" (")[0], *place.aliases]:
                    n = name.lower().strip()
                    hit = f" {n} " in low.replace(",", " ").replace("-", " ").replace("\u00b7", " ") if len(n) <= 4 else n in low
                    if n and hit and (best is None or len(n) > best[0]):
                        best = (len(n), f"{pname}.{pid}")
        return best[1] if best else None

    def find_shared(self, place_name: str) -> str | None:
        """A friend's shared place ("Library") -> "pack.place", searching campus packs first."""
        for pname, p in sorted(self.packs.items(), key=lambda kv: kv[1].kind != "campus"):
            if (pid := p.find(place_name)) is not None:
                return f"{pname}.{pid}"
        return None


def parse_pack(text: str) -> PlacePack:
    """YAML or JSON (JSON is YAML) into a validated pack; raises ValueError with what is wrong."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("a place pack is a mapping with name, places and edges")
    return PlacePack.model_validate(raw)


def _attach(pack: PlacePack, folders: list[Path]) -> None:
    """Load the pack's `network` map file, from the pack's own folder or any other pack folder. Without the file the
    pack routes over its edges as before."""
    if not pack.network or "/" in pack.network or "\\" in pack.network:
        return
    from contexts.play.domain.walknet import WalkNetwork

    for folder in folders:
        if (folder / pack.network).is_file():
            try:
                kw = {"speed_mps": pack.walk_speed_mps} if pack.walk_speed_mps else {}
                pack.attach_network(WalkNetwork.load(folder / pack.network, **kw))
            except (ValueError, KeyError, TypeError, IndexError, OSError):
                pass  # a broken map file is not a reason to lose the pack
            return


def _reachable(start: str, g: dict[str, list[tuple[str, float]]]) -> set[str]:
    seen, stack = {start}, [start]
    while stack:
        for q, _ in g[stack.pop()]:
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def _meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(a))
