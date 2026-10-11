"""Build the UCF campus map and place pack from OpenStreetMap.

    python scripts/build_campus_pack.py                  # download (two Overpass queries), then write both files
    python scripts/build_campus_pack.py --cache /tmp/osm # reuse /tmp/osm/ucf-overpass.json and ucf-poi.json if present
    python scripts/build_campus_pack.py --check          # build in memory and print the numbers, write nothing

Writes two files, both committed:

- config/places/ucf.map.json  the campus as drawn and walked: building footprints with
  names, green and water areas, roads, and the walkable network (footways, paths,
  pedestrian areas, steps, crossings, service roads and campus streets) as a graph.
- config/places/ucf.yaml      the place pack. Place ids, kinds, aliases and nested packs
  are kept from the file as it is; coordinates and names come from OSM; edges are the
  real shortest walks between every two places on the network.

Map data © OpenStreetMap contributors, licensed under the Open Database License (ODbL),
https://www.openstreetmap.org/copyright. The raw download is a cache and is not committed.

The request carries only a plain User-Agent. Run it from the repository root.

Map file format (all coordinates are integers: `origin` + n * `unit` degrees, flat [lat, lon, lat, lon, ...]):

    bbox       [south, west, north, east]
    buildings  [{id, name?, short?, kind, c: [lat, lon] label point, p: ring}]
    areas      [{k: water | green | wood | pitch | pool | parking | plaza, name?, p: ring}]
    roads      [{k: major | minor | service, name?, p: line}]
    walk       {nodes: flat points, edges: [[a, b, kind, points between]]}   kind 0 path, 1 steps, 2 crossing, 3 road
    places     {place id: {osm, name, c, doors: flat points}}               where each pack place was found
"""

import argparse
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from contexts.play.domain.walknet import ATTRIBUTION, CROSSING, FOOT, ROAD, STEPS, WalkNetwork  # noqa: E402

BBOX = (28.5915, -81.2090, 28.6115, -81.1895)  # south, west, north, east: UCF main campus
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "atlas-knighthacks-demo/1.0"
UNIT = 1e-6
R = 6_371_000.0

GEOMETRY_QUERY = """[out:json][timeout:90][bbox:{s},{w},{n},{e}];
(
  way["building"];
  relation["building"];
  way["highway"];
  way["natural"="water"];
  relation["natural"="water"];
  way["leisure"~"^(park|pitch|garden|stadium|track|swimming_pool)$"];
  way["landuse"~"^(grass|forest|recreation_ground|meadow)$"];
  way["natural"~"^(wood|scrub|wetland)$"];
  way["amenity"="parking"];
  node["entrance"];
);
out body geom;
"""
# Named things that are not buildings (a lawn, a dining hall inside a commons): only their tags and centre.
POI_QUERY = """[out:json][timeout:60][bbox:{s},{w},{n},{e}];
(
  nwr["name"~"63|Memory|Dining|Marketplace|Mall",i];
  nwr["amenity"~"^(restaurant|fast_food|cafe|food_court)$"];
);
out tags center;
"""

# How each pack place is found in OSM. `building` matches a building's name; `poi` matches a named
# thing from the second query. With several matches the one nearest the pack's current coordinates wins.
PLACES = {
    "union": {"building": r"^Student Union$"},
    "library": {"building": r"^John C\. Hitt Library$"},
    "hec": {"building": r"Harris Engineering Center$"},
    "eng2": {"building": r"^Engineering II$"},
    "cb1": {"building": r"^Classroom Building I$"},
    "memory_mall": {"poi": r"^Memory Mall$"},
    "rwc": {"building": r"^Recreation and Wellness Center$"},
    "dining_63": {"poi": r"^'?63 South$"},
    "towers": {"building": r"^Tower (I|II|III|IV)$"},
}

WALK_KINDS = {"footway": FOOT, "path": FOOT, "pedestrian": FOOT, "cycleway": FOOT, "track": FOOT, "living_street": FOOT,
              "steps": STEPS, "service": ROAD, "residential": ROAD, "unclassified": ROAD, "tertiary": ROAD}
ROAD_CLASS = {"primary": "major", "primary_link": "major", "secondary": "major", "tertiary": "minor",
              "unclassified": "minor", "residential": "minor", "service": "service"}


# ---------- download ----------

def fetch(query: str, cache: Path | None, name: str) -> dict:
    if cache and (cache / name).is_file():
        return json.loads((cache / name).read_text(encoding="utf-8"))
    s, w, n, e = BBOX
    body = urllib.parse.urlencode({"data": query.format(s=s, w=w, n=n, e=e)}).encode()
    req = urllib.request.Request(OVERPASS, data=body, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as res:  # noqa: S310 (a fixed https URL)
        raw = res.read()
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(raw)
    return json.loads(raw)


# ---------- geometry, in metres east and north of the box's south-west corner ----------

K = math.cos(math.radians((BBOX[0] + BBOX[2]) / 2))
BOX = (0.0, 0.0, math.radians(BBOX[3] - BBOX[1]) * R * K, math.radians(BBOX[2] - BBOX[0]) * R)


def xy(lat: float, lon: float) -> tuple[float, float]:
    return math.radians(lon - BBOX[1]) * R * K, math.radians(lat - BBOX[0]) * R


def q(p: tuple[float, float]) -> tuple[int, int]:
    """Metres back to integer (lat, lon) offsets from the origin."""
    return round(math.degrees(p[1] / R) / UNIT), round(math.degrees(p[0] / (R * K)) / UNIT)


def flat(points: list[tuple[float, float]]) -> list[int]:
    out: list[int] = []
    for p in points:
        a, b = q(p)
        if len(out) < 2 or (out[-2], out[-1]) != (a, b):
            out += [a, b]
    return out


def inside(p: tuple[float, float]) -> bool:
    return BOX[0] <= p[0] <= BOX[2] and BOX[1] <= p[1] <= BOX[3]


def simplify(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Douglas-Peucker: drop points that are within `tol` metres of the line through their neighbours."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (ax, ay), (bx, by) = points[i], points[j]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        far, at = 0.0, -1
        for n in range(i + 1, j):
            px, py = points[n]
            if seg2 <= 0:
                d = math.hypot(px - ax, py - ay)
            else:
                g = min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
                d = math.hypot(px - ax - dx * g, py - ay - dy * g)
            if d > far:
                far, at = d, n
        if far > tol and at > 0:
            keep[at] = True
            stack += [(i, at), (at, j)]
    return [p for p, k in zip(points, keep) if k]


def simplify_ring(ring: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """A closed ring (first point not repeated), simplified in two halves so it stays a shape."""
    if len(ring) < 5:
        return ring
    far = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]))
    a = simplify(ring[: far + 1], tol)
    b = simplify(ring[far:] + ring[:1], tol)
    return a[:-1] + b[:-1]


def area(ring: list[tuple[float, float]]) -> float:
    return abs(sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))) / 2


def centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    a = cx = cy = 0.0
    for i in range(len(ring)):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % len(ring)]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)
    return cx / (3 * a), cy / (3 * a)


def contains(ring: list[tuple[float, float]], p: tuple[float, float]) -> bool:
    hit = False
    for i in range(len(ring)):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % len(ring)]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < x0 + (p[1] - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def label_point(ring: list[tuple[float, float]]) -> tuple[float, float]:
    """The centroid, or for an L or U shape the middle of the widest stretch of the building at that height."""
    c = centroid(ring)
    if contains(ring, c):
        return c
    xs = sorted(x0 + (c[1] - y0) * (x1 - x0) / (y1 - y0)
                for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]) if (y0 > c[1]) != (y1 > c[1]))
    spans = [(xs[i + 1] - xs[i], (xs[i] + xs[i + 1]) / 2) for i in range(0, len(xs) - 1, 2)]
    return (max(spans)[1], c[1]) if spans else c


def clip_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman against the box."""
    def cut(points, keep, cross):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if keep(cur):
                if not keep(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif keep(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    for keep, cross in ((lambda p: p[0] >= BOX[0], at_x(BOX[0])), (lambda p: p[0] <= BOX[2], at_x(BOX[2])),
                        (lambda p: p[1] >= BOX[1], at_y(BOX[1])), (lambda p: p[1] <= BOX[3], at_y(BOX[3]))):
        if len(ring) < 3:
            return []
        ring = cut(ring, keep, cross)
    return ring


def clip_line(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """The stretches of a line inside the box, cut exactly at its edge."""
    runs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for a, b in zip(points, points[1:]):
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        ok = True
        for p, qq in ((-dx, a[0] - BOX[0]), (dx, BOX[2] - a[0]), (-dy, a[1] - BOX[1]), (dy, BOX[3] - a[1])):
            if p == 0:
                ok = ok and qq >= 0
            else:
                t = qq / p
                if p < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
        if not ok or t0 > t1:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        p0, p1 = (a[0] + dx * t0, a[1] + dy * t0), (a[0] + dx * t1, a[1] + dy * t1)
        if not cur or t0 > 0:
            if len(cur) > 1:
                runs.append(cur)
            cur = [p0]
        cur.append(p1)
        if t1 < 1:
            runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return runs


# ---------- OSM elements ----------

def rings_of(el: dict) -> list[list[tuple[float, float]]]:
    """Outer rings of a way or a multipolygon relation, in metres, without the repeated last point."""
    if el["type"] == "way":
        g = [xy(p["lat"], p["lon"]) for p in el.get("geometry", [])]
        return [g[:-1]] if len(g) > 3 and g[0] == g[-1] else []
    parts = [[xy(p["lat"], p["lon"]) for p in m.get("geometry", [])] for m in el.get("members", [])
             if m.get("type") == "way" and m.get("role") in ("outer", "")]
    rings, parts = [p[:-1] for p in parts if len(p) > 3 and p[0] == p[-1]], [p for p in parts if len(p) > 1 and p[0] != p[-1]]
    while parts:  # outer ways that only close when joined end to end
        ring = parts.pop()
        grew = True
        while grew and ring[0] != ring[-1]:
            grew = False
            for i, p in enumerate(parts):
                if p[0] == ring[-1] or p[-1] == ring[-1]:
                    ring += (p if p[0] == ring[-1] else p[::-1])[1:]
                    parts.pop(i)
                    grew = True
                    break
        if len(ring) > 3 and ring[0] == ring[-1]:
            rings.append(ring[:-1])
    return rings


def area_kind(tags: dict) -> str | None:
    if tags.get("natural") == "water":
        return "water"
    if tags.get("leisure") == "swimming_pool":
        return "pool"
    if tags.get("leisure") in ("pitch", "track", "stadium"):
        return "pitch"
    if tags.get("natural") in ("wood", "scrub", "wetland") or tags.get("landuse") == "forest":
        return "wood"
    if tags.get("leisure") in ("park", "garden") or tags.get("landuse") in ("grass", "meadow", "recreation_ground"):
        return "green"
    if tags.get("amenity") == "parking" and "building" not in tags:
        return "parking"
    if tags.get("highway") == "pedestrian" and tags.get("area") == "yes":
        return "plaza"
    return None


def walkable(tags: dict) -> int | None:
    kind = WALK_KINDS.get(tags.get("highway", ""))
    if kind is None or tags.get("access") in ("private", "no") or tags.get("foot") == "no" or tags.get("service") == "drive-through":
        return None
    return CROSSING if tags.get("footway") == "crossing" or tags.get("highway") == "crossing" else kind


def build_graph(ways: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[int, int, int, list[tuple[float, float]]]], dict]:
    """Walkable ways -> nodes where they meet and edges with their shape; one piece, with the bends that say nothing merged."""
    use: dict[int, int] = {}
    kept = []
    for w in ways:
        kind = walkable(w.get("tags", {}))
        if kind is None or len(w.get("nodes", [])) != len(w.get("geometry", [])):
            continue
        run: list[tuple[int, tuple[float, float]]] = []
        for nid, g in zip(w["nodes"], w["geometry"]):
            p = xy(g["lat"], g["lon"])
            if inside(p):
                run.append((nid, p))
            else:
                if len(run) > 1:
                    kept.append((kind, run))
                run = []
        if len(run) > 1:
            kept.append((kind, run))
    for _, run in kept:
        for i, (nid, _) in enumerate(run):
            use[nid] = use.get(nid, 0) + (2 if i in (0, len(run) - 1) else 1)
    where: dict[int, tuple[float, float]] = {}
    edges: list[list] = []  # [node a, node b, kind, points between]
    for kind, run in kept:
        start = 0
        for i in range(1, len(run)):
            if use[run[i][0]] > 1 or i == len(run) - 1:
                a, b = run[start][0], run[i][0]
                where[a], where[b] = run[start][1], run[i][1]
                if a != b or i - start > 2:
                    edges.append([a, b, kind, [p for _, p in run[start + 1:i]]])
                start = i
    raw_edges = len(edges)

    # merge the two edges at a node where nothing else meets and the kind does not change
    at: dict[int, list[int]] = {}
    for i, e in enumerate(edges):
        at.setdefault(e[0], []).append(i)
        at.setdefault(e[1], []).append(i)
    for node in list(at):
        mine = at[node]
        if len(mine) != 2 or mine[0] == mine[1]:
            continue
        e1, e2 = edges[mine[0]], edges[mine[1]]
        if e1 is None or e2 is None or e1[2] != e2[2] or e1[0] == e1[1] or e2[0] == e2[1]:
            continue
        first = e1[3] if e1[1] == node else e1[3][::-1]  # ... -> node
        far1 = e1[0] if e1[1] == node else e1[1]
        second = e2[3] if e2[0] == node else e2[3][::-1]  # node -> ...
        far2 = e2[1] if e2[0] == node else e2[0]
        if far1 == far2:
            continue  # would become a loop with nothing on it
        edges[mine[0]] = [far1, far2, e1[2], first + [where[node]] + second]
        edges[mine[1]] = None
        at[far2] = [mine[0] if i == mine[1] else i for i in at[far2]]
        del at[node]
    edges = [e for e in edges if e is not None]

    # keep the largest connected piece: an island nobody can walk to is no use for routes
    parent: dict[int, int] = {}

    def find(n: int) -> int:
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b, _, _ in edges:
        parent[find(a)] = find(b)
    size: dict[int, int] = {}
    for a, _, _, _ in edges:
        size[find(a)] = size.get(find(a), 0) + 1
    main = max(size, key=size.get)
    dropped = sum(1 for e in edges if find(e[0]) != main)
    edges = [e for e in edges if find(e[0]) == main]

    ids: dict[int, int] = {}
    for a, b, _, _ in edges:
        ids.setdefault(a, len(ids))
        ids.setdefault(b, len(ids))
    nodes = [where[n] for n in ids]
    out = [(ids[a], ids[b], kind, simplify([where[a], *mid, where[b]], 0.6)[1:-1]) for a, b, kind, mid in edges]
    stats = {"walk_ways": len(kept), "segments_before_merge": raw_edges, "islands_dropped_edges": dropped, "pieces": len(size)}
    return nodes, out, stats


# ---------- the build ----------

def build(geometry: dict, poi: dict, old_pack: dict) -> tuple[dict, dict, dict]:
    els = geometry["elements"]
    ways = [e for e in els if e["type"] == "way"]
    entrances = {e["id"]: e for e in els if e["type"] == "node" and "entrance" in e.get("tags", {})}

    # buildings
    buildings = []
    for e in els:
        tags = e.get("tags", {})
        if "building" not in tags or e["type"] == "node":
            continue
        for ring in rings_of(e):
            ring = clip_ring(ring)
            if len(ring) < 3 or area(ring) < 20:
                continue
            doors = []
            if e["type"] == "way":
                for nid in e.get("nodes", []):
                    if nid in entrances:
                        en = entrances[nid]
                        doors.append((en["tags"].get("entrance") != "main", xy(en["lat"], en["lon"])))
            buildings.append({"id": e["id"], "type": e["type"], "name": tags.get("name"), "short": tags.get("short_name"),
                              "kind": tags["building"], "ring": simplify_ring(ring, 0.5), "label": label_point(ring),
                              "doors": [p for _, p in sorted(set(doors))]})
    buildings.sort(key=lambda b: (b["name"] is None, b["name"] or "", b["id"]))

    # areas
    areas = []
    for e in els:
        kind = area_kind(e.get("tags", {})) if e["type"] != "node" else None
        if kind is None or "building" in e.get("tags", {}):
            continue
        for ring in rings_of(e):
            ring = clip_ring(ring)
            small = {"water": 30, "pool": 30, "plaza": 30}.get(kind, 120)
            if len(ring) >= 3 and area(ring) >= small:
                ring = simplify_ring(ring, 1.0 if kind in ("green", "wood", "parking") else 0.7)
                areas.append({"k": kind, "name": e["tags"].get("name"), "ring": ring, "size": area(ring)})
    order = ["wood", "green", "parking", "pitch", "water", "pool", "plaza"]
    areas.sort(key=lambda a: (order.index(a["k"]), -a["size"]))

    # roads, for context
    roads = []
    for w in ways:
        tags = w.get("tags", {})
        cls = ROAD_CLASS.get(tags.get("highway", ""))
        if cls is None or tags.get("area") == "yes":
            continue
        line = [xy(p["lat"], p["lon"]) for p in w.get("geometry", [])]
        for run in clip_line(line):
            run = simplify(run, 1.0 if cls != "service" else 1.5)
            if len(run) > 1:
                roads.append({"k": cls, "name": tags.get("name") if cls != "service" else None, "line": run})
    roads.sort(key=lambda r: ["service", "minor", "major"].index(r["k"]))

    nodes, edges, stats = build_graph(ways)

    data = {
        "attribution": ATTRIBUTION,
        "license": "Open Database License (ODbL) 1.0, https://www.openstreetmap.org/copyright",
        "source": "OpenStreetMap via the Overpass API; trimmed and simplified by scripts/build_campus_pack.py",
        "osm_base": geometry.get("osm3s", {}).get("timestamp_osm_base"),
        "generated": date.today().isoformat(),
        "pack": old_pack["name"],
        "bbox": list(BBOX), "origin": [BBOX[0], BBOX[1]], "unit": UNIT,
        "buildings": [{k: v for k, v in (("id", b["id"]), ("name", b["name"]), ("short", b["short"]), ("kind", b["kind"]),
                                         ("c", list(q(b["label"]))), ("p", flat(b["ring"]))) if v is not None} for b in buildings],
        "areas": [{k: v for k, v in (("k", a["k"]), ("name", a["name"]), ("p", flat(a["ring"]))) if v is not None} for a in areas],
        "roads": [{k: v for k, v in (("k", r["k"]), ("name", r["name"]), ("p", flat(r["line"]))) if v is not None} for r in roads],
        "walk": {"nodes": flat_all(nodes), "edges": [[a, b, kind, flat(mid)] if mid else [a, b, kind] for a, b, kind, mid in edges]},
        "places": {},
    }

    # places: where each pack place is on the real campus
    pois = []
    for e in poi["elements"]:
        c = e.get("center", e)
        if "lat" in c and e.get("tags", {}).get("name"):
            pois.append({"ref": f"{e['type']}/{e['id']}", "name": e["tags"]["name"], "at": xy(c["lat"], c["lon"])})
    report: dict[str, dict] = {}
    for pid, place in old_pack["places"].items():
        rule = PLACES.get(pid, {})
        old = xy(place["lat"], place["lon"]) if place.get("lat") is not None and place.get("lon") is not None else None

        def near(p: tuple[float, float]) -> float:
            return math.hypot(p[0] - old[0], p[1] - old[1]) if old else 0.0

        found = None
        if "building" in rule:
            hits = [b for b in buildings if b["name"] and re.search(rule["building"], b["name"])]
            if hits:
                b = min(hits, key=lambda b: near(b["label"]))
                found = {"osm": f"{b['type']}/{b['id']}", "name": b["name"], "at": b["label"], "doors": b["doors"], "short": b["short"]}
        elif "poi" in rule:
            hits = [p for p in pois if re.search(rule["poi"], p["name"]) and inside(p["at"])]
            if hits:
                p = min(hits, key=lambda p: near(p["at"]))
                host = next((b for b in buildings if contains(b["ring"], p["at"])), None)
                doors = sorted(host["doors"], key=lambda d: math.hypot(d[0] - p["at"][0], d[1] - p["at"][1]))[:3] if host else []
                found = {"osm": p["ref"], "name": p["name"], "at": p["at"], "doors": doors, "short": None,
                         "inside": host["name"] if host else None}
        if found is None:
            report[pid] = {"snapped": False, "why": "not found in OpenStreetMap; the old coordinates are kept" if rule else "no rule for this place id"}
            continue
        report[pid] = {"snapped": True, "osm": found["osm"], "osm_name": found["name"], "moved_m": round(near(found["at"])),
                       "doors": len(found["doors"]), **({"inside": found["inside"]} if found.get("inside") else {})}
        lat, lon = q(found["at"])
        data["places"][pid] = {"osm": found["osm"], "name": found["name"], "c": [lat, lon], "doors": flat(found["doors"])}
        suffix = re.search(r" \(.*\)$", place["name"])
        base = place["name"].split(" (")[0]
        aliases = list(place.get("aliases") or [])
        for extra in (found["short"], base if base != found["name"] else None):
            if extra and extra not in aliases:
                aliases.append(extra)
        place["name"] = found["name"] + (suffix.group(0) if suffix else "")
        place["lat"], place["lon"] = round(BBOX[0] + lat * UNIT, 6), round(BBOX[1] + lon * UNIT, 6)
        if aliases:
            place["aliases"] = aliases

    stats.update(buildings=len(buildings), named_buildings=sum(1 for b in buildings if b["name"]), areas=len(areas),
                 roads=len(roads), entrances=len(entrances), graph_nodes=len(nodes), graph_edges=len(edges),
                 path_points=sum(len(m) + 2 for _, _, _, m in edges))
    return data, old_pack, {"places": report, "stats": stats}


def flat_all(points: list[tuple[float, float]]) -> list[int]:
    """Like flat(), but never drops a point: node numbers must stay put."""
    out: list[int] = []
    for p in points:
        out += q(p)
    return out


def real_edges(data: dict, pack: dict, speed: float | None) -> list[tuple[str, str, float, float]]:
    """Every two located places, with the real walk between them: (a, b, minutes, metres)."""
    net = WalkNetwork.from_map(data, **({"speed_mps": speed} if speed else {}))

    def ends(pid: str) -> list[tuple[float, float]]:
        d = net.doors.get(pid)
        return d or [(pack["places"][pid]["lat"], pack["places"][pid]["lon"])]

    ids = [pid for pid, p in pack["places"].items() if p.get("lat") is not None]
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = net.route(ends(a), ends(b))
            out.append((a, b, max(0.1, round(r.minutes, 1)), r.meters))
    return out


def pack_yaml(pack: dict, edges: list[tuple[str, str, float, float]], report: dict, old_edges: list) -> str:
    def inline(v) -> str:
        return yaml.safe_dump(v, default_flow_style=True, allow_unicode=True, width=10_000, sort_keys=False).strip().removesuffix("\n...").strip()

    lines = [
        "# Place pack (S5): UCF main campus, the demo venue.",
        "# Written by scripts/build_campus_pack.py: do not edit the coordinates or edges by hand, run the script.",
        f"# Coordinates, names and walking times come from OpenStreetMap. Map data {ATTRIBUTION} (ODbL),",
        "# https://www.openstreetmap.org/copyright.",
        "# `network` is the map file with the footpaths: routes follow it (door to door, at 1.35 m/s unless",
        "# `walk_speed_mps` says otherwise). `edges` are the same real walks in minutes, used when the map file is missing.",
        f"name: {pack['name']}",
        f"kind: {pack.get('kind', 'campus')}",
        "network: ucf.map.json",
    ]
    if pack.get("walk_speed_mps"):
        lines.append(f"walk_speed_mps: {pack['walk_speed_mps']}")
    if pack.get("entry"):
        lines.append(f"entry: {pack['entry']}")
    lines.append("places:")
    width = max(len(p) for p in pack["places"]) + 1
    for pid, p in pack["places"].items():
        r = report[pid]
        note = f"OSM {r['osm']}" + (f", inside {r['inside']}" if r.get("inside") else "") if r["snapped"] else f"NOT SNAPPED: {r['why']}"
        lines.append(f"  {(pid + ':').ljust(width)} {inline(p)}  # {note}")
    lines.append("edges:")
    for a, b, minutes, meters in edges:
        lines.append(f"  - [{a}, {b}, {minutes:g}]  # {round(meters)} m")
    if not edges:
        lines += [f"  - {inline(list(e))}" for e in old_edges]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, help="folder for the raw Overpass responses (reused when present; never committed)")
    ap.add_argument("--pack", type=Path, default=ROOT / "config/places/ucf.yaml")
    ap.add_argument("--map", type=Path, default=ROOT / "config/places/ucf.map.json")
    ap.add_argument("--check", action="store_true", help="print the numbers, write nothing")
    args = ap.parse_args()

    geometry = fetch(GEOMETRY_QUERY, args.cache, "ucf-overpass.json")
    poi = fetch(POI_QUERY, args.cache, "ucf-poi.json")
    old = yaml.safe_load(args.pack.read_text(encoding="utf-8"))
    old_edges = [tuple(e) for e in old.get("edges", [])]
    data, pack, report = build(geometry, poi, old)
    edges = real_edges(data, pack, pack.get("walk_speed_mps"))
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    out = pack_yaml(pack, edges, report["places"], old_edges)

    print(f"map: {len(text.encode()) / 1024:.0f} KB  {report['stats']}")
    for pid, r in report["places"].items():
        print(f"  {pid:12} {r}")
    was = {frozenset(e[:2]): e[2] for e in old_edges}
    for a, b, minutes, meters in edges:
        if frozenset((a, b)) in was:
            print(f"  {a:>12} - {b:12} was {was[frozenset((a, b))]:g} min, now {minutes:g} min ({round(meters)} m)")
    if not args.check:
        args.map.write_text(text + "\n", encoding="utf-8")
        args.pack.write_text(out, encoding="utf-8")
        print(f"wrote {args.map} and {args.pack}")


if __name__ == "__main__":
    main()
