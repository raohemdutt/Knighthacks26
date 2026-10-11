"""Bake the real UCF campus (config/places/ucf.map.json, OpenStreetMap) into the film's campus map, in the website's
map palette (web/components/dayrun/campusmap.module.css), and a Day Run route along the real footpaths.

    python video/scripts/campus_bake.py

Writes video/hf/assets/ucf-campus.png (the map plane's texture, square, north up) and video/hf/assets/ucf-route.json
({"stops": [{name, u, v}], "route": [[u, v], ...]} with u east and v south in 0..1 of the square). The world maps
u, v onto its 30 x 30 campus plane. Map data (c) OpenStreetMap contributors, ODbL.
"""

import heapq
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "config" / "places" / "ucf.map.json"
OUT = ROOT / "video" / "hf" / "assets"
PX, SS = 2048, 3  # output size, supersampling
M_PER_DEG = 111_320.0

# The website's map colours (campusmap.module.css).
GROUND, BUILDING, BUILDING_LINE = (241, 244, 249), (213, 220, 232), (174, 184, 200)
GREEN, WATER, PARKING, ROAD_CASING, ROAD, PATH = (220, 235, 219), (188, 215, 239), (230, 234, 241), (214, 221, 232), (255, 255, 255), (199, 207, 219)
PIN = {"home": (92, 225, 198), "class": (201, 165, 255), "study": (127, 178, 255), "eat": (242, 166, 90)}

# Day Run's day: from the dorm to class, the library, then lunch at the Union.
STOPS = [("towers", "home"), ("cb1", "class"), ("library", "study"), ("union", "eat")]


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    south, west, north, east = d["bbox"]
    lat0, lon0 = d["origin"]
    unit = d["unit"]
    kx = M_PER_DEG * math.cos(math.radians((south + north) / 2))
    w_m, h_m = (east - west) * kx, (north - south) * M_PER_DEG
    side = max(w_m, h_m)
    ox, oy = (side - w_m) / 2, (side - h_m) / 2

    def uv(dlat: int, dlon: int) -> tuple[float, float]:
        lat, lon = lat0 + dlat * unit, lon0 + dlon * unit
        return ((lon - west) * kx + ox) / side, ((north - lat) * M_PER_DEG + oy) / side

    def pts(flat: list[int]) -> list[tuple[float, float]]:
        return [uv(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]

    S = PX * SS
    img = Image.new("RGB", (S, S), GROUND)
    g = ImageDraw.Draw(img)
    px = lambda p: [(u * S, v * S) for u, v in p]
    m = S / side  # pixels per metre

    for a in d["areas"]:
        col = {"water": WATER, "pool": WATER, "parking": PARKING, "plaza": PARKING}.get(a["k"], GREEN)
        p = pts(a["p"])
        if len(p) >= 3:
            g.polygon(px(p), fill=col)
    widths = {"major": 14, "minor": 9, "service": 6}
    for casing in (True, False):
        for r in d["roads"]:
            p = px(pts(r["p"]))
            wm = widths.get(r["k"], 6)
            if len(p) >= 2:
                g.line(p, fill=ROAD_CASING if casing else ROAD, width=max(1, int((wm + (4 if casing else 0)) * m)), joint="curve")
    nodes = d["walk"]["nodes"]
    npt = [uv(nodes[i], nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
    for a, b, *_ in d["walk"]["edges"]:
        g.line(px([npt[a], npt[b]]), fill=PATH, width=max(1, int(2.2 * m)))
    for b in d["buildings"]:
        p = pts(b["p"])
        if len(p) >= 3:
            g.polygon(px(p), fill=BUILDING, outline=BUILDING_LINE, width=max(1, int(1.6 * m)))

    # The route: shortest walks between the stops on the footpath graph.
    adj: dict[int, list[tuple[int, float]]] = {}
    for a, b, *_ in d["walk"]["edges"]:
        dd = math.dist(npt[a], npt[b])
        adj.setdefault(a, []).append((b, dd))
        adj.setdefault(b, []).append((a, dd))
    graph_nodes = list(adj)

    def nearest(p: tuple[float, float]) -> int:
        return min(graph_nodes, key=lambda i: math.dist(npt[i], p))

    def walk(s: int, t: int) -> list[int]:
        dist, prev, q = {s: 0.0}, {}, [(0.0, s)]
        while q:
            du, u = heapq.heappop(q)
            if u == t:
                break
            if du > dist.get(u, 1e9):
                continue
            for v, w in adj[u]:
                if du + w < dist.get(v, 1e9):
                    dist[v], prev[v] = du + w, u
                    heapq.heappush(q, (du + w, v))
        path = [t]
        while path[-1] != s:
            path.append(prev[path[-1]])
        return path[::-1]

    stops = [(key, kind, uv(*d["places"][key]["c"])) for key, kind in STOPS]
    route: list[tuple[float, float]] = []
    for (_, _, a), (_, _, b) in zip(stops, stops[1:]):
        seg = [npt[i] for i in walk(nearest(a), nearest(b))]
        route += seg if not route else seg[1:]
    # thin the polyline: drop points closer than ~6 m to the last kept one
    thin = [route[0]]
    for p in route[1:]:
        if math.dist(p, thin[-1]) * side > 6:
            thin.append(p)
    for key, kind, (u, v) in stops:
        r = 13 * SS
        g.ellipse([u * S - r - 6 * SS, v * S - r - 6 * SS, u * S + r + 6 * SS, v * S + r + 6 * SS], fill=(255, 255, 255))
        g.ellipse([u * S - r, v * S - r, u * S + r, v * S + r], fill=PIN[kind])

    OUT.mkdir(parents=True, exist_ok=True)
    # Fade the square's edges out so the map floats on the film's surface instead of ending in a hard border.
    out_img = img.resize((PX, PX), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (PX, PX), 0)
    mp = mask.load()
    for y in range(PX):
        for x in range(PX):
            r = math.hypot(x / PX - 0.5, y / PX - 0.5)
            mp[x, y] = int(255 * max(0.0, min(1.0, (0.5 - r) / 0.14)))
    out_img.putalpha(mask)
    out_img.save(OUT / "ucf-campus.png", optimize=True)
    out = {"source": "config/places/ucf.map.json (c) OpenStreetMap contributors, ODbL", "side_m": round(side),
           "stops": [{"name": d["places"][k]["name"], "kind": kind, "u": round(u, 4), "v": round(v, 4)} for k, kind, (u, v) in stops],
           "route": [[round(u, 4), round(v, 4)] for u, v in thin]}
    (OUT / "ucf-route.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"campus {side:.0f} m square, route {len(thin)} points, {sum(math.dist(a, b) for a, b in zip(thin, thin[1:])) * side:.0f} m")


if __name__ == "__main__":
    main()
