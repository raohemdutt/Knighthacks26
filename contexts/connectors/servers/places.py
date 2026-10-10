"""Nearby places (K8) as an MCP server: python -m contexts.connectors.servers.places

`nearby(kind, radius_m)` returns restaurants, cafes, fast food and dining halls
around the box's fixed location (a setting, never live GPS) with name,
distance and cuisine, from OpenStreetMap through the Overpass API.

Access confirmed 2026-10-10: the public Overpass servers need no key but are
busy (504/500s), so the server tries several endpoints, sends a clear
User-Agent, and caches results for a day. Places barely change, so a cached
answer is fine and keeps Atlas useful offline.

options:
  lat: 28.6024          # the box's fixed location (default: UCF main campus)
  lon: -81.2001
  cache_dir: data/connectors/places
  endpoints: [https://overpass-api.de/api/interpreter, https://overpass.kumi.systems/api/interpreter]
"""

import hashlib
import json
import math
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

KINDS = {
    "food": "restaurant|fast_food|cafe|food_court|ice_cream",
    "coffee": "cafe",
    "dining_hall": "food_court|canteen",
}
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter"]
CACHE_TTL_S = 24 * 3600
USER_AGENT = "Atlas/0.1 (Knight Hacks project; nearby places, cached daily)"


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)))


def overpass_query(kind: str, lat: float, lon: float, radius_m: int, limit: int) -> str:
    amenities = KINDS.get(kind, KINDS["food"])
    return (f'[out:json][timeout:20];(nwr["amenity"~"^({amenities})$"]["name"](around:{radius_m},{lat},{lon}););'
            f"out center tags {limit * 3};")


def to_places(elements: list[dict], lat: float, lon: float, limit: int) -> list[dict]:
    out = {}
    for e in elements:
        tags = e.get("tags", {})
        where = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
        if not tags.get("name") or where.get("lat") is None:
            continue
        d = distance_m(lat, lon, where["lat"], where["lon"])
        key = tags["name"].lower()
        if key in out and out[key]["distance_m"] <= d:
            continue  # the same chain twice: keep the closer one
        cuisine = (tags.get("cuisine") or tags["amenity"]).replace(";", ", ").replace("_", " ")
        out[key] = {
            "external_id": f"osm:{e.get('type', 'node')}/{e.get('id')}",
            "kind": "place",
            "title": tags["name"],
            "distance_m": d,
            "cuisine": cuisine,
            "summary": f"{cuisine}, {d} m away" + (f", open {tags['opening_hours']}" if tags.get("opening_hours") else ""),
            "url": f"https://www.openstreetmap.org/{e.get('type', 'node')}/{e.get('id')}",
        }
    return sorted(out.values(), key=lambda p: p["distance_m"])[:limit]


def nearby_places(options: dict, kind: str, radius_m: int, limit: int, post, now: float | None = None) -> list[dict]:
    lat, lon = float(options.get("lat", 28.6024)), float(options.get("lon", -81.2001))
    query = overpass_query(kind, lat, lon, radius_m, limit)
    cache = Path(options.get("cache_dir", "data/connectors/places")) / (hashlib.sha256(query.encode()).hexdigest()[:16] + ".json")
    now = now or time.time()
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached and now - cached["at"] < CACHE_TTL_S:
        return to_places(cached["elements"], lat, lon, limit)
    for url in options.get("endpoints", ENDPOINTS):
        try:
            elements = post(url, query)
        except Exception:
            continue  # busy mirror: try the next
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"at": now, "elements": elements}))
        return to_places(elements, lat, lon, limit)
    if cached:  # every server busy: a day-old list of restaurants is still right
        return to_places(cached["elements"], lat, lon, limit)
    raise RuntimeError("no Overpass server answered and nothing is cached yet")


def _post(url: str, query: str) -> list[dict]:
    r = httpx.post(url, data={"data": query}, timeout=40, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()["elements"]


mcp = FastMCP("atlas-places")


@mcp.tool()
def nearby(kind: str = "food", radius_m: int = 1500, limit: int = 8) -> list[dict]:
    """Places to eat near the box (kind: food, coffee, dining_hall), closest first, with distance and cuisine."""
    return nearby_places(server_options(), kind, radius_m, limit, _post)


if __name__ == "__main__":
    mcp.run()
