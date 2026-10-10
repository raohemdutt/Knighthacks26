import pytest

from contexts.connectors.servers.places import distance_m, nearby_places

ELEMENTS = [
    {"type": "node", "id": 1, "lat": 28.6030, "lon": -81.2001, "tags": {"name": "Knightro's", "amenity": "restaurant", "cuisine": "pizza;dessert"}},
    {"type": "way", "id": 2, "center": {"lat": 28.6100, "lon": -81.2001}, "tags": {"name": "Subway", "amenity": "fast_food", "cuisine": "sandwich"}},
    {"type": "node", "id": 3, "lat": 28.6025, "lon": -81.2001, "tags": {"name": "Subway", "amenity": "fast_food"}},
    {"type": "node", "id": 4, "lat": 28.6, "lon": -81.2, "tags": {"amenity": "cafe"}},  # no name: skipped
]
OPTIONS = {"lat": 28.6024, "lon": -81.2001}


def test_distance():
    assert 60 < distance_m(28.6024, -81.2001, 28.6030, -81.2001) < 75


def test_closest_first_dedupes_chains_and_caches(tmp_path):
    calls = []

    def post(url, query):
        calls.append(url)
        return ELEMENTS

    opts = {**OPTIONS, "cache_dir": str(tmp_path), "endpoints": ["https://a.test", "https://b.test"]}
    places = nearby_places(opts, "food", 1500, 5, post, now=1000.0)
    assert [(p["title"], p["cuisine"]) for p in places] == [("Subway", "fast food"), ("Knightro's", "pizza, dessert")]
    assert places[0]["url"] == "https://www.openstreetmap.org/node/3" and "m away" in places[0]["summary"]
    nearby_places(opts, "food", 1500, 5, post, now=2000.0)
    assert calls == ["https://a.test"]  # second call served from the day cache


def test_busy_servers_fall_back_to_the_next_then_to_stale_cache(tmp_path):
    def busy(url, query):
        raise RuntimeError("504")

    opts = {**OPTIONS, "cache_dir": str(tmp_path), "endpoints": ["https://a.test", "https://b.test"]}
    with pytest.raises(RuntimeError, match="nothing is cached"):
        nearby_places(opts, "food", 1500, 5, busy, now=0.0)
    nearby_places(opts, "food", 1500, 5, lambda u, q: ELEMENTS if u == "https://b.test" else busy(u, q), now=0.0)
    stale = nearby_places(opts, "food", 1500, 5, busy, now=10 * 24 * 3600.0)  # a week later, every server down
    assert stale and stale[0]["title"] == "Subway"
