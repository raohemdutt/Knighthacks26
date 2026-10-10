"""Day Run's view of the place packs (S5): whole-minute walks with their legs.

Everything goes through `contexts.play.domain.places.PlaceSet` (pack loading, validation,
nesting and A*). This only adds what the simulation needs on top: a walk rounded up to whole
minutes, the route cut into legs with coordinates for the map, and lookups by kind.
"""

import math

from pydantic import BaseModel, Field

from contexts.play.domain.places import Place, PlaceSet


class Leg(BaseModel):
    """One edge walked: what the map draws. Coordinates are (lat, lon), or None indoors."""

    a: str = Field(serialization_alias="from")
    b: str = Field(serialization_alias="to")
    from_name: str
    to_name: str
    minutes: float
    from_xy: tuple[float, float] | None = None
    to_xy: tuple[float, float] | None = None


class Trip(BaseModel):
    places: list[str]
    minutes: float
    legs: list[Leg]


class Walks:
    def __init__(self, places: PlaceSet) -> None:
        self.places = places

    def ref(self, ref: str) -> str:
        """The canonical "pack.place"; ValueError when there is no such place."""
        pack, _, place = ref.partition(".")
        found = self.places.packs[pack].find(place) if pack in self.places.packs and place else None
        if found is None:
            raise ValueError(f"unknown place {ref!r} (use pack.place, e.g. ucf.library)")
        return f"{pack}.{found}"

    def get(self, ref: str) -> Place:
        pack, place = self.ref(ref).split(".", 1)
        return self.places.packs[pack].places[place]

    def name(self, ref: str) -> str:
        return self.get(ref).name

    def kind(self, ref: str) -> str:
        return self.get(ref).kind

    def of_kind(self, kinds: list[str], packs: list[str]) -> list[str]:
        return [f"{pid}.{place}" for pid in packs if pid in self.places.packs
                for place, p in self.places.packs[pid].places.items() if p.kind in kinds]

    def locate(self, text: str | None) -> str | None:
        return self.places.locate(text)

    def shared(self, place_name: str) -> str | None:
        return self.places.find_shared(place_name)

    def trip(self, a: str, b: str) -> Trip:
        path, minutes = self.places.route(a, b)
        legs = []
        for x, y in zip(path, path[1:]):
            px, py = self.get(x), self.get(y)
            legs.append(Leg(a=x, b=y, from_name=px.name, to_name=py.name, minutes=self._edge(x, y),
                            from_xy=(px.lat, px.lon) if px.lat is not None and px.lon is not None else None,
                            to_xy=(py.lat, py.lon) if py.lat is not None and py.lon is not None else None))
        return Trip(places=path, minutes=minutes, legs=legs)

    def _edge(self, x: str, y: str) -> float:
        (pack_x, a), (pack_y, b) = x.split(".", 1), y.split(".", 1)
        if pack_x != pack_y:
            return 0.0  # stepping through the door between a pack and the one nested in it
        return next(m for p, q, m in self.places.packs[pack_x].edges if {p, q} == {a, b})

    def walk(self, a: str | None, b: str | None) -> tuple[int, Trip | None]:
        """Whole minutes from a to b, rounded up, and the trip; (0, None) when there is nowhere to go."""
        if not a or not b or self.ref(a) == self.ref(b):
            return 0, None
        trip = self.trip(a, b)
        return max(1, math.ceil(trip.minutes - 1e-9)), trip
