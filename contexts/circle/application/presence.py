"""Opt-in friend presence (G5).

A member can share a coarse place (a named spot like "Library", "Dorm", "Our
table"), never coordinates or video, and it expires. Shares come from the
phone (an explicit tap) or from desk presence (the camera sees someone at our
table), but only for members who opted in. Places outside the household's
list are refused, so nothing finer-grained can slip through.

Shares are events (circle.place.shared). The Neon `presence` table is where
they go for friends' boxes and phones (N1 outbox); PresenceBoard is the
local view the planner reads for "where should I go?".
"""

from datetime import datetime, timedelta

from contracts.circle import PlaceShared
from contracts.perception import ZoneEvent

DESK_TTL = timedelta(hours=1)
DESK_PLACE = "Our table"


class PresenceBoard:
    def __init__(self, circle_id: str, me: str, places: list[str], opted_in: bool, desk_zone: str = "desk") -> None:
        self.circle_id = circle_id
        self.me = me
        self.places = places
        self.opted_in = opted_in
        self.desk_zone = desk_zone
        self.current: dict[str, PlaceShared] = {}

    def share(self, place: str, source: str, now: datetime, ttl: timedelta = timedelta(hours=2)) -> PlaceShared:
        if not self.opted_in:
            raise PermissionError("presence sharing is off for this member (config: circle.share_presence)")
        if place not in self.places:
            raise ValueError(f"{place!r} is not one of the named places: {', '.join(self.places)}")
        return PlaceShared(ts=now, circle_id=self.circle_id, member_id=self.me, place=place, source=source,
                           expires_at=now + ttl)

    def on_shared(self, e: PlaceShared) -> None:
        """Bus handler (replay-safe): mine and friends' shares, latest wins per member."""
        prev = self.current.get(e.member_id)
        if prev is None or e.ts >= prev.ts:
            self.current[e.member_id] = e

    def on_zone_event(self, e: ZoneEvent) -> list[PlaceShared]:
        """Someone sat down at our table: share "Our table" for an hour, if opted in."""
        if not self.opted_in or e.label != "person" or e.zone_id != self.desk_zone or e.kind != "enter":
            return []
        mine = self.current.get(self.me)
        if mine and mine.place == DESK_PLACE and mine.expires_at - e.ts > DESK_TTL / 2:
            return []  # still fresh
        return [self.share(DESK_PLACE, "desk", e.ts, DESK_TTL)]

    def friends(self, now: datetime) -> list[PlaceShared]:
        return sorted(
            (p for m, p in self.current.items() if m != self.me and p.expires_at > now),
            key=lambda p: p.place,
        )
