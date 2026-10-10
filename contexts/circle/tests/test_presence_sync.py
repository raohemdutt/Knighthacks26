from datetime import UTC, datetime, timedelta

from contexts.circle.application.presence_sync import InMemoryPresenceRemote, PresenceSync
from contexts.circle.application.sharing import CircleOffline
from contracts.circle import PlaceShared

NOW = datetime(2026, 10, 10, 23, tzinfo=UTC)


def share(member, place, at=0, hours=2):
    ts = NOW + timedelta(minutes=at)
    return PlaceShared(ts=ts, circle_id="team", member_id=member, place=place, source="phone", expires_at=ts + timedelta(hours=hours))


def test_my_place_goes_up_and_friends_places_come_back_once(tmp_path):
    remote = InMemoryPresenceRemote()
    dom = PresenceSync(remote, "team", "dom")
    hem = PresenceSync(remote, "team", "hemdutt")

    dom.on_shared(share("dom", "Library"))
    hem.on_shared(share("hemdutt", "Our table"))
    hem.on_shared(share("dom", "Dorm"))  # not hemdutt's own: never pushed by hemdutt's box
    assert dom.sync(NOW) == []  # dom sees no friends yet
    [seen] = hem.sync(NOW + timedelta(minutes=1))
    assert (seen.member_id, seen.place) == ("dom", "Library")
    assert hem.sync(NOW + timedelta(minutes=2)) == []  # no duplicates
    assert remote.rows["dom"]["place"] == "Library"


def test_expired_places_disappear_and_offline_waits(tmp_path):
    remote = InMemoryPresenceRemote()
    dom = PresenceSync(remote, "team", "dom")
    hem = PresenceSync(remote, "team", "hemdutt")
    dom.on_shared(share("dom", "Library", hours=1))
    dom.sync(NOW)
    assert hem.sync(NOW + timedelta(hours=2)) == []

    class Down(InMemoryPresenceRemote):
        def push(self, row):
            raise CircleOffline("no network")

    offline = PresenceSync(Down(), "team", "dom")
    offline.on_shared(share("dom", "Gym"))
    assert offline.sync(NOW) == []
    assert offline.pending is not None  # kept for the next tick
