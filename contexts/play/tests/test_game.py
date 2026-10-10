from datetime import UTC, datetime, timedelta

from contexts.play.domain.game import SortIt
from contexts.play.domain.pathfinding import SortingRules, ZoneMap
from contracts.perception import ZoneEvent
from contracts.play import FacesCounted, RoundScored, RoundStarted

T0 = datetime(2026, 10, 11, 15, tzinfo=UTC)


def game(items=("can", "cup")):
    return SortIt(ZoneMap.load("cam-door"), SortingRules.load(), items=list(items))


def faces(at, facing=1):
    return FacesCounted(ts=T0 + timedelta(seconds=at), camera_id="cam-door", count=facing, facing=facing)


def drop(at, zone):
    return ZoneEvent(ts=T0 + timedelta(seconds=at), camera_id="cam-door", track_id=9, label="bottle", zone_id=zone, kind="enter")


def test_face_held_two_seconds_starts_a_round_with_a_route():
    g = game()
    assert g.on_faces(faces(0)) == []
    [start] = g.on_faces(faces(2))
    assert isinstance(start, RoundStarted) and (start.item, start.target_bin) == ("can", "recycling") and start.path
    assert g.on_faces(faces(3)) == []  # one round at a time


def test_right_bin_fast_scores_150_wrong_bin_scores_0():
    g = game()
    g.on_faces(faces(0))
    g.on_faces(faces(2))
    [scored] = g.on_zone_event(drop(3, "recycling"))
    assert isinstance(scored, RoundScored) and scored.correct and scored.score == 150.0
    g.on_faces(faces(4))
    [start] = g.on_faces(faces(6))
    assert start.item == "cup"
    [miss] = g.on_zone_event(drop(8, "recycling"))
    assert not miss.correct and miss.score == 0


def test_walking_away_abandons_the_round_and_desk_zones_are_ignored():
    g = game()
    g.on_faces(faces(0))
    g.on_faces(faces(2))
    assert g.on_zone_event(drop(3, "door")) == []
    g.on_faces(faces(13, facing=0))
    assert g.round is None


def test_accuracy_first_round_vs_fifth():
    g = game(items=("cup",))
    t = 0
    for zone in ("recycling", "trash", "trash", "trash", "trash"):
        g.on_faces(faces(t))
        g.on_faces(faces(t + 2))
        g.on_zone_event(drop(t + 3, zone))
        t += 5
    assert g.accuracy(g.session_id) == (0.0, 1.0)
