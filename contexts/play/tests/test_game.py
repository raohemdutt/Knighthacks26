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


def drop(at, zone, label="bottle"):
    return ZoneEvent(ts=T0 + timedelta(seconds=at), camera_id="cam-door", track_id=9, label=label, zone_id=zone, kind="enter")


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
    [scored] = g.on_zone_event(drop(3, "recycling", "can"))
    assert isinstance(scored, RoundScored) and scored.correct and scored.score == 150.0
    g.on_faces(faces(4))
    [start] = g.on_faces(faces(6))
    assert start.item == "cup"
    [miss] = g.on_zone_event(drop(8, "recycling", "cup"))
    assert not miss.correct and miss.score == 0


def test_walking_away_abandons_the_round_and_desk_zones_are_ignored():
    g = game()
    g.on_faces(faces(0))
    g.on_faces(faces(2))
    assert g.on_zone_event(drop(3, "door")) == []
    walker = drop(3, "trash").model_copy(update={"label": "person"})
    assert g.on_zone_event(walker) == []  # people passing a bin zone don't score
    g.on_faces(faces(13, facing=0))
    assert g.round is None


def test_accuracy_first_round_vs_fifth():
    g = game(items=("cup",))
    t = 0
    for zone in ("recycling", "trash", "trash", "trash", "trash"):
        g.on_faces(faces(t))
        g.on_faces(faces(t + 2))
        g.on_zone_event(drop(t + 3, zone, "cup"))
        t += 5
    assert g.accuracy(g.session_id) == (0.0, 1.0)


def test_only_the_rounds_item_scores():
    g = game(items=("can",))
    g.on_faces(faces(0))
    [start] = g.on_faces(faces(2))
    assert start.item == "can"
    other = drop(3, "recycling").model_copy(update={"label": "cup"})
    assert g.on_zone_event(other) == []  # a different sortable item doesn't end the round
    alias = drop(4, "recycling").model_copy(update={"label": "soda can"})
    [scored] = g.on_zone_event(alias)
    assert scored.correct


def test_scored_rounds_rebuild_from_the_log_after_a_restart():
    """Replaying play.round.scored into a new game restores the history without starting or scoring anything."""
    g = game(items=("cup",))
    log = []
    t = 0
    for zone in ("recycling", "trash", "trash", "trash", "trash"):
        g.on_faces(faces(t))
        g.on_faces(faces(t + 2))
        log += g.on_zone_event(drop(t + 3, zone, "cup"))
        t += 5
    for scored in log:  # live, the game also hears its own events: they are already counted
        assert g.on_scored(scored) == []
    assert g.results[g.session_id] == [False, True, True, True, True]

    restarted = game(items=("cup",))
    assert restarted.accuracy(restarted.session_id) is None
    for scored in log:
        assert restarted.on_scored(scored) == []
    assert restarted.session_id == g.session_id and restarted.round is None
    assert restarted.results == g.results and restarted.accuracy(restarted.session_id) == (0.0, 1.0)
    for scored in log:  # a second replay of the same log does not double anything
        restarted.on_scored(scored)
    assert restarted.results == g.results

    restarted.on_faces(faces(t + 60))  # still the same sitting: the session carries on
    assert restarted.session_id == g.session_id
    restarted.on_faces(faces(t + 600))  # back much later: a new session, on event time
    assert restarted.session_id != g.session_id
