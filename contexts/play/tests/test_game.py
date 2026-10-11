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


# --- where the loop is, for the screen's face chip ---


def test_state_follows_the_loop_looking_holding_round_ending():
    g = game()
    s = g.state()
    assert (s["phase"], s["faces"], s["heard"], s["round"]) == ("looking", 0, None, None)
    assert (s["hold_to_start_s"], s["idle_to_end_s"]) == (2.0, 10.0)
    g.on_faces(faces(0, facing=2))
    g.on_faces(faces(1.2, facing=2))
    s = g.state()
    assert (s["phase"], s["faces"], s["count"], s["held_s"]) == ("holding", 2, 2, 1.2)
    [start] = g.on_faces(faces(2, facing=2))
    s = g.state()
    assert s["phase"] == "round" and s["round"] == {"round_id": start.round_id, "item": "can", "players": 2, "started": start.ts.isoformat()}
    g.on_faces(faces(3, facing=2))
    g.on_faces(faces(10, facing=0))
    s = g.state()
    assert (s["phase"], s["faces"], s["idle_s"], s["round"]["players"]) == ("ending", 0, 7.0, 2)  # the players are the faces that started it
    g.on_faces(faces(13, facing=0))  # ten seconds with nobody: abandoned
    s = g.state()
    assert (s["phase"], s["round"], s["held_s"], s["idle_s"]) == ("looking", None, 0.0, 0.0)
    g.on_faces(faces(14))
    g.on_faces(faces(14.5, facing=0))  # looked away: the hold starts over
    assert g.state()["phase"] == "looking"
    g.on_faces(faces(15))
    assert (g.state()["phase"], g.state()["held_s"]) == ("holding", 0.0)


def test_state_compares_the_first_and_the_latest_round_only_when_both_were_played():
    g = game(items=("cup",))
    assert g.state()["rounds"] == {"session_id": g.session_id, "played": 0, "first": None, "latest": None}
    log, t = [], 0
    for took in (9, 3):
        [start] = g.on_faces(faces(t)) + g.on_faces(faces(t + 2))  # the second round starts at once: the gaze was held since the drop
        log += g.on_zone_event(drop(t + 2 + took, "trash", "cup"))
        if t == 0:
            assert (g.state()["rounds"]["played"], g.state()["rounds"]["first"]) == (1, None)  # one round: nothing to compare
        t += 20
    rounds = g.state()["rounds"]
    estimate = round(len(start.path) * 0.6, 1)
    assert rounds["played"] == 2
    assert rounds["first"] == {"round": 1, "item": "cup", "correct": True, "seconds": 9.0, "estimate_s": estimate, "score": log[0].score,
                               "players": 1, "ts": log[0].ts.isoformat()}
    assert (rounds["latest"]["round"], rounds["latest"]["seconds"], rounds["latest"]["score"]) == (2, 5.0, log[1].score)
    for scored in log:  # the game hears its own events: not counted twice
        g.on_scored(scored)
    assert g.state()["rounds"] == rounds
    restarted = game(items=("cup",))
    for scored in log:
        restarted.on_scored(scored)
    assert restarted.state()["rounds"] == rounds  # the same numbers from the log after a restart
    assert restarted.state()["phase"] == "looking"
