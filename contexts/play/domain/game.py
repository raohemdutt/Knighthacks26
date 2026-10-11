"""Sort It game loop (S1).

    faces held 2 s -> round starts: an item to sort, the route to the right bin
    item enters a bin zone -> round scored: correct bin + time against the route
    no faces for 10 s -> round abandoned

Pure state machine on event time, so replays score the same. Score is 100 for
the right bin plus up to 50 for beating the route's time estimate.

`state()` says where the loop is right now, in the game's own terms, for the screen's face chip: looking for a
player, a face held for so long, a round on with so many players, or nobody facing the camera and the round about
to end. Counts and seconds only; a face is never more than a number here.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from uuid import uuid4

from contexts.play.domain.pathfinding import SortingRules, ZoneMap, route
from contracts.perception import ZoneEvent
from contracts.play import FacesCounted, RoundScored, RoundStarted

HOLD_TO_START = timedelta(seconds=2)
IDLE_TO_END = timedelta(seconds=10)
NEW_SESSION_AFTER = timedelta(minutes=2)
SECONDS_PER_CELL = 0.6  # walking pace on the grid, for the time estimate


class SortIt:
    def __init__(self, zone_map: ZoneMap, rules: SortingRules, items: list[str] | None = None,
                 start_xy: tuple[float, float] = (0.75, 0.75)) -> None:
        self.map = zone_map
        self.rules = rules
        self.items = items or list(rules.items)
        self.start_xy = start_xy
        self.bin_zones = set(rules.bins.values())
        self.round: RoundStarted | None = None
        self.session_id = uuid4().hex
        self._faces_since: datetime | None = None
        self._last_face: datetime | None = None
        self._turn = 0
        self.results: dict[str, list[bool]] = defaultdict(list)  # session -> correct per round
        self._scored: set[str] = set()  # round ids already in `results`
        self.history: dict[str, list[dict]] = defaultdict(list)  # session -> each scored round: seconds, estimate, score
        self._heard: datetime | None = None  # the newest face count's time
        self._facing = 0  # ... and how many faces it had looking at the camera
        self._count = 0
        self.pick_item = None  # optional () -> item | None: what a camera sees in the player's hand (room twin)

    def on_faces(self, e: FacesCounted) -> list:
        self._heard, self._facing, self._count = e.ts, e.facing, e.count
        if e.facing > 0:
            if self._last_face and e.ts - self._last_face > NEW_SESSION_AFTER:
                self.session_id = uuid4().hex
            self._last_face = e.ts
            self._faces_since = self._faces_since or e.ts
            if self.round is None and e.ts - self._faces_since >= HOLD_TO_START:
                return [self._start(e.ts, players=e.facing)]
            return []
        self._faces_since = None
        if self.round and self._last_face and e.ts - self._last_face >= IDLE_TO_END:
            self.round = None  # abandoned: nobody is playing
        return []

    def _start(self, now: datetime, players: int) -> RoundStarted:
        item = (self.pick_item() if self.pick_item else None) or self.items[self._turn % len(self.items)]
        self._turn += 1
        zone, path = route(self.map, self.rules, item, self.start_xy)
        self.round = RoundStarted(ts=now, round_id=uuid4().hex, session_id=self.session_id,
                                  players=players, item=item, target_bin=zone, path=path)
        return self.round

    def on_zone_event(self, e: ZoneEvent) -> list:
        r = self.round
        if r is None or e.kind != "enter" or e.zone_id not in self.bin_zones or not self.rules.same_item(e.label, r.item):
            return []  # only the round's item scores: a person or another object passing a bin zone doesn't
        correct = e.zone_id == r.target_bin
        seconds = (e.ts - r.ts).total_seconds()
        estimate = max(len(r.path), 1) * SECONDS_PER_CELL
        speed = max(0.0, min(1.0, 2 - seconds / estimate))  # 1.0 at or under the estimate, 0 at twice it
        self.round = None
        self._faces_since = e.ts  # hold the camera's gaze again for the next round
        self._scored.add(r.round_id)
        self.results[r.session_id].append(correct)
        scored = RoundScored(ts=e.ts, round_id=r.round_id, session_id=r.session_id, players=r.players, item=r.item,
                             correct=correct, seconds=seconds, path_len=len(r.path),
                             score=round((100 if correct else 0) + 50 * speed * correct, 1))
        self._remember(scored)
        return [scored]

    def _remember(self, e: RoundScored) -> None:
        """One scored round for the "round 1 against the latest" line: its time, the route's estimate, its score."""
        self.history[e.session_id].append({"round": len(self.history[e.session_id]) + 1, "item": e.item, "correct": e.correct,
                                           "seconds": round(e.seconds, 1), "estimate_s": round(max(e.path_len, 1) * SECONDS_PER_CELL, 1),
                                           "score": e.score, "players": e.players, "ts": e.ts.isoformat()})

    def on_scored(self, e: RoundScored) -> list:
        """Rebuild the history from the log (replay): rounds scored before a restart count again.

        State only: it starts no round and returns no events. A round this run scored itself is
        already counted, so the live copy of its own event changes nothing."""
        if e.round_id in self._scored:
            return []
        self._scored.add(e.round_id)
        self.results[e.session_id].append(e.correct)
        self._remember(e)
        self.session_id = e.session_id  # the latest session stays the one on screen until someone plays again
        self._last_face = e.ts  # event time: a face more than NEW_SESSION_AFTER later starts a new session
        return []

    def accuracy(self, session_id: str, first: int = 1, last: int = 5) -> tuple[float, float] | None:
        """Accuracy over the first `first` rounds vs round `last` onward: did the player learn?"""
        rounds = self.results.get(session_id, [])
        if len(rounds) < last:
            return None
        early, late = rounds[:first], rounds[last - 1:]
        return sum(early) / len(early), sum(late) / len(late)

    def state(self) -> dict:
        """GET /api/play/state: where the loop is, as of the newest face count (event time).

        phase: "looking" (nobody facing the camera), "holding" (a face held, `held_s` of `hold_to_start_s`),
        "round" (a round is on and someone is facing the camera), "ending" (a round is on, nobody has faced the
        camera for `idle_s` of `idle_to_end_s`). `rounds` is this sitting's scored rounds: the first and the latest,
        when there are two or more to compare."""
        now, r = self._heard, self.round
        facing = self._facing
        if r is not None:
            phase = "round" if facing > 0 or self._last_face is None else "ending"
        else:
            phase = "holding" if facing > 0 and self._faces_since is not None else "looking"
        held = (now - self._faces_since).total_seconds() if phase == "holding" and now else 0.0
        idle = (now - self._last_face).total_seconds() if phase == "ending" and now and self._last_face else 0.0
        played = self.history.get(self.session_id, [])
        return {"phase": phase, "faces": facing, "count": self._count, "heard": now.isoformat() if now else None,
                "held_s": round(max(0.0, min(held, HOLD_TO_START.total_seconds())), 2), "hold_to_start_s": HOLD_TO_START.total_seconds(),
                "idle_s": round(max(0.0, idle), 2), "idle_to_end_s": IDLE_TO_END.total_seconds(),
                "round": None if r is None else {"round_id": r.round_id, "item": r.item, "players": r.players, "started": r.ts.isoformat()},
                "rounds": {"session_id": self.session_id, "played": len(played), "first": played[0] if len(played) > 1 else None,
                           "latest": played[-1] if len(played) > 1 else None}}
