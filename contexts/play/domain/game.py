"""Sort It game loop (S1).

    faces held 2 s -> round starts: an item to sort, the route to the right bin
    item enters a bin zone -> round scored: correct bin + time against the route
    no faces for 10 s -> round abandoned

Pure state machine on event time, so replays score the same. Score is 100 for
the right bin plus up to 50 for beating the route's time estimate.
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

    def on_faces(self, e: FacesCounted) -> list:
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
        item = self.items[self._turn % len(self.items)]
        self._turn += 1
        zone, path = route(self.map, self.rules, item, self.start_xy)
        self.round = RoundStarted(ts=now, round_id=uuid4().hex, session_id=self.session_id,
                                  players=players, item=item, target_bin=zone, path=path)
        return self.round

    def on_zone_event(self, e: ZoneEvent) -> list:
        r = self.round
        if r is None or e.kind != "enter" or e.zone_id not in self.bin_zones or e.label not in self.rules.items:
            return []  # only sortable items score: a person walking through a bin zone doesn't
        correct = e.zone_id == r.target_bin
        seconds = (e.ts - r.ts).total_seconds()
        estimate = max(len(r.path), 1) * SECONDS_PER_CELL
        speed = max(0.0, min(1.0, 2 - seconds / estimate))  # 1.0 at or under the estimate, 0 at twice it
        self.round = None
        self._faces_since = e.ts  # hold the camera's gaze again for the next round
        self.results[r.session_id].append(correct)
        return [RoundScored(ts=e.ts, round_id=r.round_id, session_id=r.session_id, players=r.players, item=r.item,
                            correct=correct, seconds=seconds, path_len=len(r.path),
                            score=round((100 if correct else 0) + 50 * speed * correct, 1))]

    def accuracy(self, session_id: str, first: int = 1, last: int = 5) -> tuple[float, float] | None:
        """Accuracy over the first `first` rounds vs round `last` onward: did the player learn?"""
        rounds = self.results.get(session_id, [])
        if len(rounds) < last:
            return None
        early, late = rounds[:first], rounds[last - 1:]
        return sum(early) / len(early), sum(late) / len(late)
