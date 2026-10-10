"""Waste Watch: counting (E1) and impact metrics (E2).

E1: a single-use item tracked on camera is `impact.item.seen` once; when its
track enters a bin zone it is `impact.item.disposed`, correct or not per
config/sorting.yaml. E2: every change updates the counts (items seen,
diversion rate, contamination), split before and after nudges were turned on
so the slide shows measured numbers, not estimates.
"""

from datetime import datetime

from contexts.play.domain.pathfinding import SortingRules
from contracts.impact import CountUpdated, ItemDisposed, ItemSeen
from contracts.perception import ObservationRecorded, ZoneEvent


class WasteWatch:
    def __init__(self, rules: SortingRules, nudges_on_at: datetime | None = None) -> None:
        self.rules = rules
        self.bin_by_zone = {zone: b for b, zone in rules.bins.items()}
        self.nudges_on_at = nudges_on_at
        self.seen: dict[str, ItemSeen] = {}
        self.disposed: dict[str, ItemDisposed] = {}

    def tracked(self, label: str) -> bool:
        return label in self.rules.items

    def recyclable(self, label: str) -> bool:
        return self.rules.bin_for(label) == "recycling"

    @staticmethod
    def item_id(camera_id: str, track_id: int) -> str:
        return f"{camera_id}:{track_id}"

    # --- E1: counting ---

    def on_observation(self, e: ObservationRecorded) -> list:
        out = []
        for t in e.tracks:
            iid = self.item_id(e.camera_id, t.track_id)
            if self.tracked(t.label) and iid not in self.seen:
                seen = ItemSeen(ts=e.ts, item_id=iid, label=t.label, camera_id=e.camera_id,
                                recyclable=self.recyclable(t.label), frame_ref=e.frame_ref)
                self.seen[iid] = seen
                out.append(seen)
        return out + ([self.counts(e.ts)] if out else [])

    def on_zone_event(self, e: ZoneEvent) -> list:
        iid = self.item_id(e.camera_id, e.track_id)
        if e.kind != "enter" or e.zone_id not in self.bin_by_zone or not self.tracked(e.label) or iid in self.disposed:
            return []
        out = []
        if iid not in self.seen:  # first seen at the bin: still counts
            out += self.on_observation_like(e)
        b = self.bin_by_zone[e.zone_id]
        done = ItemDisposed(ts=e.ts, item_id=iid, label=e.label, zone=e.zone_id, bin=b,
                            correct=self.rules.bin_for(e.label) == b, frame_ref=e.frame_ref)
        self.disposed[iid] = done
        return out + [done, self.counts(e.ts)]

    def on_observation_like(self, e: ZoneEvent) -> list:
        seen = ItemSeen(ts=e.ts, item_id=self.item_id(e.camera_id, e.track_id), label=e.label,
                        camera_id=e.camera_id, recyclable=self.recyclable(e.label), frame_ref=e.frame_ref)
        self.seen[seen.item_id] = seen
        return [seen]

    # --- E2: metrics ---

    def metrics(self, since: datetime | None = None, until: datetime | None = None) -> dict:
        def inside(ts):
            return (since is None or ts >= since) and (until is None or ts < until)

        seen = [s for s in self.seen.values() if inside(s.ts)]
        recyclable_ids = {s.item_id for s in seen if s.recyclable}
        disposed = [d for d in self.disposed.values() if inside(d.ts)]
        recycled = sum(1 for d in disposed if d.bin == "recycling" and d.item_id in recyclable_ids)
        contamination = sum(1 for d in disposed if d.bin == "recycling" and not self.recyclable(d.label))
        return {
            "items_seen": len(seen),
            "recyclable_seen": len(recyclable_ids),
            "recycled": recycled,
            "contamination": contamination,
            "diversion_rate": round(recycled / len(recyclable_ids), 3) if recyclable_ids else 0.0,
        }

    def counts(self, ts: datetime) -> CountUpdated:
        return CountUpdated(ts=ts, **self.metrics())

    def before_after(self) -> dict | None:
        if self.nudges_on_at is None:
            return None
        return {"before": self.metrics(until=self.nudges_on_at), "after": self.metrics(since=self.nudges_on_at)}
