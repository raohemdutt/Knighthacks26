"""Event log -> Memory tables (M2, H3).

An episode is a bounded span of activity on one camera. It opens on the first
activity and closes after QUIET_S without any, measured in event time, so a
replay builds the same rows as the live run. Activity is an observation with
at least one track, a zone event or a caption. An episode that never goes
quiet (a laptop always in view) is cut at MAX_EPISODE_S so hours stay
separable.

Handlers follow the bus convention (event in, events out) and are safe to
replay: an event that was already projected is skipped by `event_id`.
"""

from datetime import datetime, timedelta

from bus import EventBus
from contexts.memory.infrastructure.store import MemoryStore
from contracts.connectors import ItemSynced
from contracts.memory import Episode, EpisodeClosed, EpisodeOpened, Item, Sighting
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent

QUIET_S = 30
MAX_EPISODE_S = 600
MAX_KEYFRAMES = 8


def episode_id(camera_id: str, start: datetime) -> str:
    """From the camera and the first event's time, so a replay gives the same id."""
    return f"{camera_id}-{int(start.timestamp() * 1000)}"


class MemoryProjection:
    def __init__(self, store: MemoryStore, quiet_s: float = QUIET_S, max_episode_s: float = MAX_EPISODE_S) -> None:
        self.store = store
        self.quiet = timedelta(seconds=quiet_s)
        self.max_length = timedelta(seconds=max_episode_s)

    def subscribe(self, bus: EventBus) -> None:
        """All of these rebuild state from the log, so all of them replay."""
        bus.subscribe("perception.observation", self.on_observation, replay=True)
        bus.subscribe("perception.zone", self.on_zone_event, replay=True)
        bus.subscribe("perception.scene", self.on_scene, replay=True)
        bus.subscribe("connector.item.synced", self.on_item, replay=True)
        bus.subscribe("memory.episode.closed", self.on_episode_closed, replay=True)

    # --- handlers ---

    def on_observation(self, e: ObservationRecorded) -> list:
        if not e.tracks:  # an empty frame is quiet: it only moves the clock
            return self.flush(e.ts)
        with self.store.db:
            if not self.store.first_time(e.event_id):
                return []
            return self._activity(e.camera_id, e.ts, labels=[t.label for t in e.tracks], first_frame=e.frame_ref)

    def on_zone_event(self, e: ZoneEvent) -> list:
        with self.store.db:
            if not self.store.first_time(e.event_id):
                return []
            return self._activity(e.camera_id, e.ts, labels=[e.label], keyframe=e.frame_ref)

    def on_scene(self, e: SceneDescribed) -> list:
        with self.store.db:
            if not self.store.first_time(e.event_id):
                return []
            for o in e.objects:
                self.store.add_sighting(Sighting(label=o.label, camera_id=e.camera_id, zone_id=e.zone_id, ts=e.ts,
                                                 frame_ref=e.frame_ref, position=o.position))
            return self._activity(e.camera_id, e.ts, caption=e.caption, keyframe=e.frame_ref)

    def on_item(self, e: ItemSynced) -> list:
        with self.store.db:
            if self.store.first_time(e.event_id):
                item = Item(connector=e.connector, kind=e.kind, external_id=e.external_id, title=e.title,
                            when=e.when, url=e.url, summary=e.summary)
                self.store.put_item(item, synced_at=e.ts)
        return []

    def on_episode_closed(self, e: EpisodeClosed) -> list:
        """Our own event, read back on replay: the ticker's closes are in the log but its ticks are not."""
        with self.store.db:
            if self.store.first_time(e.event_id):
                ep = self.store.episode(e.episode_id)
                if ep and ep.end is None:
                    ep.end = e.end
                    self.store.put_episode(ep, e.end)
        return []

    def flush(self, now: datetime) -> list[EpisodeClosed]:
        """Close every episode that has been quiet for QUIET_S at `now`. The app's ticker calls this."""
        with self.store.db:
            return [self._close(ep, last, at=last + self.quiet)
                    for ep, last in self.store.open_episodes() if now - last > self.quiet]

    # --- episode building ---

    def _activity(self, camera_id: str, ts: datetime, labels: list[str] = (), caption: str | None = None,
                  keyframe: str | None = None, first_frame: str | None = None) -> list:
        out: list = [self._close(ep, last, at=last + self.quiet)
                     for ep, last in self.store.open_episodes() if ts - last > self.quiet]
        ep, last = next(((ep, last) for ep, last in self.store.open_episodes() if ep.camera_id == camera_id), (None, ts))
        if ep and ts - ep.start >= self.max_length:
            out.append(self._close(ep, last, at=ts))
            ep, last = None, ts
        if ep is None:
            ep = Episode(id=episode_id(camera_id, ts), camera_id=camera_id, start=ts)
            out.append(EpisodeOpened(ts=ts, episode_id=ep.id, camera_id=camera_id))
        ep.labels += [label for label in dict.fromkeys(labels) if label not in ep.labels]
        if caption and caption not in ep.captions:
            ep.captions.append(caption)
            self.store.index("caption", ep.id, caption)
        # Observations arrive every second: one frame from them is enough; zone events and captions keep theirs.
        frame = keyframe or (first_frame if not ep.keyframe_refs else None)
        if frame and frame not in ep.keyframe_refs and len(ep.keyframe_refs) < MAX_KEYFRAMES:
            ep.keyframe_refs.append(frame)
        self.store.put_episode(ep, max(last, ts))
        return out

    def _close(self, ep: Episode, last: datetime, at: datetime) -> EpisodeClosed:
        ep.end = last
        self.store.put_episode(ep, last)
        return EpisodeClosed(ts=at, episode_id=ep.id, camera_id=ep.camera_id, start=ep.start, end=last, labels=ep.labels,
                             captions=ep.captions, summary=ep.summary, keyframe_refs=ep.keyframe_refs)
