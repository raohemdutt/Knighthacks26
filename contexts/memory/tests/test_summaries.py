"""Episode summaries (M3) and the agent's hourly notes in search."""

import asyncio
from datetime import timedelta

from bus import EventBus
from contexts.memory.__main__ import rebuild
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.application.summarizer import EpisodeSummarizer
from contexts.memory.infrastructure.store import MemoryStore
from contexts.memory.tests.test_projection import at, ref, script, seen
from contracts.agent import NoteWritten
from contracts.memory import EpisodeClosed
from runtime.fake import FakeRuntime

KEYS = "A person puts a set of keys next to the coffee mug."
SUMMARY = "Someone dropped their keys by the mug and left."


class Summaries(FakeRuntime):
    def summarize(self, captions: list[str]) -> str:
        super().summarize(captions)
        return SUMMARY + " "


def live(tmp_path, events, runtime, db: str = "memory.sqlite") -> tuple[MemoryStore, list]:
    """The app's wiring: projection on replay, summarizer live only. Waits for the background model calls."""
    store = MemoryStore(tmp_path / db)
    bus = EventBus(tmp_path / "events")
    emitted: list = []
    bus.subscribe("memory.", emitted.append)
    MemoryProjection(store).subscribe(bus)
    EpisodeSummarizer(store, runtime).subscribe(bus)

    async def main() -> None:
        await bus.publish_all(events)
        await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))

    asyncio.run(main())
    return store, emitted


def summarize_calls(runtime: FakeRuntime) -> list:
    return [args for name, args in runtime.calls if name == "summarize"]


def test_a_closed_episode_gets_a_summary_from_its_captions(tmp_path):
    runtime = Summaries()
    store, emitted = live(tmp_path, script(), runtime)
    first, door, second = SqliteMemory(store).episodes(at(-1), at(1000))

    assert summarize_calls(runtime) == [[KEYS]]  # one call, for the one closed episode that has captions
    assert first.summary == SUMMARY
    assert (first.end, first.captions) == (at(20), [KEYS])  # the rest of the row is untouched
    assert door.summary is None  # closed, but nothing was captioned at the door
    assert second.end is None and second.summary is None  # still open
    assert len([e for e in emitted if isinstance(e, EpisodeClosed)]) == 2


def test_an_episode_without_captions_is_not_summarized(tmp_path):
    runtime = Summaries()
    store, emitted = live(tmp_path, [seen(0, "person"), seen(5, "person"), seen(60)], runtime)
    [closed] = [e for e in emitted if isinstance(e, EpisodeClosed)]
    [ep] = SqliteMemory(store).episodes(at(-1), at(1000))

    assert closed.captions == [] and runtime.calls == []
    assert ep.end == at(5) and ep.summary is None


def test_the_summary_is_in_the_window_and_in_search(tmp_path):
    store, _ = live(tmp_path, script(), Summaries())
    memory = SqliteMemory(store)
    first = memory.episodes(at(-1), at(1000))[0]

    hour = memory.window(at(-1), at(1000)).hours[0]
    assert SUMMARY in hour.captions and KEYS not in hour.captions  # the summary stands in for its captions

    hit = memory.search("who dropped something?")[0]  # "dropped" is only in the summary
    assert (hit.episode_id, hit.camera_id, hit.ts, hit.kind) == (first.id, "cam-desk", at(0), "episode")
    assert hit.text == SUMMARY and hit.frame_ref == first.keyframe_refs[0]


def test_nothing_is_summarized_during_replay(tmp_path):
    runtime = Summaries()
    store, _ = live(tmp_path, script(), runtime)
    expected = store.dump()
    assert len(summarize_calls(runtime)) == 1

    # A restart: the log replays into the same file. No model call, and the summary is still there.
    bus = EventBus(tmp_path / "events")
    MemoryProjection(store).subscribe(bus)
    EpisodeSummarizer(store, runtime).subscribe(bus)
    assert asyncio.run(bus.replay_into()) > len(script())
    assert len(summarize_calls(runtime)) == 1 and store.dump() == expected
    assert SqliteMemory(store).episodes(at(-1), at(1000))[0].summary == SUMMARY

    # A replay into an empty file, wired like the app: episodes come back, the model is still not called.
    fresh = MemoryStore(tmp_path / "fresh.sqlite")
    bus = EventBus(tmp_path / "events")
    MemoryProjection(fresh).subscribe(bus)
    EpisodeSummarizer(fresh, runtime).subscribe(bus)
    asyncio.run(bus.replay_into())
    assert len(summarize_calls(runtime)) == 1
    assert [e.summary for e in SqliteMemory(fresh).episodes(at(-1), at(1000))] == [None, None, None]

    # Summaries are not in the log yet, so the rebuild command leaves them empty too.
    rebuild(tmp_path / "events", tmp_path / "rebuilt.sqlite")
    rebuilt = MemoryStore(tmp_path / "rebuilt.sqlite")
    assert [e.summary for e in SqliteMemory(rebuilt).episodes(at(-1), at(1000))] == [None, None, None]
    assert rebuilt.dump() == fresh.dump()


def test_an_episode_is_summarized_once(tmp_path):
    runtime = Summaries()
    store, emitted = live(tmp_path, script(), runtime)
    closed = next(e for e in emitted if isinstance(e, EpisodeClosed) and e.captions)
    summarizer = EpisodeSummarizer(store, runtime)

    assert asyncio.run(summarizer.on_episode_closed(closed)) == []  # it has a summary already
    assert asyncio.run(summarizer.on_episode_closed(closed.model_copy(update={"episode_id": "cam-desk-0"}))) == []  # unknown
    assert len(summarize_calls(runtime)) == 1
    assert len([r for r in store.dump()["memory_fts"] if r[1] == "summary"]) == 1


def test_hourly_notes_are_searchable_and_rebuild_from_the_log(tmp_path):
    hour = at(0).replace(minute=0, second=0)
    note = NoteWritten(ts=hour + timedelta(hours=1), note_id="n1", hour=hour, text="Dom debugged the planner, then ate pizza.")
    store, _ = live(tmp_path, script(), FakeRuntime())
    first = SqliteMemory(store).episodes(at(-1), at(1000))[0]
    bus = EventBus(tmp_path / "events")
    MemoryProjection(store).subscribe(bus)
    asyncio.run(bus.publish_all([note.model_copy(update={"episode_ids": ["gone", first.id]}),
                                 note.model_copy(update={"event_id": "e2", "note_id": "n2", "text": "Pizza boxes piled up."})]))
    memory = SqliteMemory(store)

    hit = memory.search("who debugged the planner?")[0]
    assert (hit.kind, hit.text, hit.ts) == ("note", "Dom debugged the planner, then ate pizza.", hour)
    assert (hit.episode_id, hit.camera_id, hit.frame_ref) == (first.id, "cam-desk", ref("cam-desk", 0))  # cites its first episode
    loose = memory.search("boxes piled")[0]
    assert (loose.episode_id, loose.camera_id, loose.frame_ref, loose.kind) == ("note:n2", "agent", None, "note")

    rebuild(tmp_path / "events", tmp_path / "rebuilt.sqlite")
    rebuilt = MemoryStore(tmp_path / "rebuilt.sqlite")
    assert rebuilt.dump()["memory_notes"] == store.dump()["memory_notes"] and len(store.dump()["memory_notes"]) == 2
    assert SqliteMemory(rebuilt).search("who debugged the planner?")[0].kind == "note"
