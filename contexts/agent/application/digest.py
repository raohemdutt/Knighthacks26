"""The daily digest (A6): one tap, a summary of the day's episodes.

Written from the data, with no model call, the way the visit recap is (A12): the counts, what was seen most,
the busiest hour and a handful of moments in time order, each one cited. On the Jetson a 4B model garbles long
lists and takes seconds per paragraph; a digest of a whole day is exactly that kind of list.

The day is the household's local day (SleepSchedule.tz), midnight to midnight, cut at "now" for today.
"""

from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from contexts.agent.application.memory_port import MemoryReader
from contracts.memory import Episode
from contracts.types import Citation

MAX_MOMENTS = 6  # hours read out as moments; the busiest ones, then put back in time order
MAX_LABELS = 3  # "seen most" names this many things
QUIET_LABELS = {"person"}  # counted, but a digest that says "seen most: person" every day says nothing


class Moment(BaseModel):
    hour: datetime  # local start of the hour
    episodes: int
    text: str  # the hour's first summary or caption
    citation: Citation


class DailyDigest(BaseModel):
    day: date
    start: datetime
    end: datetime
    text: str  # the digest as one paragraph, ready to read out or show
    episodes: int
    cameras: dict[str, int]  # camera -> episodes
    seen_most: list[tuple[str, int]]  # label -> episodes it was in
    people_episodes: int
    first: datetime | None
    last: datetime | None
    busiest_hour: datetime | None
    moments: list[Moment]
    citations: list[Citation]  # one per moment first, then every other episode of the day


def episode_text(e: Episode) -> str:
    return (e.summary or next((c for c in e.captions if c.strip()), None) or ", ".join(e.labels) or "activity").strip()


def clock(ts: datetime, tz: ZoneInfo) -> str:
    """ "7:30 AM", portable (no %-I on Windows)."""
    return ts.astimezone(tz).strftime("%I:%M %p").lstrip("0")


def hour_name(ts: datetime, tz: ZoneInfo) -> str:
    return ts.astimezone(tz).strftime("%I %p").lstrip("0")


def sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return (text[:1].upper() + text[1:] + ".") if text else ""


class Digester:
    def __init__(self, memory: MemoryReader, clock: Callable[[], datetime], tz: ZoneInfo) -> None:
        self.memory = memory
        self.clock = clock
        self.tz = tz

    def window(self, day: date | None = None) -> tuple[date, datetime, datetime]:
        now = self.clock().astimezone(self.tz)
        day = day or now.date()
        start = datetime.combine(day, time(0), self.tz)
        end = min(start + timedelta(days=1), now) if day <= now.date() else start
        return day, start, max(start, end)

    def build(self, day: date | None = None) -> DailyDigest:
        day, start, end = self.window(day)
        episodes = sorted(self.memory.episodes(start, end), key=lambda e: e.start) if end > start else []
        label = "today" if day == self.clock().astimezone(self.tz).date() else day.strftime("%A, %B %d").replace(" 0", " ")
        if not episodes:
            text = f"Nothing on the cameras {label} so far." if label == "today" else f"Nothing on the cameras on {label}."
            return DailyDigest(day=day, start=start, end=end, text=text, episodes=0, cameras={}, seen_most=[], people_episodes=0,
                               first=None, last=None, busiest_hour=None, moments=[], citations=[])

        cameras = Counter(e.camera_id for e in episodes)
        labels = Counter(lbl for e in episodes for lbl in set(e.labels))
        people = labels.get("person", 0)
        seen_most = [(lbl, n) for lbl, n in labels.most_common() if lbl not in QUIET_LABELS][:MAX_LABELS]

        by_hour: dict[datetime, list[Episode]] = {}
        for e in episodes:
            local = e.start.astimezone(self.tz)
            by_hour.setdefault(local.replace(minute=0, second=0, microsecond=0), []).append(e)
        busiest = max(by_hour, key=lambda h: (len(by_hour[h]), -h.timestamp()))  # most episodes; the earlier hour on a tie
        chosen = sorted(sorted(by_hour, key=lambda h: (-len(by_hour[h]), h))[:MAX_MOMENTS])
        moments = []
        for h in chosen:
            first = by_hour[h][0]
            ref = first.keyframe_refs[0] if first.keyframe_refs else None
            moments.append(Moment(hour=h, episodes=len(by_hour[h]), text=episode_text(first),
                                  citation=Citation(episode_id=first.id, frame_ref=ref, ts=first.start)))
        cited = {m.citation.episode_id for m in moments}
        citations = [m.citation for m in moments] + [Citation(episode_id=e.id, ts=e.start) for e in episodes if e.id not in cited]

        n, k = len(episodes), len(cameras)
        parts = [f"{label[:1].upper() + label[1:]}: {n} moment{'s' * (n != 1)} on {k} camera{'s' * (k != 1)}, "
                 f"from {clock(episodes[0].start, self.tz)} to {clock(episodes[-1].start, self.tz)}."]
        if seen_most:
            parts.append("Seen most: " + ", ".join(f"{lbl} ({c})" for lbl, c in seen_most) + ".")
        if people:
            parts.append(f"People in {people} of them.")
        if len(by_hour) > 1:
            parts.append(f"Busiest hour: {hour_name(busiest, self.tz)} ({len(by_hour[busiest])}).")
        parts += [f"{hour_name(m.hour, self.tz)}: {sentence(m.text)}" for m in moments]
        if len(by_hour) > len(moments):
            parts.append(f"And {len(by_hour) - len(moments)} quieter hour{'s' * (len(by_hour) - len(moments) != 1)}.")
        return DailyDigest(day=day, start=start, end=end, text=" ".join(parts), episodes=n, cameras=dict(cameras),
                           seen_most=seen_most, people_episodes=people, first=episodes[0].start, last=episodes[-1].start,
                           busiest_hour=busiest, moments=moments, citations=citations)
