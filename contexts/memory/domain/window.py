"""Hour buckets for "what happened overnight": same rules as the agent's FakeMemory."""

from datetime import datetime

from contracts.memory import Episode, EpisodeWindow, HourBucket

MAX_CAPTIONS_PER_HOUR = 6


def group_by_hour(start: datetime, end: datetime, episodes: list[Episode]) -> EpisodeWindow:
    """One bucket per hour that has episodes; a summary stands in for its episode's captions."""
    buckets: dict[datetime, HourBucket] = {}
    for ep in sorted(episodes, key=lambda e: e.start):
        hour = ep.start.replace(minute=0, second=0, microsecond=0)
        b = buckets.setdefault(hour, HourBucket(hour=hour, episode_ids=[], captions=[]))
        b.episode_ids.append(ep.id)
        for c in [ep.summary] if ep.summary else ep.captions:
            if c not in b.captions and len(b.captions) < MAX_CAPTIONS_PER_HOUR:
                b.captions.append(c)
        b.keyframe_refs.extend(ep.keyframe_refs[:1])
    return EpisodeWindow(start=start, end=end, hours=list(buckets.values()))
