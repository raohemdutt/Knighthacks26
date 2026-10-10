import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from contexts.connectors.servers.ics import expand, read_feeds

NY = ZoneInfo("America/New_York")

# Exchange-style feed: Windows zone name with its own VTIMEZONE, a weekly
# lecture with one moved instance, and an all-day deadline.
FEED = b"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:test
BEGIN:VTIMEZONE
TZID:Eastern Standard Time
BEGIN:STANDARD
DTSTART:16011104T020000
RRULE:FREQ=YEARLY;BYDAY=1SU;BYMONTH=11
TZOFFSETFROM:-0400
TZOFFSETTO:-0500
END:STANDARD
BEGIN:DAYLIGHT
DTSTART:16010311T020000
RRULE:FREQ=YEARLY;BYDAY=2SU;BYMONTH=3
TZOFFSETFROM:-0500
TZOFFSETTO:-0400
END:DAYLIGHT
END:VTIMEZONE
BEGIN:VEVENT
UID:lecture-1
SUMMARY:COP 3502 Lecture
LOCATION:HEC 101
DTSTART;TZID=Eastern Standard Time:20261005T100000
DTEND;TZID=Eastern Standard Time:20261005T111500
RRULE:FREQ=WEEKLY;COUNT=4
END:VEVENT
BEGIN:VEVENT
UID:lecture-1
RECURRENCE-ID;TZID=Eastern Standard Time:20261012T100000
SUMMARY:COP 3502 Lecture (moved)
DTSTART;TZID=Eastern Standard Time:20261013T140000
DTEND;TZID=Eastern Standard Time:20261013T151500
END:VEVENT
BEGIN:VEVENT
UID:hw3
SUMMARY:HW 3 due
DTSTART;VALUE=DATE:20261011
DTEND;VALUE=DATE:20261012
END:VEVENT
END:VCALENDAR
"""


def test_expands_recurrence_overrides_and_windows_zones():
    start = dt.datetime(2026, 10, 10, tzinfo=NY)
    items = expand(FEED, start, start + dt.timedelta(days=14), "outlook", "calendar_event")
    assert [(i["title"], i["when"]) for i in items] == [
        ("HW 3 due", "2026-10-11T00:00:00-04:00"),
        ("COP 3502 Lecture (moved)", "2026-10-13T14:00:00-04:00"),
        ("COP 3502 Lecture", "2026-10-19T10:00:00-04:00"),
    ]
    assert items[0]["summary"] == "all day"
    assert items[2]["summary"] == "until 11:15 AM · HEC 101"
    assert len({i["external_id"] for i in items}) == 3


def test_feed_url_comes_from_env(monkeypatch):
    now = dt.datetime(2026, 10, 10, 12, tzinfo=NY)
    options = {"feeds": [{"name": "outlook", "url_env": "TEST_ICS_URL"}]}
    with pytest.raises(RuntimeError, match="TEST_ICS_URL"):
        read_feeds(options, now, lambda url: FEED)
    monkeypatch.setenv("TEST_ICS_URL", "https://secret.example/feed.ics")
    fetched = []
    items = read_feeds(options, now, lambda url: fetched.append(url) or FEED)
    assert fetched == ["https://secret.example/feed.ics"] and len(items) == 3


def test_invites_round_trip_through_any_calendar(tmp_path):
    import icalendar

    from contexts.connectors.servers.ics import write_invite

    start = dt.datetime(2026, 10, 11, 14, tzinfo=NY)
    path = write_invite([
        {"title": "Study: COP 3502", "start": start.isoformat(), "end": (start + dt.timedelta(hours=2)).isoformat(),
         "location": "Library", "attendees": ["hemdutt@example.edu"]},
        {"title": "Study: COP 3502", "start": start + dt.timedelta(hours=4), "end": start + dt.timedelta(hours=6)},
    ], tmp_path, organizer="dom@example.edu")
    cal = icalendar.Calendar.from_ical(path.read_bytes())
    events = list(cal.walk("VEVENT"))
    assert cal["METHOD"] == "REQUEST" and len(events) == 2
    assert events[0]["SUMMARY"] == "Study: COP 3502" and events[0]["LOCATION"] == "Library"
    assert events[0]["DTSTART"].dt == start and "hemdutt@example.edu" in str(events[0]["ATTENDEE"])
    # and our own reader parses what we wrote
    assert len(expand(path.read_bytes(), start - dt.timedelta(hours=1), start + dt.timedelta(days=1), "mine", "calendar_event")) == 2
