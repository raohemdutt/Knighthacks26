"""The demo people: made-up templates marked as demo data, a week of the owner's looks, and a story that did not change."""

import asyncio
import math
import shutil
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from contexts.demo import checkin, circle, people
from contexts.demo.__main__ import main
from contexts.demo.seed import seed, settings, summary
from contexts.demo.story import story
from contexts.day.domain.checkin import EXPR_SHARE, EXPR_WINDOW_MIN
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import build
from contexts.perception.application.identity import IdentityError, shared_member_id
from contexts.perception.domain.expression import STRAINED

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 15, 20, tzinfo=TZ)
MAYA = shared_member_id("maya")


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    root = tmp_path_factory.mktemp("people-demo")
    cfg = circle.config_dir(root / "config")
    plain = asyncio.run(seed(root / "plain", now=NOW, config_dir=cfg, extra=circle.circle_events))
    out = asyncio.run(seed(root / "data", now=NOW, config_dir=cfg, extra=circle.circle_events, people=True))
    yield root, cfg, plain, out
    plain.app.projection.store.close()
    out.app.projection.store.close()


@pytest.fixture(scope="module")
def phone(seeded, tmp_path_factory):
    """The gateway on a copy of the seeded directory (log and vault), rebuilt by replay as after a restart."""
    root, cfg, _, _ = seeded
    data = tmp_path_factory.mktemp("people-replayed")
    shutil.copytree(root / "data" / "events", data / "events")
    shutil.copytree(root / "data" / "vault", data / "vault")
    s = settings(data, cfg)
    s.replay = True
    atlas = build(s)
    with TestClient(create_app(atlas=atlas)) as c:
        yield c, atlas
    atlas.projection.store.close()


def test_the_vectors_are_made_up_the_same_every_time_and_under_no_recognizers_name():
    v = people.vector("anyone", 3)
    assert v == people.vector("anyone", 3) != people.vector("anyone", 4)
    assert len(v) == people.DIMENSIONS and math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9)
    from contexts.perception.infrastructure.recognizer import build_recognizer

    real = build_recognizer()  # None on a machine without the model file
    assert people.MODEL == "demo-synthetic" and (real is None or real.name != people.MODEL)
    assert all(m.demo and m.model == people.MODEL for m in people.members(NOW, circle=True))


def test_three_people_each_in_their_true_state_and_marked_as_demo_data(seeded, phone):
    _, _, _, out = seeded
    assert out.people == [people.OWNER, people.PHOTOS_ONLY, MAYA]
    c, atlas = phone
    rows = {m["member_id"]: m for m in c.get("/identity/members").json()}
    assert set(rows) == {people.OWNER, people.PHOTOS_ONLY, MAYA} and all(m["demo"] is True for m in rows.values())
    me, photos, maya = rows[people.OWNER], rows[people.PHOTOS_ONLY], rows[MAYA]
    assert (me["name"], me["source"], me["scanned"], me["recognised"], me["expressions"], me["circle_member_id"]) == (
        "Hemdutt", "scan", True, True, True, "hemdutt")
    assert (photos["name"], photos["source"], photos["scanned"], photos["recognised"], photos["expressions"]) == (
        "Dominick", "photos", False, False, False)
    assert (maya["name"], maya["source"], maya["scanned"], maya["shared_by"]) == ("Maya", "circle", True, "maya")
    assert datetime.fromisoformat(maya["expires_at"]) - datetime.fromisoformat(maya["received_at"]) == timedelta(days=people.SHARE_DAYS)
    assert all("vector" not in m and "vectors" not in m for m in rows.values())  # never the template
    # the same member the check-in story reads, so the two demos name one person
    assert people.OWNER == checkin.MEMBER


def test_without_a_face_model_the_api_says_so_and_still_lists_the_saved_people(phone):
    c, atlas = phone
    s = c.get("/identity/settings").json()
    if atlas.identity.available:
        pytest.skip("this machine has the face model: the no-model state cannot be shown here")
    assert (s["available"], s["identity_enabled"], s["expressions_available"]) == (False, False, False)
    assert "not installed" in s["reason"] and len(c.get("/identity/members").json()) == 3
    assert c.get("/identity/present").json() == {"present": []}  # nobody is recognised live


def test_the_owners_looks_are_there_for_the_screen_and_only_while_the_story_has_someone_at_the_desk(seeded, phone):
    c, atlas = phone
    events = people.looks(NOW, story(NOW))
    assert [e.model_dump_json() for e in people.looks(NOW, story(NOW))] == [e.model_dump_json() for e in events]
    assert all(e.event_id.startswith("demo-people-looks-") and e.member_id == people.OWNER and e.camera_id == checkin.DESK_CAM for e in events)
    assert len({e.event_id for e in events}) == len(events) > 60 and events[-1].window_end <= NOW
    assert all(a.window_end <= b.window_start for a, b in zip(events, events[1:]))  # no stretch overlaps the next
    assert NOW - events[0].window_start > timedelta(days=6)  # a week
    began = min(e.ts for e in story(NOW))
    stretches = people.desk_stretches(story(NOW))
    inside = [e for e in events if e.window_start >= began]
    assert inside and all(any(a <= e.window_start and e.window_end <= b for a, b in stretches) for e in inside)
    # what the People screen asks for (the last 48 hours) comes back from the replayed log
    got = atlas.identity.expressions(people.OWNER, hours=48, now=NOW)
    assert got["samples"] > 0 and len(got["windows"]) == len([e for e in events if e.window_end >= NOW - timedelta(hours=48)])
    assert set(got["looked"]) <= {"neutral", "upbeat", "tired", "tense"} and max(got["looked"], key=got["looked"].get) == "neutral"


def test_the_looks_agree_with_the_check_in_story_and_never_raise_one(seeded):
    _, _, _, out = seeded
    events = people.looks(NOW, story(NOW))
    began = min(e.ts for e in story(NOW))

    def strained(e) -> float:
        return sum(share for label, share in e.shares.items() if label in STRAINED)

    today = NOW.astimezone(TZ).date()
    for back in checkin.LIKE_THIS:  # the days the check-in story calls "like this": the afternoon read as strained
        at = datetime.combine(today - timedelta(days=back), checkin.FELL_AT[back], TZ)
        that = [e for e in events if at <= e.window_start < at + timedelta(hours=1)]
        if at + timedelta(minutes=3) <= began:
            assert that and all(e.shares == people.STRAINED and strained(e) >= EXPR_SHARE for e in that)
        else:
            assert all(e.window_start >= began for e in that)  # the story covers that afternoon: it is the story's to tell
    # inside the story no twenty minutes read as strained, so the Day loop's rule has nothing to say about them
    recent = [e for e in events if e.window_start >= began]
    assert all(strained(e) < EXPR_SHARE for e in recent) and EXPR_WINDOW_MIN >= 3
    assert out.written["day.checkin.raised"] == 0


def test_the_story_is_exactly_what_it_was(seeded):
    _, _, plain, out = seeded
    more = Counter(out.written)
    more.subtract(plain.written)
    assert {t: n for t, n in more.items() if n} == {"perception.expression.observed": out.looks}
    assert out.looks == len(people.looks(NOW, story(NOW))) and out.source_events == plain.source_events + out.looks
    for kept, fields in (("alerts", ("rule_id", "message")), ("nudges", ("habit_id", "message")), ("plans", ("subject", "text")),
                         ("rounds", ("item", "correct", "score"))):  # ids are made up per run: the words and the times are the story
        assert [(e.ts, *(getattr(e, f) for f in fields)) for e in getattr(out, kept)] == [
            (e.ts, *(getattr(e, f) for f in fields)) for e in getattr(plain, kept)]
    assert out.app.habits.digest(now=NOW) == plain.app.habits.digest(now=NOW)
    assert out.app.waste.metrics() == plain.app.waste.metrics()
    assert out.app.memory.stats() == plain.app.memory.stats()
    assert out.app.league.view(NOW)["standings"] == plain.app.league.view(NOW)["standings"]
    # nobody is "identified" and no switch is flipped: Habits and the Impact story count from those
    assert not any(t.startswith(("perception.person", "perception.identity")) for t in out.written)
    assert not out.app.identity.store.enabled


def test_nothing_about_them_leaves_the_box(seeded, phone):
    root, _, _, out = seeded
    c, atlas = phone
    # no Circle message, share or export was made of them: the only event type the people seed adds is the looks
    sent = [t for t in out.written if t.startswith(("circle.face_share", "learning.export", "vault.export"))]
    assert sent == []
    with pytest.raises(IdentityError):
        atlas.identity.own_template("hemdutt")  # demo data is not this box's to share
    assert atlas.face_share is None or not atlas.face_share.enabled
    # on disk they are vault blobs: no name and no number of a vector in plain text
    for blob in (root / "data" / "vault" / "identity").glob("*.face"):
        raw = blob.read_bytes()
        assert b"Hemdutt" not in raw and b"Maya" not in raw and b"Dominick" not in raw and b"demo" not in raw
    log = b"".join(f.read_bytes() for f in (root / "data" / "events").glob("*.jsonl"))
    assert b"demo-synthetic" not in log and b'"vector' not in log


def test_demo_people_cannot_be_matched_added_to_or_switched(phone):
    c, atlas = phone
    identity = atlas.identity
    assert all(m.demo for m in identity.store.members())
    if identity.recognizer is not None:
        assert identity._templates() == {}
    r = c.put(f"/identity/members/{people.PHOTOS_ONLY}", json={"expressions": True})
    assert r.status_code == 409 and "demo data" in r.json()["detail"]


def test_seeding_twice_adds_nothing_and_leaves_the_people_alone(seeded):
    root, cfg, _, out = seeded
    before = sum(1 for f in (root / "data" / "events").glob("*.jsonl") for _ in f.open())
    files = {p.name: p.read_bytes() for p in (root / "data" / "vault" / "identity").glob("*.face")}
    again = asyncio.run(seed(root / "data", now=NOW, force=True, config_dir=cfg, extra=circle.circle_events, people=True))
    text = summary(again)
    again.app.projection.store.close()
    assert sum(again.written.values()) == 0 and again.people == []
    assert sum(1 for f in (root / "data" / "events").glob("*.jsonl") for _ in f.open()) == before
    assert {p.name: p.read_bytes() for p in (root / "data" / "vault" / "identity").glob("*.face")} == files
    assert "People (DEMO DATA: made-up vectors, never a face" in text and "Dominick   photos only: scan to finish" in text


def test_the_plain_seed_has_no_people_and_the_flag_adds_the_two_who_need_no_circle(tmp_path, capsys):
    assert main(["seed", "--data-dir", str(tmp_path / "plain")]) == 0
    assert "DEMO DATA" not in capsys.readouterr().out and not (tmp_path / "plain" / "vault" / "identity").exists()
    assert main(["seed", "--data-dir", str(tmp_path / "people"), "--people"]) == 0
    out = capsys.readouterr().out
    assert "Hemdutt    scanned, reads expressions" in out and "Dominick   photos only: scan to finish" in out
    assert "recognised through the Circle" not in out  # no Circle on the plain box, so no friend's share
    assert len(list((tmp_path / "people" / "vault" / "identity").glob("*.face"))) == 2
