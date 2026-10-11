"""The Impact story over Waste Watch: real units as estimated ranges, the disposal feed with its rules, the weekly
goal and momentum, milestones, the practice set, and one event stream judged under two rules files."""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.impact.api import impact_router
from contexts.impact.application.story import Factors, ImpactStory, RuleBook, week_of
from contexts.impact.domain.waste_watch import WasteWatch
from contracts.habits import NudgeRaised
from contracts.impact import ItemDisposed, MilestoneReached, PracticeAdded
from contracts.play import RoundScored
from contracts.perception import IdentityChanged, ObservationRecorded, PersonIdentified, PersonLost, ZoneEvent
from contracts.types import Box, Track

TZ = ZoneInfo("America/New_York")
MONDAY = date(2026, 10, 5)
BOX = Box(x1=0, y1=0, x2=0.1, y2=0.1)
UCF = Path("config/sorting.yaml")
HOME = Path("config/profiles/family-home/sorting.yaml")
CAM = "cam-door"


def at(day: int, hour: int = 12, minute: int = 0, second: int = 0) -> datetime:
    """`day` 0 is Monday of the week under test."""
    d = MONDAY + timedelta(days=day)
    return datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=TZ)


def toss(ts: datetime, track: int, label: str, zone: str) -> list:
    """What the camera publishes when one item is carried into a bin zone."""
    ref = f"frames:{CAM}:{int(ts.timestamp() * 1000)}"
    return [ObservationRecorded(event_id=f"o-{track}", ts=ts, camera_id=CAM, frame_ref=ref,
                                tracks=[Track(track_id=track, label=label, confidence=0.9, box=BOX)]),
            ZoneEvent(event_id=f"z-{track}", ts=ts + timedelta(seconds=1), camera_id=CAM, track_id=track, label=label, zone_id=zone,
                      kind="enter", frame_ref=ref)]


# One week at the table: Monday clean, Tuesday a cup in recycling and a can in trash, Wednesday clean.
WEEK = [*toss(at(0, 10), 1, "can", "recycling"), *toss(at(0, 11), 2, "bottle", "recycling"), *toss(at(0, 12), 3, "napkin", "trash"),
        *toss(at(1, 9), 4, "cup", "recycling"), *toss(at(1, 10), 5, "can", "trash"), *toss(at(1, 11), 6, "can", "recycling"),
        *toss(at(2, 9), 7, "can", "recycling"), *toss(at(2, 10), 8, "food container", "trash")]


def wired(tmp_path, rules: Path = UCF, factors: Factors | None = None, nudges_on_at=None):
    bus = EventBus(tmp_path / "events")
    waste = WasteWatch(RuleBook.load(rules).rules, nudges_on_at=nudges_on_at)
    bus.subscribe("perception.observation", waste.on_observation, replay=True)
    bus.subscribe("perception.zone", waste.on_zone_event, replay=True)
    bus.subscribe("habits.nudge.raised", waste.on_nudge, replay=True)
    story = ImpactStory(waste, RuleBook.load(rules, "config"), factors or Factors.load("config/impact.yaml"), tz=TZ,
                        frames_dir=tmp_path / "frames")
    story.subscribe(bus)
    return bus, waste, story


def lived(tmp_path, events=WEEK, **kw):
    bus, waste, story = wired(tmp_path, **kw)

    async def run():
        for e in events:
            await bus.publish(e)

    asyncio.run(run())
    return bus, waste, story


# --- what it means, in real units ---


def test_equivalents_are_ranges_from_the_factors_file_and_say_they_are_estimates(tmp_path):
    _, waste, story = lived(tmp_path)
    eq = story.view(at(2, 20))["equivalents"]
    # three cans and a bottle reached recycling: 3 x (13.2..13.34 g) + (9.25..23.9 g)
    assert eq["items"] == 4 and eq["kept_out_g"] == {"low": 49, "high": 64}
    assert eq["estimate"] is True and "not measurements" in eq["disclaimer"] and "not UCF's own" in eq["disclaimer"]
    assert eq["energy_kwh"]["low"] < eq["energy_kwh"]["high"] and eq["co2e_kg"]["low"] < eq["co2e_kg"]["high"]
    # 39.6 g of aluminium x 44.5 kWh/kg + 9.25 g of PET x 8.4 kWh/kg, up to 40.02 g x 54.4 + 23.9 g x 10.3
    assert eq["energy_kwh"] == {"low": 1.84, "high": 2.42}
    # fewest charges: the least energy into the hungriest phone. 1.84 kWh / 19.4 Wh up to 2.42 kWh / 19 Wh (EPA: 0.019 kWh a charge)
    assert eq["phone_charges"] == {"low": 94, "high": 127}
    assert eq["phone_charge_wh"] == {"low": 19.0, "high": 19.4} and "epa.gov" in eq["phone_charge_url"] and "not electricity" in eq["energy_note"]
    can = next(r for r in eq["by_item"] if r["item"] == "can")
    assert can["count"] == 3 and can["material"] == "aluminium" and can["mass_confidence"] == "high"
    assert "WARM" in can["material_source"] and can["material_confidence"] == "medium" and can["mass_source"]
    # the factors travel with the row, so the screen can show count x mass x factor with its source
    assert can["each_g"] == {"low": 13.2, "high": 13.34} and can["energy_kwh_per_kg"] == {"low": 44.5, "high": 54.4}
    assert can["mass_url"].startswith("https://www.epa.gov/") and can["material_url"].startswith("https://www.epa.gov/")


def test_every_factor_has_a_source_and_a_confidence_and_every_range_is_a_range():
    raw = yaml.safe_load(Path("config/impact.yaml").read_text())
    for name, f in {**raw["items"], **raw["materials"]}.items():
        assert f["source"] and f["confidence"] in ("high", "medium", "low"), name
        for key in ("mass_g", "energy_kwh_per_kg", "co2e_kg_per_kg"):
            if key in f:
                assert 0 < f[key]["low"] < f[key]["high"], (name, key)
    for name, f in raw["items"].items():
        assert f["material"] == "mixed" or f["material"] in raw["materials"], name


def test_every_factor_cites_a_document_with_a_url_and_nothing_is_recalled_from_memory():
    """The track is judged on measurable impact: a factor with no openable source is not in the file at all."""
    text = Path("config/impact.yaml").read_text()
    for phrase in ("recalled offline", "from memory", "illustrative", "order of magnitude", "rule of thumb, not a measurement"):
        assert phrase not in text.lower(), phrase
    raw = yaml.safe_load(text)
    for name, f in {**raw["items"], **raw["materials"]}.items():
        assert "https://" in f["source"] and f["url"].startswith("https://") and f["url"] in f["source"], name
        assert f["confidence"] in ("high", "medium", "low"), name
    eq = raw["equivalents"]
    assert eq["phone_charge_url"] in eq["phone_charge_source"] and eq["phone_charge_confidence"] in ("high", "medium", "low")
    assert 0 < eq["phone_charge_wh"]["low"] <= eq["phone_charge_wh"]["high"]
    c = raw["contamination"]
    assert c["quote"] in c["text"] and c["url"] in c["source"] and c["confidence"] in ("high", "medium", "low")
    assert c["rule_of_thumb"] is False and c["threshold"] is None  # a quoted statement, and no invented number
    for r in c["rates"]:
        assert r["scope"] in ("local", "national") and "https://" in r["source"], r
    assert raw["problem"]["text"] and all("https://" in s["source"] and s["scope"] for s in raw["problem"]["sources"])
    assert set(raw["items"]) == {"can", "bottle"}  # the only items a published mass was found for


def test_the_ucf_rules_cite_ucfs_page_and_say_which_items_are_our_assumption():
    raw = yaml.safe_load(UCF.read_text())
    assert raw["source"]["url"] == "https://fo.ucf.edu/recycling-at-ucf/" and raw["source"]["read"]
    assert set(raw["basis"]) == set(raw["items"]) and set(raw["basis"].values()) == {"stated", "assumed"}
    assert {i for i, b in raw["basis"].items() if b == "assumed"} == {"cup", "food container", "chip bag"}
    # what UCF's page states outright
    assert raw["items"]["can"] == raw["items"]["bottle"] == raw["items"]["paper"] == raw["items"]["cardboard"] == "recycling"
    assert raw["items"]["wine glass"] == raw["items"]["napkin"] == raw["items"]["banana"] == "trash"
    summary = RuleBook.load(UCF, "config").summary()
    assert summary["source"]["publisher"] == "UCF Facilities Operations"
    assert {i["label"]: i["basis"] for i in summary["items"]}["chip bag"] == "assumed"
    glass = next(c for c in summary["compare"] if c["item"].startswith("Glass"))
    assert (glass["ucf"], glass["orange_county"], glass["orlando"]) == ("trash", "recycling", "recycling")
    assert RuleBook.load(HOME, "config").summary()["source"] is None  # the example file claims no source


def test_an_item_with_no_mass_is_counted_but_claims_nothing_and_no_factors_file_means_no_equivalents(tmp_path):
    f = Factors({"items": {"can": {"material": "aluminium", "mass_g": {"low": 13, "high": 15}}}, "materials": {}}, loaded=True)
    eq = f.equivalents(["can", "bottle"])
    assert eq["kept_out_g"] == {"low": 13, "high": 15} and eq["unweighed"] == [{"item": "bottle", "count": 1}]
    assert eq["energy_kwh"] == {"low": 0, "high": 0}  # the material has no factor: nothing is claimed
    assert Factors.load(tmp_path / "missing.yaml").equivalents(["can"]) is None


def test_the_cost_of_contamination_is_a_quoted_statement_with_no_invented_threshold(tmp_path):
    _, _, story = lived(tmp_path)
    c = story.view(at(2, 20))["contamination"]
    assert c["wrong_in_recycling"] == 1 and c["spoils_bag_at"] is None and c["at_risk"]  # no threshold: one wrong item is the risk
    assert c["quote"] == "Non-recyclable items can contaminate a whole load of recyclables, causing them to all be thrown out."
    assert c["attribution"].startswith("US EPA") and c["url"] == "https://www.epa.gov/recycle/frequent-questions-recycling"
    assert c["rule_of_thumb"] is False and c["confidence"] == "medium" and c["quote"] in c["text"]
    assert [r["scope"] for r in c["rates"]] == ["local", "national"]
    assert c["good_items_sharing_the_bin"] == 1  # Tuesday's one can shared the bin with the cup
    # a file with a threshold of its own still works, and an unsourced file is flagged as a rule of thumb
    old = Factors({"contamination": {"spoils_bag_at": 2, "text": "{n} wrong items can spoil a bag."}}, loaded=True).contamination(1, 0)
    assert old["spoils_bag_at"] == 2 and not old["at_risk"] and old["rule_of_thumb"] and old["text"].startswith("2 wrong items")


# --- the story the screen plays ---


def test_the_headline_and_the_flow_replay_the_weeks_disposals_in_order(tmp_path):
    _, waste, story = lived(tmp_path)
    v = story.view(at(2, 20))
    assert v["headline"] == "This week this table kept 4 items out of landfill and got 2 wrong."
    flow = v["flow"]
    assert [(i["item"], i["bin"], i["correct"]) for i in flow["items"]] == [
        ("can", "recycling", True), ("bottle", "recycling", True), ("napkin", "trash", True), ("cup", "recycling", False),
        ("can", "trash", False), ("can", "recycling", True), ("can", "recycling", True), ("food container", "trash", True)]
    assert flow["items"][3]["rule"].startswith("Cups go in trash at UCF") and flow["items"][3]["belongs_in"] == "trash"
    # where the recyclables went adds up to what Waste Watch counted
    assert (flow["recyclable_seen"], flow["recycled"], flow["recyclable_in_trash"], flow["not_binned"]) == (5, 4, 1, 0)
    assert flow["recycled"] == waste.metrics()["recycled"] and flow["wrong_in_recycling"] == waste.metrics()["contamination"] == 1
    assert v["problem"]["text"].startswith("UCF collects recycling single-stream") and len(v["problem"]["sources"]) == 5
    assert Factors({}, loaded=True).problem is None


def test_a_recyclable_that_never_reaches_a_bin_is_said_so_and_an_empty_week_has_its_own_headline(tmp_path):
    carried = ObservationRecorded(event_id="o-9", ts=at(0, 9), camera_id=CAM, frame_ref=None,
                                  tracks=[Track(track_id=9, label="bottle", confidence=0.9, box=BOX)])
    _, _, story = lived(tmp_path, [carried, *toss(at(0, 10), 1, "can", "recycling")])
    v = story.view(at(0, 20))
    assert (v["flow"]["recyclable_seen"], v["flow"]["recycled"], v["flow"]["not_binned"]) == (2, 1, 1)
    assert v["headline"] == "This week this table kept 1 item out of landfill and got none wrong."
    assert story.view(at(8, 9))["headline"] == "Nothing has gone into the bins at this table yet this week."


def test_the_loop_counts_what_the_log_holds_and_claims_a_trend_only_with_two_weeks(tmp_path):
    def played(n, hour, item, correct):
        return RoundScored(event_id=f"r{n}", ts=at(2, hour), round_id=f"r{n}", session_id="s", players=1, item=item, correct=correct,
                           seconds=4, path_len=3, score=80 if correct else 0)

    rounds = [played(1, 13, "cup", True), played(2, 14, "soda can", False), played(3, 15, "napkin", True)]
    bus, _, story = lived(tmp_path, [*WEEK, PracticeAdded(ts=at(2, 12), label="cup"), *rounds])
    loop = story.view(at(2, 20))["loop"]
    assert loop["seen"] == 2 and loop["taught"] == 1  # the cup and the can; "teach me" was tapped on the cup
    assert loop["practised"] == {"rounds": 2, "right": 1}  # the napkin round is not practice: nobody got a napkin wrong
    assert loop["weeks"] == [{"week_start": "2026-10-05", "disposals": 8, "mistakes": 2, "by_item": {"cup": 1, "can": 1}}]
    assert not loop["trend"]["available"] and loop["trend"]["previous"] is None and "second week" in loop["trend"]["text"]

    # the next week, with one mistake: now there is a week to compare with, and it is still called a direction
    async def more():
        for e in [*toss(at(7, 10), 30, "cup", "recycling"), *toss(at(7, 11), 31, "can", "recycling")]:
            await bus.publish(e)

    asyncio.run(more())
    loop = story.view(at(8, 9))["loop"]
    assert loop["trend"] == {"available": True, "previous": 2, "current": 1,
                             "text": "1 wrong this week against 2 last week (measured; small numbers, so a direction, not a result)."}
    assert [w["mistakes"] for w in loop["weeks"]] == [2, 1]


# --- the moment of disposal ---


def test_the_feed_is_newest_first_with_the_rule_for_a_wrong_one(tmp_path):
    _, waste, story = lived(tmp_path)
    feed = story.view(at(2, 20))["feed"]
    assert [f["item"] for f in feed] == ["food container", "can", "can", "can", "cup", "napkin", "bottle", "can"]
    cup = next(f for f in feed if f["item"] == "cup")
    assert (cup["bin"], cup["correct"], cup["belongs_in"]) == ("recycling", False, "trash")
    assert cup["rule"] == "Cups go in trash at UCF: UCF does not recycle plastic cups, and it does not list paper cups, so they stay out too."
    assert cup["frame_ref"].startswith("frames:cam-door:") and cup["frame_stored"] is False and cup["in_practice"]
    assert all(f["correct"] == waste.disposed[f["item_id"]].correct for f in feed)  # Waste Watch stays the judge
    # a stored keyframe is said to be stored
    camera, ms = cup["frame_ref"].split(":")[1:]
    (tmp_path / "frames" / camera).mkdir(parents=True)
    (tmp_path / "frames" / camera / f"{ms}.jpg").write_bytes(b"jpeg")
    assert next(f for f in story.feed(at(2, 20)) if f["item"] == "cup")["frame_stored"] is True


def test_with_owner_recognition_a_disposal_belongs_to_the_one_recognised_person_there(tmp_path):
    events = [IdentityChanged(ts=at(0, 9), enabled=True, members=["hem", "maya"]),
              PersonIdentified(ts=at(0, 9, 59), camera_id=CAM, track_id=90, member_id="hem"),
              *toss(at(0, 10), 1, "cup", "recycling"),
              PersonIdentified(ts=at(0, 10, 30), camera_id=CAM, track_id=91, member_id="maya"),
              *toss(at(0, 11), 2, "can", "trash"),  # two recognised people there: the table's
              PersonLost(ts=at(0, 11, 30), camera_id=CAM, track_id=90),
              *toss(at(0, 12), 3, "can", "recycling")]
    _, _, story = lived(tmp_path, events)
    v = story.view(at(0, 20))
    assert v["scope"]["per_person"] and v["scope"]["members"] == ["hem", "maya"]
    assert {f["item_id"]: f["member_id"] for f in v["feed"]} == {"cam-door:1": "hem", "cam-door:2": None, "cam-door:3": "maya"}
    assert [f["item"] for f in story.view(at(0, 20), member="hem")["feed"]] == ["cup"]
    assert [p["item"] for p in story.practice(at(0, 20), member="hem")["items"]] == ["cup"]
    assert [p["item"] for p in story.practice(at(0, 20))["items"]] == ["can", "cup"]
    # off: everything is the table's, and the screen says so
    _, _, plain = lived(tmp_path / "plain")
    assert not plain.view(at(2, 20))["scope"]["per_person"] and "table" in plain.view(at(2, 20))["scope"]["note"]


# --- goals and momentum ---


def test_the_weekly_goal_clean_days_best_day_and_projection(tmp_path):
    _, waste, story = lived(tmp_path)
    v = story.view(at(2, 20))
    assert v["week"] == {"start": "2026-10-05", "end": "2026-10-11", "today": "2026-10-07"}
    assert v["this_week"] == waste.metrics() == v["total"]  # the same numbers Waste Watch gives
    g = v["goal"]["diversion"]
    assert (g["recycled"], g["recyclable"], g["value"], g["target"], g["met"]) == (4, 5, 0.8, 0.8, True)
    assert v["goal"]["contamination"] == {"ceiling": 2, "value": 1, "met": True, "left": 1}
    assert [d["clean"] for d in v["days"][:3]] == [True, False, True] and v["clean_streak"] == 1  # Tuesday broke Monday's
    assert v["best_day"]["day"] == "2026-10-05" and v["best_day"]["right_share"] == 1.0  # both clean days tie; Monday had more items
    p = v["projection"]
    assert p["enough"] and p["days_with_data"] == 3 and p["days_left"] == 4 and p["recycled_by_week_end"] == round(4 + 4 / 3 * 4)
    # the view is the week as it stood at `now`: on Monday evening there is one day of data, too early to project
    monday = story.view(at(0, 20))
    assert not monday["projection"]["enough"] and "Too early" in monday["projection"]["text"] and monday["this_week"]["items_seen"] == 3
    # and on Tuesday, after the can went in trash, the goal says how far it is
    tuesday = story.view(at(1, 10, 30))["goal"]["diversion"]
    assert (tuesday["recycled"], tuesday["recyclable"], tuesday["met"], tuesday["to_go"]) == (2, 3, False, 2)  # 2 of 3 -> 4 of 5
    assert "The next 2 in the right bin, with none missed, reaches 80%" in tuesday["text"]


def test_before_and_after_nudges_carry_n_and_say_not_enough_data_under_the_minimum(tmp_path):
    nudge = NudgeRaised(ts=at(1, 8), nudge_id="n1", habit_id="sort_right", message="sort")
    _, _, story = lived(tmp_path, sorted([*WEEK, nudge], key=lambda e: e.ts))
    ba = story.view(at(2, 20))["before_after"]
    assert ba["available"] and not ba["enough"] and ba["before"]["n"] == 2 and ba["after"]["n"] == 3
    assert ba["text"].startswith("Not enough data yet: 2 recyclables before") and ba["min_n"] == 5
    # with five on each side the bars are drawn, each with its n
    many = [*[e for i in range(5) for e in toss(at(0, 10, i), 100 + i, "can", "recycling" if i < 2 else "trash")], nudge,
            *[e for i in range(5) for e in toss(at(1, 10, i), 200 + i, "can", "recycling" if i < 4 else "trash")]]
    _, _, enough = lived(tmp_path / "enough", many)
    ba = enough.view(at(2, 20))["before_after"]
    assert ba["enough"] and (ba["before"]["n"], ba["before"]["diversion_rate"]) == (5, 0.4)
    assert (ba["after"]["n"], ba["after"]["diversion_rate"]) == (5, 0.8)
    # and with no sorting nudge there is nothing to compare
    assert lived(tmp_path / "none")[2].view(at(2, 20))["before_after"]["available"] is False


def test_milestones_are_published_once_and_a_restart_does_not_say_them_again(tmp_path):
    bus, _, story = lived(tmp_path)
    said = [e for e in bus.read_log() if e.type == "impact.milestone.reached"]
    assert [(m.threshold, m.value, m.text) for m in said] == [
        (1, 1, "First item kept out of landfill this week."), (3, 3, "That's 3 items kept out of landfill this week.")]
    assert all(isinstance(m, MilestoneReached) and m.week_start == MONDAY and m.kind == "kept_out" for m in said)
    assert said[0].ts == at(0, 10, 0, 1)  # event time: the disposal that made it
    v = story.view(at(2, 20))
    assert v["milestones"]["latest"]["threshold"] == 3 and v["milestones"]["next"] == {
        "kind": "kept_out", "threshold": 5, "to_go": 1, "text": "1 more in recycling makes 5 this week."}

    again_bus, _, again = wired(tmp_path)  # the same log, a restarted box
    assert asyncio.run(again_bus.replay_into()) == len(list(bus.read_log()))
    assert len([e for e in again_bus.read_log() if e.type == "impact.milestone.reached"]) == 2  # nothing new was said
    assert again.view(at(2, 20)) == v
    # the next one is said live, once, after the restart
    asyncio.run(again_bus.publish_all(toss(at(3, 9), 9, "bottle", "recycling")))
    assert [e.threshold for e in again_bus.read_log() if e.type == "impact.milestone.reached"] == [1, 3, 5]
    assert again.celebrate(ItemDisposed(ts=at(3, 9, 5), item_id="x", label="can", zone="recycling", bin="recycling", correct=True)) == []


def test_a_clean_streak_is_a_milestone_too(tmp_path):
    days = [e for d in range(3) for e in toss(at(d, 10), 10 + d, "napkin", "trash")]
    bus, _, story = lived(tmp_path, days)
    (m,) = [e for e in bus.read_log() if e.type == "impact.milestone.reached"]
    assert (m.kind, m.threshold, m.text) == ("clean_streak", 3, "3 clean days in a row: nothing in the wrong bin.")
    assert story.clean_streak(at(4)) == 3  # a day with nothing thrown away neither counts nor breaks it


# --- linked to the rest ---


def test_the_practice_set_ranks_the_weeks_mistakes_and_teach_me_adds_to_it(tmp_path):
    bus, _, story = lived(tmp_path, [*WEEK, *toss(at(2, 12), 20, "cup", "recycling")])
    p = story.practice(at(2, 20))
    assert [(r["item"], r["mistakes"], r["went_to"], r["belongs_in"], r["zone"]) for r in p["items"]] == [
        ("cup", 2, "recycling", "trash", "trash"), ("can", 1, "trash", "recycling", "recycling")]
    assert p["size"] == 2 and p["week_start"] == "2026-10-05" and p["rules"]["file"] == "config/sorting.yaml"
    assert p["items"][0]["rule"].startswith("Cups go in trash at UCF")

    app = FastAPI()
    app.include_router(impact_router(story, publish=bus.publish, clock=lambda: at(2, 20)))
    c = TestClient(app)
    assert c.get("/api/impact/practice").json() == p
    got = c.post("/api/impact/practice", json={"label": "Chip Bag"}).json()
    assert got["added"] == "chip bag" and [r["item"] for r in got["practice"]["items"]] == ["cup", "can", "chip bag"]
    (added,) = [e for e in bus.read_log() if e.type == "impact.practice.added"]
    assert isinstance(added, PracticeAdded) and added.label == "chip bag"
    assert c.post("/api/impact/practice", json={"label": "unicorn"}).status_code == 422
    assert c.get("/api/impact/story").json()["practice"]["size"] == 3
    # next week starts empty
    assert story.practice(at(7, 9))["items"] == []
    # and a box without Waste Watch says so
    off = FastAPI()
    off.include_router(impact_router(None))
    assert TestClient(off).get("/api/impact/story").status_code == 404


def test_the_league_gets_the_weeks_milestone_and_the_practice_size_and_the_review_one_clause(tmp_path):
    from contexts.play.league.tests.helpers import league

    _, _, story = lived(tmp_path)
    assert story.week_brief(at(2, 20)) == {"milestone": "That's 3 items kept out of landfill this week.", "practice_size": 2}
    lg = league()
    assert (lg.view(at(2, 20))["team_waste"]["milestone"], lg.view(at(2, 20))["team_waste"]["practice_size"]) == (None, None)
    lg.impact = story
    team = lg.view(at(2, 20))["team_waste"]
    assert team["milestone"] == "That's 3 items kept out of landfill this week." and team["practice_size"] == 2
    assert team["line"].startswith("team:")  # the line itself is unchanged

    assert story.day_clause(MONDAY) == "all 3 things you threw away went in the right bin"
    assert story.day_clause(MONDAY + timedelta(days=1)) == (
        "1 of 3 things you threw away went in the right bin; the cup belonged in trash (and 1 more)")
    assert story.day_clause(MONDAY + timedelta(days=5)) is None


# --- another campus swaps one rules file ---


def test_the_same_event_stream_gets_different_verdicts_and_explanations_under_two_rules_files(tmp_path):
    stream = [*toss(at(0, 10), 1, "cup", "recycling"), *toss(at(0, 11), 2, "food container", "recycling"),
              *toss(at(0, 12), 3, "milk carton", "recycling"), *toss(at(0, 13), 4, "can", "recycling"),
              *toss(at(0, 14), 5, "napkin", "recycling")]
    _, ucf_waste, ucf = lived(tmp_path / "ucf", stream, rules=UCF)
    _, home_waste, home = lived(tmp_path / "home", stream, rules=HOME)

    def verdicts(story):
        return {f["item"]: (f["correct"], f["rule"]) for f in story.view(at(0, 20))["feed"]}

    a, b = verdicts(ucf), verdicts(home)
    assert a["cup"] == (False, "Cups go in trash at UCF: UCF does not recycle plastic cups, and it does not list paper cups, so they stay out too.")
    assert b["cup"] == (True, "Paper cups go in recycling at home: this kerbside programme accepts them empty, lids off.")
    assert a["food container"][0] is False and "go in trash at UCF" in a["food container"][1]
    assert b["food container"][0] is True and "go in recycling at home once they are rinsed" in b["food container"][1]
    assert b["milk carton"] == (True, "Milk and juice cartons go in recycling at home: this programme takes cartons, rinsed.")
    assert "milk carton" not in a  # not on UCF's list: Waste Watch does not track it there at all
    assert a["can"][0] is b["can"][0] is True and a["napkin"][0] is b["napkin"][0] is False  # what the two agree on
    assert a["can"][1] != b["can"][1]  # ... in each place's own words

    differing = {i for i in RuleBook.load(HOME).rules.items if RuleBook.load(UCF).rules.bin_for(i) != RuleBook.load(HOME).rules.bin_for(i)}
    assert differing == {"cup", "food container", "milk carton", "juice box"}  # at least three items sorted differently
    assert ucf_waste.metrics()["contamination"] == 3 and home_waste.metrics()["contamination"] == 1
    assert [p["item"] for p in ucf.practice(at(0, 20))["items"]] == ["cup", "food container", "napkin"]
    assert [p["item"] for p in home.practice(at(0, 20))["items"]] == ["napkin"]

    # the screen says which file is active, and that a profile's file replaced the base one
    assert ucf.view(at(0, 20))["rules"]["file"] == "config/sorting.yaml" and not ucf.view(at(0, 20))["rules"]["overridden"]
    r = home.view(at(0, 20))["rules"]
    assert (r["file"], r["overridden"], r["place"]) == ("config/profiles/family-home/sorting.yaml", True, "home")
    assert "not a real city's" in r["note"]


def test_both_rules_files_are_complete_every_item_has_a_bin_zone_and_a_sentence():
    zones = yaml.safe_load(Path("config/zones.yaml").read_text())["cam-door"]
    for path in (UCF, HOME):
        book = RuleBook.load(path, "config")
        assert set(book.rules.bins.values()) <= set(zones), path
        for item, bin_ in book.rules.items.items():
            assert bin_ in book.rules.bins and item in book.reasons and book.reasons[item].endswith("."), (path, item)
        assert book.default_reason and book.name and book.place
        assert book.reason("soda can") == book.reasons["can"]  # an alias is explained like its item
        assert book.reason("traffic cone") == book.default_reason


def test_a_rules_file_without_sentences_still_explains_itself(tmp_path):
    (tmp_path / "sorting.yaml").write_text(yaml.safe_dump({"bins": {"recycling": "recycling", "trash": "trash"}, "items": {"can": "recycling"}}))
    book = RuleBook.load(tmp_path / "sorting.yaml")
    assert book.reason("can") == "A can goes in recycling." and book.verdict("can", "trash")["correct"] is False


# --- honest empty state ---


def test_with_no_camera_data_the_view_explains_what_will_appear(tmp_path):
    _, _, story = wired(tmp_path)
    v = story.view(at(2, 20))
    assert v["empty"] and "When a camera sees a can" in v["empty_text"]
    assert v["feed"] == [] and v["practice"]["size"] == 0 and v["milestones"]["reached"] == []
    assert v["goal"]["diversion"]["value"] is None and v["goal"]["diversion"]["text"] == "Nothing recyclable seen yet this week."
    assert v["equivalents"]["kept_out_g"] == {"low": 0, "high": 0} and v["best_day"] is None and v["clean_streak"] == 0
    assert not v["projection"]["enough"] and not v["before_after"]["available"]
    assert week_of(date(2026, 10, 11)) == MONDAY
