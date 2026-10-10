"""The projection on a real bus: the boundary in event time, replay identity, the cards, the events' bounds."""

import asyncio
from datetime import date, timedelta

import pytest

from bus import EventBus
from contexts.play.league.tests.helpers import (FakeCircle, FakeHub, MONDAY, NAMES, at, day_closed, league, shared,
                                                week_of_sleep)
from contexts.workspace.domain.approvals import Approvals
from contracts.agent import ActionProposed
from contracts.experience import ApprovalDecided
from contracts.impact import CountUpdated
from contracts.perception import ZoneEvent
from contracts.play import LeagueWeekClosed, Stake, Standing

NEXT_MONDAY = MONDAY + timedelta(days=7)


def run(coro):
    return asyncio.run(coro)


def tick(ts):
    """Any event after the boundary: a zone event, like a camera would send."""
    return ZoneEvent(ts=ts, camera_id="cam-door", track_id=1, label="person", zone_id="door", kind="enter")


async def seeded(tmp_path, hub=None, circle=None):
    bus = EventBus(tmp_path / "events")
    lg = league(hub=hub, circle=circle)
    lg.subscribe(bus)
    for e in week_of_sleep(lg, "hemdutt", 6) + week_of_sleep(lg, "maya", 2):
        await bus.publish(e)
    return bus, lg


# --- the boundary ---


def test_the_week_closes_on_the_first_event_after_the_boundary_in_event_time_not_before(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        await bus.publish(tick(at(NEXT_MONDAY, 2, 59)))  # Monday 02:59 local: inside the grace
        assert lg.closed == {}
        await bus.publish(tick(at(NEXT_MONDAY, 3, 0)))
        [closed] = lg.closed.values()
        assert (closed.week_start, closed.week_end, closed.worst, closed.best) == (MONDAY, date(2026, 10, 11), "maya", "hemdutt")
        assert closed.summary == "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place."
        assert closed.ts == at(NEXT_MONDAY, 3, 0)  # event time, not the clock
        logged = [e for e in bus.read_log() if e.type == "play.league.week.closed"]
        assert len(logged) == 1 and logged[0].event_id == "league-week-team-2026-10-05"
        await bus.publish(tick(at(NEXT_MONDAY, 9)))  # later events never close it again
        assert len([e for e in bus.read_log() if e.type == "play.league.week.closed"]) == 1

    run(go())


def test_the_ticker_closes_a_week_the_clock_passed_with_no_event(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        out = lg.flush(at(NEXT_MONDAY, 4))
        assert [e.type for e in out] == ["play.league.week.closed"] and lg.flush(at(NEXT_MONDAY, 5)) == []

    run(go())


def test_a_week_with_nothing_judged_is_not_declared(tmp_path):
    async def go():
        bus = EventBus(tmp_path / "events")
        lg = league()
        lg.subscribe(bus)
        await bus.publish(tick(at(MONDAY - timedelta(days=20))))
        await bus.publish(tick(at(NEXT_MONDAY, 12)))
        assert lg.closed == {} and not [e for e in bus.read_log() if e.type.startswith("play.league")]

    run(go())


def test_replaying_the_log_rebuilds_the_same_standings_and_declares_nothing_twice(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        await bus.publish(tick(at(NEXT_MONDAY, 3, 30)))
        await bus.publish(day_closed("meals", NEXT_MONDAY, True, 3))  # the new week is under way
        live = (lg.view(at(NEXT_MONDAY, 12)), lg.me_view(at(NEXT_MONDAY, 12)), lg.closed[MONDAY])
        n_before = sum(1 for _ in bus.read_log())

        again = EventBus(tmp_path / "events")
        lg2 = league()
        lg2.subscribe(again)
        await again.replay_into()
        assert (lg2.view(at(NEXT_MONDAY, 12)), lg2.me_view(at(NEXT_MONDAY, 12)), lg2.closed[MONDAY]) == live
        assert sum(1 for _ in again.read_log()) == n_before  # replay wrote nothing
        assert lg2.closed[MONDAY].standings == live[2].standings

    run(go())


# --- what the closed week carries ---


def test_the_closed_week_carries_standings_lines_only_never_values_or_raw_signals(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        await bus.publish(tick(at(NEXT_MONDAY, 4)))
        closed = lg.closed[MONDAY]
        dumped = closed.model_dump(mode="json")
        assert set(dumped) == {"event_id", "type", "version", "ts", "circle_id", "week_start", "week_end", "standings", "worst",
                               "best", "stake", "summary"}
        for s in dumped["standings"]:
            assert set(s) == set(Standing.model_fields)
            assert all(isinstance(r, str) and len(r) < 120 for r in s["reasons"])
        assert "4.5" not in closed.model_dump_json() and "value" not in closed.model_dump_json()  # the hours slept never travel
        assert len(dumped["standings"]) == 3 and dumped["standings"][2]["score"] is None  # dom: nothing judged

    run(go())


# --- follow-ups: approval cards, never sends ---


def test_closing_the_week_proposes_the_group_post_and_the_dinner_plan_and_approves_nothing(tmp_path):
    async def go():
        hub, circle = FakeHub(), FakeCircle()
        bus, lg = await seeded(tmp_path, hub=hub, circle=circle)
        approvals = Approvals({"circle_share": lg.executor, "connector_write": lambda r: "sent"})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        await bus.publish(tick(at(NEXT_MONDAY, 4)))
        cards = {r.kind: r for r in approvals.pending()}
        assert set(cards) == {"connector_write", "circle_share"}
        post = cards["connector_write"]
        assert post.target == "groupme.post" and post.params["group"] == "friends"
        assert post.params["text"] == ("League, week of Oct 5: 1. Hemdutt 86%, 2. Maya 29%, 3. Dominick (nothing judged) "
                                       "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place.")
        plan = cards["circle_share"]
        assert plan.params["kind"] == "event_proposal"
        assert plan.params["proposal"]["title"] == "Dinner on Maya, Hemdutt picks the place"
        assert plan.params["proposal"]["start"] == "2026-10-16T19:00:00-04:00"  # the coming Friday evening
        assert sorted(plan.params["proposal"]["invitees"]) == ["dom", "maya"]
        assert all(r.status == "proposed" for r in approvals.pending()) and hub.calls == [] and circle.posted == []

    run(go())


def test_only_the_best_weeks_box_proposes_the_follow_ups(tmp_path):
    async def go():
        hub = FakeHub()
        bus = EventBus(tmp_path / "events")
        lg = league(me="maya", hub=hub, circle=FakeCircle(member_id="maya"))  # Maya's box
        lg.subscribe(bus)
        for e in week_of_sleep(lg, "maya", 2) + week_of_sleep(lg, "hemdutt", 6):
            await bus.publish(e)
        await bus.publish(tick(at(NEXT_MONDAY, 4)))
        assert lg.closed[MONDAY].best == "hemdutt"
        assert not [e for e in bus.read_log() if e.type == "agent.action.proposed"]

    run(go())


def test_picks_the_event_names_the_winner_and_a_custom_stake_proposes_only_the_post(tmp_path):
    async def go():
        hub = FakeHub()
        bus, lg = await seeded(tmp_path, hub=hub, circle=FakeCircle())
        lg.stakes[MONDAY] = Stake(kind="picks_the_event")
        out = lg.on_week_closed_live(lg._close_week(MONDAY, at(NEXT_MONDAY, 4)))
        assert [a.kind for a in out] == ["connector_write", "circle_share"]
        assert out[1].params["proposal"]["title"] == "Hemdutt's pick: the next outing"
        lg.stakes[MONDAY] = Stake(kind="custom", text="loser does the dishes")
        out = lg.on_week_closed_live(lg._close_week(MONDAY, at(NEXT_MONDAY, 4)))
        assert [a.kind for a in out] == ["connector_write"] and out[0].params["text"].endswith("loser does the dishes.")

    run(go())


def test_without_groupme_or_a_circle_nothing_is_proposed(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path, hub=FakeHub(writes=()), circle=None)
        await bus.publish(tick(at(NEXT_MONDAY, 4)))
        assert lg.closed and not [e for e in bus.read_log() if e.type == "agent.action.proposed"]

    run(go())


# --- the stake ---


def test_setting_the_stake_is_a_card_and_the_stake_changes_only_when_it_is_approved(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        approvals = Approvals({"circle_share": lg.executor})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        card = lg.propose_stake(Stake(kind="picks_the_event"), at(MONDAY, 9))
        assert card.kind == "circle_share" and card.target == "team" and card.params["kind"] == "league_stake"
        assert card.reason == "Set the League stake from the week of Oct 5: The best week picks the next outing."
        await bus.publish(card)
        assert lg.stake_for(MONDAY).kind == "pays_for_food" and [r.status for r in approvals.pending()] == ["proposed"]
        assert not [e for e in bus.read_log() if e.type == "play.league.stake.set"]

        await bus.publish(ApprovalDecided(ts=at(MONDAY, 9, 1), action_id=card.action_id, decision="approve"))
        assert lg.stake_for(MONDAY).kind == "picks_the_event" and lg.stake_for(NEXT_MONDAY).kind == "picks_the_event"
        [e] = [e for e in bus.read_log() if e.type == "play.league.stake.set"]
        assert (e.circle_id, e.week_start, e.set_by, e.stake.kind) == ("team", MONDAY, "hemdutt", "picks_the_event")
        assert approvals.requests[card.action_id].result == "League stake set: The best week picks the next outing."

    run(go())


def test_a_denied_stake_changes_nothing_and_a_custom_stake_needs_its_text(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        approvals = Approvals({"circle_share": lg.executor})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        card = lg.propose_stake(Stake(kind="custom", text="dishes"), at(MONDAY, 9))
        await bus.publish(card)
        await bus.publish(ApprovalDecided(ts=at(MONDAY, 9, 1), action_id=card.action_id, decision="deny"))
        assert lg.stakes == {} and approvals.requests[card.action_id].status == "denied"
        with pytest.raises(ValueError):
            lg.propose_stake(Stake(kind="custom"), at(MONDAY, 9))

    run(go())


def test_the_executor_hands_event_proposals_on_to_the_circles_own_executor(tmp_path):
    async def go():
        circle = FakeCircle()
        bus, lg = await seeded(tmp_path, circle=circle)
        lg.fallback = circle.share_executor
        approvals = Approvals({"circle_share": lg.executor})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        from contracts.circle import Proposal

        card = circle.propose(Proposal(title="Lunch", start=at(MONDAY, 12), end=at(MONDAY, 13)), "lunch")
        await bus.publish(card)
        await bus.publish(ApprovalDecided(ts=at(MONDAY, 9, 1), action_id=card.action_id, decision="approve"))
        assert [r.params["kind"] for r in circle.posted] == ["event_proposal"]
        lg.fallback = None
        await bus.publish(ActionProposed(ts=at(MONDAY, 10), action_id="x1", kind="circle_share", target="team", reason="?",
                                         params={"kind": "something_else"}))
        await bus.publish(ApprovalDecided(ts=at(MONDAY, 10, 1), action_id="x1", decision="approve"))
        assert approvals.requests["x1"].status == "failed"

    run(go())


# --- dares ---


def test_a_dare_is_a_card_then_judged_from_the_daily_summaries_and_declared_at_its_end(tmp_path):
    async def go():
        hub = FakeHub()
        bus, lg = await seeded(tmp_path, hub=hub)
        lg.clock = lambda: at(MONDAY, 18, 1)  # the executor stamps the dare with the box's clock at approval time
        approvals = Approvals({"circle_share": lg.executor, "connector_write": lambda r: "sent"})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        card = lg.propose_challenge("no_energy_drinks", 3, ["maya", "hemdutt"], at(MONDAY, 18))
        assert card.reason == "Start a League dare: No energy drinks for 3 days, inviting Maya"
        c = card.params["challenge"]
        assert (c["starts"], c["ends"], c["members"], c["direction"], c["target"]) == ("2026-10-06", "2026-10-08", ["hemdutt", "maya"], "at_most", 0)
        await bus.publish(card)
        assert lg.challenges == {}
        await bus.publish(ApprovalDecided(ts=at(MONDAY, 18, 1), action_id=card.action_id, decision="approve"))
        assert list(lg.challenges) == [c["challenge_id"]]
        assert lg.challenges_view(at(MONDAY, 19))[0]["status"] == "upcoming"

        for i in range(3):
            day = MONDAY + timedelta(days=1 + i)
            await bus.publish(day_closed("energy_drinks", day, True, 0))  # I had none
            await bus.publish(shared("maya", "energy_drinks", day, True, 1 if i == 1 else 0))  # Maya slipped on day 2
        view = lg.challenges_view(at(MONDAY + timedelta(days=4), 1))
        assert view[0]["status"] == "running"
        assert [(m["member_id"], m["met_days"], m["judged_days"]) for m in view[0]["members"]] == [("hemdutt", 3, 3), ("maya", 2, 3)]

        await bus.publish(tick(at(MONDAY + timedelta(days=4), 3)))  # the day after it ends, past the grace
        [closed] = lg.challenge_results.values()
        assert closed.winners == ["hemdutt"] and closed.summary == "No energy drinks, 3 days: Hemdutt made it."
        assert [(r.member_id, r.met_days, r.done) for r in closed.results] == [("hemdutt", 3, True), ("maya", 2, False)]
        post = [r for r in approvals.pending() if r.kind == "connector_write"]
        assert len(post) == 1 and post[0].params["text"] == "League dare over. No energy drinks, 3 days: Hemdutt made it."
        assert lg.challenges_view(at(MONDAY + timedelta(days=5)))[0]["status"] == "closed"

    run(go())


def test_a_dare_checks_its_preset_days_and_invitees():
    lg = league()
    with pytest.raises(ValueError, match="unknown dare"):
        lg.propose_challenge("skydiving", 3, [], at(MONDAY))
    with pytest.raises(ValueError, match="days"):
        lg.propose_challenge("sleep_six", 0, [], at(MONDAY))
    with pytest.raises(ValueError, match="not in the circle"):
        lg.propose_challenge("sleep_six", 3, ["stranger"], at(MONDAY))
    card = lg.propose_challenge("sleep_six", 5, [], at(MONDAY), target=7)
    assert card.params["challenge"]["target"] == 7 and card.params["challenge"]["members"] == ["hemdutt"]


# --- other sources ---


def test_day_run_stars_come_from_the_committed_plans_calendar_card_and_counts_from_waste_watch(tmp_path):
    async def go():
        bus, lg = await seeded(tmp_path)
        await bus.publish(ActionProposed(ts=at(MONDAY, 8), action_id="dr1", kind="connector_write", target="calendar.create_event",
                                         reason="Add 3 Day Run block(s) for Mon Oct 5 (3-star plan)", params={"events": []}))
        await bus.publish(ActionProposed(ts=at(MONDAY, 9), action_id="other", kind="send_message", target="dom", reason="(3-star plan)"))
        await bus.publish(CountUpdated(ts=at(MONDAY, 10), items_seen=5, recyclable_seen=4, recycled=4, contamination=0, diversion_rate=1))
        me = lg.me_view(at(MONDAY, 12))
        by = {b["target"]: b for b in me["breakdown"]}
        assert by["dayrun_stars"] == {"target": "dayrun_stars", "label": "Day Run stars", "unit": "plans", "weight": 1, "met": 1,
                                      "judged": 1, "share": 1.0, "detail": None}
        assert by["recycling"]["met"] == 1 and "100%" in by["recycling"]["detail"]

    run(go())


def test_summaries_from_another_circle_are_ignored_and_a_friends_name_is_learned(tmp_path):
    async def go():
        bus = EventBus(tmp_path / "events")
        lg = league(names={})
        lg.subscribe(bus)
        await bus.publish(shared("maya", "sleep", MONDAY, True, 7, circle_id="elsewhere", name="Maya"))
        assert lg.habits == {} and "maya" not in lg.names
        await bus.publish(shared("maya", "sleep", MONDAY, True, 7, name="Maya"))
        assert lg.names["maya"] == "Maya" and lg.view(at(MONDAY, 12))["standings"][0]["display_name"] == "Maya"

    run(go())
