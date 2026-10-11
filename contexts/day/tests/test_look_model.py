"""The look memory, stage two: what Atlas learns from the looks it remembers, that it says so honestly, and that it
stays on the box. The learner is a per-person table of counts over a handful of remembered moments
(contexts/day/domain/look_model.py). It is not an expression recogniser and trains none.

Every value below is invented for the test: nothing here was measured on anybody."""

import asyncio
import json
import logging
import random
import re
from datetime import timedelta
from pathlib import Path

from contexts.day.application.loop import checkin_push
from contexts.day.domain import checkin as rule
from contexts.day.domain import look_model as lm
from contexts.day.domain import looks as memory
from contexts.day.domain.narration import BOUNDS, DIAGNOSIS, check, render
from contexts.day.tests.test_checkin import NOW, afternoon, at, box, of
from contexts.day.tests.test_day import TZ, places  # noqa: F401
from contexts.day.tests.test_looks import FRIDAY, MEMBER, app_of, friday, scanned, tuesday, windows
from contexts.perception.domain.scenes import scene
from contracts.day import CheckinSuggestion, DayCheckinRaised, DayNarrated

# ---------- the model (pure) ----------


def row(did: str | None = "up", eased: bool | None = True, known: bool = True, look: str = "tired", **features) -> dict:
    """One labelled example, in the shape of looks.labelled_examples: what the person did, and whether the look eased."""
    outcome = {"known": known, "got_up": {"up": True, "drink": False, "stay": False}.get(did), "away_min": None,
               "drink": True if did == "drink" else None, "eased": eased if known else None, "answer": None}
    return {"look": look, "features": {"minute_of_day": None, "weekday": 1, "sitting_min": None, "since_drink_min": None, "energy_drinks": None,
                                       "energy_limit": None, "rhr_up": None, "short_night": None, "activity": None, "doing": None,
                                       "deadline": None, "deadline_hours": None} | features, "advice": [], "outcome": outcome}


def rows(up: tuple[int, int] = (0, 0), drink: tuple[int, int] = (0, 0), stay: tuple[int, int] = (0, 0), **features) -> list[dict]:
    """`(eased, seen)` for each arm, as that many examples."""
    return [row(did, i < k, **features) for did, (k, n) in (("up", up), ("drink", drink), ("stay", stay)) for i in range(n)]


SEEDED = rows(up=(5, 6), drink=(0, 2), stay=(0, 2))  # the demo owner's two weeks (contexts/demo/checkin.py THEN)
ADVICE = [CheckinSuggestion(kind="drink", text="Drink some water"), CheckinSuggestion(kind="walk", text="Take ten minutes outside"),
          CheckinSuggestion(kind="move_block", text="Move the study block to after dinner")]


def test_every_number_is_a_count_of_what_the_person_was_seen_to_do():
    m = lm.fit(SEEDED + [row(known=False), row(None, True), row("up", None)])
    assert (m.total, m.known, m.unknown, m.eased, m.enough) == (13, 11, 2, 6, True)  # two left out: too soon, and one the box could not see ease
    up, drink, stay = (m.arms[k] for k in lm.ARMS)
    assert (up.eased, up.n, up.rest_eased, up.rest_n) == (5, 6, 1, 5) and (drink.eased, drink.n) == (0, 2) and (stay.eased, stay.n) == (0, 2)
    assert lm.did({"got_up": True, "drink": True}) == "up" and lm.did({"got_up": False, "drink": True}) == "drink"  # "water alone" is a drink without getting up
    assert lm.did({"got_up": False, "drink": None}) == "stay" and lm.did({"got_up": None, "drink": None}) is None
    # the smoothed rate: PRIOR pretend-observations at the person's own overall rate
    pooled = (6 + 1) / (11 + 2)
    assert m.pooled == pooled and up.rate == (5 + lm.PRIOR * pooled) / (6 + lm.PRIOR) and drink.rate == (lm.PRIOR * pooled) / (2 + lm.PRIOR)
    assert 0.5 < up.rate < 5 / 6 and 0 < drink.rate < pooled  # pulled towards the average: a lot with two looks, a little with six
    assert (round(up.low, 2), round(up.high, 2)) == (0.57, 0.95) and (round(drink.low, 2), round(drink.high, 2)) == (0.0, 0.45)  # the Wilson band


def test_a_prediction_is_explained_by_the_counts_in_the_same_kind_of_moment():
    """Getting up eased it after a long sit and not after a short one: the prediction for a long sit is higher, and
    the reason is two counts anybody can check."""
    examples = (rows(up=(4, 4), sitting_min=120) + rows(up=(0, 3), sitting_min=20) + rows(stay=(1, 3), sitting_min=120))
    m = lm.fit(examples)
    long_, short = lm.predict(m, {"sitting_min": 110}), lm.predict(m, {"sitting_min": 15})
    assert long_["up"] > lm.predict(m)["up"] > short["up"] and lm.predict(m)["up"] == m.arms["up"].rate
    assert lm.explain(m, "up", {"sitting_min": 110}) == ["4 of 7 overall", "4 of 4 with sit = long"]
    assert lm.explain(m, "up", {"sitting_min": 15}) == ["4 of 7 overall", "0 of 3 with sit = short"]
    assert long_["up"] == (4 + lm.PRIOR * m.arms["up"].rate) / (4 + lm.PRIOR)  # counts in the bin, pulled towards the arm's own rate
    # a feature the box did not know is left out, never guessed; an arm never seen sits at the person's own average
    assert lm.bins({"sitting_min": None, "minute_of_day": 9 * 60}) == {"part": "morning"}
    assert lm.predict(m)["drink"] == m.pooled and m.arms["drink"].n == 0


def test_the_advice_order_comes_from_the_counts_and_the_advice_set_is_kept():
    m = lm.fit(SEEDED)
    assert lm.lead(m) == "walk" and [s.kind for s in lm.rerank(ADVICE, m)] == ["walk", "drink", "move_block"]
    water = lm.fit(rows(up=(0, 4), drink=(4, 4)))  # another person: a drink is what was followed by the look easing
    assert lm.lead(water) == "drink" and [s.kind for s in lm.rerank(list(reversed(ADVICE)), water)] == ["drink", "walk", "move_block"]
    assert sorted(s.text for s in lm.rerank(ADVICE, m)) == sorted(s.text for s in ADVICE)  # the same advice, the same words
    same = lm.fit(rows(up=(2, 4), drink=(2, 4)))
    assert lm.lead(same) is None and lm.rerank(ADVICE, same) == ADVICE  # the counts do not tell them apart: the rule's own order


def test_below_the_minimum_it_says_not_enough_yet_and_nothing_else():
    few = lm.fit(rows(up=(3, 3)) + [row(known=False), row(known=False)])
    assert lm.MIN_KNOWN == 4 and (few.total, few.known, few.enough) == (5, 3, False)
    assert lm.learned_line(few) == "" and lm.pattern_line(few) == "" and lm.lead(few) is None and lm.rerank(ADVICE, few) == ADVICE
    assert lm.ahead_line(few, 90) == "" and not any(a.helps for a in few.arms.values())
    shown = lm.view(few)
    assert shown["enough"] is False and shown["findings"] == [] and shown["evidence"] == (
        "From 5 remembered looks, 3 with a known outcome. 2 are left out: I couldn't see how they went.")
    assert shown["not_enough"] == "Not enough yet. I need to see how 1 more look went before I say what helps (3 of 4 so far)."
    assert lm.said_helps(few) == "I don't know yet. I've seen how three looks went, and I need four before I'll say."
    assert lm.said_causes(few) == "Nothing stands out yet in the five looks I remember. I can't tell you what causes a look, only what they share."
    assert lm.said_causes(lm.fit(rows(up=(1, 1)))) == "I don't know yet. I remember one look, and I need five before I'll say what they share."
    assert lm.said_how(lm.Model()) == "I don't, yet: I haven't got a look remembered on this box."
    assert lm.fit(rows(up=(4, 4))).enough  # four known outcomes: the first time anything is said


def test_a_finding_needs_counts_that_chance_would_rarely_give():
    """Atlas always says the counts. The word "helps" needs Fisher's exact test to give ALPHA or less."""
    seeded = lm.fit(SEEDED)
    assert seeded.arms["up"].helps and round(seeded.arms["up"].p, 3) == 0.024 and lm.ALPHA == 0.05  # 5 of 6 against 0 of 4
    assert not seeded.arms["drink"].helps and not seeded.arms["stay"].helps
    assert lm.fisher(5, 6, 0, 4) == 6 / 252 and lm.fisher(3, 4, 0, 3) == 4 / 35 and lm.fisher(2, 2, 2, 2) == 1.0
    suggestive = lm.fit(rows(up=(3, 4), stay=(0, 3)))  # three of four against none of three: one time in nine by chance
    assert not suggestive.arms["up"].helps and lm.lead(suggestive) == "walk"  # still ranked first, never called a finding
    assert lm.learned_line(suggestive) == "Getting up has eased this three of the last four times; staying at the desk hasn't any of the three times."
    assert lm.view(suggestive)["findings"][0]["text"] == "Getting up: the look eased 3 of the 4 times you got up. Too few to call it either way."
    assert lm.said_helps(suggestive).startswith("I can't say for sure yet. The look eased three of the four times you got up")
    even = lm.fit(rows(up=(10, 20), drink=(10, 20)))  # plenty of looks, no difference: no finding however many there are
    assert not any(a.helps for a in even.arms.values())
    # nothing is ever called a finding the other way: "water alone hasn't" is a count
    never = lm.fit(rows(up=(9, 10), drink=(0, 10)))
    assert never.arms["up"].helps and never.arms["drink"].helps is False and "doesn't" not in json.dumps(lm.view(never))


def test_what_it_says_for_a_check_in_is_counts_in_plain_words():
    said = {
        "Getting up has eased this five of the last six times; water alone hasn't either time.": SEEDED,
        "Getting up has eased this five of the last six times; staying at the desk hasn't any of the four times.": rows(up=(5, 6), stay=(0, 4)),
        "Getting up has eased this each of the last four times.": rows(up=(4, 4)),
        "Getting up has eased this both times; water alone has one of two.": rows(up=(2, 2), drink=(1, 2)),
        "Water alone has eased this three of the last four times; getting up hasn't the one time.": rows(up=(0, 1), drink=(3, 4)) ,
        "Staying at the desk hasn't eased this any of the last four times.": rows(stay=(0, 4)),
    }
    for line, examples in said.items():
        assert lm.learned_line(lm.fit(examples)) == line
    assert lm.evidence_line(lm.fit(SEEDED)) == "From 10 remembered looks, 10 with a known outcome."
    assert lm.evidence_line(lm.fit(SEEDED[:1])) == "From 1 remembered look, 1 with a known outcome."


def sits(*minutes: int, **more) -> list[dict]:
    return [row("up", True, sitting_min=m, **more) for m in minutes]


def test_what_the_looks_share_is_said_as_a_count_and_only_when_it_is_strong():
    bunched = lm.fit(sits(114, 102, 126, 96, 105, 111, 90, 108, 93, 120, 99, 15, 15))  # the demo owner's, with two short ones
    [c] = bunched.commons
    assert (c.key, c.k, c.n, c.minutes) == ("sitting", 11, 13, 90) and bunched.onset_min == 90
    assert lm.pattern_line(bunched) == "It usually shows up after about ninety minutes at the desk."
    assert lm.said_causes(bunched) == ("I can't say what causes it, only what the looks share: eleven of the thirteen looks I remember "
                                       "started after ninety minutes or more at the desk.")
    assert lm.view(bunched)["findings"][-1]["text"] == "It usually starts after about 90 minutes at the desk: 11 of 13 looks did."
    assert "not a cause" in lm.view(bunched)["shared_note"]
    # spread out over the afternoon, too few, or always soon after sitting down: nothing is said about a usual time
    for minutes in ((20, 60, 100, 140, 180, 220, 240), (100, 105, 110, 95), (10, 12, 15, 20, 25, 8)):
        assert lm.fit(sits(*minutes)).onset_min is None, minutes
    # a yes/no feature: most of the looks, and more than a coin flip would give
    nights = lm.fit(sits(*[30] * 7, short_night=True) + sits(30, short_night=False))
    assert [(c.key, c.k, c.n) for c in nights.commons if c.key == "short_night"] == [("short_night", 7, 8)]
    assert lm.common_line(nights.commons[0]) == "It usually shows up after a short night."
    assert not [c for c in lm.fit(sits(*[30] * 6, short_night=True) + sits(30, 30, short_night=False)).commons if c.key == "short_night"]  # 6 of 8: a coin does that
    assert lm.coin(7, 8) == 9 / 256 and lm.coin(6, 8) == 37 / 256 and lm.COMMON_ALPHA == 0.05
    assert lm.fit(sits(*[30] * 9)).commons == ()  # unknown is not "no": nothing is said of a feature the box did not know
    afternoons = lm.fit(sits(*[30] * 8, minute_of_day=14 * 60) + sits(30, minute_of_day=9 * 60))
    assert [(c.key, c.part, c.k, c.n) for c in afternoons.commons] == [("part", "afternoon", 8, 9)]
    assert lm.common_line(afternoons.commons[0]) == "It usually shows up in the afternoon."
    assert lm.said_causes(afternoons) == "I can't say what causes it, only what the looks share: eight of the nine looks I remember started in the afternoon."


def test_the_cards_quiet_line_is_only_around_the_usual_time_and_only_with_a_finding():
    learned = lm.fit([*sits(114, 102, 126, 96, 105, 111, 90, 108, 93, 120)[:6], *rows(stay=(0, 4), sitting_min=100)])
    assert learned.onset_min == 90 and learned.arms["up"].helps
    line = "You usually start to look strained around now. A short break tends to help."
    assert [lm.ahead_line(learned, n) for n in (60, 74, 75, 90, 105, 120, 121, 200)] == ["", "", line, line, line, line, "", ""]
    assert lm.ahead_line(learned, None) == ""  # nobody at the desk
    unsure = lm.fit(sits(114, 102, 126, 96, 105, 111))  # a usual time, and no finding about what helps: one sentence, no advice
    assert unsure.onset_min == 90 and lm.ahead_line(unsure, 90) == "You usually start to look strained around now."
    assert lm.ahead_line(lm.fit(sits(20, 60, 100, 140, 180, 220, 240)), 90) == ""  # no usual time: nothing


def test_the_same_looks_give_the_same_model_in_any_order():
    examples = SEEDED + sits(114, 102, 126, 96, 105, short_night=True) + [row(known=False)]
    first = lm.fit(examples)
    for seed in range(5):
        shuffled = examples[:]
        random.Random(seed).shuffle(shuffled)
        again = lm.fit(shuffled)
        assert again == first and lm.view(again) == lm.view(first) and lm.learned_line(again) == lm.learned_line(first)
    assert lm.fit([]) == lm.Model(arms=lm.fit([]).arms) and lm.fit([]).total == 0 and not lm.fit([]).enough


def test_the_module_is_standard_library_only_and_reads_no_clock_no_dice_no_picture_and_no_network():
    text = Path(lm.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^(?:from|import) ([\w.]+)", text, re.M)
    assert sorted(imports) == ["contexts.day.domain", "contexts.day.domain.narration", "contracts.day", "dataclasses", "math"]
    code = text.split('"""', 2)[2]  # everything after the module's docstring
    for word in ("random", "datetime", "time.", "numpy", "torch", "socket", "http", "urllib", "requests", "open(", "frame", "embedding",
                 "template", "vector", "cv2", "logging", "member_id"):
        assert word not in code, word


def all_sentences(m: lm.Model) -> list[str]:
    shown = lm.view(m)
    return [lm.learned_line(m), lm.pattern_line(m), lm.evidence_line(m), lm.ahead_line(m, m.onset_min), lm.said_helps(m), lm.said_causes(m),
            lm.said_how(m), shown["evidence"], shown["not_enough"], shown["note"], shown["shared_note"],
            *(f["text"] + " " + f["band"] for f in shown["findings"]), *(lm.common_line(c) for c in m.commons)]


def test_no_sentence_it_can_build_is_a_diagnosis():
    rng = random.Random(7)
    models = [lm.fit(SEEDED), lm.fit([]), lm.fit(rows(up=(1, 1)))]
    for _ in range(300):
        n = rng.randrange(0, 30)
        models.append(lm.fit([row(rng.choice(("up", "drink", "stay", None)), rng.random() < 0.5, known=rng.random() < 0.8,
                                  sitting_min=rng.choice((None, 15, 95, 100, 110)), since_drink_min=rng.choice((None, 30, 120)),
                                  short_night=rng.choice((None, True, True, False)), rhr_up=rng.choice((None, True, True)),
                                  energy_drinks=rng.choice((None, 3)), energy_limit=2, deadline=rng.choice((None, "Milestone 1")),
                                  minute_of_day=rng.choice((600, 900, 905, 1200))) for _ in range(n)]))
    kinds = set()
    for m in models:
        kinds |= {c.key for c in m.commons}
        for sentence in all_sentences(m):
            words = set(re.findall(r"[a-z]+", sentence.lower()))
            assert not words & set(DIAGNOSIS), sentence  # the check-in's own rule, held for every sentence here
            assert not words & {"tired", "tense", "medical", "sick", "healthy", "cure", "treat", "treatment", "feel", "feeling"} - {"feel"}, sentence
    assert kinds >= {"sitting", "short_night", "rhr_up", "part"}  # the sweep really reached the cause side's sentences
    assert "how you feel" in lm.said_how(lm.fit(SEEDED)) and "not a reading of how you feel" in lm.said_how(lm.fit(SEEDED))


# ---------- the evaluation: synthetic people with a known truth ----------


def person(rng: random.Random, n: int, rates: dict[str, float], onset: tuple[int, int] = (5, 240)) -> list[dict]:
    """`n` remembered looks of an invented person: what they did (getting up 40% of the time, a drink 35%, neither
    25%), and whether the look eased, drawn from `rates`. The minutes at the desk are drawn evenly from `onset`."""
    out = []
    for _ in range(n):
        did = rng.choices(lm.ARMS, (0.40, 0.35, 0.25))[0]
        out.append(row(did, rng.random() < rates[did], sitting_min=rng.randrange(*onset), minute_of_day=rng.randrange(8 * 60, 22 * 60),
                       short_night=rng.random() < 0.5))
    return out


PEOPLE = {"breaks": {"up": 0.8, "drink": 0.2, "stay": 0.2},  # a break helps 80% of the time; water alone and staying put 20%
          "null": {"up": 0.5, "drink": 0.5, "stay": 0.5}}  # nothing makes a difference
SIZES, TRIALS = (3, 4, 8, 12, 20, 30, 50), 400


def evaluate() -> dict:
    """For each invented person and each number of remembered looks, over TRIALS people (a fixed seed each): how
    often getting up is ranked first, how often Atlas calls getting up a finding, how often it calls ANY arm a
    finding, and how often it says the looks share something (they share nothing: every feature is drawn evenly)."""
    table = {}
    for name, rates in PEOPLE.items():
        for n in SIZES:
            rng = random.Random(f"atlas-look-model:{name}:{n}")
            ranked = found = any_ = shared = 0
            for _ in range(TRIALS):
                m = lm.fit(person(rng, n, rates))
                score = lm.predict(m)
                ranked += m.enough and score["up"] > score["drink"] and score["up"] > score["stay"]
                found += m.arms["up"].helps
                any_ += any(a.helps for a in m.arms.values())
                shared += bool(m.commons)
            table[name, n] = tuple(round(x / TRIALS, 3) for x in (ranked, found, any_, shared))
    return table


def test_on_invented_people_it_recovers_the_ranking_with_enough_looks_and_claims_nothing_without(capsys):
    """THE NUMBERS (python -m pytest contexts/day/tests/test_look_model.py -k invented -s prints the table). This is a
    per-person table of counts over a handful of observations, not a trained expression recogniser."""
    table = evaluate()
    with capsys.disabled():
        print("\n  person  looks  up ranked first  'getting up helps'  any finding  'the looks share ...'")
        for (name, n), (ranked, found, any_, shared) in table.items():
            print(f"  {name:6}  {n:5}  {ranked:15.3f}  {found:18.3f}  {any_:11.3f}  {shared:20.3f}")
    # under the minimum nothing is ranked and nothing is claimed, whoever the person is
    assert table["breaks", 3] == table["null", 3] == (0.0, 0.0, 0.0, 0.0)
    assert table["breaks", 4][1:3] == (0.0, 0.0)  # four looks: ranked, never called a finding
    # the person a break helps: the ranking is recovered with a week or two of looks, the finding takes longer, and it gets there
    assert table["breaks", 8][0] >= 0.80 and table["breaks", 20][0] >= 0.95 and table["breaks", 50][0] >= 0.99
    assert table["breaks", 8][1] <= 0.25 and table["breaks", 20][1] >= 0.65 and table["breaks", 30][1] >= 0.85 and table["breaks", 50][1] >= 0.97
    # the null person: something is ranked first (something always is), and a finding is claimed for under one person in ten
    assert all(table["null", n][2] <= 0.10 for n in SIZES) and all(table["null", n][1] <= 0.05 for n in SIZES)
    assert all(0.2 <= table["null", n][0] <= 0.45 for n in SIZES if n >= 4)  # getting up first about a third of the time: chance
    # nobody here has looks that share anything: said for well under one in ten
    assert all(table[name, n][3] <= 0.08 for name in PEOPLE for n in SIZES)


def test_on_an_invented_person_whose_looks_start_at_about_the_same_time_it_finds_that_time():
    rng, found, wrong = random.Random("atlas-look-model:onset"), 0, 0
    for _ in range(TRIALS):
        m = lm.fit(person(rng, 10, PEOPLE["null"], onset=(85, 125)))  # ten looks, each after 85 to 125 minutes at the desk
        found += m.onset_min is not None
        wrong += m.onset_min not in (None, 75, 90, 105)
    assert found / TRIALS >= 0.95 and wrong == 0


# ---------- the service, on a bus ----------

SIPPED = "A person at the desk sipping from a water bottle."
# Earlier afternoons for the tests' member, by day of October: what they did after the look held, and whether it eased.
# With test_looks' Friday (got up, eased): getting up 4 of 5, water alone 0 of 2, staying 0 of 1. Eight known outcomes.
WEEK = ((1, "up", True), (2, "drink", False), (5, "up", True), (6, "stay", False), (7, "up", False), (8, "drink", False), (12, "up", True))


def earlier_look(day: int, did: str, eased: bool, sit: int = 105, member: str = MEMBER, hour: int = 13) -> list:
    """One earlier afternoon as the desk camera's windows of looks, with no record from the desk zone: sat down `sit`
    minutes before `hour`, strained from a quarter of an hour before it (so the rule holds on the hour). Then up six
    minutes later (ten minutes away, or four when it did not ease) and calmer or still strained after; or put, with
    a drink seen or without, the look still there half an hour on (or calmer from twelve minutes on, when it eased)."""
    held = at(hour, 0, day)
    out = [*windows(held - timedelta(minutes=sit), (sit - 15) // 3, 0.1, member)]
    if did == "up":
        back = held + timedelta(minutes=16 if eased else 10)
        out += [*windows(held - timedelta(minutes=15), 7, 0.7, member), *windows(back, 6 if eased else 7, 0.1 if eased else 0.7, member)]
        return out + ([] if eased else windows(back + timedelta(minutes=21), 5, 0.1, member))
    still = 12 if eased else 33
    out += [*windows(held - timedelta(minutes=15), (15 + still) // 3, 0.7, member), *windows(held + timedelta(minutes=still), 8 if eased else 5, 0.1, member)]
    if did == "drink":
        sip = held + timedelta(minutes=5)
        out.append(scene("cam-desk", f"frames:cam-desk:{int(sip.timestamp() * 1000)}", sip, SIPPED, "test"))
    return sorted(out, key=lambda e: e.ts)


def week(days=WEEK, member: str = MEMBER) -> list:
    return sorted([e for day, did, eased in days for e in earlier_look(day, did, eased, member=member)], key=lambda e: e.ts)


LEARNED = "Getting up has eased this four of the last five times; water alone hasn't either time."
PATTERN = "It usually shows up after about an hour and three quarters at the desk."
BACK = "That's the same look you had Friday afternoon, when you'd gone two hours without moving."
SAID = ("You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty minutes. "
        f"{BACK} {LEARNED} "
        "You've been at the desk for about an hour and three quarters straight, and I haven't seen you pick up a drink since five past two. "
        "Take ten minutes outside, drink some water, and I can move the study block to after dinner if you say yes.")
AHEAD = "You usually start to look strained around now."


def test_with_enough_known_outcomes_the_check_in_says_what_the_counts_say_and_orders_the_advice_by_them(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    known = [(m.day.day, lm.did(vars(m.outcome)), m.outcome.eased) for m in reversed(loop.looks.remembered(NOW)) if m.outcome.known]
    assert known == [*WEEK[:6], (FRIDAY, "up", True), WEEK[6]]  # each afternoon is remembered as what the box saw of it
    [e] = of(out, DayCheckinRaised)
    back = e.callback
    assert (back.said, back.learned, back.lead) == (BACK, LEARNED, "walk")  # the callback is still stage one's most similar look
    assert back.evidence == "From 9 remembered looks, 8 with a known outcome." and back.pattern == PATTERN
    assert back.lines[-1] == back.evidence and back.lines[1].startswith("Ten minutes away from the desk helped")
    assert [s.kind for s in e.suggestions] == ["walk", "drink", "move_block"]
    assert sorted(s.text for s in e.suggestions) == ["Drink some water", "Move the study block to after dinner", "Take ten minutes outside"]
    # said once, through the narrator's own path and the arbiter, inside the check-in's bound, and never a diagnosis word
    [told] = [n for n in of(out, DayNarrated) if n.moment == "checkin"]
    sentences, chars = BOUNDS["checkin"]
    assert told.spoken == SAID and len(re.split(r"(?<=[.!?])\s+", told.spoken)) == sentences and len(told.spoken) <= chars
    assert rule.diagnosis_in(told.spoken) == [] and rule.diagnosis_in(e.text) == [] and rule.diagnosis_in(e.ahead) == []
    facts = rule.facts(e.signals, e.suggestions, e.history, TZ, callback=back)
    assert check(told.spoken, facts, render(facts).spoken) == []  # the grounding check a model's rewrite must pass holds for it
    # the clause about when it shows up did not fit in five sentences: it is on the screen, not in the ear
    assert PATTERN in e.text and PATTERN in told.display and PATTERN not in told.spoken
    # the counts it says are the model's, and the model is fitted from the labelled rows and nothing else
    model = loop.looks.model(MEMBER, NOW)
    assert model == lm.fit(loop.looks.examples(MEMBER, NOW)) and (model.arms["up"].eased, model.arms["up"].n, model.arms["drink"].n) == (4, 5, 2)
    assert not model.arms["up"].helps  # four of five against none of three: one time in fourteen by chance, not yet a finding
    assert e.ahead == AHEAD and loop.checkin.card(e)["ahead"] == AHEAD  # ... so the card's line gives no advice


def test_the_clause_about_when_it_shows_up_is_said_aloud_only_while_the_check_in_stays_in_its_bound():
    from contracts.day import CheckinCallback, CheckinSignal

    back = CheckinCallback(moment_id="look-m-ana-20261009-1500", day=at(15, 0, FRIDAY).date(), said=BACK, learned=LEARNED, lead="walk",
                           pattern=PATTERN, evidence="From 9 remembered looks, 8 with a known outcome.")
    looked = CheckinSignal(kind="expression", summary="Looked strained 20 min", detail="", minutes=20)
    sat = CheckinSignal(kind="sitting", summary="At the desk 100 min", detail="", minutes=100, since=at(14, 0))
    short = render(rule.facts([looked], ADVICE[1:2], None, TZ, callback=back))
    assert short.spoken == f"You've looked strained for the last twenty minutes. {BACK} {LEARNED} {PATTERN} Take ten minutes outside."
    full = render(rule.facts([CheckinSignal(kind="sleep", summary="Sleep: short", detail="", minutes=302), looked, sat], ADVICE[:2], None, TZ, callback=back))
    assert PATTERN in full.display and PATTERN not in full.spoken and LEARNED in full.spoken
    assert len(re.split(r"(?<=[.!?])\s+", full.spoken)) == BOUNDS["checkin"][0]
    # with what the counts say and no single look to call back to, it is still said, after how the person looks
    alone = render(rule.facts([looked], ADVICE[1:2], None, TZ, callback=back.model_copy(update={"said": "", "pattern": ""})))
    assert alone.spoken == f"You've looked strained for the last twenty minutes. {LEARNED} Take ten minutes outside."


def test_below_the_minimum_the_check_in_is_stage_ones_unchanged(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(WEEK[:2]), *friday()]))
    assert loop.looks.model(MEMBER, NOW).known == 3 and not loop.looks.model(MEMBER, NOW).enough
    [e] = of(out, DayCheckinRaised)
    back = e.callback
    assert (back.learned, back.lead, back.pattern, back.evidence, e.ahead) == ("Ten minutes away from the desk helped then.", "walk", "", "", "")
    assert len(back.lines) == 2 and [s.kind for s in e.suggestions][:2] == ["walk", "drink"]
    alone, _, out_alone, _ = asyncio.run(tuesday(places, tmp_path / "one", [scanned(MEMBER), *friday()], name="one"))
    [single] = of(out_alone, DayCheckinRaised)
    assert single.callback == back and single.text == e.text and single.suggestions == e.suggestions  # word for word what one look gave
    shown = loop.looks.view(NOW, member_id=MEMBER)["learned"]
    assert shown["enough"] is False and shown["findings"] == [] and shown["not_enough"].startswith("Not enough yet.")
    assert loop.looks.said("helps", NOW) == "I don't know yet. I've seen how three looks went, and I need four before I'll say."


def test_a_person_a_drink_helps_is_told_so_and_their_advice_leads_with_it(places, tmp_path):  # noqa: F811
    """The same code, another person's looks: the order is theirs. Friday is the one time getting up was seen."""
    drinks = ((1, "drink", True), (2, "drink", True), (5, "stay", False), (6, "drink", True), (8, "stay", False), (12, "up", False))
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(drinks), *friday(got_up=False)]))
    model = loop.looks.model(MEMBER, NOW)
    assert [(a.name, a.eased, a.n) for a in model.arms.values()] == [("up", 0, 1), ("drink", 3, 3), ("stay", 0, 3)]
    [e] = of(out, DayCheckinRaised)
    assert e.callback.learned == "Water alone has eased this each of the last three times; staying at the desk hasn't any of the three times."
    assert e.callback.lead == "drink" and [s.kind for s in e.suggestions] == ["drink", "walk", "move_block"]


def test_each_person_has_their_own_model_and_the_rows_it_reads_carry_no_id_and_no_picture(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    ben = week(((1, "stay", False), (2, "up", False), (5, "drink", True), (6, "drink", True)), member="m-ben")
    asyncio.run(bus.publish_all(sorted([scanned(MEMBER, "m-ben"), *week(), *ben], key=lambda e: e.ts)))
    ana, other = loop.looks.model(MEMBER), loop.looks.model("m-ben")
    assert (ana.known, ana.arms["up"].eased, ana.arms["up"].n) == (7, 3, 4) and (other.known, other.arms["drink"].eased, other.arms["up"].eased) == (4, 2, 0)
    assert lm.lead(ana) == "walk" and lm.lead(other) == "drink"
    assert loop.looks.view(NOW, member_id="m-ben")["learned"]["known"] == 4 and loop.looks.view(NOW, member_id=MEMBER)["learned"]["known"] == 7
    mine = loop.looks.examples(MEMBER)
    assert len(mine) == 7 and len(loop.looks.examples("m-ben")) == 4 and len(loop.looks.examples()) == 11
    assert all(set(x) == {"look", "features", "advice", "outcome"} for x in mine)
    body = json.dumps(mine)
    assert not [word for word in ("m-ana", "m-ben", "look-m-", "frames:", "frame_ref", "embedding", "template", "vector", "bpm", "resting_hr") if word in body]


def test_the_model_is_never_kept_and_a_replay_gives_the_same_model_and_the_same_sentences(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    [e] = of(out, DayCheckinRaised)
    later = NOW + timedelta(hours=1)
    assert not [name for name, value in vars(loop.looks).items() if isinstance(value, lm.Model | lm.Arm) or "model" in name]  # fitted when asked
    asked = {k: loop.looks.said(k, later) for k in ("helps", "causes", "how")}

    async def restart():
        again, bus2, out2, ntfy2 = box(places, tmp_path, "b")
        await bus2.replay_into()
        await bus2.publish_all(again.tick(NOW + timedelta(minutes=5)))
        return again, out2, ntfy2

    again, out2, ntfy2 = asyncio.run(restart())
    assert again.looks.model(MEMBER, later) == loop.looks.model(MEMBER, later)
    assert again.looks.view(later)["learned"] == loop.looks.view(later)["learned"] and {k: again.looks.said(k, later) for k in asked} == asked
    assert again.looks.examples(MEMBER) == loop.looks.examples(MEMBER) and again.looks.callback(NOW) == e.callback
    assert again.checkin.card(again.checkin.raised[-1])["ahead"] == e.ahead and out2 == [] and ntfy2.sent == []  # and nothing is said again


# ---------- only for who switched it on, and forgetting forgets what was learned ----------


def learned(loop, member: str = MEMBER) -> dict | None:
    return loop.looks.view(NOW, member_id=member)["learned"]


def test_nothing_is_learned_about_anyone_who_did_not_scan_and_switch_expressions_on(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all([scanned("m-ben"), *week(), *friday()]))  # these looks are somebody's who is not on the list
    assert loop.looks.examples() == [] and loop.looks.model(MEMBER).total == 0 and learned(loop) is None and loop.looks.ahead(NOW) == ""
    loop, bus, out, _ = box(places, tmp_path / "off", "off")
    asyncio.run(bus.publish_all([scanned(), *week(), *friday()]))  # nobody switched it on
    assert loop.looks.examples() == [] and learned(loop) is None and loop.looks.said("helps", NOW) is None and loop.looks.said("how", NOW) is None


def test_switching_expressions_off_or_forgetting_a_member_removes_what_was_learned(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all(sorted([scanned(MEMBER, "m-ben"), *week(), *week(WEEK[:4], member="m-ben")], key=lambda e: e.ts)))
    assert learned(loop)["enough"] and learned(loop, "m-ben")["enough"]
    asyncio.run(bus.publish(scanned("m-ben", ts=at(18, 0, 12))))  # MEMBER switched it off, or was forgotten: off the list either way
    assert loop.looks.examples(MEMBER) == [] and loop.looks.model(MEMBER) == lm.fit([]) and learned(loop) is None
    assert learned(loop, "m-ben")["known"] == 4  # and nobody else's is touched

    async def restart():
        again, bus2, _, _ = box(places, tmp_path, "b")
        await bus2.replay_into()
        return again

    again = asyncio.run(restart())
    assert again.looks.examples(MEMBER) == [] and learned(again) is None and learned(again, "m-ben")["known"] == 4  # the same from the log alone


def test_forget_what_it_learned_is_forgetting_the_looks_and_one_forgotten_look_changes_the_counts(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    [e] = of(out, DayCheckinRaised)
    c = app_of(loop, bus)
    body = c.get("/api/day/looks", params={"member": MEMBER}).json()
    shown = body["learned"]
    assert shown["enough"] and (shown["looks"], shown["known"], shown["left_out"]) == (9, 8, 1)
    assert [f["text"] for f in shown["findings"]] == [
        "Getting up: the look eased 4 of the 5 times you got up. Too few to call it either way.",
        "Water alone: the look eased 0 of the 2 times you only had a drink. Too few to call it either way.",
        "It usually starts after about 105 minutes at the desk: 8 of 9 looks did."]
    assert shown["evidence"] == "From 9 remembered looks, 8 with a known outcome. 1 is left out: I couldn't see how it went."
    assert shown["findings"][0]["band"] == "Likely between 51% and 94% of the time, on this little."
    assert "frames:" not in json.dumps(body) and "bpm" not in json.dumps(body)
    assert c.delete("/api/day/looks/look-m-ana-20261001-1300").status_code == 200  # one look forgotten: an afternoon getting up eased it
    assert learned(loop)["findings"][0]["text"].startswith("Getting up: the look eased 3 of the 4 times you got up")
    assert c.delete("/api/day/looks", params={"member": MEMBER}).json()["left"] == 0  # "Forget what it learned": forget all of the looks
    assert learned(loop) is None and loop.looks.model(MEMBER) == lm.fit([]) and loop.looks.examples() == []
    assert loop.looks.said("helps", NOW) == memory.NOTHING and loop.looks.ahead(NOW) == "" and loop.checkin.card(e)["ahead"] == ""
    assert loop.checkin.card(e)["callback"] is None

    async def restart():
        again, bus2, _, _ = box(places, tmp_path, "b")
        await bus2.replay_into()
        return again

    assert learned(asyncio.run(restart())) is None  # forgotten on a replay of the log too


def test_a_wipe_of_everything_removes_what_was_learned(places, tmp_path):  # noqa: F811
    from contracts.vault import WipeRequested

    loop, bus, out, _ = box(places, tmp_path)
    asyncio.run(bus.publish_all([scanned(MEMBER), *week()]))
    assert learned(loop)["enough"]
    loop.looks.on_wipe(WipeRequested(ts=NOW, scope="memory"))  # the log is still there, and so is what is worked out from it
    assert learned(loop)["enough"]
    loop.looks.on_wipe(WipeRequested(ts=NOW, scope="all"))
    assert loop.looks.examples() == [] and loop.looks.model(MEMBER) == lm.fit([]) and learned(loop) is None


# ---------- the card's quiet line: never said, never pushed ----------


def test_the_quiet_line_is_on_the_card_only_and_a_setting_turns_it_off(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    [e] = of(out, DayCheckinRaised)
    assert e.ahead == AHEAD and loop.looks.look_ahead is True  # on unless agent.yaml says `day: {look_ahead: false}`
    narrated = of(out, DayNarrated)
    assert not [n for n in narrated if "usually start" in n.spoken or "usually start" in n.display]  # never said, on the box or the phone
    assert not [x for x in ntfy.sent if "usually" in " ".join(x)] and len([n for n in narrated if n.moment == "checkin"]) == 1
    assert {x.type for x in out} <= {"day.plan.proposed", "day.status.changed", "day.coach.noted", "day.checkin.raised", "day.narrated",
                                     "workspace.approval.requested"}  # no event of its own: nothing new to speak or push from
    assert "ahead" not in rule.facts(e.signals, e.suggestions, e.history, TZ, callback=e.callback)  # the narrator is never given it
    # away from the usual time it is not there; neither is it for a box with the setting off
    assert loop.looks.ahead(NOW) == AHEAD and loop.looks.ahead(NOW - timedelta(minutes=70)) == "" and loop.looks.ahead(NOW + timedelta(hours=2)) == ""
    loop.looks.look_ahead = False
    assert loop.looks.ahead(NOW) == "" and loop.checkin.card(e)["ahead"] == ""
    from contexts.day.application.loop import DayLoop

    assert "look_ahead" in Path(DayLoop.__init__.__code__.co_filename).read_text(encoding="utf-8")


def test_the_now_screen_gets_the_quiet_line_when_no_check_in_is_showing(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = box(places, tmp_path)
    sitting = windows(at(13, 55), 35, 0.1)  # today: at the desk since five to two, looking fine
    asyncio.run(bus.publish_all([scanned(MEMBER), *week(), *friday(), *sitting]))
    assert of(out, DayCheckinRaised) == []
    before = len(out)
    assert app_of(loop, bus).get("/api/day/now").json()["ahead"] == AHEAD and loop.looks.ahead(NOW) == AHEAD
    assert len(out) == before and not [n for n in of(out, DayNarrated) if "usually" in n.spoken + n.display]  # reading it says and publishes nothing


# ---------- box-only: nothing of the model leaves ----------


def model_words(loop, raised: DayCheckinRaised) -> list[str]:
    """Every word of what was learned that would give it away if it showed up somewhere it must not."""
    shown = learned(loop)
    said = [raised.callback.learned, raised.callback.pattern, raised.callback.evidence, raised.ahead, shown["evidence"], shown["note"],
            *(f["text"] for f in shown["findings"]), *(f["band"] for f in shown["findings"]), *(loop.looks.said(k, NOW) for k in ("helps", "causes", "how")),
            "eased", "usually", "remembered look", "known outcome", "Getting up", "water alone", "Water alone", "of the last", "look_model"]
    assert all(said)
    return said


def test_the_push_keeps_the_rules_own_order_and_says_nothing_that_was_learned(places, tmp_path):  # noqa: F811
    loop, bus, out, ntfy = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    [e] = of(out, DayCheckinRaised)
    assert [s.kind for s in e.suggestions][:2] == ["walk", "drink"] and e.callback.evidence  # on the box: the order the counts support
    sent = " ".join(f"{title} {body}" for title, body in ntfy.sent)
    assert checkin_push(e) == "Atlas has a check-in for you. Drink some water, take ten minutes outside. Open Now to see why." and checkin_push(e) in sent
    assert not [word for word in (*model_words(loop, e), "Friday", "strained", "look") if word in sent]
    # the same afternoon on a box that has learned nothing yet pushes word for word the same, so the push tells nobody which it is
    other, _, out2, _ = asyncio.run(tuesday(places, tmp_path / "none", [scanned(MEMBER)], name="none"))
    [plain] = of(out2, DayCheckinRaised)
    assert plain.callback is None and checkin_push(plain) == checkin_push(e)


def test_a_circle_share_and_an_export_built_while_a_model_exists_have_none_of_it(places, tmp_path):  # noqa: F811
    from contexts.agent.application.memory_port import FakeMemory
    from contexts.circle.application.sharing import HABIT_FIELDS
    from contexts.circle.tests.test_health_privacy import wire
    from contexts.habits.tests.helpers import at as on
    from contexts.learning.application.export import DatasetExporter, teach
    from contexts.workspace.domain.approvals import Approvals
    from contracts.experience import ApprovalDecided
    from contracts.learning import TaskSpec
    from contracts.memory import Episode

    async def run():
        loop, bus, out, ntfy = box(places, tmp_path)
        habits, sharing = wire(tmp_path, bus)  # Habits and the opted-in Circle sharing on the Day loop's own bus
        await bus.publish_all([scanned(MEMBER), *week(), *friday()])
        await bus.publish_all(loop.planner.plan(at(8, 10)))
        loop.checkin.enabled = False
        await bus.publish_all(afternoon())
        loop.checkin.enabled = True
        await bus.publish_all(loop.tick(NOW))
        await bus.publish_all(loop.tick(NOW + timedelta(seconds=90)))
        [e] = of(out, DayCheckinRaised)
        assert e.callback.evidence and learned(loop)["enough"]
        await bus.publish_all(habits.flush(on(14, "00:05")))  # the day closes: this is everything that leaves for the Circle
        shared = sharing.outbox.pending()
        assert shared and {r.kind for r in shared} == {"habit_summary"} and all(set(r.payload) == HABIT_FIELDS for r in shared)
        leaving = json.dumps([r.payload for r in shared])
        assert not [word for word in (*model_words(loop, e), "strained", "look", "callback") if word in leaving]
        assert len(sharing.sync(on(14, "00:06"))) == len(shared)  # ... and what is pushed to the shared database is those rows
        return loop, e

    loop, e = asyncio.run(run())
    target = tmp_path / "datasets"
    target.mkdir()
    published = []
    episode = Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(hours=1), captions=["a person at the desk"], keyframe_refs=["frames:cam-desk:e1"])
    approvals = Approvals({"export_training_data": DatasetExporter(FakeMemory([episode]), lambda ref: b"jpeg", published.append, out_dir=target)})
    _, proposal = teach(TaskSpec(task="desk-objects", instruction="List small objects on the desk as JSON.", camera="cam-desk"), NOW)
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result
    written = b"".join(p.read_bytes() for p in target.rglob("*") if p.is_file()).decode(errors="replace") + published[0].model_dump_json()
    assert not [word for word in (*model_words(loop, e), "strained", "callback", "day.look") if word in written]


def test_nothing_that_was_learned_is_logged_at_info_level(places, tmp_path, caplog):  # noqa: F811
    with caplog.at_level(logging.INFO):
        loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
        [e] = of(out, DayCheckinRaised)
        c = app_of(loop, bus)
        c.get("/api/day/looks", params={"member": MEMBER}), c.get("/api/day/now"), c.get("/api/day/checkin")
        words = model_words(loop, e)
        loop.looks.model(MEMBER), loop.looks.ahead(NOW), loop.looks.examples(MEMBER)
        c.delete("/api/day/looks")
    logged = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO and r.name.startswith("atlas"))
    assert e.callback.evidence and not [word for word in words if word in logged]


def test_nothing_that_leaves_the_box_can_reach_the_model():
    """Structural. The code that leaves the box (the Circle and its shared database, the League's sync, the Learning
    lab and its pod uploads, the lab scripts, the phone push, the connectors, the vault's exports) names neither the
    model nor the look memory, so none of it can be handed a count. The agent reaches three sentences through the
    port the composition root wires, and imports nothing of Day. The model itself is in no event and no file."""
    root = Path(__file__).resolve().parents[3]
    for folder in ("contexts/circle", "contexts/play/league", "contexts/learning", "contexts/experience", "contexts/vault", "contexts/connectors",
                   "contexts/habits", "contexts/memory", "lab"):
        for path in (root / folder).rglob("*.py"):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"look_model|LookMemory|domain\.looks|application\.looks|labelled_examples|\.examples\(\)|CheckinCallback", text), path
    agent = "".join(p.read_text(encoding="utf-8") for p in (root / "contexts/agent").rglob("*.py") if "tests" not in p.parts)
    assert not re.search(r"look_model|contexts\.day\.(?:application|domain)\.looks|LookMemory", agent)
    contracts = (root / "contracts" / "day.py").read_text(encoding="utf-8")
    assert not re.search(r"class \w*(?:LookModel|Learned)\w*\(", contracts)  # no event or shape carries a model
    from contracts.day import BOX_ONLY_EVENTS

    assert "day.checkin.raised" in BOX_ONLY_EVENTS  # the one event its sentences are in never leaves the box
    used = [p.relative_to(root).as_posix() for p in (root / "contexts").rglob("*.py") if "tests" not in p.parts and "look_model" in p.read_text(encoding="utf-8")]
    assert sorted(used) == ["contexts/day/api.py", "contexts/day/application/looks.py", "contexts/day/domain/looks.py", "contexts/day/domain/narration.py",
                            "contexts/demo/checkin.py"]  # the look memory fits it; the API, the rule and the demo name it in a docstring only


# ---------- asked ----------


def test_three_questions_are_answered_from_what_it_learned_for_the_ear(places, tmp_path):  # noqa: F811
    loop, bus, out, _ = asyncio.run(tuesday(places, tmp_path, [scanned(MEMBER), *week(), *friday()]))
    ask = loop.looks.said
    assert ask("helps", NOW) == ("I can't say for sure yet. The look eased four of the five times you got up, and eased neither of the two "
                                 "times you only had a drink.")
    assert ask("causes", NOW) == ("I can't say what causes it, only what the looks share: eight of the nine looks I remember started after "
                                  "an hour and three quarters or more at the desk.")
    assert ask("how", NOW) == ("From what I saw on this box, nothing more: getting up eased it four of five times, and water alone none of two. "
                               "It's a count over the nine looks I remember, not a reading of how you feel.")
    for kind in ("helps", "causes", "how"):
        assert rule.diagnosis_in(ask(kind, NOW)) == [] and len(re.split(r"(?<=[.!?])\s+", ask(kind, NOW))) <= 2  # one or two short sentences
    # stage one's three are what they were
    assert ask("helped", NOW) == "Last time, yesterday afternoon, ten minutes away from the desk helped: the look eased within half an hour."
    assert lm.said_helps(lm.fit(SEEDED)) == ("Getting up, from what I've seen. The look eased five of the six times you got up, and eased neither "
                                             "of the two times you only had a drink.")
