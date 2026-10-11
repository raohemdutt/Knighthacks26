"""The look memory, stage one: remember a sustained look with what was going on around it, and call back to it.
Pure: no clock, no model, no I/O. contexts/day/application/looks.py keeps the moments (a projection of the log).
Stage two, what Atlas learns from all of them, is contexts/day/domain/look_model.py; it reads `labelled_examples`.

    then = similar(context_now, today, member, moments)        # an earlier moment like this one, or None
    said = callback(then, today, tz)                           # "That's the same look you had Tuesday afternoon, when ..."
    helped, lead = learned(then, words)                        # "Ten minutes away from the desk helped then.", "walk"

The rules, in plain words (every time is event time):

- A look is what the check-in already calls one: the last EXPR_WINDOW_MIN minutes of counted looks read as strained
  (contexts/day/domain/checkin.py `looked_strained`, its own thresholds). A stretch is that rule holding from one
  30-second-to-three-minute window to the next; ONE moment is remembered per stretch, with its context taken when the
  stretch began to hold. The kind of look is one of the two words Perception already uses for people on the People
  screen (contexts/perception/domain/expression.py LOOKS): "tired" when most of the strained looks read as sad, "tense"
  when most read as angry, disgusted or fearful. It is how a face looked to a small model: not an emotion and nothing
  about health. A sentence that goes into a check-in (`callback`, `learned`, the card's lines) never uses either word:
  narration.DIAGNOSIS holds there as it does for the rest of the check-in. The chip and the answers to a question
  say "looked tired", as the People screen does.
- The context is labels and whole numbers only: minutes at the desk, minutes since a drink was seen picked up, energy
  drinks so far against the limit, whether the resting heart rate was above the person's usual (a boolean, never the
  value), whether the night was short (a boolean), what they were doing (the plan block), a deadline within a day, the
  minute of the day and the weekday. No picture, no frame ref, no face vector.
- The outcome is what the box saw in the EASE_MIN minutes after: did they leave the desk for SIT_BREAK_S or more within
  GOT_UP_MIN minutes, was a drink seen picked up, did the looks counted in the second half of that half hour read as
  strained for less than EXPR_SHARE (eased) or not, and how the check-in was answered. Each is None when the box could
  not see it. Nothing is said about an outcome that is not known.
- Similar (`similar`): the same look and member, a different local day within CALLBACK_DAYS days, and a distance of
  SIMILAR_MAX or less. The distance is the mean of up to four parts, each 0 (the same) to 1: minutes at the desk
  (a difference of SIT_MIN minutes or more is 1), minutes since a drink (DRINK_GAP_MIN), what they were doing (the same
  kind of block is 0, another is 1), and the time of day (CLOCK_MIN minutes apart or more is 1). A part missing on
  either side is left out; with fewer than MIN_SHARED parts (the time of day and one more) nothing is similar: a
  moment known only by its hour is not "the same kind of moment". Among the moments within NEAR_TIE of
  the closest, the most recent wins. When nothing is similar enough nothing is said about the past.
- What it learned (`learned`): getting up, or a drink, followed by the look easing is said ("... helped then.") and
  that advice is said first; staying put and the look not easing is said gently; anything else is not said.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, tzinfo

from contexts.day.domain import checkin as rule
from contexts.day.domain.coach import Span, stretches
from contexts.day.domain.narration import WEEKDAYS, Words, cap, clock, hhmm, part_of_day, say_minutes, say_number
from contracts.day import CheckinSuggestion

# The kinds of look stage one remembers, in the words Perception uses for people; a trained model (stage two) may add more.
LOOK_WORDS = {"sad": "tired", "angry": "tense", "disgust": "tense", "fearful": "tense"}
TIRED, TENSE = "tired", "tense"
CALLBACK_DAYS = 21  # how far back a callback may reach
KEEP_DAYS = 90  # how long a moment is kept at all (the examples a later training step would read)
GOT_UP_MIN = 15  # "got up": left the desk within this many minutes of the moment
EASE_MIN = 30  # the outcome is known this long after the moment
CLOCK_MIN = 240  # two times of day four hours apart are another part of the day: as different as they get
SIMILAR_MAX = 0.5  # the largest distance that is still "the same look in the same kind of moment"
NEAR_TIE = 0.1  # within this of the closest: the most recent wins
MIN_SHARED = 2  # parts both moments must have: the time of day and one more, or the comparison means nothing
RECENT = timedelta(hours=3)  # "why do you think I look tired?" is about a look this recent
WALK_KINDS = ("walk", "break")  # the check-in's two ways of saying "get up"

DOING = {"study": "studying", "eat": "eating", "rest": "taking a break", "class": "in a lecture", "event": "at an event"}
CLASS_WORDS = ("lecture", "class", "lab", "seminar", "recitation", "section", "discussion")


@dataclass(frozen=True)
class Context:
    """What was going on around a look. Empty means the box did not know, never zero."""

    minute_of_day: int  # local
    weekday: int  # Monday is 0
    sitting_min: int | None = None  # at the desk without a break of three minutes
    since_drink_min: int | None = None  # since a drink was seen picked up (or since sitting down); None with no container in view
    energy_drinks: int | None = None  # so far that day
    energy_limit: int | None = None
    rhr_up: bool | None = None  # resting heart rate above the person's own usual: a boolean, never the value
    short_night: bool | None = None
    activity: str | None = None  # the kind of plan block: "study", "class", "event", "eat", "rest"
    doing: str | None = None  # the same in words: "studying", "in a lecture"
    deadline: str | None = None  # something due within a day, by its short title
    deadline_hours: int | None = None


@dataclass(frozen=True)
class Outcome:
    """What happened in the half hour after. `known` False: that half hour has not passed yet."""

    known: bool = False
    got_up: bool | None = None
    away_min: int | None = None  # how long they were away, when they came back inside the half hour
    drink: bool | None = None
    eased: bool | None = None
    answer: str | None = None  # how the check-in raised in the moment was answered: "done", "move"


@dataclass(frozen=True)
class Moment:
    id: str  # "look-<member>-<yyyymmdd>-<hhmm>" of when the stretch began to hold, local
    member_id: str
    at: datetime  # when the stretch began to hold
    until: datetime  # the last window in which it still held
    day: date  # local
    context: Context
    look: str = TIRED  # "tired" or "tense" (`look_word`)
    checkin_id: str | None = None  # the check-in raised in this moment, if one was
    suggested: tuple[str, ...] = ()  # ... and the kinds of advice it gave
    outcome: Outcome = field(default_factory=Outcome)


def look_word(shares: dict[str, float]) -> str:
    """The kind of look in a stretch, from the labels' shares added up over its windows: "tired" or "tense"."""
    by_word = {TIRED: 0.0, TENSE: 0.0}
    for label, share in shares.items():
        if label in LOOK_WORDS:
            by_word[LOOK_WORDS[label]] += share
    return TIRED if by_word[TIRED] >= by_word[TENSE] else TENSE


def moment_id(member_id: str, at: datetime, tz: tzinfo) -> str:
    return f"look-{member_id}-{at.astimezone(tz):%Y%m%d-%H%M}"


# ---------- what happened next ----------


def outcome(at: datetime, as_of: datetime, spans: list[Span], drinks: list[datetime], container: bool, looks: list[rule.Look],
            answer: str | None = None) -> Outcome:
    """The outcome of the moment at `at`, from what the box recorded up to `as_of`: `spans` when a person was at the desk,
    `drinks` when one was seen picked up, `container` whether a drink container was in view, `looks` the windows."""
    if as_of < at + timedelta(minutes=EASE_MIN):
        return Outcome(answer=answer)
    end = at + timedelta(minutes=EASE_MIN)
    got_up = away = None
    runs = stretches([s for s in spans if s[0] <= end], rule.SIT_BREAK_S)
    here = next((i for i, (a, b) in enumerate(runs) if a <= at <= b + rule.AT_DESK_WITHIN), None)
    if here is not None:
        left = runs[here][1]
        back = runs[here + 1][0] if here + 1 < len(runs) else None
        got_up = left < at + timedelta(minutes=GOT_UP_MIN) and (back or as_of) - left >= timedelta(seconds=rule.SIT_BREAK_S)
        if got_up and back is not None and back <= end:
            away = max(1, round((back - max(left, at)).total_seconds() / 60))
    seen = any(at < d <= end for d in drinks)
    later = [x for x in looks if at + timedelta(minutes=EASE_MIN // 2) < x.end <= end and x.samples > 0]
    total = sum(x.samples for x in later)
    eased = (sum(x.strained for x in later) / total < rule.EXPR_SHARE) if total >= rule.EXPR_MIN_SAMPLES else None
    return Outcome(known=True, got_up=got_up, away_min=away, drink=True if seen else (False if container else None), eased=eased, answer=answer)


# ---------- the most similar earlier moment ----------


def distance(a: Context, b: Context) -> float | None:
    """0 (the same kind of moment) to 1, or None when the two share too little to compare."""
    parts: list[float] = []
    if a.sitting_min is not None and b.sitting_min is not None:
        parts.append(min(1.0, abs(a.sitting_min - b.sitting_min) / rule.SIT_MIN))
    if a.since_drink_min is not None and b.since_drink_min is not None:
        parts.append(min(1.0, abs(a.since_drink_min - b.since_drink_min) / rule.DRINK_GAP_MIN))
    if a.activity and b.activity:
        parts.append(0.0 if a.activity == b.activity else 1.0)
    apart = abs(a.minute_of_day - b.minute_of_day)
    parts.append(min(1.0, min(apart, 1440 - apart) / CLOCK_MIN))
    return sum(parts) / len(parts) if len(parts) >= MIN_SHARED else None


def similar(now: Context, today: date, member_id: str, moments: list[Moment], look: str = TIRED) -> Moment | None:
    """The earlier moment to call back to, or None: never a forced one. `look`: the kind of look now."""
    scored = []
    for m in moments:
        if m.member_id != member_id or m.look != look or not 0 < (today - m.day).days <= CALLBACK_DAYS:
            continue
        d = distance(now, m.context)
        if d is not None and d <= SIMILAR_MAX:
            scored.append((d, m))
    if not scored:
        return None
    best = min(d for d, _ in scored)
    return max((m for d, m in scored if d <= best + NEAR_TIE), key=lambda m: m.at)


# ---------- the words ----------


def when_words(at: datetime, today: date, tz: tzinfo) -> str:
    """"yesterday afternoon", "Tuesday afternoon", "last Tuesday afternoon" (a week ago to the day), "the Tuesday before
    last", "about two weeks ago". The small hours belong to the night before."""
    local = at.astimezone(tz)
    part = part_of_day(hhmm(local))
    named = local.date() - timedelta(days=1) if local.hour < 4 else local.date()
    gap, name = (today - named).days, WEEKDAYS[named.weekday()]
    if gap <= 0:
        return "tonight" if part == "night" else f"this {part}"
    if gap == 1:
        return "last night" if part == "night" else f"yesterday {part}"
    if gap <= 6:
        return f"{name} {part}"
    if gap == 7:
        return f"last {name} {part}"
    if gap <= 13:
        return f"the {name} before last"
    return "about two weeks ago" if gap < 18 else "about three weeks ago"


def short_day(at: datetime, today: date, tz: tzinfo) -> str:
    """For the chip: "yesterday", "Tue", "Oct 6"."""
    local = at.astimezone(tz)
    gap = (today - local.date()).days
    return "today" if gap <= 0 else "yesterday" if gap == 1 else WEEKDAYS[local.weekday()][:3] if gap <= 6 else f"{local:%b} {local.day}"


def telling(c: Context) -> str | None:
    """The one fact of a moment worth saying: the one furthest past its own threshold, else the deadline, else
    what they were doing."""
    over = []
    if c.sitting_min is not None and c.sitting_min >= rule.SIT_MIN:
        over.append((c.sitting_min / rule.SIT_MIN, f"you'd gone {say_minutes(c.sitting_min)} without moving"))
    if c.since_drink_min is not None and c.since_drink_min >= rule.DRINK_GAP_MIN:
        over.append((c.since_drink_min / rule.DRINK_GAP_MIN, f"you hadn't picked up a drink in {say_minutes(c.since_drink_min)}"))
    if c.energy_drinks and c.energy_limit and c.energy_drinks > c.energy_limit:
        over.append((c.energy_drinks / c.energy_limit, f"you'd had {say_number(c.energy_drinks)} energy drinks"))
    if over:
        return max(over, key=lambda x: x[0])[1]
    if c.deadline:
        return f"{c.deadline} was due {'that day' if (c.deadline_hours or 0) <= 8 else 'the next day'}"
    return f"you were {c.doing}" if c.doing else None


def callback(m: Moment, today: date, tz: tzinfo) -> str:
    fact = telling(m.context)
    return f"That's the same look you had {when_words(m.at, today, tz)}" + (f", when {fact}." if fact else ".")


def learned(m: Moment, w: Words = Words()) -> tuple[str, str | None]:
    """(what followed last time in a sentence, the advice to say first) or ("", None) when there is nothing to say."""
    o = m.outcome
    if not o.known or o.eased is None:
        return "", None
    if o.eased and o.got_up:
        return (f"{cap(say_minutes(o.away_min))} away from the {w.workplace} helped then." if o.away_min else "Getting up helped then."), "walk"
    if o.eased and o.drink:
        return "A drink helped then.", "drink"
    if not o.eased and o.got_up is False and not o.drink:
        return "You pushed through then, and it was still there half an hour later.", None
    return "", None


def rerank(suggestions: list[CheckinSuggestion], lead: str | None) -> list[CheckinSuggestion]:
    """The same advice, with what was followed by the look easing last time said first. The offer to move a block stays last."""
    first = WALK_KINDS if lead == "walk" else (lead,) if lead else ()
    return sorted(suggestions, key=lambda s: (s.kind == "move_block", s.kind not in first))


def going_on(c: Context, w: Words = Words()) -> str:
    """What was going on, as one plain line for the screen."""
    parts = []
    if c.sitting_min is not None and c.sitting_min >= 5:
        parts.append(f"at the {w.workplace} for {say_minutes(c.sitting_min)}")
    if c.since_drink_min is not None and c.since_drink_min >= 30:
        parts.append(f"no drink seen for {say_minutes(c.since_drink_min)}")
    if c.energy_drinks:
        parts.append(f"{say_number(c.energy_drinks)} energy drink{'' if c.energy_drinks == 1 else 's'} so far"
                     + (f" (your limit is {say_number(c.energy_limit)})" if c.energy_limit else ""))
    if c.short_night:
        parts.append("after a short night")
    if c.rhr_up:
        parts.append("resting heart rate above your usual")
    if c.doing:
        parts.append(c.doing)
    if c.deadline:
        parts.append(f"{c.deadline} due within a day")
    return cap(", ".join(parts)) + "." if parts else "I only have the time for this one."


def went(m: Moment, w: Words = Words()) -> str:
    """What helped, or how it went, as one plain line for the screen."""
    o = m.outcome
    if not o.known:
        return "Too soon to say how it went."
    if o.eased is None:
        return "I couldn't see how it went after."
    if o.eased and o.got_up:
        return (f"{cap(say_minutes(o.away_min))} away from the {w.workplace}" if o.away_min else "Getting up") + " helped: the look eased within half an hour."
    if o.eased and o.drink:
        return "A drink helped: the look eased within half an hour."
    if o.eased:
        return "The look eased within half an hour. I didn't see what helped."
    if o.got_up:
        return "You got up, and the look was still there half an hour later."
    return f"You stayed at the {w.workplace}, and the look was still there half an hour later."


def view(m: Moment, today: date, tz: tzinfo, w: Words = Words()) -> dict:
    """One remembered moment for the owner's own screen: labels, numbers and plain lines. No picture, no heart value."""
    local = m.at.astimezone(tz)
    about = local.replace(second=0, microsecond=0) + timedelta(minutes=round(local.minute / 5) * 5 - local.minute)  # "around": to five minutes
    return {"id": m.id, "member_id": m.member_id, "look": m.look, "at": m.at.isoformat(), "until": m.until.isoformat(),
            "day": m.day.isoformat(), "weekday": WEEKDAYS[local.weekday()], "part": part_of_day(hhmm(local)),
            "when": cap(when_words(m.at, today, tz)), "time": clock(hhmm(about)), "short": short_day(m.at, today, tz),
            "context": asdict(m.context), "going_on": going_on(m.context, w), "helped": went(m, w),
            "suggested": list(m.suggested), "checkin_id": m.checkin_id, "outcome": asdict(m.outcome)}


def chip(m: Moment, today: date, tz: tzinfo) -> str:
    return f"Looked {m.look} · like {short_day(m.at, today, tz)}"


# ---------- answers to a question (the agent's port; for the ear: contexts/day/domain/ear.py's rule) ----------

NOTHING = "I haven't got a look like that remembered yet."


def said_last(m: Moment | None, today: date, tz: tzinfo) -> str:
    """ "When did I last look this tired?" """
    if m is None:
        return NOTHING
    fact = telling(m.context)
    return f"The last time you looked this {m.look} was {when_words(m.at, today, tz)}" + (f", when {fact}." if fact else ".")


def said_why(now: Moment | None, then: Moment | None, today: date, tz: tzinfo, w: Words = Words()) -> str:
    """ "Why do you think I'm tired?": what was measured, never what it means. `now`: the look of the last hours, if any."""
    if now is None:
        return "I don't think that right now: your face hasn't looked tired or tense lately. I can only go by how you look to the camera."
    c, seen = now.context, [f"your face has looked {now.look} for a while"]
    if c.sitting_min is not None and c.sitting_min >= rule.SIT_MIN:
        seen.append(f"you'd been at the {w.workplace} for {say_minutes(c.sitting_min)}")
    if c.since_drink_min is not None and c.since_drink_min >= rule.DRINK_GAP_MIN:
        seen.append(f"I hadn't seen you pick up a drink in {say_minutes(c.since_drink_min)}")
    if c.short_night:
        seen.append("you slept short")
    if c.rhr_up:
        seen.append("your resting heart rate is up")
    seen = seen[:3]
    listed = seen[0] if len(seen) == 1 else " and ".join(seen) if len(seen) == 2 else ", ".join(seen[:-1]) + f", and {seen[-1]}"
    out = f"I can't know how you feel, only what I saw: {listed}."
    return f"{out} {callback(then, today, tz)}" if then is not None else out


def said_helped(m: Moment | None, today: date, tz: tzinfo, w: Words = Words()) -> str:
    """ "What helped last time?" """
    if m is None:
        return NOTHING
    o, when = m.outcome, when_words(m.at, today, tz)
    if not o.known or o.eased is None:
        return f"I don't know yet. I saw the look {when}, but not how it went after."
    if o.eased and (o.got_up or o.drink):
        return f"Last time, {when}, " + went(m, w)[0].lower() + went(m, w)[1:]
    if o.eased:
        return f"Last time, {when}, it eased within half an hour, and I didn't see what helped."
    return f"Nothing I saw. Last time, {when}, " + went(m, w)[0].lower() + went(m, w)[1:]


# ---------- the seam for stage two (contexts/day/domain/look_model.py): labelled examples, and nothing else ----------


def labelled_examples(moments: list[Moment]) -> list[dict]:
    """READ-ONLY. The remembered moments as labelled examples: the look (the label), the context (the features), the
    advice given and the outcome (the target). No picture, no frame ref, no member id, no heart value: only what a
    moment already holds. The learning step on the box (look_model.fit) reads this and nothing else from the look memory."""
    return [{"look": m.look, "features": asdict(m.context), "advice": list(m.suggested), "outcome": asdict(m.outcome)}
            for m in sorted(moments, key=lambda m: m.at)]


def with_outcome(m: Moment, o: Outcome) -> Moment:
    return replace(m, outcome=o)
