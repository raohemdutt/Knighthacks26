"""The look memory, stage two: what Atlas has learned, for one person, from every look it remembers. Pure: no clock,
no randomness, no I/O, standard library only. contexts/day/application/looks.py fits it, per member, whenever it is asked.

    model = fit(examples)                       # `examples`: LookMemory.examples(member) rows and nothing else
    order = rerank(suggestions, model, now)     # the same advice, in the order the counts support
    said = learned_line(model, now, words)      # "Getting up has eased this four of the last five times; water alone hasn't either time."

WHAT THIS IS, PLAINLY. A per-person table of counts over a handful of remembered moments, with a little smoothing
and one significance test. It is NOT a trained expression recogniser and trains none: the thing that reads a face
stays the pre-trained model from OpenCV's zoo (contexts/perception), untouched. Nothing here sees a picture, a face
crop or a face vector: `fit` takes the labelled rows of contexts/day/domain/looks.py `labelled_examples` (the kind
of look, whole-number and yes/no context, the advice given, what was seen in the half hour after) and nothing else.

WHY THIS ONE. A person gives Atlas tens of examples at most, a few a week. A neural net or even a logistic
regression over a dozen features would fit noise at that size and could not say why it thinks what it thinks.
Counting can: every number below is "k of the n times", so every sentence Atlas says can be checked against the
list on the People screen. Three plain pieces:

1. What eased it (the outcome side). Each remembered look whose outcome is KNOWN (the half hour after has passed and
   the box could see whether the look eased) is put in one arm by what the person was seen to do in that half hour:
   "up" (left the desk for three minutes or more, with or without a drink), "drink" (a drink seen picked up and
   they stayed: "water alone"), "stay" (neither). "Follows the advice" is read as "was seen doing it", whether or
   not Atlas had suggested it: most looks are remembered with no check-in raised. The offer to move a block is not
   learned (it is answered through an approval card, not seen in half an hour) and stays last.
   For an arm with k eased of n: the estimate is a smoothed rate, (k + PRIOR * pooled) / (n + PRIOR), where `pooled`
   is the person's own overall rate of a look easing, (K + 1) / (N + 2). PRIOR pretend-observations at the pooled
   rate: an arm seen once or twice is pulled most of the way back to the person's average, an arm seen ten times
   hardly at all (partial pooling). "In this kind of context" is the same step once more: for each binned feature
   known right now (long at the desk or not, long without a drink or not, the part of the day, a short night or
   not) the arm's counts among the looks in that bin are smoothed towards the arm's own estimate, and the bins'
   estimates are averaged. The advice order is those estimates, highest first.
2. Is it a finding (the honesty gate). An arm "helps" only when Fisher's exact test, one-sided, on the 2x2 table
   (this arm or not, eased or not) gives ALPHA or less: if what the person did made no difference, counts this
   lopsided in its favour would turn up less than one time in twenty. Otherwise Atlas still says the counts,
   because they are true, but never the word "helps". Nothing is ever called a finding the other way ("does not
   help"): "water alone hasn't either time" is a count and stays one. The band next to a rate is the Wilson
   interval at BAND on the raw counts: wide on purpose with this little.
3. What the looks have in common (the cause side). The box only keeps the moments a look STARTED; it keeps no
   record of the afternoons a person looked fine. So this side cannot say what causes a look, and does not: it says
   what the remembered looks share ("9 of 11 started after ninety minutes or more at the desk"), and only when the
   share is high (COMMON_SHARE) and, for a yes/no fact (a short night, no drink for a while, the part of the day),
   more than a coin flip would give (the binomial tail against a half is COMMON_ALPHA or less). For the minutes at
   the desk the test is that the onsets are bunched: the middle half of them within ONSET_SPREAD minutes of each
   other, at ONSET_MIN minutes or more. Every look counts here, known outcome or not.

EVIDENCE, STATED. A model carries how many looks it was given, how many had a known outcome and how many were left
out because the box could not see how they went. Under MIN_KNOWN known outcomes it says "not enough yet" (`enough`
is False): nothing is ranked, nothing is said, and the check-in behaves exactly as stage one did. Under MIN_ONSET
looks nothing is said about what they share.

DETERMINISTIC. `fit` is a function of the rows alone, in any order: the same remembered moments give the same model
and the same sentences on a live box and on a replay of its log. BOX-ONLY: the model is never stored, sent or
logged; it is rebuilt from the look memory when asked, so forgetting the looks forgets it.

Never a diagnosis: no sentence here says how a person feels or what a look means, and the ones that go into a
check-in pass narration.DIAGNOSIS like the rest of it ("strained", never "tired").
"""

from dataclasses import dataclass, field
from math import comb, sqrt

from contexts.day.domain import checkin as rule
from contexts.day.domain.narration import Words, cap, say_minutes, say_number
from contracts.day import CheckinSuggestion

MIN_KNOWN = 4  # known outcomes before anything is ranked or said about what eases a look
MIN_ARM = 3  # times an arm was seen, and times it was not, before it can be called a finding
PRIOR = 2.0  # pretend-observations at the pooled rate: the strength of the pull towards it
ALPHA = 0.05  # Fisher's exact test, one-sided: a finding at this or less
BAND = 0.80  # the Wilson interval shown next to a rate
Z = 1.2816  # the normal quantile for BAND, two-sided
MIN_ONSET = 5  # looks with a feature known before anything is said about what they share
COMMON_SHARE = 0.75  # "usually": this share of them or more, and
COMMON_ALPHA = 0.05  # ... k of n or more would come up by a coin flip this often or less
ONSET_SPREAD = 45  # minutes at the desk: the middle half of the onsets lies within this, or there is no "usual" time
ONSET_MIN = 45  # ... and the usual time is at least this long, or it is not worth a sentence
ONSET_STEP = 15  # the usual time is said to a quarter of an hour
AHEAD_MIN, AROUND_MIN = 15, 30  # "around now": from this long before the usual time to this long after it

UP, DRINK, STAY = "up", "drink", "stay"
ARMS = (UP, DRINK, STAY)
ADVICE_ARM = {"walk": UP, "break": UP, "drink": DRINK}  # the check-in's advice set; "move_block" is not learned
FEATURES = ("sit", "dry", "part", "night")


def did(outcome: dict) -> str | None:
    """The arm a known outcome falls in, or None when the box could not see what the person did."""
    if outcome.get("got_up"):
        return UP
    if outcome.get("drink"):
        return DRINK
    return STAY if outcome.get("got_up") is False else None


def bins(features: dict) -> dict[str, str]:
    """A context as a few coarse labels; a feature the box did not know is left out, never guessed."""
    out: dict[str, str] = {}
    if features.get("sitting_min") is not None:
        out["sit"] = "long" if features["sitting_min"] >= rule.SIT_MIN else "short"
    if features.get("since_drink_min") is not None:
        out["dry"] = "yes" if features["since_drink_min"] >= rule.DRINK_GAP_MIN else "no"
    if features.get("minute_of_day") is not None:
        hour = features["minute_of_day"] // 60
        out["part"] = "morning" if 4 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening"
    if features.get("short_night") is not None:
        out["night"] = "short" if features["short_night"] else "usual"
    return out


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval for k of n: where the true rate plausibly lies. (0, 1) with nothing seen."""
    if n <= 0:
        return 0.0, 1.0
    p, zz = k / n, z * z
    mid, half = (p + zz / (2 * n)) / (1 + zz / n), z * sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return max(0.0, mid - half), min(1.0, mid + half)


def fisher(k: int, n: int, k_rest: int, n_rest: int) -> float:
    """Fisher's exact test, one-sided: how often k or more of the n would have eased by chance alone, had what the
    person did made no difference. 1.0 when either side is empty."""
    total, eased = n + n_rest, k + k_rest
    if n <= 0 or n_rest <= 0:
        return 1.0
    return min(1.0, sum(comb(n, x) * comb(n_rest, eased - x) for x in range(k, min(n, eased) + 1)) / comb(total, eased))


def coin(k: int, n: int) -> float:
    """How often k or more of n would come up if each were a coin flip."""
    return sum(comb(n, x) for x in range(k, n + 1)) / 2 ** n if n > 0 else 1.0


@dataclass(frozen=True)
class Arm:
    """One thing the person was seen to do after a look, and how the look went."""

    name: str
    n: int = 0  # times seen, among the known outcomes
    eased: int = 0
    rate: float = 0.0  # smoothed towards the person's overall rate
    low: float = 0.0  # the Wilson band on the raw counts
    high: float = 1.0
    rest_n: int = 0  # the known outcomes where they did something else
    rest_eased: int = 0
    p: float = 1.0  # Fisher, one-sided: how often counts this much in its favour would come up by chance
    helps: bool = False  # a finding: the look eases more often with it than without
    by: dict = field(default_factory=dict)  # feature -> bin -> (n, eased)


@dataclass(frozen=True)
class Common:
    """Something the remembered looks share: `k` of the `n` looks that had it known."""

    key: str  # "sitting", "dry", "short_night", "rhr_up", "energy", "deadline", "part"
    k: int
    n: int
    minutes: int | None = None  # "sitting": the usual time at the desk, to a quarter of an hour
    part: str | None = None  # "part": which part of the day


@dataclass(frozen=True)
class Model:
    total: int = 0  # looks it was given
    known: int = 0  # ... with a known outcome: the ones the outcome side is fitted on
    eased: int = 0
    pooled: float = 0.5  # the person's overall rate of a look easing, smoothed
    arms: dict = field(default_factory=dict)  # name -> Arm, for every arm (n may be 0)
    commons: tuple = ()  # what the looks share, strongest first

    @property
    def unknown(self) -> int:
        return self.total - self.known

    @property
    def enough(self) -> bool:
        return self.known >= MIN_KNOWN

    @property
    def onset_min(self) -> int | None:
        """The usual minutes at the desk when a look starts, when that is a finding."""
        return next((c.minutes for c in self.commons if c.key == "sitting"), None)


def fit(examples: list[dict]) -> Model:
    """The model of one person's looks, from the look memory's labelled rows. The order of the rows does not matter."""
    rows = [(x.get("features") or {}, x.get("outcome") or {}) for x in examples]
    known = [(f, o) for f, o in rows if o.get("known") and o.get("eased") is not None]
    n_all, k_all = len(known), sum(1 for _, o in known if o["eased"])
    pooled = (k_all + 1) / (n_all + 2)
    arms: dict[str, Arm] = {}
    for name in ARMS:
        mine = [(f, o) for f, o in known if did(o) == name]
        n, k = len(mine), sum(1 for _, o in mine if o["eased"])
        by: dict[str, dict[str, tuple[int, int]]] = {}
        for f, o in mine:
            for key, value in bins(f).items():
                a, b = by.setdefault(key, {}).get(value, (0, 0))
                by[key][value] = (a + 1, b + int(bool(o["eased"])))
        low, high = wilson(k, n)
        rest_n, rest_k = n_all - n, k_all - k
        p = fisher(k, n, rest_k, rest_n)
        helps = n_all >= MIN_KNOWN and n >= MIN_ARM and rest_n >= MIN_ARM and p <= ALPHA
        arms[name] = Arm(name=name, n=n, eased=k, rate=(k + PRIOR * pooled) / (n + PRIOR), low=low, high=high, rest_n=rest_n, rest_eased=rest_k,
                         p=p, helps=helps, by={key: dict(sorted(v.items())) for key, v in sorted(by.items())})
    return Model(total=len(rows), known=n_all, eased=k_all, pooled=pooled, arms=arms, commons=tuple(commons([f for f, _ in rows])))


def _strong(k: int, n: int) -> bool:
    return n >= MIN_ONSET and k / n >= COMMON_SHARE and coin(k, n) <= COMMON_ALPHA


def commons(features: list[dict]) -> list[Common]:
    """What the looks share, strongest first: counts, never causes (the box keeps no moment without a look to compare with)."""
    out: list[Common] = []
    sat = sorted(f["sitting_min"] for f in features if f.get("sitting_min") is not None)
    if len(sat) >= MIN_ONSET:
        q1, q3 = sat[(len(sat) - 1) // 4], sat[-((len(sat) - 1) // 4) - 1]
        usual = q1 // ONSET_STEP * ONSET_STEP
        k = sum(1 for v in sat if v >= usual)
        if usual >= ONSET_MIN and q3 - q1 <= ONSET_SPREAD and k / len(sat) >= COMMON_SHARE:  # bunched: that is the test here
            out.append(Common("sitting", k, len(sat), minutes=usual))
    flags = {
        "dry": lambda f: None if f.get("since_drink_min") is None else f["since_drink_min"] >= rule.DRINK_GAP_MIN,
        "short_night": lambda f: f.get("short_night"),
        "rhr_up": lambda f: f.get("rhr_up"),
        "energy": lambda f: None if f.get("energy_drinks") is None or not f.get("energy_limit") else f["energy_drinks"] > f["energy_limit"],
        "deadline": lambda f: None if "deadline" not in f else bool(f["deadline"]),
    }
    for key, read in flags.items():
        got = [v for v in (read(f) for f in features) if v is not None]
        if key == "deadline":  # no deadline is not "unknown" once the box knows of any at all
            got = got if any(got) else []
        k = sum(1 for v in got if v)
        if _strong(k, len(got)):
            out.append(Common(key, k, len(got)))
    parts = [b["part"] for b in map(bins, features) if "part" in b]
    for part in ("morning", "afternoon", "evening"):
        if _strong(parts.count(part), len(parts)):
            out.append(Common("part", parts.count(part), len(parts), part=part))
    return sorted(out, key=lambda c: (c.key != "sitting", -c.k / c.n, -c.n, c.key))


# ---------- predictions: the advice order ----------


def predict(model: Model, features: dict | None = None) -> dict[str, float]:
    """For each arm, how likely the look is to ease in a context like `features`: the arm's smoothed rate, moved
    towards its counts in each bin the context falls in. Every number is counts and PRIOR, nothing else."""
    now = bins(features or {})
    out = {}
    for name, arm in model.arms.items():
        parts = []
        for key, value in now.items():
            n, k = arm.by.get(key, {}).get(value, (0, 0))
            parts.append((k + PRIOR * arm.rate) / (n + PRIOR))
        out[name] = sum(parts) / len(parts) if parts else arm.rate
    return out


def explain(model: Model, name: str, features: dict | None = None) -> list[str]:
    """Why `predict` gives what it gives for an arm: the counts behind it, overall and in each bin of the context."""
    arm = model.arms[name]
    lines = [f"{arm.eased} of {arm.n} overall"]
    for key, value in bins(features or {}).items():
        n, k = arm.by.get(key, {}).get(value, (0, 0))
        lines.append(f"{k} of {n} with {key} = {value}")
    return lines


def lead(model: Model, features: dict | None = None) -> str | None:
    """The advice to say first in a context like `features`: "walk" (getting up: the check-in's walk or break) when the
    counts favour it over a drink, "drink" when they favour that, None when there is not enough yet or they do not
    tell the two apart. An arm never seen sits at the person's own average, so one seen to work goes ahead of it
    and one seen not to goes behind."""
    if not model.enough:
        return None
    score = predict(model, features)
    up, drink = round(score[UP], 9), round(score[DRINK], 9)
    return "walk" if up > drink else "drink" if drink > up else None


def rerank(suggestions: list[CheckinSuggestion], model: Model, features: dict | None = None) -> list[CheckinSuggestion]:
    """The same advice (the same set, the same words) in the order the counts support. Unchanged when the model has not
    enough to go on or does not tell the advice apart. The offer to move a block stays last, as in stage one."""
    first = lead(model, features)
    if first is None:
        return list(suggestions)
    return sorted(suggestions, key=lambda s: (s.kind == "move_block", ADVICE_ARM.get(s.kind) != ADVICE_ARM[first]))


# ---------- the words ----------

ARM_WORDS = {UP: "getting up", DRINK: "water alone", STAY: "staying at the {workplace}"}
ARM_DID = {UP: "you got up", DRINK: "you only had a drink", STAY: "you stayed at the {workplace}"}


def _arm(name: str, w: Words, table: dict = ARM_WORDS) -> str:
    return table[name].format(workplace=w.workplace)


def _ranked(model: Model, features: dict | None = None) -> list[Arm]:
    """The arms that were seen at all, best first by the prediction for this context; ties in the order of ARMS."""
    score = predict(model, features)
    return sorted((a for a in model.arms.values() if a.n), key=lambda a: (-round(score[a.name], 9), ARMS.index(a.name)))


def pair(model: Model, features: dict | None = None) -> tuple[Arm | None, Arm | None]:
    """(the arm the counts favour in this context, one other for contrast: water alone when it was seen, else
    staying put, else getting up)."""
    ranked = _ranked(model, features)
    if not ranked:
        return None, None
    return ranked[0], next((a for name in (DRINK, STAY, UP) for a in ranked[1:] if a.name == name), None)


def _lead_words(a: Arm, w: Words) -> str:
    subject = cap(_arm(a.name, w))
    if a.eased == 0:
        return f"{subject} hasn't eased this {'the one time' if a.n == 1 else 'either time' if a.n == 2 else 'any of the last ' + say_number(a.n) + ' times'}"
    if a.eased == a.n:
        return f"{subject} has eased this {'the one time' if a.n == 1 else 'both times' if a.n == 2 else 'each of the last ' + say_number(a.n) + ' times'}"
    return f"{subject} has eased this {say_number(a.eased)} of the last {say_number(a.n)} times"


def _other_words(a: Arm, w: Words) -> str:
    subject = _arm(a.name, w)
    if a.eased == 0:
        return f"{subject} hasn't " + ("the one time" if a.n == 1 else "either time" if a.n == 2 else f"any of the {say_number(a.n)} times")
    if a.eased == a.n:
        return f"{subject} has too, " + ("the one time" if a.n == 1 else "both times" if a.n == 2 else f"all {say_number(a.n)} times")
    return f"{subject} has {say_number(a.eased)} of {say_number(a.n)}"


def learned_line(model: Model, features: dict | None = None, w: Words = Words()) -> str:
    """What the counts say, for a check-in, in one short sentence: the arm the counts favour, and one other for
    contrast (water alone when it was seen, else staying put). "" when there is not enough yet. Counts only: it
    never says "helps", and it passes the check-in's rule against diagnosis words."""
    if not model.enough:
        return ""
    lead, other = pair(model, features)
    if lead is None:
        return ""
    return _lead_words(lead, w) + (f"; {_other_words(other, w)}." if other else ".")


def common_line(c: Common, w: Words = Words()) -> str:
    """One thing the looks share, for the ear, as one clause-sized sentence."""
    if c.key == "sitting":
        return f"It usually shows up after about {say_minutes(c.minutes).removeprefix('about ')} at the {w.workplace}."
    return {"dry": "It usually shows up when I haven't seen you pick up a drink in a while.",
            "short_night": "It usually shows up after a short night.",
            "rhr_up": "It usually shows up on days your resting heart rate is up.",
            "energy": "It usually shows up on days you're over your energy drink limit.",
            "deadline": f"It usually shows up with {'an' if w.deadline[:1] in 'aeiou' else 'a'} {w.deadline} due within a day.",
            "part": f"It usually shows up in the {c.part}."}[c.key]


def pattern_line(model: Model, w: Words = Words()) -> str:
    """The one cause-side clause a check-in may add: the strongest thing the looks share, or ""."""
    return common_line(model.commons[0], w) if model.enough and model.commons else ""


def evidence_line(model: Model) -> str:
    """The model's own evidence: "From 9 remembered looks, 8 with a known outcome." """
    looks = f"{model.total} remembered look{'' if model.total == 1 else 's'}"
    return f"From {looks}, {model.known} with a known outcome."


def ahead_line(model: Model, sitting_min: int | None, w: Words = Words()) -> str:
    """For the check-in card only, never said aloud or pushed: the person has been at the desk about as long as it
    usually takes for a look to start. "" unless that is a finding and it is "around now"."""
    usual = model.onset_min if model.enough else None
    if usual is None or sitting_min is None or not usual - AHEAD_MIN <= sitting_min <= usual + AROUND_MIN:
        return ""
    helps = model.arms[UP].helps
    return "You usually start to look strained around now." + (" A short break tends to help." if helps else "")


# ---------- for the People screen: findings as sentences with their counts ----------


def _percent(x: float) -> int:
    return int(round(x * 100))


def arm_finding(a: Arm, w: Words = Words()) -> dict:
    """One arm as a sentence with its counts, its band and whether it is a finding."""
    name, did_ = cap(_arm(a.name, w)), _arm(a.name, w, ARM_DID)
    counts = f"the look {'eased' if a.eased else 'did not ease'} the one time {did_}" if a.n == 1 else f"the look eased {a.eased} of the {a.n} times {did_}"
    rest = f"{a.rest_eased} of the {a.rest_n} times you didn't" if a.rest_n > 1 else ""
    if a.helps:
        text = f"{name} helps: {counts}" + (f", against {rest}." if rest else ".")
    else:
        text = f"{name}: {counts}. Too few to call it either way."
    return {"kind": a.name, "text": text, "eased": a.eased, "n": a.n, "finding": a.helps,
            "band": f"Likely between {_percent(a.low)}% and {_percent(a.high)}% of the time, on this little."}


def common_finding(c: Common, w: Words = Words()) -> dict:
    share = f"{c.k} of {c.n} looks"
    text = {"sitting": f"It usually starts after about {c.minutes} minutes at the {w.workplace}: {share} did.",
            "dry": f"It usually starts when no drink was seen for {rule.DRINK_GAP_MIN} minutes or more: {share}.",
            "short_night": f"It usually comes after a short night: {share}.",
            "rhr_up": f"It usually comes on days your resting heart rate is above your usual: {share}.",
            "energy": f"It usually comes on days over your energy drink limit: {share}.",
            "deadline": f"It usually comes with {'an' if w.deadline[:1] in 'aeiou' else 'a'} {w.deadline} due within a day: {share}.",
            "part": f"It usually starts in the {c.part}: {share}."}[c.key]
    low, high = wilson(c.k, c.n)
    return {"kind": c.key, "text": text, "eased": c.k, "n": c.n, "finding": True,
            "band": f"Likely between {_percent(low)}% and {_percent(high)}% of your looks."}


SHARED_NOTE = ("What the looks share is a count, not a cause: I keep the times a look started, not the afternoons you looked fine, "
               "so I have nothing to compare them with.")
HOW_NOTE = ("Worked out on this box from the remembered looks above, by counting. No picture is used and nothing is sent anywhere. "
            "It is a count, not a reading of how you feel.")


def view(model: Model, w: Words = Words()) -> dict:
    """What Atlas has learned, for the owner's own screen: two or three findings as sentences with their counts, the
    evidence line, and "not enough yet" when that is the truth. Words and whole numbers only."""
    left_out = (f" {model.unknown} {'is' if model.unknown == 1 else 'are'} left out: I couldn't see how "
                f"{'it' if model.unknown == 1 else 'they'} went.") if model.unknown else ""
    base = {"enough": model.enough, "looks": model.total, "known": model.known, "left_out": model.unknown, "min_known": MIN_KNOWN,
            "evidence": evidence_line(model) + left_out, "note": HOW_NOTE, "shared_note": SHARED_NOTE if model.commons else ""}
    if not model.enough:
        need = MIN_KNOWN - model.known
        return base | {"findings": [], "not_enough": f"Not enough yet. I need to see how {need} more look{'' if need == 1 else 's'} "
                                                     f"went before I say what helps ({model.known} of {MIN_KNOWN} so far)."}
    arms = [arm_finding(a, w) for a in pair(model) if a is not None]
    shared = [common_finding(c, w) for c in model.commons[:1]]
    return base | {"findings": arms + shared, "not_enough": ""}


# ---------- answers to a question (the agent's port; for the ear: contexts/day/domain/ear.py's rule) ----------

NOT_YET = "I don't know yet."


def _said_counts(a: Arm, w: Words) -> str:
    did_ = _arm(a.name, w, ARM_DID)
    if a.n == 1:
        return f"the look {'eased' if a.eased else 'did not ease'} the one time {did_}"
    if a.n == 2 and not a.eased:
        return f"the look eased neither of the two times {did_}"
    return f"the look eased {say_number(a.eased) if a.eased else 'none'} of the {say_number(a.n)} times {did_}"


def said_helps(model: Model, w: Words = Words()) -> str:
    """ "What usually helps when I look tired?" """
    if not model.enough:
        seen = "how one look went" if model.known == 1 else f"how {say_number(model.known)} looks went" if model.known else "no look through to the end"
        return f"{NOT_YET} I've seen {seen}, and I need {say_number(MIN_KNOWN)} before I'll say."
    best, other = pair(model)
    if best is None:
        return f"{NOT_YET} I couldn't see what you did after the looks I remember."
    counts = cap(_said_counts(best, w)) + (f", and {_said_counts(other, w).removeprefix('the look ')}" if other else "") + "."
    if best.helps:
        return f"{cap(_arm(best.name, w))}, from what I've seen. {counts}"
    return f"I can't say for sure yet. {counts}"


def said_causes(model: Model, w: Words = Words()) -> str:
    """ "What makes me look tired?": what the remembered looks share. Never a cause, and it says so."""
    if model.total < MIN_ONSET:
        have = "one look" if model.total == 1 else f"{say_number(model.total)} looks"
        return f"{NOT_YET} I remember {have}, and I need {say_number(MIN_ONSET)} before I'll say what they share."
    if not model.commons:
        return f"Nothing stands out yet in the {say_number(model.total)} looks I remember. I can't tell you what causes a look, only what they share."
    c = model.commons[0]
    of = f"{say_number(c.k)} of the {say_number(c.n)} looks I remember"
    shared = {"sitting": f"started after {say_minutes(c.minutes or 0).removeprefix('about ')} or more at the {w.workplace}",
              "dry": "started when I hadn't seen you pick up a drink in a while", "short_night": "came after a short night",
              "rhr_up": "came on days your resting heart rate was up", "energy": "came on days you were over your energy drink limit",
              "deadline": f"came with {'an' if w.deadline[:1] in 'aeiou' else 'a'} {w.deadline} due within a day",
              "part": f"started in the {c.part}"}[c.key]
    return f"I can't say what causes it, only what the looks share: {of} {shared}."


def said_how(model: Model, w: Words = Words()) -> str:
    """ "How do you know?": the counts, where they came from, and that it is no more than that."""
    if not model.total:
        return "I don't, yet: I haven't got a look remembered on this box."
    have = "the one look" if model.total == 1 else f"the {say_number(model.total)} looks"
    best, other = pair(model) if model.enough else (None, None)
    if best is not None:
        counts = f"{_arm(best.name, w)} eased it {say_number(best.eased) if best.eased else 'none'} of {say_number(best.n)} times"
        if other is not None:
            counts += f", and {_arm(other.name, w)} {say_number(other.eased) if other.eased else 'none'} of {say_number(other.n)}"
        return f"From what I saw on this box, nothing more: {counts}. It's a count over {have} I remember, not a reading of how you feel."
    went = "how one went" if model.known == 1 else f"how {say_number(model.known)} went" if model.known else "how none went yet"
    return f"From what I saw on this box, nothing more: I remember {have.removeprefix('the ')} and saw {went}. It's a count, not a reading of how you feel."
