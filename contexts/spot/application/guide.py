"""Look guide: from "where do I look for X?" to a place on the Inspection screen, and the words to walk someone there.

    Guide(spot).resolve("where do I look for heat exchanger 4?")
    -> {"target": "asset:HX-204", "label": "Heat exchanger 4 (HX-204)", "confidence": 1.0, "alternatives": [],
        "steps": [{"reveal": "tab:shift", "say": "I am going back to Your shift."},
                  {"reveal": "map:floor:1", "say": "I am switching the plan to floor 1."},
                  {"say": "Look at the plan of the plant floor, floor 1."},
                  {"say": "The red mark at Pipe Gallery, on the right, is Heat exchanger 4."}]}

Deterministic: names are matched by code (ids first, then words with a little tolerance for spelling); no model is
called. A `target` is a guide id that the screen tags with `data-guide` (web/lib/guideSpot.ts lists them and knows
how to bring each one into view). A step with `reveal` is only spoken when the screen really had to do that (switch
a tab, change the floor, open a fold); the last step is said when the pointer arrives.

When nothing matches well enough, `target` is None and `alternatives` holds the two or three closest ("Did you mean").
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from contexts.spot.application import plain
from contexts.spot.application.service import Spot
from contexts.spot.domain.directions import directions
from contexts.spot.domain.route import WalkNet, heading

TRACED_NOTE = "Paths traced from the kit's floor-plan picture; the kit has no walkway data."
TRACED_REASON = "traced from the picture"
TAP_SNAP_PX = 60.0  # a tap this close to walkable floor is taken to mean that floor

CONFIDENT = 0.75  # at or above: point there. Below: offer the closest instead of guessing
OFFER = 0.34  # the least an alternative may score

# "where do I look for the battery chart?" -> "battery chart"
LEAD = re.compile(
    r"^(?:(?:hey |ok |okay )?atlas[, ]+)?(?:please |can you |could you |so |and |ok |okay )*"
    r"(?:where (?:do|can|should|would|could) (?:i|we|you) (?:look|go|see|find|check|get|start)(?: (?:for|at|to))?"
    r"|where (?:is|are|s)|wheres|where (?:did|does|do) (?:.+? )??(?:go|live|sit)(?= )"
    r"|where (?:can|do|could|would) (?:i|we) (?:find|see|get to|get|read|check|change|set|write)"
    r"|(?:show|point|take|guide|walk|bring|direct|lead) me(?: (?:to|at|where|towards|through))?(?: (?:is|are))?"
    r"|(?:show|find|locate|highlight|point (?:to|at|out)|go to|open|look (?:for|at)|guide me to)"
    r"|how do i (?:get to|find|see|open|reach)|i (?:am looking|m looking|want to see|need to find|cant find|can t find|need)(?: for)?"
    r"|which (?:one|part|bit) is|what about)\b ?")
TRAIL = re.compile(r"\b(?:please|thanks|thank you|on (?:the|this) (?:screen|page|plan|map)|on screen|for me|right now|again|at)$")
FILLER = {"the", "a", "an", "my", "our", "that", "this", "those", "these", "to", "of", "for", "is", "are", "it", "i", "me", "on", "in", "at",
          "some", "any", "do", "can", "where", "look", "find", "show", "please", "s", "about", "be", "with", "and", "or", "does", "which", "what"}
# words that say "a kind of place" or "a kind of screen part": they count a little, never enough to match on their own
SAME = {"bay": "room", "area": "room", "hall": "room", "station": "room", "zone": "room", "panel": "panel", "section": "card", "box": "card",
        "button": "card", "list": "card", "chart": "card", "picture": "card", "graph": "card", "explainer": "card", "slider": "card",
        "page": "tab", "screen": "tab", "view": "tab", "plan": "plan", "upstairs": "floor 2", "downstairs": "floor 1",
        "anomaly": "alarm", "anomalies": "alarm", "mission": "round", "missions": "round", "waypoint": "stop", "inspection": "check",
        "inspections": "check", "v1": "version", "v2": "version"}
LIGHT = {"room", "card", "tab", "floor", "stop", "equipment", "spot", "check", "thing", "part", "one"}
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "first": "1", "second": "2", "third": "3", "1st": "1", "2nd": "2", "3rd": "3", "ground": "1"}
ORDINAL = ["", "first", "second", "third"]
MARK = {"act": "red mark", "watch": "amber triangle", "ok": "green tick"}
EXPLAINERS = {  # id -> (title, what it shows, what people call it)
    "hidden": ("A completed round can hide failed checks", "which completed rounds had a check that failed",
               ["hidden failures", "completed but failed", "completed rounds that failed", "hidden failed checks explainer"]),
    "alarms": ("Are the alarms real", "which alarms came from a check that worked",
               ["alarms", "alarm", "are the alarms real", "alarms not confirmed", "not confirmed alarms", "false alarms"]),
    "battery": ("Low battery, more failed checks", "how checks fail more often as the battery runs down",
                ["battery", "battery explainer", "low battery", "does battery matter", "battery and failed checks", "charge", "battery effect"]),
    "version": ("Is the new setup better", "the old setup next to the new one",
                ["is v2 better", "v2 better than v1", "version", "versions", "old setup new setup", "new setup", "setup update",
                 "is spot getting better", "v1 against v2", "improved setup"]),
    "weak": ("Where SPOT struggles", "the places and the kinds of check that fail most",
             ["weak spots", "where spot struggles", "where checks fail", "least reliable", "weakest place", "which checks fail most",
              "where failures are", "trouble spots"]),
    "missing": ("What the data cannot say", "what is missing from the data",
                ["missing data", "what is missing", "what the data cannot say", "gaps in the data"]),
}


def words(text: str) -> list[str]:
    """Lower-case content words, with numbers and near-synonyms made the same: 'Heat Exchanger 02' -> heat exchanger 2."""
    out: list[str] = []
    low = text.lower().replace("’", "'").replace("'s", " ").replace("'", " ")
    tokens: list[str] = []
    for whole in re.findall(r"[a-z0-9]+", low):  # "v2" is a word of its own; "hx202" is a tag and a number
        tokens.extend([whole] if whole in SAME else re.findall(r"[a-z]+|\d+", whole))
    for raw in tokens:
        w = NUMBER_WORDS.get(raw, raw)
        if w.isdigit():
            out.append(str(int(w)))
            continue
        if w in FILLER:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.extend(SAME.get(w, w).split())
    return out


def _weight(w: str) -> float:
    return 1.5 if w.isdigit() else 0.3 if w in LIGHT else 1.0


def _same(a: str, b: str) -> float:
    """1 for the same word, a little less for a near spelling ('exchnager'), 0 otherwise. Numbers must be equal."""
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit() or min(len(a), len(b)) < 4:
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.78 else 0.0


def similarity(asked: list[str], name: list[str]) -> float:
    """How well the asked words name this thing: how much of the name was said, and how much of what was said is the name."""
    if not asked or not name:
        return 0.0
    got_name = sum(_weight(n) * max(_same(n, a) for a in asked) for n in name)
    got_asked = sum(_weight(a) * max(_same(a, n) for n in name) for a in asked)
    if max(got_name, got_asked) < 0.75:  # nothing but a light word in common
        return 0.0
    return round(0.55 * got_name / sum(map(_weight, name)) + 0.45 * got_asked / sum(map(_weight, asked)), 3)


@dataclass
class Target:
    id: str
    label: str
    names: list[str]
    steps: list[dict]
    keys: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.keys = [words(n) for n in self.names]


def _where(col: float, row: float) -> str:
    """Thirds of a box, in words."""
    c = "on the left" if col < 0.34 else "on the right" if col > 0.66 else ""
    r = "at the top" if row < 0.34 else "at the bottom" if row > 0.66 else ""
    if c and r:
        return f"{r} {c.replace('on the ', '')}"
    return c or r or "near the middle"


SHIFT = {"reveal": "tab:shift", "say": "I am going back to Your shift."}
ANALYSIS = {"reveal": "tab:analysis", "say": "I am opening the Analysis tab."}
RUN = {"reveal": "tab:shift-run", "say": "I am opening Shift Run."}


class Guide:
    def __init__(self, spot: Spot) -> None:
        self.spot = spot
        self._targets: list[Target] | None = None
        self._net: WalkNet | None = None

    # --- what is on the screen ---

    def _spot_on_plan(self, location_id: str) -> str:
        kit, plan = self.spot.kit, self.spot.plan
        loc = kit.location[location_id]
        point = (plan or {}).get("waypoints", {}).get(location_id)
        panel = (plan or {}).get("panels", {}).get(str(loc.floor))
        if point and panel:
            return _where((point["x"] - panel["x0"]) / max(panel["x1"] - panel["x0"], 1e-6), (point["y"] - panel["y0"]) / max(panel["y1"] - panel["y0"], 1e-6))
        return _where(loc.x_coord / 100, loc.y_coord / 100)

    def targets(self) -> list[Target]:
        if self._targets is not None:
            return self._targets
        kit = self.spot.kit
        view = plain.shift(self.spot)
        floors = sorted({x.floor for x in kit.locations})
        out: list[Target] = []
        add = lambda id, label, names, *steps: out.append(Target(id, label, names, list(steps)))  # noqa: E731
        say = lambda text: {"say": text}  # noqa: E731

        # the first screen
        add("status", "Your shift, in one line", ["status", "status line", "summary", "shift summary", "headline", "how many need me", "overview", "how many things need me"],
            SHIFT, say("Look at the very top of the page."), say("This line is your shift in one sentence."))
        if view.get("trust"):
            add("trust", "Can you trust a completed round", ["trust", "can i trust a completed round", "trust the last round", "failed checks", "hidden failures",
                                                              "completed round", "how much can i trust", "trust line"],
                SHIFT, say("Look just under the top line."), say("This says how often a completed round still had a failed check. Tap it for the why."))
        add("card:do-first", "Do this first", ["do this first", "what to do first", "do first", "what do i do first", "what should i check first", "what needs me",
                                                "priorities", "to do", "urgent", "what needs attention", "my jobs", "where do i start", "start"],
            SHIFT, say("Look at the card called Do this first."), say("These are your jobs, in order. Start with number 1."))
        for n, it in enumerate(view["do_first"], 1):
            add(f"do-first:{n}", it["headline"], [f"{ORDINAL[n]} thing", f"{ORDINAL[n]} job", f"number {n}", f"job {n}", f"item {n}", f"do first {n}", f"{ORDINAL[n]} thing to do",
                                                   f"{ORDINAL[n]} priority"],
                SHIFT, say("Look at the card called Do this first."), say(f"Number {n}: {it['headline']}."))
        add("card:watch", "Watch out", ["watch out", "watch", "patterns", "warnings", "what to watch", "what to keep an eye on", "watch out card", "patterns today"],
            SHIFT, say("Look at the card called Watch out."), say("These are the patterns that affect today."))
        add("card:fine", "Looks fine", ["looks fine", "fine", "what is fine", "what is ok", "what is okay", "healthy", "what is working", "nothing wrong", "good equipment"],
            SHIFT, say("Look at the card called Looks fine."), say("This counts the equipment with a good recent check and no open alarm."))
        add("all-equipment", "See all equipment", ["all equipment", "every piece of equipment", "equipment list", "full list", "see all", "list of equipment", "everything"],
            SHIFT, say("Look at the card called Looks fine."), say("This button lists every piece of equipment, worst first."))
        add("map", "The plant floor", ["map", "floor plan", "plan", "plant floor", "layout", "the plant", "plan of the plant", "where things are"],
            SHIFT, say("Look at the plan of the plant floor."), say("One mark for each piece of equipment. Red needs you."))
        for f in floors:
            add(f"map:floor:{f}", f"Floor {f}", [f"floor {f}", f"level {f}", f"floor {f} button", f"{ORDINAL[f] if f < 4 else f} floor"],
                SHIFT, say("Look at the top of the plan of the plant floor."), say(f"This button shows floor {f}."))
        add("tabs", "The three views", ["tabs", "views", "the three views", "switch view", "other screens"],
            say("Look at the top of the page, on the right."), say("These three buttons switch between your shift, Shift Run and the analysis."))
        add("tab:shift", "Your shift", ["your shift", "my shift", "shift tab", "first screen", "home", "back"], say("Look at the buttons at the top."), say("This one is Your shift."))
        add("tab:shift-run", "Shift Run", ["shift run", "shift run tab", "plan the next round", "plan a round", "plan next round", "plan my round", "the game", "training",
                                           "simulate a round", "simulation", "practice"],
            say("Look at the buttons at the top."), say("This one is Shift Run. It is where you plan SPOT's next round."))
        add("tab:analysis", "Analysis", ["analysis", "analysis tab", "the numbers", "evidence", "statistics", "stats", "the analysis", "the data behind this", "proof"],
            say("Look at the buttons at the top."), say("This one is Analysis. Every sentence is backed up there."))
        add("ask", "Ask SPOT's data", ["ask", "ask box", "ask a question", "question box", "ask spot", "type a question", "search", "where do i ask", "ask spot data"],
            SHIFT, say("Look at the card called Ask SPOT's data."), say("Type a question here, in your own words."))
        add("handoff", "Write my shift handoff", ["handoff", "shift handoff", "hand off", "hand over", "handover", "notes for the next shift", "next operator", "write my handoff",
                                                   "end of shift note"],
            SHIFT, say("Look at the card called Ask SPOT's data."), say("This button writes your shift handoff as a note you can copy."))

        # a round, replayed
        add("round", "A round, step by step", ["round", "rounds", "the round", "which round", "last round", "latest round", "replay", "round step by step", "what spot did",
                                               "a round replayed", "round replay"],
            SHIFT, say("Look below the cards, at A round, step by step."), say("This is one round. Each square is one check."))
        add("round:pick", "Choose a round", ["choose a round", "pick a round", "another round", "change round", "round list", "older rounds", "round picker", "select a round"],
            SHIFT, say("Look at A round, step by step."), say("This list picks which round you are looking at."))
        add("round:steps", "The steps of the round", ["steps", "steps of the round", "the checks", "checks in the round", "each check", "squares"],
            SHIFT, say("Look at A round, step by step."), say("One square for each check. A tick worked, a cross failed."))
        add("round:play", "Play the round", ["play", "play button", "play the round", "replay the round", "watch the round", "watch it unfold", "start the replay"],
            SHIFT, say("Look at A round, step by step."), say("Press Play to watch the round as it happened."))
        add("round:battery", "SPOT's battery in this round", ["spot battery", "battery level", "battery now", "how much battery", "battery in the round", "battery percent",
                                                              "battery gauge"],
            SHIFT, say("Look at A round, step by step."), say("This is SPOT's battery at this step of the round."))
        add("round:map", "Where SPOT went", ["where spot went", "route", "spot route", "the route", "path", "where did spot go", "spot path"],
            SHIFT, say("Look at A round, step by step."), say("This plan shows the route SPOT walked in this round."))
        for r in view["rounds"]:
            n = r["number"]
            add(f"round:{n}", r["label"], [f"round {n}", f"mission {n}", r["mission_id"], f"round number {n}"],
                SHIFT, {"reveal": f"round:pick:{r['mission_id']}", "say": f"I am picking round {n}."}, say("Look at A round, step by step."),
                say(f"This is round {n}, {r['when']}: {r['result']}."))

        # Shift Run
        add("shiftrun:turn-back", "When SPOT turns back to charge", ["turn back", "turn back slider", "turn back to charge", "when to charge", "when spot turns back",
                                                                     "charge slider", "turn around", "battery slider", "turn back at"],
            RUN, say("Look at the panel beside the plan."), say("Drag this slider to choose the battery level where SPOT turns back to charge."))
        add("shiftrun:run", "Run my plan", ["run my plan", "run it", "run the plan", "run button", "try my plan", "start the run"],
            RUN, say("Look at the panel beside the plan."), say("This button runs your plan and shows how it holds up."))
        add("shiftrun:pool", "Add a stop", ["add a stop", "add stops", "stops to add", "pick stops", "choose stops", "what spot can check", "candidates"],
            RUN, say("Look at the panel beside the plan."), say("Tap a line here to add that stop to your round."))
        add("shiftrun:last", "Start from last round", ["start from last round", "last round plan", "copy the last round", "same as last round"],
            RUN, say("Look at the panel beside the plan."), say("This button starts your plan from SPOT's last round."))
        add("shiftrun:map", "The plan you are building", ["shift run plan", "my route", "my plan", "planning plan", "tap equipment"],
            RUN, say("Look at the plan of the plant floor."), say("Tap a mark to add it to your round."))

        # the analysis
        for eid, (title, shows, names) in EXPLAINERS.items():
            add(f"explainer:{eid}", title, names, ANALYSIS, say("Look at the pictures under the heading."), say(f"This one is {title}. It shows {shows}."))
        add("analysis:findings", "Every finding in one sentence", ["all findings", "every finding", "findings", "list of findings", "findings list", "everything you found"],
            ANALYSIS, say("Look below the pictures."), say("This fold lists every finding in one sentence."))
        add("analysis:tables", "The full tables", ["tables", "full tables", "the tables", "ranked queue", "triage queue", "trained scorer", "scorer", "the model", "intervals",
                                                    "key numbers", "raw numbers"],
            ANALYSIS, say("Look below the pictures."), say("This fold holds the full tables: the ranked queue, the intervals and the trained scorer."))
        add("analysis:assumptions", "Assumptions and method", ["assumptions", "method", "assumptions and method", "how it was worked out", "methodology"],
            ANALYSIS, say("Look at the top of the analysis, on the right."), say("This opens every assumption the analysis makes."))
        add("analysis:data-check", "Data check", ["data check", "validation", "data quality", "check the data", "does the data agree"],
            ANALYSIS, say("Look at the top of the analysis, on the right."), say("This compares our numbers with the kit's own."))
        for f in view["findings"]:
            add(f"finding:{f['id']}", f["headline"], [f["headline"], f["id"].replace("_", " ")],
                ANALYSIS, {"reveal": "open:analysis:findings", "say": "I am opening the list of findings."}, say("Look at the list of findings."),
                say(f"This line: {f['headline']}."))

        # equipment and places, on the plan
        by_place: dict[str, list[dict]] = {}
        for a in view["assets"]:
            by_place.setdefault(a["location_id"], []).append(a)
        for a in view["assets"]:
            raw = kit.asset[a["asset_id"]]
            prefix, _, digits = a["asset_id"].partition("-")
            names = [a["name"], a["asset_id"], f"{prefix} {digits}", a["label"], f"{raw.asset_type} {digits}"]
            if raw.asset_type and len(raw.asset_type.split()) > 1:  # "exchanger 4", "gauge 9"
                names.append(f"{raw.asset_type.split()[-1]} {a['name'].split()[-1]}")
            place = plain.place_name(kit, a["location_id"])
            mark = MARK[a["level"]]
            n_here = len(by_place[a["location_id"]])
            which = f"The {mark} at {place}, {self._spot_on_plan(a['location_id'])}," if n_here == 1 else f"One of the {n_here} marks at {place}, {self._spot_on_plan(a['location_id'])},"
            add(f"asset:{a['asset_id']}", a["label"], names, SHIFT, {"reveal": f"map:floor:{a['floor']}", "say": f"I am switching the plan to floor {a['floor']}."},
                say(f"Look at the plan of the plant floor, floor {a['floor']}."), say(f"{which} is {a['name']}. {a['level_words']}."))
        for x in kit.locations:
            name = plain.place_name(kit, x.location_id)
            n = int(re.sub(r"\D", "", x.location_id) or 0)
            here = by_place.get(x.location_id, [])
            tail = f"{len(here)} piece{'s' if len(here) != 1 else ''} of equipment {'are' if len(here) != 1 else 'is'} here." if here else "SPOT walks through here; nothing is checked."
            add(f"place:{x.location_id}", plain.place_label(kit, x.location_id), [name, x.location_id, f"wp {n}", f"stop {n}", f"{name} room"],
                SHIFT, {"reveal": f"map:floor:{x.floor}", "say": f"I am switching the plan to floor {x.floor}."},
                say(f"Look at the plan of the plant floor, floor {x.floor}."), say(f"{name} is {self._spot_on_plan(x.location_id)} of the plan. {tail}"))
        self._targets = out
        return out

    # --- the question ---

    @staticmethod
    def thing(question: str) -> str:
        """The question without its lead-in: 'Where do I look for the battery?' -> 'the battery'."""
        low = re.sub(r"[^a-z0-9:\-_ ]", " ", question.lower().replace("’", "'").replace("'", " "))
        low = re.sub(r"\s+", " ", low).strip()
        for _ in range(2):
            low = LEAD.sub("", low, count=1).strip()
            low = TRAIL.sub("", low).strip()
        return low

    def resolve(self, question: str) -> dict:
        base = {"configured": self.spot.configured, "query": question, "target": None, "label": None, "steps": [], "confidence": 0.0, "alternatives": []}
        if not self.spot.configured:
            return {**base, "steps": [{"say": "No inspection data is loaded on this box yet, so there is nothing to point at."}]}
        targets = self.targets()
        by_id = {t.id.lower(): t for t in targets}
        thing = self.thing(question)
        exact = by_id.get(question.strip().lower()) or by_id.get(thing.replace(" ", ""))
        if exact is None:
            exact = self._by_id(thing, by_id)
        if exact is not None:
            return {**base, **self._pack(exact, 1.0)}
        asked = words(thing) or words(re.sub(r"[^a-z0-9 ]", " ", question.lower()))  # "where do I start?" is all lead-in
        scored = sorted(((max((similarity(asked, k) for k in t.keys), default=0.0), t) for t in targets), key=lambda p: -p[0])
        best, top = scored[0]
        level = lambda t: t.id.split(":")[0]  # noqa: E731
        rest = [(s, t) for s, t in scored[1:] if s >= OFFER]
        # several things equally well named ("the heat exchanger"): do not pick one for the person
        tied = [t for s, t in rest if best - s < 0.02 and level(t) == level(top)]
        if best >= CONFIDENT and not tied:
            return {**base, **self._pack(top, best), "alternatives": [self._brief(t, s) for s, t in rest[:2] if s >= 0.6]}
        offers = [(best, top), *rest][:3] if best >= OFFER else []
        if not offers:  # nothing close: the three places most questions are about
            offers = [(0.0, by_id[k]) for k in ("card:do-first", "map", "round")]
        names = [t.label for _, t in offers]
        sentence = "I am not sure what to point at. Did you mean " + (", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else names[0]) + "?"
        return {**base, "confidence": best, "steps": [{"say": sentence}], "alternatives": [self._brief(t, s) for s, t in offers]}

    def _by_id(self, thing: str, by_id: dict[str, Target]) -> Target | None:
        """An id said or typed outright: HX-202, hx 202, WP05, waypoint 5, M027, round 27, floor 2."""
        kit = self.spot.kit
        m = re.search(r"\b([a-z]{2,6})[- ]?(\d{3})\b", thing)
        if m and f"{m.group(1)}-{m.group(2)}".upper() in kit.asset:
            return by_id[f"asset:{m.group(1)}-{m.group(2)}".lower()]
        m = re.search(r"\b(?:wp|waypoint|stop) ?0*(\d{1,2})\b", thing)
        if m and f"place:wp{int(m.group(1)):02d}" in by_id:
            return by_id[f"place:wp{int(m.group(1)):02d}"]
        m = re.search(r"\b(?:m|round |mission |round number )0*(\d{1,3})\b", thing)
        if m and f"round:{int(m.group(1))}" in by_id:
            return by_id[f"round:{int(m.group(1))}"]
        return None

    # --- the way there, on foot ---

    def _place_of(self, ref: str) -> str | None:
        """A waypoint from a waypoint id, an equipment tag, or a guide id ('asset:HX-202', 'place:WP05')."""
        kit = self.spot.kit
        key = ref.strip().upper().split(":")[-1]
        if key in kit.location:
            return key
        return kit.asset[key].location_id if key in kit.asset else None

    def _plan_xy(self, location_id: str) -> tuple[float, float]:
        """Where a waypoint is drawn: on the plan picture when the box has one, else the kit's own x/y."""
        point = (self.spot.plan or {}).get("waypoints", {}).get(location_id)
        loc = self.spot.kit.location[location_id]
        return (point["x"] * 100, point["y"] * 100) if point else (loc.x_coord, loc.y_coord)

    def start(self) -> str | None:
        """Where a route starts when nobody said: the first stop of SPOT's latest round."""
        kit = self.spot.kit
        acts = kit.actions_of.get(kit.missions[-1].mission_id, []) if kit.missions else []
        return acts[0].location_id if acts else (kit.locations[0].location_id if kit.locations else None)

    def leg_route(self, start: str | None, goal: str) -> dict:
        """A walking route between two places (or pieces of equipment), over legs SPOT has walked, with the words for it."""
        note = "Route follows paths SPOT has walked in the recorded rounds. Lengths are plan units, not metres."
        base = {"configured": self.spot.configured, "from": None, "to": None, "path": [], "legs": [], "length": 0.0, "floor_changes": 0,
                "found": False, "say": [], "summary": "", "note": note, "places": []}
        if not self.spot.configured:
            return base
        kit = self.spot.kit
        if self._net is None:
            self._net = WalkNet.of(kit)
        a = self._place_of(start) if start else self.start()
        b = self._place_of(goal)
        places = [{"location_id": x.location_id, "name": plain.place_name(kit, x.location_id), "floor": x.floor,
                   "walked": bool(self._net.neighbours(x.location_id))} for x in kit.locations]
        base = {**base, "from": a, "to": b, "places": places}
        if a is None or b is None:
            return {**base, "say": ["I do not know that place on this plan."], "summary": "Unknown place."}
        name = lambda loc: plain.place_name(kit, loc)  # noqa: E731
        if a == b:
            return {**base, "found": True, "path": [a], "say": [f"You are already at {name(a)}."], "summary": f"You are at {name(a)}."}
        found = self._net.route(a, b)
        if found is None:
            line = f"SPOT has never walked a path that joins {name(a)} and {name(b)}, so I will not draw one."
            return {**base, "say": [line], "summary": line}
        say = []
        for i, leg in enumerate(found["legs"]):
            lead = f"From {name(leg['from'])}, go" if i == 0 else "Then"
            if leg["floor_change"]:
                say.append(f"{lead} to floor {kit.location[leg['to']].floor}, to {name(leg['to'])}.")
            else:
                say.append(f"{lead} {heading(self._plan_xy(leg['from']), self._plan_xy(leg['to']))} on the plan to {name(leg['to'])}.")
        n = len(found["legs"])
        summary = (f"{name(a)} to {name(b)}: {n} leg{'s' if n != 1 else ''}"
                   + (f", {found['floor_changes']} change of floor" if found["floor_changes"] else "") + ", all walked by SPOT before.")
        return {**base, **found, "found": True, "say": say[:4] + ([f"{n - 4} more legs after that."] if n > 4 else []), "summary": summary}

    # --- the way there on the traced floor ---

    def _point(self, ref: str | None) -> tuple[int, float, float] | None:
        """A tapped start, 'xy:0.31,0.52' (normalised on the plan picture) -> (floor, x, y)."""
        m = re.fullmatch(r"xy:(0?\.\d+|[01](?:\.\d+)?),(0?\.\d+|[01](?:\.\d+)?)", (ref or "").strip().lower())
        wk = self.spot.walkable
        if not m or wk is None:
            return None
        x, y = float(m.group(1)), float(m.group(2))
        floor = wk.floor_at(x, y)
        return (floor, x, y) if floor is not None else None

    def route(self, start: str | None, goal: str) -> dict:
        """A walking route between two places, pieces of equipment, or from a tapped point, on the traced walkable floor.

        The legs SPOT walked stay in the answer (`via_walked_legs`, and the old top-level `path`, `legs`, `length`): they are
        what the data vouches for. The routed line is ours, from the picture, so it is always `precision: "about"`.
        Without a traced floor (no plan picture on the box) the answer is the walked-legs route, as before.
        """
        tapped = (start or "").strip().lower().startswith("xy:")
        legs = self.leg_route(None if tapped else start, goal)
        wk = self.spot.walkable if self.spot.configured else None
        if wk is None:
            return legs
        kit = self.spot.kit
        via = {k: legs[k] for k in ("found", "path", "legs", "length", "floor_changes", "say", "summary", "note")}
        base = {**legs, "found": False, "say": [], "steps": [], "summary": "", "note": TRACED_NOTE, "via_walked_legs": None if tapped else via,
                "precision": "about", "precision_reason": TRACED_REASON, "walked": False, "times_walked": 0, "routed": None}
        b = legs["to"]
        if b is None:
            return {**base, "say": ["I do not know that place on this plan."], "summary": "Unknown place."}
        key = goal.strip().upper().split(":")[-1]
        goal_name = plain.asset_name(kit, key) if key in kit.asset else plain.place_name(kit, b)
        end = wk.stand(b)
        if tapped:
            point = self._point(start)
            got = wk.snap(point[0], point[1], point[2], limit_px=TAP_SNAP_PX) if point else None
            if got is None or point is None:
                line = "That spot is not on the floor I traced. Tap a corridor or a room on the plan."
                return {**base, "from": None, "say": [line], "summary": line}
            a, start_name, begin, start_floor = None, "where you are", got, point[0]
        else:
            a = legs["from"]
            if a is None:
                return {**base, "say": ["I do not know that place on this plan."], "summary": "Unknown place."}
            start_name, begin, start_floor = plain.place_name(kit, a), wk.stand(a), kit.location[a].floor
        if begin is None or end is None:
            line = "I could not place that waypoint on the traced floor."
            return {**base, "from": a, "say": [line], "summary": line}
        goal_floor = kit.location[b].floor
        pin = wk.pins[b]
        ends = {"start": {"floor": start_floor, "x": begin[0], "y": begin[1], "snapped_px": begin[2], "waypoint": a, "name": start_name},
                "end": {"floor": goal_floor, "x": end[0], "y": end[1], "snapped_px": end[2], "waypoint": b, "name": goal_name,
                        "pin": {"x": pin["x"], "y": pin["y"]}}}
        if a == b:
            line = f"You are already at {plain.place_name(kit, b)}."
            return {**base, "from": a, "found": True, "say": [line], "summary": line, "walked": True,
                    "steps": [{"say": line, "floor": goal_floor, "length_px": 0.0, "upto": 1.0, "arrive": True}],
                    "routed": {"found": True, "connector": None, "length_px": 0.0, "floors": [], **ends}}
        routed = wk.route((start_floor, begin[0], begin[1]), (goal_floor, end[0], end[1]))
        if not routed["found"]:
            line = f"I cannot find a way from {start_name} to {goal_name} on the floor I traced."
            return {**base, "from": a, "say": [line], "summary": line}
        times = self._net.walked.get((min(a, b), max(a, b)), 0) if a else 0
        steps = directions(wk, routed, start_name, goal_name, b, a)
        for part in routed["floors"]:
            part["walked"] = times > 0
        summary = (f"{start_name[0].upper()}{start_name[1:]} to {goal_name}: {len(steps)} steps"
                   + (f", changing floor by {routed.get('connector_name', 'the stairs')}" if routed["connector"] else "") + ". "
                   + (f"SPOT has walked between these two places {times} time{'s' if times != 1 else ''}." if times
                      else "SPOT has not walked this leg; the line is from the traced plan only."))
        return {**base, "from": a, "found": True, "say": [s["say"] for s in steps], "steps": steps, "summary": summary, "walked": times > 0,
                "times_walked": times, "routed": {**routed, **ends}}

    def walkable(self) -> dict:
        """The traced floor, for drawing: rooms, doors, obstacles, stairs, the grid's free cells, and where each waypoint stands."""
        wk = self.spot.walkable if self.spot.configured else None
        if wk is None:
            return {"configured": self.spot.configured, "available": False, "floors": {}, "waypoints": {}, "caption": ""}
        points = {}
        for loc in self.spot.kit.locations:
            pin, stand = wk.pins.get(loc.location_id), wk.stand(loc.location_id)
            if pin:
                points[loc.location_id] = {"floor": loc.floor, "pin": {"x": pin["x"], "y": pin["y"]},
                                           "stand": {"x": stand[0], "y": stand[1]} if stand else None, "snapped_px": stand[2] if stand else None}
        return {"configured": True, "available": True, "source": wk.source, "caption": TRACED_NOTE, "precision": "about",
                "precision_reason": TRACED_REASON, "width": wk.width, "height": wk.height, "cell_px": wk.cell, "clearance_px": wk.clearance,
                "snap_px": wk.snap_px, "floors": wk.geometry(), "waypoints": points}

    def walkable_legs(self) -> dict:
        """The routed line between every pair of waypoints (to draw a round or a plan along real paths), and whether SPOT walked it."""
        wk = self.spot.walkable if self.spot.configured else None
        if wk is None:
            return {"configured": self.spot.configured, "available": False, "legs": {}}
        if self._net is None:
            self._net = WalkNet.of(self.spot.kit)
        ids = sorted(wk.pins)
        legs = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                leg = wk.leg(a, b)
                if leg:
                    n = self._net.walked.get((a, b), 0)
                    legs[f"{a}>{b}"] = {"floors": leg["floors"], "connector": leg["connector"], "walked": n > 0, "times_walked": n}
        return {"configured": True, "available": True, "legs": legs, "note": "Keyed 'A>B' with A before B; walk the points backwards for B to A."}

    @staticmethod
    def _brief(t: Target, score: float) -> dict:
        return {"target": t.id, "label": t.label, "confidence": round(min(score, 1.0), 2)}

    @staticmethod
    def _pack(t: Target, score: float) -> dict:
        return {"target": t.id, "label": t.label, "steps": t.steps, "confidence": round(min(score, 1.0), 2)}
