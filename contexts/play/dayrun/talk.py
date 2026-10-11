"""Change the plan by saying it (POST /api/dayrun/edit, and the voice layer on the Today's Plan screen).

    out = edit(cfg, walks, board, moves, "move lunch to one")

A small grammar, no model: a verb, a block of the plan, and a time, a length or another block.

    move | shift | push | pull | bring | delay   <block>  to <time> | by <length> | <length> later | earlier
    drop | remove | skip | cancel | delete       <block>
    add | schedule | put | plan                  <kind or event> [at <place>] [at <time>] [for <length>] [before | after <block>]
    swap | switch                                <block> and <block>
    shorten | cut | trim / lengthen | extend     <block> by <length> | to <length>
    <kind> <length> before | after <block>       "study two hours before judging"
    <block> at <time> / <block> for <length>     "lunch at one", "study for two hours"

Times: "one", "1", "1pm", "13:30", "half past two", "noon"; an hour with no AM or PM is read as the one nearest
the block it moves (else the next one in waking hours). Lengths: "two hours", "90 minutes", "an hour and a half".

Whatever it changes is run through the engine before it answers. When it cannot tell what was meant it changes
nothing, says what it did understand and offers the two nearest edits: never a silent no-op. When it could read
nothing at all ("banana") it offers no edit: it says so and gives two or three things that can be said, each about
a block of this very plan (`examples`).
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from contexts.play.dayrun import fixes, words
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Board, EditOption, EditResult, Move, OpenSlot, OptionalEvent, PlannedBlock, SimResult
from contexts.play.dayrun.routes import Walks

VERBS = {
    "swap": ["swap", "switch", "exchange", "flip"],
    "drop": ["drop", "remove", "skip", "cancel", "delete", "scrap", "lose", "ditch"],
    "shorten": ["shorten", "cut", "trim", "reduce", "shorter"],
    "lengthen": ["lengthen", "extend", "stretch", "longer"],
    "move": ["move", "shift", "push", "pull", "bring", "delay", "reschedule", "bump", "slide"],
    "add": ["add", "schedule", "put", "plan", "insert", "book", "include"],
}
KINDS = {
    "eat": ["eat", "meal", "food", "lunch", "dinner", "breakfast", "brunch", "snack", "supper"],
    "study": ["study", "studying", "work", "homework", "focus", "revise", "revision", "practice", "review"],
    "sleep": ["sleep", "bed", "nap", "bedtime"],
    "rest": ["rest", "break", "relax", "chill", "downtime"],
}
MEALS = {"breakfast": (5, 11), "brunch": (9, 13), "lunch": (11, 15), "dinner": (16.5, 22), "supper": (16.5, 22)}
DEFAULT_MINUTES = {"study": 90, "eat": 30, "rest": 30}
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fortyfive": 45, "fifty": 50,
           "sixty": 60, "ninety": 90}
FILLER = {"hey", "hi", "ok", "okay", "atlas", "please", "can", "could", "you", "the", "my", "a", "an", "to", "at", "in", "on",
          "of", "for", "by", "it", "me", "i", "want", "would", "like", "need", "lets", "let", "us", "just", "session", "block",
          "time", "some", "and", "with", "then", "from", "this", "that", "today", "tonight", "tomorrow", "instead", "so",
          "is", "be", "until", "till", "up", "back", "out", "make", "do", "have", "get", "go", "going", "s", "one"}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": -1}
STEP = 30  # "later" or "earlier" with no length
TIME = re.compile(r"\b(?:(noon|midday|midnight)|(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)\b")
LENGTH = re.compile(r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m)\b(?:\s*(?:and\s*)?(\d+)\s*(?:minutes?|mins?|min|m)\b)?")


@dataclass
class Thing:
    """Something in the plan a sentence can name: an accepted move, a move that did not fit, or a fixed block."""

    name: str
    tokens: set[str]
    kind: str
    start: datetime
    minutes: int
    move: int | None = None  # index into the moves; None for a fixed block
    block: PlannedBlock | None = None


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b([ap])\.m\.?", r"\1m", t)  # a.m. -> am, before the dots go
    t = re.sub(r"[^a-z0-9:.' ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"\bforty[ -]five\b", "45", t)
    t = re.sub(r"\ban? hour and a half\b|\bhour and a half\b|\bone and a half hours?\b", "90 minutes", t)
    t = re.sub(r"\bhalf an hour\b|\bhalf hour\b", "30 minutes", t)
    t = re.sub(r"\b(a )?quarter of an hour\b", "15 minutes", t)
    t = re.sub(r"\ba couple of hours\b", "2 hours", t)
    t = re.sub(r"\ban? hour\b", "1 hour", t)
    t = re.sub(r"\ba minute\b", "1 minute", t)
    t = re.sub(r"\bhalf past (\w+)\b", lambda m: f"{NUMBERS.get(m[1], m[1])}:30", t)
    t = re.sub(r"\bquarter past (\w+)\b", lambda m: f"{NUMBERS.get(m[1], m[1])}:15", t)
    t = re.sub(r"\b(" + "|".join(NUMBERS) + r")\b", lambda m: str(NUMBERS[m[1]]), t)
    t = re.sub(r"\b(\d{1,2}) (30|15|45)\b(?! ?(?:minutes?|mins?|min|m|hours?|h)\b)", r"\1:\2", t)  # "two thirty"
    t = re.sub(r"\b(\d+)\s*o'?clock\b", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens_of(text: str) -> set[str]:
    return {w[:-2] if w.endswith("'s") else w.strip("'") for w in re.findall(r"[a-z0-9']+", text.lower())} - {""}


def call(kind: str, start: datetime, place_name: str | None, title: str, cfg: DayRunConfig) -> str:
    """What a person would call a block: lunch, study at the Library, sleep."""
    place = fixes.place_word(place_name)
    if kind == "eat":
        hour = start.hour + start.minute / 60
        return "breakfast" if hour < 10.5 else "lunch" if hour < 15.5 else "dinner"
    if kind == "study":
        return f"{cfg.vocabulary.activity} at {place}" if place else cfg.vocabulary.activity
    if kind == "sleep":
        return "sleep"
    if kind == "rest":
        return "the rest"
    return words.short(title)


def things(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], result: SimResult) -> list[Thing]:
    out: list[Thing] = []

    def place_tokens(ref: str | None) -> set[str]:
        if not ref:
            return set()
        try:
            p = w.get(ref)
        except (ValueError, KeyError):
            return set()
        return tokens_of(" ".join([p.name, p.kind, *getattr(p, "aliases", [])]))

    for b in result.blocks:
        name = call(b.kind, b.start, b.place_name, b.title, cfg)
        toks = tokens_of(b.title) | tokens_of(name) | place_tokens(b.place) | set(KINDS.get(b.kind, []))
        if b.kind == "eat":  # a meal answers to the name of its time of day only
            hour = b.start.hour + b.start.minute / 60
            toks -= {m for m, (lo, hi) in MEALS.items() if not lo <= hour < hi}
        if b.fixed:
            toks -= set(KINDS["study"]) - tokens_of(b.title)
        out.append(Thing(name=name if not b.fixed else words.short(b.title), tokens=toks - FILLER, kind=b.kind, start=b.start,
                         minutes=b.minutes, move=b.move, block=b))
    for r in result.rejected:  # a move that did not fit can still be named, to drop or move it
        slot = next((s for s in board.slots if s.id == r.move.slot), None)
        start = r.move.start or (slot.start if slot else board.start)
        place = w.name(r.move.place) if r.move.place else None
        name = call(r.move.kind, start, place, fixes.what(r.move, board, w, cfg), cfg)
        toks = tokens_of(name) | place_tokens(r.move.place) | set(KINDS.get(r.move.kind, []))
        out.append(Thing(name=name, tokens=toks - FILLER, kind=r.move.kind, start=start, minutes=r.move.minutes or 0, move=r.index))
    return sorted(out, key=lambda t: t.start)


def best(candidates: list[Thing], said: set[str], only_moves: bool = False) -> tuple[Thing | None, list[Thing]]:
    """The thing the words name, and every thing ranked (best first) for when it is not clear."""
    scored = []
    for t in candidates:
        if only_moves and t.move is None:
            continue
        hits = said & t.tokens
        if hits:
            scored.append((len(hits) + 0.5 * sum(len(h) > 3 for h in hits) + (0.25 if t.move is not None else 0), t))
    scored.sort(key=lambda x: (-x[0], x[1].start))
    ranked = [t for _, t in scored]
    if not ranked:
        return None, []
    tied = sorted((t for s, t in scored if s == scored[0][0]), key=lambda t: t.start)
    if len(tied) > 1:  # several fit equally well: "the first", "the last" picks one; otherwise it is not guessed
        which = next((n for word, n in ORDINALS.items() if word in said), None)
        if which is None or which >= len(tied):
            return None, ranked
        return tied[which], ranked
    return ranked[0], ranked


def length_in(text: str) -> tuple[int | None, str]:
    m = LENGTH.search(text)
    if not m:
        return None, text
    n = float(m[1]) * (60 if m[2][0] == "h" else 1) + (int(m[3]) if m[3] else 0)
    return round(n), (text[:m.start()] + " " + text[m.end():]).strip()


def time_in(text: str) -> tuple[tuple[int, int, str | None] | None, str]:
    """The first clock time said: (hour, minute, "am" | "pm" | None), and the text without it."""
    for m in TIME.finditer(text):
        if m[1]:
            found = (0, 0, "am") if m[1] == "midnight" else (12, 0, "pm")
        else:
            hour, minute = int(m[2]), int(m[3] or 0)
            before = text[:m.start()].rstrip().rsplit(" ", 1)[-1]
            if not (m[3] or m[4] or before in ("to", "at", "by", "from", "until", "till", "around", "for")) or hour > 23 or minute > 59:
                continue
            if before == "for" and not (m[3] or m[4]):
                continue
            found = (hour, minute, m[4] if hour <= 12 else None)
        return found, (text[:m.start()] + " " + text[m.end():]).strip()
    return None, text


def when(board: Board, said: tuple[int, int, str | None], near: datetime | None) -> datetime | None:
    """The moment inside this board the clock time means."""
    hour, minute, half = said
    hours = [hour] if hour > 12 or (half is None and hour == 0) else (
        [hour % 12 + (12 if half == "pm" else 0)] if half else [hour % 12, hour % 12 + 12])
    options = []
    for day in range(3):
        for h in hours:
            t = (board.start + timedelta(days=day)).replace(hour=h, minute=minute, second=0, microsecond=0)
            if board.start <= t < board.end:
                options.append(t)
    if not options:
        return None
    if near is not None:
        return min(options, key=lambda t: abs((t - near).total_seconds()))
    awake = [t for t in options if 7 <= t.hour <= 23]
    return min(awake or options)


def slot_at(board: Board, t: datetime) -> OpenSlot | None:
    return next((s for s in board.slots if s.start <= t < s.end), None)


def starts_of(moves: list[Move], result: SimResult) -> list[datetime | None]:
    """When each move starts in `result` (None for one that did not fit), in the order of `moves`."""
    began = {b.move: b.start for b in result.blocks if b.move is not None}
    return [began.get(i) for i in range(len(moves))]


def placed(moves: list[Move], move: Move, board: Board, starts: list[datetime | None], at: datetime | None) -> tuple[list[Move], int]:
    """`move` put into its slot's run by time: before the first move of the slot that starts at or after `at`."""
    if at is None:
        return fixes.insert(moves, move, board)
    order = {s.id: n for n, s in enumerate(board.slots)}
    for i, m in enumerate(moves):
        began = starts[i] or m.start or board.end
        if order.get(m.slot, 99) > order.get(move.slot, 99) or (m.slot == move.slot and began >= at):
            return [*moves[:i], move, *moves[i:]], i
    return [*moves, move], len(moves)


class Editor:
    def __init__(self, cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move]) -> None:
        self.cfg, self.w, self.board, self.moves = cfg, w, board, list(moves)
        self.result = simulate(cfg, w, board, self.moves)
        self.things = things(cfg, w, board, self.moves, self.result)
        self.starts = starts_of(self.moves, self.result)
        self.note: str | None = None  # why an edit could not be made as asked

    def run(self, moves: list[Move]) -> SimResult:
        return simulate(self.cfg, self.w, self.board, moves)

    # --- answers ---

    def done(self, understood: str, moves: list[Move], changed: int | None) -> EditResult:
        after = self.run(moves)
        warning = None
        bad = next((r for r in after.rejected if r.index == changed), None) if changed is not None else None
        if bad:
            name = fixes.what(bad.move, self.board, self.w, self.cfg)
            warning = words.problem_of_rejected(bad, self.board, name[0].upper() + name[1:])
        elif len(after.rejected) > len(self.result.rejected):  # it fits, but something after it no longer does
            pushed = next((r for r in after.rejected if r.index != changed), None)
            if pushed:
                warning = f"That leaves no room for {fixes.what(pushed.move, self.board, self.w, self.cfg)}"
        return EditResult(ok=True, understood=understood, moves=moves, changed=changed, warning=warning)

    def unsure(self, understood: str, options: list[EditOption]) -> EditResult:
        seen, out = set(), []
        for o in [*options, *self.fallback()]:
            if o.label not in seen:
                seen.add(o.label)
                out.append(o)
        return EditResult(ok=False, understood=understood, moves=self.moves, options=out[:2])

    def lost(self, understood: str) -> EditResult:
        """Nothing in the words could be read: no edit is guessed at; what can be said here is shown instead."""
        return EditResult(ok=False, understood=understood, moves=self.moves, examples=self.examples())

    def examples(self) -> list[str]:
        """Up to three things to say about this plan, one each to move, drop and add. Each is checked against the
        same matching and placing `edit` uses, so none of them is a phrase it would not read."""
        mine = [t for t in self.things if t.move is not None and t.block is not None]
        awake = [t for t in mine if t.kind != "sleep"]
        out: list[str] = [] if mine else ["add lunch", "add study for 2 hours"]  # an empty day: what to start with
        used: Thing | None = None
        for t in sorted(awake, key=lambda t: (t.kind != "eat", t.start)):  # a meal first: "move lunch to 1 PM"
            if best(self.things, tokens_of(t.name) - FILLER)[0] is not t:
                continue  # two blocks answer to this name: not one to give as an example
            hour = (t.start + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            at = next((x for x in (hour, hour + timedelta(hours=1)) if not isinstance(self.move_to(t, x), str)), None)
            if at is not None:
                out.append(f"move {t.name} to {words.clock12(at)}")
                used = t
                break
        work = self.cfg.vocabulary.activity
        twins = [t for t in awake if t.kind == "study"]
        second = f"drop the second {work} block"
        if len(twins) > 1 and best(self.things, tokens_of(second) - FILLER - set(VERBS["drop"]))[0] is twins[1]:
            out.append(second)
        else:
            for t in reversed(awake):
                if t is not used and best(self.things, tokens_of(t.name) - FILLER)[0] is t:
                    out.append(f"drop {t.name}")
                    break
        offered = next((e for e in [*self.board.suggestions, *self.board.events] if not any(m.event == e.id for m in self.moves)), None)
        after = used or next(iter(awake), None)
        if offered is not None and self.add_option("event", None, None, None, None, offered) is not None:
            out.append(f"add {words.short(offered.title)}")
        elif after is not None and self.add_option("rest", None, None, None, ("after", after)) is not None:
            out.append(f"add a break after {after.name}")
        elif not mine:
            out.append("add sleep at 11")
        self.note = None  # trying an example must not leave its "only room for" note behind
        return out[:3]

    def named(self, t: Thing) -> str:
        """A thing's name, with its time when another thing in the plan goes by the same name."""
        twin = any(x is not t and x.name == t.name for x in self.things)
        return f"{t.name} at {words.clock12(t.start)}" if twin else t.name

    def fallback(self) -> list[EditOption]:
        """Edits that are always worth offering: what would mend the plan, else a meal and a block of work in a gap."""
        out = [EditOption(label=f.label, moves=f.moves, changed=f.changed)
               for f in fixes.suggest(self.cfg, self.w, self.board, self.moves, self.result, waits=False) if f.label and f.moves is not None]
        return out

    # --- edits ---

    def move_to(self, t: Thing, at: datetime) -> tuple[list[Move], int] | str:
        slot = slot_at(self.board, at)
        if slot is None:
            busy = next((b for b in self.board.blocks if b.start <= at < b.end), None)
            return f"{words.clock12(at)} is taken by {words.short(busy.title)}" if busy else f"{words.clock12(at)} is outside this day"
        old = self.moves[t.move]
        rest = [m for i, m in enumerate(self.moves) if i != t.move]
        moved = old.model_copy(update={"slot": slot.id, "start": at})
        out, index = placed(rest, moved, self.board, [x for i, x in enumerate(self.starts) if i != t.move], at)
        began = next((b.start for b in self.run(out).blocks if b.move == index), None)
        if began and began > at:  # the walk came out of the time asked for: leave that much earlier, when the slot allows
            earlier = max(slot.start, at - (began - at))
            out[index] = moved.model_copy(update={"start": earlier})
        return out, index

    def add_option(self, kind: str, place: str | None, at: datetime | None, minutes: int | None,
                   anchor: tuple[str, Thing] | None, event: OptionalEvent | None = None) -> EditOption | None:
        board, cfg = self.board, self.cfg
        if event is not None:
            slot = next((s for s in board.slots if s.id == event.slot), None) or next(
                (s for s in board.slots if event.start < s.end and event.end > s.start), None)
            if slot is None:
                return None
            move = Move(slot=slot.id, kind="event", event=event.id, minutes=minutes)
            out, index = placed(self.moves, move, board, self.starts, event.start)
            return EditOption(label=f"Add {words.short(event.title)}, {words.clock12(event.start)}", moves=out, changed=index)
        want = minutes or DEFAULT_MINUTES.get(kind)
        start: datetime | None = at
        if at is not None:  # a time in the middle of something already planned is not free
            busy = next((x for x in self.things if x.move is not None and x.start <= at < x.start + timedelta(minutes=x.minutes)), None)
            if busy:
                self.note = f"{words.clock12(at)} is in the middle of {busy.name}"
                start = at = busy.start + timedelta(minutes=busy.minutes)
        slots = list(board.slots)
        if anchor:
            side, thing = anchor
            if side == "before":
                slots = [s for s in board.slots if s.end <= thing.start and s.end >= thing.start - timedelta(minutes=1)] or [
                    s for s in board.slots if s.start < thing.start <= s.end]
                slots = slots[-1:]
            else:
                end = thing.start + timedelta(minutes=thing.minutes)
                slots = [s for s in board.slots if s.start <= end < s.end][:1]
                start = end if slots else None
        elif at is not None:
            slots = [s for s in board.slots if s.start <= at < s.end]
        free = at is None and not anchor  # no time was said: the first gap that holds all of it, else the first that holds some
        for slot in [(s, True) for s in slots if free] + [(s, False) for s in slots]:
            slot, whole = slot
            options = [o for o in slot.moves if o.kind == kind and (place is None or o.place == place)]
            if place and not options and kind in ("study", "eat", "rest"):
                continue
            feasible = sorted((o for o in options if o.feasible and not (whole and want and o.max_minutes < want)),
                              key=lambda o: (o.walk_minutes + o.back_minutes, o.place or ""))
            if not feasible:
                continue
            o = feasible[0]
            length = min(want, o.max_minutes) if want else None
            if length is not None and length < cfg.min_move_minutes:
                continue
            begin = start
            if anchor and anchor[0] == "before" and length:
                room = minutes_free = int((anchor[1].start - slot.start).total_seconds() // 60) - int(o.back_minutes + o.walk_minutes)
                if room < length:
                    self.note = f"There's only {words.dur(max(room, 0))} free before {anchor[1].name}"
                    length = room - room % 5
                    if length < cfg.min_move_minutes:
                        continue
                begin = max(slot.start, anchor[1].start - timedelta(minutes=length + o.back_minutes + o.walk_minutes))
                del minutes_free
            move = Move(slot=slot.id, kind=kind, place=o.place if kind != "rest" or place else None, minutes=length, start=begin)
            if begin is not None:
                tries = [placed(self.moves, move, board, self.starts, begin)]
            else:  # no time said: the earliest spot in this gap where it fits without pushing anything out
                run = [i for i, mv in enumerate(self.moves) if mv.slot == slot.id]
                spots = [run[0], *[i + 1 for i in run]] if run else [fixes.insert(self.moves, move, board)[1]]
                tries = [([*self.moves[:i], move, *self.moves[i:]], i) for i in spots]
            began, out, index = None, tries[0][0], tries[0][1]
            for cand, i in tries:
                after = self.run(cand)
                block = next((b for b in after.blocks if b.move == i), None)
                if block is not None and (begin is not None or len(after.rejected) <= len(self.result.rejected)):
                    began, out, index = block, cand, i
                    break
            if began is None and at is None and not anchor:
                continue  # it does not fit in this gap after all: try the next
            if began is None and length:  # a time or a block was named and the length asked for does not fit: the most that does
                for shorter in [n for n in (90, 60, 45, 30, 20, 15) if cfg.min_move_minutes <= n < length]:
                    small = move.model_copy(update={"minutes": shorter})
                    if anchor and anchor[0] == "before":
                        small.start = max(slot.start, anchor[1].start - timedelta(minutes=shorter + o.back_minutes + o.walk_minutes))
                    cand, i = placed(self.moves, small, board, self.starts, small.start)
                    after = self.run(cand)
                    block = next((b for b in after.blocks if b.move == i), None)
                    if block is not None and len(after.rejected) <= len(self.result.rejected):
                        began, out, index = block, cand, i
                        where = f"before {anchor[1].name}" if anchor else f"at {words.clock12(block.start)}"
                        self.note = f"There's only room for {words.dur(shorter)} {where}"
                        break
                if began is None:
                    return None
            name = call(kind, began.start if began else (begin or slot.start), o.place_name, o.title, cfg)
            told = f"Add {name}" + (f", {words.clock12(began.start)}" if began else "") + (f" for {words.dur(began.minutes)}" if began and kind != "eat" else "")
            return EditOption(label=told, moves=out, changed=index)
        return None

    def place_named(self, said: set[str]) -> str | None:
        """A place of the board's packs the words name ("the union", "the library")."""
        scored = []
        for slot in self.board.slots:
            for o in slot.moves:
                if not o.place:
                    continue
                try:
                    p = self.w.get(o.place)
                except (ValueError, KeyError):
                    continue
                toks = tokens_of(" ".join([p.name, *getattr(p, "aliases", [])])) - FILLER - {"at"}
                hits = said & toks
                if hits:
                    scored.append((len(hits), o.place))
        return max(scored)[1] if scored else None


def edit(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], text: str) -> EditResult:
    ed = Editor(cfg, w, board, moves)
    t = normalise(text)
    if not t:
        return ed.lost("I didn't catch anything to change.")
    all_words = t.split()
    verb = next((v for v, forms in VERBS.items() if any(f in all_words for f in forms)), None)
    later = any(x in all_words for x in ("later", "back")) or (verb == "move" and any(x in all_words for x in ("push", "delay")))
    earlier = any(x in all_words for x in ("earlier", "sooner", "forward", "ahead")) or (verb == "move" and "pull" in all_words)
    if "earlier" in all_words or "sooner" in all_words:
        later = False
    for forms in VERBS.values():
        t = " ".join(x for x in t.split() if x not in forms)
    t = " ".join(x for x in t.split() if x not in ("later", "earlier", "sooner"))

    # "... before judging" / "... after lunch": the block named after the word
    anchor: tuple[str, Thing] | None = None
    m = re.search(r"\b(before|after)\b(.*)$", t)
    if m:
        named, _ = best(ed.things, tokens_of(m[2]) - FILLER)
        if named:
            anchor, t = (m[1], named), t[:m.start()].strip()
    minutes, t = length_in(t)
    clock, t = time_in(t)
    said = tokens_of(t) - FILLER
    kind = next((k for k, forms in KINDS.items() if said & set(forms)), None)

    # --- swap ---
    if verb == "swap":
        halves = re.split(r"\b(?:and|with|for)\b", t, maxsplit=1)
        a, _ = best(ed.things, tokens_of(halves[0]) - FILLER, only_moves=True)
        b, _ = best([x for x in ed.things if x is not a], tokens_of(halves[1]) - FILLER, only_moves=True) if len(halves) > 1 else (None, [])
        if a is None or b is None or a.move is None or b.move is None:
            return ed.unsure("I heard “swap”, but I need two blocks you can move.", [])
        out = list(ed.moves)
        ma, mb = out[a.move], out[b.move]
        out[a.move] = mb.model_copy(update={"slot": ma.slot, "start": ma.start})
        out[b.move] = ma.model_copy(update={"slot": mb.slot, "start": mb.start})
        return ed.done(f"Swapped {a.name} and {b.name}.", out, a.move)

    target, ranked = best(ed.things, said)
    movable = [x for x in ranked if x.move is not None]

    # --- add ---
    event = None
    if verb in ("add", None):
        pool = [e for e in [*board.suggestions, *board.events] if not any(mv.event == e.id for mv in moves)]
        scored = sorted(((len(said & (tokens_of(e.title) - FILLER)), e) for e in pool), key=lambda x: -x[0])
        if scored and scored[0][0] > 0 and (verb == "add" or target is None):
            event = scored[0][1]
    implicit_add = verb is None and kind is not None and (anchor is not None or (target is None and not movable))
    if verb == "add" or implicit_add or (verb is None and event is not None):
        if event is not None:
            option = ed.add_option("event", None, None, None, None, event)
            if option:
                there = next((b.start for b in ed.run(option.moves).blocks if b.move == option.changed), None)
                late = there is not None and (there - event.start).total_seconds() > 300  # what is before it runs into it
                told = f"; you'd get there at {words.clock12(there)}" if late else f" at {words.clock12(event.start)}"
                return ed.done(f"Added {words.short(event.title)}{told}.", option.moves, option.changed)
            return ed.unsure(f"{words.short(event.title)} doesn't fit around what's fixed that day.", [])
        if kind is None:
            return ed.unsure("I heard “add”, but not what: a meal, study, a rest or sleep?", [])
        usual = board.sleep_window[0] if kind == "sleep" and board.sleep_window else None  # "sleep at 11" is the evening one
        at = when(board, clock, anchor[1].start if anchor else usual) if clock else None
        if clock and at is None:
            return ed.unsure("That time isn't in this day.", [])
        meal = next((m for m in MEALS if m in said), None) if kind == "eat" else None
        if meal and at is None and not anchor:  # "add dinner": in the first gap of that meal's own hours
            lo, hi = MEALS[meal]
            for slot in board.slots:
                for day in range(2):
                    a = (board.start + timedelta(days=day)).replace(hour=int(lo), minute=int(lo % 1 * 60), second=0, microsecond=0) + timedelta(hours=1)
                    b = a + timedelta(hours=hi - lo - 1)
                    if at is None and slot.start < b and slot.end > a:
                        at = max(slot.start, a)
        if "nap" in said and not minutes:
            minutes = 30
        option = ed.add_option(kind, ed.place_named(said - set(KINDS[kind])), at, minutes, anchor)
        if option is not None and ed.note:
            return ed.unsure(f"{ed.note}.", [option])
        if option is None:
            where = f" {anchor[0]} {anchor[1].name}" if anchor else f" at {words.clock12(at)}" if at else ""
            ed.note = None
            near = ed.add_option(kind, None, None, minutes, None)
            other = ed.add_option(kind, None, None, 30 if kind != "sleep" else None, None) if minutes and minutes > 30 else None
            return ed.unsure(f"There's no room to add {kind if kind != 'eat' else 'a meal'}{where}.", [o for o in (near, other) if o])
        return ed.done(option.label.replace("Add ", "Added ", 1) + ".", option.moves, option.changed)

    # --- everything else names a block already in the plan ---
    if target is None:
        if verb is None:  # no verb and no block of the plan: nothing was read, so no edit is offered
            return ed.lost("I couldn't tell what to change.")
        heard = f"I heard “{verb}”"
        options: list[EditOption] = []
        for x in (movable or [x for x in ed.things if x.move is not None])[:2]:
            if verb == "drop":
                options.append(EditOption(label=f"Drop {ed.named(x)}", moves=[mv for i, mv in enumerate(ed.moves) if i != x.move]))
            elif verb in ("move", None):
                got = ed.move_to(x, x.start + timedelta(minutes=STEP))
                if not isinstance(got, str):
                    options.append(EditOption(label=f"Move {x.name} to {words.clock12(x.start + timedelta(minutes=STEP))}", moves=got[0], changed=got[1]))
            elif verb in ("shorten", "lengthen") and x.minutes:
                new = max(cfg.min_move_minutes, x.minutes + (15 if verb == "lengthen" else -15))
                options.append(EditOption(label=f"Make {ed.named(x)} {words.dur(new)}", changed=x.move,
                                          moves=[mv.model_copy(update={"minutes": new}) if i == x.move else mv for i, mv in enumerate(ed.moves)]))
        which = ", but not which block" if verb else ""
        return ed.unsure(f"{heard}{which}. Did you mean one of these?", options)
    if target.move is None:
        nearby = [x for x in ed.things if x.move is not None]
        nearby.sort(key=lambda x: abs((x.start - target.start).total_seconds()))
        options = [EditOption(label=f"Drop {x.name}", moves=[mv for i, mv in enumerate(ed.moves) if i != x.move]) for x in nearby[:2]]
        return ed.unsure(f"{target.name} is on your calendar, so Atlas plans around it.", options)

    old = ed.moves[target.move]
    if verb == "drop":
        return ed.done(f"Dropped {target.name}.", [mv for i, mv in enumerate(ed.moves) if i != target.move], None)

    if verb in ("shorten", "lengthen") or (verb is None and minutes and not clock and not later and not earlier):
        by_or_to = "to" if re.search(r"\bto\b", normalise(text)) or verb is None else "by"
        step = minutes or 15
        new = step if by_or_to == "to" and minutes else target.minutes + (step if verb == "lengthen" else -step)
        if new < cfg.min_move_minutes:
            return ed.unsure(f"{target.name} can't be shorter than {cfg.min_move_minutes} minutes.",
                             [EditOption(label=f"Drop {target.name}", moves=[mv for i, mv in enumerate(ed.moves) if i != target.move])])
        out = [mv.model_copy(update={"minutes": new}) if i == target.move else mv for i, mv in enumerate(ed.moves)]
        return ed.done(f"{target.name[0].upper()}{target.name[1:]} is now {words.dur(new)}.", out, target.move)

    # --- move ---
    at: datetime | None = None
    if clock:
        at = when(board, clock, target.start)
        if at is None:
            return ed.unsure("That time isn't in this day.", [])
    elif later or earlier or (verb == "move" and minutes):
        at = target.start + timedelta(minutes=(minutes or STEP) * (-1 if earlier else 1))
    elif anchor:
        thing = anchor[1]
        at = thing.start - timedelta(minutes=target.minutes or STEP) if anchor[0] == "before" else thing.start + timedelta(minutes=thing.minutes)
    if at is None:
        options = []
        for delta in (STEP, -STEP):
            got = ed.move_to(target, target.start + timedelta(minutes=delta))
            if not isinstance(got, str):
                options.append(EditOption(label=f"Move {target.name} to {words.clock12(target.start + timedelta(minutes=delta))}", moves=got[0], changed=got[1]))
        return ed.unsure(f"I can move {target.name}, but I didn't catch when.", options)
    got = ed.move_to(target, at)
    if not isinstance(got, str):
        block = next((b for b in ed.run(got[0]).blocks if b.move == got[1]), None)
        if block is not None and abs((block.start - at).total_seconds()) > 600:  # it had to queue behind something
            prev = max((b for b in ed.run(got[0]).blocks if b.end <= block.start), key=lambda b: b.end, default=None)
            after = f", after {call(prev.kind, prev.start, prev.place_name, prev.title, cfg)}" if prev else ""
            return ed.done(f"{target.name[0].upper()}{target.name[1:]} now starts at {words.clock12(block.start)}{after}.", got[0], got[1])
    if isinstance(got, str):
        options = []  # the nearest free times on either side of what is in the way
        after_it = min((s for s in board.slots if s.start >= at), key=lambda s: s.start, default=None)
        before_it = max((s for s in board.slots if s.end <= at), key=lambda s: s.end, default=None)
        sides = ([after_it.start] if after_it else []) + (
            [max(before_it.start, before_it.end - timedelta(minutes=(target.minutes or STEP) + 15))] if before_it else [])
        for t_alt in sides:
            alt = ed.move_to(target, t_alt)
            if not isinstance(alt, str):
                options.append(EditOption(label=f"Move {target.name} to {words.clock12(t_alt)}", moves=alt[0], changed=alt[1]))
        return ed.unsure(f"{got}, so {target.name} can't go there.", options)
    return ed.done(f"Moved {target.name} to {words.clock12(at)}.", got[0], got[1])

