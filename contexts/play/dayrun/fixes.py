"""What needs fixing in a plan, and the one edit that would mend each thing (POST /api/dayrun/explain).

    fixes = suggest(cfg, walks, board, moves, result)

A plan that does not work is a plan that needs fixing, not a failure: each problem is one plain sentence
(contexts/play/dayrun/words.py) and one suggested edit, as the whole list of moves with the edit applied, so the
phone applies it in one tap and re-simulates. Every suggestion is tried through the engine first: `mends` says
whether it makes that problem go away. Pure: no clock, no I/O, never a model.
"""

from datetime import datetime

from contexts.play.dayrun import words
from contexts.play.dayrun.board import minutes_between
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Board, Fix, Move, MoveOption, OpenSlot, SimResult
from contexts.play.dayrun.routes import Walks

MEAL_MINUTES, STUDY_MINUTES = 30, 120
MAX_TRIES = 8  # candidates simulated per problem
Try = tuple[str, list[Move], int | None, bool]  # label, the plan with the edit, the changed move, say its start time


def insert(moves: list[Move], move: Move, board: Board, first: bool = False) -> tuple[list[Move], int]:
    """`move` put at the end (or the start) of its slot's run, slots kept in board order; and its index."""
    order = {s.id: n for n, s in enumerate(board.slots)}
    rank = order.get(move.slot, len(order))
    at = len(moves)
    for i, m in enumerate(moves):
        r = order.get(m.slot, len(order))
        if r > rank or (first and r == rank):
            at = i
            break
    return [*moves[:at], move, *moves[at:]], at


def place_word(name: str | None) -> str:
    return (name or "").split(" (")[0]


def what(move: Move, board: Board, w: Walks, cfg: DayRunConfig) -> str:
    """A move in a few words, for a label: "the meal at '63 South", "study at the Library", "sleep"."""
    place = place_word(w.name(move.place)) if move.place else None
    if move.kind == "eat":
        return f"the meal at {place}" if place else "the meal"
    if move.kind == "study":
        return f"{cfg.vocabulary.activity} at {place}" if place else cfg.vocabulary.activity
    if move.kind == "event":
        e = next((e for e in [*board.events, *board.suggestions] if e.id == move.event), None)
        return words.short(e.title) if e else "the event"
    return "sleep" if move.kind == "sleep" else f"the rest at {place}" if place else "the rest"


def _check(result: SimResult, check_id: str):
    return next((c for c in result.outcome.checks if c.id == check_id), None)


def _started(result: SimResult, index: int) -> datetime | None:
    return next((b.start for b in result.blocks if b.move == index), None)


def _worse(before: SimResult, after: SimResult, but: str) -> bool:
    """True when the edit breaks something that held: a new move that does not fit, or another check that now fails."""
    if len(after.rejected) > len(before.rejected):
        return True
    was = {c.id: c for c in before.outcome.checks}
    for c in after.outcome.checks:
        old = was.get(c.id)
        if c.id == but or old is None:
            continue
        if old.ok and not c.ok:
            return True
        if not old.ok and (c.value > old.value + 0.02 if c.status else c.value < old.value - 0.05):  # a failing check made worse
            return True
    return False


def _options(slot: OpenSlot, kind: str) -> list[MoveOption]:
    return sorted((o for o in slot.moves if o.kind == kind and o.feasible), key=lambda o: (o.walk_minutes + o.back_minutes, o.place or ""))


def _swap(moves: list[Move], index: int, **update) -> list[Move]:
    return [m.model_copy(update=update) if i == index else m for i, m in enumerate(moves)]


def suggest(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], result: SimResult, limit: int = 6) -> list[Fix]:
    out: list[Fix] = []

    def run(candidate: list[Move]) -> SimResult:
        return simulate(cfg, w, board, candidate)

    # --- a move that does not fit ---
    for r in result.rejected:
        name = what(r.move, board, w, cfg)
        fix = Fix(id=f"move:{r.index}", problem=words.problem_of_rejected(r, board, name[0].upper() + name[1:]))
        tries: list[tuple[str, list[Move], int | None]] = []
        if r.minutes and r.move.minutes and r.move.minutes - r.minutes >= cfg.min_move_minutes:
            shorter = r.move.minutes - r.minutes
            shorter = max(cfg.min_move_minutes, shorter - shorter % 5)  # a round length a person would pick
            tries.append((f"Shorten {name} to {words.dur(shorter)}", _swap(moves, r.index, minutes=shorter), r.index))
        slot = next((s for s in board.slots if s.id == r.move.slot), None)
        if slot and r.move.kind in ("eat", "study"):
            for o in _options(slot, r.move.kind)[:2]:
                if o.place == r.move.place:
                    continue
                minutes = min(r.move.minutes, o.max_minutes) if r.move.minutes else None
                verb = "Eat" if r.move.kind == "eat" else cfg.vocabulary.activity.capitalize()
                tries.append((f"{verb} at {place_word(o.place_name)} instead", _swap(moves, r.index, place=o.place, minutes=minutes), r.index))
        tries.append((f"Drop {name}", [m for i, m in enumerate(moves) if i != r.index], None))
        for label, candidate, changed in tries:
            after = run(candidate)
            fits = changed is None or all(x.index != changed for x in after.rejected)
            if fits and len(after.rejected) < len(result.rejected):
                fix = fix.model_copy(update={"label": label, "moves": candidate, "changed": changed, "mends": True})
                break
        out.append(fix)

    # --- a check that fails ---
    for c in result.outcome.checks:
        if c.ok:
            continue
        fix = Fix(id=c.id, problem=words.problem_of_check(c, board))
        best: tuple[float, str, list[Move], int | None, bool] | None = None
        for label, candidate, changed, timed in _candidates(cfg, w, board, moves, result, c.id)[:MAX_TRIES]:
            after = run(candidate)
            now = _check(after, c.id)
            if now is None or _worse(result, after, c.id):
                continue
            gain = (c.value - now.value) if c.status else (now.value - c.value)  # a deadline's value is its risk: lower is better
            if gain <= 0 and not now.ok:
                continue
            score = gain + (1000 if now.ok else 0)
            if best is None or score > best[0]:
                at = _started(after, changed) if timed and changed is not None else None
                best = (score, label + (f", {words.clock12(at)}" if at else ""), candidate, changed, now.ok)
            if now.ok:
                break
        if best:
            fix = fix.model_copy(update={"label": best[1], "moves": best[2], "changed": best[3], "mends": best[4]})
        out.append(fix)
    return out[:limit]


def _candidates(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], result: SimResult, check_id: str) -> list[Try]:
    out: list[Try] = []
    if check_id == "meals":
        for slot in board.slots:
            for o in _options(slot, "eat")[:1]:
                if o.max_minutes < cfg.meal_min_minutes:
                    continue
                move = Move(slot=slot.id, kind="eat", place=o.place, minutes=min(MEAL_MINUTES, o.max_minutes))
                for first in (True, False):  # before what the slot already holds, else after it
                    candidate, at = insert(moves, move, board, first=first)
                    out.append((f"Add a meal at {place_word(o.place_name)}", candidate, at, True))
        return out
    if check_id == "sleep":
        night = board.sleep_window
        for i, m in enumerate(moves):  # a sleep that is there but short: let it run to the end of its slot
            if m.kind == "sleep" and m.minutes is not None:
                out.append(("Sleep until the morning", _swap(moves, i, minutes=None), i, False))
        for i, m in enumerate(moves):  # a long move late in the day is what pushes the night out: end it at bedtime
            block = next((b for b in result.blocks if b.move == i), None)
            if night and block and m.kind not in ("sleep", "event") and block.start < night[0] < block.end:
                keep = minutes_between(block.start, night[0])
                if keep < cfg.min_move_minutes:
                    continue
                cut = _swap(moves, i, minutes=keep)
                if any(x.kind == "sleep" and x.slot == m.slot for x in cut):
                    cut = [x.model_copy(update={"start": night[0], "minutes": None}) if x.kind == "sleep" and x.slot == m.slot else x for x in cut]
                else:
                    cut, _ = insert(cut, Move(slot=m.slot, kind="sleep", start=night[0]), board)
                out.append((f"Stop {what(m, board, w, cfg)} at {words.clock12(night[0])} and sleep", cut, i, False))
        if night:
            for slot in board.slots:
                if slot.start < night[1] and slot.end > night[0] and not any(m.kind == "sleep" and m.slot == slot.id for m in moves):
                    start = max(slot.start, night[0])
                    candidate, at = insert(moves, Move(slot=slot.id, kind="sleep", start=start), board)
                    out.append((f"Add sleep from {words.clock12(start)}", candidate, at, False))
        if board.slots and not any(m.kind == "sleep" for m in moves):
            candidate, at = insert(moves, Move(slot=board.slots[-1].id, kind="sleep"), board)
            out.append(("Add sleep at the end of the day", candidate, at, False))
        return out
    deadline = next((d for d in board.deadlines if d.id == check_id), None)
    if deadline is None:
        return out
    ranked: list[tuple[int, OpenSlot, MoveOption]] = []
    for slot in board.slots:
        if slot.start >= deadline.due:
            continue
        def good(o: MoveOption) -> float:  # how much work a minute there turns into
            return cfg.study_place_factor.get(w.kind(o.place), cfg.study_place_factor.get("default", 0.7)) if o.place else 0.7

        opts = sorted(_options(slot, "study"), key=lambda o: (-min(o.max_minutes, STUDY_MINUTES) * good(o), o.walk_minutes + o.back_minutes))
        ranked += [(min(o.max_minutes, STUDY_MINUTES), slot, o) for o in opts[:2] if o.max_minutes >= cfg.min_move_minutes]
    for minutes, slot, o in sorted(ranked, key=lambda x: (-x[0], x[1].start)):
        for first in (False, True):
            move = Move(slot=slot.id, kind="study", place=o.place, minutes=minutes, deadline=deadline.id)
            candidate, at = insert(moves, move, board, first=first)
            out.append((f"Add {words.dur(minutes)} of {cfg.vocabulary.activity} at {place_word(o.place_name)}", candidate, at, True))
    return out


def explain(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], result: SimResult) -> dict:
    """The fixes, and one true sentence for each number a person might question."""
    fixes = suggest(cfg, w, board, moves, result)
    checks = {c.id: c for c in result.outcome.checks}
    return {
        "headline": words.headline(len(fixes)),
        "fixes": fixes,
        "deadlines": {d.id: {"title": words.assignment(d.title), "due": words.day_clock(d.due), "work": words.why_work(d),
                             "status": checks[d.id].status if d.id in checks else None,
                             "risk": words.why_risk(checks[d.id], board) if d.id in checks else None} for d in board.deadlines},
        "sleep": words.why_sleep(result, board), "meals": words.why_meals(result, board),
        "points": words.why_points(result), "stars": words.why_stars(result),
        "walks": {str(r.index): words.why_walk(r, board) for r in result.routes},
    }
