"""`python -m contexts.play.dayrun demo`: the scripted Sunday, printed.

The board, a bad plan and why it fails, a good plan with three stars and its walking legs,
then hard mode: two disruptions and the re-route. Runs on fixtures/play/demo.yaml with the
shipped place packs and config/dayrun.yaml; nothing is read from the clock or the network.
"""

from pathlib import Path

from contexts.play.dayrun.board import hm
from contexts.play.dayrun.models import Board, Move, SimResult
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet

BARS = " .:-=+*#%@"


def strip(values: list[float]) -> str:
    return "".join(BARS[min(9, int(v / 100 * 10))] for v in values)


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show_board(b: Board) -> None:
    i = b.initial
    print(f"{b.start:%A %b %d, %H:%M} to {b.end:%A %H:%M}, starting at {b.start_place}")
    print(f"Start: energy {i.energy:g}/100, focus {i.focus:g}/100. Last night {i.sleep_hours:g} h of sleep "
          f"(debt {i.sleep_debt:g} h), {i.meals} meals so far today. [{i.source}]")
    print("\nFixed blocks")
    for f in b.blocks:
        print(f"  {hm(f.start)}-{hm(f.end)}  {f.title}  @ {f.place_name or 'anywhere'}")
    print("Deadlines")
    for d in b.deadlines:
        print(f"  {d.title}: due {d.due:%a %H:%M}, {d.work_left:g} focused minutes of work left, risk {d.risk:.2f}")
    print("Friends")
    for f in b.friends:
        print(f"  {f.member} at {f.place_name}, {hm(f.start)}-{hm(f.end)}")
    print("Optional events")
    for e in b.events:
        print(f"  {hm(e.start)}-{hm(e.end)}  {e.title}  @ {e.place_name}")
    print("Open slots, and what each can hold (walk there / walk on to the next block / minutes left)")
    for s in b.slots:
        print(f"  {s.id}  {hm(s.start)}-{hm(s.end)}  {s.minutes} min, from {s.from_place}" + (f", then to {s.to_place}" if s.to_place else ""))
        for o in s.moves:
            who = f"  with {', '.join(o.friends)}" if o.friends else ""
            fits = f"{o.walk_minutes:g} / {o.back_minutes:g} / {o.max_minutes}" if o.feasible else f"NO: {o.reason}"
            print(f"        {o.title:<44} {fits}{who}")


def show_plan(moves: list[Move]) -> None:
    for n, m in enumerate(moves):
        bits = [m.kind, m.place or "", f"{m.minutes} min" if m.minutes else "fill the slot", f"with {m.with_friend}" if m.with_friend else "",
                f"from {hm(m.start)}" if m.start else ""]
        print(f"  {n}. [{m.slot}] " + ", ".join(b for b in bits if b))


def show_result(r: SimResult) -> None:
    print("\nWhat happens")
    for b in r.blocks:
        print(f"  {hm(b.start)}-{hm(b.end)}  {b.title}{'' if b.fixed else f'  @ {b.place_name}'}{'  (fixed)' if b.fixed else ''}")
    for x in r.rejected:
        print(f"  REJECTED move {x.index} ({x.move.kind} at {x.move.place}): {x.reason}")
    for note in r.notes:
        print(f"  note: {note}")
    hours = [t for t in r.timeline if t.t.minute == 0][::2]
    print("\nMeters (one character per 15 minutes; ' ' = 0, '@' = 100)")
    print(f"  time    {hm(r.start)}{' ' * (len(r.timeline) - 10)}{hm(r.end)}")
    print(f"  energy  {strip([t.energy for t in r.timeline])}")
    print(f"  focus   {strip([t.focus for t in r.timeline])}")
    print(f"  risk    {strip([t.deadline_risk * 100 for t in r.timeline])}")
    print("  " + "  ".join(f"{hm(t.t)} e{t.energy:.0f} f{t.focus:.0f} r{t.deadline_risk:.2f}" for t in hours[:7]))
    print("  " + "  ".join(f"{hm(t.t)} e{t.energy:.0f} f{t.focus:.0f} r{t.deadline_risk:.2f}" for t in hours[7:]))
    t = r.totals
    print(f"\nTotals: {t.study_minutes} study minutes -> {t.focused_minutes:g} focused minutes of work, {t.sleep_hours:g} h sleep, "
          f"{t.meals} meals, {t.walk_minutes:g} min walking, energy low {t.energy_min:g}, end {t.energy_end:g}, {t.points} points")
    for c in r.outcome.checks:
        print(f"  [{'ok' if c.ok else 'FAIL'}] {c.text}")
    print(f"OUTCOME: {'WIN' if r.outcome.won else 'FAIL'}  {'*' * r.stars or 'no stars'}  {r.outcome.summary}")


def show_legs(r: SimResult, service: DayRun) -> None:
    print("\nRoute legs (A* over the place pack)")
    for route in r.routes:
        print(f"  {hm(route.depart)}  {route.purpose}: {route.minutes:g} min")
        for leg in (x for x in r.legs if x.route == route.index):
            print(f"           {leg.from_name} -> {leg.to_name}: {leg.minutes:g} min")


def main(scenario: str = "demo") -> None:
    service = DayRun(PlaceSet.load("config/places"), config_dir="config", data_dir="data", fixtures_dir=Path("fixtures/play"))
    sc = service.scenario(scenario)

    heading("1. THE BOARD: the next 24 hours")
    show_board(service.board(scenario=scenario))

    heading("2. A BAD PLAN: skip the food, cram alone all night")
    show_plan(sc.plans["bad"])
    show_result(service.simulate(sc.plans["bad"], scenario=scenario))

    heading("3. A GOOD PLAN: eat, study at the library with Maya, sleep")
    show_plan(sc.plans["good"])
    good = service.simulate(sc.plans["good"], scenario=scenario)
    show_result(good)
    show_legs(good, service)

    heading("4. HARD MODE: the same good plan, disrupted")
    h = service.hard(sc.plans["good"], scenario=scenario, disruptions=sc.disruptions)
    for d in h.disruptions:
        print(f"  DISRUPTION: {d.text}")
    print("\nThe engine re-validates every move and re-routes:")
    for c in h.changes:
        print(f"  [{c.kind}] {c.text}")
    b, a = h.before, h.after
    print(f"\nBefore: {'WIN' if b.outcome.won else 'FAIL'} {'*' * b.stars}, {b.totals.walk_minutes:g} min walking, {b.totals.meals} meals, {b.totals.points} points")
    print(f"After:  {'WIN' if a.outcome.won else 'FAIL'} {'*' * a.stars}, {a.totals.walk_minutes:g} min walking, {a.totals.meals} meals, {a.totals.points} points")
    print(f"        {a.outcome.summary}")
    seeded = service.hard(sc.plans["good"], scenario=scenario, seed=7, count=2)
    print("\nSeeded (seed 7, the same every run): " + "; ".join(d.text for d in seeded.disruptions))
