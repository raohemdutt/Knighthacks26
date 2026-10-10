"""Reality score and forecast calibration: the plan against what Atlas measured.

The intended check is face presence at the desk: was a face at the desk during each study
block? That signal is not built yet. Until it is, this uses what the habit tracker already
measures for the day: desk time (the Focus habit), meals seen and sleep (the empty-desk
window). Those are day totals, so they are dealt out to the planned blocks in time order.
When a per-block measurement exists (`Measured.blocks`, where face presence will plug in),
it wins. A study block away from the desk cannot be measured by anything yet and is left
out of the score rather than counted as missed.
"""

from datetime import date

from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.models import BlockScore, Estimate, ForecastResult, ForecastRow, Gap, Measured, PlannedBlock, Review, Totals

MEASURED_BY = ("the habit tracker's desk time, meals and sleep for the day, dealt out to the blocks in time order "
               "(face presence at the desk is the intended check and is not built yet)")


def reality(cfg: DayRunConfig, desk: str, day: date, blocks: list[PlannedBlock], predicted: Totals, measured: Measured) -> Review:
    pools = {
        "study": measured.focus_minutes,
        "eat": measured.meals,
        "sleep": measured.sleep_hours * 60 if measured.sleep_hours is not None else None,
    }
    rows: list[BlockScore] = []
    for b in sorted((b for b in blocks if not b.fixed and b.kind in pools), key=lambda b: b.start):
        row = BlockScore(id=b.id, title=b.title, kind=b.kind, planned_minutes=b.minutes)
        if b.id in measured.blocks:
            row.measured_minutes = measured.blocks[b.id]
            row.note = "measured for this block"
        elif b.kind == "study" and b.place != desk:
            row.note = f"at {b.place_name}, away from the box: nothing measures it yet"
        elif pools[b.kind] is None:
            row.note = "no measurement for this day"
        elif b.kind == "eat":
            row.measured_minutes = b.minutes if pools["eat"] >= 1 else 0.0
            pools["eat"] -= 1
            row.note = "a meal was seen" if row.measured_minutes else "no meal was seen for this one"
        else:
            row.measured_minutes = round(min(b.minutes, max(pools[b.kind], 0.0)), 1)
            pools[b.kind] -= row.measured_minutes
            row.note = "from the day's desk time" if b.kind == "study" else "from the night's empty-desk window"
        if row.measured_minutes is not None:
            row.followed = round(min(1.0, row.measured_minutes / b.minutes), 2) if b.minutes else 1.0
        rows.append(row)

    weights = {"study": cfg.review.weights.get("study", 0), "eat": cfg.review.weights.get("meals", 0), "sleep": cfg.review.weights.get("sleep", 0)}
    total = share = 0.0
    gaps: list[tuple[float, BlockScore]] = []
    for kind, weight in weights.items():
        scored = [r for r in rows if r.kind == kind and r.followed is not None]
        minutes = sum(r.planned_minutes for r in scored)
        if not scored or not minutes:
            continue
        total += weight
        share += weight * sum(r.followed * r.planned_minutes for r in scored) / minutes
        gaps += [(weight * (1 - r.followed) * r.planned_minutes / minutes, r) for r in scored]
    score = round(100 * share / total) if total else None

    gap = None
    worst = max(gaps, key=lambda g: g[0], default=None)
    if worst and worst[0] > 0:
        r = worst[1]
        start = next(b.start for b in blocks if b.id == r.id)
        what = "no meal was seen" if r.kind == "eat" else f"{r.measured_minutes:g} of {r.planned_minutes:g} planned minutes happened"
        gap = Gap(block=r.id, kind=r.kind, planned=r.planned_minutes, measured=r.measured_minutes or 0.0,
                  text=f"{r.title} at {start:%H:%M}: {what}")

    planned = {"study_minutes": float(sum(r.planned_minutes for r in rows if r.kind == "study")),
               "meals": float(sum(1 for r in rows if r.kind == "eat")),
               "sleep_hours": round(sum(r.planned_minutes for r in rows if r.kind == "sleep") / 60, 2)}

    def seen(value: float | None, unit: str) -> str:
        return "not measured" if value is None else f"{value:g} {unit}"

    unmeasured = sum(1 for r in rows if r.followed is None)
    text = (
        f"Simulation: {predicted.sleep_hours:g} h of sleep, {predicted.meals} meals, {planned['study_minutes']:g} {cfg.vocabulary.activity} minutes planned. "
        f"Reality: {seen(measured.sleep_hours, 'h of sleep')}, {seen(measured.meals, 'meals')}, "
        f"{seen(measured.focus_minutes, f'minutes at the {cfg.vocabulary.workplace}')}. "
        + (f"You followed {score}% of what could be measured. " if score is not None else "Nothing in this plan could be measured. ")
        + (f"Biggest gap: {gap.text}. " if gap else "No gap: every measured block happened. " if score is not None else "")
        + (f"{unmeasured} block(s) could not be measured." if unmeasured else "")
    ).strip()
    return Review(day=day, score=score, blocks=rows, planned=planned, measured=measured, biggest_gap=gap, text=text, measured_by=MEASURED_BY)


def forecast(estimates: list[Estimate], measured: dict[str, float]) -> ForecastResult:
    """How well the player's own time estimates matched the measured minutes, block by block."""
    rows = [ForecastRow(block=e.block, estimated=e.minutes, measured=measured[e.block],
                        error=round(abs(e.minutes - measured[e.block]) / max(e.minutes, measured[e.block], 1e-9), 3),
                        off_by=round(e.minutes - measured[e.block], 1))
            for e in estimates if e.block in measured]
    unmeasured = [e.block for e in estimates if e.block not in measured]
    if not rows:
        return ForecastResult(score=0, bias=0.0, multiplier=1.0, rows=[], unmeasured=unmeasured,
                              text="None of these blocks has a measurement yet, so there is nothing to score.")
    est, real = sum(r.estimated for r in rows), sum(r.measured for r in rows)
    score = round(100 * (1 - sum(r.error for r in rows) / len(rows)))
    bias = round((est - real) / real, 3) if real else 0.0
    multiplier = round(real / est, 2) if est else 1.0
    worst = max(rows, key=lambda r: r.error)
    lean = "spot on overall" if abs(bias) < 0.05 else f"you {'over' if bias > 0 else 'under'}estimate by {abs(bias):.0%}"
    text = (f"Calibration {score}/100: {lean}. Furthest off: {worst.block}, estimated {worst.estimated:g} minutes, "
            f"took {worst.measured:g}. Multiply your estimates by {multiplier:g} next time.")
    return ForecastResult(score=score, bias=bias, multiplier=multiplier, rows=rows, unmeasured=unmeasured, text=text)
