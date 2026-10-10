"""Patterns from the habit history: simple, honest statements with their counts, nothing fitted.

    patterns(habits) -> ["On the 9 nights under 6 h, next-day desk time averaged 2.1 h (3.8 h on the other 12).", ...]

Reads Habits through its `history_for(habit_id, days)` view only. A statement needs at least MIN_DAYS days
on each side, otherwise it is not made.
"""

MIN_DAYS = 3
SHORT_NIGHT_H = 6.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def patterns(habits, days: int = 90) -> list[dict]:
    if habits is None:
        return []
    sleep = {r["day"]: r for r in habits.history_for("sleep", days) if r.get("closed") and r.get("applies", True)}
    focus = {r["day"]: r for r in habits.history_for("focus", days) if r.get("closed") and r.get("applies", True)}
    meals = {r["day"]: r for r in habits.history_for("meals", days) if r.get("closed") and r.get("applies", True)}
    out: list[dict] = []

    def split(by: dict, of: dict, cut, what: str, unit: str, scale: float = 1.0) -> None:
        both = [(by[d]["value"], of[d]["value"] * scale) for d in by if d in of]
        low = [v for s, v in both if cut(s)]
        high = [v for s, v in both if not cut(s)]
        if len(low) >= MIN_DAYS and len(high) >= MIN_DAYS:
            out.append({"text": f"On the {len(low)} days {what}, {unit} averaged {_mean(low):.1f} against {_mean(high):.1f} on the other {len(high)}.",
                        "n": len(both)})

    split(sleep, focus, lambda h: h < SHORT_NIGHT_H, f"after a night under {SHORT_NIGHT_H:g} h", "desk time (h)")
    split(sleep, meals, lambda h: h < SHORT_NIGHT_H, f"after a night under {SHORT_NIGHT_H:g} h", "meals seen")
    split(meals, focus, lambda m: m < 2, "with fewer than 2 meals seen", "desk time (h)")
    if sleep and len(sleep) >= MIN_DAYS:
        short = sum(1 for r in sleep.values() if r["value"] < SHORT_NIGHT_H)
        out.append({"text": f"{short} of the last {len(sleep)} measured nights were under {SHORT_NIGHT_H:g} h.", "n": len(sleep)})
    return out
