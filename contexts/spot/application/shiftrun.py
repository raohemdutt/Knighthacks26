"""Shift Run in words: the board to plan on, a simulated result with reasons, and Atlas's plan beside it.

    run = ShiftRun(spot)
    run.board()                                        # GET  /api/spot/shiftrun/board
    run.simulate(["HX-202", "HX-204"], turn_back_at=40)  # POST /api/spot/shiftrun/simulate
    run.recommended()                                  # GET  /api/spot/shiftrun/recommended

The model is `domain/shiftrun.py` (a toy built from the kit's recorded actions; read its header). This file only names
things the way the shift screen does (`application/plain.py`) and turns the numbers into reasons a trainee can read.
No model is called and nothing is written: the same plan always gives the same result.
"""

from contexts.spot.application import plain
from contexts.spot.application.service import Spot, not_configured
from contexts.spot.domain import shiftrun as sr

FOOTNOTE = "Simulated from {rounds} recorded rounds; a training aid, not a prediction."
HOW = ("Each check's chance of failing comes from how often checks like it failed in the recorded rounds: by battery level, kind of check "
       "and place. Battery use and time per stop are the recorded averages. There is no physics and no map distance in it.")


def chance(p: float) -> str:
    """0.31 -> 'about a 1 in 3 chance of failing'."""
    return "more likely to fail than to work" if p >= 0.5 else f"about a 1 in {round(1 / max(p, 0.01))} chance of failing"


class ShiftRun:
    def __init__(self, spot: Spot) -> None:
        self.spot = spot
        self._model: dict | None = None
        self._best: dict[str | None, dict] = {}

    @property
    def model(self) -> dict:
        if self._model is None:
            self._model = sr.build_model(self.spot.kit, self.spot.triage, self.spot.uninspected)
        return self._model

    def best(self, blocked: str | None = None) -> dict:
        if blocked not in self._best:
            self._best[blocked] = sr.recommend(self.model, blocked=blocked)
        return self._best[blocked]

    # --- words ---

    def _why_worth(self, a: dict) -> str:
        """Why checking this asset is worth what it is, in a few words."""
        state = plain.asset_state(self.spot, a["asset_id"])
        if state["reasons"]:
            return state["reasons"][0]
        if state["alarm"]["count"]:
            n = state["alarm"]["count"]
            return f"flagged {n} time{'s' if n != 1 else ''}, though the last check worked"
        hours = a["hours_since_good"]
        return f"checked {plain.span(hours)} ago and fine" if hours is not None else "never checked successfully"

    def _candidate(self, a: dict) -> dict:
        kit = self.spot.kit
        state = plain.asset_state(self.spot, a["asset_id"])
        return {"asset_id": a["asset_id"], "label": plain.asset_label(kit, a["asset_id"]), "name": plain.asset_name(kit, a["asset_id"]),
                "location_id": a["location_id"], "place": plain.place_label(kit, a["location_id"]), "floor": a["floor"],
                "check": plain.CHECKS.get(a["check"], ("", "", "camera"))[2], "check_words": plain.check_word(a["check"]),
                "value": a["value"], "level": state["level"], "why": self._why_worth(a)}

    def _step(self, s: dict) -> dict:
        kit = self.spot.kit
        name = plain.asset_name(kit, s["asset_id"])
        what = f"{plain.check_word(s['check']).capitalize()} at {name}"
        battery = f"{s['battery_pct']:.0f}% battery"
        if s["outcome"] == "checked":
            text = f"Stop {s['n']} · {what} · {battery} · {chance(s['p_fail'])}"
        elif s["outcome"] == "blocked":
            text = f"Stop {s['n']} · {what} · the way is blocked, so SPOT skips it"
        elif s["outcome"] == "stranded":
            text = f"Stop {s['n']} · {what} · the battery ran out on the way"
        else:
            text = f"Stop {s['n']} · {what} · not reached: SPOT had turned back"
        played = {"ok": "worked", "failed": "failed", "blocked": "skipped", "stranded": "battery ran out", "not_reached": "not reached"}[s["played"]]
        return {**s, "label": plain.asset_label(kit, s["asset_id"]), "name": name, "place": plain.place_label(kit, s["location_id"]),
                "check": plain.CHECKS.get(s["check"], ("", "", "camera"))[2], "sentence": text, "played_words": played}

    def _reasons(self, result: dict, best: dict) -> list[dict]:
        """What made this plan score as it did: at most five sentences, the heaviest first."""
        m, kit = self.model, self.spot.kit
        out: list[tuple[float, str, str]] = []  # (weight, tone, text)
        steps = result["steps"]
        by_asset = {s["asset_id"]: s for s in reversed(steps)}
        ranked = sorted(m["assets"].values(), key=lambda a: -a["value"])
        for place, a in enumerate(ranked[:3]):
            label = plain.asset_label(kit, a["asset_id"])
            s = by_asset.get(a["asset_id"])
            if s is None and a["asset_id"] not in best["route"]:
                continue  # Atlas leaves it out too
            if s is None:
                out.append((a["value"], "bad", f"You left out {label}: {self._why_worth(a)}." + (" It is the most valuable stop." if place == 0 else "")))
            elif s["outcome"] == "not_reached":
                out.append((a["value"], "bad", f"{label} was stop {s['n']}, but SPOT had already turned back at {result['turn_back_at']:.0f}% battery, so it was never checked."))
            elif s["outcome"] == "blocked":
                out.append((a["value"] * 0.8, "bad", f"The way to {label} was blocked, so that stop was skipped."))
            elif s["outcome"] == "checked" and s["low"]:
                out.append((a["value"] * s["p_fail"] * 2, "bad", f"You left {label} for stop {s['n']}: SPOT reaches it at {s['battery_pct']:.0f}% battery, "
                                                                 f"where the check {'is' if s['p_fail'] >= 0.5 else 'has'} {chance(s['p_fail'])}."))
            elif s["outcome"] == "checked":
                out.append((a["value"] * 0.25, "good", f"Good: {label} is stop {s['n']}, checked at {s['battery_pct']:.0f}% battery."))
        danger = [s for s in steps if s["outcome"] == "checked" and s.get("danger")]
        if result["stopped"] == "stranded":
            out.append((200, "bad", "The battery ran out on the floor. Someone has to carry SPOT back: turn back sooner or plan fewer stops."))
        elif danger:
            out.append((40 + 10 * len(danger), "bad", f"SPOT was still checking at {danger[0]['battery_pct']:.0f}% battery. Under {sr.DANGER:.0f}% most checks fail: "
                                                      f"turn back sooner."))
        elif result["end_battery_pct"] < sr.DANGER:
            out.append((35, "bad", f"SPOT got back with {result['end_battery_pct']:.0f}% battery: too close."))
        skipped = [s for s in steps if s["outcome"] == "not_reached"]
        if skipped and result["stopped"] == "turn_back":
            names = ", ".join(plain.asset_name(kit, s["asset_id"]) for s in skipped[:3])
            out.append((sum(s["value"] for s in skipped) * 0.4, "bad", f"Turning back at {result['turn_back_at']:.0f}% left {len(skipped)} "
                                                                       f"stop{'s' if len(skipped) != 1 else ''} unchecked: {names}."))
        elif not danger and result["stopped"] is None and steps and not any(s.get("low") for s in steps):
            out.append((12, "good", "Good: every check ran above the battery level where checks start to fail."))
        cheap = [s for s in steps if s["outcome"] == "checked" and m["assets"][s["asset_id"]]["value"] <= sr.BASE_VALUE + 6]
        if cheap:
            s = cheap[0]
            out.append((14, "bad", f"{plain.asset_name(kit, s['asset_id'])} was {self._why_worth(m['assets'][s['asset_id']])}: little to gain from another check now."))
        risky = max((s for s in steps if s["outcome"] == "checked"), key=lambda s: s["p_fail"], default=None)
        if risky and risky["p_fail"] >= 0.4 and not any(risky["asset_id"] in t for _, _, t in out):
            out.append((risky["value"] * risky["p_fail"], "bad", f"The riskiest stop was {plain.asset_name(kit, risky['asset_id'])} at {plain.place_label(kit, risky['location_id'])}: "
                                                                 f"{chance(risky['p_fail'])}."))
        if not steps:
            out.append((100, "bad", "No stops yet. Tap equipment on the plan to add it to the round."))
        out.sort(key=lambda r: -r[0])
        return [{"tone": tone, "text": text} for _, tone, text in out[:5]]

    def _differences(self, result: dict, best: dict) -> list[str]:
        """The two or three moves that separate this plan from Atlas's."""
        kit, m = self.spot.kit, self.model
        mine, theirs = result["route"], best["route"]
        out: list[str] = []
        if mine == theirs and result["turn_back_at"] == best["turn_back_at"]:
            return ["Your plan is the same as Atlas's."]
        missing = sorted((a for a in theirs if a not in mine), key=lambda a: -m["assets"][a]["value"])
        if missing:
            out.append("Atlas also visits " + ", ".join(plain.asset_label(kit, a) for a in missing[:2])
                       + (f" and {len(missing) - 2} more" if len(missing) > 2 else "") + f": {self._why_worth(m['assets'][missing[0]])}.")
        if theirs and mine and theirs[0] != mine[0]:
            first = m["assets"][theirs[0]]
            out.append(f"Atlas starts with {plain.asset_label(kit, theirs[0])}, while the battery is full: {self._why_worth(first)}.")
        extra = [a for a in mine if a not in theirs]
        if extra:
            out.append("Atlas skips " + ", ".join(plain.asset_name(kit, a) for a in extra[:2]) + ": too little to gain for the battery it costs.")
        if result["turn_back_at"] != best["turn_back_at"]:
            out.append(f"Atlas turns back at {best['turn_back_at']:.0f}% battery; you chose {result['turn_back_at']:.0f}%.")
        return out[:3] or ["Atlas visits the same equipment in a different order, so the important stops come while the battery is high."]

    def _pack(self, result: dict) -> dict:
        return {**result, "steps": [self._step(s) for s in result["steps"]], "route_places": [s["location_id"] for s in result["steps"]]}

    # --- the three calls ---

    def board(self) -> dict:
        if not self.spot.configured:
            return {**not_configured(), "candidates": []}
        m, kit = self.model, self.spot.kit
        worst = m["worst_location"]
        return {"configured": True, "candidates": [self._candidate(a) for a in sorted(m["assets"].values(), key=lambda a: (-a["value"], a["asset_id"]))],
                "start_battery_pct": m["start_battery"], "turn_back": {"choices": list(sr.TURN_BACK_CHOICES), "default": 40, "low": sr.LOW, "danger": sr.DANGER},
                "max_stops": sr.MAX_STOPS, "last_route": m["last_route"],
                "per_stop": {"battery_pct": m["drop"]["same"], "battery_pct_other_floor": m["drop"]["other"],
                             "minutes": round(m["seconds"]["same"] / 60, 1), "minutes_other_floor": round(m["seconds"]["other"] / 60, 1)},
                "hard": {"blocked": worst, "words": f"Hard mode: the way to {plain.place_label(kit, worst)} is blocked. The manual says the operator then walks SPOT "
                                                    f"past it and skips that part of the mission ({plain.GUIDE}, p. 43)." if worst else None},
                "footnote": FOOTNOTE.format(rounds=m["rounds"]), "how": HOW,
                "manual": [plain.manual(plain.MANUAL_BATTERY), plain.manual(plain.MANUAL_FAILURE["PATH_BLOCKED"])]}

    def recommended(self, hard: bool = False) -> dict:
        if not self.spot.configured:
            return {**not_configured(), "route": [], "steps": []}
        blocked = self.model["worst_location"] if hard else None
        best = self.best(blocked)
        return {"configured": True, **self._pack(best), "stars": 3, "footnote": FOOTNOTE.format(rounds=self.model["rounds"])}

    def simulate(self, route: list[str], turn_back_at: float, seed: int = 0, hard: bool = False) -> dict:
        if not self.spot.configured:
            return {**not_configured(), "route": [], "steps": []}
        m = self.model
        blocked = m["worst_location"] if hard else None
        result = sr.simulate(m, [a.upper() for a in route], turn_back_at, seed=seed, blocked=blocked)
        best = self.best(blocked)
        n = sr.stars(result["score"], best["score"])
        verdict = {3: "A strong plan.", 2: "A decent plan with something to fix.", 1: "This plan leaves a lot on the floor.", 0: "This plan needs a rethink."}[n]
        return {"configured": True, **self._pack(result), "stars": n, "verdict": verdict, "best_score": best["score"],
                "summary": (f"{result['checked']} stops checked, about {result['expected_ok']:.0f} expected to work. "
                            f"About {result['minutes']} minutes. Battery {result['start_battery_pct']:.0f}% to {result['end_battery_pct']:.0f}%."),
                "reasons": self._reasons(result, best), "differences": self._differences(result, best),
                "recommended": {**self._pack(best), "stars": 3,
                                "summary": (f"{best['checked']} stops, about {best['expected_ok']:.0f} expected to work, {best['minutes']} minutes, "
                                            f"back with {best['end_battery_pct']:.0f}% battery, turning back at {best['turn_back_at']:.0f}%.")},
                "footnote": FOOTNOTE.format(rounds=m["rounds"])}
