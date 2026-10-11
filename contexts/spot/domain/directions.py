"""A routed walk in words, from its geometry: one short instruction per stretch, in the walker's own left and right.

    steps = directions(wk, routed, start_name="Main Entrance", goal_name="Heat exchanger 2", goal_waypoint="WP05")
    -> [{"say": "From Main Entrance, go along the entrance path.", "floor": 1, "upto": 0.11},
        {"say": "Turn left and take the entrance steps.", ...},
        ...
        {"say": "Heat exchanger 2 is on your right, by the east pipe runs.", "floor": 1, "upto": 1.0, "arrive": True}]

How the words are made (all by code, nothing from a model):
- The polyline is cut into stretches wherever it turns by more than 20 degrees. A very short jog is folded into the
  next stretch. A turn is named against the direction of travel: keep going, bear, turn, turn back.
- A stretch is named by what it goes through in the trace: a door ("go through the Pump Bay opening"), the corridor or
  room it ends in, and the nearest traced piece of equipment it passes, with the side it is on.
- The picture has no scale, so no distance is ever given in metres. When a stretch ends part of the way down a
  corridor, the words say how far in relative terms ("about a third of the way along").
- A change of floor is one step ("Take the centre stairs up to floor 2").
- The last step says which side the thing is on, from where the equipment of that waypoint was traced.

`upto` is the share of the whole route's length walked when the step ends, so a screen can speak each step as its dot
gets there. Pure.
"""

import math

from contexts.spot.domain.walkable import Floor, Walkable

TURN_DEG = 20.0  # a bend smaller than this is the same stretch
SHORT_PX = 14.0  # a jog shorter than this is not worth a sentence of its own
NEAR_PX = 30.0  # equipment this close to a stretch is "passed"
LONG_PX = 60.0  # a corridor at least this long gets "a third of the way along"


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed turn from heading a to heading b in degrees; positive is to the walker's right (y grows downwards)."""
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def turn_words(deg: float) -> str:
    side = "right" if deg > 0 else "left"
    size = abs(deg)
    if size < 25:
        return "Keep going"
    if size < 60:
        return f"Bear {side}"
    if size <= 135:
        return f"Turn {side}"
    return f"Turn back to the {side}"


def side_words(deg: float) -> str:
    """Where something is, given the turn you would make to face it."""
    size = abs(deg)
    if size < 30:
        return "straight ahead"
    if size > 150:
        return "just behind you"
    return "on your right" if deg > 0 else "on your left"


def along_words(share: float) -> str:
    if share < 0.2:
        return ""
    if share < 0.42:
        return "about a third of the way along"
    if share < 0.58:
        return "about halfway along"
    if share < 0.8:
        return "about two thirds of the way along"
    return "to the end"


def _rect_gap(rect: list[float], px: float, py: float, w: int, h: int) -> float:
    dx = max(rect[0] * w - px, 0.0, px - rect[2] * w)
    dy = max(rect[1] * h - py, 0.0, py - rect[3] * h)
    return math.hypot(dx, dy)


def _stretches(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Index ranges of the polyline that run straight enough to be one instruction."""
    if len(pts) < 2:
        return []
    out, first = [], 0
    for i in range(1, len(pts) - 1):
        a = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        b = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if abs(_angle(a, b)) >= TURN_DEG:
            out.append((first, i))
            first = i
    out.append((first, len(pts) - 1))
    return out


def _samples(pts: list[tuple[float, float]], lo: int, hi: int, every: float = 4.0) -> list[tuple[float, float, float, float]]:
    """Points along a stretch with the heading there: (x, y, dx, dy) in pixels."""
    out = []
    for i in range(lo, hi):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d / every))
        for k in range(n + (1 if i == hi - 1 else 0)):
            out.append((ax + (bx - ax) * k / n, ay + (by - ay) * k / n, bx - ax, by - ay))
    return out


def _passed(f: Floor, samples: list, w: int, h: int, skip: set[str]) -> tuple[str, str] | None:
    """The nearest named piece of equipment a stretch passes, and the side it is on."""
    best = None
    for o in f.obstacles:
        label = o.get("label", "")
        if not label.startswith("the ") or label in skip or o.get("kind") == "stairs":
            continue
        for x, y, dx, dy in samples[1:-1] or samples:
            gap = _rect_gap(o["rect"], x, y, w, h)
            if gap <= NEAR_PX and (best is None or gap < best[0]):
                cx, cy = (o["rect"][0] + o["rect"][2]) / 2 * w, (o["rect"][1] + o["rect"][3]) / 2 * h
                best = (gap, label, _angle((dx, dy), (cx - x, cy - y)))
    if best is None:
        return None
    return best[1], "right" if best[2] > 0 else "left"


def _floor_steps(wk: Walkable, f: Floor, points: list[list[float]], lead: str, start_waypoint: str | None) -> list[dict]:
    w, h = wk.width, wk.height
    pts = [(p[0] * w, p[1] * h) for p in points]
    out: list[dict] = []
    heading: tuple[float, float] | None = None
    # setting off from a waypoint's own room, that room is not news ("From Pump Bay, go into the Pump Bay")
    home = f.region(points[0][0], points[0][1]) if points and start_waypoint else None
    said_region: str | None = home["id"] if home is not None and home.get("wp") == start_waypoint else None
    said_marks: set[str] = set()
    said_doors: set[str] = set()
    carried = 0.0
    stretches = _stretches(pts)
    for n, (lo, hi) in enumerate(stretches):
        length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(lo, hi))
        samples = _samples(pts, lo, hi)
        regions: list[dict] = []
        counts: dict[str, int] = {}
        for x, y, _, _ in samples:
            r = f.region(x / w, y / h)
            if r is None:
                continue
            counts[r["id"]] = counts.get(r["id"], 0) + 1
            if all(r["id"] != seen["id"] for seen in regions):
                regions.append(r)
        doors = [r for r in regions if "seen" in r and r["id"] not in said_doors]
        final = n == len(stretches) - 1
        if length < SHORT_PX and not doors and not final:
            carried += length  # a small jog: said as part of the next stretch
            continue
        vec = (pts[hi][0] - pts[lo][0], pts[hi][1] - pts[lo][1])
        end = next((r for r in reversed(regions) if "seen" not in r), None)
        # what the stretch is mostly in (not the place just named), the steps it climbs on the way, and the room it ends in
        fresh = [r for r in regions if "seen" not in r and r["id"] != said_region]
        level = [r for r in fresh if r.get("kind") != "steps"] or fresh
        main = max(level, key=lambda r: counts[r["id"]]) if level else None
        climbs = [r for r in fresh if r.get("kind") == "steps" and r is not main]
        start = lead if heading is None else turn_words(_angle(heading, vec))
        bits = [f"take {r['name']}" for r in climbs]
        named = main
        if doors:
            bits.append("go through " + ", then ".join(dict.fromkeys(d["name"] for d in doors)))
            said_doors.update(d["id"] for d in doors)
            if main is not None and main.get("kind") == "corridor" and regions.index(main) > regions.index(doors[-1]):
                bits.append(f"along {main['name']}")
            else:
                named = None
        elif main is not None:
            bits.append({"corridor": f"go along {main['name']}", "steps": f"take {main['name']}"}.get(main.get("kind", ""), f"go into {main['name']}"))
            if end is not None and end is not main and end.get("kind") in ("room", "yard") and end["id"] != said_region:
                bits.append(f"into {end['name']}")
                named = end
        body = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1] if bits else ""
        body = body.replace(" and along ", " and go along ").replace(" and into ", " into ")
        if heading is None:
            sentence = f"{start} {body or 'set off'}"
        elif not body:
            sentence = start
        elif start == "Keep going":
            sentence = body[0].upper() + body[1:]
        else:
            sentence = f"{start}, {body}" if len(bits) > 1 and climbs else f"{start} and {body}"
        mark = _passed(f, samples, w, h, said_marks)
        if mark and not final:
            sentence += f", past {mark[0]} on your {mark[1]}"
            said_marks.add(mark[0])
        if end is not None and end is main and named is main and end.get("kind") == "corridor" and not final:
            x0, y0, x1, y1 = end["rect"]
            wide = (x1 - x0) * w >= (y1 - y0) * h
            lo_px, hi_px = (x0 * w, x1 * w) if wide else (y0 * h, y1 * h)
            if hi_px - lo_px >= LONG_PX:
                at = pts[hi][0] if wide else pts[hi][1]
                share = (at - lo_px) / (hi_px - lo_px)
                if (vec[0] if wide else vec[1]) < 0:
                    share = 1 - share
                words = along_words(min(max(share, 0.0), 1.0))
                if words:
                    sentence += f", {words}"
        out.append({"say": sentence + ".", "floor": f.number, "length_px": round(length + carried, 1), "heading": vec})
        carried = 0.0
        heading = vec
        if named is not None:
            said_region = named["id"]
        elif doors and end is not None:
            said_region = end["id"]  # through a door named after its room: that room is not news in the next sentence
    if carried and out:
        out[-1]["length_px"] = round(out[-1]["length_px"] + carried, 1)
    return out


def _arrival(wk: Walkable, f: Floor, at: tuple[float, float], heading: tuple[float, float] | None, goal_name: str, goal_waypoint: str | None) -> str:
    """Where the thing is at the end of the walk: towards the traced equipment of that waypoint, else the pin itself."""
    if heading is None:
        return f"{goal_name} is here."
    w, h = wk.width, wk.height
    x, y = at[0] * w, at[1] * h
    near = sorted((( _rect_gap(o["rect"], x, y, w, h), o) for o in f.obstacles if goal_waypoint and o.get("wp") == goal_waypoint), key=lambda p: p[0])
    if near and near[0][0] <= 3 * NEAR_PX:
        o = near[0][1]
        cx, cy = (o["rect"][0] + o["rect"][2]) / 2 * w, (o["rect"][1] + o["rect"][3]) / 2 * h
        return f"{goal_name} is {side_words(_angle(heading, (cx - x, cy - y)))}, by {o['label']}."
    pin = wk.pins.get(goal_waypoint or "")
    if pin and math.hypot(pin["x"] * w - x, pin["y"] * h - y) > wk.cell:
        return f"{goal_name} is {side_words(_angle(heading, (pin['x'] * w - x, pin['y'] * h - y)))}."
    return f"{goal_name} is here."


def directions(wk: Walkable, routed: dict, start_name: str, goal_name: str, goal_waypoint: str | None = None, start_waypoint: str | None = None) -> list[dict]:
    """The steps of a routed walk (Walkable.route's answer). Empty when there is no route."""
    if not routed.get("found") or not routed.get("floors"):
        return []
    parts = routed["floors"]
    steps: list[dict] = []
    for i, part in enumerate(parts):
        f = wk.floors[part["floor"]]
        lead = f"From {start_name}," if i == 0 else f"On floor {part['floor']},"
        steps.extend(_floor_steps(wk, f, part["points"], lead, start_waypoint if i == 0 else None))
        if i < len(parts) - 1:
            nxt = parts[i + 1]["floor"]
            name = routed.get("connector_name") or "the stairs"
            steps.append({"say": f"Take {name} {'up' if nxt > part['floor'] else 'down'} to floor {nxt}.", "floor": part["floor"], "length_px": 0.0,
                          "heading": steps[-1]["heading"] if steps else None, "stairs": True})
    last = parts[-1]
    heading = next((s["heading"] for s in reversed(steps) if s.get("heading") and s["floor"] == last["floor"] and not s.get("stairs")), None)
    steps.append({"say": _arrival(wk, wk.floors[last["floor"]], tuple(last["points"][-1]), heading, goal_name, goal_waypoint), "floor": last["floor"],
                  "length_px": 0.0, "arrive": True})
    total = sum(s["length_px"] for s in steps) or 1.0
    walked = 0.0
    for s in steps:
        walked += s["length_px"]
        s["upto"] = round(walked / total, 3)
        s.pop("heading", None)
    return steps
