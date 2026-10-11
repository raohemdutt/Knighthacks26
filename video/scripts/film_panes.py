"""Write the 2D overlays (compositions/<id>.html) of the v4.7 scenes from one pane library, timed from film.json.

    python video/scripts/film_build.py && python video/scripts/film_panes.py

Every demo scene gets Atlas's app window in the top-right corner (the website's soft UI): a title bar with the tab
name and panes that switch on the spoken lines, like a screen recording of the app. Spec scenes get labelled cards.
Times are written as "<line>:<fraction>" (a point inside spoken line n) or "end-<s>" (seconds before the scene ends)
and resolved against film.json, so the overlays move with the words. Numbers on screen come from FILM_PLAN.md's
sources; nothing is typed in that the plan does not cite.
"""

import json
import re
import sys
from pathlib import Path

HF = Path(__file__).resolve().parents[1] / "hf"

BASE_CSS = """
#root { position: absolute; inset: 0; font-family: var(--font); overflow: hidden; }
.chip-ch { position: absolute; left: 96px; top: 72px; padding: 14px 28px; font-size: 28px; opacity: 0; }
.win { position: absolute; left: 1196px; top: 60px; width: 664px; height: 640px; opacity: 0; padding: 0; overflow: hidden; }
.win-bar { height: 76px; display: flex; align-items: center; gap: 14px; padding: 0 28px; border-bottom: 2px solid var(--surface-2); }
.win-dot { width: 18px; height: 18px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 6px rgba(69,108,179,0.16); }
.win-app { font-family: var(--font); font-size: 30px; font-weight: 700; color: var(--ink); letter-spacing: 0.02em; }
.win-tabs { position: relative; flex: 1; height: 40px; }
.win-tab { position: absolute; right: 0; top: 0; padding: 4px 18px; border-radius: 999px; font-size: 24px; font-weight: 700; color: var(--accent); background: var(--surface-2); box-shadow: inset 3px 3px 7px var(--shade), inset -3px -3px 7px var(--light); opacity: 0; white-space: nowrap; }
.pane { position: absolute; left: 28px; right: 28px; top: 100px; bottom: 24px; opacity: 0; }
.ph { font-family: var(--font); font-size: 38px; font-weight: 700; color: var(--ink); letter-spacing: -0.01em; }
.pk { font-family: var(--font-body); font-size: 24px; font-weight: 600; color: var(--muted); }
.row { display: flex; align-items: center; gap: 18px; padding: 14px 20px; margin-top: 12px; border-radius: 20px; background: var(--surface); box-shadow: 6px 6px 14px var(--shade), -6px -6px 14px var(--light); }
.row.dash { box-shadow: none; border: 3px dashed var(--shade); }
.row .t { width: 96px; font-family: var(--font); font-size: 28px; font-weight: 700; color: var(--accent); }
.row .w { font-family: var(--font-body); font-size: 28px; font-weight: 600; color: var(--ink); line-height: 1.2; }
.row .w small { display: block; font-size: 22px; color: var(--muted); font-weight: 500; }
.bub { max-width: 520px; margin-top: 16px; padding: 16px 22px; border-radius: 24px 24px 24px 8px; background: var(--surface); box-shadow: 6px 6px 14px var(--shade), -6px -6px 14px var(--light); font-family: var(--font-body); font-size: 26px; font-weight: 500; color: var(--ink); line-height: 1.3; }
.bub b { display: block; font-size: 22px; color: var(--accent); }
.bub.me { margin-left: auto; border-radius: 24px 24px 8px 24px; box-shadow: 6px 6px 14px var(--shade), -6px -6px 14px var(--light), inset 0 0 0 3px var(--accent-bright); }
.chips { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 18px; }
.chips .site-tag { font-size: 24px; }
.big { font-family: var(--font); font-size: 92px; font-weight: 700; color: var(--accent); line-height: 1; }
.card { position: absolute; padding: 26px 34px; opacity: 0; }
.card .ph { font-size: 40px; }
.cardrow { position: absolute; display: flex; gap: 26px; opacity: 0; }
.mini { padding: 20px 28px; }
.btnrow { display: flex; gap: 18px; margin-top: 26px; }
.ok { color: var(--ok); }
.svgbox { margin-top: 14px; border-radius: 22px; overflow: hidden; }
"""

PAGE = """<!doctype html>
<html>
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <link rel="stylesheet" href="world/soft.css" />
      <style>{css}{extra}</style>
      <div id="root" data-composition-id="{id}" data-width="1920" data-height="1080">
        <div id="{id}-chapter" class="chip-ch soft-raised soft-muted">{title}</div>
{body}
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
          tl.fromTo("#{id}-chapter", {{ opacity: 0, y: -12 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }}, 0.15);
          tl.to("#{id}-chapter", {{ opacity: 0, duration: 0.4 }}, {chapter_out:.3f});
{tweens}
          window.__timelines["{id}"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


class Scene:
    def __init__(self, sc: dict):
        self.sc, self.id, self.d = sc, sc["id"], sc["duration"]
        self.body, self.tweens, self.extra = [], [], ""
        self.n = 0

    def t(self, spec) -> float:
        """'2:0.5' = halfway through line 2; 'end-1.5' = 1.5 s before the scene ends; a number = local seconds."""
        if isinstance(spec, (int, float)):
            return float(spec)
        if spec.startswith("end-"):
            return max(0.0, self.d - float(spec[4:]))
        i, f = spec.split(":")
        lines = self.sc["lines"]
        i = min(int(i), len(lines) - 1)
        l = lines[i]
        return round(l["start"] - self.sc["start"] + float(f) * l["duration"], 3)

    def uid(self, kind: str) -> str:
        self.n += 1
        return f"{self.id}-{kind}{self.n}"

    def show(self, sel: str, at, out=None, y=14, dur=0.45):
        a = self.t(at)
        self.tweens.append(f'          tl.fromTo("{sel}", {{ opacity: 0, y: {y} }}, {{ opacity: 1, y: 0, duration: {dur}, ease: "power3.out" }}, {a:.3f});')
        if out is not None:
            self.tweens.append(f'          tl.to("{sel}", {{ opacity: 0, duration: 0.35, ease: "power2.in" }}, {self.t(out) - 0.35:.3f});')

    def window(self, panes: list[tuple[str, str, str]], at=0.4, out="end-0.4", top=60, height=640):
        """panes: (tab name, inner html, when) in order; each pane shows until the next one starts."""
        wid = self.uid("win")
        inner = []
        for k, (tab, html, when) in enumerate(panes):
            inner.append(f'<div class="win-tab" id="{wid}-tab{k}">{tab}</div>')
        body = [f'        <div class="win soft-raised" id="{wid}" style="top:{top}px;height:{height}px">',
                f'          <div class="win-bar"><div class="win-dot"></div><div class="win-app">Atlas</div><div class="win-tabs">{"".join(inner)}</div></div>']
        for k, (tab, html, when) in enumerate(panes):
            body.append(f'          <div class="pane" id="{wid}-p{k}">{html}</div>')
        body.append("        </div>")
        self.body.extend(body)
        self.show(f"#{wid}", at, out, y=24, dur=0.6)
        for k, (tab, html, when) in enumerate(panes):
            nxt = panes[k + 1][2] if k + 1 < len(panes) else out
            a, b = self.t(when), self.t(nxt)
            for sel in (f"#{wid}-p{k}", f"#{wid}-tab{k}"):
                self.tweens.append(f'          tl.fromTo("{sel}", {{ opacity: 0, y: 10 }}, {{ opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }}, {a:.3f});')
                if k + 1 < len(panes):
                    self.tweens.append(f'          tl.to("{sel}", {{ opacity: 0, duration: 0.25, ease: "power2.in" }}, {b - 0.25:.3f});')
        return wid

    def card(self, html: str, x: int, y: int, at, out="end-0.4", w=None, cls="card soft-raised"):
        cid = self.uid("card")
        style = f"left:{x}px;top:{y}px" + (f";width:{w}px" if w else "")
        self.body.append(f'        <div class="{cls}" id="{cid}" style="{style}">{html}</div>')
        self.show(f"#{cid}", at, out)
        return cid

    def tween(self, line: str):
        self.tweens.append("          " + line)

    def write(self):
        out = self.t("end-0.3")
        chapter_out = min(3.2, max(1.0, out - 0.5))
        html = PAGE.format(css=BASE_CSS, extra=self.extra, id=self.id, title=self.sc["title"], body="\n".join(self.body),
                           tweens="\n".join(self.tweens), chapter_out=chapter_out)
        (HF / "compositions" / f"{self.id}.html").write_text(html, encoding="utf-8")


def rows(items):
    out = []
    for t, w, small, dash in items:
        out.append(f'<div class="row{" dash" if dash else ""}"><div class="t">{t}</div><div class="w">{w}{f"<small>{small}</small>" if small else ""}</div></div>')
    return "".join(out)


ROUTE_SVG = """<svg viewBox="0 0 600 300" width="600" height="300"><rect width="600" height="300" fill="#e6ebf3"/>
<g stroke="#c9d3e3" stroke-width="14" stroke-linecap="round" fill="none"><path d="M40 250 H560"/><path d="M120 30 V280"/><path d="M300 20 V280"/><path d="M40 140 H560"/><path d="M460 30 V280"/></g>
<path id="ROUTE" d="M120 250 L120 140 L300 140 L300 70 L460 70" fill="none" stroke="#456cb3" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="120" cy="250" r="14" fill="#1b2433"/><circle cx="460" cy="70" r="16" fill="#456cb3"/>
<text x="140" y="275" font-size="22" font-family="Segoe UI" fill="#566176">Your dorm</text><text x="350" y="50" font-size="22" font-family="Segoe UI" fill="#566176">Engineering II</text></svg>"""


def morning(s: Scene):
    today = '<div class="ph">Today</div><div class="pk">calendar blocks solid · Atlas\'s ideas dashed</div>' + rows([
        ("9:20", "Mentor check-in", "Engineering II", False), ("11:00", "Submissions close", "", False),
        ("12:00", "Judging", "", False), ("2:00", "Maya's study session", "John C. Hitt Library", False), ("8:00", "Board game night", "optional", True)])
    route = '<div class="ph">Route</div><div class="pk">real UCF map · dorm to Engineering II</div><div class="svgbox">' + ROUTE_SVG + \
        '</div><div class="chips"><span class="site-tag">Leave by 9:10</span><span class="site-tag">680 m</span><span class="site-tag">10 min walk</span></div>'
    plan = '<div class="ph">Today\'s Plan</div><div class="pk">Try your day before you live it.</div><div class="svgbox">' + ROUTE_SVG.replace('id="ROUTE"', 'id="PLANROUTE"') + \
        '</div><div class="chips"><span class="site-tag">9:20 ✓</span><span class="site-tag">11:00 ✓</span><span class="site-tag">noon ✓</span><span class="site-tag">2:00 ✓</span></div>'
    week = '<div class="ph">This week</div>' + rows([("Tue", "Algorithms writeup", "due 11:59 PM", False)])
    opt = '<div class="ph">Optional</div>' + rows([("8:00", "Board game night", "Student Union · with Sam, Riley", True)]) + \
        '<div class="chips"><span class="site-tag">via GroupMe</span><span class="site-tag">free</span></div>'
    approve = ('<div class="ph">Lock today\'s plan?</div><div class="pk">nothing is booked until you say yes</div>' + rows([("Goal", "2 real meals today", "you asked for it", False)]) +
               f'<div class="btnrow"><span class="site-btn" id="{s.id}-approve">Approve</span><span class="site-chip">Change</span></div>'
               f'<div class="ph ok" id="{s.id}-locked" style="margin-top:22px;opacity:0">Locked ✓</div>')
    keys = ('<div class="bub me">Where are my keys?</div><div class="bub"><b>Atlas</b>Behind the mug on your desk. You left them there last night.</div>'
            '<div class="svgbox"><svg viewBox="0 0 600 190" width="600" height="190"><rect width="600" height="190" fill="#e6ebf3"/>'
            '<rect x="40" y="110" width="520" height="22" rx="8" fill="#d8dfeb"/><rect x="250" y="60" width="54" height="56" rx="10" fill="#cfd7e5"/>'
            '<circle cx="330" cy="104" r="34" fill="none" stroke="#c0332b" stroke-width="6"/><circle cx="330" cy="104" r="8" fill="#566176"/>'
            '<text x="380" y="70" font-size="24" font-family="Segoe UI" font-weight="700" fill="#c0332b">Keys here</text></svg></div>')
    s.window([("Today", today, "2:0"), ("Route", route, "2:0.55"), ("Today's Plan", plan, "3:0"), ("This week", week, "3:0.78"),
              ("Optional", opt, "4:0"), ("Approve", approve, "5:0.2"), ("Ask", keys, "8:0")], at="2:0")
    a = s.t("7:0.7")
    s.tween(f'tl.to("#{s.id}-approve", {{ scale: 0.94, duration: 0.12, yoyo: true, repeat: 1 }}, {a:.3f});')
    s.tween(f'tl.to("#{s.id}-locked", {{ opacity: 1, duration: 0.3 }}, {a + 0.35:.3f});')
    for rid, at in (("ROUTE", "2:0.6"), ("PLANROUTE", "3:0.05")):
        s.tween(f'tl.fromTo("#{rid}", {{ strokeDasharray: 900, strokeDashoffset: 900 }}, {{ strokeDashoffset: 0, duration: 2.2, ease: "power2.inOut" }}, {s.t(at):.3f});')


def problem(s: Scene):
    apps = [("Calendar", "9:20 Mentor check-in", 180, 200), ("Group chat", "Maya: study session at 2?", 760, 150),
            ("Notes", "algorithms writeup: runtime", 1340, 260), ("Sleep", "7 h 28 m", 300, 640), ("Map", "Engineering II · 680 m", 1220, 640)]
    for name, sub, x, y in apps:
        cid = s.card(f'<div class="pk">{name}</div><div class="ph" style="font-size:34px">{sub}</div>', x, y, 0.2, out="1:0.5", w=420)
        s.tween(f'tl.to("#{cid}", {{ x: {960 - x - 210}, y: {560 - y}, scale: 0.15, duration: 1.4, ease: "power3.in" }}, {s.t("1:0.05"):.3f});')


def sources(s: Scene):
    names = ["Camera", "Apple Watch", "Outlook", "Canvas", "GroupMe", "UCF Events", "OpenStreetMap"]
    chips = "".join(f'<span class="site-chip">{n}</span>' for n in names)
    s.card(f'<div class="pk">what fed the brief</div><div class="chips">{chips}</div><div class="chips"><span class="site-tag">written on the Jetson · under 3 s</span><span class="site-tag">Approve? sent to the phone</span></div>',
           220, 520, "0:0", out="1:0", w=1480)
    steps = [("≈10,700", "frames"), ("9,738", "things seen"), ("19", "episodes"), ("55", "answers")]
    cells = "".join(f'<div class="site-card mini" style="text-align:center"><div class="big" style="font-size:72px">{v}</div><div class="pk">{k}</div></div>' + ('<div class="ph" style="align-self:center">→</div>' if i < 3 else "")
                    for i, (v, k) in enumerate(steps))
    s.card(f'<div class="pk">what the camera captured at our table, Saturday (the box\'s own log)</div><div style="display:flex;gap:22px;margin-top:16px">{cells}</div>'
           '<div class="chips"><span class="site-tag">with your OK: the examples it learns from</span></div>', 200, 500, "1:0", w=1520)


def model(s: Scene):
    s.card('<div class="pk">on the Jetson</div><div class="ph">Qwen · 4 billion parameters</div>', 140, 620, "0:0", w=620)
    s.card('<div class="chips"><span class="site-chip">Fast</span><span class="site-chip">Accurate</span><span class="site-chip">Private</span></div>', 140, 820, "0:0.25")
    s.card('<div class="pk">demo questions</div><div class="big">48 / 48</div>', 1380, 640, "0:0.6", w=420)


def desk(s: Scene):
    hr = ('<svg viewBox="0 0 600 120" width="600" height="120"><polyline fill="none" stroke="#c0332b" stroke-width="5" points="0,90 60,88 120,84 180,80 240,74 300,70 360,62 420,58 480,50 540,46 600,40"/>'
          '<text x="8" y="24" font-size="22" font-family="Segoe UI" fill="#566176">resting heart rate · from your watch</text></svg>')
    check = ('<div class="ph">Check-in</div><div class="svgbox">' + hr + '</div><div class="chips"><span class="site-tag">Looked tired</span><span class="site-tag">Sat 1 h 45</span>'
             '<span class="site-tag">No water 100 min</span><span class="site-tag">3rd can · limit 2</span></div>')
    chat = ('<div class="ph">COP4533 · class chat</div><div class="pk">GroupMe</div>'
            '<div class="bub"><b>Maya</b>prof said in class half the milestone grade is the runtime analysis, we should go over weeks 5 and 6 together</div>'
            '<div class="bub"><b>Jordan</b>weeks 5 and 6. the knapsack and interval scheduling slides basically walk through it</div>')
    notes = ('<div class="ph">Notes for the writeup</div>' + rows([("1", "Runtime analysis: half the grade", "GroupMe · class chat", False), ("2", "Knapsack, weeks 5 and 6", "r/ucf thread", False),
                                                                 ("3", "Interval scheduling proof", "two articles", False)]))
    first = '<div class="ph">But first</div>' + rows([("💧", "Fill your bottle", "no water in 90 min", False), ("🚶", "Walk a few minutes", "sat 1 h 45", False)]) + '<div class="pk" style="margin-top:16px">I\'ll hold the notes.</div>'
    s.window([("Check-in", check, "0:0"), ("GroupMe", chat, "1:0"), ("Notes", notes, "1:0.7"), ("Check-in", first, "2:0")], at="0:0")


def faces(s: Scene):
    face = lambda blur, label, sub: (f'<div class="site-card mini" style="width:380px;text-align:center"><svg viewBox="0 0 200 200" width="200" height="200" style="filter:blur({blur}px)">'
                                    '<circle cx="100" cy="100" r="80" fill="#d8ad8a"/><circle cx="72" cy="88" r="9" fill="#1b2433"/><circle cx="128" cy="88" r="9" fill="#1b2433"/>'
                                    '<path d="M70 134 Q100 120 130 134" stroke="#1b2433" stroke-width="7" fill="none" stroke-linecap="round"/></svg>'
                                    f'<div class="ph" style="font-size:32px">{label}</div><div class="pk">{sub}</div></div>')
    row = face(0, "Your own screen", "live · clear") + face(14, "Saved pictures", "blurred") + face(16, "Anything that leaves", "blurred")
    cid = s.card(f'<div style="display:flex;gap:30px">{row}</div>', 260, 560, "0:0.35", cls="cardrow")
    s.card('<span class="site-tag">looked tired · 30 s window · read on the device · opt-in</span>', 600, 470, "0:0.05", cls="cardrow")


def kitchen(s: Scene):
    bins = ('<svg viewBox="0 0 600 220" width="600" height="220"><rect width="600" height="220" fill="#e6ebf3"/>'
            '<rect id="BLUE" x="60" y="70" width="130" height="130" rx="16" fill="#456cb3"/><rect x="235" y="70" width="130" height="130" rx="16" fill="#0f7b5f" opacity="0.5"/>'
            '<rect x="410" y="70" width="130" height="130" rx="16" fill="#8a93a3" opacity="0.5"/><text x="70" y="55" font-size="22" font-family="Segoe UI" font-weight="700" fill="#456cb3">Recycling</text>'
            '<text x="245" y="55" font-size="22" font-family="Segoe UI" fill="#566176">Compost</text><text x="420" y="55" font-size="22" font-family="Segoe UI" fill="#566176">Trash</text></svg>')
    sort = '<div class="ph">Can goes in recycling</div><div class="pk">UCF\'s rule · the blue bin</div><div class="svgbox">' + bins + '</div><div class="chips"><span class="site-tag">the kitchen camera saw the can</span></div>'
    ring = ('<div class="ph">This week</div><div style="display:flex;align-items:center;gap:30px;margin-top:10px"><svg viewBox="0 0 200 200" width="220" height="220">'
            '<circle cx="100" cy="100" r="80" fill="none" stroke="#e6ebf3" stroke-width="22"/><circle id="KRING" cx="100" cy="100" r="80" fill="none" stroke="#456cb3" stroke-width="22" stroke-linecap="round" transform="rotate(-90 100 100)" stroke-dasharray="502.6" stroke-dashoffset="166"/>'
            '<line x1="100" y1="8" x2="100" y2="34" stroke="#1b2433" stroke-width="5" transform="rotate(288 100 100)"/></svg>'
            '<div><div class="big" id="KPCT">71%</div><div class="pk">5 of 7 right bin · goal 80%</div></div></div>'
            '<div class="chips"><span class="site-tag">sources: config/impact.yaml</span></div>')
    s.window([("Recycling", sort, "0:0"), ("Impact", ring, "1:0")], at="0:0")
    s.tween(f'tl.fromTo("#KRING", {{ strokeDashoffset: 166 }}, {{ strokeDashoffset: 146, duration: 1.0, ease: "power2.out" }}, {s.t("1:0.1"):.3f});')


def sortit(s: Scene):
    round_ = ('<div class="ph">Sort It</div><div class="pk">hold it up for two seconds</div>' + rows([("Cup", "goes in trash", "the grey bin · route drawn", False)]) +
              f'<div class="ph ok" id="{s.id}-score" style="margin-top:22px;opacity:0">Right bin · 6.0 s · 150 points</div>')
    s.window([("Sort It", round_, "0:0")], at="0:0")
    s.tween(f'tl.to("#{s.id}-score", {{ opacity: 1, duration: 0.3 }}, {s.t("end-1.6"):.3f});')


def sorttech(s: Scene):
    s.card('<div class="pk">on the Jetson</div><div class="ph">YOLO11 · 41 frames a second</div><div class="chips"><span class="site-tag">tracks every item</span><span class="site-tag">bin zones</span></div>', 110, 600, "0:0", w=860)
    s.card('<div class="ph">UCF\'s own bin rules</div><div class="chips"><span class="site-tag">config/sorting.yaml</span><span class="site-tag">impact numbers cite their sources</span><span class="site-tag">only counts leave</span></div>', 1000, 600, "0:0.55", w=820)
    s.card('<div class="ph">Look guide</div><div class="pk">follows your eyes to point the way · the frames never leave the browser</div>', 1000, 360, "1:0", w=820)


def lab(s: Scene):
    s.card('<div class="pk">measured on AMD Strix Halo · lab/refine/metrics.jsonl</div><div class="ph">Teacher: 11 → 51 tokens a second</div><div class="pk">batched, at about 96 W</div>'
           '<div class="chips"><span class="site-tag">8.8 J → 1.9 J per token</span><span class="site-tag">pod idle ≈ 10 W</span><span class="site-tag">Jetson idle 12.6 W</span></div>', 110, 450, "0:0.2", w=900)
    steps = "".join(f'<span class="site-chip">{x}</span>' for x in ["Test gate", "Hugging Face", "Box checks it", "Swaps in live", "Rolls back on failure"])
    s.card(f'<div class="chips">{steps}</div>', 110, 740, "1:0.1", w=1700, cls="cardrow")


def device(s: Scene):
    chips = "".join(f'<span class="site-chip{" on" if x != "learning (paused)" else ""}">{x}</span>' for x in ["seeing", "remembering", "answering", "speaking", "learning (paused)"])
    s.card(f'<div class="pk">network cable pulled</div><div class="chips">{chips}</div><div class="chips"><span class="site-tag">your question goes ahead of the cameras</span></div>', 110, 500, "0:0.1", w=1200)
    s.card('<div class="pk">ledger of what left</div><div style="display:flex;gap:40px"><div><div class="big">13</div><div class="pk">allowed</div></div><div><div class="big" style="color:var(--danger)">13</div><div class="pk">blocked</div></div></div>', 1400, 480, "1:0", w=420)


def league(s: Scene):
    table = ('<div class="ph">This week\'s league</div>' + rows([("1st", "Dominick", "most goals · picks tonight", False), ("2nd", "Ilan", "", False), ("3rd", "Hemdutt", "roughest week · driving", False)]))
    cal = ('<div class="ph">Three calendars</div><div class="pk">each box reads only its own</div>'
           '<div style="display:flex;gap:14px;margin-top:14px">' + "".join(
               f'<div class="site-card mini" style="flex:1;padding:14px"><div class="pk">{n}</div><div class="row dash" style="padding:8px 10px"><div class="w" style="font-size:22px">{b}</div></div>'
               f'<div class="row" id="{s.id}-c{i}" style="padding:8px 10px;opacity:0"><div class="w" style="font-size:22px">8:30 Board games</div></div></div>'
               for i, (n, b) in enumerate([("Hemdutt", "busy"), ("Ilan", "busy at 8"), ("Dominick", "busy")])) + '</div>'
           '<div class="bub" style="font-size:22px"><b>Hemdutt\'s box</b>Free.</div><div class="bub me" style="font-size:22px">Busy at 8. How about 8:30? <b>Ilan\'s box</b></div>'
           '<div class="bub" style="font-size:22px"><b>Dominick\'s box</b>8:30 works.</div>')
    dinner = ('<div class="ph">Ilan is at \'63 South</div><div class="pk">shared place · expires tonight</div>' + rows([("now", "Dinner with Ilan", "free till 8:30", True)]) +
              '<div class="chips"><span class="site-tag">Agreed, three of three · 8:30</span></div>')
    meals = ('<div class="ph">Meals today</div><div class="big">2 of 2</div><div class="pk">goal met · the one you set this morning</div>' +
             rows([("3rd", "Hemdutt", "league row updated", False)]))
    s.window([("League", table, 0.4), ("Calendars", cal, "0:0"), ("Places", dinner, "1:0"), ("Meals", meals, "3:0")], at=0.3)
    for i in range(3):
        s.tween(f'tl.to("#{s.id}-c{i}", {{ opacity: 1, duration: 0.3 }}, {s.t("0:1") + 0.6:.3f});')


def friends(s: Scene):
    s.card('<div class="pk">where the boxes meet</div><div class="ph">Neon Postgres</div><div class="chips"><span class="site-tag">only what each person chose to share</span><span class="site-tag">presence · league · RSVPs</span><span class="site-tag">works over 443</span></div>', 110, 620, "0:0.45", w=900)
    s.card('<div class="pk">a group you trust</div><div class="chips"><span class="site-chip on">Ilan</span><span class="site-chip on">Dominick</span><span class="site-chip on">Hemdutt</span></div>'
           '<div class="chips"><span class="site-tag">boxes recognise each other · face numbers, never pictures</span><span class="site-tag">learn from each other\'s examples</span></div>', 1050, 600, "1:0", w=760)


def lives(s: Scene):
    vs = [("Night shift", "10 PM to 6 AM · keys, badge, phone, charger"), ("Family", "Leo: bins out · Ava: science fair poster"), ("Plant floor", "inspection rounds")]
    row = "".join(f'<div class="site-card mini" style="width:520px"><div class="ph" style="font-size:34px">{a}</div><div class="pk">{b}</div></div>' for a, b in vs)
    s.card(f'<div style="display:flex;gap:26px">{row}</div>', 110, 640, "0:0", cls="cardrow")
    s.card('<span class="site-tag">12 W at idle</span>', 840, 880, "0:0.7", cls="cardrow")


def plant(s: Scene):
    # The triage is the real one (docs/spot.md, "Triage"): HX-202 re-inspect, HX-204 send a technician, GAUGE-209 48 h.
    tri = "".join(f'<div class="site-card mini" style="width:420px"><div class="pk">{a}</div><div class="ph" style="font-size:32px">{b}</div></div>'
                  for a, b in [("Do this first", "HX-202 · re-inspect"), ("Then", "HX-204 · a technician"), ("Keep an eye", "GAUGE-209 · 48 h")])
    s.card(f'<div class="pk">the brief becomes a shift triage</div><div style="display:flex;gap:22px;margin-top:12px">{tri}</div>', 110, 560, "0:0.1", out="1:0", w=1400)
    s.card('<div class="pk">every number is computed from the rows and cites them</div><div class="chips"><span class="site-tag">21 of 26 completed missions hide a failed action</span></div>', 110, 600, "1:0", w=820)
    s.card("<div class=\"pk\">the lab's SPOT test</div><div class=\"ph\">cites the right rows · every expected number</div><div class=\"chips\"><span class=\"site-tag\">before it's allowed on the box</span></div>", 980, 560, "1:0.45", w=840)


def shift(s: Scene):
    tri = ('<div class="ph">Your shift</div>' + rows([("1", "Heat exchanger 2 · HX-202", "alarm from a failed check · no good check in 88 h", False),
                                                     ("2", "HX-204 · send a technician", "critical anomaly, confirmed", True), ("3", "GAUGE-209", "48 h without a good check", True)]))
    plan_svg = ('<svg viewBox="0 0 600 300" width="600" height="300"><rect width="600" height="300" fill="#e6ebf3"/>'
                '<g fill="#d8dfeb"><rect x="30" y="30" width="200" height="110" rx="8"/><rect x="260" y="30" width="310" height="110" rx="8"/><rect x="30" y="170" width="540" height="100" rx="8"/></g>'
                '<path id="SPOTPATH" d="M80 250 L80 155 L250 155 L250 80 L470 80" fill="none" stroke="#c0332b" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
                '<circle cx="470" cy="80" r="16" fill="#c0332b"/><text x="420" y="60" font-size="22" font-family="Segoe UI" font-weight="700" fill="#c0332b">HX-202</text>'
                '<text x="40" y="290" font-size="20" font-family="Segoe UI" fill="#566176">control room</text></svg>')
    where = '<div class="ph">Where is it?</div><div class="svgbox">' + plan_svg + '</div><div class="chips"><span class="site-tag">left</span><span class="site-tag">entrance steps</span><span class="site-tag">west passage</span></div>'
    trust = ('<div class="ph">What the numbers hide</div><div class="big" style="font-size:72px">96%</div><div class="pk">missions say complete</div>'
             '<div class="row"><div class="w">most of them hide a failed check<small>docs/spot.md · computed from the rows</small></div></div>')
    send = (f'<div class="ph">Send SPOT back to heat exchanger 2?</div><div class="btnrow"><span class="site-btn" id="{s.id}-yes">Yes</span><span class="site-chip">Not now</span></div>'
            f'<div class="ph ok" id="{s.id}-done" style="margin-top:22px;opacity:0">Done · work order on this box</div>')
    run = ('<div class="ph">Shift Run</div><div class="pk">the trainee plans the next round</div><div class="big" style="margin-top:12px">★★☆</div>'
           '<div class="pk">check the far exchangers before the battery drops</div>'
           f'<div class="site-btn" id="{s.id}-handoff" style="margin-top:24px;opacity:0">Shift handed off ✓</div>')
    s.window([("Inspection", tri, "0:0.4"), ("Floor plan", where, "2:0.5"), ("Trust", trust, "3:0"), ("Approve", send, "4:0"), ("Shift Run", run, "5:0.6")],
             at="0:0.2", height=700)
    s.tween(f'tl.fromTo("#SPOTPATH", {{ strokeDasharray: 700, strokeDashoffset: 700 }}, {{ strokeDashoffset: 0, duration: 1.8, ease: "none" }}, {s.t("2:0.6"):.3f});')
    a = s.t("4:0.8")
    s.tween(f'tl.to("#{s.id}-yes", {{ scale: 0.94, duration: 0.12, yoyo: true, repeat: 1 }}, {a:.3f});')
    s.tween(f'tl.to("#{s.id}-done", {{ opacity: 1, duration: 0.3 }}, {a + 0.3:.3f});')
    s.tween(f'tl.to("#{s.id}-handoff", {{ opacity: 1, duration: 0.3 }}, {s.t("end-1.2"):.3f});')


BUILDERS = {"problem": problem, "morning": morning, "sources": sources, "model": model, "desk": desk, "faces": faces,
            "kitchen": kitchen, "sortit": sortit, "sorttech": sorttech, "lab": lab, "device": device, "league": league,
            "friends": friends, "lives": lives, "plant": plant, "shift": shift}


def main() -> int:
    film = json.loads((HF / "film.json").read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    for sc in film["scenes"]:
        if sc["id"] in BUILDERS and (not only or sc["id"] in only):
            s = Scene(sc)
            BUILDERS[sc["id"]](s)
            s.write()
            print(f"wrote compositions/{sc['id']}.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
