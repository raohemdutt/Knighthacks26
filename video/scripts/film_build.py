"""Build the film: speak video/hf/film.yaml, lay the scenes out, write video/hf/index.html.

film.yaml is generated from SCRIPT.md by film_from_script.py. A line is either plain text (the narrator) or
{who, text, hold}: `who` picks the voice from `voices` and the caption's speaker name from `names`; `hold` is seconds
of silence after the line (action). Every scene gets safe line access: an old scene file that indexes a line the
scene no longer has gets a synthetic line inside the scene instead of killing the whole timeline, and every scene's
choreography runs in its own try/catch.

    python video/scripts/film_build.py            # synthesise what changed, write film.json + index.html, print the layout
    python video/scripts/film_build.py --check    # no writes: exit 1 if a scene file is missing or a cached line is
    python video/scripts/film_render.py day       # render one scene fast (15 fps draft), see film_render.py

Every line is spoken once (edge-tts, cached under video/public/voice/ by text hash, copied into video/hf/assets/voice/).
Scenes run back to back: start = previous end + pad_s; duration = max(min_s, lead_s + speech + gaps + tail_s).
video/hf/film.json holds the resolved layout (fps, total, scenes with start/duration/lines) and is what the scene
choreographies, the overlays and the tests read. index.html is the root composition: one world canvas, one host per
scene (compositions/<id>.html), the captions sub-composition and every spoken line as an <audio> clip. Never edit
index.html by hand.
"""

import argparse
import asyncio
import hashlib
import re
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "hf"
FILM = HF / "film.yaml"
OUT_JSON = HF / "film.json"
INDEX = HF / "index.html"
CACHE = ROOT / "public" / "voice"
ASSETS = HF / "assets" / "voice"
FPS = 30
GSAP = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"
THREE = "0.181.2"

OVERLAY_STUB = """<!doctype html>
<html>
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <link rel="stylesheet" href="world/soft.css" />
      <style>
        #root {{ position: absolute; inset: 0; font-family: var(--font); }}
        #{id}-chapter {{ position: absolute; left: 96px; top: 72px; padding: 14px 28px; font-size: 28px; opacity: 0; }}
      </style>
      <div id="root" data-composition-id="{id}" data-width="1920" data-height="1080">
        <div id="{id}-chapter" class="soft-raised soft-muted">{title}</div>
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
          tl.fromTo("#{id}-chapter", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6, ease: "sine.inOut" }}, 0.2); tl.fromTo("#{id}-chapter", {{ y: -12 }}, {{ y: 0, duration: 0.6, ease: "power3.out" }}, 0.2);
          window.__timelines["{id}"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""

CHOREO_STUB = """// Scene "{id}": {title}. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = {{ id, title, start, duration, lines: [{{ text, start, duration }}] }} with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {{
  const {{ cam, state }} = world, t0 = scene.start;
  world.leg(tl, t0, {{ x: 2.0, y: 2.0, z: 7.8, fov: 34 }}, scene.duration * 0.9, "none");
}}
"""

CAPTIONS = """<!doctype html>
<html>
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <link rel="stylesheet" href="world/soft.css" />
      <style>
        #root {{ position: absolute; inset: 0; pointer-events: none; }}
        .cap {{ position: absolute; left: 0; right: 0; bottom: 64px; display: flex; justify-content: center; }}
        .cap-card {{ padding: 22px 40px; max-width: 1560px; opacity: 0; }}
        .cap-text {{ font-size: 32px; text-align: center; }}
        .cap-who {{ font-weight: 700; color: var(--accent); margin-right: 14px; }}
      </style>
      <div id="root" data-composition-id="captions" data-width="1920" data-height="1080">
{clips}
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
{tweens}
          window.__timelines["captions"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


SAFE = r"""      // Safe line access: an index past the scene's lines gets a synthetic line inside the scene, never a throw.
      const safeScene = (s) => {
        const real = s.lines, last = real.length ? real[real.length - 1] : null;
        const base = last ? last.start + last.duration : s.start, room = Math.max(0.5, s.start + s.duration - base);
        const lines = new Proxy(real, { get(arr, k) {
          if (typeof k === "string" && /^\d+$/.test(k) && +k >= arr.length) {
            const j = +k - arr.length + 1;
            return { n: +k, who: "N", name: "", text: "", start: base + room * 0.9 * (1 - Math.pow(0.5, j)), duration: room * 0.9 * Math.pow(0.5, j + 1) };
          }
          return arr[k];
        } });
        return { ...s, lines };
      };
      const run = (fn, id) => {
        try { fn(tl, world, safeScene(film.scenes.find((s) => s.id === id))); }
        catch (e) { console.error("scene " + id + " failed", e); }
      };"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def caption(sid: str, l: dict, at: float, cd: float) -> str:
    who = f'<span class="cap-who">{esc(l["name"])}</span>' if l.get("name") else ""
    return (f'        <div class="clip cap" id="cap-{sid}-{l["n"]}" data-start="{at:.3f}" data-duration="{cd:.3f}" data-track-index="0">'
            f'<div class="cap-card soft-raised"><div class="cap-text soft-body">{who}{esc(l["text"])}</div></div></div>')


def cap_tweens(sid: str, l: dict, at: float, cd: float) -> list[str]:
    n = l["n"]
    return [f'          tl.fromTo("#cap-{sid}-{n} .cap-card", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.35, ease: "sine.inOut" }}, {at:.3f}); tl.fromTo("#cap-{sid}-{n} .cap-card", {{ y: 12 }}, {{ y: 0, duration: 0.3, ease: "power2.out" }}, {at:.3f});',
            f'          tl.to("#cap-{sid}-{n} .cap-card", {{ opacity: 0, duration: 0.3, ease: "power2.in" }}, {at + cd - 0.32:.3f});']


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


async def speak(text: str, voice: str, rate: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))



# How the voice says a word that it would otherwise misread (the caption keeps the written word).
PRONOUNCE = {r"YOLO": "Yoh-loh"}


def say(text: str) -> str:
    for pattern, spoken in PRONOUNCE.items():
        text = re.sub(pattern, spoken, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scene-previews", action="store_true", help="also write scene-<id>.html (film_render.py does this and removes them)")
    args = ap.parse_args()
    cfg = yaml.safe_load(FILM.read_text(encoding="utf-8"))
    rate = cfg.get("rate", "+0%")
    voices = cfg.get("voices") or {"N": cfg["voice"]}
    names = cfg.get("names") or {}
    gap, pad, tail = float(cfg["gap_s"]), float(cfg["pad_s"]), float(cfg["tail_s"])
    CACHE.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    problems, scenes, t = [], [], 0.0
    for sc in cfg["scenes"]:
        sid, title = sc["id"], sc["title"]
        for path, stub in ((HF / "compositions" / f"{sid}.html", OVERLAY_STUB), (HF / "world" / "scenes" / f"{sid}.js", CHOREO_STUB)):
            if not path.exists():
                problems.append(f"missing {path.relative_to(HF).as_posix()}")
                if not args.check:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(stub.format(id=sid, title=title), encoding="utf-8")
        start, cursor = t, t + float(sc.get("lead_s", 0.0))
        lines = []
        for n, raw in enumerate(sc.get("lines", [])):
            line = raw if isinstance(raw, dict) else {"who": "N", "text": raw}
            who, text, hold = line.get("who", "N"), line["text"].strip(), float(line.get("hold", 0.0))
            voice = voices.get(who, voices["N"])
            spoken = say(text)  # the words as the voice should say them; the caption keeps the written form
            digest = hashlib.sha1(f"{voice}|{rate}|{spoken}".encode()).hexdigest()[:10]
            name = f"{sid}-{n}-{digest}.mp3"
            cached = CACHE / name
            if not cached.exists():
                legacy = [p for p in CACHE.glob(f"*-{digest}.mp3")]  # the same words spoken for the ten-slot video
                if legacy:
                    shutil.copy(legacy[0], cached)
                elif args.check:
                    continue  # CI has no voice cache; --check verifies the scene files and that film.json matches the words
                else:
                    asyncio.run(speak(spoken, voice, rate, cached))
            if not (ASSETS / name).exists():
                shutil.copy(cached, ASSETS / name)
            if args.check:
                continue
            d = duration(cached)
            lines.append({"n": n, "who": who, "name": names.get(who, ""), "text": text, "start": round(cursor, 3),
                          "duration": round(d, 3), "file": f"assets/voice/{name}"})
            cursor += d + gap + hold
        speech_end = (cursor - gap) if lines else start + float(sc.get("lead_s", 0.0))
        dur = max(float(sc.get("min_s", 0)), speech_end - start + tail)
        scenes.append({"id": sid, "title": title, "start": round(start, 3), "duration": round(dur, 3), "lines": lines,
                       "speech_s": round(sum(l["duration"] for l in lines), 1)})
        t = start + dur + pad
    total = round(t - pad, 1)
    for sc in scenes:
        print(f"{sc['id']:<8} {sc['title']:<18} starts {sc['start']:6.1f}  lasts {sc['duration']:5.1f} s  speech {sc['speech_s']:5.1f} s")
    print(f"total {total:.1f} s ({int(total // 60)}:{total % 60:04.1f})")
    for p in problems:
        print("  " + p)
    if args.check:
        committed = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {"scenes": []}
        said = [[l["text"] for l in sc["lines"]] for sc in committed["scenes"]]
        want = [[(t["text"] if isinstance(t, dict) else t).strip() for t in sc.get("lines", [])] for sc in cfg["scenes"]]
        if said != want:
            problems.append("film.json is stale: run python video/scripts/film_build.py")
        return 1 if problems else 0
    for stale in ASSETS.glob("*.mp3"):
        if not any(l["file"].endswith(stale.name) for sc in scenes for l in sc["lines"]):
            stale.unlink()
    OUT_JSON.write_text(json.dumps({"fps": FPS, "total_s": total, "scenes": scenes}, indent=2) + "\n", encoding="utf-8")

    hosts, audio, caps, tweens, imports, calls = [], [], [], [], [], []
    for sc in scenes:
        sid = sc["id"]
        hosts.append(f'      <div id="host-{sid}" data-composition-id="{sid}" data-composition-src="compositions/{sid}.html" '
                     f'data-start="{sc["start"]:.3f}" data-duration="{sc["duration"]:.3f}" data-track-index="2" data-width="1920" data-height="1080"></div>')
        imports.append(f'      import {sid} from "./world/scenes/{sid}.js";')
        calls.append(f"        run({sid}, \"{sid}\");")
        for l in sc["lines"]:
            at, d, n = l["start"], l["duration"], l["n"]
            audio.append(f'      <audio id="voice-{sid}-{n}" src="{l["file"]}" data-start="{at:.3f}" data-duration="{d:.3f}" data-track-index="3" data-volume="1"></audio>')
            cd = d + 0.25  # less than the 0.3 s gap, so two captions never overlap
            caps.append(caption(sid, l, at, cd))
            tweens.extend(cap_tweens(sid, l, at, cd))
    (HF / "compositions" / "captions.html").write_text(CAPTIONS.format(clips=chr(10).join(caps), tweens=chr(10).join(tweens)), encoding="utf-8")
    nl = chr(10)
    html = f"""<!doctype html>
<!-- GENERATED by video/scripts/film_build.py from video/hf/film.yaml. Do not edit; change film.yaml or the scene files. -->
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="{GSAP}"></script>
    <script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@{THREE}/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@{THREE}/examples/jsm/"}}}}</script>
    <link rel="stylesheet" href="world/soft.css" />
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ width: 1920px; height: 1080px; overflow: hidden; background: var(--surface); }}
      #root {{ position: relative; width: 100%; height: 100%; overflow: hidden; background: var(--surface); }}
      #world {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; }}
      #root > div[data-composition-src] {{ position: absolute; inset: 0; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="atlas" data-start="0" data-duration="{total:.1f}" data-fps="{FPS}" data-width="1920" data-height="1080">
      <canvas id="world" class="clip" data-start="0" data-duration="{total:.1f}" data-track-index="0" width="1920" height="1080"></canvas>
      <div class="soft-grain"></div>
{nl.join(hosts)}
      <div id="host-captions" data-track-kind="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="{total:.1f}" data-track-index="4" data-width="1920" data-height="1080"></div>
{nl.join(audio)}
    </div>
    <script>
      window.__hf = window.__hf || {{}}; window.__hf.buildReady = window.__hf.buildReady || {{}};
      window.__hf.buildReady.world = new Promise((resolve) => {{ window.__worldReady = resolve; }});
    </script>
    <script type="module">
      import {{ createWorld }} from "./world/world.js";
      import film from "./film.json" with {{ type: "json" }};
{nl.join(imports)}
      const world = await createWorld(document.getElementById("world"));
      window.__world = world;
{SAFE}
      const tl = gsap.timeline({{ paused: true }});
      const clock = {{ t: 0 }};
      tl.to(clock, {{ t: {total:.1f}, duration: {total:.1f}, ease: "none", onUpdate: () => world.render(clock.t) }}, 0);
{nl.join(calls)}
      window.__timelines["atlas"] = tl;
      tl.seek(0); world.render(0);
      if (window.__hfForceTimelineRebind) window.__hfForceTimelineRebind();
      window.__worldReady();
    </script>
  </body>
</html>
"""
    INDEX.write_text(html, encoding="utf-8")
    # One preview file per scene (scene-<id>.html): the same world and the same timeline, played only over that
    # scene, its overlay and its voice at local time. `hyperframes render -c scene-<id>.html` renders 20 s, not 3 min.
    # No captions in previews (the captions file is laid out on the film's absolute clock).
    # They exist only while film_render.py renders: a second root file trips lint (multiple_root_compositions).
    for old in [*HF.glob("scene-*.html"), *(HF / "compositions").glob("_cap-*.html")]:
        old.unlink()
    for k, sc in enumerate(scenes if args.scene_previews else []):
        sid, s0 = sc["id"], sc["start"]
        d = (scenes[k + 1]["start"] - s0) if k + 1 < len(scenes) else sc["duration"]  # up to the next scene: the pad too
        cid = f"_cap-{sid}"
        lc, lt = [], []
        for l in sc["lines"]:
            lc.append(caption(sid, l, l["start"] - s0, l["duration"] + 0.25))
            lt.extend(cap_tweens(sid, l, l["start"] - s0, l["duration"] + 0.25))
        (HF / "compositions" / f"{cid}.html").write_text(
            CAPTIONS.format(clips=nl.join(lc), tweens=nl.join(lt)).replace('"captions"', f'"{cid}"'), encoding="utf-8")
        host = (f'      <div id="host-{sid}" data-composition-id="{sid}" data-composition-src="compositions/{sid}.html" '
                f'data-start="0" data-duration="{d:.3f}" data-track-index="2" data-width="1920" data-height="1080"></div>')
        voices = [f'      <audio id="voice-{sid}-{l["n"]}" src="{l["file"]}" data-start="{l["start"] - s0:.3f}" '
                  f'data-duration="{l["duration"]:.3f}" data-track-index="3" data-volume="1"></audio>' for l in sc["lines"]]
        page = html.split("<body>")[0] + f"""<body>
    <div id="root" data-composition-id="atlas" data-start="0" data-duration="{d:.3f}" data-fps="{FPS}" data-width="1920" data-height="1080">
      <canvas id="world" class="clip" data-start="0" data-duration="{d:.3f}" data-track-index="0" width="1920" height="1080"></canvas>
      <div class="soft-grain"></div>
{host}
      <div id="host-{cid}" data-track-kind="captions" data-composition-id="{cid}" data-composition-src="compositions/{cid}.html" data-start="0" data-duration="{d:.3f}" data-track-index="4" data-width="1920" data-height="1080"></div>
{nl.join(voices)}
    </div>
    <script>
      window.__hf = window.__hf || {{}}; window.__hf.buildReady = window.__hf.buildReady || {{}};
      window.__hf.buildReady.world = new Promise((resolve) => {{ window.__worldReady = resolve; }});
    </script>
    <script type="module">
      import {{ createWorld }} from "./world/world.js";
      import film from "./film.json" with {{ type: "json" }};
{nl.join(imports)}
      const world = await createWorld(document.getElementById("world"));
      window.__world = world;
{SAFE.replace("(tl, world", "(full, world")}
      const full = gsap.timeline({{ paused: true }});
      const clock = {{ t: 0 }};
      full.to(clock, {{ t: {total:.1f}, duration: {total:.1f}, ease: "none", onUpdate: () => world.render(clock.t) }}, 0);
{nl.join(calls).replace("(tl, world", "(full, world")}
      const tl = gsap.timeline({{ paused: true }});
      tl.add(full.tweenFromTo({s0:.3f}, {s0 + d:.3f}, {{ ease: "none", immediateRender: false }}), 0);
      window.__timelines["atlas"] = tl;
      tl.seek(0); full.seek({s0:.3f}); world.render({s0:.3f});
      if (window.__hfForceTimelineRebind) window.__hfForceTimelineRebind();
      window.__worldReady();
    </script>
  </body>
</html>
"""
        (HF / f"scene-{sid}.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT.parent)} and {INDEX.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
