"""Assemble the HyperFrames project (video/hf/index.html) from the slot layout and the narration timings.

    python video/scripts/hf_build.py            # write index.html (+ stub slots for anything missing)
    python video/scripts/hf_build.py --check    # only report what is missing

Inputs: video/hf/timeline.json (python -m contexts.demo video json), video/src/voice.json (video/scripts/voice.py),
video/hf/world/slots/slot-N.js and video/hf/compositions/slot-N.html (hand-written per slot, see video/hf/WORLD.md).
The root composition holds the one world canvas, the ten slot overlays, every spoken line as an <audio> clip at its
absolute second, and a caption clip per line. The root GSAP timeline is built in a module script: it drives the world
clock and calls every slot's choreography at that slot's absolute start. Nothing here is edited by hand.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "hf"
TIMELINE = HF / "timeline.json"
VOICE = ROOT / "src" / "voice.json"
INDEX = HF / "index.html"
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
        #slot-{k}-title {{ position: absolute; left: 160px; top: 120px; display: flex; align-items: center; gap: 48px; }}
        #slot-{k}-key {{ transform-origin: center; }}
        #slot-{k}-name {{ opacity: 0; }}
      </style>
      <div id="root" data-composition-id="slot-{k}" data-width="1920" data-height="1080">
        <div id="slot-{k}-title">
          <div id="slot-{k}-key" class="soft-key">{k}</div>
          <div id="slot-{k}-name" class="soft-title">{title}</div>
        </div>
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
          tl.fromTo("#slot-{k}-key", {{ boxShadow: "18px 18px 36px #b7c0cd, -18px -18px 36px #ffffff" }},
            {{ boxShadow: "inset 10px 10px 20px #b7c0cd, inset -10px -10px 20px #ffffff", duration: 0.35, ease: "power2.in" }}, 0.1);
          tl.fromTo("#slot-{k}-name", {{ opacity: 0, x: -24 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power3.out" }}, 0.3);
          tl.to("#slot-{k}-name", {{ opacity: 0, duration: 0.5, ease: "power2.in" }}, 6.0);
          tl.to("#slot-{k}-key", {{ scale: 0.6, x: -40, y: -40, duration: 0.6, ease: "power2.inOut" }}, 6.0);
          window.__timelines["slot-{k}"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
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

CHOREO_STUB = """// Slot {k}: {title}. Camera + world choreography on the ROOT timeline; every time is t0 + seconds. See WORLD.md.
export default function choreograph(tl, world, t0) {{
  const {{ cam, state, HOME }} = world;
  tl.set(cam, {{ ...HOME }}, t0);
  world.leg(tl, t0 + 1.5, {{ x: 2.5, y: 2.0, z: 7.5, fov: 34 }}, 4, "power2.inOut");
  tl.to(state, {{ spin: 0.12, duration: 20, ease: "none" }}, t0 + 1.5);
  world.leg(tl, t0 + 52, {{ ...HOME }}, 4, "power2.inOut");
  tl.to(state, {{ spin: 0, duration: 4, ease: "power2.inOut" }}, t0 + 54);
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tl = json.loads(TIMELINE.read_text(encoding="utf-8"))
    voice = json.loads(VOICE.read_text(encoding="utf-8"))["slots"]
    fps, slot_frames, total_frames = tl["fps"], tl["slot_frames"], tl["duration_frames"]
    slot_s, total_s = slot_frames / fps, total_frames / fps
    missing = []
    hosts, audio, captions, imports, calls, cap_tweens = [], [], [], [], [], []
    for slot in tl["slots"]:
        k, title = slot["key"], slot["title"]
        start = slot["start_frame"] / fps
        overlay, choreo = HF / "compositions" / f"slot-{k}.html", HF / "world" / "slots" / f"slot-{k}.js"
        for path, stub in ((overlay, OVERLAY_STUB), (choreo, CHOREO_STUB)):
            if not path.exists():
                missing.append(path.relative_to(HF).as_posix())
                if not args.check:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(stub.format(k=k, title=title), encoding="utf-8")
        hosts.append(f'      <div id="host-slot-{k}" data-composition-id="slot-{k}" data-composition-src="compositions/slot-{k}.html" '
                     f'data-start="{start:.3f}" data-duration="{slot_s:.3f}" data-track-index="2" data-width="1920" data-height="1080"></div>')
        imports.append(f'      import slot{k} from "./world/slots/slot-{k}.js";')
        calls.append(f"        slot{k}(tl, world, {start:.3f});")
        for line in voice[str(k)]["lines"]:
            at, dur, n = start + line["start"], line["duration"], line["n"]
            src = "assets/" + line["file"]
            audio.append(f'      <audio id="voice-{k}-{n}" src="{src}" data-start="{at:.3f}" data-duration="{dur:.3f}" data-track-index="3" data-volume="1"></audio>')
            cap_dur = dur + 0.5
            text = line["text"].replace("&", "&amp;").replace("<", "&lt;")
            captions.append(f'        <div class="clip cap" id="cap-{k}-{n}" data-start="{at:.3f}" data-duration="{cap_dur:.3f}" data-track-index="0">'
                            f'<div class="cap-card soft-raised"><div class="cap-text soft-body">{text}</div></div></div>')
            cap_tweens.append(f'        tl.fromTo("#cap-{k}-{n} .cap-card", {{ opacity: 0, y: 12 }}, {{ opacity: 1, y: 0, duration: 0.3, ease: "power2.out" }}, {at:.3f});')
            cap_tweens.append(f'        tl.to("#cap-{k}-{n} .cap-card", {{ opacity: 0, duration: 0.3, ease: "power2.in" }}, {at + cap_dur - 0.32:.3f});')
    if args.check:
        print("\n".join(missing) if missing else "all slots present")
        return 1 if missing else 0

    html = f"""<!doctype html>
<!-- GENERATED by video/scripts/hf_build.py from video/hf/timeline.json and video/src/voice.json. Do not edit. -->
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
    <div id="root" data-composition-id="atlas" data-start="0" data-duration="{total_s:.1f}" data-fps="{fps}" data-width="1920" data-height="1080">
      <canvas id="world" class="clip" data-start="0" data-duration="{total_s:.1f}" data-track-index="0" width="1920" height="1080"></canvas>
      <div class="soft-grain"></div>
{chr(10).join(hosts)}
      <div id="host-captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="{total_s:.1f}" data-track-index="4" data-width="1920" data-height="1080"></div>
{chr(10).join(audio)}
    </div>
    <script>
      window.__hf = window.__hf || {{}}; window.__hf.buildReady = window.__hf.buildReady || {{}};
      window.__hf.buildReady.world = new Promise((resolve) => {{ window.__worldReady = resolve; }});
    </script>
    <script type="module">
      import {{ createWorld }} from "./world/world.js";
{chr(10).join(imports)}
      const world = await createWorld(document.getElementById("world"));
      window.__world = world;
      const tl = gsap.timeline({{ paused: true }});
      const clock = {{ t: 0 }};
      tl.to(clock, {{ t: {total_s:.1f}, duration: {total_s:.1f}, ease: "none", onUpdate: () => world.render(clock.t) }}, 0);
{chr(10).join(calls)}
      window.__timelines["atlas"] = tl;
      tl.seek(0); world.render(0);
      if (window.__hfForceTimelineRebind) window.__hfForceTimelineRebind();
      window.__worldReady();
    </script>
  </body>
</html>
"""
    (HF / "compositions" / "captions.html").write_text(CAPTIONS.format(clips=chr(10).join(captions), tweens=chr(10).join(cap_tweens)), encoding="utf-8")
    INDEX.write_text(html, encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT.parent)}: {len(hosts)} slots, {len(audio)} spoken lines, {total_s:.1f} s")
    if missing:
        print("stubs written for: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
