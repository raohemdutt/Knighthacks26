# The submission video (D10, D12)

A programmed video in [Remotion](https://www.remotion.dev): React components rendered frame by frame to an MP4,
locally. It is 9:58 at 30 fps, exactly 17,940 frames in ten slots of 1,794, so YouTube's number keys (which jump to
tenths of the video) land at the start of each slot. Every judge watches slot 0, presses their track's key and watches
two slots. The plan and why: Notion, Team Comms → "Rubric self-check + submission video plan".

```
cd video
npm install
npm run timeline     # slots, tracks and clips from contexts/demo/video.py -> src/timeline.json
npm run totals       # measured hackathon totals from ../data's event log -> src/totals.json (D11)
npm run studio       # preview and scrub in the browser
npm run render       # out/atlas.mp4 (about 30 minutes on a laptop)
npm run check        # length within 0.1 s of 9:58, and where each key lands
```

## The three-minute film (HyperFrames)

`video/hf/` is the short cinematic cut, built with [HyperFrames](https://github.com/heygen-com/hyperframes): HTML,
GSAP and Three.js rendered by headless Chrome. One continuous 3D world (the box and its four layers, the privacy
membrane, the time ring on the desk, the lab, friends' boxes, the campus) under soft-UI overlays, ten scenes back to
back, no keys and no app recordings. `video/hf/WORLD.md` is the contract.

```
python video/scripts/film_build.py       # speaks video/hf/film.yaml, writes film.json + index.html (never edit index.html)
cd video/hf
npx hyperframes@0.8.145 lint             # 0 errors before anything else
npx hyperframes@0.8.145 snapshot --at 8,40,100   # frames at absolute seconds, read the PNGs
npx hyperframes@0.8.145 preview          # the browser editor (click anything, scrub)
npx hyperframes@0.8.145 render --quality delivery --output out/atlas-film.mp4
```

Words live in `film.yaml`; each scene is a choreography (`world/scenes/<id>.js`, numbers tweened on the root
timeline at absolute seconds from `film.json`) plus an overlay (`compositions/<id>.html`, a sub-composition in local
time). It renders natively on Windows, macOS and Linux (Node 22, FFmpeg); Docker is optional (`render --docker`).

## Narration and diagrams

- `script.yaml` is the script, one entry per slot, plain words. `python video/scripts/voice.py` speaks every line with
  a male neural voice (edge-tts, `en-US-AndrewMultilingualNeural`), caches the audio under `public/voice/` (not
  committed; the script needs the internet once per changed line) and writes `src/voice.json` with each line's start
  and length. It prints a fit report per slot and fails if a slot's narration would run past its 59.8 s.
- A slot with no recording shows its bird's-eye diagram from `src/scenes.ts` (nodes are soft cards, edges draw
  themselves, a pulse shows the flow) while the narrator speaks; captions follow the voice line by line.
- `python video/scripts/record_screens.py` records the phone app's screens from a gateway on the seeded demo night
  into `public/clips/slot-<k>.mp4` (slots 1, 8 and 9 today); see its docstring for the gateway command.

## Renders

`video/renders/` holds dated full renders for the team to watch (`atlas-draft-2026-10-10.mp4`: the design with slot 0
finished and placeholders in slots 1 to 9). `out/` stays local. The final upload is the newest file here once the
recordings are in.

## Adding a recording

Drop a screen or phone recording at `public/clips/slot-<key>.mp4` (1920x1080, 30 fps, at least 58 s), run
`npm run timeline` again, and that slot plays it after its title card instead of the placeholder. Recordings are
not committed (`.gitignore`); keep them in the team drive. What each slot must show is in `contexts/demo/video.py`
and on the placeholder itself.

## The look

Soft UI (neumorphism), minimal: one surface colour, depth only from a light to the top left and a shade to the
bottom right, one accent for the key a judge presses, four type sizes, one idea per screen. The tokens and the
`raised` / `pressed` / `press(t)` styles are in `src/theme.ts`; the pieces (`Key`, `Card`, `Well`, `Muted`) in
`src/Soft.tsx`. The 3D box in the intro uses the same surface colour and light direction, with a soft shadow texture
instead of shadow maps, so it reads as the same material as the cards. A recording dropped into a slot is shown as
is; keep the screens it shows on the box's own light theme where you can.

## Rules

- The slot order, titles and tracks live only in `contexts/demo/video.py`. Change them there, never in
  `src/timeline.json`.
- The intro's totals card shows only rows the event log measured (`src/totals.json`, empty by default: no card).
  Run `npm run totals` against the Jetson's real `data/` (or copy its `--json` output) before the final render.
- Charts and numbers come from measured files (the lab ledger, the eval JSON, `lab/slide_numbers.py`), never typed in.
- Slot 5 says "the on-device Qwen VL understands the scene".
- Remotion is free for individuals and small teams; read its license once before relying on it.
