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
