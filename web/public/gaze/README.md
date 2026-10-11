# Gaze files (served by this app, never a CDN)

The look guide's opt-in eye tracking ("Use my eyes to guide me", Settings) runs in the browser. Everything it loads comes
from this folder on the box; the page fetches nothing from another origin.

| File | What | Source | Licence |
| --- | --- | --- | --- |
| `face_landmarker.task` | MediaPipe Face Landmarker, float16, version 1 (face mesh with iris points). 3,758,596 bytes. sha256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | Apache 2.0 |
| `wasm/vision_wasm_internal.{js,wasm}`, `wasm/vision_wasm_nosimd_internal.{js,wasm}` | The MediaPipe Tasks runtime. Not committed (11 MB each): `npm run build` copies them from `node_modules/@mediapipe/tasks-vision/wasm/` (`scripts/fetch-gaze-model.mjs`, the prebuild step). | npm `@mediapipe/tasks-vision` **0.10.35** (`sha512-HOvadwVRE6JC+45nyYhmnywnr5h/J8KZvOeUNVOG9q/0875pZgItznFB9bRTvLc264YSJqiZ1NsIpCStJw/egg==`) | Apache 2.0 |

sha256 of the runtime files as copied from 0.10.35:

    e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c  vision_wasm_internal.js
    6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc  vision_wasm_internal.wasm
    438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296  vision_wasm_nosimd_internal.js
    8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31  vision_wasm_nosimd_internal.wasm

## Why 0.10.35 and not the newest

From 1.0.0 on (checked: 1.0.1 and 1.1.0), `vision_bundle.mjs` carries a usage logger that posts to
`https://odml.pa.googleapis.com/v1/log` once a minute while a task is alive. This app tells people "Nothing is stored or
sent", so the dependency is pinned to 0.10.35, the last release whose bundle contains no URL at all, and the prebuild
step stops the build if the installed bundle contains one. Do not bump the version without reading the bundle again.

## Getting the files

    node scripts/fetch-gaze-model.mjs            # copy the runtime from node_modules, check the model's checksum (prebuild runs this)
    node scripts/fetch-gaze-model.mjs --fetch    # also download the model if it is missing (needs the internet once; the checksum must match)

Without these files the app still works: the switch says the gaze model is not installed and the pointer starts from
the middle of the screen.

## What never happens

A camera frame is handed to the model and dropped. No frame, landmark or gaze point is uploaded, drawn to a canvas for
export, logged or stored. `localStorage` keeps the on/off choice and the calibration fit (a few weights and its measured
error) under `atlas.gaze`; nothing else.
