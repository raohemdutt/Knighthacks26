# AMD segment: run of show for the two live moments

Ilan's 2.5-minute AMD segment has two live screens: moment C (beat 5, the Learning lab) and moment B (beat 6, the
live power trace). The script is `work/demo/AMD_SCRIPT.md`; the slides are separate and not covered here. Everything
below was checked live from the workstation on Sun 2026-10-11 between 3:20 and 3:35 AM ET.

Box address: `100.124.34.36` (Tailscale, the `jetson` key of the workspace `.env`). The gateway serves the web app and
its API on port 8000. Fallback images are in `C:\Users\hackathon\Code\Knighthacks26\work\demo\`.

## Windows to have open, left to right

| Tab | URL | Zoom | Why |
|---|---|---|---|
| 1 | `http://100.124.34.36:8000/lab/` (Training run tab) | 150 percent | Moment C |
| 2 | `http://127.0.0.1:8787/` (live power page) | 150 percent | Moment B |
| 3 | `http://100.124.34.36:8000/lab/?tab=speed` (Speed and energy tab) | 150 percent | Fallback for C, already loaded |
| 4 | File Explorer on `work\demo\` | | The fallback images |

Load tabs 1 and 3 at least a minute before the segment. The web app shows its splash for about 10 seconds on a cold
load, and the Speed and energy tab waits about 16 seconds for its data (the bench endpoint answered in 16.1 s).

Chrome zoom: Ctrl+0, then Ctrl+plus four times (110, 125, 150). At 150 percent a 1920x1080 projector shows a
1280x720 page, which is what every screenshot here was taken at.

## Helper for moment B (start it first)

The power page needs a local helper: the pod's power sampler is only reachable through the hub proxy with the hub
token, which a browser page cannot send. The helper reads the token from the workspace `.env` itself and never
prints it.

```
python C:\Users\hackathon\Code\Knighthacks26\work\demo-amd\tooling\demo\power_proxy.py
```

It prints `power page: http://127.0.0.1:8787/` and keeps running. Leave that terminal open. It finds the `.env` by
walking up from its own folder. Options: `--port`, `--env <file>`, `--account C`, `--box <address>`.

What it does: serves `power.html`, forwards `/pod/samples` and `/pod/status` to pod C's sampler
(`lab/pod_power.py`, port 8790) with the token, forwards "Ask the box" to the gateway's `POST /api/ask`, and forwards
"Ask the teacher" to the teacher's `/v1/chat/completions` on pod C (streamed, thinking off, temperature 0, at most 96
tokens). The box lane needs no helper: the page opens the gateway's `/ws` directly, which already carries
`telemetry.jetson.power` (VDD_IN every 350 ms) and the question and answer events.

## Moment C: the Learning lab, Training run tab (beat 5, 0:25)

State at 3:30 AM: the real run is on the box. `GET /api/learning/lab` answers `sample: false`, `real_runs: 1`, run
`train-20261011T032149Z-7dd098`, status `deployed`; L27 (PR #382) has landed.

URL: `http://100.124.34.36:8000/lab/`, zoom 150 percent. At that zoom two cards fit on screen at a time, so scroll
slowly once from top to bottom during the ten seconds (the whole run page is about 3,300 px tall at this width).

What the audience sees, card by card:

1. Taught from: empty in this run ("No split to show", "No example rows"), because the box has no export or split
   for the dataset. Do not linger.
2. Training: 18 steps, loss 0.01 now, 11 min 57 s of training, the loss curve. No power trace (no pod-b samples in
   the box's log).
3. Before / after: 24 of 24 before, 24 of 24 after, 0 fixed, 0 regressions.
4. The gate: "Not proven: not deployed. The model on the box did NOT change." p = 1, 24 questions, run-to-run noise 0.
5. On the box: was serving atlas-base, serving since 2:42 AM the new model; energy per answer "not in the log".

Watch out: card 4 says the model did not change, while card 5 and the header say the fine-tuned model is serving. A
judge can read both. Say the line already in the script ("the gate refused the first student") and, if asked, "the
gate said no change, so what is on the box is as good as base".

Alert toasts ("Activity at the desk ... during sleep hours") can slide over the top of the page. Close them with the
x before the segment.

Fallback, in order:
1. The screenshots, taken at 150 percent: `lab-run-150pct-top.png` (header and cards 1 and 2),
   `lab-run-150pct-card3.png`, `lab-run-150pct-card4.png`, `lab-run-150pct-card5.png`, and
   `lab-run-150pct-full.png` (the whole page in one image).
2. Tab 3, the Speed and energy tab, already loaded; it shows real rows (teacher batching on pod-a, the A100
   reference). Screenshot: `lab-speed-150pct.png`.

## Moment B: the live power trace (beat 6, 0:25)

URL: `http://127.0.0.1:8787/`, zoom 150 percent. Two lanes: the box (Jetson whole board, VDD_IN, dashed idle line
15.76 W from T13) and the AMD teacher (pod C, Strix Halo APU socket power, dashed idle floor 10.27 W from T10). Each
lane scrolls 120 seconds. The yellow panel at the bottom right gives the energy of the last answer: the power
integrated over the answer's window, gross, as the telemetry context defines joules per answer.

The script's beat 6 talks about the AMD chip ("it waits at ten watts... that answer cost about X joules"), and the
box's own answers run on the Jetson, not on the pod. So the live action for beat 6 is "Ask the teacher":

- Type: `In one sentence, why does a smaller model on the box save energy?`
- Press: Ask the teacher.
- Expected answer (temperature 0, tested at 3:29 AM): "A smaller model saves energy because it requires fewer
  computational operations and less memory bandwidth to process data, leading to lower power consumption during
  inference." It took 5.2 s.
- What the audience sees: the orange lane sitting on the 10 W line, a jump to about 100 W inside a shaded band for
  about five seconds, back to 10 W, and the yellow line "That answer cost N J" with a smaller line under it
  ("On the AMD teacher, APU socket: 5.2 s at ... W average, ... samples, gross; ... J above its 10.27 W idle floor").
- The test value was 518 J, but the pod was already at about 108 W from someone else's work at that moment, so that
  figure is not this answer's alone; the page says so in the small line when the pod was busy before the question.
  At idle, expect roughly 5 to 8 seconds at about 96 W, so about 500 to 800 J. Read the number off the screen.
- Read it as low confidence: the pod sampler is 1 Hz, so a 5-second answer has about 5 samples.

Alternative if Ilan wants the box itself to answer (it takes longer, about 25 s):

- Type `What happened at our table overnight?` (fixture `golden-01`, `fixtures/learning/tool-calling-labels/`;
  the fine-tuned student got it right before and after in the gate, 24 of 24) and press Ask the box, or ask the box
  by voice: the page reacts to any question the box gets, typed, spoken or from the phone.
- Expected: the student calls `episodes_between` for last night and answers hour by hour from the box's own memory.
  Tested at 3:27 AM: "Sat 8:00 PM: A person in a white "ALGO" sweatshirt sat at a table ... Sun 3:00 AM: A laptop and
  water bottle sat on a wooden table in a classroom." Model `atlas-tool-calling-train-20261011T032149Z-7dd098`.
- Measured: 25.5 s at 18.1 W average, 71 samples, "That answer cost 463 J" (gross, whole board), 61 J above the
  15.76 W idle line. The blue lane goes from its usual 5 W / 18 W sawtooth (the box captions its camera by itself)
  to a flat 17 to 20 W block for the length of the answer.

Fallback: `power-answer-cost.png` (the box answer above, with the 463 J line), `power-ask-1.png` (mid-answer,
"so far 216 J", the shaded band growing), `power-idle-150pct.png` (both lanes idle at 150 percent, the teacher on
its 10 W floor). If the helper or the pod proxy is down, show `power-answer-cost.png` and say the number.

## 60-second rehearsal checklist (10:30 AM)

1. Helper terminal shows `power page: http://127.0.0.1:8787/` (start it with the command above if not).
2. Tab 2: both dots green, "box live" and "pod C sampler live"; the blue lane moves every second.
3. Orange lane sits near 10 W. If it sits near 100 W, someone is using the teacher: ask on the team channel to pause
   it from 10:55 to 11:10, or plan to use the box alternative.
4. Press Ask the teacher with the question above; the yellow line reads "That answer cost ... J" within about 10 s.
5. Tab 1: the header says "Deployed", the five cards load, no alert toast covering the tabs.
6. Tab 3 has finished loading (not "Loading the lab's measurements...").
7. Zoom is 150 percent on tabs 1 to 3; the screenshot folder is open in tab 4.

Do not restart the teacher, the box's gateway or its model server for any of this; nothing here needs it.
