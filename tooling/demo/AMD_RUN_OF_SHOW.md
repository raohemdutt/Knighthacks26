# AMD segment: run of show for the two live moments

Ilan's 2.5-minute AMD segment has two live screens: moment C (beat 5, the Learning lab) and moment B (beat 6, the
Shared brain card on the Circle screen). The script is `work/demo/AMD_SCRIPT.md`; the slides are separate and not
covered here. Everything below was checked live from the workstation on Sun 2026-10-11 between 3:20 and 3:50 AM ET.

Box address: `100.124.34.36` (Tailscale, the `jetson` key of the workspace `.env`). The box's gateway serves the web
app and its API on port 8000. Fallbacks are in `C:\Users\hackathon\Code\Knighthacks26\work\demo\`.

## Before 10:30: put the Shared brain card on the box (morning operator, about 5 minutes)

Moment B needs PR #389 (`ctx/circle/shared-brain-card`, "G4: Shared brain card on the Circle") running on the box.
The box's repo copy (`~/Knighthacks26`) is not a git repo: code reaches it as a tar over ssh (`jetson/push.sh`), and
the gateway serves the static web export in `~/Knighthacks26/web/out` at `:8000`. Do not use `jetson/push.sh` for
this: it replaces the whole tree, and the box's gateway files differ from main right now (about 20 files differ; for
example the outbound gate of PR #381 is not on it). The PR carries a script that changes only what the card needs.

On the workstation, in Git Bash:

```
cd /c/Users/hackathon/Code/Knighthacks26/work/G4b
git pull                                   # the PR branch, ctx/circle/shared-brain-card
tooling/demo/deploy_shared_brain.sh        # builds web/out, copies it and brain_power.py, patches the wiring,
                                           # adds the account C keys to the box .env if missing; restarts nothing
```

It prints the backup folder (`~/shared-brain-backup-<time>` on the box) and ends with `account C keys in the box
.env: 3 of 3`. The new Circle screen is served at once. The routes need the gateway restarted, which needs sudo
(Xander, or whoever holds the box password) and is not done by an agent:

```
ssh -t alexander@100.124.34.36 sudo systemctl restart atlas-gateway
```

The gateway replays its log on start; wait until `curl -s http://100.124.34.36:8000/health` answers (up to two
minutes), then check the route:

```
curl -s http://100.124.34.36:8000/api/circle/brain/power | head -c 300
```

It should start with `{"configured":true,"node":"pod-a","host":"sjc.aupcloud.io"` and carry `"error":null`. If it
says `"configured":false`, its `error` names the missing key. Rollback: copy `app.py` and `bootstrap.py` back from
the backup folder into `~/Knighthacks26/contexts/gateway/`, move its `web-out` back to `~/Knighthacks26/web/out`, and
restart the gateway the same way.

## Windows to have open, left to right

| Tab | URL | Zoom | Why |
|---|---|---|---|
| 1 | `http://100.124.34.36:8000/lab/` (Training run tab) | 150 percent | Moment C |
| 2 | `http://100.124.34.36:8000/circle/` (People tab, Shared brain card at the top) | 150 percent | Moment B |
| 3 | `http://100.124.34.36:8000/lab/?tab=speed` (Speed and energy tab) | 150 percent | Fallback for C, already loaded |
| 4 | `work\demo\circle-brain-clip.webm` dragged into a Chrome tab, paused at 0:00 | 150 percent | Fallback for B |
| 5 | File Explorer on `work\demo\` | | The fallback images |

Load tabs 1 to 3 at least a minute before the segment. The web app shows its splash for about 10 seconds on a cold
load, and the Speed and energy tab waits about 16 seconds for its data (the bench endpoint answered in 16.1 s).

Chrome zoom: Ctrl+0, then Ctrl+plus four times (110, 125, 150). At 150 percent a 1920x1080 projector shows a
1280x720 page, which is what every screenshot and the clip here were taken at.

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

## Moment B: the Shared brain card on the Circle screen (beat 6, 0:25)

URL: `http://100.124.34.36:8000/circle/`, zoom 150 percent, People tab. The card sits at the top under "Shared
brain": "Runs on the teacher on the lab's AMD Strix Halo pod (Radeon 8060S), through the hub (sjc.aupcloud.io)", two
numbers (friends' boxes on the network, and the brain's watts right now with the 10.27 W idle floor), a two-minute
scrolling trace with the idle floor dashed, the line about the last answer, and an ask box.

The action: type the question in the card's box and press "Ask the brain". The first ask on the box also turns on
this member's opt-in for the group brain (the card says so under the box); that is one event in the box's log.

- Question (recommended): `Dominick and Hemdutt are both at Our table until noon. Where should we meet at 11?`
  The group brain answers only from the facts it is given (the places friends share, plus what the question says),
  so a question that carries its own facts gets a direct answer, expected along the lines of "Meet at Our table at
  11 AM." This exact wording was not sent tonight; the teacher answered tonight's questions in 2 to 4 seconds.
- Tested at 3:43 AM: `Maya is at the Library until 9 PM and Sam is at the Gym until 8 PM. Where should we meet
  tonight?` The answer was "The facts state Maya is at the Library until 9 PM and Sam is at the Gym until 8 PM. They
  do not specify a mutual meeting location or time. Therefore, the facts do not settle where you should meet." So
  avoid questions whose facts do not settle the answer.
- What the audience sees: the trace sits on the dashed 10 W floor, jumps to about 100 W in a shaded band for 2 to 4
  seconds, falls back, and the line changes to "that answer cost N J". Measured tonight from the card: 2.0 s, 121 J
  (100 J above idle), and 4.2 s at 84 W average, 353 J (310 J above idle). The small line under it gives the
  seconds, the average watts, gross, the joules above the idle floor, and "Low confidence: N samples in the window
  (the sampler reads once a second)". Read the number off the screen.
- The line shows the last answer until a new one is measured, so after the rehearsal the card shows the
  rehearsal's number. That is expected; the new number replaces it about two seconds after the answer.
- If the pod sits well above 10 W before the question (someone else using the teacher), the figure includes their
  work and the small line says so. See the rehearsal checklist.

Fallback: tab 4, the recorded clip `work\demo\circle-brain-clip.webm` (1280x720, under a minute, recorded at 3:43 AM
against the real pod C sampler from this branch running locally: the card at idle, the question typed, the trace
jumping to 100 W, "that answer cost 353 J"). Stills: `circle-brain-clip-end.png` (the 353 J line),
`circle-brain-answer.png` (the 121 J line), `circle-brain-idle.png` (the card at idle). If the card is not on the box
but the pod is up, the standalone page from branch `ctx/docs/amd-demo-setup` still runs on the workstation
(`python work/demo-amd/tooling/demo/power_proxy.py`, then `http://127.0.0.1:8787/`).

## 60-second rehearsal checklist (10:30 AM)

1. `curl -s http://100.124.34.36:8000/api/circle/brain/power | head -c 200` says `"configured":true` and
   `"error":null` (if not, the operator steps above are not finished).
2. Tab 2: the Shared brain card shows watts that change every second; the trace sits near the dashed 10 W line.
   If it sits near 100 W, someone is using the teacher: ask on the team channel to pause it from 10:55 to 11:10.
3. Ask the recommended question once; the line reads "that answer cost ... J" within about 5 s.
4. Tab 1: the header says "Deployed", the five cards load, no alert toast covering the tabs.
5. Tab 3 has finished loading (not "Loading the lab's measurements...").
6. Tab 4 plays the clip; pause it at 0:00.
7. Zoom is 150 percent on tabs 1 to 4.

Do not restart the teacher or the box's model server for any of this; the only restart is the operator's gateway
restart above, once, before 10:30.
