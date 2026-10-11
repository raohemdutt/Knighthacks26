# Refinement goal for the plumbing track

Created 2026-10-10 (Task Board O3). This file says what the plumbing track is judged on: pod control on the two
AUP Learning Cloud accounts (one R9700 each), the box-to-lab link through the hub proxy, and credits. A pass that
does not follow it is invalid and its edits are reverted. How passes run (triggers, rounds, the cap, the merge
pass) is set by `lab/refine/PASS_RULES.md`, and the file formats by `lab/refine/TEMPLATES.md`. Where PASS_RULES.md
and this file disagree about process, PASS_RULES.md wins; where they disagree about what the track is judged on,
this file wins (PASS_RULES.md, opening section).

## The goal

Objective: pods that are reliable, cheap, and reachable with no hand steps (O3 task page).

Refine the plumbing track (the board tasks it covers, the experiments queued on T0 for its metrics, and what they
produce) until all eight plumbing tests below pass, each backed by a measured number in the metrics file or the
ledger. A run ends at convergence, at its tenth pass, or when Ilan ends it (PASS_RULES.md, "Runs, the cap and
convergence"). Convergence = a pass that walks every task and queued experiment of the track, and every failure
scenario in the walk list, against all eight tests and changes nothing: no board edit, no new queued experiment,
and no new ruling.

How passes are run is set by PASS_RULES.md ("How passes are run", "When a pass is triggered", "The merge pass"):
each pass is a fresh Opus agent on its own board task, started through the open-ticket skill; passes on this
track run strictly in sequence, the four tracks run in parallel in rounds, and a Fable merge pass follows each
round. A pass edits only the board rows of this track that are Ilan's alone, not Locked, and not Done; queues
experiments on T0; and writes only `REFINE_LOG.md`, `refine_pass.txt`, and `pass_reports/` in this folder and the
notes under `lab/refine/research/plumbing/`. It never writes `OWNER_RULINGS.md` or this file: it raises a ruling
in its log entry, and the merge pass copies it into `OWNER_RULINGS.md`.

## The two universal rules

These bind every pass in every track, in the words PASS_RULES.md gives them ("The two universal rules"), which
every rubric repeats.

1. Old versus new. Every task and experiment names the metrics it is meant to move, reads their current values
   from the metrics file or ledger before starting, and ends by measuring them the same way. The pass entry shows
   the values before and after and the conditions they were measured under. A change that moves nothing is not an
   improvement. A regression is reverted, or it goes to a ruling with the trade-off stated. The loop itself is
   judged by whether the track's numbers moved across passes.
2. Source-forced. Every change to a method or a plan item cites at least one source next to its before and after.
   Each pass starts with a bounded research step on the track's open problems, written into
   `lab/refine/research/plumbing/` with citations. A source-backed idea that has not been tested becomes a queued
   experiment with its citation and expected movement. A change that depends on a fact no source in hand
   establishes is not made; the fact becomes a queued measurement or a ruling.

A pass entry with no metrics-before block, or with any change that has no citation, is invalid and is reverted
(PASS_RULES.md, "Invalid passes").

For this track, "measured the same way" means the same node (this machine, the Jetson, pod A, or pod B), the same
account, the same port, and the same payload size; the conditions in the log name all four.

## Primary sources (the ONLY things that may justify a change)

In the order PASS_RULES.md sets for every track ("What a pass reads"). When two disagree, the higher one wins and
the disagreement is recorded in the log, never averaged.

1. A filled ruling from Ilan: an `Answer:` line in `OWNER_RULINGS.md` in this folder (a blank one is not a rule).
   The owner rules that bind every pass sit at the same level: the GPU Queue rules (the GPU Queue database and the
   open-ticket skill, section 4b) and `AGENTS.md`.
2. The Design Doc for this track's ground: sections 4 (system overview), 6 and 6b (integration rules and failure
   modes), 8 (network and GPU server lifecycle), and 13 (risks).
3. The track's research notes in `lab/refine/research/plumbing/` with their `SOURCES.md`, ranked inside by label
   (`lab/refine/RESEARCH_FORMAT.md`). For this track a finding measured on our hub (our own run, citing its ledger
   line, PR, or verified task-page line) outranks a prediction read from source code or docs about the same
   behavior, because the public AUP repo shows a reference deployment, not the Dublin hub's settings. Source code
   is cited at a pinned tag or commit: JupyterHub 5.4.4 (the version our hub reports at `GET /hub/api/`), the AUP
   repo at commit 9b4af19, vLLM b1388b1 (the commit pod A serves, `lab/pod_a_teacher.sh` L11), and, until the
   pod's own versions are confirmed (open item 9), the newest releases at the time of writing:
   jupyter-server-proxy 4.6.0, Tornado 6.5.10, and huggingface_hub 2.2.0; Jupyter Server is pinned to 2.20.0, the
   version the SJC Strix Halo pods report (plumbing ruling 4, carried out by merge 2 for the pods that run;
   measurements M24; DUB's pod A ran 2.19.0, M6). Unpinned "stable"
   docs do not count: JupyterHub's stable REST reference is already 6.0.1.
4. The metrics file `lab/refine/metrics.jsonl`, later the ledger `lab/state/ledger.jsonl`.

Until the research notes exist, the sources at the end of this file, the verified lines on the R2 and R9 task
pages, the Q1 answer on Ilan's Team Comms page, and PR #37 with `lab/README.md` stand at level 3 under the same
rule.

A pass MUST open the source before changing a plan item and MUST cite it in the log: file and line at a pinned tag
or commit, doc URL and section, ledger line, or Notion page and line. No citation, no change. Nothing from memory.

## The rubric: eight plumbing tests

Every pass checks all eight against the whole track. A finding under any test is logged whether or not it is fixed
in that pass. Each test says what passing means, what the sources predict, how it is checked, which metric it
moves, and where it stands today. A cycle is the controller's keep-alive and poll interval, 10 minutes (R9
Done-when; Design Doc section 13), which is also the idle culler's check interval.

The checks under "Checked by" are carried out by the track's tasks and queued experiments, not by a pass: a pass
starts no pod or model server, runs no GPU job, and takes no queue slot (PASS_RULES.md, "What a pass may never
change"); it reads their results from the metrics file or the ledger. The unit tests run on this machine against
recorded hub and proxy answers under `fixtures/`, never a live call (`.claude/testing/AGENT_ANTIPATTERNS.md`,
"Atlas harness facts"). A live check takes a GPU Queue slot on pod B and Ilan's go, except read-only calls (hub API
reads, requests to pod A, calls to an account whose server is stopped), which a pass may make itself.

### T1. Every controller command is safe to re-run

- Passes when: each controller command (launch, stop, status, keep-alive, push, pull, start the model server) run
  twice in a row ends in the same state, exits 0 both times, and on the second run says there was nothing to do.
  Launch while running and stop while stopped are no-ops.
- Sources predict: the hub's own answers are not all no-ops. In JupyterHub 5.4.4 a launch returns 202 while a
  spawn is pending, 400 "is already running" when the server is up, and 400 "is pending stop" while it stops; a
  stop returns 204 when nothing runs and 202 while a stop is in progress (`jupyterhub/apihandlers/users.py`
  L622-L744). The controller therefore reads 400 "already running" as success and waits out "pending stop" before
  launching.
- Checked by: unit tests against recorded hub answers for the five states (no server, spawn pending, ready, stop
  pending, stopped), then one live double run of each command on pod B in a GPU Queue slot.
- Moves: failed-call rate (a re-run that errors is a failed call).
- Today: no controller exists on main (`lab/` holds the pod scripts only), and R9's first build was released with
  nothing pushed (R9 task page). The pod scripts already skip finished steps (`lab/pod_a_teacher.sh` L3,
  `lab/pod_b_trainer.sh` L3).

### T2. A pod lost mid-job is detected and the job resumes from the persistent home volume within one cycle

- Passes when: after a pod disappears (culled, runtime ended, stopped, or failed), the controller notices within
  one cycle, relaunches it, and the job continues from its last checkpoint under `/home/jovyan`, losing at most the
  work since that checkpoint; the recovery time goes to the ledger.
- Sources predict: nothing restarts a lost pod for us. The AUP spawner sets `restartPolicy: Never`
  (`runtime/hub/core/spawner/kubernetes.py` L1010-L1013) and stops the server once start + runtime_minutes x 60
  has passed, checking every 60 s (L1240-L1250, L1332-L1352); the reference chart culls a server after 3600 s
  without activity, checking every 600 s (`runtime/chart/values.yaml` L854-L862; `lab/README.md` L15-L16). The
  home volume survives a relaunch and `/opt/workspace` does not (`lab/README.md` L12-L13), so checkpoints, models,
  and builds belong under the home directory. huggingface_hub 2.2.0 skips files already downloaded and restarts an
  interrupted file from zero in a fresh temporary file (`file_download.py` L1966-L2003), so a lost pod costs the
  files in flight, not the whole checkpoint.
- Checked by: a unit test where a fake hub drops the server in the middle of a job; live, one slot on pod B that
  stops the server under a running job and times detection and resume.
- Moves: recovery time after a lost pod.
- Today: the plan is "the controller relaunches (15 s) and restarts the model servers from the persistent home"
  (Design Doc section 13); no code does it yet (R9 task page).

### T3. Credits per hour are tracked per account, with projected spend against balance, and keep-alive runs only while something is working

- Passes when: one command shows, for each account, the balance, the running session's minutes and credits so far,
  credits per hour over the last window, and the projected balance at the session's scheduled end; the keep-alive
  pings only while a job runs or a served model is in use, and the tracker shows 0 credits per hour when nothing
  is working.
- Sources predict: the R9700 costs 4 credits per minute (`runtime/values.yaml` L259-L265; `lab/README.md` L14),
  with about 60,000 per account (`lab/README.md` L14). The hub deducts a session only when it ends, for its whole
  elapsed minutes (`runtime/hub/core/quota/manager.py` `end_usage_session` L354-L411, called from the spawner's
  `stop` L1286-L1310), so the balance from `GET /hub/api/quota/me` (R9 task page) does not move while a pod runs
  and spend has to be projected from our own launch time. A launch whose runtime_minutes x rate exceeds the
  balance is refused with 403 "Insufficient quota for N min" (manager.py L118-L167; spawner L1046-L1072). At hub
  startup the stale-session cleanup closes usage records older than 480 minutes with their duration capped
  (manager.py L436-L474; `runtime/hub/core/setup.py` L356), so the hub's own record can disagree with elapsed time
  and the ledger keeps our own launch and stop times. Every request through the proxy is recorded as activity
  (`jupyter_server_proxy/handlers.py` L236-L259, called at L413; idle-culler README, "How it works"), so a
  keep-alive on an idle pod defeats the culler and spends 240 credits per hour per account.
- Checked by: unit tests of the projection against recorded quota answers; live, two `quota/me` reads around one
  stop, compared with the projection.
- Moves: credits per hour.
- Today: the rate is in the metrics file (`pod.credits_per_min`, lines 19 and 20); its balances are marked low
  confidence (`pod.balance_credits`, lines 21 and 22); nothing tracks or projects spend, and balances are read by
  hand (R9 task page). Both pods were running when R2 reported (R2 task page).

### T4. A test greps logs, repo, and Notion posts for tokens and fails on any hit

- Passes when: one command scans the repo's tracked files, the lab logs and the ledger on this machine and on the
  Jetson, and the Notion pages agents write to (the track's task pages, Team Comms, the Build Log, Overnight
  Status), exits non-zero on any hit, and prints only the location and the `.env` key name, never the value.
- Sources predict: JupyterHub tokens are `uuid.uuid4().hex`, 32 lowercase hex characters (`jupyterhub/utils.py`
  L420-L425). Atlas event ids have the same shape (`contracts/events.py` L21), and so do Notion page ids in links
  (the GPU Queue database id is even shaped like a version-4 UUID), so a bare 32-hex pattern flags every event
  and most Notion links. The scan matches (a) the exact token values, read in-process from `.env` and never
  printed, and (b) a token in context: `Authorization: token `, `token=`, or `"token":` followed by 32 hex
  characters. No pytest test reads `.env` (the workspace testing rules, `.claude/testing/AGENT_ANTIPATTERNS.md`,
  "Atlas harness facts"), so the exact-value scan is a script, and its matching is unit-tested with fake tokens.
- Checked by: the scan's output pasted in the log on every pass; a unit test that plants a fake token in a
  temporary directory and expects the scan to fail.
- Moves: none of the metrics. Any hit fails the track until the hit is removed and Ilan has replaced the token
  (tokens are credentials, his call).
- Today: no scan exists; the rules are written down (`AGENTS.md` L183; `lab/README.md` L4-L5; R9 Done-when).

### T5. The box-side call path is exercised end to end from a non-pod machine, with latency measured and failure modes mapped

- Passes when: from this machine and from the Jetson, a probe calls
  `https://dub.aupcloud.io/user/<account>/proxy/<port>/` with the account token (`lab/README.md` L37-L46; Design
  Doc section 8), records time to first byte and total time for a short completion, streamed and not, and fills a
  failure map with the observed status code and the client's response for: a bad or revoked token; the server
  stopped; the pod up with nothing listening or the model still loading; a hub restart; a streamed response; and a
  launch refused for quota.
- Sources predict:
  - Server stopped: 424, not 404. JupyterHub 5.4.4 answers requests for a stopped server itself with 424 (a JSON
    body for non-GET or JSON-accepting requests, an HTML page otherwise), and with 503 only if the hub sets
    `use_legacy_stopped_server_status_code`; while a spawn is pending, a browser GET gets a 303 to the
    spawn-pending page (`jupyterhub/handlers/base.py` L1653-L1689, L1717-L1841).
  - Bad token on a running server: jupyter-server-proxy's `prepare` applies tornado's `web.authenticated`
    (`jupyter_server_proxy/handlers.py` L139-L170), which redirects a GET to the login page and answers other
    methods with 403 (`tornado/web.py` L3514-L3545). The task text's 401 is not what the source predicts, and the
    hub's single-user auth may answer first, so the map records what actually comes back.
  - Nothing listening: the proxy's HTTP client fails with 599 (handlers.py L502-L511 streamed, L527-L545
    buffered); while vLLM loads, its `/health` answers 503 (vLLM b1388b1
    `vllm/entrypoints/serve/instrumentator/health.py` L22-L33).
  - Streaming: the proxy streams only when the request's Accept header is exactly `text/event-stream` (handlers.py
    L438-L443); every other request is buffered whole (L527 onward), which is why server-sent events have arrived
    in one burst (O7 task page, Scope). Buffered requests time out after 300 s (L712-L717), streamed ones after
    7200 s (L498-L500).
  - Hub restart: in the reference chart the proxy runs as its own configurable-http-proxy 5.2.0 deployment
    (`runtime/chart/values.yaml` L333-L349), so routes to running servers should keep working while the hub API
    is down. Observe it; never cause it.
  - Quota refusal: 403 with the balance and the largest runtime it allows (T3 sources).
- Checked by: a probe script whose handling of each code is unit-tested against recorded answers; live, requests
  to pod A (a shared service: requests only) and to pod B's account while it is stopped (observing a stopped server
  needs no slot).
- Moves: failed-call rate; the latency numbers feed time from launch to serving.
- Today: the path works end to end (Q1 answer: a test server on port 8099; PR #37: teacher calls with text and 720p
  images from outside); `/proxy/absolute/<port>/` does not (`lab/README.md` L46). No latency is measured from
  outside the pod (on the pod, TTFT is 0.55 s for text, `lab/README.md` L69), and no failure map exists.

### T6. Push and pull speeds are measured and acceptable for the real sizes

- Passes when: a real dataset export pushed to pod B and a real model package (a few GB: the GGUF, vision
  projector, and manifest that R10 serves, R10 task page) pulled to the Jetson are each measured in MB/s at their
  real size with sha256 checked, both meet the bound set by ruling (open item 10), and the 18 GB teacher never
  crosses the link (it downloads on the pod from Hugging Face, `lab/README.md` L53).
- Sources predict:
  - Push: Jupyter Server refuses a request body over `max_body_size`, 512 MiB by default (`serverapp.py`
    L1237-L1248 at v2.20.0), and the contents API carries a file base64-encoded inside JSON. Its default contents
    manager saves in chunks (`chunk` 1, 2, and so on, and -1 for the last; `AsyncLargeFileManager`, serverapp.py
    L1617-L1621 and `largefilemanager.py` L85-L133 at v2.20.0), so a push of more than a few hundred MB must be
    chunked.
  - Pull: jupyter-server-proxy fetches from the pod port with tornado's AsyncHTTPClient at its defaults
    (handlers.py L429-L437; JupyterHub's single-user extension only adds TLS defaults, `singleuser/extension.py`
    L328-L341), which accepts at most a 100 MiB response body (`tornado/simple_httpclient.py` L106-L141;
    `tornado/http1connection.py` L146 and L642), and a buffered request ends after 300 s (handlers.py L712-L717).
    A package of a few GB through `/proxy/<port>/` is therefore predicted to fail, which puts R10's Done-when (a
    file server on the pod behind the proxy) and L6's pull at risk until a path is measured.
- Checked by: a transfer script that writes size, seconds, MB/s, chunk or part size, and the sha256 result per run
  to the ledger; live on pod B in a slot.
- Moves: push speed and pull speed.
- Today: nothing measured; R6 and R10 are Not started (R6 and R10 task pages).

### T7. Pod state (running, ready, port, model) is readable from one command and mirrored to the ledger

- Passes when: one command prints, for both accounts, running, ready, pending, started, scheduled end, last
  activity, the serving port, the served model name, the model server's health, and the balance, and appends one
  ledger row per account (verb `status`). As a `python -m lab` verb it answers with one JSON line and exit code 0
  or 1 (`lab/__main__.py`, module docstring).
- Sources predict: `GET /hub/api/users/{name}` gives each server's `ready`, `pending`, `started`, `last_activity`,
  `url`, and `progress_url` (JupyterHub 5.4.4 `rest-api.yml`, Server schema; `tutorial/server-api.md`);
  `GET /hub/api/quota/me` gives the balance (R9 task page); through the proxy, vLLM answers `GET /v1/models` with
  the served name and `GET /health` with 200 or 503 (vLLM b1388b1 `vllm/entrypoints/openai/models/api_router.py`
  L20-L21, `instrumentator/health.py` L22-L33); pod A serves `teacher` on port 8000 (`lab/README.md` L54). Ledger
  rows carry ts, actor, tier, task, verb, outcome, metrics, and refs (O15 task page).
- Checked by: a unit test of the status command against recorded answers that checks every printed field and the
  ledger row; live, a read-only run (no slot needed).
- Moves: none directly; it is the instrument the other metrics are read with.
- Today: state is read by hand (R9 task page).

### T8. A fresh pod reaches serving from scripts alone in a measured time

- Passes when: from a stopped account, one command launches the pod, uploads `lab/`, starts the model server, and
  waits until `GET /v1/models` answers through the proxy, with no browser step; the time from the launch request
  to that answer is in the ledger for the warm case (build and checkpoint already in the home volume) and, once,
  for the cold case (empty home).
- Sources predict: the pod is ready about 15 s after the launch (R9 task page; Q1 answer; `pod.start_s`,
  metrics.jsonl line 18); vLLM's engine init
  takes about 270 s (`lab/README.md` L73; PR #37); a cold home adds the vLLM build, about 1 h (`lab/README.md`
  L52), and the 18 GB checkpoint download (unmeasured). Three steps are by hand today: jupyter-server-proxy needs
  one relaunch after its install (`lab/README.md` L37; `lab/pod_a_teacher.sh` L20-L24), the private repo cannot be
  cloned at launch so `lab/` is uploaded after it (`lab/README.md` L32-L33), and the server is started from a
  terminal (Jupyter Server's `POST /api/terminals`, `api.yaml` L608-L632 at v2.20.0; Design Doc section 8 lists
  the terminal and REST as the only access).
- Checked by: one timed warm run on pod B in a slot; the cold case only when pod B's home is rebuilt anyway.
- Moves: time from launch to serving.
- Today: the parts are known (about 15 s plus about 270 s warm), but there is no end-to-end number.

## The metric set

Names follow the dotted, lowercase style of `lab/refine/metrics.jsonl`, which already holds `pod.start_s`,
`pod.credits_per_min`, and `pod.balance_credits` for the nodes `pod-a` and `pod-b` (lines 18 to 22); the six below
are new to this track, and the merge pass reconciles a name if another track chose a different one. Every row
names its node and its conditions; for this track the node is `pod-a` or `pod-b` (one account each), or the
calling machine for link metrics. MB means 10^6 bytes.

| Metric | Name | Unit | How it is measured | Before (2026-10-10) and source | Better |
|---|---|---|---|---|---|
| Credits per hour | `pod.credits_per_hour` | credits/h | Credits per wall-clock hour over a stated window, per account (node), from our own launch and stop times at the posted rate, checked against `quota/me` around each stop | 240 per running pod-hour by rate (`pod.credits_per_min` = 4 for `pod-a` and `pod-b`, metrics.jsonl lines 19 and 20; `runtime/values.yaml` L265); no window average yet | lower at equal work |
| Time from launch to serving | `pod.launch_to_serving_s` | s | From the launch request to the first 200 from `GET /v1/models` through the proxy; conditions say warm or cold | not measured end to end; parts: 15 s to ready (`pod.start_s`, metrics.jsonl line 18) and about 270 s engine init (`lab/README.md` L73) | lower |
| Push speed | `link.push_mb_s` | MB/s | Bytes over seconds for a real export pushed to pod B, sha256 checked; conditions give bytes and chunk size | unmeasured | higher |
| Pull speed | `link.pull_mb_s` | MB/s | Bytes over seconds for a real package pulled from pod B, sha256 checked; conditions give bytes and path | unmeasured | higher |
| Recovery time after a lost pod | `pod.recovery_s` | s | From the loss (the server gone from `GET /hub/api/users/{name}`, or a 424 through the proxy) to the job running again | unmeasured | lower |
| Failed-call rate | `link.failed_call_rate` | fraction | Failed calls over all calls through the proxy in a stated window; a failure is any non-2xx answer other than an expected no-op, a timeout, or a 599 | unmeasured | lower |

## The walk list (track items and failure scenarios)

Each pass walks every track item and at least two failure scenarios, rotating so that all fourteen are walked
across a run; a pass can only converge if it walked all fourteen. The log says which were walked. Walk each
scenario through the items as they stand (code on main, open PRs, and each task's Done-when), step by step, and
record every place with no automated step, no mapped response, or no number.

Track items (the starting set in PASS_RULES.md, "The four tracks": R6, R9, R10, and the credit rules that T3
judges):
- R9 pod controller: launch, stop, keep-alive, long sessions, status (R9 task page).
- R6 box-side transfers: push a dataset with the contents API, pull a package through the proxy (R6 task page).
  R6 is shared with Xander, so a pass proposes changes to it and never edits it (PASS_RULES.md, "What a pass may
  never change").
- R10 pod-side package file server (R10 task page).
- R13 controller live test, R14 push and pull speed measurement, R15 recovery drill, R16 secrets grep test,
  R17 credit tracker and R18 proxy failure-mode map: added by merge 2 under plumbing ruling 3 (owner,
  2026-10-10: the walk list is every plumbing task in Ilan's lane, R6 and R9 to R18, under the same rules).
  R11 and R12 inside that range are the kernels track's (PASS_RULES.md, "The four tracks"; kernels ruling 1),
  and the merge log's track map records the boundary.
- Experiments queued on T0 whose metric is in this file's metric set (PASS_RULES.md, "The four tracks").
- Any plumbing task a pass or the merge pass creates.

Walked for context, owned elsewhere (a finding about them goes in the entry for the merge pass):
- L6, deploy to the Jetson over the hub proxy: its pull step uses this track's link (learning track; L6 task page).
- E1 on T0, senses contention: its result decides the controller's stop and start policy (T0 task page), but it
  moves the serving and training metrics of other tracks.
- The GPU Queue procedure for launches and stops on pod B (GPU Queue database; open-ticket skill, section 4b): an
  owner rule that only Ilan changes.

Failure scenarios:
- P1 Launch while the server is up (400 "is already running"; T1).
- P2 Stop while stopped (204) and stop while stopping (202) (T1).
- P3 Launch refused for quota (403 "Insufficient quota for N min"; T3).
- P4 The idle culler stops the pod mid-job (3600 s without activity, checked every 600 s; T2).
- P5 The session's runtime ends mid-job (start + runtime_minutes x 60; T2).
- P6 Two agents launch or stop the same account within seconds (T1; the GPU Queue).
- P7 The box calls the proxy while the server is stopped (424 predicted, 404 in the task text; T5).
- P8 The token is wrong or revoked (a login redirect or 403 predicted, 401 in the task text; T5).
- P9 The pod is up but the model server is down or still loading (599, then `/health` 503; T5).
- P10 A streamed completion through the proxy (one burst unless `Accept: text/event-stream`; T5).
- P11 A package pull larger than 100 MiB through the proxy (predicted to fail; T6).
- P12 A dataset push larger than 512 MiB in one request (predicted to be refused; T6).
- P13 The hub restarts while a pod serves (T5).
- P14 A relaunch on an account with a persisted home: what survives (the home volume: models, builds,
  `~/.local`) and what does not (`/opt/workspace`, running processes) (T2, T8).

## What a pass may NOT change

Everything PASS_RULES.md lists under "What a pass may never change" holds here: other people's and shared tasks,
the Demo Plan, priorities, Locked tasks, the rulings file, this file, other tracks' rows, an earlier pass's change
without a newer source, and anything outside plans. For this track that means in particular:

- Pod A's teacher server. Requests only; it is never stopped, restarted, or relaunched (GPU Queue rules: pod A is
  a shared service all weekend).
- Pods and slots. A pass starts no pod or model server, runs no GPU job, and takes no queue slot (PASS_RULES.md).
  The track's tasks and experiments do the live work, each with a GPU Queue slot on pod B and Ilan's explicit go
  naming the node and the job (open-ticket skill, sections 4b, 5, and 7).
- Tokens. Never printed, logged, committed, posted, or written to the ledger; refer to the `.env` key name only
  (`AGENTS.md` L183; R9 Done-when).
- What crosses the link. The 18 GB teacher checkpoint never crosses it (O3 task page, test 6). Raw frames never
  leave the Jetson; only approved, blurred exports go to the pods (Design Doc sections 4 and 8).
- Dead paths stay dead: Tailscale on the pods (blocked, Q1 answer), vLLM on the Jetson and the Strix Halo node
  (open-ticket skill, section 7), and the `auto` accelerator, which picks the cheapest eligible accelerator rather
  than the R9700 (`runtime/hub/core/spawner/kubernetes.py` L800-L866).
- The launch keys. The spawner reads `runtime_minutes`, `resource_type`, `gpu_selection`, `repo_url`,
  `repo_branch`, and `repo_persist` from the launch request (`kubernetes.py` L998-L1008, L1122-L1123, L1173;
  `lab/README.md` L26-L28); a pass does not invent others.
- Billing. Every elapsed minute of every session counts at the posted rate; a pass never plans around a gap in the
  hub's accounting (T3).
- This file. Only the merge pass edits it, and only to carry out a filled ruling that names the change
  (PASS_RULES.md, "The merge pass"); a pass that finds a test, a metric, or a universal rule wrong raises a ruling.

## The pass procedure (every pass, in this order)

PASS_RULES.md ("What a pass reads", "What a pass does") is the procedure; these are its steps with what they mean
for this track.

1. Read PASS_RULES.md, this file, and `lab/refine/TEMPLATES.md`, fresh. Read `refine_pass.txt` in this folder;
   this is pass N, its value plus one.
2. Read `REFINE_LOG.md` from the closing section of the latest run ("State at the end of the run") to the end, or
   all of it if no run has closed; then `OWNER_RULINGS.md`, the metrics file, the notes under
   `lab/refine/research/plumbing/` with their `SOURCES.md`, the board rows of the track, the experiment queue on
   T0, and the latest entry of `lab/refine/MERGE_LOG.md`.
3. Metrics before. For each metric in the set, copy its latest value, unit, node, conditions, and source line. A
   metric with no value is listed as having no baseline, never estimated, and a measurement for it is queued on T0.
4. Research, bounded. Open the sources that bear on what you expect to change, within the search bound of
   `lab/refine/RESEARCH_FORMAT.md` (until it exists: 30 minutes and at most six new sources). Write the findings
   into `lab/refine/research/plumbing/` in that format, each labeled, with its citation.
5. Walk every track item and at least two failure scenarios the previous pass did not walk. Record every place a
   path breaks, needs a hand step, has no mapped response, or has no number.
6. Check all eight tests against the whole track. List EVERY finding with its location (task ID and Done-when
   clause, file and line at a commit, or experiment), whether or not you change it.
7. Make the changes the findings justify, highest impact first: (a) a path that loses work or spends credits
   unnoticed, (b) a hand step with no script, (c) a failure mode with no mapped response, (d) a metric with
   neither a measurement nor a queued experiment, (e) wording precision. Each change records BEFORE verbatim,
   AFTER verbatim, SOURCE, WHY in one sentence, and the metric it should move with the expected direction, and
   its line goes on the Plan changes post on Ilan's Team Comms page when the change is made (PASS_RULES.md,
   "Telling Ilan what changed"). An idea not yet tested is queued on T0 with its expected movement, never logged
   as an improvement. A finding about a task owned elsewhere goes in the entry for the merge pass.
8. Raise a ruling in the entry for anything the sources cannot decide: Now, Recommend, Source of tension, and a
   blank `Answer:` line.
9. Metrics after. If the pass ran anything (a read-only call), record the after values, measured the same way
   under the same conditions; if it ran nothing, the entry says so. No movement is no improvement; a regression is
   reverted or raised for a ruling with the trade-off stated.
10. Checks, pasting the real output: `python -m pytest -q` from the clone root is green; the token scan (T4) exits
    0, or until it exists `git grep -nE "(Authorization: token |token=)[0-9a-f]{32}"` prints nothing; and
    `grep -cP "[\x{2013}\x{2014}]" <file>` prints 0 for every file the pass wrote (plain hyphens only).
11. Append the entry to `REFINE_LOG.md` in the `TEMPLATES.md` format (the minimum is below) and write the pass
    report, which states the time the pass ended.
12. Write N to `refine_pass.txt`, overwriting. This is the last step. A pass that fails before it does not move
    the counter, reverts its board changes, and releases its task (PASS_RULES.md, "What a pass does").
13. A pass that walked every track item and all fourteen scenarios and changed nothing ends its entry with
    `CONVERGED at pass N`. The tenth pass of a run that does not converge ends with `CAP REACHED at pass N, not
    converged` and raises the cap ruling (PASS_RULES.md, "The cap"). Either way the counter still moves.

## Log entry format

`lab/refine/TEMPLATES.md` is the standard once it exists. Until then, and as the minimum it must keep, an entry
carries these blocks in this order:

```
## Pass N, YYYY-MM-DD

Research: <sources opened, one per line> :: <note file written under lab/refine/research/plumbing/>
Metrics before:
- <metric> = <value> <unit> (node, conditions) :: <ledger or metrics.jsonl line>, or "no baseline; queued as E<n>"
Walked: <track items>; scenarios P<a>, P<b> :: <where each broke, or "held">
Findings (all, by test):
- T<n> <location>: <what is wrong>

Changes (highest impact first):
1. <location> (T<n>, P<x> if a scenario)
   BEFORE: <verbatim>
   AFTER:  <verbatim>
   SOURCE: <file and line at a tag or commit / doc URL and section / ledger line / Notion page and line>
   WHY: <one sentence>
   MOVES: <metric>, expected <direction and size>
   POSTED: <the Plan changes line, as posted>

For the merge pass: <findings about tasks owned elsewhere> or none
Rulings raised: <question>; Now: <...>; Recommend: <...>; Source of tension: <...>; Answer: (blank) - or none
Metrics after: <as above, or "nothing run this pass">
Checks: pytest <summary line>; token scan <exit code>; dashes 0
Left for the next pass: <count> findings; scenarios not yet walked: <list>
Pass report: <file under pass_reports/>, ended <time>
```

## Known open items for pass 1

Found on 2026-10-10 from the sources below; none is measured yet.

1. Streaming through the proxy (T5, P10). The proxy streams only for `Accept: text/event-stream` (handlers.py
   L438-L443). Experiment: the same streamed teacher call from this machine with and without that header; expected,
   time to first token through the proxy falls from the whole-response time to near the on-pod 0.55 s plus the
   network (`lab/README.md` L69).
2. Packages over 100 MiB (T6, P11; R10, L6). The proxy's client caps a response at 100 MiB and a buffered request
   at 300 s (T6 sources). Experiment: pull a 150 MB file through `/proxy/<port>/` (expected to fail), then compare
   candidate paths for a 3 GB package, such as parts under 90 MB with a sha256 per part, or a route that bypasses
   jupyter-server-proxy found in the research step. R10's Done-when may need a rewrite once this is measured.
3. Pushes over 512 MiB (T6, P12; R6). Experiment: one 600 MB push in a single contents-API request (expected to be
   refused) against chunked pushes at two chunk sizes; record MB/s per chunk size.
4. Status codes (T5, P7, P8). The sources predict 424 for a stopped server and a login redirect or 403 for a bad
   token, where the task text says 404 and 401. Record what our hub returns; the observation goes in the failure
   map, and the task text is corrected through a ruling if it differs.
5. Spend is invisible until a stop (T3). The balance moves only when a session ends (manager.py L354-L411), so the
   tracker projects from launch time. Queue the check for a pod B slot: two `quota/me` reads around one stop,
   compared with elapsed whole minutes x 4.
6. Keep-alive and the culler (T3, P4). Proxied requests are recorded as activity (handlers.py L236-L259, called on
   every request at L413; idle-culler README, "How it works"), and the hub refreshes `last_activity` every 300 s
   (`jupyterhub/app.py` L502-L504). Read-only check: `servers[""].last_activity` from `GET /hub/api/users/<name>`
   before and after one proxied call to pod A, at least 300 s apart. Ruling needed: whether pod A counts as working
   through an idle night (the GPU Queue rules and Design Doc section 13 keep it up all weekend) at 240 credits per
   idle hour.
7. Token-shaped ids (T4). A bare 32-hex pattern flags every event id and Notion link (T4 sources); the scan uses
   exact values plus token-in-context patterns.
8. Hand steps on a fresh pod (T8): the jupyter-server-proxy relaunch, the `lab/` upload, and the terminal start
   (`lab/README.md` L32-L33 and L37; T8 sources). Each becomes a controller step or a queued experiment.
9. Versions on the pod (sources level 3). The hub reports 5.4.4 (`GET https://dub.aupcloud.io/hub/api/`, read
   without a token on 2026-10-10), while JupyterHub's stable docs are 6.0.1, so cite 5.4.4 sources only.
   jupyter-server-proxy was installed with `pip install --user` on 2026-10-10 (Q1 answer), when PyPI's newest was
   4.6.0 (released 2026-09-21). Queue `pip show jupyter-server-proxy jupyter-server tornado huggingface_hub` for
   the start of the next pod B slot, and correct any pinned citation that differs. Read since: jupyter_server
   2.19.0 on DUB's pod A (M6), 2.21.1 on the first REST-launched SJC pod (M20) and 2.20.0 on the SJC Strix Halo
   pods (M24); merge 2 pinned the Jupyter Server citations to 2.20.0 under ruling 4; tornado and huggingface_hub
   are still unread on a pod.
10. Acceptable transfer speed (T6) has no owner rule yet. Recommended for a ruling: a few-GB package pull within
    10 minutes (about 5 MB/s for 3 GB), because a trainer run finishes in tens of minutes (Design Doc section 8)
    and shipping should not take longer than training. Until it is answered, a pass records the numbers and does
    not mark T6 passed.

## Sources (opened 2026-10-10)

Our own pages and repo:
- O3 task page, this rubric's spec: https://app.notion.com/p/3f596ef5fec5813ba635cba31a445b1e
- R2 task page, pod facts and the R2 progress line: https://app.notion.com/p/3f596ef5fec5816bac9ce34a28989784
- R9 task page, Done-when and the verified launch, stop, and quota line: https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230
- R6 task page, box-side transfers: https://app.notion.com/p/3f596ef5fec581a29055dcde2edee80f
- R10 task page, pod-side package file server: https://app.notion.com/p/3f596ef5fec581638880c7f06ee849bf
- L6 task page, deploy over the hub proxy: https://app.notion.com/p/3f596ef5fec581658fd3cd890933faa9
- T0 task page, the experiment queue: https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1
- O1 task page, the shared wording of the universal rules: https://app.notion.com/p/3f596ef5fec581859648f9824af01daf
- O7 task page, plumbing research scope and the one-burst observation: https://app.notion.com/p/3f596ef5fec58181bf00cb833fa190ec
- O9 task page, file formats and who writes the rulings: https://app.notion.com/p/3f596ef5fec581c38706ec07a84acf72
- O10 task page, metrics file schema and names: https://app.notion.com/p/3f596ef5fec5812c9c58dd375d7e8d8d
- O13 task page, pass rules: https://app.notion.com/p/3f596ef5fec581b395cdef8efccafd9e
- O15 task page, ledger schema: https://app.notion.com/p/3f596ef5fec581499019c49da7a9d322
- Ilan's Team Comms page, the Q1 answer (verified pod facts): https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7
- Design Doc, sections 4, 6, 6b, 8, and 13: https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917
- GPU Queue (R9700 pods) database: https://app.notion.com/p/c69762a396874e049a4af6819f762615
- PR #37, R2 GPU lab setup with the measured numbers: https://github.com/raohemdutt/Knighthacks26/pull/37
- `lab/README.md` on main: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/README.md
- `lab/pod_a_teacher.sh` on main: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/pod_a_teacher.sh
- `lab/pod_b_trainer.sh` on main: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/pod_b_trainer.sh
- `lab/refine/PASS_RULES.md`, the pass rules every track follows: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/refine/PASS_RULES.md
- `lab/refine/metrics.jsonl`, the metrics file (lines 18 to 22 hold the pod rows): https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/refine/metrics.jsonl
- `lab/__main__.py`, the `python -m lab` dispatcher and its JSON contract: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/lab/__main__.py
- `contracts/events.py` L21, the event id shape: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/contracts/events.py#L21
- `AGENTS.md` L183, secrets: https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5c9b021822bb212d465edfe81e3db48280/AGENTS.md#L183

JupyterHub (our hub runs 5.4.4):
- Live version, read without a token: https://dub.aupcloud.io/hub/api/
- REST API spec at 5.4.4: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/docs/source/_static/rest-api.yml
- Starting servers with the API, 5.4.4: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/docs/source/tutorial/server-api.md
- Using the REST API, 5.4.4: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/docs/source/howto/rest.md
- Server start and stop handlers: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py#L619-L744
- Requests for a stopped server: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/handlers/base.py#L1625-L1841
- Activity refresh interval: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/app.py#L502-L504
- Token generator: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/utils.py#L420-L425
- Single-user HTTP client defaults: https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/singleuser/extension.py#L328-L341
- Stable REST reference (6.0.1, for the version check only): https://jupyterhub.readthedocs.io/en/stable/reference/rest-api.html
- Idle culler README, "Caveats" and "How it works": https://github.com/jupyterhub/jupyterhub-idle-culler

jupyter-server-proxy:
- Proxy handlers at v4.6.0: https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py
- Accessing arbitrary ports or hosts: https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html
- Server process options (`update_last_activity`): https://jupyter-server-proxy.readthedocs.io/en/latest/server-process.html
- Release history (4.6.0, 2026-09-21): https://pypi.org/project/jupyter-server-proxy/

Jupyter Server and Tornado:
- Body limits (L1237-L1258) and the default contents manager (L1617-L1621), at v2.20.0, the version the SJC Strix Halo pods report (M24): https://github.com/jupyter-server/jupyter_server/blob/v2.20.0/jupyter_server/serverapp.py#L1237-L1258
- Chunked saves: https://github.com/jupyter-server/jupyter_server/blob/v2.20.0/jupyter_server/services/contents/largefilemanager.py
- REST API spec, terminals: https://github.com/jupyter-server/jupyter_server/blob/v2.20.0/jupyter_server/services/api/api.yaml
- REST API docs: https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html
- Client body limits: https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/simple_httpclient.py#L100-L141
- Body limit enforcement: https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/http1connection.py#L146
- `web.authenticated`: https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/web.py#L3514-L3545
- HTTP client docs: https://www.tornadoweb.org/en/stable/httpclient.html

AUP Learning Cloud (public repo at commit 9b4af19):
- Repository: https://github.com/AMDResearch/aup-learning-cloud
- Spawner (launch keys, quota check, restart policy, runtime timer, stop): https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/hub/core/spawner/kubernetes.py
- Quota manager (launch check, charge at stop, stale-session cleanup): https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/hub/core/quota/manager.py
- Cleanup at hub startup: https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/hub/core/setup.py#L356
- Accelerator rates (r9700 quotaRate 4): https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/values.yaml#L259-L265
- Chart values (proxy, culler): https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/chart/values.yaml#L854-L862

Model server and downloads:
- vLLM `GET /v1/models` at b1388b1: https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/models/api_router.py#L20-L21
- vLLM `GET /health` at b1388b1: https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/serve/instrumentator/health.py#L22-L33
- huggingface_hub download to a temporary file, v2.2.0: https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/file_download.py#L1966-L2003
