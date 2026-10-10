# Idle culling, keep-alive, credits, and recovering work on a lost pod

Research note for the plumbing track (O3 rubric tests T2 and T3), opened 2026-10-10; sources indexed in
`SOURCES.md`. Question: what keeps a pod alive or gets it culled, how are credits actually charged, and how
should a job be written so that a pod lost mid-job costs as little as possible?

Labels, keys and metric names as in `hub-api-lifecycle.md`: `measured` = observed on a running system with the
setup (ours: `measurements-2026-10-10.md`, M1 to M15); `practitioner` = documented behavior or recommendation
from the people who build or run the system (vendor docs, maintainers' source at a pinned version,
maintainers' PRs); `anecdote` = one report without data; `opinion` = argument without evidence. Metrics (O3
rubric [R8]): CPH `pod.credits_per_hour`, TTS `pod.launch_to_serving_s`, MB/s `link.push_mb_s` and
`link.pull_mb_s`, RT `pod.recovery_s`, FCR `link.failed_call_rate`.

## 1. What the culler sees

### Takeaway
A pod is culled once the hub's record of its last activity is an hour old, checked every ten minutes. Any request
through the proxy refreshes that record, including a health check and including a failed call, and so does any API
call to the pod's Jupyter server. Reads of the hub's own API do not. At 11:00Z the teacher pod had gone 38 minutes
with nothing touching it.

### Cited findings
- **[practitioner]** AUP's chart enables the idle culler with `timeout: 3600`, `every: 600`, `maxAge: 0` and `users: false`, and the hub runs `jupyterhub_idle_culler` with those flags. Source: AUP `runtime/chart/values.yaml`, `cull`, L854-L862, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/chart/values.yaml#L854-L862 ; `runtime/hub/core/jupyterhub_config.py` L360-L400, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/jupyterhub_config.py#L360-L400 [S37] [S36]

  Implies: an untouched pod keeps billing for 60 to 70 minutes after its last recorded activity before the culler stops it, about 240 to 280 credits per idle episode at 4 per minute (CPH); `maxAge: 0` means no age limit, so only the runtime timer bounds a busy pod.

  Experiment (IDL-1): after a known last request, leave a pod untouched and record when it leaves the user model. Expected: 60 to 75 minutes later. Minimum sample: 1 (it costs a relaunch, so Ilan decides when).

- **[practitioner]** jupyterhub-idle-culler 1.4.0 (the version in k8s-hub 4.3.3) culls a ready server when `now - server.last_activity >= timeout`, skips servers with a pending action or not yet ready, and applies `max_age` only when set; its README says the timeout should exceed the websocket ping interval (30 s) plus `JupyterHub.last_activity_interval` (5 min). Source: jupyterhub-idle-culler 1.4.0 `jupyterhub_idle_culler/__init__.py`, `cull_idle`, L180-L262, https://github.com/jupyterhub/jupyterhub-idle-culler/blob/1.4.0/jupyterhub_idle_culler/__init__.py#L180-L262 ; README, "Caveats", https://github.com/jupyterhub/jupyterhub-idle-culler/blob/1.4.0/README.md#caveats [S29] [S06]

  Implies: a keep-alive ping every 10 minutes leaves about 45 minutes of margin, and a ping is only needed while work is running (T3).

  Experiment: IDL-1.

- **[practitioner]** The hub's per-server `last_activity` is the latest of the proxy's route activity, polled every `last_activity_interval` = 300 s, and the single-user server's own reports, sent every `hub_activity_interval` = 300 s; updates less than `activity_resolution` = 30 s newer are dropped. The culler README describes the same two sources and says activity of apps behind jupyter-server-proxy is covered. Source: JupyterHub 5.4.4 `app.py` L490-L504 and `update_last_activity` L3555-L3600, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/app.py#L3555-L3600 ; `singleuser/extension.py`, `hub_activity_interval`, L384-L399, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/singleuser/extension.py#L384-L399 ; culler README, "How it works", https://github.com/jupyterhub/jupyterhub-idle-culler/blob/1.4.0/README.md#how-it-works [S23] [S24] [S06]

  Implies: the hub learns of activity up to about 5 minutes late, small next to the 60-minute timeout (CPH).

  Experiment (IDL-2): repeat the propagation measurement of M10 on pod B's next session. Expected: under 300 s each time. Minimum sample: 3 observations.

- **[practitioner]** The pod's Jupyter server reports `last_activity()` = the latest of its start, kernel activity, every setting whose key ends in `_last_activity` (API requests, terminals) and every entry in `last_activity_times`; jupyter-server-proxy writes `last_activity_times["jupyter-server-proxy"]` at the start and end of every proxied request; authenticated API requests count unless they carry `?no_track_activity`. Source: jupyter_server v2.19.0 `serverapp.py`, `last_activity`, L542-L559, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/serverapp.py#L542-L559 ; `base/handlers.py`, `update_api_activity`, L804-L812, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/base/handlers.py#L804-L812 ; jupyter-server-proxy v4.6.0 `_record_activity`, L236-L259, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L236-L259 [S26] [S25]

  Implies: every proxied call keeps a pod alive, even a failed one (the proxy stamps activity before the backend answers), and `no_track_activity` does not help for proxied calls; the design doc's health check every 5 s through the proxy would keep both pods alive forever, at 240 credits per hour each whether or not they work (T3, CPH).

  Experiment: IDL-3.

- **[practitioner]** configurable-http-proxy 5.2.0 records route activity when a proxied request finishes with a status below 300, and on request body data; it no longer records activity at the start of a request. Source: CHP 5.2.0 `lib/configproxy.js`, `handleProxy`, L645-L672, https://github.com/jupyterhub/configurable-http-proxy/blob/5.2.0/lib/configproxy.js#L645-L672 [S30]

  Implies: failed proxied calls (302, 500, 503) do not count at the CHP level, but they still count through the pod's own report (previous finding), so a box retrying against a broken model server also keeps the pod alive (CPH).

  Experiment: none separate.

- **[measured]** At 11:00:56 hub time pod A's server `last_activity` was 10:22:21, so the teacher had been idle 38.6 minutes with no keep-alive running. Our proxied reads moved it to 11:01:14 within 127 s, and a later read showed up the same way. Reads of the hub API moved only the user-level `last_activity`, never the server's. Source: `measurements-2026-10-10.md` M3 and M10 [R1]

  Implies: until the R9 controller exists, the shared teacher is culled about an hour after its last use, and each of our probes reset that clock; a status command built only on hub API reads has no keep-alive side effect (T3, CPH).

  Experiment (IDL-3): a hub-API-only status poll every 10 minutes for an hour on a pod nobody else uses. Expected: server `last_activity` unchanged by the polls. Minimum sample: 6 polls.

- **[measured]** Pod A, untouched after our last read at 11:04:47 hub time, was stopped at 12:13:39, 68.9 minutes later, with no stop issued by our tools; the 60-minute timeout plus the 10-minute culler cycle predicts exactly that window. Its balance then fell by 664 credits, which is 166 whole minutes x 4 for a session of 166.25 minutes. Source: `measurements-2026-10-10.md` M10 and M15 [R1]

  Implies: IDL-1's prediction holds on the live hub, so the shared teacher dies about 70 minutes after its last use unless something works on it, and its next start costs a full TTS (T2, T3, CPH); the charge confirms the rounded-down per-minute billing of section 3.

  Experiment: IDL-1 is answered once (n=1); repeat on the next idle episode to confirm the window. Expected: stop between 60 and 75 minutes after the last activity. Minimum sample: 2 more episodes.

- **[measured]** Account B's 600-minute session ended at 10:53:37 hub time after 78 minutes (`Session 776: 78 min @ r9700 (4/min)`, -312 credits), and the stopped server's record shows `last_activity` equal to the stop time, so a cull cannot be told apart from a deliberate stop through the hub API. Source: `measurements-2026-10-10.md` M4 and M5 [R1]

  Implies: the ledger must record every stop the controller issues; any stop without a controller record is a cull or a runtime expiry, which is the signal T2 needs to start recovery (RT).

  Experiment (IDL-4): the controller logs each DELETE with a reason. Expected: every stop this weekend is explained by a log line, a cull or an expiry. Minimum sample: all stops.

## 2. A "working" signal for keep-alive

### Takeaway
Keep-alive should depend on the pod doing work, not on a timer. The teacher reports its own queue through
`/metrics`, and a training job can be judged from GPU activity, which AMD SMI reads without root.

### Cited findings
- **[measured]** Through the proxy, `vllm:num_requests_running` and `vllm:num_requests_waiting` were both 0 at 11:00Z, and the time-to-first-token histogram had counted 5 requests since the engine started. Source: `measurements-2026-10-10.md` M8 [R1]; metric definitions in vLLM docs, "Production Metrics", https://docs.vllm.ai/en/latest/usage/metrics.html [S10]

  Implies: "the teacher is working" can be read as running + waiting > 0, or as the histogram count rising since the last check; a rule of "ping only if that moved in the last 10 minutes" meets T3 for the teacher (CPH).

  Experiment (IDL-5): log the two gauges and the histogram count every 10 minutes for a day. Expected: nonzero only around real box traffic. Minimum sample: 100 samples.

- **[practitioner]** AMD SMI's CLI reports engine usage with `amd-smi metric --usage` and power with `--power`, lists GPU processes with `amd-smi process` (process names may need elevated permissions and show `N/A` without sudo), repeats with `--watch` or `--iterations`, and prints `--json`; the page marks only `set`, `reset` and some RAS features as needing sudo. Source: ROCm docs, "AMD SMI CLI tool usage", Commands `amd-smi metric` and `amd-smi process`, https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html [S08]

  Implies: a training job's "working" signal can be GPU activity read without root, and pod state (T7) can carry GPU activity and power into the ledger (CPH).

  Experiment (IDL-6): sample `amd-smi metric --usage --json` every 10 s during a LoRA run and while idle. Expected: clearly separated activity levels. Minimum sample: 30 samples per state (queued pod B slot).

- **[measured]** On a live pod with no root, `rocm-smi --showpower` and `amd-smi metric --power` both worked. Source: Ilan's Team Comms page, Q1 answer, Sat 5:25 AM, https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 [R5]

  Implies: the AMD SMI signal above is available on our pods today.

  Experiment: IDL-6.

## 3. How credits are charged

### Takeaway
AUP charges a session once, when it stops, for whole elapsed minutes times the rate. The balance does not move
while a pod runs, so credits per hour must be computed from start times rather than read from the balance. A hub
restart more than eight hours into a session can leave that session unbilled, so our ledger computes spend itself.

### Cited findings
- **[practitioner]** AUP opens a usage session at the top of the spawner's `start()` (before the pod exists) and settles it in `stop()`: `duration_minutes = int(elapsed_seconds / 60)` (rounded down), `quota_consumed = duration_minutes * rate`, and the balance never goes below 0; if the hub lost the session id it recovers the user's active session from the database. Source: AUP `quota/manager.py`, `end_usage_session`, L354-L411, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/quota/manager.py#L354-L411 ; `spawner/kubernetes.py`, `start` L1038-L1044 and `stop` L1286-L1330, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L1286-L1330 [S35] [S34]

  Implies: a session costs `floor((stop - session start) / 60 s) x 4`, with the clock starting before the pod is up; a 59-second probe costs nothing and the 15 s pod start is billed (CPH).

  Experiment (IDL-7): for every stop, compare the predicted charge with the transaction. Expected: equal, or one minute (4 credits) apart when the session start precedes our recorded launch. Minimum sample: every stop, at least 4.

- **[measured]** Account A's balance stayed at 59,960 while its session ran for about 96 minutes; its past transactions are whole minutes times 4 (`Session 773: 3 min` = -12, `Session 758: 7 min` = -28), as are account B's (`Session 776: 78 min` = -312, `Session 759: 15 min` = -60). Source: `measurements-2026-10-10.md` M5 [R1]

  Implies: credits per hour while a pod runs come from `(now - started) x 4` per running server, read from the hub user model, and are reconciled with the transaction at the stop (T3, CPH).

  Experiment: IDL-7.

- **[practitioner]** At every hub startup AUP runs `cleanup_stale_sessions()`, which marks active sessions older than 480 minutes as `cleaned_up` without deducting anything, and `end_usage_session` later returns (0, 0) for any session that is not `active`. Source: AUP `runtime/hub/core/setup.py` L351-L360, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/setup.py#L351-L360 ; `quota/manager.py`, `cleanup_stale_sessions` L436-L473 and `end_usage_session` L371-L372, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/quota/manager.py#L436-L473 [S36] [S35]

  Implies: a long session that spans a hub restart after its eighth hour may never be billed, so balance-based accounting would under-count; our ledger computes spend itself and flags any stop without a matching transaction (CPH). Whether to tell AMD staff is Ilan's call.

  Experiment: none possible without a hub restart; the IDL-7 reconciliation flags it.

- **[measured]** Both servers were launched with `runtime_minutes: 600` at a rate of 4 credits per minute; account A's server started at 09:27:24Z, so AUP's timer will stop the teacher at about 19:27Z whatever the keep-alive does. Source: `measurements-2026-10-10.md` M3 and M5 [R1]

  Implies: an all-weekend teacher needs a longer `runtime_minutes` at launch (the server enforces no cap) or a scheduled relaunch, and each relaunch costs one TTS (about 15 s plus the vLLM init); a running pod costs 240 credits per hour, and both pods for 36 hours would cost about 17,280 of the 119,588 left (CPH).

  Experiment (IDL-8): record the actual stop time of this session. Expected: between 19:27:24Z and 19:28:30Z hub time. Minimum sample: 1.

## 4. Recovering work on a lost pod (T2)

### Takeaway
A lost pod gives its jobs about a second (see `hub-api-lifecycle.md`), so recovery depends on what was already
written to the persistent home. Trainers should checkpoint there on a wall-clock schedule, downloads already
resume, and the teacher's restart cost is dominated by the vLLM engine start, which a persistent compile cache may
shorten.

### Cited findings
- **[measured]** `/home/jovyan` is a persistent NFS volume (models, `pip --user` packages and builds survive a relaunch) and `/opt/workspace`, the Jupyter root, is ephemeral. Source: `lab/README.md` L12-L13, "What the pods are", at bb351e6 (PR #37), https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md#L12-L13 [R2]; Ilan's Team Comms page, Q1 answer, https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 [R5]

  Implies: every checkpoint, dataset and log a resumed job needs must live under `~/` (RT).

  Experiment: XFR-1 in `transfers.md`.

- **[practitioner]** Transformers' `Trainer.train(resume_from_checkpoint=True)` loads the last checkpoint in `output_dir` and resumes "from the model/optimizer/scheduler states loaded here"; `save_strategy` defaults to `"steps"` with `save_steps` = 500, `save_total_limit` prunes old checkpoints, and `save_only_model=True` "prevents resuming training from the checkpoint". Source: Transformers v5.19 docs, "Trainer", `Trainer.train` and `TrainingArguments` "Checkpointing & Saving", https://huggingface.co/docs/transformers/en/main_classes/trainer [S13]

  Implies: at the smoke-run rate of 0.51 samples/s on pod B (batch 4, max_length 256, rank 16; metrics file line 13 and PR #37 [R7] [R3]), the default 500 steps is over an hour of training, so a lost pod could cost far more than one cycle; the L3 trainer should set `save_steps` from wall time (a few minutes), write `output_dir` under `~/`, keep `save_only_model` off and resume with `True` (RT).

  Experiment (IDL-9): stop pod B mid-run in a queued slot, relaunch, rerun with `resume_from_checkpoint=True`. Expected: training continues from the last `checkpoint-N` with an unbroken loss curve; RT = launch (about 15 s) + model load + the steps since the last save. Minimum sample: 2.

- **[measured]** On pod A the vLLM engine took about 270 s to initialize (graph capture included) for `RedHatAI/Qwen3.8-27B-INT4` at 16k context, holding 84% of VRAM. Source: `lab/README.md` L73, "Teacher (pod A)", at bb351e6 (PR #37), https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md#L73 [R2]

  Implies: the teacher's RT and TTS are at least 15 s plus 270 s, about 5 minutes, and the engine start is the term worth shrinking.

  Experiment: IDL-10.

- **[practitioner]** vLLM caches `torch.compile` output under `~/.cache/vllm/torch_compile_cache` by default; on a later run "Inductor compilation is completely bypassed, and we will load from disk"; the cache can be disabled with `VLLM_DISABLE_COMPILE_CACHE=1`, and `cudagraph_capture_sizes` limits which batch sizes get graphs captured. Source: vLLM docs, "torch.compile integration", "Compilation Cache" and "Cudagraph Capture", https://docs.vllm.ai/en/latest/design/torch_compile.html [S11]

  Implies: because `~` is the persistent home, the second and later teacher starts on an account should beat the 270 s cold start; fewer capture sizes would cut graph capture further at some cost to batched decode (TTS, RT).

  Experiment (IDL-10): time `serve` to the first 200 on `/v1/models` with the cache present and with `VLLM_DISABLE_COMPILE_CACHE=1`. Expected: the cached start is faster, by an amount nobody has published for gfx1201. Minimum sample: 3 per arm.

- **[practitioner]** jupyter_server_terminals does not cull terminals by default (`cull_inactive_timeout = 0`, "Values of 0 or lower disable culling"), and terminals end with the pod; five R2 terminals were still open about 85 minutes later (M6). Source: jupyter_server_terminals v0.5.4 `terminalmanager.py` L32-L44, https://github.com/jupyter-server/jupyter_server_terminals/blob/v0.5.4/jupyter_server_terminals/terminalmanager.py#L32-L44 [S32]; `measurements-2026-10-10.md` M6 [R1]

  Implies: a job launched from a terminal survives the controller's websocket disconnecting but not a stop; it should run under `nohup` with its log under `~/`, as `lab/pod_a_teacher.sh` already does, so the controller can read progress after reconnecting (RT).

  Experiment (IDL-11): start a 5-minute dummy job from a terminal, drop the websocket, reconnect. Expected: the job finished and its log is complete. Minimum sample: 2.

## Gaps
Pod B's stop cause is unknown (M4). The cull timing, runtime-expiry timing and checkpoint-resume drill all need a
pod stopped on purpose, which is Ilan's call and a queue slot. How much the vLLM compile cache saves on gfx1201 is
undocumented and unmeasured, and the 270 s figure is not broken down into weight loading from NFS, compilation and
graph capture.

## 5. Activity and culling with the front door down (plumbing pass 1, 2026-10-10)

### Takeaway
On AMD's current config the culler is a service of the hub process and pods report their activity to the hub
inside the cluster, so the public front door going down stops our box-side keep-alive pings but not the culler.
A pod that has nothing of its own counting as activity is then stopped about an hour into the outage.

### Cited findings
- **[practitioner]** AUP's hub config sets `cleanup_servers = False` and `last_activity_interval = 60`, points `hub_connect_url` at the in-cluster hub service, and starts the idle culler as a hub-managed service that calls the hub at `localhost:8081`; z2jh 4.3.3 sets the same in-cluster `hub_connect_url`. Source: AUP `runtime/hub/core/jupyterhub_config.py` L85-L86, L204-L206 and L360-L400, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/jupyterhub_config.py#L360-L400 [S36]; z2jh 4.3.3 `jupyterhub/files/hub/jupyterhub_config.py` L97-L107, https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/4.3.3/jupyterhub/files/hub/jupyterhub_config.py#L97-L107 [S46]

  Implies: on this config the hub polls the proxy's route activity every 60 s, not JupyterHub's default 300 s, so section 1's "up to about 5 minutes late" is an upper bound (M10's 127 s fits both), if the live build (M13) carries the line. More important for long sessions: with the hub up and the front door down (M16), proxied pings from the box cannot land while the culler keeps running, and a loaded model with no requests is not activity (M15), so a pod with nothing pod-side generating activity is stopped 60 to 70 minutes into such an outage; with the hub itself down nothing is culled, but its runtime timer can be lost (`hub-api-lifecycle.md` section 3). A keep-alive generated on the pod, an authenticated request to its own Jupyter API, would not depend on the front door (CPH, RT).

  Experiment (IDL-12): during a long session on pod B, run a pod-side loop that makes an authenticated request to the pod's own Jupyter API every 10 minutes, pause the box-side ping for 80 minutes, and read the server's `last_activity` from the hub API every 10 minutes. Expected: `last_activity` advances about every 10 minutes and the server is not culled; credits per hour stay at 240. Minimum sample: one 80-minute window.
