# Hub API and pod lifecycle: launch, stop, state, tokens

Research note for the plumbing track (O3 rubric tests T1, T2, T4, T7 and T8), opened 2026-10-10; every
source was opened that day and is indexed in `SOURCES.md`. Question: what do the JupyterHub REST API
and AMD's AUP Learning Cloud spawner actually do when our controller launches, stops, polls and
authenticates, and which of our metrics does each behavior move?

Labels: `measured` = a number or behavior observed on a running system, with the setup (our own read-only
probes are in `measurements-2026-10-10.md`, cited as M1 to M14); `practitioner` = documented behavior or a
documented recommendation from the people who build or run the system (vendor docs, maintainers' source code
and docstrings at a pinned version, maintainers' PRs); `anecdote` = one report without data; `opinion` =
argument without evidence. Every finding ends with `[S..]` or `[R..]` keys into `SOURCES.md`; source links pin
a tag or commit and their line anchors point at the function cited. Under each finding, "Implies" maps it onto
our metric set and "Experiment" proposes a test with the expected movement and the minimum sample. Metric set
(names from the O3 rubric, `lab/refine/plumbing/RUBRIC.md` [R8]): credits per hour `pod.credits_per_hour` (CPH), time from launch to
serving `pod.launch_to_serving_s` (TTS), push and pull speed `link.push_mb_s` and `link.pull_mb_s` (MB/s),
recovery time after a lost pod `pod.recovery_s` (RT), failed-call rate `link.failed_call_rate` (FCR).
Experiments that launch, stop or generate on a pod need Ilan's go and a GPU Queue slot.

## 1. What runs where

### Takeaway
The live hub is JupyterHub 5.4.4 behind Cloudflare, built from AMD's hub image on z2jh 4.3.3, and the pods run
jupyter_server 2.19.0 with jupyter-server-proxy 4.6.0. AMD's public source moves faster than the Dublin
deployment, so every AUP finding below is dated against the live build.

### Cited findings
- **[measured]** The hub reports JupyterHub 5.4.4 and every response carries `server: cloudflare`; pod A runs jupyter_server 2.19.0 and the `jupyterhub-jupyter-server-proxy` 4.6.0 extension. Source: our read-only probe of 2026-10-10 11:00Z, `measurements-2026-10-10.md` M1 and M6 [R1]

  Implies: hub citations must be pinned to 5.4.4 and pod citations to jupyter_server 2.19.0; Cloudflare's limits apply to every controller and box call (see `proxy-path.md`), which touches FCR and MB/s.

  Experiment (HUB-1): read `GET /hub/api/` and `GET /user/<acct>/api` at every pod launch and write both versions to the ledger. Expected: no change all weekend. Minimum sample: every launch, at least 4.

- **[practitioner]** AMD's hub image is built `FROM quay.io/jupyterhub/k8s-hub:4.3.3`, whose pinned requirements are jupyterhub 5.4.4, jupyterhub-kubespawner 7.0.0, jupyterhub-idle-culler 1.4.0 and tornado 6.5.5; the single-user base image pins `jupyterhub==5.4.4` and `jupyterlab==4.5.6`. Source: AUP `dockerfiles/Hub/Dockerfile` L22, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/dockerfiles/Hub/Dockerfile#L22 ; z2jh 4.3.3 `images/hub/requirements.txt` L83-L235, https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/4.3.3/images/hub/requirements.txt#L83-L235 ; AUP `dockerfiles/Base/Dockerfile.rocm` L196-L201, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/dockerfiles/Base/Dockerfile.rocm#L196-L201 [S38] [S33]

  Implies: KubeSpawner 7.0.0 and culler 1.4.0 are the versions to read for stop and cull behavior (RT, CPH).

  Experiment: none beyond HUB-1; a hub version change would invalidate these pins. Expected: none.

- **[measured]** The live spawn form bundle contains `repo_persist`, which entered AUP on 2026-06-25 (commit 6e09fad), while the pod still roots Jupyter at `/opt/workspace`, which AUP changed to `/home/jovyan` on 2026-07-06 (commit b5b9b7f8); the Dublin build is therefore from between those dates. Source: `measurements-2026-10-10.md` M13 [R1]; AUP git history [S39]

  Implies: AUP code older than 2026-06-25 (quota charging, the runtime timer, `cleanup_servers`, the preStop clone cleanup) applies to our hub; later code (for example `restartPolicy: Never`, 2026-07-28) may not.

  Experiment (HUB-2): re-fetch `/hub/static/spawn-ui/assets/index.js` once a day and compare its size and markers. Expected: unchanged. Minimum sample: 2 reads, one per day.

## 2. Launch and stop are not symmetric (T1)

### Takeaway
Stopping a stopped server is a native no-op (204). Launching a running server is not: the hub answers 400.
The controller has to read state first or map that 400 to success, otherwise "launch when running" counts as
a failed call.

### Cited findings
- **[practitioner]** `POST /hub/api/users/{name}/server` returns 202 with no action when a spawn is already pending, 400 `"<name> is already running"` when the server is ready and its poll says it is alive, 400 `"is pending stop"` while a stop is in flight, and otherwise spawns and returns 202 (still launching) or 201 (ready at once). Source: JupyterHub 5.4.4 `jupyterhub/apihandlers/users.py`, `UserServerAPIHandler.post`, L623-L670, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py#L623-L670 [S20]

  Implies: a naive retrying launch inflates FCR with false failures and can race a stop; "launch is a no-op when running" (T1) must be built into the controller.

  Experiment (HUB-3): call the controller's launch twice on a running server and once while a stop is pending. Expected: the second launch exits 0 with "already running"; the pending-stop launch waits and retries. Minimum sample: 3 per state (9 calls).

- **[practitioner]** `DELETE /hub/api/users/{name}/server` returns 204 when the server is stopped, including when it was already stopped (neither stop branch runs), 202 while the stop is still pending, and cancels a pending spawn before stopping it. Source: JupyterHub 5.4.4 `UserServerAPIHandler.delete`, L673-L744, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py#L673-L744 [S20]

  Implies: stop is idempotent for free; a stop issued mid-launch cancels the launch, which a controller retry loop must not mistake for a lost pod (RT).

  Experiment (HUB-4): DELETE twice in a row on a stopped server. Expected: 204, then 204. Minimum sample: 3.

- **[practitioner]** JupyterHub's server-API guide: 201 means the server is ready at its URL immediately, 202 means it has begun launching; wait by polling the server model until `ready` is true (`pending` names a transitional state such as `start` or `stop`) or read the progress EventStream; there is no progress API for stopping, so poll the user model until the server disappears. Source: JupyterHub docs, "Starting servers with the JupyterHub API", sections "Starting servers", "Waiting for a server to start" and "Stopping servers", https://jupyterhub.readthedocs.io/en/stable/tutorial/server-api.html [S01]

  Implies: TTS starts at the POST and its pod part ends at the progress stream's `ready` event; for RT, the pod counts as gone at the first poll where the server has left the user model.

  Experiment (HUB-5): time POST to `ready` through the progress stream on every launch. Expected: about 15 s (next finding). Minimum sample: 5 launches.

- **[practitioner]** The progress stream sends a keepalive every 8 s and ends with `{"progress": 100, "failed": true, "message": "Spawn failed: ..."}` when a spawn fails, which the guide does not mention. Source: JupyterHub 5.4.4 `SpawnProgressAPIHandler`, L770-L880, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py#L770-L880 [S20]

  Implies: launch failures (quota, no free GPU, start timeout) can be classified without extra polling, so FCR can carry a reason for each failed launch.

  Experiment (HUB-6): log the last progress event of every launch over the weekend. Expected: `ready` on all launches; any `failed` carries AUP's message. Minimum sample: all launches.

- **[measured]** On our hub, `POST /hub/api/users/{name}/server` with `{runtime_minutes, resource_type: "gpu", gpu_selection: "r9700"}` returned 202 and the pod was ready in about 15 s; `DELETE` on the same path stopped it at once. Source: R9 task page, "Verified today" line, Sat 6:35 AM, https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230 [R4]; `lab/README.md` "Launching a pod" at bb351e6 (PR #37), https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md [R2]

  Implies: the pod part of TTS is about 15 s; the 270 s vLLM engine init (see `idle-credits-recovery.md`) dominates TTS and the teacher's RT.

  Experiment: HUB-5. Expected: median 15 s, all launches under 60 s when a card is free.

## 3. AUP launch gates and what the pod inherits (T2, T3, T7, T8)

### Takeaway
AUP checks the balance before every launch, writes the start time, the requested minutes and the rate into the
pod's environment, and enforces the requested runtime with a timer that lives only in the hub's memory. A pod
being stopped gets about one second before it is killed.

### Cited findings
- **[practitioner]** AUP refuses a launch when the balance is zero or below `runtime_minutes x rate` (its message gives the largest affordable runtime, `balance // rate`), and the spawner turns that into HTTP 403 `"Cannot start container: ... Please contact administrator to add quota."`. Source: AUP `runtime/hub/core/quota/manager.py`, `can_start_container`, L118-L167, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/quota/manager.py#L118-L167 ; `runtime/hub/core/spawner/kubernetes.py`, `RemoteLabKubeSpawner.start`, L1046-L1071, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L1046-L1071 [S35] [S34]

  Implies: a 36-hour session at 4 credits per minute needs 8,640 credits available at launch (balances were 59,960 and 59,628 at 11:00Z, M5); the controller should compute the affordable runtime before POSTing so a launch never fails on quota (FCR, TTS).

  Experiment (HUB-7): the controller logs `runtime_minutes x 4` next to `GET /hub/api/quota/me` before each launch. Expected: never refused. Minimum sample: all launches.

- **[practitioner]** The spawner puts `JOB_START_TIME` (epoch seconds), `JOB_RUN_TIME` (requested minutes) and `QUOTA_RATE` into the pod's environment. Source: AUP `spawner/kubernetes.py`, `_build_runtime_metadata_env` L739-L756 and `start` L1086-L1096, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L739-L756 [S34]

  Implies: a pod-side job can compute its own deadline and checkpoint before the runtime expires, and pod state (T7) can include the deadline without a hub call (RT, CPH).

  Experiment (HUB-8): on the next launch run `env | grep -E 'JOB_|QUOTA_'` in a pod terminal. Expected: all three present. Minimum sample: 1 per account.

- **[practitioner]** The runtime limit is an in-memory hub timer: `shutdown_time = start_time + runtime_minutes * 60`, checked every 60 s through `loop.call_later`, which calls the spawner's `stop()` directly; the spawner keeps neither value in its saved state, so a hub restart drops the timer. Source: AUP `spawner/kubernetes.py`, `start` L1240-L1251 and `check_timeout` L1332-L1352, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L1240-L1352 , present since at least 2026-02-06 (commit 29b463a) [S34] [S39]

  Implies: expiry lands up to 60 s after the deadline; after a hub restart a pod runs until it is culled or stopped, so our controller must enforce its own deadline to bound CPH.

  Experiment (HUB-9): launch with `runtime_minutes: 5` and read the stop time from the hub user model. Expected: stopped 0 to 90 s after start + 300 s. Minimum sample: 3.

- **[practitioner]** AUP's hub config sets `c.JupyterHub.cleanup_servers = False`, which JupyterHub documents as "Disable if you want to be able to teardown the Hub while leaving the single-user servers running." Source: AUP `runtime/hub/core/jupyterhub_config.py` L85, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/jupyterhub_config.py#L85 ; JupyterHub 5.4.4 `app.py`, `cleanup_servers`, L1540-L1550, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/app.py#L1540-L1550 [S36] [S23]

  Implies: a hub restart does not kill our pods (good for RT), but it drops the runtime timer (previous finding) and the in-memory usage session id.

  Experiment (HUB-10): the controller polls `GET /hub/api/` every 10 minutes and logs failures. Expected: zero or a few short outages. Minimum sample: one weekend of polls (about 200).

- **[practitioner]** KubeSpawner 7.0.0 deletes a pod with `grace_period_seconds = delete_grace_period`, which defaults to 1 s (0 s when the hub stops it with `now=True`), and AUP does not override it. Source: KubeSpawner 7.0.0 `kubespawner/spawner.py`, `delete_grace_period` L1884-L1897 and `stop` L3337-L3346, https://github.com/jupyterhub/kubespawner/blob/7.0.0/kubespawner/spawner.py#L1884-L1897 [S28]

  Implies: a culled, expired or stopped pod gives our jobs about one second; nothing can be saved on the way down, so RT depends on how recently the job checkpointed to the home volume.

  Experiment (HUB-11): run a pod-side script that traps SIGTERM and appends a line to `~/sigterm.log`, then stop the pod. Expected: no line or a truncated one. Minimum sample: 3 (queued pod B slot).

- **[practitioner]** On deletion the kubelet runs any preStop hook, then sends TERM to process 1 in each container, and sends SIGKILL to whatever still runs when the grace period expires (30 s by default; `--grace-period` overrides it). Source: Kubernetes docs, "Pod Lifecycle", "Termination of Pods" and "Pod Termination Flow", https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination [S14]

  Implies: process 1 in our pods is the image's entrypoint (AUP's base image ends with `CMD ["/bin/bash", "/entrypoint.sh"]`, `dockerfiles/Base/Dockerfile.rocm` L257 [S38]), never our job; a trainer or vLLM started from a terminal may not see TERM before the 1 s SIGKILL, which strengthens the "checkpoint periodically" rule (RT).

  Experiment: HUB-11, also recording whether TERM arrived at all. Expected: not delivered to the child job in most runs.

- **[practitioner]** AUP's chart waits `singleuser.startTimeout: 300` seconds for a pod, and the spawner requests the GPU as the Kubernetes extended resource `amd.com/gpu` (guarantee and limit both 1), which the ROCm k8s device plugin allocates; its README documents GPU requests through `resources.limits` in the pod spec. Source: AUP `runtime/chart/values.yaml` L555, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/chart/values.yaml#L555 ; AUP `spawner/kubernetes.py` L929-L931, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L929-L931 ; ROCm k8s-device-plugin README, "Example workload", https://github.com/ROCm/k8s-device-plugin [S37] [S34] [S09]

  Implies: when no R9700 is free on the node, a launch waits up to 300 s and then fails, adding a capacity term to TTS and RT that our scripts cannot shorten; the controller should retry with backoff and report "no GPU free" rather than "launch broken".

  Experiment: HUB-6 with time-to-ready. Expected: ready in about 15 s when a card is free, a `failed` event after about 300 s otherwise.

## 4. Reading pod state without touching the pod (T7)

### Takeaway
One hub API call per account returns everything a status command needs, and it does not reset the pod's idle
clock. The hub's clock and ours differ, and the token cache means a hub outage reaches the box several minutes
late.

### Cited findings
- **[measured]** `GET /hub/api/users/<acct>` returned, for a running server, `ready`, `pending`, `started`, `last_activity`, `url`, `progress_url` and `user_options` (`runtime_minutes: 600`, `resource_type: gpu`, `gpu_selection: r9700`); with `?include_stopped_servers=1` a stopped server shows `stopped: true`, `started: null` and `last_activity` equal to the stop time. Source: `measurements-2026-10-10.md` M3 and M4 [R1]

  Implies: running, ready, requested runtime and deadline (`started + runtime_minutes`) come from one hub GET per account, and spend so far is `(now - started) x 4` (CPH); the port and model need one proxied `GET /v1/models`, which does reset the idle clock (see `idle-credits-recovery.md`).

  Experiment (HUB-12): a status command that reads only the hub API every 10 minutes for an hour. Expected: the server's `last_activity` does not move because of it. Minimum sample: 6 polls.

- **[measured]** The hub's `Date` header ran 46 s ahead of the workstation clock. Source: `measurements-2026-10-10.md` M14 [R1]

  Implies: durations that mix hub timestamps (`started`, quota transactions) with local clocks are off by about 46 s, a small bias in CPH and RT; compute durations in hub time.

  Experiment (HUB-13): log the hub `Date` offset with every status poll. Expected: a stable offset within 1 s. Minimum sample: 10 polls.

- **[practitioner]** On the pod, JupyterHub's auth caches token lookups for 300 s (`HubAuth.cache_max_age`); when the hub API is unreachable, uncached requests fail with 500 ("Failed to connect to Hub API"), a hub 5xx becomes 502 ("Failed to check authorization (upstream problem)"), and a hub 403 becomes 500 ("Permission failure checking authorization"). Source: JupyterHub 5.4.4 `jupyterhub/services/auth.py`, `cache_max_age` L470-L479 and `_api_request` L595-L680, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/services/auth.py#L470-L680 [S22]

  Implies: during a hub restart, box calls keep working for up to 5 minutes and then fail with 500 or 502 until the hub is back; FCR should tag these apart from model-server failures and retry with backoff.

  Experiment (HUB-14): classify box-side failures by status and body text over the weekend. Expected: any 500 or 502 bursts line up with hub outages seen by HUB-10. Minimum sample: one weekend.

## 5. Tokens (T4)

### Takeaway
Our two controller tokens never expire and can do everything their account can, including stopping servers and
minting tokens. A token meant for the box only needs proxy access to one server, and JupyterHub can mint exactly
that, with an expiry.

### Cited findings
- **[practitioner]** JupyterHub's guide says to "pass the API token in the request's Authorization header", but the single-user auth also accepts a `?token=` URL parameter, which it checks first. Source: JupyterHub docs, "Using JupyterHub's REST API", "Make an API request", https://jupyterhub.readthedocs.io/en/stable/howto/rest.html ; JupyterHub 5.4.4 `services/auth.py`, `HubAuth.get_token`, L731-L756, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/services/auth.py#L731-L756 [S02] [S22]

  Implies: any URL built with `?token=` puts the token into Cloudflare, proxy and shell logs; the T4 grep should match `token=` in URLs as well as bare token strings.

  Experiment (HUB-15): run the T4 grep for 32-hex-character strings and `token=` over the repo, logs and our Notion posts. Expected: 0 hits once Notion page links are excluded (their page ids are also 32 hex characters, as in the links in these notes, so a bare pattern gives false positives). Minimum sample: every PR.

- **[measured]** Both controller tokens (`a177` for account A and `a178` for B, note "atlas controller (Knighthacks26)") have `expires_at: None` and carry the owner's full scope set, including `servers`, `delete:servers`, `access:servers` and `tokens`; the server token the hub minted at spawn (`a179`) is narrow (`access:servers!server=<acct>/` plus activity and name scopes). Source: `measurements-2026-10-10.md` M11 [R1]

  Implies: a controller token copied onto the Jetson could stop pods and mint new tokens, with no end date; a leak there costs credits and availability (CPH, FCR).

  Experiment (HUB-16): mint a box token with scopes `["access:servers!server=<acct>/"]` and `expires_in` of 3 days. Expected: proxied `GET /v1/models` returns 200 and `DELETE /hub/api/users/<acct>/server` returns 403 with it. Minimum sample: 3 calls each.

- **[practitioner]** JupyterHub scopes: `access:servers` = "Access user servers via API or browser", `servers` = "Start and stop user servers", `delete:servers` = "Stop and delete users' servers", `tokens` = "Read, write, create and delete user tokens"; `access:servers!server=username/` limits access to that user's default server; the `inherit` metascope gives a token its owner's permissions. Source: JupyterHub docs, "Scopes in JupyterHub", sections "Scope conventions", "Metascopes" and "Available scopes", https://jupyterhub.readthedocs.io/en/stable/rbac/scopes.html [S03]

  Implies: the least-privilege box token in HUB-16 is a supported configuration, not a workaround.

  Experiment: HUB-16.

- **[practitioner]** `POST /hub/api/users/{name}/tokens` accepts `note`, `scopes` (or `roles`) and `expires_in` (integer seconds or null); the hub-wide `token_expires_in_max_seconds` defaults to 0, meaning tokens without expiry are allowed. Source: JupyterHub 5.4.4 `UserTokenListAPIHandler.post`, L406-L560, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py#L406-L560 ; `app.py`, `token_expires_in_max_seconds`, L467-L480, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/app.py#L467-L480 [S20] [S23]

  Implies: the mint script can set an expiry per token today; nothing on AMD's side forces one, which matches M11.

  Experiment (HUB-17): mint one 72-hour token and read its `expires_at` back. Expected: now + 72 h. Minimum sample: 1.

## Gaps
How often the Dublin hub restarts, and what its logs keep (request URLs, tokens in query strings), cannot be read
from outside. Whether a quota refusal reaches the caller as the POST's own error or only as a `failed` progress
event was not tested, because that needs a launch. The hub-restart behaviors above (lost runtime timer, token
cache) come from source, not observation.
