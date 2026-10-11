# Hub API and pod lifecycle: launch, stop, state, tokens

Research note for the plumbing track (O3 rubric tests T1, T2, T4, T7 and T8), opened 2026-10-10; every
source was opened that day and is indexed in `SOURCES.md`. Question: what do the JupyterHub REST API
and AMD's AUP Learning Cloud spawner actually do when our controller launches, stops, polls and
authenticates, and which of our metrics does each behavior move?

Labels: `measured` = a number or behavior observed on a running system, with the setup (our own read-only
probes are in `measurements-2026-10-10.md`, cited as M1 to M15); `practitioner` = documented behavior or a
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

## 6. The front door down (plumbing pass 1, 2026-10-10)

### Takeaway
Since about 11:20 AM ET the hub's public address has answered Cloudflare's own plain-text 502 on every path,
so nothing that goes through `dub.aupcloud.io` works: no launch, stop, status read, proxied model call or
keep-alive ping. Both servers were already stopped, so nothing was lost this time.

### Cited findings
- **[measured]** Unauthenticated GETs of `/hub/api/`, a hub static file and `/hub/login` all returned 502 in under 0.4 s with `Server: cloudflare`, `Content-Type: text/plain`, no `x-jupyterhub-version` header and the 16-byte body `error code: 502`; `python -m lab ready` got 502 for both accounts' hub API reads; the R9 task page's lock line of Sat 11:24 AM ET already records the hub answering 502. Source: `measurements-2026-10-10.md` M16 [S47]; R9 task page, lock line of Sat 11:24 AM ET, https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230 [S48]

  Implies: a front-door outage is its own failure class for FCR, told apart by `Server: cloudflare` plus the plain `error code: 502` body, distinct from a stopped server (424, `proxy-path.md` section 4) or a dead model server (500, M9); during it the controller can neither launch, stop nor read state, the box cannot reach a model server, and keep-alive pings sent through the proxy cannot land (see `idle-credits-recovery.md` section 5). The box shows "Learning paused" (Design Doc section 6b) and retries with backoff.

  Experiment (HUB-18): the controller's status poll logs every front-door 502 with its `CF-RAY` and the outage's first and last failed poll. Expected: FCR gains a front-door class and the ledger holds outage minutes per day; no movement in the other metrics. Minimum sample: every poll for the weekend.

- **[practitioner]** Cloudflare documents that a 502 or 504 originating from Cloudflare is a page without Cloudflare branding, while a Cloudflare-branded 502 or 504 is the origin's own error passed through; its listed causes of the unbranded kind are origin compression faults and data-center traffic shifts that usually last seconds. Source: Cloudflare docs, "Error 502 or 504", sections "502/504 from Cloudflare" and "502/504 from your origin web server", https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502-504/ [S44]

  Implies: our unbranded 16-byte body is Cloudflare's own error, but an outage of hours on every path is not the seconds-long kind the page lists, so from outside we cannot tell whether AMD's hub, its ingress or the route to it is down; fixing it is AMD staff's job, and our tools only detect, wait and record it (FCR).

  Experiment: none beyond HUB-18.

## 7. Whose clock is off (plumbing pass 1, 2026-10-10)

### Takeaway
The 46 s gap of M14 is our workstation running slow, not the hub running fast: three unrelated public servers
show the same gap.

### Cited findings
- **[measured]** At 16:04 UTC the `Date` headers of Google, GitHub and Cloudflare's own site ran 44 to 47 s ahead of the workstation clock, the same gap the hub's Cloudflare-served responses showed in M14 and M16. Source: `measurements-2026-10-10.md` M17 [S47]

  Implies: ledger rows, pass reports and queue timestamps written on this machine run about 46 s slow, and durations that mix hub timestamps (`started`, quota transactions) with this clock carry that error (CPH, TTS, RT); timing code should take "now" from the hub's `Date` header or from a synced clock, and the section 4 advice to compute durations in hub time stands.

  Experiment: HUB-13 as specified in section 4, logging the offset against a public clock as well as the hub's. Expected: the same 44 to 47 s until the workstation clock is resynced. Minimum sample: 10 polls.

## 8. A second hub: SJC (plumbing pass 2, 2026-10-10)

### Takeaway
After the DUB outage the pods run on AMD's SJC hub (accounts C and D). It is the same JupyterHub 5.4.4 and the
same AUP hub build behind the same Cloudflare front, so the hub-side and Cloudflare findings above carry over.
Three things differ: quota is switched off, so nothing is metered; a REST launch there gave a stock image without
a GPU, so the pods were relaunched through the web spawn form; and the pods are Strix Halo (gfx1151), whose
Jupyter server reports 2.20.0.

### Cited findings
- **[measured]** SJC's hub answers `{"version": "5.4.4"}` with `server: cloudflare` and a `cf-ray`, and its spawn-form bundle has the same size (218,673 bytes) and markers as DUB's. Pod C's Jupyter server reported 2.21.1 on the first, REST-launched pod (19:01Z) and 2.20.0 on the Strix Halo pod launched through the web form (19:31Z); DUB's pod A reported 2.19.0. Source: `measurements-2026-10-10.md` M18, M20 and M24 [S49]

  Implies: Cloudflare's 125 s read timeout and 100 MB upload cap (`proxy-path.md` section 3, `transfers.md` section 2) and the hub's own answers (sections 2 to 4) apply on SJC as on DUB, so the FCR and MB/s predictions stand; the pod-side version depends on how the pod was launched, so a pod-side citation names the version of the pod it was read on (2.20.0 for the Strix Halo pods today), and a reading taken on SJC is not one taken on DUB, because the node's account, hub and chip differ.

  Experiment (HUB-19): at every launch on either hub, write the hub's version, the pod's jupyter_server version and the launch path (REST or web form) to the ledger per account (HUB-1 for two hubs). Expected: SJC stays 5.4.4 with 2.20.0 on form-launched pods; no metric moves. Minimum sample: every launch.

- **[measured]** On SJC `GET /hub/api/quota/me` answers `balance` 0, `unlimited` true, `enabled` false for both accounts, and both servers run on `runtime_minutes` 2400: the REST-launched ones since 18:54Z, and the Strix Halo ones (`gpu_selection` `strix-halo`) launched through the web form since 19:07Z. Source: `measurements-2026-10-10.md` M19 and M24 [S49]

  Implies: credits per hour on SJC are 0 by the hub's own setting, so T3's credit tracking and projection measure nothing there, and a keep-alive costs no credits; AUP's runtime timer (section 3) would stop both Strix Halo servers near 11:07Z on 2026-10-12 if the hub stays up, while the idle culler, if this build keeps the chart default (`idle-credits-recovery.md` section 1), still stops a server about an hour after its last activity whatever its runtime (CPH, RT).

  Experiment (HUB-20): on an unmetered hub, record each server's minutes run since `started` on every status poll as the use measure in place of credits. Expected: about 60 minutes per hour per running account while credits stay 0. Minimum sample: every status poll for a day.

## 9. The DUB outage as Cloudflare classifies it, and a readiness check tied to DUB (plumbing pass 2, 2026-10-10)

### Takeaway
Asked for JSON, Cloudflare says the DUB 502 is the origin's fault (`origin_bad_gateway`, retry after 60 s, owner
action required), so the outage is on AMD's side and our tools can only detect and wait. Our own readiness check
still asks DUB alone, so it reports the lab not ready while both SJC pods run.

### Cited findings
- **[measured]** DUB answered 502 to every request from about 15:20Z to at least 19:04Z; with `Accept: application/json` the body is Cloudflare's JSON with `error_name` `origin_bad_gateway`, `error_category` `origin`, `retryable` true, `retry_after` 60 and `owner_action_required` true; a browser-like request gets the branded HTML page and curl the 16-byte text of M16. Source: `measurements-2026-10-10.md` M16 and M21 [S47] [S49]

  Implies: the failure is AMD's origin behind Cloudflare, not Cloudflare's edge, so nothing on our side fixes it (FCR). Our hub client already asks for JSON on every call (`lab/hub.py` L199-L200 at 97428e4), so a status poll can tell a front-door outage from a hub or pod error by `error_name` and `error_category` instead of the bare 502; it retries a 502 after 1, 2 and 4 s (`lab/hub.py` L57-L62 and L268), where Cloudflare asks for at least 60 s.

  Experiment: HUB-18 as specified in section 6, classifying each failed poll by `error_name` and `error_category`. Expected: every DUB failure today is `origin_bad_gateway`; no other metric moves. Minimum sample: every poll.

- **[practitioner]** Cloudflare's "Error responses" reference says structured responses cover the 1xxx errors and Cloudflare-generated 5xx (500, 502, 504, 520 to 526), while 5xx generated by the origin are passed through unchanged; the format follows the `Accept` header (`application/json` gives JSON, `text/html`, `*/*` or none gives HTML); `error_category` `origin` means the origin is responsible and the client should retry with backoff using `retry_after`; retryable errors also carry a `Retry-After` header. Source: [S50] Cloudflare docs, "Error responses", sections "Content negotiation", "Response fields", "Error categories" and "Retry-After header", https://developers.cloudflare.com/fundamentals/reference/error-responses/

  Implies: a 5xx without that JSON shape came from the hub or the pod itself, so the box-side client (R6) and the controller can split FCR into front-door, hub and pod failures from the body alone, and pace retries by `Retry-After`.

  Experiment: HUB-18, as in the previous finding.

- **[measured]** `python -m lab ready` from a clone of main answered `not_ready` with "hub account A answered 502" and "hub account B answered 502": it checks accounts A and B on DUB only (`lab/machine/ready.py` L37-L41 at 97428e4), and the session prompt of every lab sitting says to hand off when it reports not ready (`lab/machine/prompts/opus.md`, `fable.md` and `sonnet.md` L12). Source: `measurements-2026-10-10.md` M23 [S49] [S52]

  Implies: while DUB is down every pool sitting hands off at its first step, including those that need no pod, although both SJC pods run; task O54 moves `lab/hub.py` and the controller to a hub per account, and its Done when does not name `ready.py` (FCR of the lab's own checks).

  Experiment (HUB-21): once `ready.py` reads the hub per account, run `python -m lab ready` with the C and D keys while DUB is down. Expected: ready answers ok; the readiness check's failure rate on SJC falls from 1.0 to 0. Minimum sample: 3 runs.

## 10. What the day's live runs settled on SJC (plumbing pass 3, 2026-10-10)

### Takeaway
By evening every part of a pod's life on SJC had been run once from scripts: the controller's web-form launch (R13),
its keep-alive (R13), the read of use on an unmetered hub (R17), the teacher's build, download and start on pod D
(R30) and its start through the proxy (R18). What no run has produced yet is one number from a stopped account to
the first answer of the model server (T8), or a reading of running minutes under the name owner ruling 6 gives it.

### Cited findings
- **[measured]** On pod D (SJC account D, Strix Halo, gfx1151) the controller's web-form launch gave a ready pod 5.55 s after the launch request, the first request through the hub (`GET /user/<acct>/api/me`, 0.21 s) came 11.7 s after the launch call began, `keep-alive --interval 600 --minutes 70` made 7 pings with 0 failures and the pod was still running afterwards, and credits per hour read 0 (quota disabled, `strix-halo` listed at 3 a minute). Source: `lab/refine/metrics.jsonl:38`, `lab/refine/metrics.jsonl:39`, `lab/refine/metrics.jsonl:40` and `lab/refine/metrics.jsonl:41` [S65]; `lab/README.md`, section "Controlling the pods: `lab/controller.py`", paragraph "Measured live (R13, ...)" [S66]

  Implies: on SJC the launch costs seconds, not the R9700's 15 s (`metrics.jsonl:18`, old, other chip), and ruling 1's ping every 10 minutes holds a pod past the culler's hour (CPH, TTS); pod.launch_to_serving_s still has no reading, because pod B had no model server when R13 ran ("pod B has no serving script yet"), so T8 has its first part (launch to ready) and not its whole.

  Experiment (HUB-22): on pod D, from a stopped account and inside one GPU Queue slot, run the controller's form launch, then the teacher's serve step through `run-script` (the build and checkpoint R30 left in the home), and poll `GET /v1/models` through the proxy until the first 200; record pod.launch_to_serving_s with the hub, account, launch path and whether the compile cache was warm. Expected: about 5.55 s to ready plus about 294 s to the first 200 (`metrics.jsonl:53`), so about 300 s warm; the cold case adds R30's 523 s engine start in place of the 294 s. Minimum sample: 2 warm runs.

- **[measured]** On pod D the R30 teacher (vLLM 0.19.2+gfx1151, RedHatAI/Qwen3.8-27B-INT4) started with a warm compile cache answered `GET /v1/models` through the proxy 294.1 s after its serve step was sent, with Jupyter Server's 500 page on every 5 s poll until then, and its first, cold start was ready 523 s after launch; R30 found pod D's Jupyter root to be `/home/jovyan` on the persistent 984 GB volume, so the build (`~/vllm`) and the checkpoint (`~/models`) it left there are in the home. Source: `lab/refine/metrics.jsonl:52` and `lab/refine/metrics.jsonl:53` [S65]; R30 task page, lines of Sat 5:40 PM and 6:10 PM ET (Hemdutt's agent) [S61]; `measurements-2026-10-10.md` M25 [S63]

  Implies: the serving half of pod.launch_to_serving_s is about 294 s warm and 523 s cold on gfx1151, against the R9700's 270 s engine init (`metrics.jsonl:9`, old, other chip); and on the form-launched SJC pods a file written to the Jupyter root lands in the persistent home, so scenario P14's worry from pass 2 (the root's `vllm` and `models` lost at a relaunch) does not hold there, though no relaunch of pod C or D has yet been watched with the build in place (TTS, RT).

  Experiment: HUB-22, as in the previous finding; its second warm run follows a stop and form relaunch of pod D, which also shows that `~/vllm` and `~/models` survive a relaunch.

- **[practitioner]** `python -m lab credits` on an unmetered hub reads the hub's usage statistics (`GET /hub/api/stats/me`) and writes, per account, `pod.usage_min_today`, `pod.sessions_today` and `pod.usage_min_window` (the minutes of every session that overlaps the window), counting the running session from its start; it writes no metric named pod.running_minutes. Source: `lab/credits.py`, module docstring and `_assess_unmetered`, main at fba248a [S64]

  Implies: owner ruling 6 judges T3 on an unmetered hub by running minutes per account from each server's `started` time, under the name pod.running_minutes, while the merged tracker reads the same use from the hub's statistics under other names; the two agree for a single running session and differ when a window holds a stopped session as well (pod D was stopped and relaunched at 21:24Z, M25), so the metric set needs one name and one definition before T3 has a reading (CPH).

  Experiment (HUB-23): read each account's `servers[""].started` from `GET /hub/api/users/<name>` by the hub's clock and, at the same minute, `python -m lab credits --account C,D --window-hours 24`; record pod.running_minutes per account with the hub and account in the conditions, beside the tracker's pod.usage_min_window. Expected: pod.running_minutes equals minutes since `started` (331.7 for C and 194.8 for D at 00:38Z, M25); pod.usage_min_window is larger for D by the session R13 stopped. Minimum sample: 1 read per account.

- **[measured]** `python -m lab ready` from a clone of main at fba248a still answers not_ready with "hub account A answered 502" and "hub account B answered 502" and the DUB address, while `jetson/egress.py` now admits the aupcloud.io host of every `AUP_HUB_URL_<X>` and the controller, the credit tracker and the package pull take a hub per account. Source: `measurements-2026-10-10.md` M27 [S63]; `jetson/egress.py` L56-L61 and L119-L120, main at fba248a [S64]

  Implies: for scenario P15 (a hub site lost for hours, the pods moved to another site's accounts, owner ruling 7) the box's path and the pods' tools have moved with the accounts, and the lab's own readiness check has not, so every lab sitting that starts with `python -m lab ready` still hands off while both SJC pods run (FCR of the lab's own checks).

  Experiment: HUB-21, as in section 9.

## Open questions

- Is the Dublin outage the hub process itself or only the route from Cloudflare to it? It decides whether the idle culler keeps running during an outage (`idle-credits-recovery.md` section 5).
- Does the live build carry AUP's `last_activity_interval = 60`? M13 dates the build between 2026-06-25 and 2026-07-06, older than the 9b4af19 config read here.
- Does a job's output count as terminal activity when no browser is attached to its terminal, or only websocket traffic?
- Partly answered by section 9 (M21): Cloudflare puts the DUB outage on the origin, but whether the hub process, its ingress or the route behind Cloudflare failed, and so whether DUB's culler ran during it, is still not visible from outside.
- Does the SJC build keep the chart's idle culler (3600 s, checked every 600 s)? Its spawn bundle matches DUB's (M18), which suggests the same chart, but no SJC server has been left idle and watched.

## Search log

- 2026-10-10, plumbing pass 1: budget about 2 h (`lab/refine/RESEARCH_FORMAT.md` section 8), used about 35 minutes (11:58 AM to 12:12 PM ET, workstation clock). Starting points: the rubric's known open items 1, 2, 3, 6 and 9, the Gaps sections of these notes, and the 502 outage the R9 task page reports. Opened: the live front door (M16) and three public `Date` headers (M17); Cloudflare's "Error 502 or 504" page (S44); CPython `Lib/http/server.py` at v3.12.11, v3.13.7 and main (S45); AUP's `jupyterhub_config.py` at 9b4af19 (S36) and z2jh 4.3.3's (S46). Re-opened before a change rested on them: jupyter-server-proxy v4.6.0 `handlers.py` (S25), Tornado v6.5.10 `simple_httpclient.py` (S27) and Cloudflare's "Connection limits" (S15). Stopped when those last three repeated what `proxy-path.md` already says.
- 2026-10-10, plumbing pass 2: budget about 2 h, used about 15 minutes (2:57 to 3:12 PM ET, workstation clock, which runs about 46 s slow, M22). Starting points: the move to the SJC hub (Ilan's Team Comms FYI of Sat 2:47 PM ET, task O54), the DUB outage, R9's merged controller, this note's open questions, and E1, which merge 1 gave to this track. Opened: read-only probes of both hubs (M18 to M23: unauthenticated front-door reads, hub API reads for accounts C and D, two reads of pod C's Jupyter server); Cloudflare's "Error 502 or 504" page again (S44); the `type` URL in Cloudflare's JSON error (404, not cited); Cloudflare's "Error responses" reference (S50), found through one web search whose result snippets were not cited; vLLM's `vllm/config/cache.py` at b1388b1 (S51); `lab/machine/ready.py`, the sitting prompts, `lab/hub.py`, `lab/controller.py`, `jetson/egress.py` and `lab/pod_a_teacher.sh` on main at 97428e4 (S52 to S55). Stopped because the step's questions were answered by our own probes and the last two web sources added only the error format, not new limits.
- 2026-10-10, plumbing pass 3: budget about 2 h (`lab/refine/RESEARCH_FORMAT.md` section 8), used about 40 minutes (8:30 to 9:10 PM ET, workstation clock, which runs about 47 s slow, M26). Starting points: plumbing rulings 6 to 9 (answered in d7e9558), the open findings and "Left for the next pass" of pass 2, the day's merged measurements (R13, R17, R18, R29, R30, T6), the tasks that shipped the link and the pods' tools (R6, R10, R31, O55, R46), and scenario P15. Opened: read-only probes of SJC accounts C and D and of each pod's Jupyter API (M25 to M27); `lab/refine/metrics.jsonl` lines 38 to 58 and the R13, R18 and R30 sections of `lab/README.md` (S65, S66); `lab/credits.py`, `lab/package_pull.py`, `lab/transfer.py` and `jetson/egress.py` on main at fba248a (S64); the R10, R13, R14, R15, R17, R18, R30, R31 and R43 task pages (S61, S67); jupyter_server's `largefilemanager.py` at v2.19.0 and v2.20.0 and `serverapp.py` at v2.20.0 (S68), re-opened under owner ruling 8; Cloudflare's "Connection limits" (S15), re-opened for E8. Stopped because the step's questions were answered by our own runs and records, and the two re-opened web sources repeated what `proxy-path.md` and `transfers.md` already say.
