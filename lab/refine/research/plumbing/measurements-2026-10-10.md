# Plumbing measurements, 2026-10-10 (read-only probes of the AUP hub)

This file is the measurement record the plumbing notes cite as `measured` (it stands in for ledger
lines until the lab ledger exists). Every probe was a read-only HTTP GET from the Windows
workstation on the venue network to `https://dub.aupcloud.io`, using the account tokens from the
workspace `.env` in an `Authorization: token ...` header and a browser-like User-Agent. Nothing was
launched, stopped, written or generated. Pod B was never contacted (hub API reads only for account
B). Tokens and account names are redacted as `<acct>`. Times are UTC; the hub's clock ran 46 s ahead
of the workstation's (M14), and hub-side timestamps below are in hub time.

## M1. Front door and hub version (probe 1, 11:00:10 local)

`GET /hub/api/` with no auth returned 200 in 136 ms with body `{"version": "5.4.4"}` and headers
`server: cloudflare`, `x-jupyterhub-version: 5.4.4`. Every other response in these probes (hub API,
the user server's API, proxied vLLM responses, static files) also carried `server: cloudflare`.

## M2. Hub API latency (probe 1)

`GET /hub/api/users/<acct>` (account A), 10 sequential calls: p50 146 ms, min 141 ms, max 149 ms.

## M3. Account A server state (probes 1 and 2)

Probe 1 (hub time about 11:00:56): server `''` ready, not pending, `started` 2026-10-10T09:27:24Z,
`user_options` = `{runtime_minutes: 600, resource_type: gpu, gpu_selection: r9700}`, server
`last_activity` 2026-10-10T10:22:21Z, so the teacher pod had been idle for about 38.6 minutes with
nothing keeping it alive. `GET /user/<acct>/api/status` agreed: `connections 0, kernels 0,
last_activity 10:22:21Z, started 09:27:27Z`.

## M4. Account B server state (probes 1, 2, 5)

No running server. `GET /hub/api/users/<acct>?include_stopped_servers=1` returned server `''` with
`stopped: true`, `started: null`, `last_activity` 2026-10-10T10:53:38Z (the stop time) and
`user_options` `{runtime_minutes: 600, resource_type: gpu, gpu_selection: r9700}`. The quota
transaction for that stop (M5) reads `Session 776: 78 min @ r9700 (4/min)`, amount -312, at
10:53:37. Who or what stopped it cannot be read from the hub API.

## M5. Quota and transactions (probe 2, 11:02:35 local)

`GET /hub/api/quota/me`: account A balance 59,960, account B 59,628, `rate_r9700` 4, quota enabled,
not unlimited. Unchanged on every read between 11:00 and 11:10 while account A's server ran.
`GET /hub/admin/api/quota/<acct>` (own account) `recent_transactions`:

| Account | Amount | Type | Before -> after | Hub time | Description |
|---|---|---|---|---|---|
| A | -12 | usage | 59,972 -> 59,960 | 2026-10-10 09:12:45 | Session 773: 3 min @ r9700 (4/min) |
| A | -28 | usage | 60,000 -> 59,972 | 2026-10-10 01:21:32 | Session 758: 7 min @ r9700 (4/min) |
| A | 60,000 | set | 0 -> 60,000 | 2026-10-08 13:59:55 | Balance set to 60000 |
| B | -312 | usage | 59,940 -> 59,628 | 2026-10-10 10:53:37 | Session 776: 78 min @ r9700 (4/min) |
| B | -60 | usage | 60,000 -> 59,940 | 2026-10-10 03:01:47 | Session 759: 15 min @ r9700 (4/min) |
| B | 60,000 | set | 0 -> 60,000 | 2026-10-08 13:59:57 | Balance set to 60000 |

Account A's running session (started 09:27:24) had no transaction yet after about 96 minutes.

## M6. Pod A's Jupyter server (probe 1)

`GET /user/<acct>/api` returned `{"version": "2.19.0"}` (jupyter_server 2.19.0).
`GET /user/<acct>/lab/api/extensions` listed `jupyterhub-jupyter-server-proxy==4.6.0`.
`GET /user/<acct>/api/contents/` (the Jupyter root) returned 31 entries, all files uploaded by the
R2 session (scripts, logs, two diffs), largest 34,177 bytes, and no `home` entry, so the root is not
`/` (it is the ephemeral `/opt/workspace` recorded in `lab/README.md`).
`GET /user/<acct>/api/terminals` listed at least five terminals with `last_activity` between 09:34Z
and 09:37Z, still open about 85 minutes later.

## M7. Proxy latency to the teacher (probe 1)

`GET /user/<acct>/proxy/8000/v1/models`, 20 sequential calls 0.5 s apart: all 200, p50 138 ms,
p95 142 ms, min 135 ms, max 143 ms, model ids `['teacher']`, headers `transfer-encoding: chunked`,
`server: cloudflare`.

## M8. vLLM metrics through the proxy (probe 1)

`GET /user/<acct>/proxy/8000/metrics`: 200 in 182 ms, 49,760 bytes. Selected lines:
`vllm:num_requests_running{...} 0.0`, `vllm:num_requests_waiting{...} 0.0`,
`vllm:time_to_first_token_seconds_count{...} 5.0`, `vllm:time_to_first_token_seconds_sum{...} 9.83`,
`vllm:inter_token_latency_seconds_count{...} 581.0`, `vllm:e2e_request_latency_seconds_count{...} 5.0`.

## M9. Failure modes on the proxy path (probe 1, all GET, redirects not followed)

| Case | Request | Result |
|---|---|---|
| No token | `GET /user/<acct>/proxy/8000/v1/models` | 302 to `/hub/api/oauth2/authorize?client_id=jupyterhub-user-<acct>&...`, 136 ms |
| Random bad token | same | 302 to the same OAuth URL, 139 ms |
| Bad token, `Accept: application/json` | same | 302 to the same OAuth URL, 139 ms |
| Absolute form, good token | `GET /user/<acct>/proxy/absolute/8000/v1/models` | 404 `{"detail":"Not Found"}` (application/json, from vLLM), 144 ms |
| Nothing listening on the port | `GET /user/<acct>/proxy/65111/` | 500, Jupyter Server HTML error page, 152 ms |
| Hub API, bad token | `GET /hub/api/users/<acct>` | 403 `{"status": 403, "message": "Missing or invalid credentials."}`, 138 ms |
| Another user's URL, good token | `GET /user/atlas-probe-no-such-user/api/status` | 302 to `/hub/user/atlas-probe-no-such-user/api/status`, 136 ms |

## M10. Activity propagation to the hub (probes 1, 2, 4, 5)

Probe 1 sent proxied requests to pod A between about 11:00:56 and 11:01:14 hub time. Probe 2, at
11:03:21 hub time, read server `last_activity` = 11:01:14.67, so the hub learned of the traffic
within 127 s. Probe 4 (files, about 11:04:44 hub time) showed up as 11:04:47 in probe 5. Probes 2, 3
and 5 used only hub API endpoints: they moved the user-level `last_activity` (token use) but not the
server-level `last_activity`.

## M11. Token metadata (probe 3, `GET /hub/api/users/<acct>/tokens`, values never returned)

| Account | Id | Note | Created | expires_at | Scopes (abridged) |
|---|---|---|---|---|---|
| A | a177 | atlas controller (Knighthacks26) | 08:24:47Z | None | servers, delete:servers, access:servers, tokens, read:users, read:servers, users:activity, ... (all `!user=<acct>`) |
| B | a178 | atlas controller (Knighthacks26) | 08:24:49Z | None | same set for account B |
| A | a179 | Server at /user/<acct>/ | 09:27:24Z | None | users:activity, read:users:activity, read:users:groups, access:servers!server=<acct>/, read:users:name |

## M12. Pulling a file through `/files/` (probe 4)

The largest file at pod A's Jupyter root (34,177 bytes): `GET /user/<acct>/files/<name>` with
`Range: bytes=0-1023` returned 206, 1,024 bytes, `content-range: bytes 0-1023/34177`,
`accept-ranges: bytes`, `server: cloudflare`. Three full GETs took 0.16 to 0.17 s each, which is
latency-bound, so no throughput figure can be drawn from them.

## M13. Which AUP build Dublin runs (static file, no auth)

`GET /hub/static/spawn-ui/assets/index.js` returned 200, 218,673 bytes. It contains `repo_persist`
(1 occurrence), `gpu_selection` (2) and `allowGitClone` (4). In the AUP repository `repo_persist`
entered the spawn app in commit 6e09fad (2026-06-25), and the base image's `WORKDIR` moved from
`/opt/workspace` to `/home/jovyan` in b5b9b7f8 (2026-07-06). The live pod still roots Jupyter at
`/opt/workspace` (M6), so the Dublin build is consistent with a version from between those two dates.

## M14. Clock offset (probe 2)

Hub `Date` header `Sat, 10 Oct 2026 11:03:23 GMT` against the workstation's UTC clock
`11:02:37`: the hub is about 46 s ahead.

## M15. Pod A stopped about 69 minutes after its last activity (probe 6, about 13:52 UTC, hub API only)

`GET /hub/api/users/<acct>?include_stopped_servers=1` for account A: server `''` `stopped: true`,
`last_activity` 2026-10-10T12:13:39Z (the stop time). The last activity recorded before it was
11:04:47Z (M10), so the server stopped 68.9 minutes after its last activity, with nothing issuing a
stop that we know of. Account A's balance fell from 59,960 to 59,296 (-664 = 166 minutes x 4);
09:27:24 to 12:13:39 is 166.25 minutes. Account B was unchanged (59,628).

## M16. The front door answers Cloudflare 502 on every path (plumbing pass 1, 15:58 to 16:04 UTC)

`python -m lab ready` (about 15:58 UTC, workstation clock) reported `hub account A answered 502` and
`hub account B answered 502` for its authenticated hub API reads. At 16:03:52 to 16:03:53 workstation
time, unauthenticated GETs of `/hub/api/`, `/hub/static/spawn-ui/assets/index.js` and `/hub/login`
each returned 502 in 0.36 to 0.39 s, with `Content-Type: text/plain; charset=UTF-8`,
`Server: cloudflare`, a `CF-RAY` ending in `-ATL`, no `x-jupyterhub-version` header, and the 16-byte
body `error code: 502`. The R9 task page's lock line of Sat 11:24 AM ET already says the hub is
answering 502. No proxied call was made: both servers had stopped before the outage (M4, M15).

## M17. The 46-second offset is the workstation's clock (plumbing pass 1, 16:04 UTC)

At 16:04:03 workstation time the `Date` headers of `https://www.google.com`, `https://github.com` and
`https://www.cloudflare.com` read 16:04:49, 16:04:47 and 16:04:50 GMT, and the Cloudflare 502
responses of M16 carried 16:04:39 to 16:04:40 against 16:03:52 to 16:03:53 local. The workstation
clock is 44 to 47 s behind all four, so the 46 s of M14 is this machine running slow, not the hub
running fast.

## M18. The SJC hub's front door (plumbing pass 2, 19:01 UTC)

The pods moved to AMD's second hub, `https://sjc.aupcloud.io` (accounts C and D), after the DUB
outage of M16. At 19:01:28 workstation time an unauthenticated `GET /hub/api/` returned 200 in
0.181 s with body `{"version": "5.4.4"}` and headers `server: cloudflare`, a `cf-ray`, and
`x-jupyterhub-version: 5.4.4`. `GET /hub/static/spawn-ui/assets/index.js` returned 200 in 0.426 s,
218,673 bytes, with `repo_persist` (1 occurrence), `gpu_selection` (2) and `allowGitClone` (4): the
same size and markers as DUB's bundle in M13, so SJC runs the same AUP hub build behind the same
Cloudflare front.

## M19. SJC accounts C and D (plumbing pass 2, hub API reads only, 19:01 UTC)

`GET /hub/api/users/<acct>` (no `include_stopped_servers`): for account C, server `''` ready, not
pending, `started` 2026-10-10T18:54:27Z, server `last_activity` 19:01:09Z (hub time),
`user_options` `{runtime_minutes: 2400, resource_type: gpu, gpu_selection: r9700}`; for account D,
ready, not pending, `started` 18:54:59Z, `last_activity` 19:00:03Z, the same `user_options`. Both
answered in 0.19 to 0.31 s with `server: cloudflare`. `GET /hub/api/quota/me` for both: `balance` 0,
`unlimited` true, `enabled` false, and the same rate table as DUB (`r9700` 4, `9070xt` 4,
`strix-halo` 3, `strix` 2, `phx` 2, `cpu` 1). These reads touch only the hub API, which does not
move a server's `last_activity` (M10).

## M20. Pod C's Jupyter server on SJC (plumbing pass 2, 19:01 to 19:03 UTC)

`GET /user/<acct>/api` returned `{"version": "2.21.1"}` (jupyter_server 2.21.1; DUB's pod A ran
2.19.0, M6) in 0.109 s. `GET /user/<acct>/api/contents/` (the Jupyter root) listed 9 entries at
19:01 and 11 at 19:03: hour-0 and GPU probe scripts and their outputs written since 18:59 (by
another session) plus two directories, `checkpoints` and `outputs`, whose `last_modified` is
2026-09-22 (12:45:41Z and 13:02:10Z), and no `home` or `opt` entry. Directories dated weeks before
the account existed come from the image rather than a fresh home volume, which points to the root
being the image's working directory again (ephemeral, as `/opt/workspace` on DUB) and not
`/home/jovyan`; the contents API does not show an absolute path, so this is not proven. These three
requests went through the pod's Jupyter server and count as activity on pod C.

## M21. DUB still down, as Cloudflare classifies it (plumbing pass 2, 19:01 to 19:04 UTC)

Every request to `https://dub.aupcloud.io/hub/api/` still answered 502, so the outage that began
about 15:20 UTC (M16) had lasted at least 3 h 40 min. The body depends on the request headers. A
client sending a browser User-Agent and `Accept: */*` (httpx) got Cloudflare's branded HTML error
page (6,432 bytes) that reads "Browser Working", "Atlanta Cloudflare Working" and
"dub.aupcloud.io Host Error". With `Accept: application/json` it got an 869-byte JSON body:
`"error_code": 502`, `"error_name": "origin_bad_gateway"`, `"error_category": "origin"`,
`"detail": "The origin web server returned an invalid or incomplete response to Cloudflare. This
typically indicates the origin is overloaded or misconfigured."`, `"retryable": true`,
`"retry_after": 60`, `"owner_action_required": true`, `"cloudflare_error": true`,
`"zone": "dub.aupcloud.io"`. curl, with its own or a browser User-Agent, still got the 16-byte
`error code: 502` text of M16. The `type` URL in the JSON body
(`https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502/`)
answered 404 when opened.

## M22. The workstation clock is still slow (plumbing pass 2, 19:01 UTC)

The `Date` headers of the SJC and DUB answers in M18 to M21 ran 45.4 to 46.1 s ahead of the
workstation clock, and `https://www.google.com` 47 s ahead (19:02:18 GMT against 19:01:31 local),
the same offset as M17.

## M23. The lab's readiness check reads DUB only (plumbing pass 2, about 18:59 UTC)

`python -m lab ready` from a fresh clone of main at 97428e4 answered
`{"ok": false, "ready": false, "outcome": "not_ready", "reasons": ["hub account A answered 502",
"hub account B answered 502"], ...}` with `"hub": {"url": "https://dub.aupcloud.io", ...}`: it
checks accounts A and B on DUB only (`lab/machine/ready.py`, `ACCOUNTS`, `REQUIRED_ENV` and
`HUB_URL`), so it reports the lab not ready while both SJC pods run.

## M24. The SJC pods after their relaunch as Strix Halo (plumbing pass 2, 19:31 UTC)

The dispatcher reports that SJC ignored the REST launch (a stock image with no GPU) and that both pods were
relaunched through the hub's web spawn form, as Strix Halo (Ilan1 and Ilan2 dispatcher page, lines of Sat 3:46
and 4:00 PM ET by that page's clock), so M19 and M20 describe the first, REST-launched pods. At 19:31:06 to
19:31:09 workstation time, hub API reads showed account C's server ready since 19:07:12Z and account D's since
19:07:32Z, both with `user_options` `{runtime_minutes: 2400, resource_type: gpu, gpu_selection: strix-halo}`;
server `last_activity` 19:29:51Z for C and 19:08:32Z for D; quota as in M19 (balance 0, unlimited true, enabled
false; `strix-halo` rate 3 in the table). `GET /user/<acct>/api` on pod C returned `{"version": "2.20.0"}`, and its
Jupyter root listed 20 entries with the directories `checkpoints`, `lab`, `models`, `outputs`, `proxytest`,
`serve` and `vllm`, and no `home` or `opt` entry.
