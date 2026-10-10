# The box-to-pod call path through the hub proxy: latency, streaming, limits, failure modes

Research note for the plumbing track (O3 rubric tests T5, T6, and T4 for token handling), opened 2026-10-10;
sources indexed in `SOURCES.md`. Question: what sits between the Jetson and a model server on a pod at
`https://dub.aupcloud.io/user/<account>/proxy/<port>/`, which limits and failure codes does each hop impose,
and why do server-sent events (SSE) arrive in one burst?

Labels, keys and metric names as in `hub-api-lifecycle.md`: `measured` = observed on a running system with the
setup (ours: `measurements-2026-10-10.md`, M1 to M15); `practitioner` = documented behavior or recommendation
from the people who build or run the system (vendor docs, maintainers' source at a pinned version,
maintainers' PRs); `anecdote` = one report without data; `opinion` = argument without evidence. Metrics (O3
rubric [R8]): CPH `pod.credits_per_hour`, TTS `pod.launch_to_serving_s`, MB/s `link.push_mb_s` and
`link.pull_mb_s`, RT `pod.recovery_s`, FCR `link.failed_call_rate`. Generation experiments on pod A need
Ilan's go even though pod A is a shared service.

## 1. The hops

### Takeaway
A box call goes Jetson, Cloudflare, the hub's proxy (configurable-http-proxy in AUP's chart), the pod's Jupyter
server running jupyter-server-proxy, then the model server. Cloudflare was not in any of our planning notes, and
its limits are part of the path. The proxy itself costs nothing measurable: a proxied GET is as fast as a hub
API GET.

### Cited findings
- **[measured]** Every response from `dub.aupcloud.io` carried `server: cloudflare`: the hub API, the user server's API, vLLM responses through the proxy and the hub's static files. Source: `measurements-2026-10-10.md` M1, M7 and M12 [R1]

  Implies: Cloudflare's request-size and timeout limits apply to every push, pull and model call (MB/s, FCR); `lab/README.md` and the R6 task describe the path without it.

  Experiment (PRX-1): log the `server` and `cf-ray` headers on 20 proxied calls from the Jetson. Expected: Cloudflare on all 20 and a single edge location. Minimum sample: 20 calls.

- **[practitioner]** AUP's chart runs `quay.io/jupyterhub/configurable-http-proxy:5.2.0` as the hub proxy, with the hub as `--default-target` and `/hub/error` as `--error-target`. Source: AUP `runtime/chart/values.yaml`, `proxy.chp.image`, L344-L349, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/chart/values.yaml#L344-L349 ; `runtime/chart/templates/proxy/deployment.yaml` L94-L105, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/chart/templates/proxy/deployment.yaml#L94-L105 [S37]

  Implies: CHP's error codes (section 4) and its route-activity rule (`idle-credits-recovery.md`) apply, unless Dublin overrides the chart default, which cannot be seen from outside.

  Experiment: none possible from outside; the 503 and 404 page bodies seen in PRX-9 will show whether CHP answers.

- **[measured]** From the workstation, 20 sequential `GET /user/<acct>/proxy/8000/v1/models` calls all returned 200 with p50 138 ms and p95 142 ms (min 135, max 143); 10 hub API GETs had p50 146 ms. Source: `measurements-2026-10-10.md` M2 and M7 [R1]

  Implies: the proxy adds no measurable overhead over a hub API call; about 140 ms is the floor for any box-to-lab call from the venue, so a 1.09 s caption TTFT on the pod (R2) is at least about 1.23 s on the box (TTS of a caption, FCR budgets).

  Experiment (PRX-2): the same 20-call loop from the Jetson, then 100 calls spread over a day. Expected: p50 within 20 ms of 138 ms if the Jetson shares the venue uplink. Minimum sample: 100 calls.

## 2. Why SSE arrives in one burst, and the fix

### Takeaway
jupyter-server-proxy buffers the whole upstream response unless the request's `Accept` header is exactly
`text/event-stream`. Our burst is the proxy's buffered mode, not vLLM. Sending that header with `stream: true`
should stream token by token. Cloudflare is a second hop that could still buffer, and that has not been tested.
TTFT can already be read from vLLM's `/metrics` through the proxy.

### Cited findings
- **[practitioner]** jupyter-server-proxy 4.6.0 picks its mode per request: if `self.request.headers.get("Accept") == "text/event-stream"` (string equality) it proxies "progressively", writing and flushing every chunk from a `streaming_callback`; otherwise Tornado's client fetches the whole response and only then writes it. Source: jupyter-server-proxy v4.6.0 `jupyter_server_proxy/handlers.py`, `ProxyHandler.proxy` L437-L443, `_proxy_progressive` L445-L525, `_proxy_buffered` L527-L560, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L437-L560 [S25]

  Implies: the R2 observation "the proxy buffers SSE, so TTFT is only measurable on the pod" is the buffered branch. A client that sends `Accept: */*` (httpx's default), `application/json`, or `text/event-stream, application/json` still gets a buffered response; only the exact header streams (box-side first-token time, and FCR when long buffered calls time out).

  Experiment (PRX-3): the same streamed chat completion through the proxy with `Accept: text/event-stream` and with `Accept: */*`, logging each chunk's arrival time. Expected: the first chunk at about pod TTFT + 140 ms with the exact header, every chunk at completion without it. Minimum sample: 10 per arm.

- **[practitioner]** Streaming came from PR #479 "feat: support stream api" (merged 2024-08-29, first released in v4.4.0), whose description reads "the normal request works fine, but the stream API never response"; issue #502 (open) says only `text/event-stream` is supported, not other stream types. Source: jupyter-server-proxy PR #479, https://github.com/jupyterhub/jupyter-server-proxy/pull/479 ; changelog v4.4.0, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/docs/source/changelog.md ; issue #502, https://github.com/jupyterhub/jupyter-server-proxy/issues/502 [S40] [S41]

  Implies: the pod needs jupyter-server-proxy 4.4.0 or later (we have 4.6.0, M6); a chunked file download is never streamed by the proxy, whatever its headers.

  Experiment: PRX-3.

- **[practitioner]** The rendered docs page "Accessing Arbitrary Ports or Hosts" documents `/proxy/<port>` (prefix removed before forwarding) and `/proxy/absolute/<port>` (the server receives the full URL) and says nothing about streaming, buffering or size limits. Source: jupyter-server-proxy docs, "Accessing Arbitrary Ports or Hosts", https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html [S04]

  Implies: the header rule lives only in the source, the changelog and the PR; our own README should state it so nobody rediscovers the burst (FCR).

  Experiment: none (documentation fact).

- **[measured]** `GET /user/<acct>/proxy/absolute/8000/v1/models` returned 404 `{"detail":"Not Found"}` from vLLM while `/proxy/8000/v1/models` returned 200. Source: `measurements-2026-10-10.md` M9 [R1]

  Implies: the "absolute form 404s" note in `lab/README.md` is the backend rejecting the prefixed path, as the docs describe; never use the absolute form for vLLM (FCR).

  Experiment: none needed.

- **[practitioner]** vLLM exposes Prometheus metrics at `/metrics` on its OpenAI-compatible server, including the histogram `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:inter_token_latency_seconds` and the gauges `vllm:num_requests_running` and `vllm:num_requests_waiting`. Source: vLLM docs, "Production Metrics", "General Metrics", https://docs.vllm.ai/en/latest/usage/metrics.html [S10]

  Implies: TTFT measured on the pod can be read from anywhere by differencing the histogram's `_sum` and `_count`, with no streaming needed.

  Experiment: PRX-4.

- **[measured]** `GET /user/<acct>/proxy/8000/metrics` returned 200 in 182 ms (49,760 bytes) and showed `vllm:time_to_first_token_seconds_count 5` with `_sum 9.83`, `vllm:num_requests_running 0` and `vllm:num_requests_waiting 0`. Source: `measurements-2026-10-10.md` M8 [R1]

  Implies: the box can measure the teacher's TTFT today (mean over the 5 requests served since the engine started: 1.97 s, warm-up included) and can tell whether the teacher is working (for keep-alive, see `idle-credits-recovery.md`).

  Experiment (PRX-4): read `/metrics` before and after 20 caption calls. Expected: the batch's mean TTFT within 20% of the 1.09 s R2 image figure. Minimum sample: 20 calls.

## 3. Time and size limits on the path

### Takeaway
Three limits bound a single call. Cloudflare answers 524 if the origin sends nothing for 125 s, jupyter-server-proxy
gives a buffered request 300 s, and Tornado's HTTP client inside the proxy refuses any response larger than
100 MB, streamed or not. Long non-streamed generations and large pulls through `/proxy/<port>/` will fail.

### Cited findings
- **[practitioner]** Buffered proxy requests use `connect_timeout=250.0, request_timeout=300.0`; progressive requests override them to 600 s and 7200 s; a timeout comes back as HTTP 599. Source: jupyter-server-proxy v4.6.0 `proxy_request_options` L712-L717, `_proxy_progressive` L497-L508 and `_proxy_buffered` L536-L542, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L712-L717 [S25]

  Implies: a non-streamed call longer than 300 s fails with 599 at the proxy (FCR); in practice Cloudflare's limit (next finding) fails it first.

  Experiment: PRX-5.

- **[practitioner]** Cloudflare returns 524 when the origin has not answered within the Proxy Read Timeout of 125 s (adjustable only on Enterprise, up to 6,000 s); the Proxy Write Timeout is 30 s, and idle origin connections close after 900 s (520). Source: Cloudflare docs, "Connection limits", https://developers.cloudflare.com/fundamentals/reference/connection-limits/ ; "Error 524", https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/ [S15] [S17]

  Implies: in buffered mode the proxy sends nothing until vLLM finishes, so any non-streamed generation longer than about 125 s (roughly 950 tokens at the measured 7.7 tok/s) fails with 524 (FCR); a streamed call sends its first bytes at TTFT and should not. Cloudflare's own advice is status polling or streaming.

  Experiment (PRX-5): one non-streamed and one streamed chat completion sized to take about 150 s. Expected: 524 for the first, success for the second. Minimum sample: 3 per arm.

- **[practitioner]** Tornado's default `SimpleAsyncHTTPClient` refuses a response body larger than `max_body_size`, which defaults to `max_buffer_size` = 100 MB, and "with a `streaming_callback` only `max_body_size`" applies; the default is the same in Tornado 6.4.0, 6.5.5 and 6.5.10. jupyter-server-proxy builds `httpclient.AsyncHTTPClient(force_instance=True)` without setting either. Source: Tornado v6.5.10 `tornado/simple_httpclient.py`, `SimpleAsyncHTTPClient` docstring and `initialize`, L100-L141, https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/simple_httpclient.py#L100-L141 ; jupyter-server-proxy v4.6.0 `handlers.py` L437, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L437 [S27] [S25]

  Implies: pulling a model package (a Q4 GGUF plus projector is a few GB) through `/proxy/<port>/` from a pod-side file server fails in both modes, so R10's server must answer HTTP Range requests and the box must pull in pieces of at most 100 MB, or the pull should use `/files/` (see `transfers.md`) (MB/s, FCR).

  Experiment (PRX-6): serve a 150 MB file from a pod-side `python3 -m http.server`, GET it whole through the proxy, then in two Range requests of 75 MB. Expected: the whole GET fails (599 or a dropped connection), the ranged GETs succeed. Minimum sample: 3 per arm.

- **[anecdote]** Issue #281 (open since 2021, bumped again 2026-05-11): a user's downloads through jupyter-server-proxy failed at once with 599 and they traced it to Tornado's 100 MB limit ("the proxy requests come back immediatley 599 when the size of the file is >100MB"); a maintainer called it a valid use case; PR #282, which would have exposed the client options, was closed unmerged on 2024-06-12. Source: jupyter-server-proxy issue #281, https://github.com/jupyterhub/jupyter-server-proxy/issues/281 ; PR #282, https://github.com/jupyterhub/jupyter-server-proxy/pull/282 [S42] [S43]

  Implies: no configuration switch exists in 4.6.0; the limit has to be designed around (MB/s).

  Experiment: PRX-6 confirms it on our pods.

## 4. Auth and failure codes (T5)

### Takeaway
The rubric's expected failure map is wrong in two places. A bad or missing token never produces 401: GETs are
redirected (302) to the hub's login and other methods get 403. A stopped server produces 424, not 404; a 404
means the token has no access to that server. A client that follows redirects would see a 200 login page and
think the call worked.

### Cited findings
- **[measured]** On a running server: no token and a random bad token both returned 302 to `/hub/api/oauth2/authorize?...`, even with `Accept: application/json`; a port with nothing listening returned 500 (a Jupyter Server HTML page) in 152 ms; the absolute form returned the backend's 404; the hub API with a bad token returned 403 `{"status": 403, "message": "Missing or invalid credentials."}`; another user's `/user/<name>/...` URL returned 302 to `/hub/user/<name>/...`. Source: `measurements-2026-10-10.md` M9 [R1]

  Implies: "model server down while the pod is up" is a fast 500 and "bad token" is a 302 or 403, so the ledger can classify FCR by cause from the status code alone; clients must not follow redirects.

  Experiment (PRX-7): classify every non-2xx box call over the weekend by status. Expected: 302 or 403 = token, 424 = pod stopped, 500 = model server down, 503 = pod gone with its route still present, 524 = Cloudflare timeout, 599 = proxy timeout or size. Minimum sample: one weekend of calls.

- **[practitioner]** jupyter-server-proxy's `ProxyHandler.prepare` enforces auth with `web.authenticated`, "which does a 302 to the appropriate login url" (websocket upgrades get 403), and jupyter_server's `JupyterHandler.prepare` "redirects to the login page on GET and HEAD and otherwise raises 403". Source: jupyter-server-proxy v4.6.0 `ProxyHandler.prepare` L139-L170, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L139-L170 ; jupyter_server v2.19.0 `jupyter_server/base/handlers.py`, `JupyterHandler.prepare`, L598-L660, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/base/handlers.py#L598-L660 [S25] [S26]

  Implies: a POST chat completion with an expired or wrong token gets 403 and a GET health check gets 302; the box's retry logic must treat both as "fix the token", not "retry" (FCR).

  Experiment (PRX-8): one POST with a bad token through the proxy. Expected: 403 before the request reaches vLLM. Minimum sample: 3.

- **[practitioner]** When the user's server is not running, the hub's `UserUrlHandler` answers 424 (503 only if `use_legacy_stopped_server_status_code` is set) with JSON "JupyterHub server no longer running at ... Restart the server at ..." for non-GET requests and for GETs that accept JSON or have an `api` path segment; other GETs get a 424 HTML page, or a 303 to the spawn-pending page while a spawn is pending; a token without access to that server gets 404 "No access to resources or resources not found". Source: JupyterHub 5.4.4 `jupyterhub/handlers/base.py`, `UserUrlHandler`, L1625-L1840, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/handlers/base.py#L1625-L1840 [S21]

  Implies: "pod down" is 424, so the rubric's "404 when the server is down" should read 424, and a box health check should send `Accept: application/json` to get the JSON form (FCR, and RT detection).

  Experiment (PRX-9): after a planned stop, GET and POST the proxy URL. Expected: 424 JSON for the POST and for the GET with `Accept: application/json`. Minimum sample: 3.

- **[practitioner]** configurable-http-proxy 5.2.0 answers 404 when no route matches and 503 (through the error target) when a route's target does not respond; the hub notices a dead server on its next poll, every 30 s by default (`Spawner.poll_interval`). Source: CHP 5.2.0 `lib/configproxy.js`, `handleProxy`, L605-L680, https://github.com/jupyterhub/configurable-http-proxy/blob/5.2.0/lib/configproxy.js#L605-L680 ; JupyterHub 5.4.4 `spawner.py`, `poll_interval`, L558-L575, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/spawner.py#L558-L575 [S30] [S24]

  Implies: right after a pod dies, box calls get 503 for up to about 30 s, then 424 once the hub removes the route; detection of a lost pod (RT) can start at the first 503 instead of waiting for the hub.

  Experiment (PRX-10): stop a pod while polling its proxy URL once a second. Expected: 200, then 503 for up to 30 s, then 424. Minimum sample: 3.

- **[practitioner]** jupyter-server-proxy forwards every request header to the backend (`proxy_request_headers` copies `self.request.headers`), so the hub token in `Authorization: token ...` reaches the model server. Source: jupyter-server-proxy v4.6.0 `proxy_request_headers` L700-L706, https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py#L700-L706 [S25]

  Implies: a pod-side server that logs request headers (R10's file server, a debug vLLM) writes the hub token to the home volume, a T4 failure; vLLM's own `--api-key` cannot share the `Authorization` header with the hub token.

  Experiment (PRX-11): run the pod-side file server for 20 requests and grep its log for the token. Expected: 0 hits. Minimum sample: 20 requests.

- **[practitioner]** Our runtime client, `LlamaServerRuntime` in `runtime/llama.py`, builds `httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)` with `timeout_s=120.0` and posts non-streamed `/v1/chat/completions`; it has no way to add an `Authorization` header and no streaming path. Source: `runtime/llama.py`, `LlamaServerRuntime.__init__` L62-L78 and `_complete` L159-L168, main at bf482f4, https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/runtime/llama.py#L62-L168 [R6]

  Implies: as written, the box cannot call the teacher through the hub proxy at all (no token header) and can never stream; its 120 s timeout caps a teacher answer at about 900 tokens at 7.7 tok/s (FCR). The fix belongs to the Runtime tasks (R5, R6), not to this note.

  Experiment (PRX-12): once a headers parameter and an `Accept: text/event-stream` streaming path land, measure box-side TTFT on 20 caption calls. Expected: pod TTFT plus about 140 ms. Minimum sample: 20 calls.

## Gaps
No official Cloudflare page on how it treats `text/event-stream` was found (only second-hand search snippets,
not cited), so whether Cloudflare adds its own buffering on our zone is untested; PRX-3 answers it. AMD's
Cloudflare plan is unknown, and the 125 s and 100 MB figures assume a non-Enterprise plan. Whether Dublin runs
CHP or another proxy behind Cloudflare was not observable.
