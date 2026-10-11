# lab/ - the Strix Halo side (AUP Learning Cloud)

Two pods, one AMD Strix Halo each (Radeon 8060S, gfx1151, RDNA 3.5, 64 GB shared with the CPU), on the SJC hub
https://sjc.aupcloud.io: ROCm 7.13, torch 2.9.1+rocm7.13.0, triton 3.5.1, Python 3.12, image
`ghcr.io/amdresearch/auplc-base:latest`. Account C is **pod A, the teacher** (vLLM 0.19.2 built for gfx1151 serves
Qwen3.8-27B as model `teacher` on port 8000, behind the hub proxy; only called, never reserved). Account D is
**pod B, the exclusive card** (trainer: LoRA, eval, packaging, and the kernels experiments), used only through the
GPU Queue. Some notes and ledger rows call them pod C and pod D, after the account letters. The AMD Radeon AI PRO
R9700 (gfx1201, RDNA 4, 32 GB) on the DUB hub (accounts A and B) is **the other chip**: the scripts were first built
and measured there, and every R9700 number below is labelled as that chip's, never as the lab's baseline. Logins and
hub API tokens are on the Notion Credentials page and in your local `.env`; never put them in this repo.

## Which hub is live today (2026-10-10)

| Hub | Accounts | State |
|---|---|---|
| **SJC**, https://sjc.aupcloud.io | **C** and **D** | The lab: the two Strix Halo pods (launched with accelerator `strix-halo`, through the web spawn form). Quota disabled, nothing metered. JupyterHub 5.4.4, resource key `gpu` (image `ghcr.io/amdresearch/auplc-base:latest`), accelerators `r9700`, `9070xt`, `strix-halo`, `strix`, `phx`. |
| **DUB**, https://dub.aupcloud.io | **A** and **B** | The other hub: the R9700s (gfx1201), metered at 4 credits a minute. It answered 502 all day on 2026-10-10 and is kept only for when it answers again. |

Some sections below were first written against DUB and its accounts A and B; a number that comes from there
(credits, timings, the culler) says so and is the other hub's, not the lab's, until the same thing has been measured
on SJC. Pods on SJC are started through the hub's web spawn form (next paragraphs). Pod A (account C) runs the teacher:
see "Teacher on the Strix Halo (pod A, account C)" below for what it is and what differs from the R9700 recipe.

**On SJC the REST launch is accepted but ignored; the controller launches through the web form.**
`POST /hub/api/users/<account>/server` with options answers 202 and echoes the options back, but the pod that comes up
is the stock `quay.io/jupyterhub/singleuser:latest`: no GPU device and no ROCm. Only the hub's own spawn page starts the
real pod (`ghcr.io/amdresearch/auplc-base:latest`, ROCm 7.13, and with `strix-halo` the Strix Halo GPU, gfx1151).
`Hub.spawn_form` in `lab/hub.py` does what the browser does, with cookies and no token:
`GET /hub/native/login` (the `_xsrf` cookie and hidden field), `POST /hub/native/login` with `username`, `password` and
`_xsrf` (a 302 and the `jupyterhub-hub-login` cookie mean it was accepted), `GET /hub/spawn/<user>` (200), then
`POST /hub/spawn/<user>?_xsrf=<the _xsrf cookie>` with the fields `resource_type`, `gpu_selection_<resource_type>`,
`runtime` (minutes), `repo_url`, `repo_persist` and `repo_branch` (a 302 means the spawn started). The controller then
polls the API for `servers[""].ready` as for any launch. The page's own runtime box stops at 240 minutes; the hub took
2400 from the form and records it as `runtime_minutes`, and a runtime beyond 240 still wants AMD staff's go first.

Which way a launch goes is set per account, in the environment or the workspace `.env`:

| Key | Meaning |
|---|---|
| `AUP_HUB_LAUNCH_<letter>` | `form`: always the web form. `rest`: only the API call, never a look at the pod. `auto` (also when the key is not set): the API call, then, if the account has a password, a look at the ready pod; a pod that turns out to be the stock image is stopped and launched again through the form. Set `form` for C and D. |
| `AUP_HUB_PASS_<letter>` | The web login password. Only the form reads it, and it is cut out of every answer, error message and ledger row as the token is. A `form` launch without it is refused before anything is sent (kind `no_credentials`, exit 2). |
| `AUP_HUB_GPU_<letter>` | The accelerator the account launches by default; else `strix-halo`, the SJC pods. `r9700` (the other chip, on DUB) is still a card both hubs list: set it for A and B. `--accelerator` overrides it. |
| `AUP_HUB_RESOURCE_<letter>` | The catalog resource the account launches by default; else `gpu`. `--resource` overrides it. |

The look at the pod (`auto`) runs a three-line script in a terminal on it, as `run-script` does, which prints
`$JUPYTER_IMAGE_SPEC` and whether `/dev/kfd` exists. The image name decides (`jupyterhub/singleuser` is the stock one);
without a name, a pod with no GPU device is the stock image when the resource is not a cpu one. A pod that cannot be
asked is left as it is, and the answer says `image_check: unknown` and why. `--no-wait` skips the look, as does a pod
that was already running. The answer of every launch that sent something names the path (`launch_path`: `rest` or
`form`) and, after a look, `image_check` (`ok`, `stock` or `unknown`); a launch that fell back adds `fell_back_from` and
`rest_pod_image`. The `pod launch` ledger row carries `launch_path` (and `fell_back_from`) next to `resource` and
`accelerator` in `conditions`, so a metric such as `pod.start_s` is never mixed between the two paths. On DUB the REST
call works and the look at the pod is only a few extra calls: set `AUP_HUB_LAUNCH_A=rest` and `AUP_HUB_LAUNCH_B=rest` to
launch A and B exactly as before (an account without a password already does).

**Nothing is metered on SJC.** `GET /hub/api/quota/me` answers `balance: 0, unlimited: true, enabled: false` and
`GET /hub/api/quota/rates` answers `enabled: false`: the hub keeps no credit accounts. The controller reads that as
"no limit", not as an empty balance: `quota` and `launch` say `quota disabled on this hub`, a launch is never
refused for credits, and `status` shows no credit figures. On DUB (quota enabled) a balance that cannot cover the
runtime still refuses the launch.

**Accounts and addresses.** An account is any single letter whose `AUP_HUB_USER_<letter>` and
`AUP_HUB_TOKEN_<letter>` keys exist in the environment or the workspace `.env` (the same search as `python -m lab ready`).
The hub it lives on is its `AUP_HUB_URL_<letter>` when that key is set, else `LAB_HUB_URL`, else DUB, so C and D
(`AUP_HUB_URL_C`, `AUP_HUB_URL_D` = `https://sjc.aupcloud.io`) work without renaming a key. `--account both` still means
A and B, which is also the default of `status`, `quota` and `keep-alive`: ask for the SJC pair by letter. The ledger
names the machines by role, not by account (telemetry ruling 3): `LAB_POD_ROLES=C:pod-a,D:pod-b` says that SJC
accounts C and D fill the teacher's pod-a and the trainer's pod-b (environment, then the workspace `.env`; unset, it is
`A:pod-a,B:pod-b`, the DUB pair). Never `pod-c` or `pod-d`. An account with no role in the setting is refused
(`no_pod_role`, exit 2) instead of being named `pod-<letter>`. Every `pod ...` row's conditions carry `hub` (the host)
and `account` (the letter), so a pod-a row from DUB and one from SJC tell apart. `python -m lab credits` names its rows
the same way.

```
python -m lab.controller status --account C                 # read-only: is a pod up, its scheduled end, which hub
python -m lab.controller quota --account C                  # read-only: "quota disabled on this hub"
python -m lab.controller launch --account C --minutes 240 --dry-run
python -m lab.controller launch --account C --minutes 240 [--accelerator strix-halo] [--resource gpu] [--no-wait]
python -m lab.controller wait --account C
python -m lab.controller keep-alive --account C --minutes 90
python -m lab.controller run-script --account C lab/pod_a_teacher.sh proxy
python -m lab.controller stop --account C
```

The same with `--account D`. `status` answers with `hub` (the address) when the accounts it was asked about share a
hub, and with `hubs` (one address per account) when they do not. Launching and stopping stay the owner's call.

**Reaching a pod from outside.** Once `jupyter-server-proxy` runs on the pod (the `proxy` step of the pod scripts, then
one relaunch; not yet tried on SJC), anything listening on a pod port is at

```
https://<hub>/user/<account>/proxy/<port>/
Authorization: token <hub token>
```

where `<hub>` is `sjc.aupcloud.io` for C and D or `dub.aupcloud.io` for A and B, `<account>` is the user name the hub
knows the account by (not the letter), and `<port>` is the port the server listens on.

## Credits: `python -m lab credits`

```
python -m lab credits                       # accounts C,D (the SJC hub)
python -m lab credits --account A,B --window-hours 6 --until 2026-10-11T16:00:00Z --no-ledger
```

Read-only (GET requests only; it also runs while `lab/state/STOP` exists). It answers with one JSON object: per
account the figures, `alerts`, and `lines`, one sentence per account. Exit 0 when every account was read (an alert is
a result, outcome `alert`, still exit 0), 1 when one could not be (outcome `error`), 2 for bad arguments or missing
keys. `--account all` takes every letter that has both keys; an account's hub is `AUP_HUB_URL_<letter>`, else
`$LAB_HUB_URL`, else DUB.

- **Metered hub** (DUB, the other hub: quota enabled, 4 credits a minute on the R9700, the other chip). The hub deducts a session only when it
  stops, so the balance is flat while a pod runs. The tracker adds the running session (whole minutes x rate, from the
  server's `started`) and the settled ones (the usage transactions, spread over the minutes they ran) to give
  `credits_left`, `credits_per_hour` over `--window-hours` (default 3), `burn_per_hour` (the higher of that and what
  the running servers burn now), `hours_left` and `runs_out_at`. It alerts when the credits are gone or would not last
  until `--until` (default judging, Sunday 2026-10-11 12:00 ET); after that moment it never alerts.
- **Unmetered hub** (SJC, the lab: quota disabled, balance 0, unlimited). Nothing is projected and nothing alerts; the line is
  `unmetered: usage N minutes today, M sessions` from `/hub/api/stats/me`, today being the hub's UTC day.
- Times come from the hub's clock (its `Date` header), not the workstation's.
- **Ledger.** One row per account (verb `credits`, node the role the account fills (`pod-a` for C on SJC, `LAB_POD_ROLES`), `refs.account`, `hub` and `account` in conditions, outcome `ok`, `alert`,
  `unmetered` or `error`) unless `--no-ledger`. Metered rows carry `pod.balance_credits`, `pod.credits_left`,
  `pod.credits_per_hour`, `pod.credits_burn_per_hour`, `pod.credits_hours_left`; unmetered rows `pod.usage_min_today`,
  `pod.sessions_today`, `pod.usage_min_window`. A partly read answer is marked `low_confidence`. Tokens and account
  names are never printed or written.

Tests: `tests/test_lab_credits.py` replays the recordings in `fixtures/lab/credits/` and drives `lab/hub_fake.py`.

## Secret scan: `tests/test_no_secrets.py`

`python -m pytest -q tests/test_no_secrets.py` fails on any credential-shaped text in three places: the repo tree
(what git tracks, plus untracked files it does not ignore), everything under `lab/state/` (the ledger, notes and review
files, ignored by git or not, and `$LAB_STATE_DIR` when it points elsewhere), and the Notion post texts kept in
`fixtures/secrets/notion_posts.txt`. It runs with the rest of `pytest -q` in about a second and prints only
`path:line: label`. A value is never printed.

| Label | What it matches | Left alone on purpose |
|---|---|---|
| hub token | the word `token` or `bearer` (any case, an account letter such as `_C` allowed after it), then `:`, `=` or a space, then 32 hex characters: `Authorization: token ...`, `?token=...`, `"token": "..."`, `AUP_HUB_TOKEN_C=...` | a bare 32-hex string (event ids, Notion page ids and digests have that shape), `token_id`, `tokens`, 40 and 64 hex hashes, `$AUP_HUB_TOKEN_A` |
| hugging face token | `hf_` and 30 or more letters and digits | `hf_hub_download` and other names |
| notion token | `ntn_` (current) or `secret_` (legacy) and 40 or more letters and digits | the prefixes in prose, `secret_key_base` |
| tailscale key | `tskey-`, an optional kind (`auth-`, `api-`) and 20 or more key characters | `tskey-auth-<key>` and `tskey-auth-...` |
| github token | `ghp_`, `gho_`, `ghu_`, `ghs_` or `ghr_` and 36 or more characters, or `github_pat_` and 50 or more | the prefixes in prose |
| private key | a `BEGIN ... PRIVATE KEY` header followed by a base64 body, also with JSON `\n` escapes between them | the header alone, a body written as `...` |
| account password | an upper-case `.env` line whose key has `PASS`, `PASSWORD`, `PASSWD` or `PWD` as a whole word, `=`, then 8 or more characters; or a quoted value after a `password`, `passwd` or `pwd` key (JSON, YAML, Python) | `<password>`, `$VAR`, `${VAR}`, `changeme`, masked values, code that reads a password, a value that is itself a key name |

A match is dropped when it cannot be a credential: filler such as `xxxx...`, one unit repeated (`0123456789abcdef`
twice), a run of ten of one character, or fewer than 9 distinct characters in a long value (a random 32-hex token
has fewer 3 times in a million). A fake token in a test therefore has to look like filler, or be built at run time.

**Values from the `.env` files.** When a `.env` exists (the clone's, or one of the two folders above it, found the way
`python -m lab ready` finds them, or `$LAB_ENV_FILES`), the test reads it in memory and looks for the value of every key
listed in `config/.env.example` and of every credential-named key (`TOKEN`, `SECRET`, `PASS`, `PASSWORD`, `PASSWD`, `PWD`,
`API_KEY`, `PRIVATE_KEY`, `ACCESS_KEY`, `AUTH_KEY`) in all three places, as written, JSON-escaped and percent-encoded.
Values under 8 characters and keys ending `_DIR` or `_PATH` are skipped. Without a `.env` (CI has none) there is
nothing to look for. A hit reads `env value of <KEY>`.

**When it fails**, take the value out of the file. If it was real, tell Ilan: the credential has to be replaced, not just
deleted (a hub token is minted again, a Notion or Hugging Face token is revoked).

**The proof** is in `fixtures/secrets/`. `cases.json` holds the true and false positives; secret bodies are written
as `{hex32}`, `{hf}`, `{ntn}`, `{tskey}`, `{github}`, `{pem}`, `{password}` and expand to generated values when the test
runs, so the repo never holds one. `notion_posts.txt` holds post texts modeled on real ones (page links, event ids,
hashes, key names) that have to stay clean. To teach the scan a new shape, add it to `SHAPES` with at least one true case
and one near miss (`"near"`); a test fails when a shape has neither. To stop a false alarm, add the text that raised it to
`cases.json` as a case with `"expect": []`, then change the shape.

Limits: binary files and `.gz` archives are not read, a file is read up to 64 MB, and a hub token that is not 32 hex or
a password inside prose is found only through its `.env` value.

## What the pods are

- An AMD Strix Halo APU each: Radeon 8060S, gfx1151, RDNA 3.5, 64 GB of memory shared between the CPU and the GPU.
  The R9700 (gfx1201, 32 GB of GDDR6) on DUB is the other chip.
- ROCm 7.13.0, `torch 2.9.1+rocm7.13.0`, `triton 3.5.1+rocm7.13.0`, `hipcc` on PATH, `python3` only.
- No root, no Docker, no SSH. JupyterLab terminal and the Jupyter REST API only. `pip install --user` works
  (binaries land in `~/.local/bin`; the scripts add it to PATH).
- `/home/jovyan` is a persistent NFS volume (models, venv packages, builds survive a relaunch).
  `/opt/workspace` (the Jupyter root) is ephemeral.
- Nothing is billed on SJC (quota disabled). On the other hub, DUB, billing is per elapsed minute (4 credits/min on
  the R9700, ~60,000 per account). The server enforces no runtime cap; AMD staff are fine with long sessions, and a
  runtime beyond the form's 240 minutes wants their go first. Stopping is immediate. A 60-minute idle culler
  stops a server that gets no requests, so keep the model server busy or ping it.
- **Tailscale is blocked** on the pods (every tailscale.com host is connection-reset). GitHub, Hugging Face
  and PyPI are reachable.

## Launching a pod

Web form: resource "Basic GPU Environment" (key `gpu`), accelerator `strix-halo`, runtime up to 240 on the page.
On SJC this form is the only way to get the real pod; the REST launch is accepted but ignored there (above), and
`python -m lab.controller launch` with `AUP_HUB_LAUNCH_<letter>=form` does what the form does. On the other hub, DUB,
the API launch works, with the account's hub token and the R9700:

```
POST https://dub.aupcloud.io/hub/api/users/<account>/server
Authorization: token <hub token>
{"runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "r9700"}
```

Poll `GET /hub/api/users/<account>` until `servers[""].ready`. Stop with `DELETE .../server`.
The private team repo cannot be cloned at launch (no GitHub app on the hub); upload `lab/` after launch
(JupyterLab upload, or `PUT /user/<account>/api/contents/<path>`).

## Controlling the pods: `lab/controller.py`

One command line for the calls above, from the repo root. The keys are `AUP_HUB_USER_<letter>` and `AUP_HUB_TOKEN_<letter>`
in the workspace `.env` (found the way `python -m lab ready` finds them); nothing prints a key or an account name.
Every call answers with one JSON line and exit code 0, 1 or 2, like `python -m lab`.

```
python -m lab.controller status                       # both accounts: running, scheduled end, spend so far, balance
python -m lab.controller quota --transactions         # balances, rates, recent sessions
python -m lab.controller launch --account B --minutes 600 [--dry-run] [--no-wait]
python -m lab.controller wait --account B
python -m lab.controller keep-alive --minutes 90      # a ping every 10 minutes so the idle culler never fires
python -m lab.controller run-script --account B lab/pod_b_trainer.sh smoke
python -m lab.controller stop --account B
```

- Launching and stopping a pod is the owner's call, and pod B goes through the GPU queue: a sitting is denied
  `launch` and `stop` (`lab/machine/allowed.py`). Try `launch --dry-run` first: it checks the balance and sends nothing.
- Safe to repeat: launch of a ready pod and stop of a stopped one say so and send nothing; a launch the hub answered
  with a gateway error is never sent twice.
- `--minutes` has no ceiling (the hub has none). The session is charged when it ends, 4 credits a minute, so
  `status` works the spend out from the start time and a launch the balance cannot cover is refused up front.
- `status` reads only the hub API, which does not count as activity. `keep-alive` pings the pod and so does cost
  240 credits an hour: always give it `--minutes`.
- `run-script` takes its own options before the script's name; everything after the name goes to the script. It
  uploads the script to the pod, runs it in a terminal and waits for the exit code it appends to the output.
  `--with FILE` (before the script's name, repeatable) uploads a file the script needs next to it first.
- While `lab/state/STOP` exists `launch`, `stop`, `keep-alive` and `run-script` are refused.
- Ledger rows: `controller <command>` for each call, and `pod launch`, `pod stop`, `pod status`, `pod keep-alive`
  (`alive`, `failed_ping`, `lost`) with `node` pod-a or pod-b (the role the account fills, see above) and `hub` and `account` in their conditions, which are the rows `python -m lab alerts` reads.
- Tests use `lab/hub_fake.py`, a fake hub on a fake clock built from the recorded answers, so nothing launches or sleeps.

Measured live (R13, 2026-10-10, pod D = account D on the SJC hub, Strix Halo gfx1151, `AUP_HUB_LAUNCH_D=form`): stop then
`launch --minutes 2400 --accelerator strix-halo --resource gpu` took 5.55 s to a ready pod (launch_path form, image
ghcr.io/amdresearch/auplc-base, `/dev/kfd` present, torch 2.9.1+rocm7.13.0 sees gfx1151). The first request through the hub
(`GET /user/roscon02/api/me`, 200 in 0.21 s) came 11.7 s after the launch call began; pod B has no serving script yet, so
there was no model server to ask. `keep-alive --interval 600 --minutes 70` made 7 pings with 0 failures and the pod was still
running afterwards, so the idle culler did not fire. Credits per hour is unmetered on SJC (balance 0, unlimited true,
enabled false); on a metered hub it is the 4 credits per whole minute above. The rows are in `lab/refine/metrics.jsonl`
on node `pod-d`.

## Box ↔ lab: the hub proxy

Once `pip install --user jupyter-server-proxy` has run on an account and the pod has been relaunched once,
anything listening on a pod port is reachable from outside:

```
https://<hub>/user/<account>/proxy/<port>/...        # <hub> is sjc.aupcloud.io for C and D
Authorization: token <hub token>
```

That is how the Jetson calls the teacher (`/proxy/8000/v1/chat/completions`) and how it will pull packaged
models. The `/proxy/absolute/<port>/` form does not work here.

### From the box: `lab/proxy.py` and `lab/transfer.py`

`lab/proxy.py` is the box's client for that address. It takes the hub from `lab/hub.py`, sends the token and a browser
User-Agent, never follows a redirect, and streams: jupyter-server-proxy holds any reply whose `Accept` header is not
exactly `text/event-stream` until it is whole, and on DUB Cloudflare cut a held reply after 125 s, so `chat()` always
sends `stream: true` with that header and gathers the pieces. Each failure is one error with a kind, a status, a
`retryable` flag and a hint: a wrong token is 302 or 403 (`bad_token`), a stopped pod 424 (`pod_stopped`), a port with
nothing on it 500 (`model_server_down`), a held reply cut at 125 s 524 (`gateway_timeout`), the proxy giving up 599
(`proxy_timeout`); the whole table is in the module's docstring. `proxied_runtime()` gives the stock
`LlamaServerRuntime` the same path without changing `runtime/`.

```
python -m lab.proxy probe --account C [--port 8000] [--wrong-token]    # models, health, and the server and cf-ray headers
python -m lab.proxy chat --account C --prompt "Count to 40." [--max-tokens 64] [--no-thinking]   # streams, and times it
```

`lab/transfer.py` moves files. A dataset goes up through the contents API in chunks (folders made first, a lost answer
settled by reading the size back, only an export that `lab/dataset.py` accepts). A package comes down in byte ranges
through the Jupyter server's `/files/` route, or through a file server on a pod port, with a sha256 check, and lands
under `data/models/<version>` ready for `python jetson/atlas_swap.py swap <version>`.

```
python -m lab.transfer push-dataset --account D data/exports/<dataset_id> [--to datasets] [--chunk-mib 8]
python -m lab.transfer push-file --account D LOCAL REMOTE [--chunk-mib 8]
python -m lab.transfer pull-file --account D REMOTE LOCAL [--via files|proxy --port N] [--part-mib 64] [--sha256 HEX]
python -m lab.transfer pull-package --account D REMOTE_DIR [--models-dir data/models] [--via files|proxy --port N]
```

Both print one JSON line and exit 0 (worked), 1 (did not) or 2 (bad arguments, missing keys); a transfer reports its
bytes, seconds and megabytes a second as the `link.push_mb_s` or `link.pull_mb_s` metric. A push lands in the pod's
Jupyter root, which on DUB was the ephemeral `/opt/workspace`: what must outlive a relaunch has to be moved under `~/`
on the pod (`push_dataset` reports `persistent` when the root lists a `home` folder). Nothing here has run on a real
pod or on SJC yet; `lab/proxy_fake.py` is the stand-in its tests use (Cloudflare, the hub, the Jupyter server and a
model server on a fake clock).

### Failure-mode map of the proxy path on SJC (R18)

Exercised on 2026-10-10 from a laptop (not a pod) through `lab/proxy.py`, with no retry, against pod B's role (account
D on `sjc.aupcloud.io`), port 8000, where the R30 teacher (vLLM) was started and stopped inside a GPU Queue slot. SJC
answers through Cloudflare like DUB (`server: cloudflare`, a `cf-ray` header). Pod C was not called.

| Case | Status | `lab/proxy.py` kind | error_name | Latency | What comes back |
|---|---|---|---|---|---|
| Bad token, GET | 302 | `bad_token` | none | 0.39 s | a redirect to the hub's login page (never followed) |
| Bad token, POST | 403 | `bad_token` | none | 0.13 s | Jupyter Server's 403 page |
| Nothing listening on the port (8000, 8123), GET and POST | 500 | `model_server_down` | none | 0.13 to 0.23 s | Jupyter Server's HTML error page |
| Model server mid-restart (GET `/v1/models` every 5 s) | 500, 56 times, then 200 | `model_server_down` | none | 0.2 s a poll; 294 s from start to the first 200 | the same 500 page for the whole start: vLLM opens the port only when the engine is ready, so there is no 502, no 503 and no hung call |
| Model server stopped, pod up | 500 | `model_server_down` | none | 0.13 to 0.47 s | as nothing listening |
| Pod stopped | 424 predicted | `pod_stopped` | none | not exercised | not exercised live: stopping pod D ends the keep-alive that runs from another machine, which is the owner's call (JupyterHub documents 424 with a JSON body for a stopped server) |
| 31 s stream, `Accept: text/event-stream` | 200 | ok | - | headers and first bytes at 0.92 s, last at 31.9 s | streamed: 91 reads spread over 30.9 s |
| The same stream, `Accept: application/json` | 200 | ok | - | nothing until 31.5 s | held whole: headers and all 19,994 bytes arrive together when the model has finished (2 reads in 0.008 s) |
| 5 MB request body (5,242,964 bytes), POST | 400 | `model_server_error` | none | 1.0 s | the body reaches vLLM, which refuses it for its 16,384-token context; no 413 on the way |
| 20 concurrent short chats (8 tokens each, streamed) | 200, all 20 | ok | - | 3.0 s for all; 1.05 s fastest, 2.95 s median, 2.98 s slowest; first token 2.2 s | no refusal and no rate limit at 20 |
| Front door down (from the record, DUB, M21 of `lab/refine/research/plumbing/measurements-2026-10-10.md`) | 502 | `hub_down` | `origin_bad_gateway`, error_category `origin` | - | with `Accept: application/json` Cloudflare's JSON (`retryable` true, `retry_after` 60); a browser-like request gets its HTML page |

What the box should take from it: only the outage answer carries an `error_name`; every other failure is told apart
by its status and by whether the body is the hub's, Jupyter's or the model server's. A model server that is starting
looks exactly like one that was never started (500 for about five minutes), so the caller waits and polls instead of
giving up. A reply that is not asked for with the exact event-stream header shows nothing until it is complete, which
on a long answer runs into the 125 s cut measured on DUB. The rows are in `lab/refine/metrics.jsonl` on node `pod-b`
with hub sjc and account D (`proxy.status`, `proxy.restart_to_ready_s`, `proxy.stream_first_byte_s`,
`proxy.concurrent_wall_s`), one sample each unless the row says otherwise.

## Jetson power sampler, live run (T7)

The Jetson sampler ran on the box on 2026-10-10 as one read-only 10-minute read over SSH (key login, no sudo,
`atlas-gateway` and `atlas-llm` left running and never restarted, nothing written to `data/events`), in the mode the box
was in: `nvpmodel -q` said MAXN_SUPER, mode 2, before and after, and nothing changed it. It is
`python -m contexts.telemetry.jetson_power probe --count 600 --interval-ms 1000` on a scratch copy of main under `/tmp`
(removed afterwards): nothing of ours running for the first 300 s, then back-to-back caption requests for 266 s, the
request `runtime/llama.py` sends for a caption (one image, "Describe the scene in one sentence.", 80 tokens) straight to
llama-server on :8080, one in flight at a time. The image is a 1280x720 repo fixture (939 prompt tokens, the count a
camera frame gives), not a camera frame, and the request goes straight to llama-server, not through the gateway, so
nothing reaches the event log.

The sampler returned 600 samples in 602.5 s at a measured period of 1005.8 ms (longest gap 1015.2 ms, no line unparsed
or bunched), the INA3221 averaging 512 conversions into each reading. Its own reads are cheap: the three overcurrent
counters take 0.63 ms a sample, the INA3221 settings 0.25 ms (every 10 s) and the `nvpmodel -q` query 6.6 ms (every
30 s). VDD_IN, all low confidence (one run per state, no external meter):

- Idle, 297 samples: mean 12.62 W, median 15.07 W, from 4.15 to 19.17 W. It is not one level. The gateway keeps
  captioning keyframes on its own, so the rail alternates in stretches of about 3 s between roughly 5 W (30 percent of
  the samples are under 7 W) and roughly 18 W (half are over 15 W), and the mean moves with how busy that captioning is:
  the 100 ms read in the T13 section below, an hour earlier, gave 15.76 W with three quarters of its samples over 15 W.
- Under caption load, 265 samples and 28 requests (none failed, 9.5 s each on average because the server takes one
  request at a time and the gateway's own captioning shares it, median decode 13.85 tok/s): mean 17.54 W, from 15.72
  to 19.13 W. The peak is a one-second reading that averages 512 conversions, so a shorter spike is smoothed away.
- Per rail, mean mW idle then loaded: VDD_IN 12620 then 17536, VDD_CPU_GPU_CV 5848 then 8947, VDD_SOC 3337 then 4408.
- The overcurrent counter oc3 rose by 471 over the idle stretch and by 3684 over the load stretch (about 1.6 and 14 a
  second); oc1 and oc2 stayed at 0. A joules-per-answer value is marked low confidence when a Jetson overcurrent counter
  rose across its window, and on this run that is 24 of the 28 caption windows and 18 of 29 ten-second idle windows, so at
  MAXN_SUPER on this box the rule fires on ordinary work. Replayed through the joules projection, the run gives one
  value per caption, each with `idle_w` 18.23 W: the last reading before the first request, which is the captioning
  level, not the 5 W floor, because the gateway's own captioning is not a `telemetry.request`.

The rows are `jetson.idle_watts`, `student.watts_avg` and `student.watts_peak` on node `jetson` in
`lab/refine/metrics.jsonl`, each with the mode, sampler, period, temperatures, rails and counter rises in its
conditions. The raw one-second events, the request timings, the environment before and after and the read costs are in
`fixtures/telemetry/jetson_power_live/`, and `tests/test_lab_jetson_live_run.py` recomputes each row from them. The
other power modes were not run: switching `nvpmodel` needs sudo, and the box stays on MAXN SUPER.

## GPU power sampler: `lab/pod_power.py`

Reads the pod's Strix Halo (Radeon 8060S, gfx1151; the R9700 is the other chip) once a period (default 1000 ms)
with `amd-smi metric --power --clock --temperature --usage --energy`, found under `/opt/rocm/bin` since the SJC image
has it off PATH (rocm-smi as fallback; reads only, no root), keeps every read as one `telemetry.gpu.power` sample in
`~/atlas-power/samples-<UTC day>.jsonl`, and serves them on `127.0.0.1:8790` (`/samples?after=<seq>`, `/status`),
which the proxy exposes at `/user/<account>/proxy/8790/`. On this chip amd-smi reads SOCKET_POWER (the whole APU
socket, in watts) and the edge temperature, but `ENERGY` and `CLOCK` are N/A, so `rocm-smi --showtemp --showclocks`
fills the clocks (sclk, mclk) on every read and the energy always comes from integrated power. Each sample names the
tool, the period, the power field and the temperatures, and counts polls apart from distinct updates; under load
(GFX activity, else 60 W or more), readings frozen for 10 s are flagged stale; the energy counter is checked against
integrated power at start-up and once under load and is used for energy only while it agrees within 10 percent.
`serve` refuses any chip but gfx1151. Stdlib only, so the pod needs nothing installed. Starting it on a pod is
Ilan's go:

```
python -m lab.controller run-script --account C --with lab/pod_power.py lab/pod_power.sh check            # one read, raw outputs saved
python -m lab.controller run-script --account C --with lab/pod_power.py lab/pod_power.sh start gpu-a       # status after 15 s
python -m lab.controller run-script --account C lab/pod_power.sh stop
```

The Jetson reads both pods with `ATLAS_TELEMETRY_SOURCE=pods` (keys `AUP_HUB_USER_<X>`, `AUP_HUB_TOKEN_<X>` of each
pod's account; `ATLAS_POD_ACCOUNTS` maps pods to accounts, default `gpu-a=A,gpu-b=B`, so the SJC pods need
`gpu-a=C,gpu-b=D`; `AUP_HUB_URL_<X>` or `LAB_HUB_URL`, the settings `lab/hub.py` reads, name the hub;
`ATLAS_POD_POWER_PORT`, `ATLAS_POD_POWER_POLL_S` optional): samples arrive with `received_ts`, nothing is lost over
a proxy gap the pod's buffer covers (about 5.5 hours), and a sampler restart shows as a new `sampler_started`.
The reader sends `lab/hub.py`'s browser-like User-Agent (`LAB_HUB_USER_AGENT` overrides it), since the hub's front end
answers 403 to a script's own agent. A hub host other than dub.aupcloud.io must also be admitted by the Jetson
firewall: `jetson/egress.py` adds the aupcloud.io host of every `AUP_HUB_URL_<X>` and of `LAB_HUB_URL` to its hub
group. Samples the reader rejects (another node's name, a broken event) are logged once per page with the count.
A window becomes a stored row with `python -m lab gpu-power record --samples <files> --node pod-a --metric
pod.idle_watts [--since ISO] [--until ISO] [--state idle] [--metrics-file]`: the row carries the sampler, period,
measured period, longest gap (and, from the Jetson log, the longest gap between receive times and the latest
arrival), restarts, distinct updates, starting temperatures, the energy and its source, and `error: unknown` until a
reference run measures it. Parsers and sampler are tested offline on `fixtures/telemetry/pod_power`
(tests/test_lab_pod_power.py; the `strixhalo_*` files are output recorded on the Strix Halo (account C), the others
on the R9700, the other chip).

## Jetson sampler cost and idle watts (T13)

Measured on the box on 2026-10-10 over SSH (key login, no sudo, `atlas-gateway` and `atlas-llm` left running, nothing
restarted, nothing written to `data/events`), in the mode it was in: `nvpmodel -q` said MAXN_SUPER, mode 2, never
changed. The commands are the two read-only steps T2 merged, run from a scratch copy of main under `~/t13`:
`python -m contexts.telemetry.jetson_power cpu --minutes 5 --interval-ms 100|200` (5 minutes sampler off, then 5 on),
and `probe --count 3000 --interval-ms 100` (about 5 minutes of VDD_IN). The sampler's own CPU (this process, its
`nvpmodel` children and tegrastats; the row value) is 3.51 percent of one core at the 100 ms default with the gateway
idle and 3.28 with one caption request always in flight, and 1.83 and 1.70 at 200 ms: over the 1 percent budget at
both periods, and about linear in the rate. The box-wide on-minus-off difference (the report's own `value`) came out
-5.5, -3.5, +10.0 and -10.3 points, so the gateway's own background captioning (the box sits near 110 percent of a
core with nothing asked of it) swamps the sampler and that figure cannot resolve it; it is kept in each row's
conditions. The caption request was a back-to-back loop straight to llama-server (:8080, 80 tokens, about 9 to 13 s
each under load), not `/api/ask`, which would have logged a `QuestionAsked` event on the box. Idle watts on the
VDD_IN rail with the gateway and model loaded and no request of ours: mean 15.76 W over 3000 samples (median 18.2,
min 4.7, max 19.9), bimodal: about 5 W between the gateway's keyframe captions and about 18 W while the GPU captions,
so "idle" depends on how often it captions. The starting temperature was tj 54.2 C. Rows are `telemetry.sampler_cpu_pct`
and `jetson.idle_watts` on node `jetson` in `lab/refine/metrics.jsonl`, all `low_confidence` (telemetry ruling 7:
one run per cell and no external meter, error unmeasured); the raw output (top and ps snapshots, probe lines, request
timings, temperatures) is `fixtures/telemetry/jetson_sampler_cost/`. Consequence: a sampler that fits 1 percent needs
about a 350 ms period, or tegrastats read in place of a Python loop per line.

## Teacher (pod A)

```
bash lab/pod_a_teacher.sh proxy      # once per account, then relaunch the pod
bash lab/pod_a_teacher.sh build      # vLLM from source at commit b1388b1 + lab/vllm-gfx1201.diff, for the pod's gfx target, ~1 h
bash lab/pod_a_teacher.sh download   # RedHatAI/Qwen3.8-27B-INT4 (19 GB) into ~/models
bash lab/pod_a_teacher.sh serve      # OpenAI-compatible server on 127.0.0.1:8000, model names "teacher" and "group-brain", log ~/serve/vllm.log
```

One model under two names: "teacher" for labeling and captions, "group-brain" for the Circle's planning questions
(G4, `contexts/circle/infrastructure/teacher.py`). The server runs `--scheduling-policy priority`: vLLM serves the
lowest request priority first and, when the KV cache runs out, preempts the highest. A planning question asks at -10
and `lab/label.py` sends none (0), so a question never waits behind a labeling run, and continuous batching keeps the
labels decoding beside it. At the pinned commit (b1388b1) a server on the default fcfs policy accepts the field and
ignores it, so the group brain works before the restart too, asking as "teacher" while that is the only name. Requests
are not logged (`--enable-log-requests` stays off). Which name answers, with no member text sent:
`python -m contexts.circle.brain status`. How much planning slows labeling when both run is not measured yet (T0).

`lab/vllm-gfx1201.diff` is the set of changes vLLM (commit b1388b1, April 2026) needs on these pods: `num_stages=1`
in the Triton unified attention kernel (whose default overflowed the R9700's 64 KB LDS on RDNA 4), two platform-detection fixes, the torch pins removed
so the pod's own torch is used (`use_existing_torch.py`), and the GPTQ `compat.cuh` half/half2 `atomicAdd`
fallbacks guarded out because ROCm 7.13 ships those overloads itself. Build gotchas the script handles: no root,
so `cmake`, `ninja` and `pkgconf` come from `pip --user`; the image has no libdrm, so a stub `libdrm.pc` and an
empty `libdrm.so` under `~/.local` satisfy torch's `LoadHIP.cmake`; `SETUPTOOLS_SCM_PRETEND_VERSION` avoids a
`git describe` timeout on the NFS home.

The patch keeps the name `vllm-gfx1201.diff` although the lab now runs on gfx1151. Its hunks are generic ROCm fixes
for this vLLM commit, not code for one chip, and the same file is applied unchanged for both targets (all 9 hunks
apply on gfx1151 and none is new, below). The name stays because the scripts (`pod_a_teacher.sh` looks for it next to
itself), `--with` and `--patch` command lines, the kernels rubric, the rulings and the ledger rows all cite that path;
renaming it would break the build steps and orphan that history.

On the R9700 (the other chip, DUB pod A, measured on the morning of 2026-10-10) with `RedHatAI/Qwen3.8-27B-INT4`
(bf16 activations, 16k context, one image per prompt):

| Prompt | TTFT | decode |
|---|---|---|
| text, thinking off, 76 tokens | 0.55 s | 7.8 tok/s |
| text, thinking on, 160 tokens | 0.28 s | 7.7 tok/s |
| 1280x720 JPEG caption, 28 tokens | 1.09 s | 6.6 tok/s |

On the R9700, engine init took about 270 s (CUDA-graph capture included), the model held 84 % of VRAM with ~10x
concurrency at 16k tokens, and the card drew ~200 W average, 300 W peak, while generating (about 26 J per token).
These are the other chip's numbers, not the lab's baseline; the Strix Halo's are in the next section (2.9 tok/s at
about 120 W, 9 W idle, 0.39 s to the first text token). On both chips the W4A16 layers run on `ConchLinearKernel`
(the Triton fallback; Marlin/Machete do not exist on ROCm), which is why decode is far below the 20 to 30 tok/s the
design doc estimated. That kernel is the first target of the kernel work, which now targets gfx1151 (RDNA 3.5) on
pod B.
Pass `"chat_template_kwargs": {"enable_thinking": false}` for captions, otherwise the model spends its budget
reasoning.

Checkpoints: `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` and `-MXFP4` are AMD's own Quark exports (Aug 2026) and
would be the better AMD-track story, but this vLLM commit's Quark weight mapper (`quark.py:apply_vllm_mapper`)
rejects their config. Making them load means either moving to a vLLM commit from late August 2026 or later
(which pins torch 2.10+, so torch would come from `repo.radeon.com/rocm/manylinux` rather than the image) or
backporting that one Python file's mapper, which needs no C++ rebuild. `Qwen/Qwen3.8-27B-FP8` (28.8 GB) is the
last resort and leaves little KV room.

Call it from anywhere:

```
curl https://sjc.aupcloud.io/user/<account>/proxy/8000/v1/chat/completions \
  -H "Authorization: token $AUP_HUB_TOKEN_C" -H "Content-Type: application/json" \
  -d '{"model":"teacher","messages":[{"role":"user","content":"Describe this table in one sentence."}]}'
```

Images go inline as `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` content parts.

### Teacher on the Strix Halo (pod A, account C)

Account C's pod on SJC (`roscon01`), pod A, runs the teacher, at
`https://sjc.aupcloud.io/user/roscon01/proxy/8000/v1/chat/completions` with `Authorization: token $AUP_HUB_TOKEN_C`.
The GPU is a Radeon 8060S (Strix Halo, gfx1151, RDNA 3.5) with 64 GB of memory shared with the CPU. The same
`lab/pod_a_teacher.sh` builds and serves it; what differs from the R9700 recipe above:

- Tool calling: `serve` passes `--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"` (default `qwen3_coder`, the parser for
  the `<tool_call><function=...>` blocks this checkpoint's chat template writes; `hermes` expects JSON there). Without them a tool
  call comes back as text in `content` and `lab/label.py` rejects it.
- Build target and size: `PYTORCH_ROCM_ARCH` comes from `rocminfo` (gfx1151), and compile jobs are capped by RAM
  (20 jobs on 32 threads and 62 GB instead of 48). The build took 4 min 18 s on the pod's local disk (about an
  hour on DUB's NFS home). `cmake` is pinned below 4 as vLLM asks, and a pip constraint holds the image's torch
  2.9.1, torchvision, torchaudio and triton so no dependency swaps in a CUDA wheel.
- Patch: all 9 hunks (7 files) of `lab/vllm-gfx1201.diff` apply cleanly at b1388b1 and no new hunk is needed; that commit
  already lists gfx1150/gfx1151 and guards the RDNA 3.5 paths in `csrc/rocm`.
- No `amdsmi` Python package: installed from `/opt/rocm` it loads a second copy of `libamd_smi` next to torch's own
  and hides the GPU (`torch.cuda.device_count()` 0). `deps` removes a user-site copy; the patch's
  `torch.version.hip` fallback detects the platform.
- Vision encoder attention, three serve attempts: (1) Torch SDPA ran its math kernel, because torch 2.9.1 keeps
  the AOTriton SDPA kernels off as experimental on gfx1151, and the encoder's profiling pass asked for a 256 GiB
  attention matrix (HIP out of memory); (2) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` the server started,
  but those kernels go wrong at a 720p image's ~3.5k tokens (max error 0.08 to 0.12, NaN at 4096) and captions
  came back as "!!!!"; (3) `--mm-encoder-attn-backend TRITON_ATTN` (vLLM's own Triton attention) is exact and is
  what `serve` passes on gfx115x.
- Power is read with `amd-smi metric --power` (`SOCKET_POWER`, the whole APU package); energy and clocks read N/A.

Measured on 2026-10-10 on pod A, account C (fix round 3, batch 1, 16k context, temperature 0, thinking off; TTFT and decode
with a streaming client on the pod, end to end through the proxy, which buffers the response):

| Prompt | TTFT | decode | through the proxy | watts while decoding |
|---|---|---|---|---|
| text, 31 in, 77 out | 0.39 s | 2.88 tok/s | 27.5 s | 120 W average, 129 W peak |
| 1280x720 JPEG caption, 902 in, 27 out | 3.51 s | 2.82 tok/s | 12.9 s | 118 W average |

Idle: 8 W with no server, 9.2 W with the teacher loaded. The engine is ready about 5 min after launch with a warm
torch.compile cache. These are the lab's teacher numbers. Decode is about 2.7x slower than the other chip's (the
R9700's) 7.7 tok/s, about the ratio of the two memory
bandwidths (decode is weight-bound on `ConchLinearKernel`). The rows are in `lab/refine/metrics.jsonl` on node
`pod-c` (written before the rows were named by role; they are read as pod-a-role rows of account C on the SJC hub and
stay as written); the R9700 rows stay on `pod-a` as the old, other chip.

Pod power sampler, live on pod A, account C (T6, 2026-10-10): `amd-smi metric --power --clock --temperature --usage --energy` answers on
this chip (exit 0, 0.21 s a read): SOCKET_POWER and GFX_ACTIVITY, with CLOCK, ENERGY, HOTSPOT and the other sensors N/A and
EDGE the only temperature. Idle with the teacher loaded 10.2 W (60 s); over 50 teacher requests one every 12 s (44 text, 6
720p images, 6460 prompt and 894 completion tokens) 106.8 W average inside the request windows, 129 W peak, about 9 samples
a window (2 for the answers that stop after 3 tokens). Rows on node `pod-a` (the teacher's role) with the sampler's
conditions; the per-second samples are `fixtures/telemetry/pod_power/strixhalo_2026-10-10_teacher_load_*.jsonl`. The
sampler stays up under nohup (`bash ~/lab/pod_power.sh stop`).
The 39 W `pod.idle_watts` row is the R9700, the other chip.

### The same build on pod D, from scripts alone (R30)

Account D's pod on SJC (pod B's role, the trainer; the kernels experiments run there) holds the same stock build and
checkpoint since 2026-10-10: vLLM 0.19.2+gfx1151 at b1388b1 plus `lab/vllm-gfx1201.diff` in `~/vllm`, and
`~/models/Qwen3.8-27B-INT4` (19 GB, 12 files). No server runs there: start one only inside a GPU Queue slot and stop it
after. Nothing was typed on the pod; every step went through the controller from the repo root:

```
python -m lab.controller run-script --account D lab/pod_a_teacher.sh facts       # hour-0 facts, read-only
python -m lab.controller run-script --account D --timeout 3000 --with lab/vllm-gfx1201.diff lab/pod_a_teacher.sh build
python -m lab.controller run-script --account D lab/pod_a_teacher.sh verify      # rejected hunks, extensions, platform
python -m lab.controller run-script --account D --timeout 3000 lab/pod_a_teacher.sh download
python -m lab slot request --task <ID> --what "..." --minutes 15                 # then: slot start --row <row>
python -m lab.controller run-script --account D --timeout 900 lab/pod_a_teacher.sh serve
python -m lab.controller run-script --account D lab/pod_a_teacher.sh models      # GET /v1/models on the pod
python -m lab.proxy chat --account D --model teacher --no-thinking --prompt "..." # through the hub proxy
python -m lab.controller run-script --account D lab/pod_a_teacher.sh stop        # then: slot done --row <row>
```

`--with` puts the patch next to the uploaded script, which is where the script looks for it. On SJC the Jupyter root
is `/home/jovyan` itself, so run-script's files and everything the script writes sit on the persistent volume. What
R29 did by hand on pod C and the script now does: the hour-0 facts (`facts`, R29 used a separate `hour0.sh`), the
checks after the build (`verify`), the wall-clocks (`build`, `download` and `serve` end with a
`### <step> wall_s=<seconds> rc=<code>` line), `/v1/models` on the pod (`models`) and stopping the server (`stop`).
`serve` now ends at once with the log's tail when the engine dies, instead of waiting out its 15 minutes.

What pod D showed: the recipe needed no change. All 9 hunks apply, none rejected; `_C`, `_rocm_C` and `_moe_C`
import; vLLM resolves `RocmPlatform` with one GPU. Build 292 s (pip dependencies included, 20 jobs), download 281 s,
the engine ready 523 s after a cold start (no torch.compile cache yet; pod C took 297 s warm). One chat completion
with thinking off came back through `https://sjc.aupcloud.io/user/<account D>/proxy/8000/v1/chat/completions` (21
tokens in, 47 out, 24.7 s). Idle with no server: 9 W (`amd-smi` SOCKET_POWER). The build and download wall-clocks are
ledger rows on node `pod-b` with hub sjc and account D; no metric row was written, because this is a build and not a
measurement (kernels pass 2, ruling b). The build's dependencies put numpy 2.2.6 in the user site, over the image's
2.4.6.

### Idle baseline on the Strix Halo

T10, 2026-10-10, pod C, the T3 sampler beside the running teacher. The clock states cannot be set: `rocm-smi --setperflevel high`
fails (no root; sudo answers "no new privileges") and the level stays `auto`, so the states were observed. rocm-smi's
"clock level" label does not tell them apart (it prints level 1 at 620 MHz and at 2900 MHz); the state is the sclk band.
Over 4866 samples (20:35 to 21:56 UTC): deep idle (sclk 600 to 650 MHz, no mclk line) 62 percent of the time at 10.27 W
(stdev 0.7, longest unbroken stretch 42.6 min, 2558 samples), busy (sclk 2600 MHz or more, mclk 1000) 31 percent at 110 W,
and the ramp between them 7 percent. After a request ends the power goes 110, 57, 28, 17, 13, 11, 10 W one second apart: it
settles in a median of 4 s (3 to 5 s, 15 requests), costing about 46 J above idle; the memory clock lags the shader clock by a
second or two (7 samples, 11.1 W). A second reader forking `rocm-smi --showclocks` and `amd-smi metric --power` once a second
beside the sampler lifted deep idle to 13.8 W, so a clock reader must never run beside a power sample that is stored as idle.
The five single requests at 60 s spacing ran while another client had the teacher at 96 to 110 W (53 requests running by the
end), so they gave no clean settle; the settle figure comes from the T6 run. The rows are `pod.idle_watts` per state,
`pod.settle_s` and `pod.idle_state_share` on node `pod-a`, all `low_confidence` with `error: unknown`; the readings are
`fixtures/telemetry/pod_power/strixhalo_2026-10-10_idle_baseline_*.jsonl` and `idle_baseline.py` recomputes every number.

Policy recommendation (telemetry ruling 1 stays: joules per answer are gross, with the idle stored beside). The idle beside
the value is the deep-idle figure, 10.3 W, read by the sampler alone with the model loaded and no request. Why: it is the state
the chip is in whenever no request is running (every sample outside the 4 s tail), so it is the watts a box not answering
costs; the 8 W of a pod with no server is a different state (no model loaded), the R9700's 39 W is another chip, and a
time-weighted average of the three states (about 45 W here) would describe the workload, not the machine. The 4 s settle tail
is not part of the answer's window; its 46 J (2 to 5 percent of a 9 to 27 s answer at 110 W) is stored as `excess_j_median` on
`pod.settle_s`, not folded into the idle.

## Attribution check: `python -m lab attribution-check` (T8)

Do the answers' windows hold the energy the teacher's pod spends on them? From the box, against the running teacher
on pod A and the pod sampler already running there (nothing is started, stopped or changed on the pod):

```
python -m lab attribution-check run --account C [--requests 100] [--idle-s 60] [--gap-s 10] [--wait-min 30]
python -m lab attribution-check replay --tag <date> [--record]      # the numbers again from the saved rows
```

It checks that vLLM's `/metrics` shows nothing in flight (and waits, two minutes at a time, up to `--wait-min`),
reads the idle watts for 60 s, then sends bench's text prompts one at a time (greedy, thinking off, 128 tokens,
ignore_eos), each followed by a 10 s gap with nothing in flight. Every number is gross (telemetry ruling 1): each
answer's joules come from T4's attribution (`contexts/telemetry/application/joules.py`) fed the pod's samples, the
windows and the pod's clock offset as the box feeds it; the gaps and the whole run are integrated the same way, and
answers plus gaps must equal the whole run (conservation). The gap energy above the idle watts times the gap time is
what the windows missed (the sensor's averaging lag and the tail after each answer); its share of the answers' energy
is the relative error. The verdict (`lab/machine/verdict.py`) reads each answer's joules plus the gap after it above
idle against its joules alone: no_difference, with at least 10 samples in every window, conservation, a quiet idle
window and no other client's request during the run (vLLM's counters before and after), calls the attribution
trustworthy at the sampler's rate. The raw rows go to `fixtures/telemetry/attribution_check_<tag>_{samples,requests,run}.jsonl`
and one ledger row (verb `attribution-check`, task T8, node pod-a, `low_confidence` while the sensor's own error is
unmeasured) holds the result. Tests: `tests/test_lab_attribution_check.py` (a synthetic trace through the real pod
sampler and a fake teacher, worked out by hand).

## Trainer (pod B)

```
bash lab/pod_b_trainer.sh proxy      # once, then relaunch
bash lab/pod_b_trainer.sh deps       # peft, trl, transformers, accelerate, datasets, gguf (torch stays the pod's)
bash lab/pod_b_trainer.sh llama      # llama-quantize / llama-cli / llama-server built for the pod's gfx target (rocminfo; GFX= overrides)
bash lab/pod_b_trainer.sh download   # Qwen/Qwen3.5-4B (the student the Jetson runs)
bash lab/pod_b_trainer.sh smoke      # lab/smoke_lora.py: tiny LoRA SFT, prints loss and samples/s
```

`smoke_lora.py` is the template for the real trainer (L3): bf16, PEFT LoRA on all linear layers, TRL
`SFTTrainer`, forkserver DataLoader workers, no `torch.compile`.

## Datasets: the box's export and the no-camera fixture

An approved export from the box is a folder named for its dataset id, with `manifest.json`, `examples.jsonl` and
`images/` (the shapes are in `contracts/learning.py`). `lab/dataset.py` reads and checks one, and writes the
teacher's labels back into it. Only `pydantic` and the repo's `contracts/` are needed, so it runs on a pod.

```
python lab/dataset.py <dataset folder>               # checks it against the contract, prints one JSON line
python lab/dataset.py <dataset folder> --text-only   # the same for a dataset with no pictures
```

```python
from lab.dataset import load, write_target

data = load(folder)                       # DatasetError names every problem it found
for example in data.pending():            # the examples that have no target yet
    write_target(folder, example.id, label_for(example, data.image_path(example)))
```

`write_target` replaces `examples.jsonl` whole under a lock, so a labeling run can be killed and started again
and two labelers lose nothing: `pending()` is exactly the work left. Same text again is a no-op; different text
needs `overwrite=True`.

A dataset folder can also carry a frozen split: `split.json` beside `examples.jsonl`, with the example ids that
train and the ids held out for the eval gate. It is drawn once, before any training run, and not edited afterwards
(a new split is a new baseline for the current, candidate and teacher scores). `load_split` is how the trainer and
the gate decide membership, and nothing else decides it:

```python
from lab.dataset import load_split

split = load_split("fixtures/learning/tool-calling-fixture", require_images=False)
split.train                 # DatasetExample objects, in the order split.json lists them: what the trainer reads
split.held_out              # the gate's examples, which no training run may see
split.sha256                # of split.json; the ledger row of every run and score that used the split records it
split.held_out_frame_refs   # 4: how many distinct cases the held-out examples stand on
split.meets_deploy_floor    # True: 58 held-out examples is at least the 50 a gate needs before it may deploy
```

It raises `DatasetError`, naming every problem by id and never quoting a question, caption or label, unless the
split names this dataset, lists every example once on one side, lists no id that `examples.jsonl` lacks, keeps every
`frame_ref` on one side (and with it the group of questions that share it), keeps the same question (a text-only
example's `context`, case, spacing and closing punctuation folded) off both sides, and holds at least
`MIN_HELD_OUT` = 50 examples out. 50 is the owner's ruling (`lab/refine/learning/OWNER_RULINGS.md`, ruling 1): at
least 50 held-out items per task before the gate may deploy; fewer is reported but never deploys. So a short
held-out set is read only when asked for (`min_held_out=`), and then `split.meets_deploy_floor` is False and the
gate rejects. An example added to a rebuilt dataset belongs to neither side until the split is drawn again.

`fixtures/learning/tool-calling-fixture/split.json` holds out every group that has a golden question's wording (the
questions `contexts/agent/eval.py` scores): 58 examples standing on 4 frame_refs, with the other 84 examples in 12
groups to train on. Report `held_out_frame_refs` beside the example count: 58 wordings of 4 cases are not 58
independent items. To record the freeze, once, before the first training run:

```
python lab/dataset.py fixtures/learning/tool-calling-fixture --text-only --split
```

It prints the usual line plus a `split` entry (`sha256`, `train`, `held_out`, `held_out_frame_refs`) and writes a
`dataset-check` ledger row whose `refs.split_sha256` is the hash, so the row is older than any run that cites it.
A split that does not hold up exits 1 and the row says `invalid`.

`lab/fixture_dataset.py` builds the tool-calling task's dataset without any camera from
`fixtures/golden/questions.yaml` and `memory.yaml`, in the same layout, into
`fixtures/learning/tool-calling-fixture/`:

```
python lab/fixture_dataset.py                  # rebuilds it (refuses if some examples already have targets)
python lab/fixture_dataset.py --no-variants    # only the golden questions
```

Each example is a question (`context`) with an empty `target`. A label is the JSON text
`{"tool_calls": [{"name": ..., "arguments": {...}}], "answer": "..."}`, and `manifest.json` carries its schema.
`expected.jsonl` holds what a correct label looks like, taken from the agent's own pre-fetch code; the teacher
never sees it. Read `lab/fixture_dataset.py`'s docstring for how groups and `origin` support a split that keeps
the golden questions out of training. Both scripts write a ledger row and stop while `lab/state/STOP` exists.

## Training rows for the student: `lab/format_traces.py`

The Jetson serves Qwen3.5-4B with `llama-server --jinja --reasoning-budget 0`, so the model's own chat template renders
every prompt with thinking off and the model answers a tool question in the template's tag form, not in JSON:

```
<tool_call>
<function=last_seen>
<parameter=label>
keys
</parameter>
</function>
</tool_call>
```

The teacher hands its calls back as OpenAI JSON. `lab/format_traces.py` turns the labeled examples of a tool-calling
dataset (the no-camera fixture, once the teacher has filled `target`) into the rows the trainer reads, in that form, and
writes a row only when it can prove it. It needs nothing beyond the repo's own requirements.

```
python lab/format_traces.py <dataset folder> --out rows.jsonl --split             # the training rows
python lab/format_traces.py <dataset folder> --out rows.jsonl                    # a row for every labeled example
python lab/format_traces.py <dataset folder>                                      # only checks
python lab/format_traces.py <dataset folder> --out rows.jsonl --split --tokenizer <folder>   # also encodes and decodes every trace
```

`--split` reads the dataset's frozen `split.json` the way `load_split()` does (a split that leaks or is too short is
refused) and writes rows only for the examples it lists under `train`, so a held-out example cannot reach the trainer
through this file. Make the training rows with it. The answer then has a `split` entry (the sha256 of `split.json`
and how many held-out examples were skipped), and the ledger row records the hash as `refs.split_sha256`.

It prints one JSON line (`rows`, `rejected`, `failed`, `pending`, the sha256 of the rows file, and the reasons for
the first few rejects) and exits 1 when no row passed. It writes a ledger row (`format-traces`) and stops while
`lab/state/STOP` exists.

A row is one trace, with these keys:

| key | what it is |
|---|---|
| `id`, `frame_ref` | the example's, so a split by id or by `frame_ref` can pick rows |
| `messages` | the conversation, OpenAI style: system, user, the assistant's `tool_calls` (arguments as objects), the `tool` results, the final answer |
| `tools` | the `TOOLS` schemas of `contexts/agent/application/qa.py`, as llama-server receives them |
| `text` | `messages` and `tools` rendered through the template with thinking off: what the student is trained on |
| `spans` | `[start, end)` character offsets into `text` of what the model writes in each assistant turn, after the empty thinking block up to and including `<|im_end|>` |

Train on `text` with the loss on `spans` only: `text[:start]` is exactly what the box sends before that turn (the
generation prompt ends with the empty thinking block, so the model never has to write it). If you render again from
`messages` and `tools`, read the rows with the `json` module and pass `enable_thinking=False`: the `datasets` library
gives the argument objects of different tools one shared set of keys, and the template would then print the missing
ones as `None`.

What is proven for each row, and what happens to a row that fails:

- The renderer is the template written out in Python (no Jinja), checked against what the real template produced for
  16 conversations recorded in `fixtures/lab/qwen35-template/` (revision in `source.json`). With `--tokenizer <folder>`
  holding `tokenizer.json` and `chat_template.jinja` (and `pip install tokenizers transformers`) every trace is also
  compared with `apply_chat_template(..., enable_thinking=False)` itself.
- Each assistant turn starts exactly where the generation prompt ends; the text survives encode and decode unchanged
  (needs `--tokenizer`); and the calls and the answer read back out of the detokenized text have the same function names,
  arguments and words as the trace.
- A label that is not valid JSON, calls an unknown tool, leaves out or invents an argument, or has the wrong type is
  `rejected`; a trace holding a special token or a thinking tag in its text, or whose rendering does not check out, is
  `failed`. Both are counted and left out, never repaired. Arguments are written in the order the tool declares them.
- A label holds the calls and the answer, not what the calls returned. Each call is run again through the agent's own tool
  loop against `fixtures/golden/memory.yaml` (`--golden`), which gives the text the box would have shown the model. That
  is possible only for the fixture dataset (`action_id` `fixture:golden-questions`), whose memory is in the repo.

To get the student's tokenizer files without the model weights (about 13 MB), take `tokenizer.json`,
`tokenizer_config.json` and `chat_template.jinja` from `https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<revision>/` into
one folder. If the template changes upstream, record it again with the call in `fixtures/lab/qwen35-template/source.json`.
Not proven here: that the Jetson's llama.cpp build renders the same prompt and parses these calls back into
`tool_calls`; that needs a run on the box. From the source, llama.cpp's own template engine (`common/jinja/value.cpp`
at 8b54361) writes `tojson` with the same separators and key order, and prints `True`, `False` and `None` the way the
Hugging Face renderer does, so the system block it builds from the `TOOLS` list should be the one in `text`.

## Packages: the pod's file server and the box's pull (R10)

A pod cannot reach the box, so the box pulls a finished model package (`manifest.json`, the GGUF, the vision projector)
from the trainer pod through the hub proxy. Two facts shape it. `python3 -m http.server` cannot answer a Range request
(it sends every file whole), and jupyter-server-proxy refuses any response body over 100 MiB, so a multi-GB GGUF only
crosses the proxy in Range parts. The server is `lab/package_server.py` (standard library only), the pull is
`lab/package_pull.py`, and `lab/package_fake.py` is the fake proxy their tests (and the next measurement tasks) use.

```
python -m lab.package_pull serve-start --account C           # put lab/package_server.py on the pod and start it (port 8300)
python -m lab.package_pull curl --account C --version V      # the curl lines that prove the path from outside the pod
python -m lab.package_pull probe --account C --version V     # one whole-file GET of the GGUF, once; records the proxy's status
python -m lab.package_pull pull --account C --version V --dest data/models [--part-mib 64] [--force]
python -m lab.package_pull serve-stop --account C
```

- The server serves `~/data/packages/<version>/` (the persistent home, not the Jupyter root a relaunch wipes) at
  `https://<hub>/user/<account>/proxy/8300/<version>/<file>`, `<hub>` being the address `lab/hub.py` resolves for the
  account (above). It listens on 127.0.0.1, answers `bytes=a-b`, `a-` and `-n`
  with 206 and a Content-Range (416 past the end, `If-Range` honored), lists `/` and `/<version>/`, and never writes a
  request header or a query string to its log: the proxy forwards the hub token to it. By hand on a pod:
  `nohup python3 lab/package_server.py --root ~/data/packages --port 8300 --log ~/package_server.log &`.
- `serve-start` runs a script on the pod through `Controller.run_script` that carries the server's source, so nothing is
  uploaded first. It needs a ready pod with jupyter-server-proxy loaded (the `proxy` step of the pod scripts, then one
  relaunch) and is refused while `lab/state/STOP` exists.
- `pull` reads the manifest (`ModelManifest`), pulls each file in Range parts of 64 MiB by default (never 96 MiB or more),
  checks its sha256, and writes `manifest.json` last, so a half-pulled package is never mistaken for an installed one
  (`--dest data/models` leaves `data/models/<version>/`, the folder `runtime/modelstore.py` verifies and activates).
  It continues a stopped pull from the `.part` file, tries gateway errors and dropped connections again, and does not
  retry a login redirect, a 403, 404, 424 (pod stopped) or 500 (nothing on the port). A pull that sends nothing records no speed.
- The ledger gets a row for the call (`package pull`) and the measurement (`link pull`): `link.pull_mb_s` (MB of 10^6
  bytes over the seconds the transfers took), with the node, `refs.account`, and the fields bytes, seconds, part_bytes,
  parts, retries, wall_seconds and version; conditions are the path, bytes, part size, hub and kind of machine. The
  speed is written only for a package whose checksums matched. `probe` writes `link probe` with the status, the bytes
  that arrived and the size of the file on the pod.
- Reuse: `from lab.package_pull import PackageClient, pull_package, probe_whole_file` (R14 and L18), and
  `lab.package_fake.write_package` / `FakePodProxy` to rehearse them against a fake hub.

Only a pod can show these, and nothing here has run against one yet: that the live proxy passes a 206 and its
Content-Range through, the status it answers to a whole file over 100 MiB (599 is the prediction; `probe` records it),
the real MB/s of a single stream of 64 MiB parts, and that the SJC pod image has `python3` and the proxy extension.
The `/files/` route of the Jupyter server (Range without a pod-side server, but only under the Jupyter root) is
the other path; experiment E9 on the queue compares them.
## Link speed: `python -m lab link-speed` (R14)

Measures how fast a few-GB package crosses the hub: a seeded random file (it does not compress) goes up to the pod
through the contents API in base64 chunks (`lab/transfer.py`), comes back through the proxy from R10's file server in
Range parts (`lab/package_pull.py`), and is checked by sha256, three repeats each way. A short probe first doubles one
single PUT until the contents API refuses it and keeps the last size it accepted.

```
python -m lab link-speed --account D [--sizes-mb 1000,3000] [--repeats 3] [--chunk-mib 64] [--part-mib 64]
                         [--port 8301] [--probe-ceiling-mib 256] [--no-probe] [--scratch r14-link-speed] [--report lab/R14_REPORT.md]
```

- It runs the file server on its own port (8301) from a copy named `r14_file_server.py` in `~/.r14_link_speed`, so a
  standing package server on 8300 is left alone, and it serves the folder the push wrote to. At the end it stops that
  server, deletes `~/.r14_link_speed`, the scratch folder and the controller's script files, so the pod is as found.
  No GPU and no server restart; it needs a ready pod with `jupyter-server-proxy`.
- One ledger row per run (`link push`, `link pull`) with `link.push_mb_s` or `link.pull_mb_s`, bytes, seconds,
  `chunk_bytes` or `part_bytes`, the node, `hub` and `refs.account`; a failed run is an `error` row with no speed. The probe
  is one row (`link part-probe`, metric `link.max_part_mb`). The Markdown table is written to `--report`.
- On SJC the edge caches `GET /user/<account>/api/contents/<path>?content=0` for four hours (`cache-control: public,
  max-age=14400`, `cf-cache-status: HIT`), so a size read back after a rewrite is the old size. This verb adds a
  never-repeated query parameter to every contents read (`FreshLink`); `lab/transfer.py` does not, and a push of a
  different size to a path it has read before can fail its size check. The proxy route answers `no-store`.
- A push's seconds cover the whole call (folders, chunks, size read back); a pull's seconds are its requests only.
  No MB/s bar applies; this measures.
- Rehearsal: `tests/test_lab_link_speed.py` runs it at 1 MB and 3 MB against `lab/proxy_fake.py` and `lab/package_fake.py`.

## Labeling with the teacher: `lab/label.py`

`lab/label.py` asks the teacher about every example of a dataset that has no `target` yet, checks each answer and
stores the ones that pass with `lab/dataset.py`'s `write_target`, so a run can be stopped and started again with
nothing lost and nothing asked twice.

```
python lab/label.py data/exports/<dataset_id> --account C                   # through the hub proxy to pod C's port 8000
python lab/label.py fixtures/learning/tool-calling-fixture --account C --workers 4
python lab/label.py <dataset folder> --url http://127.0.0.1:8000             # a server reached directly
python lab/label.py <dataset folder> --account C --limit 5                   # a short first run
```

- **What it reads.** Only an approved export, a `data/exports/<dataset_id>/` folder named for the dataset id in its
  manifest (whose `action_id` `lab/dataset.py` already requires), or the no-camera fixture. Anything else is refused
  before the teacher is asked. Run on a pod (`JUPYTERHUB_USER` set) it refuses an export: label an export where it
  lives and reach the teacher through the proxy, so no copy of it is ever left on a pod. Images go inline in the
  request as data URLs; vLLM keeps no request on disk unless started with `--enable-log-requests` (off by default at
  the pod's commit `b1388b1`, and `pod_a_teacher.sh` does not pass it).
- **How it asks.** Model `teacher`, thinking off on every request (`chat_template_kwargs: {"enable_thinking":
  false}`), temperature 0, top_p 1, seed 0, at most 512 new tokens (`--temperature`, `--top-p`, `--seed`,
  `--max-tokens` change them). Before the first example it reads `GET /v1/models` and stops unless `teacher` is served
  from a Qwen3.8-27B checkpoint (`--expect-model ''` drops that check). Answers are streamed with `Accept:
  text/event-stream` exactly, the only value the proxy streams for; a buffered answer (`--no-stream`) longer than
  125 s is cut by Cloudflare with a 524. Gateway errors, dropped connections and broken streams are tried again after
  1, 2 and 4 s; a login redirect (bad token), 401, 403, 404 or a stopped pod (424) ends the run, as do five examples
  in a row with no usable answer.
- **Tool-trace tasks** (the output schema has a top-level `tool_calls`, like the fixture): the instruction is the
  system prompt, the question the user turn, and the teacher gets the agent's `TOOLS`. Each call it makes runs through
  the agent's own tool code against `fixtures/golden` (`--golden` for another memory), for at most the agent's three
  rounds of calls. The label is `{"tool_calls": [{"name", "arguments"}], "answer"}`, and `traces.jsonl` beside
  `examples.jsonl` gets one row per kept label with each call's `result` as the teacher saw it.
- **Checks.** A label is dropped, counted and left pending (never repaired) when it is `truncated` (hit the token
  limit), carries a `thinking` block, is `empty`, is `not_json` or breaks the `schema` (the task's
  `TaskSpec.output_schema`, checked with `jsonschema`), passes `bad_arguments` to a tool, makes calls past the last
  round (`tool_rounds`), or answers with `tool_markup` the server did not parse. A free-text task (empty schema)
  passes its own check to `label_dataset(check=...)`.
- **Ledger.** One `label-example` row per example asked (outcome accepted, rejected or failed; `label.seconds`,
  `label.prompt_tokens`, `label.completion_tokens`, `label.tokens`, `label.requests`, `label.tool_calls`; the decode
  settings and served checkpoint as conditions; the reason as a word, never a label, caption or question), and one
  `label` row per run with the counts, rejects by reason and per-example means. The entry point prints one JSON line
  and exits 0 when every pending example was asked.

**Before a real run on a pod**: the serve command in `pod_a_teacher.sh` has neither `--enable-auto-tool-choice`
nor `--tool-call-parser`. At `b1388b1` vLLM then answers a request that offers tools with the model's call markup as
plain text and no `tool_calls`, which `label.py` rejects as `tool_markup` (and stops after five), so the teacher
must be served with both flags before the tool-calling fixture can be labeled (the parser named on the model card;
the research notes give `qwen3_coder` for the Qwen3.5 family). Image and free-text tasks need neither.

**Rehearsing without a pod.** `lab/teacher_fake.py` is a fake of vLLM's OpenAI server and of the hub proxy in front
of it (token, browser agent, the Accept rule, the 125 s cut, a stopped pod), built from vLLM's code at `b1388b1`.
The tests drive it in-process; `python -m lab.teacher_fake --port 8000` serves the fixture's reference answers on
127.0.0.1 so a copy of the fixture can be labeled end to end with `--url http://127.0.0.1:8000`
(`--no-tool-parser` answers as today's serve command would).

## Benchmark harness: `lab/bench.py` (R11)

Decode tokens per second and joules per generated token, the measuring stick of the kernels track (rubric K1, K3, K6).
A run is one cold server spawn; repeats are spawns. On the pod the server runs on (R20 runs it on pod B; it needs the
repo clone and `pip install --user httpx`):

```
python lab/bench.py run --engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm     --patch lab/vllm-gfx1201.diff --node pod-b [--batches 1,4,8,16] [--repeats 5] [--metrics-file]
python lab/bench.py run --engine llamacpp --checkpoint ~/models/<student>.gguf --node pod-b      # same harness
python -m lab bench attach --account C --batches 1 --requests-per-stream 4 --warmup 0 --no-probe  # from the box
```

- **Requests.** `fixtures/lab/bench/prompts.json`: two text prompts and two drawn images (`make_images.py` writes
  them; never camera frames). Greedy (temperature 0, top_p 1, seed 0), a fixed `--max-tokens` (128) with
  `ignore_eos`, thinking off, and a per-request nonce at the front whenever prefix caching may be on (both engines'
  default). Batch size B is B concurrent streams; the defaults are 1, 4, 8 and 16, plus 64 only when the KV-cache
  capacity in the serve log holds 64 requests of the workload (owner ruling 5), else a `skipped` row with the reason.
- **One run.** Idle watts with no server, spawn (`pod_a_teacher.sh`'s vLLM serve line, llama-server's, or
  `--serve-cmd` with `{port}` `{log}` `{batch}` `{max_len}` `{model}` `{checkpoint}`), wait for `/v1/models`, read the
  serve log and the process, loaded-idle watts, warm-up (one request per prompt, then one round at B; discarded and
  listed under `warmup`), one probe request alone (single-stream rate, `mem_busy_percent` while it decodes), the
  measured requests, stop. It refuses a port something already listens on and defaults to 8100: spawning on pod C
  would collide with the teacher serving there (port 8000 and most of the memory), so spawn-per-run measuring is
  pod B's (R20).
- **Numbers.** From the client's own token stream: TTFT, per-request decode rate (first to last token), the median
  per stream and the total over the decode window; watts from amd-smi through `lab/pod_power.py`'s reader and parser
  (or, attached, the T6 sampler on port 8790 through the proxy), integrated over the decode window: average, peak,
  energy, joules per token, with the idle readings beside them and THROTTLE_STATUS.
- **Conditions on every row** (rubric K1 b): engine commit and patch sha256, checkpoint, quantization, max model
  length, batch size, node, ROCm, torch, Triton, the environment read back from `/proc/<pid>/environ` (secrets left
  out, `GPU_MAX_HW_QUEUES` always present), the linear kernel classes, attention backends and KV-cache capacity from
  the serve log, the host (hub, account, hostname, kernel release, amdgpu driver, firmware, ASIC serial and the chip,
  from `amd-smi static`/`firmware` under /opt/rocm/bin), the KFD queue count, time since spawn. What cannot be read
  is null with its reason under `unavailable`. The harness reads the chip and never checks it.
- **Bands.** Spawns are banded by their own probe decode rates (largest-gap split, 10 % of the median, at least 4
  spawns); the rate and `mem_busy_percent` thresholds between bands come from those clusters. Each configuration
  reports per-spawn values with median, min, max, mean and sd, overall and per band with each band's share.
- **Rows.** `lab/state/bench/<session>/rows.jsonl` (one per run) and `summary.json`; a `bench.run` ledger row per run
  and the `bench` row of the whole call (`@recorded`); with `--metrics-file`, each configuration's medians in
  `lab/refine/metrics.jsonl`, `low_confidence` under 5 spawns.

Tested on the box against `lab/bench_fake.py` (a fake OpenAI-compatible server spawned as a real process, writing a
vLLM- or llama-server-style log, a fake `/proc` and `/sys`, and a fake pod whose amd-smi prints the Strix Halo output
recorded on pod C). Not yet seen on a pod, so R20 should check them first: the exact serve-log lines of vLLM b1388b1
for the kernel classes and KV cache (`fixtures/lab/bench/vllm_serve_synthetic.log` is made from the source's log
statements), the output of `amd-smi static --driver`/`--asic` and `amd-smi firmware` on the SJC image
(`amdsmi_*_synthetic.txt` follow amd-smi's layout, not a recording), and whether the pod exposes
`/sys/class/kfd/kfd/proc/<pid>/queues` and `mem_busy_percent`.

Live smoke, 2026-10-10, attached to the teacher on pod C through the proxy (4 requests at batch 1, 64 tokens each,
no spawn, no change on the pod): decode 2.83 tok/s per request (2.67 over the window), TTFT 0.52 s text and 3.63 s
for the 1280x720 image, 112 W average and 124 W peak from the T6 sampler, 42 J per generated token with nothing
subtracted. No row went to the metrics file: a single attached run is not a stock record.

## Correctness harness: `python -m lab correctness` (R21)

A speedup that changes outputs is a bug (kernels rubric K4). `lab/correctness.py` records the greedy outputs of a
running server on the R11 prompt set (`fixtures/lab/bench/prompts.json`: two text prompts and two drawn images, sent
with lab/bench.py's own `PromptSet`, `Settings` and endpoints) as a baseline fixture, and later compares a run with it.
Requests only: it never spawns, restarts or changes anything on the pod, and it sends one request at a time.

```
python -m lab correctness record  --account C --engine vllm --chip gfx1151 --node pod-a --engine-commit b1388b1 \
    --patch lab/vllm-gfx1201.diff --quantization int4 --rocm 7.13 --torch 2.9.1+rocm7.13.0 --triton 3.5.1 \
    --set enforce_eager=false --set serve_flags="..."
python -m lab correctness compare --account C <the same condition flags> \
    --baseline fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json [--allow-condition-drift] [--metrics-file]
```

- **Requests.** Greedy (temperature 0, top_p 1, seed 0), `--max-tokens` 128 with `ignore_eos`, thinking off, no
  per-request nonce (the prompt must be the same bytes every run), and `logprobs` with `return_tokens_as_token_ids`,
  so an output is token ids plus the logprob of each chosen token. A server that sends no logprobs is recorded by its
  streamed chunks (`token_kind` `chunk_text`).
- **Fixture** `fixtures/lab/correctness/<engine>-<chip>-<date>.json` (date in UTC): the conditions (engine, engine
  commit, patch sha256, checkpoint and max model length as `/v1/models` reports them, quantization, served model name,
  node, chip, ROCm, torch, Triton, the sampling settings, the prompt-set hash, anything given with `--set`) and per
  prompt the text, token ids, logprobs, finish reason and token counts. What the box cannot read from the pod is given
  with flags; what is neither given nor served is null with its reason under `unavailable`.
- **Compare.** Per prompt: exact token match, first divergence index (first differing id, or where one output ends),
  the baseline's and the run's logprob there and their difference (K4 d), and a normalised-text match (NFC,
  whitespace collapsed, case kept). Summary: prompts, matches, divergences, match rate, first diverging prompt. Exit 1
  with outcome `diverged` when any prompt diverges. The full result goes to `lab/state/correctness/<run id>.json`.
- **Conditions.** The compare is refused before a request is sent when engine, engine commit, checkpoint, quantization,
  max model length, served model, chip, ROCm, torch, Triton, token kind, the sampling settings, the prompt set or any
  `--set` value that both sides know differ, unless `--allow-condition-drift` (the row then says `condition_drift`). A
  changed patch hash is the change under test, so it is reported as `under_test`, not refused. A condition the run
  does not know is listed under `unverified`.
- **Rows.** One `correctness` ledger row per call (`@recorded`; `LAB_STATE_DIR` names the ledger) with
  `teacher.output_matches`, `.output_divergences`, `.output_prompts`, `.output_match_rate` and the first diverging
  prompt in its refs; with `--metrics-file` the match rate is appended to `lab/refine/metrics.jsonl` as
  `teacher.output_match_rate` (low confidence: one compare).

The baseline in the repo was recorded 2026-10-10 8:53 to 8:58 PM ET (2026-10-11 00:53 to 00:58 UTC) from the teacher
on pod C after its restart with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (vLLM 0.19.2+gfx1151 at
b1388b1 plus `lab/vllm-gfx1201.diff`, RedHatAI/Qwen3.8-27B-INT4, bfloat16, compiled not eager, node pod-a), with R26's
batching sweep possibly loading the same server. Both commands as run, with their output:

```
$ python -m lab correctness record --account C --engine vllm --engine-commit b1388b1 --patch lab/vllm-gfx1201.diff \
    --quantization int4 --chip gfx1151 --node pod-a --rocm 7.13 --torch 2.9.1+rocm7.13.0 --triton 3.5.1 \
    --set vllm_version=0.19.2+gfx1151 --set enforce_eager=false --set dtype=bfloat16 --set serve_flags="..."
{"ok": true, "outcome": "ok", "action": "record", "fixture": ".../fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json", "prompts": 4, "tokens": [128, 128, 128, 128], "token_kind": "token_id", ...}

$ python -m lab correctness compare <the same flags> --baseline fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json --metrics-file
{"action": "compare", "prompts": 4, "matches": 2, "divergences": 2, "match_rate": 0.5, "normalised_text_matches": 2, "first_divergence_prompt": "image-bars", "first_divergence_index": 102, "condition_drift": false, "diverged": [{"prompt": "image-bars", "first_divergence": 102, "logprob_difference": 0.060536}, {"prompt": "text-schedule", "first_divergence": 106, "logprob_difference": 0.06076}], "ok": false, "outcome": "diverged", ...}
```

**The stock server does not match itself.** Compared with its own baseline minutes later, the unchanged teacher
reproduced two of four 128-token outputs and diverged on the other two at tokens 102 and 106, by logprob gaps of 0.06
(near-ties; a first compare before the restart diverged at token 113 on two other prompts, not kept). That is the K4 (c)
floor: until the cause is known (batch composition under R26's load, compiled-graph kernel choice, or the scheduler),
a changed build is held to 0.5 here, not 1.0, and K4's `--enforce-eager` arm and a quiet-server repeat are the next
steps. The row in `lab/refine/metrics.jsonl` records it as low confidence.

Tested against `lab/bench_fake.py` (spawned as a real process; `--diverge-at`, `--retoken-at` and `--diverge-match`
make a second server answer differently from a chosen position): record, compare equal, compare divergent, a
same-text-different-token divergence, conditions that differ (refused, nothing sent, and `--allow-condition-drift`).

## Placement plan: `python -m lab placement` (R46)

Which pod serves the teacher, and where the power sampler, benchmark runs, Ax trials and LoRA jobs go, with a reason
per line, which jobs may share a chip under what memory margin, and every number traced to its file (the `[n]` in
the text resolve in its Numbers list). The rules and why: `lab/refine/research/plumbing/placement.md`.

```
python -m lab placement                                    # the plan as text (--json for one JSON object)
python -m lab placement --check "L3: smoke LoRA" --gpu A   # does a slot request contradict it? exit 1 when it does
python -m lab placement --teacher-share 0.6                # the plan if the teacher were served at a smaller share
```

- Inputs: the pods' facts folder (`--facts`, else `$LAB_PLACEMENT_FACTS`, else `lab/state/placement` when it holds a
  `pods.json`, else the recorded `fixtures/lab/placement`): the `python -m lab.controller status --probe-port 8000`
  answer, the output of `bash lab/pod_a_teacher.sh facts` per pod and the pod sampler's sample file. One chip sheet per
  pod from `lab/refine/research/kernels/data/` (the newest `chipsheet-<gfx>-<date>.json`, else
  `chipsheet-sample.json`, which documents the six fields read). The GPU Queue through the slot client (`NOTION_TOKEN`;
  `--queue-file` or `--no-queue` otherwise). Nothing is called on a pod or the hub.
- Today, from the recorded facts: the teacher stays on pod-a (it serves there, 57.6 GiB held at the serve script's
  0.90 share of 64 GiB); benchmarks, Ax trials and LoRA jobs go to pod-b alone; the sampler runs on both. The teacher
  and a LoRA job fit one pod on paper (35 GiB with the weights) but not at the 0.90 share (74.65 GiB against 58 GiB
  free); at a share of 0.63 or less they would, which is R43's interference measurement to settle.
- `python -m lab slot request` asks the same check before it adds a row and refuses a request whose kind (from its
  words, or `--kind`) the plan puts on the other card, outcome `placement_contradiction`, unless `--override "why"` is
  given; the reason goes into the row's Progress note as an Override line. A plan that cannot be built never blocks a
  request.
## Chip sheet: `lab/kernels/chipsheet/` (R33)

What the Strix Halo pods' chip (gfx1151) measures about itself, so every kernel note can cite a number taken on it:
device-to-device copy bandwidth from 1 MiB to 4 GiB, fp16 and bf16 matmul at the 27B decode shapes (batch 1, 4, 8,
16) and the 4B LoRA step, a plain Triton int4-to-fp16 dequantize-plus-matmul kernel, the LDS size and the largest
workgroup LDS allocation that launches, the wavefront size the compiler chose for a probe kernel built for wave32 and
for wave64, an occupancy curve (achieved pointer-chase throughput against live VGPRs and LDS per workgroup, beside
the compiler's VGPR count and occupancy remark), the cache sizes rocminfo and the HIP runtime report, the amd-smi
watts (idle before each section, loaded during each measurement) and the ROCm, HIP, torch and triton versions. Each
number carries its exact command, its repeats and their spread. A full run stays inside 840 s of GPU time and
`--quick` inside 110 s; a measurement that would start past the budget is written as skipped.

| File | What it is |
|---|---|
| `chipsheet.py` | the pod runner (`run --node pod-b [--quick]`) and the no-GPU `self-check` |
| `chipsheet_format.py` | the sheet's JSON format, its Markdown summary and the parsers for rocminfo, hipcc, amd-smi, the probe and the compiler's resource remarks; standard library only |
| `probe.hip` | the HIP probes: device properties, wave size, LDS launch limit, occupancy kernels |
| `pod_chipsheet.sh` | what `run-script` runs on the pod: copies the uploaded files to `~/chipsheet` and runs the sheet there |
| `ingest.py` | `python -m lab chipsheet ingest`: files a pulled sheet under `lab/refine/research/kernels/data/` and writes one ledger row per section on the sheet's node |

On pod D, inside one GPU Queue slot (the pod must be up and the slot started first):

```
python -m lab.controller run-script --account D --timeout 1200 --with lab/kernels/chipsheet/chipsheet.py --with lab/kernels/chipsheet/chipsheet_format.py --with lab/kernels/chipsheet/probe.hip --with lab/pod_power.py lab/kernels/chipsheet/pod_chipsheet.sh run --node pod-b
python -m lab.transfer pull-file --account D chipsheet/chipsheet-gfx1151-<date>.json data/chipsheet-gfx1151-<date>.json
python -m lab chipsheet ingest --sheet data/chipsheet-gfx1151-<date>.json
```

The last line writes `lab/refine/research/kernels/data/chipsheet-gfx1151-<date>.json` and `.md` (commit both) and the
`chipsheet` ledger rows on node pod-b, metrics named `chip.<entry id>`. Add `--quick` after `--node pod-b` for the
two-minute version. The runner refuses (exit 2) unless rocminfo names gfx1151.

On a machine with no GPU, `python -m lab chipsheet self-check` (or `python3 lab/kernels/chipsheet/chipsheet.py
self-check`) parses the recorded tool output in `fixtures/lab/chipsheet/`, writes a sheet marked self-check to a
temporary folder, reads it back and exits 0; it measures nothing and `ingest` refuses its sheet.

## The LoRA trainer: `lab/train.py`

LoRA SFT of the student (`Qwen/Qwen3.5-4B`, the base the Jetson runs) on the rows `lab/format_traces.py --split`
writes. Pod B's role is now held by the SJC Strix Halo pod on account D (Radeon 8060S, gfx1151, 64 GB shared, ROCm
7.13, torch 2.9.1+rocm7.13.0), whose image has torch and triton and nothing else. The module docstring holds every
rule; the short version:

```
# on the workstation: the rows, from the dataset's frozen split
python lab/format_traces.py <dataset folder> --out rows.jsonl --split
# to the pod: lab/ and contracts/ (the repo cannot be cloned there), the dataset folder and rows.jsonl
python -m lab.transfer push-dataset --account D <dataset folder> --to datasets
python -m lab.transfer push-file --account D rows.jsonl datasets/rows.jsonl
# the pod's power sampler running first, so each step gets its watts
python -m lab.controller run-script --account D --with lab/pod_power.py lab/pod_power.sh start gpu-b
# on the pod, from the Jupyter root the pushes landed in
python3 lab/train.py --dataset datasets/<dataset_id> --rows datasets/rows.jsonl --model ~/models/Qwen3.5-4B
```

- **It installs its own stack.** On the first run it finds the pins of `lab/requirements-train.txt` (PEFT 0.21.2,
  TRL 1.15.0, transformers 5.19.0, accelerate, datasets, pydantic, Pillow) missing, runs pip on that file with a
  constraint file that pins the torch, torchvision, torchaudio and triton already installed (so pip can never swap the
  pod's ROCm torch for a PyPI build), writes a `train-install` ledger row and starts itself again. `--no-install`
  skips the step.
- **It runs only on the trainer pod.** Without `--cpu` it refuses unless torch is a ROCm build and the GPU is
  `--gfx` (default gfx1151). `--cpu` is a rehearsal anywhere the stack is installed: a tiny model, the CPU, nothing
  deleted, and TRL's fused LM-head loss (a Triton GPU kernel) swapped for the same computation in torch.
- **What it trains on.** Membership comes from `load_split()` only: a row that is held out, unknown to the dataset,
  from another `frame_ref` or showing another picture refuses the run, as does a split with fewer than 50 held out.
  The loss sits on the rows' `spans` and nowhere else; a span that does not start and end on a token boundary of the
  whole text, or a text the tokenizer does not give back, refuses the run. Nothing is truncated: `--max-length` below
  the longest example refuses the run.
- **Settings.** The defaults are R36's pinned baseline for this step on gfx1151
  (`lab/refine/research/kernels/settings_training_gfx1151.md` section 13): LoRA r 16, alpha 32, dropout 0, learning
  rate 2e-4, batch 4 by gradient accumulation 4 (effective batch 16), gradient checkpointing with `use_reentrant`
  False, SDPA, fused AdamW, two forkserver workers with pinned memory, seed 42, no packing, no length grouping, no
  torch.compile. On the pod it exports `MIOPEN_FIND_MODE=NORMAL` and unsets every other variable the sheet pins
  before torch is imported. The search-space settings a flag can move: `--no-gradient-checkpointing`, `--batch`
  with `--grad-accum`, `--group-by-length`, `--aotriton-experimental`, `--blas-backend hipblaslt` and
  `--tunableop <results file>`. `refs.settings` gives the run's value of every search-space parameter under its own
  name (as `data/searchspace-training-gfx1151.json` names them), `refs.env` the environment.
- **The model.** Loaded in bfloat16 by the script; the text-only class for text rows, the image-text class and its
  processor for image rows (with the labels following the processor's image-token expansion). LoRA on every linear
  layer of the language model, the vision encoder and projector excluded and checked frozen before the first step.
  Image rows also need torchvision, which the requirements cannot pin without replacing the pod's torch: check
  `python3 -c "import torchvision"` on the pod before an image run (it is refused up front without it).
- **What it leaves.** One `train` ledger row per run whatever the outcome (run_id, dataset_id, action_id, the sha256
  of `examples.jsonl`, of `split.json` and of the rows, the base's Hub repo and commit, seed and data_seed, the
  library versions, the full LoraConfig and SFTConfig as JSON text, and `trainer.samples_per_s` measured as
  `metrics.jsonl:13` measures it); one `telemetry.train.step` event per optimizer step in `<out>/telemetry.jsonl`
  on node `pod-b` (loss, step_ms, tokens, socket_w); the adapter in `<out>/adapter/` and the answer in
  `<out>/run.json` (`--out` defaults to `~/runs/<run_id>`). The row is written to the pod's own
  `lab/state/ledger.jsonl`; bring it back with the run.
- **Power has one reading path.** The trainer never reads amd-smi. It reads the pod sampler's samples in
  `~/atlas-power` as they are written and puts every window through `lab/pod_power.py`'s `summarize()`, the summary
  `python -m lab gpu-power record` stores: a step's `socket_w` is its window's `mean_w` (null when the step is
  shorter than the sampler's period and holds no sample), its `energy_j` is set only from the card's energy counter
  (N/A on the Strix Halo, so null there), and the run's own window gives `trainer.mean_socket_w`,
  `trainer.peak_socket_w` and `trainer.energy_j` (integrated power), with the whole summary in `refs.power_window`.
- **What it deletes.** On the pod, once the node check passed, the dataset copy and the rows file are deleted however
  the run ends; `refs.data_deleted` says whether that worked, and a folder that does not look like an export is
  left and named (a trained run then ends `data-left`, exit 1).
- **Base commit.** A folder from `hf download --local-dir` records its commit (`.cache/huggingface/download/*.metadata`),
  which the row takes; a folder without it needs `--base-revision <40 hex>`.

Tests: `tests/test_lab_train.py` (rows, membership, span labels, the base's commit, the power feed, the step events,
the node check, deletion, the install, the entry point's refusals) runs everywhere; `tests/test_lab_train_cpu.py`
trains tiny Qwen3.5 models, text and image-text, on the CPU and is skipped where the training stack is not
installed (CI). `fixtures/learning/train-tiny/` is six labeled train examples of the no-camera fixture with that
split's held-out side, and the rows `format_traces.py --split` writes for them.

### Batching sweep on the pod C teacher (R26, Strix Halo, gfx1151)

`python -m lab bench attach --account C --node pod-a --batches 1,4,8,16,64 --repeats 3 --no-probe --idle-s 30 --max-len 16384
--quantization compressed-tensors-W4A16-INT4 --metric-prefix teacher_batching --metrics-file lab/refine/metrics.jsonl`,
2026-10-10 about 5:51 PM to 8:23 PM ET. **Every row is the running teacher (vLLM 0.19.2 for gfx1151, model `teacher`,
Qwen3.8-27B-INT4, 16k context) measured in attach mode through the hub proxy: no cold spawn, the one departure from
R20's method, because the teacher on pod C must not be restarted.** Rows carry node `pod-a` (the teacher's role), hub
sjc, account C (`roscon01`), prompt set bench-v1 (2 text, 2 drawn images, 4 requests per stream, 128 tokens with
`ignore_eos`, temperature 0, thinking off, per-request nonce), watts from the T6 sampler (port 8790, 1 s). Engine commit,
patch hash, ROCm, torch, Triton, serve-log kernels, KV capacity and the process environment cannot be read from the box in
attach mode and are recorded as unavailable, not guessed. Medians of 3 repeats (low confidence: verdict() wants 5);
per-run rows are in `lab/state/bench/<session>/rows.jsonl` and the ledger (`bench.run`), the medians in `metrics.jsonl`
as `teacher_batching.*`.

| Concurrent | tok/s per stream (min to max) | tok/s total | TTFT s | avg / peak W | J/token gross | J/token above idle |
|---|---|---|---|---|---|---|
| 1 | 2.70 (2.69 to 2.70, sd 0.01) | 2.62 | 1.18 | 96.0 / 97 | 36.66 | 32.67 |
| 4 | 2.52 (2.52 to 2.55, sd 0.02) | 9.32 | 4.87 | 96.0 / 97 | 10.31 | 9.19 |
| 8 | 2.32 (2.32 to 2.32, sd 0.003) | 16.36 | 7.78 | 96.0 / 97 | 5.87 | 5.23 |
| 16 | 2.07 (2.02 to 2.13, sd 0.06) | 26.95 | 14.95 | 96.0 / 97 | 3.56 | 3.18 |
| 64 | 0.89 (0.85 to 0.92, sd 0.03) | 49.53 | 24.39 | 96.0 / 98 | 1.94 | 1.73 |

- **Frontier.** Power is flat at about 96 W whatever the batch (decode is weight-bound), so J/token falls as 1 over the
  total rate: 36.7 J at one stream to 1.94 J at 64, 19 times lower for 19 times the throughput. Per-stream speed
  holds within 23 % up to 16 streams (2.70 to 2.07 tok/s) and then falls to 0.89 at 64, where TTFT is 24 s; going from 16
  to 64 buys 1.8 times the throughput and 1.8 times fewer joules for 4 times the streams. The knee for a throughput job is
  16 to 64; 16 keeps each answer under a minute for 128 tokens (about 62 s) and TTFT near 15 s.
- **Idle.** The "above idle" column subtracts 10.44 W, the loaded-idle read of the sampler (60 s, teacher up, no request,
  taken after the sweep). The session's own loaded-idle read (82.4 W, in the ledger rows and the per-run `idle_w`) is wrong: it
  caught requests of an earlier attempt of this sweep that was stopped after one run; that reading is not used. No server
  idle: 8 W (R29).
- **Watts.** 96 W average is below the 110 to 125 W of the single-stream smoke; the sampler's 1 s reads sit at 96 to 97 W through
  every window, which looks like the package at a steady power limit. Treat the energy as the sampler's, not a second meter's.
- **64 fits.** Attach cannot read the serve log, so the harness could not say whether 64 requests fit (`fits` is null in the
  rows); vLLM's `/metrics` reports 778 GPU blocks of 784 tokens and 64 requests of about 1,100 tokens ran with no error or
  rejection (about 52 in flight on average). The model is hybrid, so that block count is not a plain token count.
## Request-side settings sweep: `lab/sweep_request.py` (R52)

The settings a request carries, measured on one server spawn of the stock teacher on pod D, so no cell restarts the
server: batch size (B concurrent requests, one each), max tokens, thinking off or on, the prompt length class and the
sampling profile. The grid is committed, `lab/refine/research/kernels/data/sweep-request-grid.json`: four slices
(`base` at 128 tokens, `tokens-32`, `thinking-on-32`, `prompt-long` at about 1,100 prompt tokens) each at batch 1, 4, 8
and 16, three repeats per cell run interleaved over the grid (forward, backward, forward). `base-b1` is the pinned
baseline cell: the operating point the inference settings sheet measured (batch 1, thinking off, temperature 0, a
short text prompt). Sampling stays greedy: the sheet marks no sampling setting as changing decode work on its pinned
server (its only sampling line is the MTP acceptance rate, and MTP is a restart setting held off); a profile under
`sampling` and a slice that sets it add the axis without a code change. The server is launched from the pinned
baseline of `searchspace-inference-gfx1151.json` (its fixed launch, the baseline value's flags and variables, its
unset list removed with `env -u`), with `lab/bench.py`'s stock vLLM line as the fallback if that launch dies before it
is ready; the result says which one served.

```
python -m lab sweep-request --dry-run [--quick]          # the grid, each cell's expected seconds, the slot minutes
python -m lab sweep-request pod-script [--quick]         # writes lab/state/sweep_request/pod_run.sh (needs AUP_HUB_USER_D)
python tooling/dispatcher/pod_exec.py D term <clone>/lab/state/sweep_request/pod_run.sh 3600 > <clone>/lab/state/sweep_request/pod_run.out
python -m lab sweep-request collect lab/state/sweep_request/pod_run.out --metrics-file
```

- **Slot.** `--dry-run` gives 55 minutes for the full grid (16 cells, 48 runs) and 30 for `--quick` (the `base` slice,
  four cells), from the grid's timing model: pod D's 523 s cold start, pod C's 2.88 tok/s at batch 1, a 5 % per-stream
  slowdown per extra stream and 400 prefill tokens a second (both guesses on the slow side), plus 10 %. The pod run
  never starts a cell its deadline (the slot minutes) cannot hold; such a cell is a skipped row with the reason.
- **The pod run.** `pod_run.sh` carries the lab files it imports, the grid, the prompt sets, the search space and the
  patch as one base64 block checked by sha256 (no tab or exclamation mark, so term mode can type it), unpacks them under
  `~/sweep_request/bundle-<sha>`, installs httpx if missing and runs `pod-run`. The gate refuses before anything
  starts unless the node is pod-b, the machine x86_64, rocminfo names gfx1151, the Jupyter user hashes to account D's
  (the script holds the hash, not the name), port 8100 is free and no vLLM server runs. Then idle watts, one spawn,
  idle watts loaded, one discarded 16-token warm-up round per batch size and prompt class, every cell through
  `lab/bench.py`'s `Session.measure` (the same TTFT, decode rate, amd-smi watts, joules per token and conditions as the
  harness), and the stop, however the run ends. It prints three headline lines and the result block, and keeps both in
  `~/sweep_request/last_result.b64` (`pod_exec.py D tail sweep_request/last_result.b64 3000` fetches it if the terminal
  dropped).
- **collect** writes `lab/refine/research/kernels/data/sweep-request-gfx1151-<date>.json` and `.md` (a `-quick` suffix
  for the quick grid): per cell the median of its repeats with the lowest and highest, decode tok/s per stream and in
  total, TTFT, watts and joules per token, the change against `base-b1`; the best cell by aggregate decode tok/s and its
  change over the baseline with the range the two cells' lowest and highest repeats allow; the lowest joules per token;
  what stays open. It appends one `sweep_request.cell` ledger row per cell on node pod-b and, with `--metrics-file`, the
  baseline and best cells' headline medians and the derived change to `lab/refine/metrics.jsonl`, all low_confidence
  (three repeats on one spawn). The raw result is kept under `lab/state/sweep_request/<run>/`.
- **Tested on this machine** (`tests/test_lab_sweep_request.py`) against `lab/bench_fake.py` spawned as a real server:
  the grid parser, the cell-to-harness mapping, one spawn for the whole grid, the fallback launch, the deadline, stuck
  streams, the summary arithmetic on chosen numbers, the rows, the result block, the gate and the pod script (its
  bundle runs the dry run with the standard library alone). `fixtures/lab/sweep_request/recorded_full.out` is the
  committed grid run on the fake, as the pod prints it (`make_fixture.py` records it again; the test checks the grid's
  sha256). Not yet seen on a pod: the pinned launch's flags on vLLM b1388b1, the real rates and the warm start time.
## Eval gate: `python -m lab eval-gate` (L4)

Scores the current Jetson model, the packaged Q4_K_M candidate (from L5) and the teacher on the held-out side of a
dataset's frozen `split.json`, with Dominick's scorer (`python -m contexts.agent.eval --runtime llama --base-url
<server> --json out.json`), one scorer commit and one decode setting (thinking off, greedy, 512 output tokens),
and decides deploy or reject. The current model is scored at least twice; the spread is the run-to-run noise.

```
python -m lab eval-gate --dataset fixtures/learning/tool-calling-fixture --run-id run-3 \
    --current-url http://127.0.0.1:8080 --candidate-url http://127.0.0.1:8081 --teacher-url http://127.0.0.1:8000 \
    --candidate-manifest data/models/<version>/manifest.json --decode-attested [--preview-url URL] [--events-dir data/events]
```

The paired test is an exact sign-flip test over the question groups' mean differences (group = frame_ref, or the
episode for image tasks via `--episodes`), which is exact McNemar when each group holds one item. It deploys only
with at least 50 scored items, a gain above the noise and p below 0.05; a smaller set is reported, never deployed.
It writes one ledger row per score (`eval-gate.score`) and one for the decision (`eval-gate`), each with the
decode setting, scorer commit and split sha256; `lab/state/eval-gate/<run id>/scores.json` for the slide, a
`<role>.json` per score that `lab/slide_numbers.py --eval-before/--eval-after` reads, and the
`learning.run.evaluated` event. Tests run on `lab/eval_gate_fake.py`; the real run is `lab/L4_REAL_RUN.md`. The
committed tool-calling fixture holds 58 held-out items in only 4 groups, so its smallest possible p is 0.125 and it
can never deploy: a deploy needs at least 6 independent groups, all in favour. The scorer also asks only the golden
questions, so of those 58 it scores the 24 golden wordings (the 34 variants are reported as unscored).
## Packaging an adapter for the Jetson: `python -m lab package` (L5)

`lab/package.py` turns a trained LoRA adapter into a package the box installs under `data/models/<version>/`, on pod B:

```
python -m lab package build --adapter ~/runs/<run id>/adapter --run-id <run id> --task <task> --mmproj ~/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf
python -m lab package scores --package ~/data/packages/<version> --run-id <run id>      # after the gate (L4) has scored it
python -m lab package dry-run --package ~/data/packages/<version> [--scratch DIR]
```

`build` checks the inputs (an unquantized base, a language-only adapter, a base projector named `mmproj*.gguf`), then
runs three commands through an injectable toolchain: PEFT `merge_and_unload` into the bf16 base
(`python -m lab.package merge`), `convert_hf_to_gguf.py --outtype bf16` and `llama-quantize ... Q4_K_M`. It copies the
base mmproj unchanged, writes `manifest.json` (ModelManifest, with jetson/make_manifest.py's folder rule and a sha256
per file) and leaves `~/data/packages/<version>/`, which the package server (R10, above) serves. The version defaults
to `atlas-<task>-<run id>`; an existing package is never overwritten. Its ledger row (`package build`, refs.run_id)
carries `learning.package_mb` (model gguf plus mmproj, MB), the llama.cpp commit, the three commands and
`wall_seconds`. `scores` copies `learning.heldout.<name>` from the gate's newest ledger row of the run into the
manifest's scores (never typed); `build` does it too when that row already exists. `dry-run` installs the package
under a scratch `data/models/`, switches `current` to it (every sha256 checked) and reads it back through
`runtime/registry.py`'s `model_version()`. The adapter folder is what PEFT's `save_pretrained` writes
(`adapter_config.json`, `adapter_model.safetensors`); the trainer has no script on main yet. `--toolchain fake`
rehearses the whole step with `lab/package_toolchain_fake.py`, which writes small placeholder GGUFs. Tests:
`tests/test_lab_package.py`. The commands for the real run on pod B are in `lab/L5_REAL_RUN.md`.

## Labeled tool-calling fixture: `fixtures/learning/tool-calling-labels/` (L13)

The 142 examples of `fixtures/learning/tool-calling-fixture` with a label each, made by `lab/label.py` against the
teacher (account C, through the hub proxy, thinking off, greedy, 4 examples at a time) once the teacher was served with
`--enable-auto-tool-choice` and a tool-call parser; before that restart the same request got HTTP 400. The folder
loads with `lab/dataset.py` and goes through `lab/format_traces.py --split` unchanged (84 training rows).

| File | What it is |
|---|---|
| `examples.jsonl` | the fixture's examples, each with `target`: `{"tool_calls": [{"name", "arguments"}], "answer"}` |
| `traces.jsonl` | the same labels with what each call returned, as the teacher saw it |
| `label_manifest.json` | the source, the teacher model and checkpoint, the server behavior seen, the date, the sha256 of the input (`examples.jsonl`, `manifest.json`, `split.json`), counts, rejects by reason, per-example seconds and tokens, the sha256 of every file here |
| `rejects.jsonl` | one row per example with no label and its reason as a word; empty when all are labeled |
| `label_ledger.jsonl` | the run's `label-example` ledger rows (seconds, prompt and completion tokens, requests, tool calls per example) |
| `manifest.json`, `split.json` | the fixture's, unchanged |

`expected.jsonl` stays in the fixture folder and was not touched: it is the gold of the agreement check, not a label.

```
python lab/label.py <copy of the fixture named tool-calling-fixture> --account C --workers 4
python lab/label_fallback.py package <that copy> fixtures/learning/tool-calling-fixture fixtures/learning/tool-calling-labels --state-dir <ledger folder> --run-id <id> --teacher-info info.json
python lab/label_fallback.py convert fixtures/learning/tool-calling-fixture fixtures/learning/tool-calling-labels   # cut-off route, owner ruling 12
```

`convert` is the route for a night when the teacher is not labeling: it turns the reference traces in `expected.jsonl`
into the same folder, checks every row against the task's output schema, and writes `"source": "reference traces,
teacher unavailable"` in the manifest. This run did not need it: the manifest says `"source": "teacher"`.
## Publishing the adapter: `python -m lab publish-adapter` (L8)

Puts the trained LoRA adapter and its model card on the team Hugging Face repo. The card is filled by code from the
evaluation gate's ledger row (`lab/model_card_template.md`): the task and its instruction from a `TaskSpec` JSON file,
the current, candidate and teacher scores, n as items and as groups, the run-to-run noise and the paired p-value, none
typed by hand. The metadata sets `license: apache-2.0`, `base_model: Qwen/Qwen3.5-4B`, `base_model_relation: adapter`,
`library_name: peft` and no `datasets` entry; the upload carries `LICENSE` (the Apache 2.0 text) and a line saying the
adapter is a LoRA fine-tune of the base.

```
python -m lab publish-adapter --fake --repo <owner>/<name> --adapter-dir <dir> --task-spec <spec.json> [--run-id R]
```

`--fake` runs every step against `lab/hf_fake.py` and uploads nothing; without it the command needs `HF_TOKEN` in the
environment (the name only; the value is never printed) and `huggingface_hub`, which is imported only then. Before any
push it reads the base model's card again and stops unless the license is `apache-2.0`, and scans every file of the
upload folder for an image, a caption, a label, a `frame_ref`, a `data/` path or a camera id, stopping on any hit. The
exact command for the real publish is in `lab/L8_REAL_RUN.md`.

## Label agreement spot-check: `python -m lab label-agreement`

`lab/label_agreement.py` asks the teacher again about 30 labeled examples (the first 30, or `--sample seeded
--sample-seed N`) with another sampling seed (`--seed`, default 1; the labels were made with 0) and a reworded prompt
(`REWORD_PREFIX` in front of the instruction), compares the two labels one top-level field at a time (`tool_calls`
and `answer` for the fixture; free text and lists are one field, `label`), and writes `label_agreement.json` and
`label_agreement.md` (`--out`) with the agreement rate per field and overall, every disagreement (example id, field,
first value, second value), the examples the second pass could not label (counted as disagreeing) and a verdict:
`consistent` at or above the threshold, `inconsistent` below it, `insufficient` when fewer than 30 are labeled. The
threshold is the number LT2 in `lab/refine/learning/RUBRIC.md` names for the agreement check; it names none today,
so the default is 0.9 and the report says so (`--threshold` overrides). The report holds the rubric line the merge
pass should add, with its citation; nothing under `lab/refine/` is written. One ledger row (verb `label-agreement`,
outcome the verdict, `label.agreement_rate`, `label.agreement_sample`, `label.agreement_disagreements`). The dataset
is never written; the sample is relabeled in a temporary copy. Tests use the fake teacher (`swayed_policy` makes it
answer differently on a second seed); the run against the real teacher is in `lab/L14_REAL_RUN.md`.
## Camera clips: `jetson/record_clip.py` (L22)

The gateway (`atlas-gateway`) holds `/dev/video0` open, and `/dev/video1` is the same webcam's metadata channel (v4l2-ctl
lists Metadata Capture, no Video Capture), so a second reader gets no picture and reading `/dev/video0` directly would
also skip the face blur. The recorder goes through the gateway instead and opens no device, stops no service and needs no
sudo. The capture command that worked on 2026-10-10, from Ilan's machine over Tailscale (the script goes over scp because
the box has no git login; Python 3.10 and `gst-launch-1.0` are on the box, `ffmpeg` is not):

```
scp -o BatchMode=yes jetson/record_clip.py alexander@100.124.34.36:/tmp/record_clip.py
ssh -o BatchMode=yes alexander@100.124.34.36 python3 /tmp/record_clip.py clip package-at-door --seconds 10 --fps auto
ssh -o BatchMode=yes alexander@100.124.34.36 python3 /tmp/record_clip.py clip person-walks-through --seconds 10 --fps auto --when person-moves --wait-max 90
ssh -o BatchMode=yes alexander@100.124.34.36 python3 /tmp/record_clip.py dataset keys-on-desk package-at-door --per-clip 6
```

It polls `GET http://127.0.0.1:8000/capture/cam-desk/snapshot.jpg` on the box (the camera node the gateway reads is
`/dev/video0`; the blurred frames are the only ones the recorder keeps), keeps every new frame, and joins them with
`nice -n 19 gst-launch-1.0 multifilesrc ! jpegdec ! videoconvert ! x264enc ! mp4mux`. Clips and their frames and
`clip.json` (frame offsets, measured rate, policy, mp4 sha256) land in `~/atlas/fixtures/clips/` on the box; `dataset`
writes `desk-clips-fixture/` there in the export layout, which `python lab/dataset.py <folder>` checks (12 examples, 0
labeled). It refuses to record while the Vault policy has `blur_faces` off, and refuses a clip with a hole longer than
1 s. A `degraded` camera (under 70% of the 15 fps target) is recorded and the status is written into `clip.json`.

Frame rate: that evening the gateway reported 9.8 to 10.9 fps from the reader (`degraded`; the webcam lists 30 fps at
1280x720 and has `exposure_dynamic_framerate` on, which lets it slow down in dim light, a likely but unconfirmed cause),
and the snapshot endpoint served about half of those frames as new (it blurs and encodes each new frame on demand while
the detector and the language model run; cause of the further drop not measured), so the clips came out at 5.4 to 5.6
frames a second, below the plan's 15 to 30. `--fps auto` joins the frames at the measured rate so the clip plays in real time; `--fps 15`
refuses such a camera rather than play it too fast. More frames need more light or `v4l2-ctl -d /dev/video0 -c
exposure_dynamic_framerate=0` (not done: it changes the live camera) and a quieter box. What each clip shows, sizes and
sha256 are in `fixtures/clips/README.md`; the person clip is kept on the box because the blur missed a very close face.

## Decode profile: `lab/profile_decode.py` (R47)

Where each decode step's time goes on the stock teacher build R30 left on pod D (vLLM 0.19.2 at b1388b1 with
`lab/vllm-gfx1201.diff`, RedHatAI/Qwen3.8-27B-INT4, gfx1151), at batch 1 and batch 4, with the fixed prompt set of
`lab/bench.py`. Built and tested on the box; the pod run is one GPU Queue slot on pod D, done by the dispatcher.

```
python -m lab profile-decode --quick                         # the box: parser and writer on the synthetic fixture
python -m lab profile-decode pod-script                      # the box: writes lab/state/profile_decode/pod_run.sh
python tooling/dispatcher/pod_exec.py D run <clone>/lab/state/profile_decode/pod_run.sh r47_pod_run.sh 3000 > <clone>/lab/state/profile_decode/pod_run.out
python -m lab profile-decode collect lab/state/profile_decode/pod_run.out [--chip-sheet FILE] [--save-fixtures]
```

The third line runs from the workspace (pod_exec.py is the workspace's pod tool, not in this repo): run mode uploads
the script through the Jupyter contents API and types only its `bash` line into a terminal, then reads the output
back until the script's last line, `### END rc=<code>`. The script is about 200 KB, because it carries the lab files
it imports (`lab/bench.py` and what that needs, the three profile modules, the prompt set and its two images) as one
base64 block with its sha256; it has no tab and no exclamation mark, so term mode can type it too, but a paste that
large can stall a terminal whose shell echoes faster than the terminal server drains it.

- **The gate**, fail closed, before anything starts: rocminfo names gfx1151, the Jupyter user hashes to account D's
  (the script carries the sha256 of the user name, never the name), the server port (8100) is free and no other
  vLLM server holds the GPU (the run stops nothing it did not start). Pod C, which serves the live teacher, never
  passes.
- **Phase torch.** The server is spawned from `lab/bench.py`'s vLLM launch line (the flags `pod_a_teacher.sh serve`
  uses, `--mm-encoder-attn-backend TRITON_ATTN` on gfx115x) plus `--profiler-config
  '{"profiler":"torch","torch_profiler_dir":...,"torch_profiler_with_memory":true,"ignore_frontend":true,...}'`:
  at b1388b1 that switch replaces the VLLM_TORCH_PROFILER_DIR variable of older releases
  (`vllm/config/profiler.py`), and the profiler only exists when it is set (`vllm/entrypoints/serve/profile`). Warm-up
  (each prompt and one batch-4 round at 16 tokens), then batch 1 (one request per prompt, one at a time) and batch 4
  (four streams, stream s on prompt s) at 64 tokens unprofiled, then the same two between `POST /start_profile` and
  `POST /stop_profile`; each stop writes one trace.
- **Phase rocprof.** The same server under `rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d
  <run>/rocprof/%pid% -o run --`, with `VLLM_WORKER_MULTIPROC_METHOD=spawn` (not part of vLLM's compile-cache hash) so
  the engine core leaves through the normal interpreter exit that writes rocprofv3's buffers. The stop pauses the API
  server (SIGSTOP; its own shutdown kills the engine core after 5 s), ends the engine core and waits up to 15 minutes
  for it to finish writing, then resumes and ends the API server. A tool that fails is recorded with its error, and
  the other phase still runs.
- **Attached instead** (`pod-script --attach-port N [--serve-log FILE]`): nothing is started or stopped. The torch
  phase uses the running server's own profile routes, its trace folder read from the `--profiler-config` on its
  command line (a server started without one has no `/start_profile`; the unprofiled runs are kept and the phase
  says why it stopped there), and copies the traces into the run folder; the rocprof phase runs `rocprofv3 --attach
  <engine core> --attach-duration-msec 300000` during both batches, which needs ptrace rights the pod may not grant.
  The gate then wants a server on that port, and skips the check that no other vLLM server holds the GPU.
- **The result**, `lab/refine/research/kernels/data/decode-profile-gfx1151-<date>.json` and `.md`, written by
  `collect`: time per decode step and its spread (the client's token gaps unprofiled; vLLM's
  `execute_context_0(0)_generation_<B>(<B>)` step ranges in the torch trace; rocprof's kernels inside the client's
  decode windows, cut at a kernel that runs once per step), the top 25 kernels by self time with launches, grid and
  block (rocprof's dispatch records when that run worked), the share of a step in the W4A16 matmul, attention,
  sampling (in the torch trace: whatever is launched after the step's forward range, which is `sample_tokens`),
  host-side time (the GPU ran nothing), copies and other GPU work, achieved bandwidth (the safetensors bytes a step
  reads over the step time, and the quantized bytes over the W4A16 kernel's time) against the chip sheet's copy
  bandwidth (the best measured `copy.d2d.*` entry of R33's newest `chipsheet-gfx1151-<date>.json` there, through
  `chipsheet_format.copy_peak`; a sample or self-check sheet's numbers are never used), peak and reserved memory (the
  torch profiler's allocator samples, the serve log, amd-smi), watts for every run from amd-smi, and the
  quantized-matmul kernel vLLM chose (the serve log's `Using <Kernel> for <Scheme>`, the class's file and the Triton
  kernel it launches with its file, read with the pod's own Python). `collect` also writes ledger rows on node pod-b
  (one per batch, one per tool, one for the run), and `--save-fixtures` keeps a two-step excerpt of each trace under
  `fixtures/lab/profile_decode/recorded-<date>/`.
- **Slot.** About 27 minutes with pod D's warm torch.compile cache (two server starts of about 5 minutes each, four
  runs per phase, a trace flush, the parse); ask for 30, with the one 15-minute extension in reserve if the rocprof
  flush runs long. The script stops starting new work at 30 minutes (`--deadline-min`) and starts a phase only with
  10 minutes left.

The trace parser (`lab/trace_profile.py`) is generic: the torch profiler's Chrome trace (kernels, copies, runtime
calls with their correlation ids, record_function ranges, allocator samples) and rocprofv3's CSVs, with ordered
kernel-name categories; a training-step profile can use it as it is. Tests: `tests/test_lab_trace_profile.py` and
`tests/test_lab_profile_decode.py` on the synthetic run in `fixtures/lab/profile_decode/synthetic`
(`make_fixture.py` writes it; made in the tools' documented formats, not recorded), and a whole run against
`lab/bench_fake.py`, whose server now answers `/start_profile` and `/stop_profile` and has a rocprofv3 stand-in. Not
yet run on a pod.
