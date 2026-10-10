# lab/ — the R9700 side (AUP Learning Cloud)

Two pods, one R9700 (gfx1201, 32 GB) each, on https://dub.aupcloud.io. Account A is the **teacher**
(serves Qwen3.8-27B), account B is the **trainer** (LoRA, eval, packaging). Logins and hub API tokens
are on the Notion Credentials page and in your local `.env`; never put them in this repo.

## Which hub is live today (2026-10-10)

| Hub | Accounts | State |
|---|---|---|
| **SJC**, https://sjc.aupcloud.io | **C** and **D** | The one to use. JupyterHub 5.4.4, resource key `gpu` (image `ghcr.io/amdresearch/auplc-base:latest`), accelerators `r9700`, `9070xt`, `strix-halo`, `strix`, `phx`. |
| **DUB**, https://dub.aupcloud.io | **A** and **B** | Kept for when it answers again; it has answered 502 all day. |

The sections below were written against DUB and its accounts A and B. Pods on SJC have so far been started only through
the hub's web spawn form (next paragraphs), so read their numbers (credits, timings, the culler) as DUB's until a pod
there has been measured. On SJC, account C's pod is a Strix Halo and runs the teacher today: see "Teacher on SJC today
(pod C, Strix Halo)" below for what it is and what differs from the R9700 recipe.

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
| `AUP_HUB_GPU_<letter>` | The accelerator the account launches by default (`strix-halo` on SJC); else `r9700`. `--accelerator` overrides it. |
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

- **Metered hub** (DUB: quota enabled, 4 credits a minute on the R9700). The hub deducts a session only when it
  stops, so the balance is flat while a pod runs. The tracker adds the running session (whole minutes x rate, from the
  server's `started`) and the settled ones (the usage transactions, spread over the minutes they ran) to give
  `credits_left`, `credits_per_hour` over `--window-hours` (default 3), `burn_per_hour` (the higher of that and what
  the running servers burn now), `hours_left` and `runs_out_at`. It alerts when the credits are gone or would not last
  until `--until` (default judging, Sunday 2026-10-11 12:00 ET); after that moment it never alerts.
- **Unmetered hub** (SJC: quota disabled, balance 0, unlimited). Nothing is projected and nothing alerts; the line is
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

- ROCm 7.13.0, `torch 2.9.1+rocm7.13.0`, `triton 3.5.1+rocm7.13.0`, `hipcc` on PATH, `python3` only.
- No root, no Docker, no SSH. JupyterLab terminal and the Jupyter REST API only. `pip install --user` works
  (binaries land in `~/.local/bin`; the scripts add it to PATH).
- `/home/jovyan` is a persistent NFS volume (models, venv packages, builds survive a relaunch).
  `/opt/workspace` (the Jupyter root) is ephemeral.
- Billing is per elapsed minute (4 credits/min on the R9700, ~60,000 per account). The server enforces no
  runtime cap; AMD staff are fine with long sessions. Stopping is immediate. A 60-minute idle culler
  stops a server that gets no requests, so keep the model server busy or ping it.
- **Tailscale is blocked** on the pods (every tailscale.com host is connection-reset). GitHub, Hugging Face
  and PyPI are reachable.

## Launching a pod

Web form: resource "Basic GPU Environment" (key `gpu`), accelerator `r9700`, runtime up to 240 (the API has
no cap). API, with the account's hub token:

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
https://dub.aupcloud.io/user/<account>/proxy/<port>/...
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
(tests/test_lab_pod_power.py; the `strixhalo_*` files are output recorded on pod C, the others the R9700's).

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

`lab/vllm-gfx1201.diff` is the set of changes vLLM (commit b1388b1, April 2026) needs on this pod: `num_stages=1`
in the Triton unified attention kernel (64 KB LDS on RDNA4), two platform-detection fixes, the torch pins removed
so the pod's own torch is used (`use_existing_torch.py`), and the GPTQ `compat.cuh` half/half2 `atomicAdd`
fallbacks guarded out because ROCm 7.13 ships those overloads itself. Build gotchas the script handles: no root,
so `cmake`, `ninja` and `pkgconf` come from `pip --user`; the image has no libdrm, so a stub `libdrm.pc` and an
empty `libdrm.so` under `~/.local` satisfy torch's `LoadHIP.cmake`; `SETUPTOOLS_SCM_PRETEND_VERSION` avoids a
`git describe` timeout on the NFS home.

Measured on 2026-10-10 with `RedHatAI/Qwen3.8-27B-INT4` (bf16 activations, 16k context, one image per prompt):

| Prompt | TTFT | decode |
|---|---|---|
| text, thinking off, 76 tokens | 0.55 s | 7.8 tok/s |
| text, thinking on, 160 tokens | 0.28 s | 7.7 tok/s |
| 1280x720 JPEG caption, 28 tokens | 1.09 s | 6.6 tok/s |

Engine init takes about 270 s (CUDA-graph capture included), the model holds 84 % of VRAM with ~10x concurrency
at 16k tokens, and the card draws ~200 W average, 300 W peak, while generating. The W4A16 layers run on
`ConchLinearKernel` (the Triton fallback; Marlin/Machete do not exist on ROCm), which is why decode is far
below the 20 to 30 tok/s the design doc estimated. That kernel is the first target for the gfx1201 kernel work.
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
curl https://dub.aupcloud.io/user/<account>/proxy/8000/v1/chat/completions \
  -H "Authorization: token $AUP_HUB_TOKEN_A" -H "Content-Type: application/json" \
  -d '{"model":"teacher","messages":[{"role":"user","content":"Describe this table in one sentence."}]}'
```

Images go inline as `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` content parts.

### Teacher on SJC today (pod C, Strix Halo)

Account C's pod on SJC (`roscon01`) runs the teacher today, at
`https://sjc.aupcloud.io/user/roscon01/proxy/8000/v1/chat/completions` with `Authorization: token $AUP_HUB_TOKEN_C`.
The GPU is a Radeon 8060S (Strix Halo, gfx1151, RDNA 3.5) with 64 GB of memory shared with the CPU. The same
`lab/pod_a_teacher.sh` builds and serves it; what differs from the R9700 recipe above:

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

Measured on 2026-10-10 on pod C (fix round 3, batch 1, 16k context, temperature 0, thinking off; TTFT and decode
with a streaming client on the pod, end to end through the proxy, which buffers the response):

| Prompt | TTFT | decode | through the proxy | watts while decoding |
|---|---|---|---|---|
| text, 31 in, 77 out | 0.39 s | 2.88 tok/s | 27.5 s | 120 W average, 129 W peak |
| 1280x720 JPEG caption, 902 in, 27 out | 3.51 s | 2.82 tok/s | 12.9 s | 118 W average |

Idle: 8 W with no server, 9.2 W with the teacher loaded. The engine is ready about 5 min after launch with a warm
torch.compile cache. Decode is about 2.7x slower than the R9700's 7.7 tok/s, about the ratio of the two memory
bandwidths (decode is weight-bound on `ConchLinearKernel`). The rows are in `lab/refine/metrics.jsonl` on node
`pod-c` (written before the rows were named by role; they are read as pod-a-role rows of account C on the SJC hub and
stay as written); the R9700 rows stay on `pod-a` as the old, other chip.

Pod power sampler, live on pod C (T6, 2026-10-10): `amd-smi metric --power --clock --temperature --usage --energy` answers on
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

## Trainer (pod B)

```
bash lab/pod_b_trainer.sh proxy      # once, then relaunch
bash lab/pod_b_trainer.sh deps       # peft, trl, transformers, accelerate, datasets, gguf (torch stays the pod's)
bash lab/pod_b_trainer.sh llama      # llama-quantize / llama-cli / llama-server built for gfx1201
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
