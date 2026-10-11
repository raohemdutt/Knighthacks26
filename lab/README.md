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

## Verbs: `python -m lab <verb>`

Every verb has a page under `lab/docs/` (one line here per verb, alphabetical; `python -m lab` prints the same
list). A verb registers itself with `CLI_VERB = "name"` in its module (see the top of `lab/__main__.py`), so
adding one touches no shared file but this list.

- [`alerts`](docs/alerts.md): Scan the ledger for trouble and append what is new to today's alerts file.
- [`attribution-check`](docs/attribution-check.md): Validate joules per answer on pod A: answers, gaps and whole-run energy, idle watts, the missed share.
- [`bench`](docs/bench.md): Benchmark a model server: decode tok/s, TTFT and joules per token, one cold spawn per run.
- [`capacity`](docs/capacity.md): Probe one chip (free memory, bandwidth, bf16 matmul, name, ROCm) in under two minutes; write a capacity record.
- [`chipsheet`](docs/chipsheet.md): File a chip sheet measured on a pod and write its ledger rows: ingest | self-check.
- [`correctness`](docs/correctness.md): Record greedy outputs from a running server as a baseline, or compare a later run with it.
- [`credits`](docs/credits.md): Credits left on the pod accounts, credits per hour, and whether they last until judging.
- [`desk-score`](docs/desk-score.md): Score a model on the held-out desk-object examples (L24): python -m lab desk-score split|score.
- [`digest`](docs/digest.md): What a sitting starts from: what ran, metrics that moved, GPU queue, rulings, Ilan's lane, next tasks.
- [`draft`](docs/draft.md): Ask for a verb the lab lacks: leaves the owner a request and a stub (once per sitting; nothing runs).
- [`eval-gate`](docs/eval-gate.md): Score current, candidate and teacher on the frozen held-out set and decide deploy or reject (L4).
- [`experiment`](docs/experiment.md): Experiments as ledger rows: add, list, result (read by the verdict math), rank, and T0's checklist (--print).
- [`gpu-power`](docs/gpu-power.md): Summarize pod GPU power samples, or store a window of them as a ledger row: summary | record.
- [`handoff`](docs/handoff.md): End the session: leave the owner a note (no sitting reads it) and record the handoff in the ledger.
- [`interference`](docs/interference.md): interference: what a teacher-style decode load and a LoRA run cost each other on one Strix Halo chip.
- [`lab-push`](docs/lab-push.md): Push a training run's files to the box's Learning lab: `python -m lab lab-push`.
- [`label-agreement`](docs/label-agreement.md): Re-label 30 labeled examples with a second teacher pass and report how often the labels agree.
- [`ledger`](docs/ledger.md): Add to, tail and search the lab ledger (ledger add, tail, grep).
- [`link-speed`](docs/link-speed.md): Measure push and pull speed of a synthetic package on a pod through the hub, and write the table.
- [`metrics`](docs/metrics.md): Report each metric's latest reading against its previous one, or compare a metric across a time window.
- [`mirror`](docs/mirror.md): Mirror the lab digest to Overnight Status and the refinement plan changes to Ilan's Team Comms page.
- [`package`](docs/package.md): Package a LoRA adapter for the Jetson: merge, GGUF, Q4_K_M, manifest, scores, dry-run install.
- [`placement`](docs/placement.md): Which pod serves the teacher and where the sampler, benchmarks, Ax trials and LoRA jobs go (--check a request).
- [`pool`](docs/pool.md): Keep sittings running under the 16-unit budget (Sonnet 1, Opus 2, Fable 8); pool status shows the units.
- [`profile-decode`](docs/profile-decode.md): Decode profile of the stock teacher on gfx1151: --quick here, pod-script, pod-run on the pod, collect.
- [`profile-train`](docs/profile-train.md): Training step profile of the student LoRA run: where each optimizer step's time goes.
- [`publish-adapter`](docs/publish-adapter.md): Publish the trained LoRA adapter and its model card to the team Hugging Face repo.
- [`quark-compare`](docs/quark-compare.md): quark-compare (R23): AMD's Quark INT4 checkpoint against the RedHat INT4 teacher on pod B, with a verdict.
- [`ready`](docs/ready.md): Check what a sitting needs: no STOP, a writable state dir, .env key names; the hub only warns.
- [`referee`](docs/referee.md): Referee of the GPU queue: end slots past their cap, list complaints and rows that could start.
- [`report`](docs/report.md): The under-the-hood report: one HTML page and a JSON data pack that show the lab's optimization work.
- [`research-check`](docs/research-check.md): Check research notes and SOURCES.md indexes against lab/refine/RESEARCH_FORMAT.md.
- [`resume`](docs/resume.md): Remove the STOP file so verbs run again.
- [`review`](docs/review.md): Write the owner's morning file: what happened, numbers moved, rulings, complaints and open PRs.
- [`routing-closed-loop`](docs/routing-closed-loop.md): The routing hook's closed-loop test (T11): the same lab requests routed under policy one and policy two, with the
- [`rulings`](docs/rulings.md): Owner rulings: list the open ones, record the answered ones as owner rules, mirror the waiting ones to Notion.
- [`sampler-overhead`](docs/sampler-overhead.md): The pod sampler's own cost on pod B, measured with and without it (T9).
- [`sit`](docs/sit.md): Start one headless agent sitting for a task, a pass, the merge or the referee, and record it.
- [`slot`](docs/slot.md): The GPU queue for pod B: request, start, extend, done, complain, status.
- [`spot-score`](docs/spot-score.md): Score a model on the held-out SPOT questions: python -m lab spot-score split|score.
- [`stop`](docs/stop.md): Create the STOP file: every verb but ready, resume and stop is refused.
- [`sweep-request`](docs/sweep-request.md): Request-side settings sweep of the teacher on one spawn: the dry run, the pod script, the pod run, collect.

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
`python -m lab ready` probes the accounts `LAB_POD_ROLES` names (C and D on SJC; unset, A and B on DUB), each at its own `AUP_HUB_URL_<letter>`, and reports per account whether the hub answered (`reachable`), lists a server (`running`) and calls it `ready`, with its `role` and `hub`. An account with no keys, or with keys but no role, is reported as `skipped` and not probed. The `env` gate wants the keys of those accounts only; the hub being down is a warning, not a hand-off (exit 0).
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
python -m lab.controller run-script --account B --with lab/smoke_lora.py lab/pod_b_trainer.sh smoke
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
bash lab/pod_a_teacher.sh build      # vLLM from source at b1388b1 + lab/vllm-gfx1151.diff (PATCH=vllm-gfx1201.diff for the April one), for the pod's gfx target
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

### A100 reference (Colab)

The same checkpoint, `RedHatAI/Qwen3.8-27B-INT4`, measured on another chip as a reference (R57, 2026-10-10 evening ET,
UTC 2026-10-11): a Google Colab A100 runtime (NVIDIA A100-SXM4-40GB, 400 W limit, driver 580.82.07, CUDA 13.0) with
stock vLLM 0.31.0 from PyPI (torch 2.13.0+cu130; no build, no patch), served as `teacher` with `--max-model-len 16384
--gpu-memory-utilization 0.90`, thinking off. The run used `python -m lab bench run --node a100 --repeats 3`: one cold
spawn per run, batch sizes 1, 4, 8 and 16, plus 64 because the KV cache (288,165 tokens, fp8 as vLLM picked it under
`auto`) holds it. Warm-ups were discarded, each request was greedy with 128 tokens, and all 15 runs came back ok. The
harness reads amd-smi only, so it ran without watts. `nvidia-smi --query-gpu=timestamp,power.draw -l 1` sampled the
board power beside it, and `fixtures/lab/bench/a100-2026-10-10/analyze.py` integrates those samples over each run's
decode window. Medians of 3 spawns (peak is the max):

| batch | tok/s per stream | tok/s total | TTFT | avg W | peak W | J/token gross | idle W (no server) |
|---|---|---|---|---|---|---|---|
| 1 | 65.0 | 61.2 | 0.118 s | 306 | 408 | 5.02 | 46.1 |
| 4 | 59.2 | 198.9 | 0.495 s | 303 | 412 | 1.52 | 46.1 |
| 8 | 52.6 | 321.4 | 0.824 s | 321 | 419 | 1.00 | 46.1 |
| 16 | 36.0 | 462.3 | 1.056 s | 356 | 413 | 0.77 | 46.1 |
| 64 | 11.3 | 636.3 | 1.938 s | 382 | 425 | 0.60 | 46.1 |

Next to the Strix Halo at batch 1 (2.9 tok/s, 42 J per generated token, and a chip-sheet ceiling of about 12 tok/s for
4-bit decode), the A100 decodes about 22 times faster at about an eighth of the energy per token. The A100 runs
Marlin W4A16 kernels and FlashInfer attention, and its HBM2 has several times the Strix Halo's memory bandwidth. These
rows are a reference, not the lab's numbers. They sit in `lab/refine/metrics.jsonl` on node `a100` with `hub: colab`
and `other_chip: true`, and they are `low_confidence` (3 spawns, under the 5 that verdict() needs). Board power is
nvidia-smi's reading at 1 Hz, so a batch-1 decode window of 8.4 s holds only about 9 samples. A cold spawn took 3 to
9 minutes: torch.compile, the profiling pass and CUDA graph capture, with the first spawn's compile cached for the rest.
The fixture folder also holds the per-run rows and the harness summary, the first spawn's serve log and the raw
nvidia-smi samples. It also holds the greedy outputs. `vllm-a100-2026-10-10.json` is an R21 baseline: `python -m lab
correctness record --chip a100 --node a100` was run against the same serve line on a second Colab session, and it
gives 128 token ids per prompt with their logprobs, so `correctness compare --baseline` takes it directly.
`greedy_outputs.json` is the same four prompts run to 256 tokens without ignore_eos. Set against the Strix Halo's R21
baseline (`fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json`), the text-kitchen prompt matches token for token.
text-schedule first diverges at token 76, image-shapes at token 3 and image-bars at token 0. The image prompts go
through each engine's own image preprocessing and vision attention, and pod C matched itself on only 2 of 4 prompts,
so a cross-chip mismatch here is not yet a bug.

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

### The hybrid W4A16 build beside the serving teacher (R53)

`lab/vllm-gfx1151.diff` is the April patch plus vLLM PR 40977's RDNA hybrid W4A16 kernel (HIP skinny GEMM for
decode, Triton for prefill, with its three later gfx1151 tunings from vLLM main 276fbcff2), backported onto the same
b1388b1 so the pod's torch 2.9.1 and C++17 stay. Every vLLM that carries the kernel upstream wants torch 2.11 or newer.
The diff's header has a note per hunk; `git apply --check` passes on a fresh b1388b1 checkout (22 files, 35 hunks).
It is now the build's default patch; `PATCH=vllm-gfx1201.diff` (or `--patch`) selects the April patch alone.

What the build step does since R53:

- `--vllm-dir DIR --venv DIR` (or `VLLM_DIR`, `VLLM_VENV`) builds into its own clone and a venv made with
  `--system-site-packages`: it sees the image's ROCm torch and not the user site, so the serving teacher's vLLM
  (`~/vllm`, user site) is neither used nor replaced. A `--vllm-dir` without `--venv` is refused for that reason.
- A stamp (`<dir>/.atlas-build-stamp`: commit, patch hash, gfx) decides whether a folder is already built. A pod that
  holds the April build has no stamp (or another patch hash) and is rebuilt, not skipped because `import vllm` works.
- Compile jobs follow available memory, not total, and `--jobs N` (`BUILD_JOBS`) sets them: beside the teacher most of
  the 64 GB is the model's.
- `serve --venv DIR` serves from that venv, and the `### serve` line carries `kernel=<name>`, the kernel vLLM logged
  for the 4-bit layers (`Using <name> for CompressedTensorsWNA16`); run-script copies every `### <step> key=value`
  line into its `pod run-script` ledger row under `conditions.steps`. `verify` prints the build stamp and whether
  the hybrid op (`torch.ops._rocm_C.wvSplitK_int4_g`) is registered.

Build on pod C's CPU beside the serving teacher, from the repo root (no GPU slot, the teacher keeps serving):

```
python -m lab.controller run-script --account C --timeout 3600 --with lab/vllm-gfx1151.diff lab/pod_a_teacher.sh build --vllm-dir /home/jovyan/vllm-hybrid --venv /home/jovyan/venv-hybrid --jobs 8
python -m lab.controller run-script --account C lab/pod_a_teacher.sh verify --vllm-dir /home/jovyan/vllm-hybrid --venv /home/jovyan/venv-hybrid
```

Expected time: the stock build took 292 s at 20 jobs on an idle pod (R30); a new clone has no build cache and the
venv installs vLLM's Python dependencies once, so about 6 minutes idle and 15 to 30 minutes at 8 jobs beside the
teacher. The log is `~/vllm_build-vllm-hybrid.log`. `verify` should show `rejected_hunks:` empty, the three
extensions importing, `vllm_from:` under `/home/jovyan/vllm-hybrid` and `hybrid_w4a16_op: present`.

Swap, in the teacher's quiet window:

```
python -m lab.controller run-script --account C lab/pod_a_teacher.sh stop
python -m lab.controller run-script --account C --timeout 1200 lab/pod_a_teacher.sh serve --venv /home/jovyan/venv-hybrid
```

The serve line should end `kernel=RDNAHybridW4A16LinearKernel` (the April build says `kernel=ConchLinearKernel`).
The first start compiles torch.compile's cache anew (about 9 minutes cold on pod D). Roll back with `stop` and a
plain `serve` (no `--venv`): the April build in the user site was never touched.

What could still fail at compile time on ROCm 7.13 (hipcc), and the fix for each:

- A C++20 construct the grep missed in `skinny_gemms_int4.cu` (only `std::in_range` was found and replaced):
  compile that one file as C++20, which hipcc and torch 2.9.1's headers accept, by adding
  `set_source_files_properties(csrc/rocm/skinny_gemms_int4.cu PROPERTIES COMPILE_OPTIONS "-std=c++20")` after
  `VLLM_ROCM_EXT_SRC` in `CMakeLists.txt`, or replace the construct with its C++17 form.
- `__builtin_amdgcn_fdot2_f32_bf16` (the bf16 dot product, new to b1388b1's `csrc/rocm`) needs the gfx11 dot
  instructions; gfx1151 has them and upstream tuned this file on gfx1151. If hipcc says the target lacks the feature,
  the build was not given `PYTORCH_ROCM_ARCH=gfx1151` (the script sets it from `rocminfo`; set `GFX=gfx1151`).
- A missing declaration between the files (`wvSplitK_int4_g` in `ops.h`, the schema in `torch_bindings.cpp`): an
  undefined-symbol error at import of `vllm._rocm_C`, shown by `verify`. The three are in the patch; a rejected hunk
  in `verify` names the file to fix by hand.
- Out of memory while compiling beside the teacher (a compiler killed, error 137): rerun with `--jobs 4`; the build
  resumes from `build/` in the same folder.
- `AT_DISPATCH_REDUCED_FLOATING_TYPES`: present in torch 2.9.1 and already used by b1388b1's `skinny_gemms.cu`.

At load, not compile: the hybrid kernel takes only group sizes 32, 64 and 128 and no act-order (`g_idx`) layers; any
other checkpoint falls back to the next kernel and the serve line says which. The teacher checkpoint's
`quantization_config` (`group_size`, `actorder`) decides it.

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
## Joules per token on the teacher: the energy model and the number of the night (R56)

`lab/energy_model.py` predicts joules per generated token on the Strix Halo (gfx1151) from the committed chip sheet,
and checks a measured batch curve against the prediction; `lab/refine/research/kernels/energy_model_gfx1151.md`
derives it, gives the measurement protocol and the optimizer's objective. Standard library only, runs anywhere.

```
python -m lab.energy_model                       # predicted J/token at batch 1, 4, 8, 16 and the knees
python -m lab.energy_model --batches 1,16,48,64 --json
python -m lab.energy_model check fixtures/lab/energy_model/attach-sweep-pod-c-2026-10-10.jsonl
python -m lab.energy_model check <lab/bench.py rows.jsonl | sweep-request-gfx1151-<date>.json>
```

- **Model.** J/token = P(B) / tok_s(B). Decode is weight-read bound, so the socket draws about the same watts at batch
  1 as at 16 while tok/s grows almost with the batch. Predicted gross J/token decoding alone at a 512-token context:
  April build 41.7, 10.6, 5.4, 2.8 at batch 1, 4, 8, 16; a build reading weights at the 210.5 GB/s ceiling 8.1, 2.2,
  1.2, 0.7. The curve flattens near batch 45 to 52 for the teacher at the ceiling and 72 to 201 for the bf16 student.
- **Checked against R26** (the attach sweep above, the fixture is its rows): stated as the April build (2.88 tok/s,
  120 W at batch 1) the throughput holds within 7 % at every batch and the watts miss (96.0 W measured, flat, against
  120 to 129 W predicted), so J/token holds at 1 and 4 and misses by 15 to 23 % at 8, 16 and 64; anchored on the
  sweep's own batch-1 point every batch holds within 7 %. The real loaded idle is 10.44 W, read after the sweep; the
  82.4 W read before it caught an earlier attempt's requests.
- **Objective.** The night optimizer (`lab/refine/research/kernels/data/ax-config-night.json`) minimizes
  `joules_per_token.b1`, gross, with `decode_tok_s.b1 >= 2.88` and `ttft_s.b1 <= 2.0` as outcome constraints.

The before-and-after number of the night. Gross J/token from `lab/bench.py` (socket energy over the decode window
over the tokens in it), medians of the runs named. The "after" row is filled from the best measured configuration on
the rebuilt teacher (R53 / R60) once it is benchmarked; until then it is pending, not estimated.

| | Build | Configuration | Batch | tok/s per stream | tok/s total | Avg W | J/token gross | J/token above idle | Conditions |
|---|---|---|---|---|---|---|---|---|---|
| Before | April build: vLLM 0.19.2 at b1388b1 plus `lab/vllm-gfx1201.diff`, Conch W4A16, `RedHatAI/Qwen3.8-27B-INT4` | stock serve line, thinking off, greedy | 1 | 2.70 | 2.62 | 96.0 | 36.66 | 32.67 | R26 attach sweep, pod C (SJC, account C), bench-v1, 4 x 128 tokens a stream, 3 repeats, idle 10.44 W, PR #269 (081c496) |
| After | rebuilt teacher (R53 / R60) | best measured arm | pending | pending | pending | pending | pending | pending | pending: run, pod, prompt set, repeats, idle, commit |

For reference, the April build's own batch-1 smoke on pod C read 2.88 tok/s at 120 W, about 41.7 J/token
(teacher section above); the R26 row is the before number because it ran through the same harness the after row will.

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

## Experiment runner: `python -m lab experiment run` (O29)

One kernels-track experiment from its ledger definition to its verdict on pod B, in one call
(`lab/machine/experiment_run.py`):

```
python -m lab experiment run E3 --dry-run                  # check and prepare only: the plan and the pod script, nothing sent
python -m lab experiment run E3 [--task O29] [--row <row>] [--reply "..."] [--repeats N] [--config FILE]
```

- **Check.** The experiment is on the kernels track, its metric is one `lab/bench.py` measures, it has a run config
  (`RUN_CONFIGS`, today E3, or `--config FILE` in the same shape), the account (`--account`, default D) fills the node
  the experiment names, and the placement plan puts benchmarks there. Anything else is refused before the queue or the
  pod is asked.
- **Prepare and upload.** One `lab/bench.py run` per distinct argument set the arms need (E3's four batch sizes share
  one spawn per repeat and are split by `batch_size`), `--repeats` from the experiment's `min_n`. The harness travels
  inside the script as a base64 tarball, so the pod needs only python3 and httpx. The script and its plan go into
  the Jupyter root before the slot is asked for.
- **Slot.** `slot request` (kind bench; a 60-minute slot carries the config's justification) and `slot start`. A start
  the queue refuses ends the call with outcome `waiting`: the uploads are removed and the row stays Queued; run the
  same command again (`--row`, and `--reply` once the referee has written its critique).
- **Run, pull, record.** `run-script` runs the script; each harness call is interrupted (SIGINT, so it stops the
  server it spawned) 4 minutes before the slot's cap. The rows are pulled with `lab/transfer.py`, each measured run
  becomes a `bench.run` ledger row (refs run_id, slot, experiment, arm) and the arms go to `experiment result`.
- **Release and clean.** `slot done` with one result line, then `lab_exp/<run id>/` and every file the call put in the
  Jupyter root are removed. Whatever fails after the slot starts, the slot is still released, the pod still cleaned,
  and the `experiment run` ledger row names the failed step. The runner never launches or stops a pod and never
  touches a server the pod was running.

Tested against `lab/hub_fake.py` (its file store on disk, the pod script run by bash), `lab/machine/slots_fake.py` and
`lab/bench_fake.py`. The commands of the first real run are in `lab/O29_REAL_RUN.md`.

## The first real fine-tune on pod B (L16)

`lab/l16_run.sh` is the whole run, step by step, and `lab/L16_CONFIG.md` is its config with the source of every
number (R36's pinned baseline: LoRA r 16, alpha 32, lr 2e-4, batch 4 by accumulation 4, 3 epochs, seed 42). On the
workstation `dataset` writes the labeled train rows (L13's labels when they are in the repo, else the reference traces
of `expected.jsonl`, learning ruling 12), `bundle` packs `lab/` and `contracts/` for the pod and `dry-run` runs the
same command line on the CPU. On pod D, inside the GPU Queue slot, `prep` makes `~/venvs/atlas-train` over the image's
torch (pip runs only in the venv, with `PYTHONNOUSERSITE=1`, so the user site where pod D keeps vLLM is never touched)
and downloads the base at its pinned commit to `~/models/Qwen3.5-4B`; `train <run id> <dataset id>` runs the trainer
and leaves the adapter, telemetry, `run.json`, the log and the ledger rows in `~/adapters/l16-<run id>/`.
`fixtures/demo/` holds what came back: the trainer log, the loss curve, the ledger rows and the screen recording the
"It learns" fallback uses (owner ruling 7).
