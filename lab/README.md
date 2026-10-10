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
there has been measured.

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
calls the machines `pod-c` and `pod-d`.

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
- While `lab/state/STOP` exists `launch`, `stop`, `keep-alive` and `run-script` are refused.
- Ledger rows: `controller <command>` for each call, and `pod launch`, `pod stop`, `pod status`, `pod keep-alive`
  (`alive`, `failed_ping`, `lost`) with `node` pod-a or pod-b, which are the rows `python -m lab alerts` reads.
- Tests use `lab/hub_fake.py`, a fake hub on a fake clock built from the recorded answers, so nothing launches or sleeps.

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

## Teacher (pod A)

```
bash lab/pod_a_teacher.sh proxy      # once per account, then relaunch the pod
bash lab/pod_a_teacher.sh build      # vLLM from source at commit b1388b1 + lab/vllm-gfx1201.diff, ~1 h
bash lab/pod_a_teacher.sh download   # amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (18 GB) into ~/models
bash lab/pod_a_teacher.sh serve      # OpenAI-compatible server on 127.0.0.1:8000, model name "teacher"
```

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
