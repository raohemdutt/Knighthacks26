# Moving data between the box and the pods: push, pull, size caps, persistence

Research note for the plumbing track (O3 rubric tests T6 and T2), opened 2026-10-10; sources indexed in
`SOURCES.md`. Question: how do datasets get from the Jetson to a pod and model packages back, what caps each
request, where do pushed files land, and which path can carry a package of a few GB?

Labels, keys and metric names as in `hub-api-lifecycle.md`: `measured` = observed on a running system with the
setup (ours: `measurements-2026-10-10.md`, M1 to M15); `practitioner` = documented behavior or recommendation
from the people who build or run the system (vendor docs, maintainers' source at a pinned version,
maintainers' PRs); `anecdote` = one report without data; `opinion` = argument without evidence. Metrics (O3
rubric [R8]): CPH `pod.credits_per_hour`, TTS `pod.launch_to_serving_s`, MB/s `link.push_mb_s` and
`link.pull_mb_s`, RT `pod.recovery_s`, FCR `link.failed_call_rate`. Nothing here was uploaded or downloaded
beyond a 34 KB read (M12), so no MB/s figure exists yet; every MB/s number below is an experiment to run.

## 1. Where a push lands

### Takeaway
On the live pods the contents API writes into `/opt/workspace`, the ephemeral Jupyter root, not into the
persistent home. A dataset pushed with `PUT /user/<account>/api/contents/<path>` is gone after the next relaunch
unless a pod-side step moves it into `~/`. AMD's newer source would change the root to `/`, so the controller
should discover the root rather than hard-code it.

### Cited findings
- **[measured]** Pod A's Jupyter root lists only the 31 files uploaded during R2 and no `home` entry, and our verified pod facts record `/opt/workspace` as the ephemeral Jupyter root and `/home/jovyan` as the persistent NFS home. Source: `measurements-2026-10-10.md` M6 [R1]; `lab/README.md` L13, "What the pods are", at bb351e6 (PR #37), https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md#L13 [R2]

  Implies: R6's `push_dataset` through the contents API writes to storage that a relaunch wipes, so a job resumed after a lost pod (T2) would find its dataset missing unless the push is followed by a move to `~/data/` on the pod (RT, MB/s).

  Experiment (XFR-1): push a small marker file, relaunch the pod, list the root; then push, `mv` it to `~/data/`, relaunch, list again. Expected: gone the first time, present the second. Minimum sample: 1 each (queued pod B slot).

- **[practitioner]** AMD's current source maps a launch whose resource has a `defaultPath` (the `gpu` resource: `/home/jovyan`) to `notebook_dir = "/"`, and the base image's `WORKDIR` moved from `/opt/workspace` to `/home/jovyan` on 2026-07-06 (commit b5b9b7f8); the live build predates that change (M13). Source: AUP `spawner/kubernetes.py`, `_apply_target_path_mapping`, L544-L557, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L544-L557 ; `dockerfiles/Base/Dockerfile.rocm` L255, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/dockerfiles/Base/Dockerfile.rocm#L255 [S34] [S38] [S39]

  Implies: if AMD upgrades the Dublin hub during the weekend, contents paths change (from `data/x` to `home/jovyan/data/x`) and pushes start landing in the persistent home; a controller that reads the root with `GET /api/contents/` before pushing survives either layout (FCR).

  Experiment (XFR-2): at every launch, record whether the root listing contains `home`. Expected: never this weekend. Minimum sample: all launches.

## 2. Push: request caps and chunking

### Takeaway
One push request is capped by Cloudflare at 100 MB on its smaller plans, below jupyter_server's own 512 MiB. The
contents API has a chunked mode, which JupyterLab drives in 1 MiB pieces. At about 140 ms per round trip, chunk
size decides push MB/s.

### Cited findings
- **[practitioner]** jupyter_server's default contents manager, `AsyncLargeFileManager` (the async twin of `LargeFileManager`), accepts chunked uploads: `PUT /api/contents/<path>` with `type: "file"`, `format: "base64"` or `"text"`, `content`, and `chunk` = 1 to create the file, 2 and up to append, and -1 for the last chunk (which runs the post-save hooks); the REST API reference lists `content`, `format`, `name`, `path` and `type` but not `chunk`. Source: jupyter_server v2.19.0 `services/contents/largefilemanager.py`, `LargeFileManager.save` L13-L60 and `AsyncLargeFileManager.save` L85-L130, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/services/contents/largefilemanager.py#L13-L130 ; `serverapp.py`, `contents_manager_class`, L1607-L1610, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/serverapp.py#L1607-L1610 ; Jupyter Server docs, "The REST API", `PUT /api/contents/{path}`, https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html [S26] [S05]

  Implies: a multi-GB push is possible through the contents API only in chunks, and base64 adds a third to every byte on the wire (MB/s).

  Experiment (XFR-3): push the same 200 MB file in 1, 8, 32 and 64 MiB chunks. Expected: MB/s rises with chunk size until the Cloudflare cap. Minimum sample: 3 runs per size.

- **[practitioner]** JupyterLab 4.5.6 switches to chunked upload for files over `CHUNK_SIZE = 1024 * 1024` and warns above `LARGE_FILE_SIZE = 15 * 1024 * 1024`; it numbers chunks `end / CHUNK_SIZE` and sends -1 for the last. Source: JupyterLab v4.5.6 `packages/filebrowser/src/model.ts` L29-L34, L448 and L530-L533, https://github.com/jupyterlab/jupyterlab/blob/v4.5.6/packages/filebrowser/src/model.ts#L29-L34 [S31]

  Implies: 1 MiB is the maintainers' default, not a limit; with about 140 ms per sequential round trip (M7), 1 MiB chunks cap a single stream near 7 MiB/s before any bandwidth limit, so the push helper should use larger chunks or parallel streams (MB/s).

  Experiment: XFR-3. Expected: 1 MiB chunks at or below 7 MiB/s.

- **[practitioner]** jupyter_server rejects a request body larger than `ServerApp.max_body_size`, default 512 MiB ("applied even in streaming mode"), with `max_buffer_size` also 512 MiB. Source: jupyter_server v2.19.0 `serverapp.py` L1236-L1256, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/serverapp.py#L1236-L1256 [S26]

  Implies: the pod's own cap is above Cloudflare's, so Cloudflare sets the chunk ceiling (MB/s, FCR).

  Experiment: none separate; XFR-3 stays below 512 MiB.

- **[practitioner]** Cloudflare's maximum upload size is 100 MB on Free and Pro, 200 MB on Business and up to 5 GB on Enterprise; for larger uploads it advises to "break up requests into smaller chunks". Source: Cloudflare docs, "Error 413", https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/ [S16]

  Implies: until AMD's plan is known, keep each request body under 100 MB, so raw chunks under about 70 MiB once base64 is added (FCR, MB/s).

  Experiment (XFR-4): PUT bodies of 50, 99 and 101 MB. Expected: 413 only for 101 MB if the zone is on Free or Pro. Minimum sample: 1 per size.

## 3. Pull: `/files/` beats the proxy for large packages

### Takeaway
The Jupyter server's `/files/` route serves byte ranges, skips XSRF for token-authenticated calls and does not
pass through jupyter-server-proxy, so neither the 100 MB response cap nor the 300 s proxy timeout applies. It only
reaches files under the Jupyter root, and newer Tornado refuses symlinks that point out of it, so a package
written in `~/` may have to be copied there.

### Cited findings
- **[measured]** `GET /user/<acct>/files/<name>` with `Range: bytes=0-1023` returned 206 with `content-range: bytes 0-1023/34177` and `accept-ranges: bytes`; full GETs of the 34 KB file took 0.16 to 0.17 s, too small for an MB/s figure. Source: `measurements-2026-10-10.md` M12 [R1]

  Implies: pulls through `/files/` are resumable from any byte offset, so a dropped transfer costs only its last range (MB/s, RT).

  Experiment (XFR-5): pull a 1 GB package via `/files/` in 64 MiB ranges and the same file from a pod-side server behind `/proxy/<port>/` in 64 MiB ranges. Expected: both succeed, `/files/` at least as fast. Minimum sample: 3 per path.

- **[practitioner]** With a file-based contents manager, jupyter_server serves `/files/` through `AuthenticatedFileHandler`, a Tornado `StaticFileHandler`, which answers `Range` with 206; its `check_xsrf_cookie` returns early for token-authenticated requests, and JupyterHub's auth marks a request token-authenticated when the token came in a header or URL parameter. Source: jupyter_server v2.19.0 `base/handlers.py`, `check_xsrf_cookie` L530-L540 and `AuthenticatedFileHandler` L872-L901, https://github.com/jupyter-server/jupyter_server/blob/v2.19.0/jupyter_server/base/handlers.py#L872-L901 ; Tornado v6.5.10 `web.py`, `StaticFileHandler.get`, L2843-L2885, https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/web.py#L2843-L2885 ; JupyterHub 5.4.4 `services/auth.py` L812-L830 and L983-L992, https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/services/auth.py#L812-L830 [S26] [S27] [S22]

  Implies: the box can pull with nothing more than the token header (no cookie dance, no pod-side server) for packages under the Jupyter root, which may make R10's file server unnecessary for pulls (MB/s, FCR).

  Experiment: XFR-5.

- **[practitioner]** Since Tornado 6.5.9, `StaticFileHandler` rejects with 403 any path that resolves through a symlink to a location outside the served directory, unless `allowed_symlink_directory` names it (it defaults to the served path). Source: Tornado v6.5.10 `web.py`, `StaticFileHandler` docstring, L2738-L2792, https://github.com/tornadoweb/tornado/blob/v6.5.10/tornado/web.py#L2738-L2792 [S27]

  Implies: a symlink `/opt/workspace/pkg -> ~/packages` serves home files through `/files/` only if the pod's Tornado is older than 6.5.9; otherwise the package script must copy the finished package into the Jupyter root before the box pulls it (MB/s).

  Experiment (XFR-6): create the symlink and GET a file through it. Expected: 200 on Tornado below 6.5.9, 403 on 6.5.9 or later. Minimum sample: 1.

## 4. Other channels and resumable downloads

### Takeaway
The pods reach GitHub, Hugging Face and PyPI directly, so the Hub can relay model packages around Cloudflare and
the proxy. Downloads into the persistent home resume after a relaunch.

### Cited findings
- **[measured]** From a live pod, GitHub, Hugging Face and PyPI were reachable over a direct public IP, every tailscale.com host was connection-reset, and `pip install --user` persisted in the home volume across a relaunch. Source: Ilan's Team Comms page, Q1 answer, Sat 5:25 AM, https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 [R5]

  Implies: a pod can upload a finished package to a private Hugging Face repository and the Jetson can download it from there, bypassing the 100 MB proxy cap and the Cloudflare upload cap; the Hugging Face track wants adapters on the Hub anyway (MB/s).

  Experiment (XFR-7): upload a 1 GB test file from pod B to a private Hub repo and download it on the Jetson. Expected: both legs well above the proxy path's MB/s. Minimum sample: 1 per leg (queued slot).

- **[measured]** The private team repository cannot be cloned at launch because the hub has no GitHub app for it, so `lab/` is uploaded after launch. Source: `lab/README.md` L32, "Launching a pod", at bb351e6 (PR #37), https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md#L32 [R2]

  Implies: script uploads go through the contents API into the ephemeral root (section 1) and must be repeated after every relaunch unless copied into `~/` (TTS, RT); a pod-side `git clone` would need a GitHub credential on the pod, which is a credentials decision for Ilan.

  Experiment (XFR-8): time "upload `lab/` and copy to `~/lab`" in the bring-up script on each launch. Expected: under 10 s for the current `lab/`. Minimum sample: 3.

- **[practitioner]** For public repositories, AUP clones the launch's `repo_url` into `<home>/<repo>` with an init container; `repo_persist` defaults to true, and when it is false a preStop hook runs `rm -rf <home>/<repo>` at every stop. Source: AUP `spawner/kubernetes.py`, `start` L1160-L1190 and `_with_ephemeral_clone_cleanup` L759-L763, https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py#L1160-L1190 , added 2026-06-25 in commit 6e09fad and present on the live hub per M13 [S34] [S39]

  Implies: never pass `repo_persist: false` if anything under that clone must survive (RT); a public helper repo could be cloned at launch with no upload step (TTS).

  Experiment: none needed unless a public repo is used.

- **[practitioner]** Downloading with `local_dir` keeps metadata in `<local_dir>/.cache/huggingface/` so that a rerun fetches only files that are missing or changed, `hf download --dry-run` lists what is left, and `hf_xet` is the fast path installed with current `huggingface_hub` (`hf_transfer` is deprecated). Source: Hugging Face docs, "Download files from the Hub", sections "Download file(s) to a local folder", "Dry-run mode" and "Faster downloads", https://huggingface.co/docs/huggingface_hub/guides/download [S12]

  Implies: the teacher and student downloads in `lab/pod_a_teacher.sh` and `lab/pod_b_trainer.sh` resume after a lost pod instead of restarting, as long as `local_dir` is under `~/` (RT).

  Experiment (XFR-9): stop a pod halfway through an 18 GB download, relaunch and rerun. Expected: `--dry-run` shows only the missing shards and the rerun takes about half the full time. Minimum sample: 1.

## Gaps
No push or pull throughput was measured, because every probe here was read-only and the only file at the Jupyter
root was 34 KB; XFR-3, XFR-5 and XFR-7 produce the MB/s baselines the rubric asks for. The pod's Tornado version,
which decides the symlink behavior, could not be read over HTTP. AMD's Cloudflare plan, which decides the upload
cap, is unknown.

## 5. Range parts need a Range-capable server (plumbing pass 1, 2026-10-10)

### Takeaway
Python's built-in file server, the one R10's page proved the proxy path with, cannot answer a Range request, so
it always sends a package file whole, and through `/proxy/<port>/` a whole file over 100 MiB is predicted to
fail. Pulling a multi-GB package through the proxy needs a Range-capable server or files pre-split into parts.

### Cited findings
- **[practitioner]** CPython's `http.server` has no Range support: `SimpleHTTPRequestHandler.send_head` answers a GET for a file with 200 and the file's full `Content-Length`, and the module has no 206 path (also checked at v3.13.7 and on main). Source: CPython `Lib/http/server.py`, `SimpleHTTPRequestHandler.send_head`, L689-L770 at v3.12.11, https://github.com/python/cpython/blob/v3.12.11/Lib/http/server.py#L689-L770 [S45]

  Implies: through `/proxy/<port>/`, `python3 -m http.server` returns every package file whole, which jupyter-server-proxy 4.6.0 refuses above 100 MiB (`proxy-path.md` section 3, Tornado's default `max_body_size`), so R10's server has to answer Range requests itself, or the package script splits each file into parts under 100 MiB with a sha256 per part in the manifest, which any static server can serve; the `/files/` route (section 3) answers Range without a pod-side server (MB/s, FCR).

  Experiment (XFR-10): one 1 GB file pulled from pod B three ways: whole through `/proxy/<port>/` from `python3 -m http.server`, as 64 MiB pre-split parts through the same server, and in 64 MiB Range requests through `/files/`, sha256 checked. Expected: the whole pull fails (599 or a dropped connection) and both part-wise pulls succeed, `/files/` at least as fast. Minimum sample: 3 per arm.

## 6. Where a push lands on SJC (plumbing pass 2, 2026-10-10)

### Takeaway
On the SJC hub the Jupyter root of pod C again holds directories baked into the image weeks before the account
existed and no `home` entry, so a push through the contents API most likely lands in ephemeral storage there too.
Until a relaunch proves it either way, treat SJC pushes as lost on relaunch and move them under `~/`.

### Cited findings
- **[measured]** Pod C's Jupyter root on SJC listed the session's uploaded scripts plus `checkpoints` and `outputs` directories last modified on 2026-09-22, and no `home` or `opt` entry, both on the first, REST-launched pod (jupyter_server 2.21.1) and on the Strix Halo pod launched through the web form, where the root also holds `lab`, `models`, `serve` and `vllm` directories (jupyter_server 2.20.0). Source: `measurements-2026-10-10.md` M20 and M24 [S49]

  Implies: as on DUB (section 1), the contents API most likely writes into the image's working directory, which a relaunch wipes, so R6's push helper and E10 must move a pushed dataset under `~/data/` on the pod (or delete a test file) whichever hub they run on; the relaunch of the SJC pods through the web form would then have dropped every script uploaded to the root before it, and a vLLM build or model download kept in the root's `vllm` or `models` directory would be lost at the next relaunch (RT, MB/s, TTS).

  Experiment: XFR-1 as specified in section 1, run on SJC account D at its next relaunch. Expected: the marker file is gone from the root after the relaunch and present under `~/data/`. Minimum sample: 1 each.

## 7. The paths R6 and R10 shipped, and where the root is on SJC (plumbing pass 3, 2026-10-10)

### Takeaway
The box's push and pull are now code on main: pushes go through the contents API in 8 MiB chunks by default (at
most 64 MiB), and pulls come in 64 MiB Range parts (never 96 MiB or more) either through the Jupyter server's own
`/files/` route (`lab/transfer.py`, its default) or through `/proxy/<port>/` from R10's Range-capable server
(`lab/package_pull.py`), with one whole-file probe kept as the measurement of the 100 MiB limit. The queued
experiments E9 and E10 were written before this code, and their controls are not what it does. On the
form-launched SJC pods the Jupyter root is the persistent home, which section 6 did not expect.

### Cited findings
- **[practitioner]** `lab/transfer.py` (R6) pulls a file in HTTP Range parts of 64 MiB (never more than 96 MiB) with a sha256 check and resume, through the Jupyter server's own `/files/<path>` route by default (`--via files`) or through `/proxy/<port>/` (`--via proxy --port N`); `pull-package`, the box half of L6's deploy, takes the same default. `lab/package_pull.py` (R10) pulls each file of a package from `lab/package_server.py` on the pod through `/proxy/<port>/<version>/<file>` in the same 64 MiB parts, and its `probe` command makes one whole-file GET of the GGUF, once and with no retry, recording the status, the announced length and the bytes that arrived; every R10 pull writes link.pull_mb_s with the bytes, seconds, part size, hub and account. Source: `lab/transfer.py`, module docstring and the `--via` option (default `files`), and `lab/package_pull.py`, module docstring, main at fba248a [S64]

  Implies: the box has two shipped pull paths, `/files/` Range parts (R6's default, and so L6's) and proxy Range parts from R10's server, and E9 compares neither pair: its control is a whole file from `python3 -m http.server`, which no shipped code sends, and its second arm (files pre-split on the pod) is not shipped either; only its third arm (`/files/` Range) is a shipped path. Which of the two shipped paths is faster, and whether the whole file really fails, are untested (section 3's M12 timed `/files/` on DUB only), and R14 times one path at a time (MB/s, FCR).

  Experiment (XFR-11): one 1 GB package file pulled from pod D, arm A `/files/` Range parts (`python -m lab.transfer pull-file --via files`, R6's default and L6's path, control) and arm B proxy Range parts from R10's server (`python -m lab.package_pull pull`, port 8300), 64 MiB parts in both, sha256 checked, plus one `python -m lab.package_pull probe` of the whole file. Expected: both arms succeed, A at least as fast as B (section 3, M12), the probe refused above 100 MiB. Minimum sample: 3 runs per arm.

- **[practitioner]** `lab/transfer.py` (R6) pushes a file larger than one chunk through the contents API in chunks (`chunk` 1 creates the file, 2 and up append, -1 the last), 8 MiB of the file per request unless told otherwise and at most 64 MiB (about 85 MB on the wire, under Cloudflare's 100 MB), and re-sends a chunk only when a read-back of the file's size says it did not land. Source: `lab/transfer.py`, module docstring and `push_file`, main at fba248a [S64]

  Implies: the shipped push already runs at E10's arm B (8 MiB); E10's control, 1 MiB, is JupyterLab's default and not ours, so E10's verdict would rank chunk sizes against a path the box does not use, while R14 times pushes at whatever chunk its run picks (MB/s).

  Experiment (XFR-12): a 200 MB file pushed to pod D with `python -m lab.transfer push-file` at `--chunk-mib` 8 (the shipped default, control), 32 and 64, sha256 checked after each. Expected: MB/s rises with chunk size, 64 MiB still under Cloudflare's cap. Minimum sample: 3 runs per arm.

- **[practitioner]** jupyter_server's `largefilemanager.py` is identical at v2.19.0 (DUB's pod A) and v2.20.0 (the SJC Strix Halo pods, M25), and at v2.20.0 `serverapp.py` still caps a request body at 512 MiB (`max_body_size`, L1237-L1248) and still defaults the contents manager to `AsyncLargeFileManager` (L1617-L1621). Source: jupyter_server `jupyter_server/services/contents/largefilemanager.py` at v2.19.0 and v2.20.0 and `jupyter_server/serverapp.py` at v2.20.0, https://github.com/jupyter-server/jupyter_server/blob/v2.20.0/jupyter_server/serverapp.py#L1237-L1248 [S68]

  Implies: section 2's push findings, cited at v2.19.0, hold unchanged on the pods that run (owner ruling 8: pin to 2.20.0 and re-open the cited lines before a change rests on them), so a push line on T0 can cite v2.20.0 without changing what it predicts (MB/s).

  Experiment: XFR-12, as in the previous finding.

- **[measured]** On pod D (SJC account D, form-launched) R30 read `/home/jovyan` as a 984 GB volume of the persistent kind (ubuntu-vg docker-lv, 565 GB free) and found the Jupyter root to be that same folder, so files that `run-script` uploads land in the persistent home; at 00:38Z both pods' roots held the work of the day (pod D: `vllm`, `models`, `serve`, `chipsheet`, `r14-link-speed`, `r20_bench`; pod C: `vllm`, `models`, `serve`, `lab`, `atlas-power`). Source: R30 task page, line of Sat 5:40 PM (Hemdutt's agent), table row "Home" [S61]; `measurements-2026-10-10.md` M25 [S63]

  Implies: section 6's reading (a push on SJC most likely lands in ephemeral storage) does not hold for the form-launched pods: a pushed dataset, a build and a checkpoint in the root are in the home, so E10's note to move or delete pushed files is a matter of space, not of loss, and P14's loss of the root's `vllm` and `models` at a relaunch is not expected; no relaunch with those folders in place has yet been watched (RT, MB/s, TTS).

  Experiment: XFR-1 as specified in section 1, run on SJC account D at its next relaunch. Expected: the marker file is still in the root after the relaunch. Minimum sample: 1.
