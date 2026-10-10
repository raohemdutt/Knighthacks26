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
