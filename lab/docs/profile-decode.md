# Decode profile: `lab/profile_decode.py` (R47)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab profile-decode -h`.

Where each decode step's time goes on the stock teacher build R30 left on pod D (vLLM 0.19.2 at b1388b1 with
`lab/vllm-gfx1201.diff`, RedHatAI/Qwen3.8-27B-INT4, gfx1151), at batch 1 and batch 4, with the fixed prompt set of
`lab/bench.py`. Built and tested on the box; the pod run is one GPU Queue slot on pod D, done by the dispatcher.

```
python -m lab profile-decode --quick                         # the box: parser and writer on the synthetic fixture
python -m lab profile-decode pod-script                      # the box: writes lab/state/profile_decode/pod_run.sh
python tooling/dispatcher/pod_exec.py D term <clone>/lab/state/profile_decode/pod_run.sh 3000 > <clone>/lab/state/profile_decode/pod_run.out
python -m lab profile-decode collect lab/state/profile_decode/pod_run.out [--chip-sheet FILE] [--save-fixtures]
```

The second and third lines run from the workspace inside one GPU Queue slot on pod D: `pod-script` reads account
D's user name from the workspace `.env` to put its sha256 in the gate (it answers with an error line naming the
missing keys elsewhere), and pod_exec.py is the workspace's pod tool, not in this repo. Its term mode types the
script into a pod terminal inside a quoted heredoc, runs it and streams the output back; 3000 seconds covers the
30-minute deadline and the one 15-minute extension. The script is about 200 KB, because it carries the lab files it
imports (`lab/bench.py` and what that needs, the three profile modules, the prompt set and its two images) as one
base64 block with its sha256, in lines of 76 characters with no tab and no exclamation mark, so an interactive shell
reads it as typed. Should a paste that large stall the terminal, run mode (`pod_exec.py D run <script>
r47_pod_run.sh 3000`) uploads it through the Jupyter contents API and types only its `bash` line, then reads the
output back until the script's last line, `### END rc=<code>`.

- **The gate**, fail closed, before anything starts: rocminfo names gfx1151, the Jupyter user hashes to account D's
  (the script carries the sha256 of the user name, never the name), the server port (8100) is free and no other
  vLLM server holds the GPU (the run stops nothing it did not start). Pod C, which serves the live teacher, never
  passes.
- **Phase torch.** The server is spawned from `lab/bench.py`'s vLLM launch line (`pod_a_teacher.sh serve`'s
  checkpoint, context length, memory share, dtype and `--mm-encoder-attn-backend TRITON_ATTN` on gfx115x; the live
  teacher's second served name and `--scheduling-policy priority` are left out, as these requests carry no priority)
  plus `--profiler-config
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
  `chipsheet_format.copy_largest`, the copy over the largest buffer, since buffers that fit the 32 MiB last-level
  cache copy several times faster than memory; a sample or self-check sheet's numbers are never used), peak and
  reserved memory (the torch profiler's allocator samples, the serve log, amd-smi), watts for every run from amd-smi,
  and the quantized-matmul kernel vLLM chose (the serve log's `Using <Kernel> for <Scheme>`, the class's file and the
  Triton kernel it launches with its file, read with the pod's own Python). `collect` also writes ledger rows on node
  pod-b (one per batch, one per tool, one for the run), and `--save-fixtures` keeps a two-step excerpt of each trace
  under `fixtures/lab/profile_decode/recorded-<date>/`.
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
