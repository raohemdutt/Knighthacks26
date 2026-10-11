# L5 result: first package of the L16 adapter (pod-b, Sun 2026-10-11)

The first real package of a fine-tuned adapter for the Jetson: the L16 LoRA adapter merged into the bf16 base,
converted to GGUF, quantized to Q4_K_M, with the base vision projector and a manifest. Built on pod D (SJC account D,
node pod-b), pulled to the workstation, checked and dry-run installed. Scores are empty: the gate (L4) had not run.

## What was packaged

| | |
|---|---|
| Run id (run.json and every telemetry line) | `train-20261011T032149Z-7dd098` (the folder on the pod is named `l16-<run id>`) |
| Adapter | `/home/jovyan/adapters/l16-train-20261011T032149Z-7dd098/adapter` (adapter_model.safetensors, 129,927,008 bytes) |
| Base | `Qwen/Qwen3.5-4B`, bf16, `~/models/Qwen3.5-4B` at revision 851bf6e |
| Task | `tool-calling` |
| Version | `atlas-tool-calling-train-20261011T032149Z-7dd098` |
| llama.cpp | 23b0202a189c44a54625aadcb37a946dd1d6278d (`~/llama.cpp`) |
| lab source on the pod | main 2a7d3ea1461e7286840c81642833ae080dcf262e (`~/l5/l5-bundle.tar.gz`) |

## The package

On pod D: `/home/jovyan/packages/atlas-tool-calling-train-20261011T032149Z-7dd098/`
On the workstation: `work/PODD-ops-out/L5/package/atlas-tool-calling-train-20261011T032149Z-7dd098/` (not committed)

| File | Bytes | sha256 |
|---|---|---|
| `atlas-tool-calling-train-20261011T032149Z-7dd098-Q4_K_M.gguf` | 2,708,804,064 | b48886c0655005f59fbbee9097215a64549b230cb00dcf5b744cd5d19b18ea6d |
| `mmproj-F16.gguf` (base projector, unchanged; `unsloth/Qwen3.5-4B-GGUF` at e87f176479d0855a907a41277aca2f8ee7a09523) | 672,423,616 | cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864 |
| `manifest.json` | 533 | ea82e7bf90e8059f04c4833233d23c5c066506b0e0dd6484d95eeb407d3a33e7 |

manifest.json: version as above, base `Qwen/Qwen3.5-4B`, task `tool-calling`, gguf and mmproj as named, a sha256 for
each, `scores` {}, created 2026-10-11T04:54:20.755235Z.

## Commands, in order

GPU Queue row `3f696ef5fec581fa9f92e96936676996` (60 min asked), Running 12:51 to 12:55 AM ET (4.8 min), released.
Both steps went through `python -m lab.controller run-script --account D`, which uploads the script under its own name.

1. `l5_run.sh prep` (48 s): unpacked the bundle, built `llama-quantize` CPU-only into `~/llama.cpp/build` (31 s),
   fetched the base mmproj at its pinned commit (7 s, sha256 checked), checked the venv imports and that
   `convert_hf_to_gguf.py` starts.
2. `l5_run.sh build` (64 s; `lab package build` reports wall_seconds 63.163), which ran:
   - `~/venvs/atlas-train/bin/python -m lab.package merge --base-dir ~/models/Qwen3.5-4B --adapter <adapter> --out ~/packages/.work-<version>/merged`
   - `~/venvs/atlas-train/bin/python ~/llama.cpp/convert_hf_to_gguf.py ~/packages/.work-<version>/merged --outtype bf16 --outfile ~/packages/.work-<version>/<version>-bf16.gguf`
   - `~/llama.cpp/build/bin/llama-quantize ~/packages/.work-<version>/<version>-bf16.gguf ~/packages/.work-<version>/package/<version>-Q4_K_M.gguf Q4_K_M`

One change to `lab/l5_run.sh` as handed over: its two cmake commands run as `env -u PYTHONNOUSERSITE cmake ...`.
The script exports `PYTHONNOUSERSITE=1`, and the pod's only cmake is the pip-installed one in `~/.local/bin`, a Python
entry point that cannot import its module with that variable set (`ModuleNotFoundError: No module named 'cmake'`).
Nothing else in the script changed; the venv steps keep the variable.

## Ledger row (on the pod, `~/l5/ledger.jsonl`; copy in `work/PODD-ops-out/L5/l5-ledger.jsonl`)

verb `package build`, outcome `ok`, node `pod-b`, refs.run_id `train-20261011T032149Z-7dd098`,
metrics `learning.package_mb` 3381.22768, conditions quant `Q4_K_M`, base `Qwen/Qwen3.5-4B`,
version `atlas-tool-calling-train-20261011T032149Z-7dd098`, llama_cpp_commit 23b0202a189c44a54625aadcb37a946dd1d6278d.

## Workstation checks (2:09 to 2:10 AM ET)

1. Pull: `python -m lab.transfer pull-package --account D packages/atlas-tool-calling-train-20261011T032149Z-7dd098 --models-dir <folder>`
   answered ok, verified true: 2 files, 3,381,227,680 bytes, 59.0 s, 57.3 MB/s, 52 range requests, 0 retries, route
   `files` (the Jupyter server's own route; nothing runs on the pod for it).
2. sha256 recomputed on the pulled files: both model files match manifest.json; manifest.json is byte-identical to
   the pod's.
3. `python -m lab package dry-run --package <folder>`: ok. Installed into a scratch model store, activated (the store
   checks every sha256) and read back through `model_version()`: version, base `Qwen/Qwen3.5-4B`, task `tool-calling`,
   scores {} as the manifest says.

## To use it

- On the Jetson, from the repo root: `python -m lab.transfer pull-package --account D packages/atlas-tool-calling-train-20261011T032149Z-7dd098 --models-dir data/models`,
  then `python jetson/atlas_swap.py swap atlas-tool-calling-train-20261011T032149Z-7dd098`. The pull needs the account D
  hub keys in the `.env` (`AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D`, `AUP_HUB_URL_D`) and reaches `sjc.aupcloud.io`.
- Learning lab: `curl -X POST --data-binary @manifest.json "http://100.124.34.36:8000/api/learning/lab/import?kind=manifest&run_id=train-20261011T032149Z-7dd098"`.

## Still open

- `scores` is still empty. The gate ran on pod D at 2:12 to 2:20 AM ET (run id `train-20261011T032149Z-7dd098`,
  decision `reject`, report only: current 24/24 in both runs, candidate 24/24, teacher 22/24 on the 24 scored
  held-out items; the current arm was the base model rebuilt on pod D the way this package was built). The first
  `python -m lab package scores` then failed with `no gate row ... learning.heldout.* metrics`: the gate writes
  `eval.current`, `eval.candidate` and `eval.teacher`. The commit that adds this file makes `package scores` read
  that row; `python -m lab package scores --package <dir> --run-id train-20261011T032149Z-7dd098` copies the
  scores in once it is run with the gate's ledger as `LAB_STATE_DIR`. The manifest has not been rewritten yet.
- The base mmproj was not compared with the projector the box has installed.
- `~/l5` (4.6 MB) and `~/llama.cpp/build` (the CPU-only llama-quantize) are left on pod D.
