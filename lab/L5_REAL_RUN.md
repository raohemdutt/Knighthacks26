# L5 real run: package a trained adapter on pod B

`lab/package.py` and its tests run offline on `lab/package_toolchain_fake.py`. This file lists what was not run, and
the exact commands for the real run on the trainer pod (pod B, SJC account D), in order. Environment variables are
named here, never given values: they live in the workspace `.env` and on the Notion Credentials page.

## What the offline work did not prove (each step below closes one)

| Clause of the Done when | Proved offline by | Still needs the pod |
|---|---|---|
| PEFT merge_and_unload on the unquantized bf16 base | `test_the_merge_on_the_pod_loads_the_base_in_bf16_and_saves_merge_and_unload` (stand-in torch, transformers, PEFT), `test_a_quantized_base_is_refused_before_anything_runs` | step 6: the real merge of a real adapter, and whether `AutoModelForCausalLM` on the Qwen3.5-4B checkpoint saves a folder `convert_hf_to_gguf.py` accepts |
| convert_hf_to_gguf.py, then llama-quantize Q4_K_M | `test_the_three_toolchain_commands_run_in_order_merge_convert_quantize` | step 6: the real tools, a real GGUF |
| base mmproj kept unchanged | `test_the_package_holds_the_quantized_gguf_and_the_base_mmproj_unchanged` | step 3: the real base projector file, the same one the box runs |
| manifest.json validates as ModelManifest, one model gguf, at most one mmproj, sha256 per file | `test_manifest_validates_as_model_manifest_with_a_sha256_for_every_file`, `test_the_manifest_follows_the_folder_rule_of_make_manifest` | nothing (the same code runs on real files) |
| scores copied from the gate's ledger row by the step | `test_scores_are_copied_from_the_gate_row_named_by_the_run_id` and the other score tests | step 7: a real gate row from L4, which is not on main yet |
| llama.cpp commit, commands, learning.package_mb, wall-clock in the run's ledger row | `test_the_ledger_row_carries_commit_commands_package_mb_and_wall_clock` | step 6: real numbers |
| dry run under a scratch data/models/<version>/, model_version() reads back version, base, task, scores | `test_dry_run_installs_under_scratch_data_models_and_model_version_reads_it_back`, `test_python_m_lab_package_runs_build_scores_and_dry_run_on_the_fake` | step 8: the dry run of the real package |

Assumptions to confirm on the pod:

- The trainer (L3) has no script on main. The adapter folder expected is what PEFT's `save_pretrained` writes:
  `adapter_config.json` and `adapter_model.safetensors`; `base_model_name_or_path` names `Qwen/Qwen3.5-4B` or a local
  folder named `Qwen3.5-4B`; no target or saved module names `visual` or `vision`.
- The gate (L4) is not on main. `scores` reads its ledger row as: `refs.run_id` = the run id, metrics named
  `learning.heldout.<name>` (current, candidate, teacher, optional dotted keys such as `candidate.bf16`), and, when the
  gate names the package it scored, `conditions.version` = the package version. If L4 lands with another shape, the
  one function to change is `gate_scores` in `lab/package.py`.

## Environment variable names

Workstation (controller, run-script): `AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D`, `AUP_HUB_URL_D`, `AUP_HUB_LAUNCH_D`,
`AUP_HUB_PASS_D`, `LAB_POD_ROLES`.
Pod B, for the ledger rows: `LAB_STATE_DIR` (optional; else `lab/state` in the clone), `LAB_ACTOR`, `LAB_TIER`,
`LAB_TASK` (set it to `L5`).
Hugging Face download of the public projector: no token needed; if one is used it is `HF_TOKEN`.

## Commands, in order

Placeholders: `<run id>` the training run (L3), `<task>` its task slug, `<version>` = `atlas-<task>-<run id>` unless
`--version` is given.

1. Workstation: the pod is up and the GPU Queue slot is held (the slot is the owner's call).

   ```
   python -m lab.controller status --account D
   ```
   Expected: one JSON line, the pod `ready`, hub `https://sjc.aupcloud.io`.

2. Pod B, in the repo clone, on this branch (or main once merged): the tools.

   ```
   git fetch origin && git checkout ctx/learning/package-jetson
   bash lab/pod_b_trainer.sh deps
   bash lab/pod_b_trainer.sh llama
   bash lab/pod_b_trainer.sh download
   python3 -c "import pydantic, yaml, httpx, peft, transformers, torch; print('ok')" || pip install --user --quiet pydantic pyyaml httpx
   ls ~/llama.cpp/convert_hf_to_gguf.py ~/llama.cpp/build/bin/llama-quantize ~/models/Qwen3.5-4B/config.json
   git -C ~/llama.cpp rev-parse HEAD
   ```
   Expected: `deps` prints the torch, peft, trl and transformers versions; `llama` prints `llama-quantize present` or
   builds it; `download` prints `checkpoint present`; the import line prints `ok`; `ls` lists the three files; the
   last line prints a 40-character commit, which should be at or after the 2026-02-10 merge of llama.cpp PR #19468
   (Qwen3.5 support, research note PK-2). `grep -c quantization_config ~/models/Qwen3.5-4B/config.json` prints 0.

3. Pod B: the base vision projector, the same file the Jetson serves.

   ```
   mkdir -p ~/models/Qwen3.5-4B-GGUF
   hf download unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf --local-dir ~/models/Qwen3.5-4B-GGUF
   sha256sum ~/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf
   ```
   Expected: the sha256 equals the mmproj sha256 in the box's installed base manifest
   (`data/models/current/manifest.json` on the Jetson). If it differs, stop: the box runs another projector.

4. Pod B: the adapter is there.

   ```
   ls ~/runs/<run id>/adapter
   python3 -c "import json; c = json.load(open('$HOME/runs/<run id>/adapter/adapter_config.json')); print(c['base_model_name_or_path'], c.get('target_modules'))"
   ```
   Expected: `adapter_config.json` and `adapter_model.safetensors`; a base named `Qwen/Qwen3.5-4B` or `.../Qwen3.5-4B`;
   language-side modules only.

5. Pod B: rehearse on the fake toolchain (writes placeholders only, to a throwaway folder and ledger).

   ```
   LAB_STATE_DIR=/tmp/l5-rehearsal/state python -m lab package build --toolchain fake --adapter ~/runs/<run id>/adapter --run-id <run id> --task <task> --mmproj ~/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf --out /tmp/l5-rehearsal/packages
   ```
   Expected: `{"ok": true, "outcome": "ok", "version": "atlas-<task>-<run id>", ...}`, exit 0. Then `rm -rf /tmp/l5-rehearsal`.

6. Pod B: the real build (merge on CPU in bf16, about 8.4 GB of weights in memory; convert; quantize).

   ```
   LAB_TASK=L5 python -m lab package build --adapter ~/runs/<run id>/adapter --run-id <run id> --task <task> --mmproj ~/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf
   ls -l ~/data/packages/<version>
   ```
   Expected: one JSON line with `"outcome": "ok"`, `llama_cpp_commit` (step 2's commit), the three `commands`,
   `package_mb` and `wall_seconds`; exit 0. The folder holds `<version>-Q4_K_M.gguf`, `mmproj-F16.gguf` and
   `manifest.json`. The model gguf should be near the 2.74 GB of the published Q4_K_M of this base (research note
   PK-5 expects within 1%), so `package_mb` is that plus the projector's size. A failure answers
   `{"ok": false, "kind": "toolchain", "error": "<step> exited <code>: <last line>"}` and leaves no package; the ledger
   has a `package build` row with outcome `fail`. If `convert` fails with an unknown or text-only architecture, the
   merged folder from `AutoModelForCausalLM` is the suspect: report it on the task page rather than patching on the pod.

7. After the gate (L4) has scored `~/data/packages/<version>` and written its ledger row for `<run id>`:

   ```
   LAB_TASK=L5 python -m lab package scores --package ~/data/packages/<version> --run-id <run id>
   ```
   Expected: `{"ok": true, "scores": {"current": ..., "candidate": ..., "teacher": ...}, "gate_seq": N, ...}`, the
   numbers equal to the gate row's `learning.heldout.*` metrics. Before the gate has run it answers
   `"kind": "no_gate_row"`, exit 1, and changes nothing.

8. Pod B: the dry-run install, off the box.

   ```
   python -m lab package dry-run --package ~/data/packages/<version>
   ```
   Expected: `{"ok": true, "outcome": "ok", "installed": ".../data/models/<version>", "read_back": {"version": "<version>", "base": "Qwen/Qwen3.5-4B", "task": "<task>", "scores": {...}}}`, exit 0.

9. Pod B: the ledger rows.

   ```
   python -m lab ledger grep --ref run_id=<run id> --verb 'package*'
   ```
   Expected: a `package build` row (outcome ok, `learning.package_mb` in MB, `conditions.llama_cpp_commit`,
   `commands`, `wall_seconds`, node pod-b) and a `package scores` row whose `source` is `ledger seq <the gate row>`.

The box then pulls the package through R10's file server (`python -m lab.package_pull serve-start --account D` and
`python -m lab.package_pull pull --account D --version <version> --dest data/models`); that is not part of L5.
