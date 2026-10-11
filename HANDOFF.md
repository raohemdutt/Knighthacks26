# Handoff: L5 packaging of the L16 adapter (paused by the owner, Sat 2026-10-10 about 11:55 PM ET)

## State

- L16 is finished: PR #310 (branch `ctx/learning/first-finetune`), adapter on pod D at
  `/home/jovyan/adapters/l16-train-20261011T032149Z-7dd098/adapter` (adapter_config.json, adapter_model.safetensors 130 MB).
- L5 has not started on pod D. Nothing of this task runs there. A `lab bench` run with a vLLM server (another session,
  R20's matrix, no started queue row) was alive on pod D at 11:42 PM ET and holds most of the memory; per the owner's
  rule it was left alone. Wait until `pgrep -af "vllm|lab bench"` on the pod prints nothing.
- Queue: L5 row `3f696ef5fec581fa9f92e96936676996` (60 min, Queued). Its critique is written (by ilan2-agent, owner's rule
  for tonight); it still needs `slot start --reply`. Not started, so nothing to release yet.
- On pod D already: `~/l5/l5-bundle.tar.gz` (lab/, contracts/, jetson/make_manifest.py at main 2a7d3ea, file upload only),
  `~/venvs/atlas-train` (L16's venv), `~/models/Qwen3.5-4B` (bf16 base at 851bf6e), `~/llama.cpp` at 23b0202.
- Found on pod D: `~/llama.cpp/build-rocm/bin` has llama-cli and llama-server but NO llama-quantize, and
  `lab/package.py` runs `<llama.cpp>/build/bin/llama-quantize`. `lab/l5_run.sh prep` builds it CPU-only into
  `~/llama.cpp/build`. The converter at 23b0202 registers `Qwen3_5ForCausalLM` (conversion/qwen.py:653), so the
  text-only merge should convert. No mmproj on pod D yet.
- Base mmproj to use: `unsloth/Qwen3.5-4B-GGUF` `mmproj-F16.gguf` at commit e87f176479d0855a907a41277aca2f8ee7a09523,
  672,423,616 bytes, sha256 cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864. Not checked against the
  box's installed manifest: the Jetson did not answer SSH at 11:45 PM ET.
- A local fake-toolchain rehearsal of `python -m lab package build` passed (version
  `atlas-tool-calling-train-20261011T032149Z-7dd098`).

## Next commands (from this clone, branch ctx/learning/first-package, LAB_ACTOR=ilan2-agent LAB_TASK=L5)

```
python -m lab.controller run-script --account D --timeout 120 <a script running: pgrep -af "vllm|lab bench">   # must print nothing
python -m lab slot start --row 3f696ef5fec581fa9f92e96936676996 --reply "..."
python -m lab.controller run-script --account D --timeout 1500 lab/l5_run.sh prep
python -m lab.controller run-script --account D --timeout 3000 lab/l5_run.sh build
python -m lab.transfer pull-file --account D packages/atlas-tool-calling-train-20261011T032149Z-7dd098/<each file> packages/<version>/<file>
python -m lab.transfer pull-file --account D l5/ledger.jsonl ../L16-data/l5-ledger.jsonl
python -m lab package dry-run --package packages/atlas-tool-calling-train-20261011T032149Z-7dd098
python -m lab slot done --row 3f696ef5fec581fa9f92e96936676996 --result "..."
python -m lab.controller run-script --account D lab/l5_run.sh cleanup <job ids>
```

## What remains

The real build (merge, convert, quantize), pulling the package to `work/L16/packages/` (not to be committed) while leaving
it in `~/packages/<version>/` on the pod, the local dry-run install, `lab/L5_RESULT.md` (sizes, sha256s, commands,
wall-clock, the ledger row), the PR "L5: first package of the L16 adapter, Q4_K_M GGUF and manifest", releasing the slot.
Remove `lab/l5_run.sh` and this file from the branch or keep l5_run.sh as the run script, as the PR author prefers.
