# Packaging an adapter for the Jetson: `python -m lab package` (L5)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab package -h`.

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
