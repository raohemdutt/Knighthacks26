# Publishing the adapter: `python -m lab publish-adapter` (L8)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab publish-adapter -h`.

Puts the trained LoRA adapter and its model card on the team Hugging Face repo. The card is filled by code from the
evaluation gate's ledger row (`lab/model_card_template.md`): the task and its instruction from a `TaskSpec` JSON file,
the current, candidate and teacher scores, n as items and as groups, the run-to-run noise and the paired p-value, none
typed by hand. The metadata sets `license: apache-2.0`, `base_model: Qwen/Qwen3.5-4B`, `base_model_relation: adapter`,
`library_name: peft` and no `datasets` entry; the upload carries `LICENSE` (the Apache 2.0 text) and a line saying the
adapter is a LoRA fine-tune of the base.

```
python -m lab publish-adapter --fake --repo <owner>/<name> --adapter-dir <dir> --task-spec <spec.json> [--run-id R]
```

`--fake` runs every step against `lab/hf_fake.py` and uploads nothing; without it the command needs `HF_TOKEN` in the
environment (the name only; the value is never printed) and `huggingface_hub`, which is imported only then. Before any
push it reads the base model's card again and stops unless the license is `apache-2.0`, and scans every file of the
upload folder for an image, a caption, a label, a `frame_ref`, a `data/` path or a camera id, stopping on any hit. The
exact command for the real publish is in `lab/L8_REAL_RUN.md`.
