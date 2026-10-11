# `python -m lab group-finetune`

Group fine-tune (L10): pool a Circle's opt-in exports, label, train one LoRA, gate per member, offer to all.

```
python -m lab group-finetune --circle C --task T [--pool ~/pool] --run-id ID --teacher-url URL \
    [--teacher-token-env AUP_HUB_TOKEN_C] [--model DIR --mmproj FILE --llama-cpp DIR --trainer-python PY] \
    [--epochs --lr --rank --alpha --dropout --batch --grad-accum --seed --max-steps --base-revision] \
    [--min-held-out 50] [--decode-attested] [--gate real|fake] [--teacher-expect-model NAME] [--keep] [--dry-run]
python -m lab group-finetune pool-add --circle C --task T --member M --export data/exports/<dataset_id>
python -m lab group-finetune take --offer group_offer.json --member M [--account D] [--rollback] [--dry-run]
```

The pool is `~/pool/<circle>/<task>/pool.json` (the Circle roster: every member, uploader or not, and the model each
box serves now) beside one folder per member holding that member's approved exports. The run merges them into one
dataset, freezes a per-member split, labels it with the teacher through `lab/label.py`, trains one LoRA with
`lab/train.py`, packages it with `lab/package.py`, runs `lab/eval_gate.py` once per member against that member's
current model, writes `group_offer.json` into the package folder, records `group-finetune.run`,
`group-finetune.delete` and metrics rows, and deletes the pooled data on every way out (`--keep` off a pod only).
`--dry-run` prints the plan and writes nothing. `take` pulls, swaps and on failure rolls back one box; `take
--dry-run` checks the member against the offer and prints the commands it would run, running none.

For a smoke run, `--gate fake` scores with a stand-in gate (no model servers, canned scores): the ledger row and
the offer say `gate: fake` and no metrics row is written, so it proves the plumbing and never the model.
`--teacher-expect-model` names what the teacher's served checkpoint must hold (default the label verb's; empty
accepts any). The pod script maps these to `GATE`, `TEACHER_EXPECT`, `MIN_HELD_OUT` and `L10_STEPS` (the
trainer's `--max-steps`), and finds llama.cpp through `LLAMA_SRC` and `LLAMA_BUILD`.

Reference: the docstring at the top of `lab/group_finetune.py`, the "Group fine-tune (L10)" section of
`lab/README.md`, and the pod script `lab/pod_b_group_finetune.sh`. Tests: `tests/test_lab_group_finetune.py`, on the
fakes in `lab/group_finetune_fake.py` and the fixture pool `fixtures/learning/group-pool/`.
