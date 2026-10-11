# `python -m lab group-finetune`

Group fine-tune (L10): pool a Circle's opt-in exports, label, train one LoRA, gate per member, offer to all.

```
python -m lab group-finetune --circle C --task T [--pool ~/pool] --run-id ID --teacher-url URL \
    [--teacher-token-env AUP_HUB_TOKEN_C] [--model DIR --mmproj FILE --llama-cpp DIR --trainer-python PY] \
    [--epochs --lr --rank --alpha --dropout --batch --grad-accum --seed --max-steps --base-revision] \
    [--min-held-out 50] [--decode-attested] [--keep] [--dry-run]
python -m lab group-finetune pool-add --circle C --task T --member M --export data/exports/<dataset_id>
python -m lab group-finetune take --offer group_offer.json --member M [--account D] [--rollback]
```

The pool is `~/pool/<circle>/<task>/pool.json` (the Circle roster: every member, uploader or not, and the model each
box serves now) beside one folder per member holding that member's approved exports. The run merges them into one
dataset, freezes a per-member split, labels it with the teacher through `lab/label.py`, trains one LoRA with
`lab/train.py`, packages it with `lab/package.py`, runs `lab/eval_gate.py` once per member against that member's
current model, writes `group_offer.json` into the package folder, records `group-finetune.run`,
`group-finetune.delete` and metrics rows, and deletes the pooled data on every way out (`--keep` off a pod only).
`--dry-run` prints the plan and writes nothing. `take` pulls, swaps and on failure rolls back one box.

Reference: the docstring at the top of `lab/group_finetune.py`, the "Group fine-tune (L10)" section of
`lab/README.md`, and the pod script `lab/pod_b_group_finetune.sh`. Tests: `tests/test_lab_group_finetune.py`, on the
fakes in `lab/group_finetune_fake.py` and the fixture pool `fixtures/learning/group-pool/`.
