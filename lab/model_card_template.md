---
license: apache-2.0
base_model: $base
base_model_relation: adapter
library_name: peft
---

# $repo_name

This adapter is a LoRA fine-tune of $base.

## Task

Task: `$task`

Instruction given to the model:

> $instruction

The adapter was trained so that the small model answers this task the way a larger teacher model does.
It is meant for one household's own table, served next to $base on a small edge box.

## Held-out scores

All numbers below come from the evaluation gate's ledger row (run `$run_id`, ledger row $seq), scored on the same
held-out set with the same settings. Score is the fraction of held-out questions answered correctly.

| Model | Score |
| --- | --- |
| Current model (before) | $current |
| This adapter (candidate) | $candidate |
| Teacher | $teacher |

- Sample size: n = $n_items items in $n_groups groups. Items in one group are not independent.
- Measured run-to-run noise (same model, same set, scored repeatedly): $noise.
- Paired p-value, candidate against current: $p_value.

## License and changes

The adapter is released under the Apache License 2.0; the license text is in `LICENSE`. The base model $base is
also Apache 2.0 and is not redistributed here. The only change to the base is this LoRA fine-tune.

## Training data

No training data is published with this adapter, and none of it is described beyond the task above.
