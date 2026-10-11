# Placement plan: `python -m lab placement` (R46)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab placement -h`.

Which pod serves the teacher, and where the power sampler, benchmark runs, Ax trials and LoRA jobs go, with a reason
per line, which jobs may share a chip under what memory margin, and every number traced to its file (the `[n]` in
the text resolve in its Numbers list). The rules and why: `lab/refine/research/plumbing/placement.md`.

```
python -m lab placement                                    # the plan as text (--json for one JSON object)
python -m lab placement --check "L3: smoke LoRA" --gpu A   # does a slot request contradict it? exit 1 when it does
python -m lab placement --teacher-share 0.6                # the plan if the teacher were served at a smaller share
```

- Inputs: the pods' facts folder (`--facts`, else `$LAB_PLACEMENT_FACTS`, else `lab/state/placement` when it holds a
  `pods.json`, else the recorded `fixtures/lab/placement`): the `python -m lab.controller status --probe-port 8000`
  answer, the output of `bash lab/pod_a_teacher.sh facts` per pod and the pod sampler's sample file. One chip sheet per
  pod from `lab/refine/research/kernels/data/` (the newest `chipsheet-<gfx>-<date>.json`, else
  `chipsheet-sample.json`, which documents the six fields read). The GPU Queue through the slot client (`NOTION_TOKEN`;
  `--queue-file` or `--no-queue` otherwise). Nothing is called on a pod or the hub.
- Today, from the recorded facts: the teacher stays on pod-a (it serves there, 57.6 GiB held at the serve script's
  0.90 share of 64 GiB); benchmarks, Ax trials and LoRA jobs go to pod-b alone; the sampler runs on both. The teacher
  and a LoRA job fit one pod on paper (35 GiB with the weights) but not at the 0.90 share (74.65 GiB against 58 GiB
  free); at a share of 0.63 or less they would, which is R43's interference measurement to settle.
- `python -m lab slot request` asks the same check before it adds a row and refuses a request whose kind (from its
  words, or `--kind`) the plan puts on the other card, outcome `placement_contradiction`, unless `--override "why"` is
  given; the reason goes into the row's Progress note as an Override line. A plan that cannot be built never blocks a
  request.
