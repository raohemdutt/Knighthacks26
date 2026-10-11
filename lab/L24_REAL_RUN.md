# L24 real run: the desk score of the current Jetson model

What the offline build could not do: score the real model on Dominick's real desk export. Everything below was
proven only against `lab/desk_fake.py` and the synthetic `fixtures/learning/desk-fixture/`. The task is done when
the current Jetson model's desk score is in the ledger twice, with the split's sha256 and the scorer commit.

Env var names only; no values belong in this file.

## 0. Before you start

- The desk export is on the box at `data/exports/<dataset_id>/` (manifest.json, examples.jsonl, images/). Check it:
  `python lab/dataset.py data/exports/<dataset_id>`.
- The label the scorer compares against is each example's `target`. Confirm that Dominick's export carries the
  object label there. If it is somewhere else (the caption, a separate file), stop and ask on Team Comms: the
  scorer does not guess.
- If the export's frame_refs are one per keyframe but several keyframes come from one episode, the split by
  frame_ref keeps near-duplicate frames of an episode on both sides. Check with
  `python -c "import json;rows=[json.loads(l) for l in open('data/exports/<dataset_id>/examples.jsonl')];print(len(rows), len({r['frame_ref'] for r in rows}), len({r['id'].rsplit('-',1)[0] for r in rows}))"`
  (items, frame_refs, episodes). If episodes are far fewer than frame_refs, raise it with the L4 owner before the
  split is written, because the split is written once.

## 1. The frozen split (once, on the box)

```
cd ~/Knighthacks26          # the repo checkout on the Jetson, on the commit that holds lab/desk_score.py
git rev-parse HEAD          # the scorer commit the rows will carry
python -m lab desk-score split data/exports/<dataset_id> --held-out 50
python lab/dataset.py data/exports/<dataset_id> --split      # re-check; prints the same sha256
```

With 136 examples, `--held-out 50` holds out 50 or a few more (whole groups). Never run `split` again on the same
export: a new split is a new baseline. If the export must also go to a pod (step 3), push it after the split, so the
pod reads the same split.json and the same sha256.

## 2. The Jetson run (the number the task asks for)

The current model is the package `data/models/current` points at; `runtime/registry.py` reads its manifest
(`data/models/current/manifest.json`, ModelManifest: version, gguf, mmproj) and `jetson/atlas-llm.sh` serves it:
`llama-server -m <pkg>/<gguf> --mmproj <pkg>/<mmproj> --jinja -c 4096 -ngl 99 -np 1 --reasoning-budget 0
--cache-ram 0 --host 127.0.0.1 --port ${ATLAS_LLM_PORT:-8080}` (LLAMA_SERVER names the binary). Score that
service as it runs, so the model is served exactly as the box serves it:

```
systemctl status atlas-llm                       # running
python3 -c "import json;print(json.load(open('data/models/current/manifest.json'))['version'])"   # the model name
export LAB_ACTOR=<who>  LAB_TASK=L24             # names only; LAB_STATE_DIR if the ledger lives elsewhere
python -m lab desk-score score data/exports/<dataset_id> \
    --base-url http://127.0.0.1:8080 --model <version from the manifest> --node jetson \
    --runs 2 --json desk-current-jetson.json
```

- `--runs 2` writes two `desk-score` rows (the real score recorded twice, for run-to-run noise); the answer's
  `noise` is their spread. Both rows carry `refs.split_sha256`, `refs.commit` and `conditions.decode`.
- Greedy decoding should give the same answer twice; a non-zero noise on the Jetson is itself a finding (llama.cpp
  batching or the vision path is not deterministic) and goes on the L4 page.
- The gateway's captions share the one llama-server slot; they slow the run but do not change an answer. If the
  run is to be quiet, stop the gateway for its length (`systemctl stop atlas-gateway`, start it after).
- The live box decodes at temperature 0.2 (`runtime/llama.py`); the scorer uses greedy (DECODE) on purpose, as L4
  requires one decode setting for every score.
- Check the rows: `python -m lab ledger tail 2` (or read `lab/state/ledger.jsonl`), then bring the ledger rows and
  `desk-current-jetson.json` back with the run.

## 3. The pod run (candidate or teacher, same scorer)

On a pod, lab/ and contracts/ are copies (the repo cannot be cloned there), so the commit must be passed.

```
# on the workstation, after step 1, from the commit the Jetson run used
python -m lab.transfer push-dataset --account <letter> data/exports/<dataset_id> --to datasets
# on the pod: llama-server built by lab/pod_b_trainer.sh llama, serving the Q4_K_M GGUF and its mmproj
~/llama.cpp/build/bin/llama-server -m <model>.Q4_K_M.gguf --mmproj <mmproj>.gguf --jinja -c 4096 -ngl 99 -np 1 \
    --reasoning-budget 0 --cache-ram 0 --host 127.0.0.1 --port 8080 &
python3 -c "import httpx, pydantic"              # the scorer's two dependencies
python3 -m lab desk-score score datasets/<dataset_id> --base-url http://127.0.0.1:8080 \
    --commit <the 40-hex scorer commit> --model <package version> --node <pod-a|pod-b> --runs 2 --json desk-pod.json
```

Bring the pod's `lab/state/ledger.jsonl` rows back with the run. A pod score is the same scores dict; only the
Jetson rows close L24.

## 4. Done when, checked

- [ ] two `desk-score` rows with `node` jetson, the same `refs.split_sha256` and `refs.commit`, the model version
- [ ] `conditions.meets_deploy_floor` true (held-out side of at least 50)
- [ ] the two scores and their noise noted on the L24 and L4 task pages
