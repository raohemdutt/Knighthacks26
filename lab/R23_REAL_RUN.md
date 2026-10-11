# R23 real run: the Quark INT4 checkpoint on pod B

What this branch could not do offline, and the exact commands to do it. Everything below the "Offline" section
needs the SJC pods, the hub, the GPU Queue in Notion or the checkpoint's weights; none of it has run yet. Run the
commands from the repo root on the box unless a step says "on pod B".

## Offline (done on the branch)

- The loader hunks are appended to `lab/vllm-gfx1201.diff` (Python only; the seven build-enabling hunks are
  unchanged). `git apply --check lab/vllm-gfx1201.diff` passes on a clean vLLM checkout at b1388b1 (the v0.19.1 tag;
  the pods call this build 0.19.2 through `SETUPTOOLS_SCM_PRETEND_VERSION`), and every patched file compiles.
  To repeat it: `VLLM_B1388B1=<clean checkout at b1388b1> python -m pytest -q tests/test_lab_quark_compare.py -k b1388b1`.
- `lab/quark_keys.py` and `tests/test_lab_quark_keys.py`: every key of the checkpoint's config (fixture
  `fixtures/kernels/quark_config_august_2026.json`, byte for byte its config.json at 0f7ee2559e8d) has a mapping,
  the old mapper raises on it, the patched one does not, and the scheme's parameter shapes match the safetensors
  header.
- `lab/quark_compare.py` and `tests/test_lab_quark_compare.py`: the slot (fake queue), the serve line, the bench
  matrix and the correctness pass (fake server and pod), the verdict, the ledger row, T0's E5 result, the T0 text,
  the release and the pod-as-found check.

## Environment variables (names only; values are on the Notion Credentials page and in the box's `.env`)

`AUP_HUB_URL_C`, `AUP_HUB_USER_C`, `AUP_HUB_TOKEN_C`, `AUP_HUB_PASS_C`, `AUP_HUB_LAUNCH_C`,
`AUP_HUB_URL_D`, `AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D`, `AUP_HUB_PASS_D`, `AUP_HUB_LAUNCH_D`,
`LAB_POD_ROLES` (`C:pod-a,D:pod-b`), `NOTION_TOKEN` (the GPU Queue; box only), `LAB_ACTOR`, `LAB_TIER`,
`LAB_TASK` (`R23`). Nothing on this list goes on pod B: the slot verbs run on the box.

## 1. Before the slot (no GPU needed)

1. Pod B up and ready: `python -m lab.controller status --account D`.
2. Pod B's vLLM is the stock build: `python -m lab.controller run-script --account D lab/pod_a_teacher.sh verify`
   (commit b1388b1, no rejected hunks). The runner puts the loader hunks on itself and takes them off after; it
   refuses a tree where they neither apply nor are already in.
3. The checkpoint is on pod A's home volume. Check it is whole there, then copy it pod A to box to pod B, file by
   file, through the hub (never `hf download`; R23's page says copy, do not re-download). The weights' sha256 is
   `f1699476b8c79a9a3121469b8bdfd84c0ee55417d2af02260b0d8982257c6346` (the hub's LFS id). On SJC the Jupyter
   root is `/home/jovyan`, so `models/...` below is `~/models/...` on both pods.

   ```
   M=models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16
   python -m lab.transfer pull-file --account C $M/model.safetensors data/r23/model.safetensors --part-mib 64 \
       --sha256 f1699476b8c79a9a3121469b8bdfd84c0ee55417d2af02260b0d8982257c6346
   for f in .gitattributes LICENSE README.md chat_template.jinja config.json crc32.txt generation_config.json \
            merges.txt preprocessor_config.json processor_config.json tokenizer.json tokenizer_config.json \
            video_preprocessor_config.json vocab.json; do
     python -m lab.transfer pull-file --account C "$M/$f" "data/r23/$f"
   done
   for f in data/r23/* data/r23/.gitattributes; do
     python -m lab.transfer push-file --account D "$f" "$M/$(basename "$f")" --chunk-mib 64
   done
   ```

   19.5 GB each way in 64 MiB parts is about 300 requests each way; the link's speed is not measured yet (T0 E9).
   If the two pods turn out to share a volume, skip the copy. `data/` is git-ignored on the box; delete
   `data/r23/` after the push.
4. Put the branch's files on pod B (the private repo cannot be cloned there). From the box:

   ```
   git archive --format=tar.gz -o /tmp/r23.tar.gz HEAD lab fixtures/lab/bench fixtures/kernels
   python -m lab.transfer push-file --account D /tmp/r23.tar.gz r23/r23.tar.gz
   ```

   and a scratch script outside the repo, `/tmp/r23_pod.sh`, that unpacks it and runs one sub-verb:

   ```
   #!/bin/bash
   set -eu
   mkdir -p ~/r23/repo && tar -xzf ~/r23/r23.tar.gz -C ~/r23/repo && cd ~/r23/repo
   python3 -c "import httpx" 2>/dev/null || pip install --user --quiet httpx
   python3 -m lab quark-compare "$@"
   ```

5. Pre-flight on pod B (reads the copied checkpoint's sizes and config, prints the time plan; changes nothing):
   `python -m lab.controller run-script --account D /tmp/r23_pod.sh preflight`
   Expect `"checkpoint": {"ok": true, ...}`, scheme `QuarkW4A16Int4`, and a plan of about 64 minutes for batches
   1 and 4 (`needs_extension: true`): the time model is rough, from the README's measured start-up and decode.
6. Arm A's numbers: R20's bench rows of RedHatAI/Qwen3.8-27B-INT4 on pod B (a `rows.jsonl` under
   `lab/state/bench/<session>/` on pod B), pulled to the box:
   `python -m lab.transfer pull-file --account D r23/repo/lab/state/bench/<R20 session>/rows.jsonl data/r23/control_rows.jsonl`
   (or wherever R20 left them). They must use the same prompt set, max_tokens and engine commit; record flags any
   difference (`same_conditions`). Reference outputs: R21's `fixtures/kernels/baseline_outputs.jsonl` when it
   exists (rows `{"prompt_id", "text", "spawn"}`, two spawns give the self-match floor); without it the run serves
   arm A once more, eager, for them (the plan grows to about 71 minutes).

## 2. The slot (60 minutes on pod B, one 15-minute extension)

```
python -m lab quark-compare slot --agent <the queue Agent of the person who runs it>
# a 60-minute row waits for the referee's critique; after it:
python -m lab quark-compare slot --row <row> --reply "<answer to the critique>" --agent <same>
```

`outcome: started` means the card is yours. Then, on pod B through the controller (90 minutes of timeout covers
the slot and its extension):

```
python -m lab.controller run-script --account D --timeout 5400 /tmp/r23_pod.sh run --batches 1,4
```

When the answer names the session (`quark-<stamp>`), and before minute 60 if the bench part is done:
`python -m lab slot extend --row <row> --note "R23 bench done: <n> ok rows in <session>; correctness pass next"`
(`all` does this itself when it runs in one process; the split run cannot, since pod B has no Notion token).

The matrix's batch 8 and 16 take a second slot the same way, with `run --batches 8,16` (about 66 minutes by the
time model); `--with-64` adds the 64 arm when it fits in memory.

## 3. After the run (on the box)

```
python -m lab.transfer pull-file --account D r23/repo/lab/state/quark_compare/<session>/result.json data/r23/result-1.json
python -m lab quark-compare record --result data/r23/result-1.json [--result data/r23/result-2.json] \
    --control-rows data/r23/control_rows.jsonl [--reference fixtures/kernels/baseline_outputs.jsonl] \
    --out data/r23 --row <row> --agent <same>
```

`record` writes the `quark-compare` ledger row (task R23, node pod-b), T0's E5 result with arms A and B (E5 stays
running until arm C, the FP8 checkpoint, is read), `data/r23/verdict.json` and `data/r23/R23_T0_result.txt`, and
releases the slot with a one-line result. Post `R23_T0_result.txt` on T0 under E5 (the dispatcher posts it; an
agent does not write to a live Notion page from this branch). Also pull bench's own rows if the box's ledger
should hold them: `r23/repo/lab/state/ledger.jsonl` on pod B has the `bench.run` rows of the session.

## 4. What only the pod can show (open until the real run)

- That the backported loader serves the checkpoint at b1388b1: the serve log names `ConchLinearKernel` for the
  int4 layers (the same kernel as the RedHat checkpoint) and no "No quark compatible scheme" error. Watch the
  Gated DeltaNet `in_proj_ba` layer: it is int4 here (48 outputs per half, 6 packed int32 columns) where RedHat
  keeps it bf16, and no offline test exercised Conch on an output that narrow.
- That the nibble order matches. The `reorder` layout is taken from upstream PR #48606, not proven here; a wrong
  order shows as garbage text in the correctness pass, not as a load error. The xor with 8 (signed int4 to
  uint4b8) is backed by the checkpoint's own bytes: k_proj's nibbles peak at 0, 1 and 15, and every stored zero
  point is 0.
- Decode tok/s, joules per token and the exact-match rate themselves, and whether the plan fits the slot (the time
  model's 64 minutes for batches 1 and 4 is an estimate).
- That pod B is left as found on the real tree: the same `git status` and `git diff` in `~/vllm`, nothing on port
  8100 (`pod_as_found` in result.json), and the card released (`slot status --row <row>` shows Done).
- `lab/pod_a_teacher.sh` and `lab/README.md` (Checkpoints paragraph under "Teacher (pod A)") still say the Quark
  export fails at b1388b1; that stays true for a tree without the new hunks. Update them after the real run.
