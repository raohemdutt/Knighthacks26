# Desk-object scorer: `python -m lab desk-score` (L24)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab desk-score -h`.

Scores a model on the held-out desk-object examples of an export, served as the box serves it, in the scores shape
the eval gate (L4) reads. The module docstring of `lab/desk_score.py` holds every rule; the short version:

```
python -m lab desk-score split data/exports/<dataset_id> --held-out 50     # once: split.json by frame_ref group
python -m lab desk-score score data/exports/<dataset_id> --base-url http://127.0.0.1:8080 --runs 2 \
    --json desk-current.json --node jetson
```

- **The split.** `split` writes `split.json` once, in the format `lab/dataset.py load_split` reads: whole frame_ref
  groups in the order of the sha256 of their frame_ref, held out until the side holds at least `--held-out`
  examples. It refuses a folder that has one already and records the file's sha256 (ledger verb `desk-split`).
  `score` reads the held-out side only from that file, through `load_split`, so a leaky or short split never scores.
- **The request.** One chat request per held-out example: the picture as a data URL and the export's instruction as
  the one user message, with one decode setting (`DECODE`: temperature 0, top_k 1, seed 0, max_tokens 24,
  `enable_thinking` false). The caption is not sent.
- **The score.** Answer and label (the example's `target`) normalised the same way (lowercase, punctuation and the
  articles a, an, the dropped, the last word made singular), then exact match. The answer is
  `{"score", "model", "results", "desk.<object>", ..., "desk.n_items", "desk.n_groups"}`: the shape
  `contexts/agent/eval.py --json` writes, with per-object detail as dotted keys; every result keeps its frame_ref so
  the gate pairs by group. `score(base_url, examples, prompt=...)` is the call the gate makes.
- **The row.** One `desk-score` ledger row per run (task L24): metrics `desk.score`, `desk.<object>`,
  `desk.n_items`, `desk.n_groups`; refs `split_sha256`, `commit` (git HEAD, or `--commit` where lab/ is a copy),
  `model`, `run_id`; conditions with the decode setting, the prompt's sha256 and `meets_deploy_floor`.

Tests: `tests/test_lab_desk_score.py`, on `lab/desk_fake.py` (a fake llama-server over an httpx transport) and
`fixtures/learning/desk-fixture/` (12 synthetic examples on 8 frame_refs; `python -m lab.desk_fake build-fixture`
rebuilds it). The real runs on the Jetson and on a pod are in `lab/L24_REAL_RUN.md`.
