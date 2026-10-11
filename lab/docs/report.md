# Under the hood report: `python -m lab report` (O71)

One self-contained HTML page and a JSON data pack beside it, built only from committed files, for a judge or
the big display: `lab/refine/report/under-the-hood.html` and `lab/refine/report/under-the-hood.json`. Both are
committed, so the demo opens the page from disk without running anything. Rebuild after new rows land:

```bash
python -m lab report                    # rewrites both files under lab/refine/report/
python -m lab report --out /tmp/report  # somewhere else
```

It reads `lab/refine/metrics.jsonl`, the newest `chipsheet-gfx1151-<date>.json`, a `decode-profile-gfx1151-<date>.json`
when one is committed, the public prior (`lab/public_prior.py` on `public-strix-halo-benchmarks.csv`) and any
`ledger*.jsonl` copy, all under `lab/refine/research/kernels/data/`. Charts are inline SVG written by
`lab/report.py`: no script, no font, no network. The page follows the system's light or dark setting and has
Night and Day switches. Rows from the R9700 (by `lab/slide_numbers.py`'s `_old_chip`, the rule the Learning lab's
bench reader imports) are shown in their own panel, never on a Strix Halo chart; a missing measurement says
"not measured".

Sections: (1) `ceiling`, the copy bandwidth over the largest buffer divided by the teacher's weight bytes per token
against each build's batch-1 decode rows, with the public prior beside as a prior; (2) `frontier`, joules per token
against concurrent streams for every sweep of two or more batch sizes, with the ratio between the fewest and the
most streams; (3) `step`, milliseconds per decode step by bucket from the decode profile; (4) `engines`, the best
batch-1 row per engine; (5) `finetune`, per-step loss and step time (rows with `conditions.step`); (6) `dropped`,
experiment verdicts from a committed ledger copy and the move off the old chip.

The data pack (`format: atlas.lab_report/1`), for the Learning lab screen or any other reader:

| Key | What it holds |
|---|---|
| `format`, `title`, `data_as_of` | the pack's format, the page title, the newest row time in the metrics file |
| `chip` | the current chip (`gfx1151`), the other chip, and the rule that sets its rows apart |
| `sources` | every file read: `{file, sha256, lines}` |
| `skipped` | metrics lines that are not rows: `{line, problem}` |
| `sections` | six objects in page order, each `{id, title, reading, state, other_chip, awaiting, ...}`; `state` is `measured`, `partial` or `empty`; `reading` is the sentence to say aloud |
| `numbers` | every number on the page once, by `id` |
| `awaited` | the runs that fill a section when their rows land: `{task, what, where, sections, file_present}` |

A number is `{id, label, value, unit, kind, source: {file, line, ref}, gpu, chip, conditions, ts, low_confidence,
reason, other_chip}`; `kind` is `measured`, `derived` (with `formula` and `inputs`, the ids it was worked out
from), `stated` (an estimate a committed file states), `prior` (never a measurement) or `not measured` (value
null, with `why`). `gpu` is the row's `conditions.gpu` and `chip` its `conditions.chip`; either may be null.
Section extras: `ceiling` has `inputs`, `ceiling`, `achieved`, `shares`, `priors`; `frontier` has `energy` and
`throughput` (sweeps of `{metric, node, build, session, points: [{batch, number}]}`) and `ratios`; `step` has
`batches` (`{batch, step_ms, parts}`); `engines` has `engines` (`{engine, best, rows}`), `not_measured` and
`priors`; `finetune` has `loss`, `step_time`, `numbers`; `dropped` has `items`.
