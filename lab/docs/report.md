# Under the hood report: `python -m lab report` (O71)

One self-contained HTML page and a JSON data pack beside it, built only from committed files, for a judge or
the big display: `lab/refine/report/under-the-hood.html` and `lab/refine/report/under-the-hood.json`. Both are
committed, so the demo opens the page from disk without running anything. Rebuild after new rows land:

```bash
python -m lab report                    # rewrites both files under lab/refine/report/
python -m lab report --out /tmp/report  # somewhere else
python -m lab report --check            # writes nothing; exit 1 when the files there are stale
```

The committed page goes stale whenever a file it reads changes (a merge that adds rows to
`lab/refine/metrics.jsonl`, say): `--check` answers `{"ok": false, "outcome": "stale", "stale": [...]}` then, and
running the verb and committing the two files brings it back. Every answer lists `still_coming`, the awaited runs
whose rows are not committed yet.

## What it reads

- `lab/refine/metrics.jsonl`, every row with its line number. A line that repeats an earlier row (the same metric,
  node, value, unit and time, and no condition both rows record differs) is left out and listed under `skipped`, so
  a merge that kept both copies of a rewritten row never counts a measurement twice;
- under `lab/refine/research/kernels/data/`: the newest `chipsheet-gfx1151-<date>.json`, a
  `decode-profile-gfx1151-<date>.json` when one is committed, the public prior (`lab/public_prior.py` fitted on
  `public-strix-halo-benchmarks.csv`), any `ledger*.jsonl` copy, the `excluded` list of any `searchspace-*.json`,
  and the name of a `sweep-request-gfx1151-<date>.json`;
- a fine-tune run's own files, `fixtures/demo/<run>-telemetry.jsonl` (one `telemetry.train.step` event per line:
  step, loss, step_ms) and `fixtures/demo/<run>-run.json` beside it (the device and the run's totals); L16's are
  `l16-telemetry.jsonl` and `l16-run.json`. The newest run on the lab's chip is the one charted;
- the newest `lab/refine/research/telemetry/demo_numbers_<date>.md`, named in the footer as the page's companion.

Nothing else: no pod, no hub, no network, no Notion.

## The page

Charts are inline SVG written by `lab/report.py`: no script, no font, no network reference. The page follows the
system's light or dark setting and has Night and Day switches. Each section leads with one sentence to say aloud
and one key figure, then its chart with the chip and the source lines under it; the numbers behind a chart fold
into a list, each with its file, line, `conditions.gpu` and flags. On a phone the parts stack; on a wide screen the
chart sits beside the key figure. Rows from any other chip (the R9700 by `lab/slide_numbers.py`'s `_old_chip`, the
rule the Learning lab's bench reader imports; the A100 reference; any GPU a row names) sit in a dashed panel of
their own, with their own chart when they are a sweep, never on a Strix Halo chart. A missing measurement says
"not measured", and the section names the runs still coming (`AWAITED` in `lab/report.py`, each with what it was
doing on `STILL_COMING_AS_OF`); a run drops off that list by itself once its file or rows are committed.

Sections:

1. `ceiling`: the chip sheet's copy bandwidth over its largest buffer divided by the teacher's weight bytes per
   token, against the best batch-1 decode row of each build. A build is named from its row: engine, commit and
   patch (`vLLM b1388b1 + gfx1201 patch`), or `build not recorded` when the row says so. The public prior sits
   beside, in gray, as a prior.
2. `frontier`: joules per token against concurrent streams. The headline sweep is the one with the most stream
   counts (then the widest span, then the first measured), so it does not change when a second pod repeats it; a
   second sweep is drawn with it and said in a note. Tokens per second over all streams is a chart of its own.
3. `step`: milliseconds of one decode step by bucket, from the decode profile.
4. `engines`: one bar per engine, its best batch-1 row on the prompt the engines share when they share one; the
   page says where the rows still differ (prompts, tokens out, thinking, weights). `llamacpp` and `llama-server`
   are read as llama.cpp. A row that names no engine stands for none.
5. `finetune`: loss and step time for each logged step of the newest run, with the run's totals.
6. `dropped`: experiment verdicts from a committed ledger copy, the settings a search space excludes (ruled out
   before any run, each with its reason and line), and the move off the old chip.

## The data pack

`format: atlas.lab_report/1`, for the Learning lab screen or any other reader. Top-level keys:

| Key | What it holds |
|---|---|
| `format`, `title`, `data_as_of` | the pack's format, the page title, the newest row time in the metrics file |
| `chip` | the current chip (`gfx1151`), the other chips, and the rule that sets their rows apart |
| `sources` | every file read: `{file, role, sha256, lines}`; `role` says what it was used for |
| `skipped` | metrics lines left out: `{line, problem}`, with `repeats` (the earlier line) when it repeats a row |
| `sections` | six objects in page order, each `{id, title, reading, state, other_chip, awaiting, ...}`; `state` is `measured`, `partial` or `empty`; `reading` is the sentence to say aloud |
| `numbers` | every number on the page once, by `id` |
| `awaited` | the runs that fill a section when their rows land: `{task, what, status, as_of, where, sections, landed, evidence, file_present}` |

A number is `{id, label, value, unit, kind, source: {file, line, ref}, gpu, chip, chip_label, conditions, ts,
low_confidence, reason, other_chip}`; `kind` is `measured`, `derived` (with `formula` and `inputs`, the ids it was
worked out from), `stated` (an estimate a committed file states), `prior` (never a measurement) or `not measured`
(value null, with `why`). `gpu` is the row's `conditions.gpu` (for a fine-tune step, the device its run file
names) and `chip` its `conditions.chip`; either may be null. `chip_label` is the chip the report filed it under.

Section extras:

- `ceiling`: `inputs` (bandwidth, bytes per token), `ceiling`, `achieved` (every batch-1 row), `shares`, `builds`
  (`{build, engine, best, rows, share}`), `priors`, `notes`.
- `frontier`: `energy`, `throughput` and `power` (sweeps of `{metric, node, build, session, label, chip_label,
  points: [{batch, number}]}`), `ratios` (joules per token, fewest over most streams, one per sweep),
  `more_work` (tokens per second, most over fewest), `changes` (a settings sweep's change rows), `notes`,
  `other_ratios` and `pending_files`.
- `step`: `batches` (`{batch, step_ms, parts}`), `top` (the biggest slice's share), `file`.
- `engines`: `engines` (`{engine, name, best, rows, builds, setup}`), `compared_on` (the shared prompt, or null),
  `differences`, `unnamed`, `not_measured`, `priors`.
- `finetune`: `run` (`{name, run_id, node, gpu, file, run_file}`), `loss`, `step_time` (milliseconds, as
  logged), `median_step` (seconds), `numbers` (the run's totals and any trainer rows), `notes` (the steps that
  handled far fewer tokens than the others, when the telemetry records tokens).
- `dropped`: `items` (`{what, verdict, note, kind, source, gpu, task}`; `kind` is `measured`, `ruled out` or
  `record`).

Tests: `tests/test_lab_report.py` on two small trees, `fixtures/lab/report/` (the first rows) and
`fixtures/lab/report_landed/` (every awaited run landed), an empty folder, and this repository.
