# Research note format and citation rules

Every research note in `lab/refine/research/<track>/` follows this file. That covers the notes the track research tasks write and the bounded research step that opens each refinement pass. The rule behind it is the refinement loop's own: no citation, no change; nothing from memory. A claim goes into a note only with a source the author opened, and its label says how much that source can carry.

`python -m lab research-check <path>` checks a note, a `SOURCES.md`, or a whole track folder against this file, from the repo root. Like every lab verb it answers with one JSON line: `ok`, `outcome` (`clean` or `violations`), `count`, `violations` (one string per violation), `notes` and `findings`. The exit code is 0 when clean and 1 otherwise. Section 10 lists what it checks and what it cannot.

## 1. Where notes live

```text
lab/refine/research/<track>/<topic>.md    a note: one topic, lowercase kebab-case file name
lab/refine/research/<track>/SOURCES.md    the index of every source the track's notes cite
```

A file named `measurements*.md` in the folder is a record, not a note: raw probe results that notes cite by section (`measurements-2026-10-10.md#m1-front-door`). The checker skips it, and an index entry points at it like any in-repo source.

The tracks are `telemetry`, `learning`, `plumbing` and `kernels`. A pass adds findings to an existing note under a new heading, or starts a new note for a new topic. It never renames or deletes a heading that a log entry, a ruling or a queued experiment cites, because the heading is the address (section 9).

## 2. Note layout

```markdown
# <The question the note answers>

<Two or three sentences of prose: what the note covers and what it leaves out.>

## <Topic>

- **[label]** <The claim, with its number and setup.> <Citation>
  - Implication: <what it means for our metric set>
  - Candidate experiment: <what to run>. Expected movement: <metric, direction, size>. Minimum sample size: <n>.

## <Another topic>

...

## Open questions

- <A lead the search bound stopped before it was followed>

## Search log

- <date, budget, time used, queries and starting points, why the search stopped>
```

- One `#` title, then a short prose opening. Prose frames the note. A claim about the world goes in a finding.
- One `##` heading per topic. Each `##` heading is unique in its note, because a pass cites a finding by that heading (section 9). `###` headings may group findings inside a topic.
- Findings are bullets, one per finding (section 3). Every top-level bullet under a topic heading is a finding.
- Bullets before the first `##` heading are not findings. Neither are bullets under a framing heading: Scope, Summary, Overview, Background, Method, Search log, Queries, Sources, References, Open questions, Gaps, Limitations, Caveats, Next steps, Takeaways, Inferences, Conclusions, and Candidate experiments (ranked). A `###` heading under one of these inherits the exemption. A heading that only resembles one, such as "Sources of error", is a topic.

A complete note, short enough to read at a glance (the two findings stand for the many a real note holds):

```markdown research-note example=complete
# Decode energy on one Strix Halo: what is measured and what is documented

Joules per token for decode on the AMD pods. Training energy and the Jetson are out of scope.

## Baseline decode energy

- **[measured]** Decode ran at 2.88 tok/s at about 120 W average and 129 W peak (the whole APU socket) on one Strix Halo (Radeon 8060S, gfx1151, pod A on the SJC hub) serving RedHatAI/Qwen3.8-27B-INT4 with vLLM and its Conch Triton fallback kernel for W4A16, which is about 42 J per token (120 W / 2.88 tok/s, derived). [PR 194](https://github.com/raohemdutt/Knighthacks26/pull/194)
  - Implication: joules per token follows decode speed at near-fixed power, so a faster W4A16 kernel lowers it almost one for one. This is the baseline every kernel change is judged against; the R9700's 7.7 tok/s at about 200 W (PR 37) is the other chip's number, not this baseline.
  - Candidate experiment: rerun the PR 194 text prompt after each kernel change, sampling power over the request window only. Expected movement: joules per token falls in proportion to the tok/s gain (2.88 to 5.8 tok/s at the same 120 W is about 21 J per token). Minimum sample size: 5 runs of 256 generated tokens.

## Sampler trust

- **[practitioner]** Zeus's documentation says the AMD energy counter can return invalid values on older GPU models and ROCm versions, so at start-up Zeus reads it twice, 0.5 s apart, and compares the difference with power measurements before it trusts the counter. It also says the instantaneous power field can read N/A, in which case average power must be used. [Zeus docs, Measuring Energy, Hardware Support > AMD GPU > Initialization](https://ml.energy/zeus/measure/)
  - Implication: our sampler must cross-check the energy counter against integrated power before it reports joules, and fall back to average power when the instantaneous field is N/A. The metric set gains "counter agrees with integrated power" as a pass or fail per node.
  - Candidate experiment: on pod B, log the energy counter and 1 Hz power together for 60 s idle and for 60 s under the smoke LoRA load, then compare. Expected movement: none in joules per token; the counter stays a source only if it agrees with integrated power within 5 percent. Minimum sample size: 3 windows of 60 s per state.

## Open questions

- Does the pod's `amd-smi` report `current_socket_power`, or N/A?

## Search log

- Example entry: 2026-10-10, budget 2 h, used 40 min, stopped when the last three sources repeated the Zeus docs.
```

## 3. A finding

```markdown
- **[measured]** The claim in one or two sentences, with the number and the setup. A citation.
  - Implication: what it means for our metric set.
  - Candidate experiment: what to run. Expected movement: metric, direction, size. Minimum sample size: n.
```

1. One finding per bullet. Two claims are two bullets.
2. The label comes first and there is exactly one (section 4). Write it as `**[measured]**`. The checker also accepts `[measured]`, `(measured)`, `**measured**` and `measured:`, and a short qualifier after a comma inside the brackets, as in `**[measured, n=3]**`. A label word that is merely part of the sentence ("Measured power draw was...") is not a label, and the label does not go after the claim.
3. The citation is on the bullet itself, at the end of its first paragraph, or on a `Source:` line under it (section 5). A link that appears only in the Implication or Candidate experiment line does not cite the finding.
4. The implication and the candidate experiment sit under the finding (section 6): as nested bullets, as sibling bullets that directly follow it and start with `Implication:` or `Candidate experiment:`, or on the same line.
5. The claim says what the source says. Our inference goes in the Implication line, so a reader can tell the two apart.

## 4. The four labels

The labels run from strongest evidence to weakest. An `anecdote` or an `opinion` is a lead to test, not a result: under the loop's source-forced rule, an idea that is not yet tested becomes a queued experiment with its citation and expected movement.

If a source mixes kinds, split it into one finding per kind. If you are unsure between two labels, take the weaker. A blog post is an `anecdote` unless it shows data. A number whose setup is missing is not `measured`.

The phone project's extra labels (`own side`, `secondary`, `UNTRACEABLE`) are not used here. A source speaking for its own method is a `practitioner` when it documents what it does and an `opinion` when it argues. A figure that cannot be traced to a primary source is an `anecdote`, and its finding says so.

### measured

A number someone measured, with the setup that produced it: hardware, model, engine, batch or concurrency, prompt, measurement window. Our own runs count, and cite the PR, ledger line or file they are recorded in.

```markdown research-note label=measured
- **[measured]** Decode ran at 2.88 tok/s at about 120 W average and 129 W peak (the whole APU socket) on one Strix Halo (Radeon 8060S, gfx1151, pod A on the SJC hub) serving RedHatAI/Qwen3.8-27B-INT4 with vLLM and its Conch Triton fallback kernel for W4A16, which is about 42 J per token (120 W / 2.88 tok/s, derived). [PR 194](https://github.com/raohemdutt/Knighthacks26/pull/194)
  - Implication: joules per token follows decode speed at near-fixed power, so a faster W4A16 kernel lowers it almost one for one. This is the baseline every kernel change is judged against; the R9700's 7.7 tok/s at about 200 W (PR 37) is the other chip's number, not this baseline.
  - Candidate experiment: rerun the PR 194 text prompt after each kernel change, sampling power over the request window only. Expected movement: joules per token falls in proportion to the tok/s gain (2.88 to 5.8 tok/s at the same 120 W is about 21 J per token). Minimum sample size: 5 runs of 256 generated tokens.
```

### practitioner

A documented recommendation or procedure from someone who runs the thing in practice: a maintainer's documentation, a vendor guide, a write-up of a working setup. It has specifics but no dataset of its own.

```markdown research-note label=practitioner
- **[practitioner]** Zeus's documentation says the AMD energy counter can return invalid values on older GPU models and ROCm versions, so at start-up Zeus reads it twice, 0.5 s apart, and compares the difference with power measurements before it trusts the counter. It also says the instantaneous power field can read N/A, in which case average power must be used. [Zeus docs, Measuring Energy, Hardware Support > AMD GPU > Initialization](https://ml.energy/zeus/measure/)
  - Implication: our sampler must cross-check the energy counter against integrated power before it reports joules, and fall back to average power when the instantaneous field is N/A. The metric set gains "counter agrees with integrated power" as a pass or fail per node.
  - Candidate experiment: on pod B, log the energy counter and 1 Hz power together for 60 s idle and for 60 s under the smoke LoRA load, then compare. Expected movement: none in joules per token; the counter stays a source only if it agrees with integrated power within 5 percent. Minimum sample size: 3 windows of 60 s per state.
```

### anecdote

A single report: one user, one deployment, one forum or issue comment, or a blog post that shows no data. Say in the finding what the report lacks.

```markdown research-note label=anecdote
- **[anecdote]** One operator (TUL Corporation) reports an 8-card R9700 deployment with tensor parallelism 8 that completed 8,681 requests in five days with zero reported errors, and says its raw sweep logs are gone, so it does not offer its throughput figures as reproducible. [ROCm/vllm#1229, comment of 2026-09-24](https://github.com/ROCm/vllm/issues/1229#issuecomment-5809360213)
  - Implication: none for the metric set. The report gives no reproducible throughput or energy figure; it only shows that an 8-card deployment stayed up for five days.
  - Candidate experiment: ask the operator on the issue for the raw sweep logs and the serve flags. Expected movement: none until logs exist; with logs, a second data point for tokens per second per watt at tensor parallelism 8. Minimum sample size: 1 reply.
```

### opinion

An argument, forecast or stated goal without field evidence, including a vendor's target. It can frame a question. It is not a result.

```markdown research-note label=opinion
- **[opinion]** The author of the RDNA4 serving-port RFC argues that enabling these cards "belongs with AMD rather than with a one-person fork". [ROCm/vllm#1229](https://github.com/ROCm/vllm/issues/1229)
  - Implication: none for the metric set. It is a view about who should maintain the port, so the pass keeps carrying its own `lab/vllm-gfx1201.diff` and does not count on an upstream fix.
  - Candidate experiment: reread the issue and its linked PRs at the start of each kernels pass. Expected movement: none to our numbers unless an upstream fix lands. Minimum sample size: 1 reread per pass.
```

## 5. Citation forms

Every finding carries at least one of these. Cite the narrowest thing that supports the claim: the section, the comment permalink, the function. A finding that rests on two sources cites both.

| Form | Write it as | Example |
| --- | --- | --- |
| Paper | title, authors, venue, year, and a URL, DOI or arXiv id | Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training (You, Chung, Chowdhury; NSDI 2023), arXiv:2208.06102, https://arxiv.org/abs/2208.06102 |
| Benchmark or leaderboard | the URL of the page and the version or methodology section | a leaderboard page and the section that describes how it measures |
| Vendor or project doc | the doc name, the section, and the URL | `[Zeus docs, Measuring Energy, Hardware Support > AMD GPU > Initialization](https://ml.energy/zeus/measure/)` |
| Issue or PR | the URL, a comment permalink when one comment is the source, or `owner/repo#N`; a bare `PR #N` means a PR in this repo | `[ROCm/vllm#1229](https://github.com/ROCm/vllm/issues/1229)`, `PR #37` |
| Source file and function | repo, commit, file path and function; a permalink with the commit when the repo is public | `runtime/hub/core/spawner/kubernetes.py::RemoteLabKubeSpawner.options_from_form` in AMDResearch/aup-learning-cloud at 9b4af19 ([permalink](https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19bffb9e0cef8046cfffc03d66e929a72d5/runtime/hub/core/spawner/kubernetes.py#L308-L357)) |
| Our ledger line | `ledger:<seq>`, the sequence number of the row in `lab/state/ledger.jsonl`; a number from the baseline metrics file is `lab/refine/metrics.jsonl:<line>` | `ledger:57` (an illustrative number; use the real `seq`) |
| A file in this repo | the repo-relative path, with `::function` or `:line` when it narrows the claim, or a relative link with the section | `lab/README.md`, `[M1](measurements-2026-10-10.md#m1-front-door)` |
| A source ID | `[S49]` or `[S3, S7]`: the ID of an entry in this folder's `SOURCES.md` (one capital letter and a number, so `[L2]` and `[R1]` also work), followed by what in the source supports the claim | `Source: [S49] vLLM benchmarks/kernels/benchmark_moe.py at b1388b1, tune and benchmark_config` |

- Cite only what you opened. A paper known from memory, a title seen in a search result or a model's recollection is not opened.
- Prefer peer-reviewed papers, vendor and project docs, and lab benchmarks over blog posts. A blog post is an `anecdote` unless it shows data.
- Use inline links. Footnotes and reference-style links are not resolved by the checker.
- A paper's citation carries its year and venue as well as its link, so the reader can judge it without opening it. The checker only requires the link.

## 6. Under each finding: implication and candidate experiment

- **Implication:** what the finding means for our metric set. Name the metric from the track's rubric (joules per answer, idle watts, sampler overhead, and so on). If it changes none, write "none for the metric set" and say why, as the anecdote and opinion examples above do.
- **Candidate experiment:** one run that would test it on our hardware, with the **expected movement** (the metric, the direction and a size) and the **minimum sample size** (runs, samples or requests). An experiment is a proposal. Writing it in a note starts nothing. Several findings may share one experiment: define it once, with all its fields and an ID such as `KE5`, and let the others say `Experiment: KE5, as specified in kernels.md section 4`, `Experiment: see the previous finding`, or `Experiment: none beyond KE13`. The checker accepts these pointers without the fields.

Each track asks a little more or less of an experiment, following its research task:

| Track | Expected movement | Minimum sample size | Confidence |
| --- | --- | --- | --- |
| telemetry | required | required | not asked |
| learning | required | required | not asked |
| plumbing | required | optional | not asked |
| kernels | required | required (as minimum runs) | required: low, medium, high, or a number from 0 to 1 |

At the end of the main kernels note, a `## Candidate experiments, ranked` section lists the experiments in order of expected joules-per-token gain times confidence. The checker does not read that section; it is exempt from the finding rules like the other framing sections.

The checker takes the track from the name of the folder the note sits in. A note in any other folder gets the telemetry rules.

## 7. SOURCES.md

One `SOURCES.md` per track folder lists every source the track's notes cite, and every source opened and set aside, one table row or one bullet each. Each entry needs a link and the date it was opened, written `YYYY-MM-DD`.

```markdown research-sources example=complete
# Sources: telemetry

| Source | URL or DOI | Opened | Type | Cited in |
| --- | --- | --- | --- | --- |
| Zeus docs, Measuring Energy | https://ml.energy/zeus/measure/ | 2026-10-10 | project doc | baseline.md |
| Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training (You, Chung, Chowdhury; NSDI 2023) | arXiv:2208.06102 (https://arxiv.org/abs/2208.06102) | 2026-10-10 | paper | not used yet |
| PR 194, R29: teacher on the Strix Halo (gfx1151) | https://github.com/raohemdutt/Knighthacks26/pull/194 | 2026-10-10 | pr | baseline.md |
| ROCm/vllm issue 1229, RDNA4 serving port RFC | https://github.com/ROCm/vllm/issues/1229 | 2026-10-10 | issue | not used yet |
```

- The columns are Source, URL or DOI, Opened, Type and Cited in. Type is one of `paper`, `vendor doc`, `project doc`, `benchmark`, `issue`, `pr`, `code`, `blog`, `other`. A list works as well as a table: `- [Zeus docs](https://ml.energy/zeus/measure/), opened 2026-10-10`.
- An internal source (a file in this repo, a ledger line) may carry its path or `ledger:<seq>` in place of a URL.
- To cite by ID, start the entry with its ID (`S49`, `[S49]` or an `ID` or `Key` column first in the row). `[S07]` and `[S7]` are the same ID. A cited ID that no entry defines is `source-not-indexed`.
- Source code is read at a tag or commit, so a file URL such as `https://github.com/org/repo/blob/9b4af19/a/b.py` matches any index entry that lists the same repo at the same ref (a short commit matches the full one). Other repos and other refs do not match.
- A lead that was seen but never opened (search snippets, a paywalled page) may be recorded with the words `not opened` or `not citable` and no link. It is never cited.
- A table in `SOURCES.md` whose header has no URL, DOI, link, opened or date column (for example a table listing the notes) is not a list of sources and is skipped.
- Every URL, DOI or arXiv id that a finding cites must appear in the index. The checker matches them ignoring `http` versus `https`, `www.`, a trailing slash, a `#fragment`, the `abs` versus `pdf` form of an arXiv link, and an arXiv paper's DOI versus its arXiv link. A comment permalink therefore matches its issue.
- A URL template such as `https://host/user/<name>/proxy/` or `.../users/{name}/server` is an example, never a source. A URL inside a `code span` still counts as a citation but is not looked up in the index, so write a real source as a link.
- A source that was opened but not cited stays in the index, so the next pass does not open it again.

## 8. The search bound

- **Budget:** about two hours of search and reading per research step for telemetry, learning and plumbing, and about three hours for kernels. A pass's research step has the same budget as its track.
- **Stop rule:** stop when the last three sources opened repeat what earlier sources already said. Stop at the budget even if new findings keep arriving, and list the open leads under `## Open questions`.
- **Order:** papers, vendor and project docs, and lab benchmarks first; then issues, PRs and source code; blogs and forums last.
- **Record:** a `## Search log` entry per research step: the date, the budget, the time used, the starting points and queries, and why it stopped. The checker does not read it. The reviewer does.

## 9. Citing a finding from a pass

A pass that relies on a note's finding cites it by file and heading, and states the finding's label:

```text
SOURCE: lab/refine/research/telemetry/sampling.md#sampler-trust (practitioner)
```

- The file is the repo-relative path with forward slashes.
- The heading is the GitHub anchor of the `##` heading the finding sits under: lowercase, punctuation dropped, spaces turned into hyphens. `## Sampler trust` is `#sampler-trust`. When one heading text repeats in a file, GitHub numbers the later ones `-1`, `-2`. Cite the `##` topic heading, which the checker requires to be unique in its note.
- Headings are the addresses. A pass never renames or deletes a heading that a log, a ruling or a queued experiment cites. It adds a new heading instead.
- The pass opens the note and the finding's own source before it acts, and cites that source as well when the change alters a method. No citation, no change.
- `python -m lab research-check --ref lab/refine/research/telemetry/sampling.md#sampler-trust` checks that the file and the heading exist. Run it from the repo root, or pass `--root`.

## 10. What the checker enforces

| Code | Fires when |
| --- | --- |
| `missing-path`, `not-markdown`, `no-notes` | the path does not exist; a file is not `.md`; a folder holds no notes |
| `no-findings` | a note has no bullet under a `##` topic heading |
| `no-label`, `unknown-label`, `multiple-labels` | a finding starts with no label; with a bracketed word that is not one of the four; with two labels |
| `no-citation` | a finding carries no URL, DOI, arXiv id, `owner/repo#N`, `PR #N`, `ledger:N`, a source ID such as `[S49]`, a relative link to a file in the repo, or a file path (with a folder, or with `::function` or `:line`) |
| `no-implication`, `no-experiment` | a finding has no implication, or no candidate experiment |
| `no-expected-movement`, `no-minimum-sample`, `no-confidence` | the experiment does not state them, per the track table in section 6 |
| `duplicate-heading` | two `##` headings in one note have the same anchor |
| `sources-missing`, `sources-empty` | a folder with notes has no `SOURCES.md`; the index lists nothing |
| `source-no-url`, `source-no-date` | an index entry has no link or locator, or no date opened |
| `source-not-indexed` | a finding cites a URL, DOI, arXiv id or source ID that the index does not list |
| `bad-reference` | a `file#heading` reference names no markdown file under the root, or no heading in it |

Each violation is one string in the answer's `violations` list: `<file>:<line>: <code>: <message>`. A single note is checked against the `SOURCES.md` beside it when there is one. A folder is checked note by note, then its index. A folder with no notes of its own is checked subfolder by subfolder, so `lab/refine/research` covers all four tracks.

The checker cannot tell whether a source exists, whether it says what the finding says, whether a label is the right one, whether a number came with its setup, or whether the search stayed inside its bound. Those are the reviewer's checks. A clean run means the note is well formed, not that it is true.

## 11. Where each rule comes from

The format is the one the four track research tasks already specify. The table shows what each asks for, so a note written to its task page passes without changes.

| Rule | telemetry | learning | plumbing | kernels |
| --- | --- | --- | --- | --- |
| One finding per bullet, labeled measured, practitioner, anecdote or opinion | asked | asked | asked | asked |
| Citation on each bullet | paper, vendor doc, ledger line | paper, vendor doc | doc and section, source file and function, ledger line | paper, PR or issue URL, doc section, ledger line |
| Implication for our metric set | asked | asked | asked | asked |
| Candidate experiment with expected movement | asked | asked | asked | asked |
| Minimum sample size | asked | asked | not asked | asked, as minimum runs |
| Confidence, and a ranked list at the end of the main note | not asked | not asked | not asked | asked |
| `SOURCES.md` with a link and the date opened | asked | asked | asked | asked |
| Search budget | about 2 h | about 2 h | about 2 h | about 3 h |

Every citation form is accepted in every track. The labels and the habit of citing a page, episode or URL for each item come from the phone project's practitioner critique notes, and "no citation, no change; nothing from memory" from its refinement goal file.
