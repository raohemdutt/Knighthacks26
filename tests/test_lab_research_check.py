"""Tests for the research-note checker: labels, citations, the SOURCES index, and file-and-heading references."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lab.machine import research_check as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT_DOC = REPO_ROOT / "lab" / "refine" / "RESEARCH_FORMAT.md"

PR37_URL = "https://github.com/raohemdutt/Knighthacks26/pull/37"
ZEUS_URL = "https://ml.energy/zeus/measure/"
ISSUE_URL = "https://github.com/ROCm/vllm/issues/1229"
PR37 = f"[PR 37]({PR37_URL})"
ZEUS = f"[Zeus docs, Hardware Support: AMD GPU]({ZEUS_URL})"

IMPLICATION = "  - Implication: joules per token moves with decode speed at fixed power."
EXPERIMENT = (
    "  - Candidate experiment: rerun the same prompt after each kernel change. "
    "Expected movement: 20 percent fewer joules per token. Minimum sample size: 5 runs."
)
EXPERIMENT_NO_MINIMUM = (
    "  - Candidate experiment: rerun the probe. Expected movement: time to ready drops to 15 s."
)
CLAIM = "Decode ran at 7.7 tok/s at about 200 W average on one R9700."


def bullet(lead="**[measured]**", claim=CLAIM, cite=PR37, detail=(IMPLICATION, EXPERIMENT)):
    """One finding: the label and claim and citation on the first line, then the indented detail lines."""
    return "\n".join([f"- {lead} {claim} {cite}".rstrip(), *detail])


def note(*bullets, heading="Sampler accuracy"):
    return "# Sampler notes\n\nScope paragraph.\n\n## " + heading + "\n\n" + "\n".join(bullets) + "\n"


def put(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def line_of(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} is not in the text")


def parse(violations):
    """Violations as (line or None, code), so tests do not depend on message wording or the temp path."""
    parsed = []
    for violation in violations:
        match = re.match(r"^(?P<where>.+?): (?P<code>[a-z]+(?:-[a-z]+)*): (?P<message>.+)$", violation)
        assert match, violation
        tail = match["where"].rsplit(":", 1)
        line = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None
        parsed.append((line, match["code"]))
    return parsed


GOOD_NOTE = f"""\
# Decode energy on one R9700

Scope: joules per token for decode on the AMD pods.

## Baseline decode energy

- **[measured]** Decode ran at 7.7 tok/s at about 200 W average and 300 W peak on one R9700 serving a 4-bit 27B model with vLLM, about 26 J per token. {PR37}
  - Implication: joules per token moves with decode speed at fixed power, so a faster kernel lowers it almost one for one.
  - Candidate experiment: rerun the same prompt after each kernel change. Expected movement: 20 to 40 percent fewer joules per token. Minimum sample size: 5 runs of 256 tokens.

## Sampler trust

- **[practitioner]** Zeus reads the AMD energy counter twice, 0.5 s apart, and compares the difference with power before it trusts the counter. {ZEUS}
  - Implication: our sampler must cross-check the energy counter against integrated power before it reports joules.
  - Candidate experiment: log both for 60 s idle and under load. Expected movement: counter and integrated power agree within a few percent. Minimum sample size: 3 windows of 60 s.
- **[anecdote]** One operator reports 8,681 requests in five days on 8 R9700 cards with zero errors and says its raw sweep logs are gone. [ROCm/vllm#1229]({ISSUE_URL}#issuecomment-5809360213)
  - Implication: no throughput or energy number can be taken from this report.
  - Candidate experiment: ask for the raw logs before using any figure. Expected movement: none until logs exist. Minimum sample size: 1 reply.
- **[opinion]** The RFC author believes the RDNA4 port belongs with AMD rather than with a one-person fork. [ROCm/vllm#1229]({ISSUE_URL})
  - Implication: assume we carry our own vLLM patch for the rest of the weekend.
  - Candidate experiment: reread the issue at the start of each pass. Expected movement: none to our numbers unless an upstream fix lands. Minimum sample size: 1 reread per pass.

## Open questions

- Does the pod report current_socket_power, or N/A?
"""

BAD_NOTE = """\
# Sampler notes

Scope paragraph.

## Sampler accuracy

- Zeus polls power on a separate thread.
- **[own side]** The vendor says it is accurate. https://example.org/vendor
  - Implication: trust the vendor figure.
  - Candidate experiment: compare with a wall meter. Expected movement: none. Minimum sample size: 3 runs.
- **[measured]** Idle power was 38 W on the pod.
  - Implication: subtract 38 W from every window.
  - Candidate experiment: repeat at night. Expected movement: none. Minimum sample size: 3 runs.
- **[opinion]** Sampling slower than 1 s is fine. https://example.org/opinion
- **[anecdote]** One user saw a 5 percent gap. https://example.org/anecdote
  - Implication: a small gap is plausible.
  - Candidate experiment: compare with a wall meter.
- **[measured / practitioner]** Both at once. https://example.org/both
  - Implication: none.
  - Candidate experiment: none. Expected movement: none. Minimum sample size: 1.
"""

SOURCES_OK = f"""\
# Sources: telemetry

| Source | URL or DOI | Opened | Type |
| --- | --- | --- | --- |
| Zeus docs, Measuring Energy | {ZEUS_URL} | 2026-10-10 | project doc |
| PR 37 | {PR37_URL} | 2026-10-10 | pr |
| ROCm/vllm issue 1229 | {ISSUE_URL} | 2026-10-10 | issue |
"""


# ---- the two sample notes ---------------------------------------------------------------------------------


def test_well_formed_note_has_no_violations(tmp_path):
    path = put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    assert rc.check(path) == []


def test_badly_formed_note_reports_each_broken_rule_on_its_own_line(tmp_path):
    path = put(tmp_path, "research/telemetry/accuracy.md", BAD_NOTE)
    unlabeled, vendor, idle = (line_of(BAD_NOTE, s) for s in ("Zeus polls power", "[own side]", "Idle power was 38 W"))
    opinion, anecdote, both = (line_of(BAD_NOTE, s) for s in ("Sampling slower than 1 s", "One user saw", "measured / practitioner"))
    assert parse(rc.check(path)) == [
        (unlabeled, "no-label"),
        (unlabeled, "no-citation"),
        (unlabeled, "no-implication"),
        (unlabeled, "no-experiment"),
        (vendor, "unknown-label"),
        (idle, "no-citation"),
        (opinion, "no-implication"),
        (opinion, "no-experiment"),
        (anecdote, "no-expected-movement"),
        (anecdote, "no-minimum-sample"),
        (both, "multiple-labels"),
    ]


def test_violation_messages_name_the_file_the_line_and_the_fix(tmp_path):
    path = put(tmp_path, "research/telemetry/accuracy.md", BAD_NOTE)
    first = rc.check(path)[0]
    assert first.startswith(f"{path.as_posix()}:{line_of(BAD_NOTE, 'Zeus polls power')}: no-label: ")
    assert "measured, practitioner, anecdote, opinion" in first


# ---- labels -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", rc.LABELS)
def test_each_of_the_four_labels_is_accepted(tmp_path, label):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(lead=f"**[{label}]**")))
    assert rc.check(path) == []


@pytest.mark.parametrize(
    "lead",
    [
        "[measured]",
        "**[measured]**",
        "**[measured, n=3]**",
        "**[Measured]**",
        "*[measured]*",
        "`[measured]`",
        "`measured`",
        "`measured`:",
        "**measured**",
        "**Measured:**",
        "__measured__",
        "measured:",
        "Measured -",
        "Measured \N{EN DASH}",
        "Measured \N{EM DASH}",
        "(measured)",
        "(measured, one run)",
    ],
)
def test_natural_ways_of_writing_a_label_are_accepted(tmp_path, lead):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(lead=lead)))
    assert rc.check(path) == []


@pytest.mark.parametrize(
    "lead",
    [
        "Measured power draw was low and",
        "**Zeus** reports that",
        "[Zeus docs](https://ml.energy/zeus/measure/) report that",
        "(Zeus) reports that",
        "An opinion piece says that",
        "measuring",
    ],
)
def test_a_bullet_that_only_mentions_a_label_word_is_unlabeled(tmp_path, lead):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(lead=lead)))
    assert [code for _, code in parse(rc.check(path))] == ["no-label"]


@pytest.mark.parametrize("lead", ["**[own side]**", "**[mesaured]**", "[UNTRACEABLE]"])
def test_a_bracketed_word_that_is_not_one_of_the_four_labels_is_an_unknown_label(tmp_path, lead):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(lead=lead)))
    assert [code for _, code in parse(rc.check(path))] == ["unknown-label"]


@pytest.mark.parametrize(
    "lead, claim",
    [
        ("**[measured / practitioner]**", CLAIM),
        ("**[measured, opinion]**", CLAIM),
        ("**[measured]**", "Decode was fast. **[opinion]** and slow."),
    ],
)
def test_one_bullet_carries_exactly_one_label(tmp_path, lead, claim):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(lead=lead, claim=claim)))
    assert [code for _, code in parse(rc.check(path))] == ["multiple-labels"]


def test_label_of_returns_the_canonical_label_or_none():
    assert rc.label_of("**[Practitioner, n=40]** Farrokh books 1 in 3.") == "practitioner"
    assert rc.label_of("anecdote: a single report") == "anecdote"
    assert rc.label_of("**[own side]** Voss says so.") is None
    assert rc.label_of("Zeus samples every 100 ms.") is None


# ---- citations --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cite",
    [
        f"[Zeus docs]({ZEUS_URL})",
        f"Zeus docs, section Hardware Support, {ZEUS_URL}",
        f"<{ZEUS_URL}>",
        "Zeus (You et al., 2023, NSDI), doi:10.48550/arXiv.2208.06102",
        "Zeus (You et al., 2023, NSDI), arXiv:2208.06102",
        "ROCm/vllm#1229",
        "PR #37",
        "ledger:128",
        "ledger seq 128",
        "Ledger line 128",
        "lab/README.md",
        "lab/refine/metrics.jsonl:5",
        "runtime/hub/core/spawner/kubernetes.py::RemoteLabKubeSpawner.options_from_form",
        "kubernetes.py::options_from_form",
    ],
)
def test_each_citation_form_counts_as_a_citation(tmp_path, cite):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))
    assert rc.check(path) == []


@pytest.mark.parametrize(
    "cite",
    [
        "",
        "(Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training, 2023, NSDI)",
        "see above",
        "as measured by llama.cpp",
        "our ledger",
        "issue 1229",
        "kubernetes.py",
    ],
)
def test_a_title_a_vague_pointer_or_a_bare_name_is_not_a_citation(tmp_path, cite):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))
    assert [code for _, code in parse(rc.check(path))] == ["no-citation"]


def test_a_source_line_under_the_finding_supplies_its_citation(tmp_path):
    detail = (f"  - Source: {PR37}", IMPLICATION, EXPERIMENT)
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(cite="", detail=detail)))
    assert rc.check(path) == []


@pytest.mark.parametrize(
    "template",
    [
        "https://dub.aupcloud.io/user/<account>/proxy/<port>/",
        "https://dub.aupcloud.io/hub/api/users/{name}/server",
    ],
)
def test_a_url_template_in_a_claim_is_neither_a_citation_nor_a_source_to_index(tmp_path, template):
    claim = f"The proxy path is {template} with the account token."
    put(tmp_path, "research/telemetry/a.md", note(bullet(claim=claim)))
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK)
    assert rc.check(tmp_path / "research" / "telemetry") == []
    uncited = put(tmp_path, "research/telemetry/b.md", note(bullet(claim=claim, cite="")))
    assert [code for _, code in parse(rc.check(uncited))] == ["no-citation"]


def test_an_example_url_in_a_code_span_is_not_matched_against_the_index(tmp_path):
    claim = "The hub answers at `https://dub.aupcloud.io/hub/api/` when the token is valid."
    put(tmp_path, "research/telemetry/a.md", note(bullet(claim=claim)))
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK)
    assert rc.check(tmp_path / "research" / "telemetry") == []


def test_a_link_inside_the_implication_is_not_the_findings_citation(tmp_path):
    detail = (f"  - Implication: see {ZEUS_URL} for the counter check.", EXPERIMENT)
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(cite="", detail=detail)))
    assert [code for _, code in parse(rc.check(path))] == ["no-citation"]


@pytest.mark.parametrize("cite", ["Source: [S49] vLLM benchmark", "[S3, S7]", "([S12])"])
def test_a_source_id_is_a_citation_when_there_is_no_index_to_check_it_against(tmp_path, cite):
    assert rc.check(put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))) == []


INDEX_WITH_IDS = """# Sources

| ID | Source | URL or DOI | Opened |
| --- | --- | --- | --- |
| S01 | Zeus docs | https://ml.energy/zeus/measure/ | 2026-10-10 |
| S49 | vLLM benchmark | https://github.com/vllm-project/vllm | 2026-10-10 |

- [S7] PR 37 - https://github.com/raohemdutt/Knighthacks26/pull/37 - opened 2026-10-10
"""


@pytest.mark.parametrize("cite", ["[S49]", "[S1]", "[S01]", "[S7]", "[S49, S7]"])
def test_a_source_id_must_name_an_entry_of_the_index(tmp_path, cite):
    put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))
    put(tmp_path, "research/telemetry/SOURCES.md", INDEX_WITH_IDS)
    assert rc.check(tmp_path / "research" / "telemetry") == []


def test_a_source_id_the_index_does_not_define_is_reported(tmp_path):
    body = note(bullet(cite="[S49, S50]"))
    put(tmp_path, "research/telemetry/a.md", body)
    put(tmp_path, "research/telemetry/SOURCES.md", INDEX_WITH_IDS)
    result = rc.check(tmp_path / "research" / "telemetry")
    assert parse(result) == [(line_of(body, "- **[measured]**"), "source-not-indexed")]
    assert "[S50]" in result[0]


def test_a_source_id_in_a_code_span_is_not_a_citation(tmp_path):
    assert [c for _, c in parse(rc.check(put(tmp_path, "research/telemetry/a.md", note(bullet(cite="`[S49]`")))))] == ["no-citation"]


@pytest.mark.parametrize(
    "cite", ["[log M1](measurements-2026-10-10.md#m1-front-door)", "[proxy](proxy-path.md)", "[x](../other/file.json#row-3)"]
)
def test_a_relative_link_to_a_record_in_the_repo_is_a_citation(tmp_path, cite):
    assert rc.check(put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))) == []


@pytest.mark.parametrize("cite", ["[see below](#section)", "[home](nothing)", "[page](https:/broken)"])
def test_an_in_page_anchor_or_a_link_to_nothing_is_not_a_citation(tmp_path, cite):
    assert [c for _, c in parse(rc.check(put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cite)))))] == ["no-citation"]


@pytest.mark.parametrize(
    "pointer",
    [
        "see the previous finding.",
        "covered by the cull timing experiment above.",
        "KE5, as specified in `triton_gfx12.md` section 4.",
        "KE15 (energy-aware selection) reuses a tuner like this; see section 3.",
        "the same experiment as the one below.",
    ],
)
def test_an_experiment_that_points_at_another_experiment_need_not_repeat_its_fields(tmp_path, pointer):
    detail = ("  - Implication: joules per token follows decode speed.", f"  - Experiment: {pointer}")
    assert rc.check(put(tmp_path, "research/kernels/a.md", note(bullet(detail=detail)))) == []


def test_a_sentence_that_merely_contains_the_word_see_is_not_a_pointer(tmp_path):
    detail = ("  - Implication: x.", "  - Experiment: rerun it to see whether the number holds.")
    codes = [c for _, c in parse(rc.check(put(tmp_path, "research/telemetry/a.md", note(bullet(detail=detail)))))]
    assert codes == ["no-expected-movement", "no-minimum-sample"]


@pytest.mark.parametrize("cite", ["[L2]", "[R1]", "Source: `measurements-2026-10-10.md` M1 and M6 [R1]"])
def test_other_one_letter_ids_work_like_source_ids(tmp_path, cite):
    path = put(tmp_path, "research/plumbing/a.md", note(bullet(cite=cite)))
    assert rc.check(path) == []
    index = (
        "# Sources\n\n"
        "- [L2] T0 page - https://app.notion.com/p/abc - 2026-10-10\n"
        "- [R1] record - lab/refine/research/plumbing/measurements-2026-10-10.md - 2026-10-10\n"
    )
    put(tmp_path, "research/plumbing/SOURCES.md", index)
    assert rc.check(tmp_path / "research" / "plumbing") == []


@pytest.mark.parametrize(
    "pointer",
    [
        "none beyond KE13.",
        "none from the headline; the kernels below map to KE3 and KE9.",
        "see the measured propagation below; repeat it on pod B's next session; minimum n: 3 observations.",
    ],
)
def test_an_experiment_may_say_none_or_point_below(tmp_path, pointer):
    detail = ("  - Implication: x.", f"  - Experiment: {pointer}")
    assert rc.check(put(tmp_path, "research/kernels/a.md", note(bullet(detail=detail)))) == []


def test_a_measurements_record_is_not_a_note_and_a_notes_table_in_the_index_is_not_a_source(tmp_path):
    put(tmp_path, "research/plumbing/a.md", note(bullet(cite="[log](measurements-2026-10-10.md#m1)")))
    put(tmp_path, "research/plumbing/measurements-2026-10-10.md", "# Record\n\n## M1. Probe\n\nPlain prose, no bullets.\n")
    sources = (
        "# Sources\n\n| Note | Covers |\n| --- | --- |\n| `a.md` | decode energy |\n\n"
        "| ID | Source | URL | Opened |\n| --- | --- | --- | --- |\n"
        "| S1 | Zeus | https://ml.energy/zeus/measure/ | 2026-10-10 |\n"
    )
    put(tmp_path, "research/plumbing/SOURCES.md", sources)
    assert rc.check(tmp_path / "research" / "plumbing") == []


# ---- which bullets are findings ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "Scope",
        "1. Scope",
        "Summary",
        "Method",
        "Search log",
        "Search bound (two hours)",
        "Sources",
        "Sources opened",
        "Open questions",
        "Gaps",
        "Inferences",
        "Takeaway",
        "Candidate experiments, ranked by expected gain times confidence",
        "Ranked experiments",
    ],
)
def test_bullets_under_non_finding_headings_are_not_checked(tmp_path, heading):
    text = note(bullet(), "", f"## {heading}", "", "- Some plain list item with no label or link.")
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert rc.check(path) == []


def test_non_finding_status_is_inherited_by_subsections(tmp_path):
    text = note(bullet(), "", "## Open questions", "", "### On sampling", "", "- Is 1 Hz enough?")
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert rc.check(path) == []


@pytest.mark.parametrize("heading", ["Experiments others ran", "Sources of error in sampling", "Search-time power"])
def test_a_topic_heading_that_only_resembles_an_exempt_one_is_still_a_findings_section(tmp_path, heading):
    text = note(bullet(), "", f"## {heading}", "", "- Zeus polls power on a thread.")
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert "no-label" in [code for _, code in parse(rc.check(path))]


def test_bullets_before_the_first_topic_heading_are_not_findings(tmp_path):
    text = "# Title\n\n- Track: telemetry\n- Date: 2026-10-10\n\n## Topic\n\n" + bullet() + "\n"
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert rc.check(path) == []


def test_a_note_with_no_findings_is_rejected(tmp_path):
    path = put(tmp_path, "research/telemetry/a.md", "# Title\n\nJust prose.\n\n## Open questions\n\n- Why?\n")
    assert parse(rc.check(path)) == [(None, "no-findings")]


def test_a_numbered_finding_is_checked_like_a_bullet(tmp_path):
    good = f"1. **[measured]** {CLAIM} {PR37}\n   - Implication: x.\n   - {EXPERIMENT.strip()[2:]}"
    bad = "2. Zeus polls power on a thread."
    text = "# T\n\n## Topic\n\n" + good + "\n" + bad + "\n"
    path = put(tmp_path, "research/telemetry/a.md", text)
    at = line_of(text, "Zeus polls power")
    assert parse(rc.check(path)) == [
        (at, "no-label"),
        (at, "no-citation"),
        (at, "no-implication"),
        (at, "no-experiment"),
    ]


PHONE_PROJECT_STYLE = f"""\
# Energy per token: sampling and baselines

Source labels used throughout: **[measured]** = a number someone measured; **[opinion]** = argument without evidence.

## 1. How power is sampled

### Takeaway

Prefer the energy counter where one exists.

### Cited Findings

- **[measured, n=3]** {CLAIM} - {PR37}
  - **What it implies for our metrics:** log the sampling rate next to every joules number.
  - **Candidate experiment:** sweep the rate. *Expected delta:* 3 percent. *Min n:* 3 windows.

### Inferences

- The counter beats polling.

### Gaps

- No source found for sampler overhead.

## 2. Idle baseline

### Takeaway

One policy everywhere.

### Cited Findings

- **[opinion]** Idle subtraction should use a pre-run baseline. [Blog](https://example.org/idle)
  - Implication: record idle watts at the start of each sitting.
  - Candidate experiment: compare baselines. Expected movement: 5 percent. Minimum sample size: 5 requests.

### Gaps

- None.
"""


def test_the_phone_projects_takeaway_findings_inferences_gaps_layout_is_accepted(tmp_path):
    path = put(tmp_path, "research/telemetry/a.md", PHONE_PROJECT_STYLE)
    assert rc.check(path) == []


def test_findings_under_a_repeated_subheading_are_still_checked(tmp_path):
    broken = PHONE_PROJECT_STYLE.replace("**[opinion]** Idle subtraction", "Idle subtraction")
    path = put(tmp_path, "research/telemetry/a.md", broken)
    at = line_of(broken, "Idle subtraction")
    assert parse(rc.check(path)) == [(at, "no-label")]


def test_a_numbered_ranking_at_the_end_of_a_kernels_note_is_not_a_finding(tmp_path):
    experiment = EXPERIMENT + " Confidence: medium."
    ranking = "\n## Candidate experiments, ranked\n\n1. Concurrency sweep.\n2. Tuned kernel.\n"
    path = put(tmp_path, "research/kernels/a.md", note(bullet(detail=(IMPLICATION, experiment))) + ranking)
    assert rc.check(path) == []


# ---- how a finding is read --------------------------------------------------------------------------------


def test_implication_and_experiment_may_be_sibling_bullets_that_follow_the_finding(tmp_path):
    flat = "\n".join(
        [
            f"- **[measured]** {CLAIM} {PR37}",
            "- Implication: joules per token follows decode speed.",
            "- Candidate experiment: rerun after each change. Expected movement: lower. Minimum sample size: 5 runs.",
        ]
    )
    path = put(tmp_path, "research/telemetry/a.md", note(flat))
    assert rc.check(path) == []


def test_implication_and_experiment_may_sit_on_the_findings_own_line(tmp_path):
    inline = (
        f"- **[measured]** {CLAIM} {PR37} Implication: joules per token follows decode speed. "
        "Candidate experiment: rerun after each change. Expected movement: lower. Minimum sample size: 5 runs."
    )
    path = put(tmp_path, "research/telemetry/a.md", note(inline))
    assert rc.check(path) == []


def test_a_continuation_line_belongs_to_the_bullet_above_it(tmp_path):
    wrapped = f"- **[measured]** {CLAIM}\n  The source is {PR37}.\n{IMPLICATION}\n{EXPERIMENT}"
    path = put(tmp_path, "research/telemetry/a.md", note(wrapped))
    assert rc.check(path) == []


def experiment_with(text):
    return ("  - Implication: joules per token follows decode speed.", f"  - Candidate experiment: rerun the probe. {text}")


@pytest.mark.parametrize(
    "stated",
    [
        "Expected movement: 20 percent fewer joules per token. Minimum sample size: 5 runs.",
        "Expected delta: -5 J/token. Min n: 3.",
        "Expected: lower. n = 30.",
        "We expect a drop. n>=30.",
        "Predicted movement: down. n \N{GREATER-THAN OR EQUAL TO} 30.",
        "Expected movement: down. At least 20 requests.",
        "Expected movement: down. 30+ samples.",
        "Expected movement: down. Sample size: 12.",
        "Expected movement: down. Minimum runs: 5.",
    ],
)
def test_each_natural_way_of_stating_movement_and_sample_size_is_accepted(tmp_path, stated):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(detail=experiment_with(stated))))
    assert rc.check(path) == []


@pytest.mark.parametrize(
    "stated, codes",
    [
        ("Minimum sample size: 5 runs.", ["no-expected-movement"]),
        ("Expected movement: lower. A few runs.", ["no-minimum-sample"]),
        ("Expected movement: lower. Enough data to be sure.", ["no-minimum-sample"]),
        ("", ["no-expected-movement", "no-minimum-sample"]),
    ],
)
def test_an_experiment_that_leaves_out_movement_or_sample_size_is_reported(tmp_path, stated, codes):
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(detail=experiment_with(stated))))
    assert [code for _, code in parse(rc.check(path))] == codes


def test_expected_movement_in_the_implication_does_not_stand_in_for_the_experiments(tmp_path):
    detail = ("  - Implication: we expect a small gap.", "  - Candidate experiment: rerun the probe.")
    path = put(tmp_path, "research/telemetry/a.md", note(bullet(detail=detail)))
    assert [code for _, code in parse(rc.check(path))] == ["no-expected-movement", "no-minimum-sample"]


def test_bullets_in_code_fences_html_comments_and_front_matter_are_ignored(tmp_path):
    text = (
        "---\ntitle: x\n---\n# T\n\n## Topic\n\n"
        + bullet()
        + "\n\n```\n- an unlabeled bullet inside a fence\n```\n\n<!--\n- a bullet inside a comment\n-->\n"
    )
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert rc.check(path) == []


def test_windows_line_endings_and_tab_indented_details_are_accepted(tmp_path):
    text = note(bullet(detail=("\t- Implication: x.", "\t- " + EXPERIMENT.strip()[2:])))
    path = tmp_path / "research" / "telemetry" / "a.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert rc.check(path) == []


def test_a_level_two_heading_used_twice_is_ambiguous_to_cite(tmp_path):
    text = note(bullet()) + "\n## Sampler accuracy\n\n" + bullet() + "\n"
    path = put(tmp_path, "research/telemetry/a.md", text)
    second = text.splitlines().index("## Sampler accuracy", text.splitlines().index("## Sampler accuracy") + 1) + 1
    assert parse(rc.check(path)) == [(second, "duplicate-heading")]


def test_a_level_three_heading_may_repeat_under_different_topics(tmp_path):
    section = "### Cited findings\n\n" + bullet() + "\n"
    text = "# T\n\n## First topic\n\n" + section + "\n## Second topic\n\n" + section
    path = put(tmp_path, "research/telemetry/a.md", text)
    assert rc.check(path) == []


# ---- what each track asks of an experiment ----------------------------------------------------------------


def test_kernels_experiments_state_a_confidence(tmp_path):
    text = note(bullet())
    path = put(tmp_path, "research/kernels/ports.md", text)
    assert parse(rc.check(path)) == [(line_of(text, "- **[measured]**"), "no-confidence")]


def test_kernels_experiment_with_a_confidence_passes(tmp_path):
    experiment = EXPERIMENT + " Confidence: medium."
    path = put(tmp_path, "research/kernels/ports.md", note(bullet(detail=(IMPLICATION, experiment))))
    assert rc.check(path) == []


def test_plumbing_experiments_may_omit_the_minimum_sample_size(tmp_path):
    text = note(bullet(detail=(IMPLICATION, EXPERIMENT_NO_MINIMUM)))
    assert rc.check(put(tmp_path, "research/plumbing/a.md", text)) == []


@pytest.mark.parametrize("track", ["telemetry", "learning", "kernels", "elsewhere"])
def test_the_other_tracks_and_unknown_folders_require_the_minimum_sample_size(tmp_path, track):
    text = note(bullet(detail=(IMPLICATION, EXPERIMENT_NO_MINIMUM + " Confidence: low.")))
    path = put(tmp_path, f"research/{track}/a.md", text)
    assert parse(rc.check(path)) == [(line_of(text, "- **[measured]**"), "no-minimum-sample")]


# ---- paths and folders ------------------------------------------------------------------------------------


def test_a_missing_path_is_reported(tmp_path):
    assert parse(rc.check(tmp_path / "nope.md")) == [(None, "missing-path")]


def test_a_file_that_is_not_markdown_is_reported(tmp_path):
    path = put(tmp_path, "research/telemetry/a.txt", GOOD_NOTE)
    assert parse(rc.check(path)) == [(None, "not-markdown")]


def test_an_empty_folder_is_reported(tmp_path):
    (tmp_path / "research").mkdir()
    assert parse(rc.check(tmp_path / "research")) == [(None, "no-notes")]


def test_a_folder_is_checked_note_by_note_and_its_index_and_readme_are_not_notes(tmp_path):
    put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    put(tmp_path, "research/telemetry/accuracy.md", BAD_NOTE)
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK)
    put(tmp_path, "research/telemetry/README.md", "- an unlabeled bullet that is not a note\n")
    violations = rc.check(tmp_path / "research" / "telemetry")
    assert violations
    assert all("/accuracy.md:" in v.replace("\\", "/") for v in violations)


def test_a_folder_without_a_sources_index_is_reported(tmp_path):
    put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    assert parse(rc.check(tmp_path / "research" / "telemetry")) == [(None, "sources-missing")]


def test_a_root_folder_with_no_notes_of_its_own_checks_each_track_folder(tmp_path):
    put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK)
    put(tmp_path, "research/learning/accuracy.md", BAD_NOTE)
    put(tmp_path, "research/learning/SOURCES.md", SOURCES_OK)
    (tmp_path / "research" / "kernels").mkdir()
    violations = rc.check(tmp_path / "research")
    assert violations
    assert all("/learning/" in v.replace("\\", "/") for v in violations)


def test_a_single_note_is_checked_against_the_sources_index_beside_it_when_there_is_one(tmp_path):
    body = note(bullet(), bullet(lead="**[practitioner]**", claim="Zeus checks the counter.", cite=ZEUS))
    path = put(tmp_path, "research/telemetry/a.md", body)
    assert rc.check(path) == []
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK.replace(ZEUS_URL, "https://example.org/other"))
    assert parse(rc.check(path)) == [(line_of(body, "Zeus checks"), "source-not-indexed")]


# ---- the SOURCES index ------------------------------------------------------------------------------------


def sources_check(tmp_path, text):
    path = put(tmp_path, "research/telemetry/SOURCES.md", text)
    return parse(rc.check(path))


def test_a_sources_table_with_a_url_and_a_date_on_every_row_is_valid(tmp_path):
    assert sources_check(tmp_path, SOURCES_OK) == []


def test_a_sources_list_is_valid_with_a_url_and_a_date_on_every_item(tmp_path):
    text = f"# Sources\n\n- [Zeus docs]({ZEUS_URL}), opened 2026-10-10\n- {PR37_URL} - opened Oct 10, 2026\n"
    assert sources_check(tmp_path, text) == []


def test_a_source_without_a_url_or_without_a_date_is_reported_on_its_line(tmp_path):
    text = (
        "# Sources\n\n"
        f"- Zeus docs {ZEUS_URL}\n"
        "- A paper that was read on 2026-10-10\n"
        f"- {PR37_URL} opened 2026-10-10\n"
    )
    assert sources_check(tmp_path, text) == [(3, "source-no-date"), (4, "source-no-url")]


def test_table_header_and_separator_rows_are_not_sources(tmp_path):
    text = "| Source | URL or DOI | Opened |\n| :--- | :--- | :--- |\n" f"| Zeus | {ZEUS_URL} | 2026-10-10 |\n"
    assert sources_check(tmp_path, text) == []


def test_an_index_with_no_sources_is_reported(tmp_path):
    assert sources_check(tmp_path, "# Sources\n\nNothing yet.\n") == [(None, "sources-empty")]


@pytest.mark.parametrize(
    "cited, indexed",
    [
        (ZEUS_URL, "http://www.ml.energy/zeus/measure"),
        (f"{ISSUE_URL}#issuecomment-5809360213", "https://github.com/rocm/vllm/issues/1229"),
        ("https://arxiv.org/abs/2208.06102", "https://arxiv.org/pdf/2208.06102v2"),
        ("https://arxiv.org/abs/2208.06102", "arXiv:2208.06102"),
        ("doi:10.48550/arXiv.2208.06102", "https://doi.org/10.48550/arXiv.2208.06102"),
        ("doi:10.48550/arXiv.2208.06102", "https://arxiv.org/abs/2208.06102"),
        ("https://dl.acm.org/doi/10.1145/3620665.3640411", "doi:10.1145/3620665.3640411"),
    ],
)
def test_a_cited_source_matches_its_index_entry_despite_url_spelling(tmp_path, cited, indexed):
    put(tmp_path, "research/telemetry/a.md", note(bullet(cite=cited)))
    put(tmp_path, "research/telemetry/SOURCES.md", f"# Sources\n\n- {indexed}, opened 2026-10-10\n")
    assert rc.check(tmp_path / "research" / "telemetry") == []


def test_a_cited_source_missing_from_the_index_is_reported_on_the_finding_line(tmp_path):
    body = note(bullet(cite="[Other](https://example.org/other)"))
    put(tmp_path, "research/telemetry/a.md", body)
    put(tmp_path, "research/telemetry/SOURCES.md", SOURCES_OK)
    assert parse(rc.check(tmp_path / "research" / "telemetry")) == [(line_of(body, "- **[measured]**"), "source-not-indexed")]


REPO = "https://github.com/AMDResearch/aup-learning-cloud"
SHA = "9b4af19bffb9e0cef8046cfffc03d66e929a72d5"


@pytest.mark.parametrize(
    "cited, indexed",
    [
        (f"{REPO}/blob/{SHA}/dockerfiles/Hub/Dockerfile#L22", f"{REPO}/blob/{SHA}/dockerfiles/Base/Dockerfile.rocm"),
        (f"{REPO}/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py", f"{REPO}/blob/{SHA}/README.md"),
        (f"{REPO}/blob/{SHA}/README.md", f"{REPO}/tree/9b4af19/runtime"),
    ],
)
def test_source_code_is_indexed_at_the_granularity_of_the_repo_and_ref_that_was_read(tmp_path, cited, indexed):
    put(tmp_path, "research/plumbing/a.md", note(bullet(cite=cited)))
    put(tmp_path, "research/plumbing/SOURCES.md", f"# Sources\n\n- {indexed}, opened 2026-10-10\n")
    assert rc.check(tmp_path / "research" / "plumbing") == []


def test_source_code_at_another_ref_or_in_another_repo_is_not_indexed(tmp_path):
    body = note(bullet(cite=f"{REPO}/blob/main/README.md"), bullet(cite="https://github.com/jupyterhub/jupyterhub/blob/9b4af19/a.py"))
    put(tmp_path, "research/plumbing/a.md", body)
    put(tmp_path, "research/plumbing/SOURCES.md", f"# Sources\n\n- {REPO}/blob/{SHA}/README.md, opened 2026-10-10\n")
    assert [c for _, c in parse(rc.check(tmp_path / "research" / "plumbing"))] == ["source-not-indexed", "source-not-indexed"]


def test_an_index_entry_may_record_a_lead_that_was_never_opened(tmp_path):
    entry = "- Web search results on SSE buffering (snippets only) - n/a - 2026-10-10 - not opened as pages, so not citable\n"
    assert sources_check(tmp_path, "# Sources\n\n" + entry) == []


# ---- slugs and references ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, slug",
    [
        ("Sampler trust", "sampler-trust"),
        ("1. Scope & method", "1-scope--method"),
        ("Joules per token (J/token)", "joules-per-token-jtoken"),
        ("`amd-smi` power reads", "amd-smi-power-reads"),
        ("[Zeus](https://ml.energy/zeus/) docs", "zeus-docs"),
        ("Idle_baseline  policy", "idle_baseline--policy"),
    ],
)
def test_slugify_follows_the_github_anchor_rules(heading, slug):
    assert rc.slugify(heading) == slug


@pytest.fixture
def repo(tmp_path):
    put(tmp_path, "lab/refine/research/telemetry/sampling.md", "# T\n\n## Sampler trust\n\n### Findings\n\n## Other\n\n### Findings\n")
    return tmp_path


@pytest.mark.parametrize(
    "ref",
    [
        "lab/refine/research/telemetry/sampling.md#sampler-trust",
        "lab/refine/research/telemetry/sampling.md#Sampler trust",
        "lab/refine/research/telemetry/sampling.md#findings",
        "lab/refine/research/telemetry/sampling.md#findings-1",
    ],
)
def test_a_reference_to_an_existing_file_and_heading_is_valid(repo, ref):
    assert rc.check_reference(ref, repo) == []


@pytest.mark.parametrize(
    "ref",
    [
        "lab/refine/research/telemetry/missing.md#sampler-trust",
        "lab/refine/research/telemetry/sampling.md#no-such-heading",
        "lab/refine/research/telemetry/sampling.md#findings-2",
        "lab/refine/research/telemetry/sampling.md",
        "lab/refine/research/telemetry/sampling.md#",
        "../outside.md#sampler-trust",
        "lab/refine/research/telemetry#sampler-trust",
    ],
)
def test_a_reference_without_a_real_file_and_heading_is_rejected(repo, ref):
    assert [code for _, code in parse(rc.check_reference(ref, repo))] == ["bad-reference"]


# ---- the research-check verb ------------------------------------------------------------------------------


def test_main_returns_every_violation_with_the_counts(tmp_path):
    path = put(tmp_path, "research/telemetry/accuracy.md", BAD_NOTE)
    assert rc.main([str(path)]) == {
        "ok": False,
        "outcome": "violations",
        "count": 11,
        "violations": rc.check(path),
        "notes": 1,
        "findings": 6,
    }


def test_main_reports_a_clean_note(tmp_path):
    path = put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    assert rc.main([str(path)]) == {
        "ok": True,
        "outcome": "clean",
        "count": 0,
        "violations": [],
        "notes": 1,
        "findings": 4,
    }


def test_main_reports_a_missing_path(tmp_path):
    result = rc.main([str(tmp_path / "nope")])
    assert (result["ok"], result["notes"], result["findings"]) == (False, 0, 0)
    assert parse(result["violations"]) == [(None, "missing-path")]


def test_main_checks_references_with_the_ref_option(repo):
    good = rc.main(["--root", str(repo), "--ref", "lab/refine/research/telemetry/sampling.md#sampler-trust"])
    assert (good["ok"], good["violations"]) == (True, [])
    bad = rc.main(["--root", str(repo), "--ref", "lab/refine/research/telemetry/sampling.md#nope"])
    assert (bad["ok"], parse(bad["violations"])) == (False, [(None, "bad-reference")])


def test_main_expands_wildcards_itself_because_powershell_does_not(tmp_path):
    put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    result = rc.main([str(tmp_path / "research" / "*" / "*.md")])
    assert (result["ok"], result["notes"]) == (True, 1)


def test_main_without_a_path_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as raised:
        rc.main([])
    assert raised.value.code == 2
    assert "give at least one path or --ref" in capsys.readouterr().err


@pytest.fixture
def lab_state(tmp_path, monkeypatch):
    """A private state dir, so a STOP file in the real one cannot refuse the verb."""
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))


def run_lab(*args):
    return subprocess.run(
        [sys.executable, "-m", "lab", "research-check", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def test_the_dispatcher_answers_a_clean_note_with_one_json_line_and_exit_0(tmp_path, lab_state):
    path = put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    done = run_lab(str(path))
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.splitlines()) == 1
    assert json.loads(done.stdout) == rc.main([str(path)])


def test_the_dispatcher_answers_a_bad_note_with_the_violations_and_exit_1(tmp_path, lab_state):
    path = put(tmp_path, "research/telemetry/accuracy.md", BAD_NOTE)
    done = run_lab(str(path))
    assert done.returncode == 1
    assert json.loads(done.stdout)["violations"] == rc.check(path)


def test_the_dispatcher_lists_the_verb_and_answers_its_help(lab_state):
    helped = run_lab("-h")
    assert helped.returncode == 0
    assert helped.stdout.startswith("usage: python -m lab research-check")
    listing = subprocess.run(
        [sys.executable, "-m", "lab", "--help"], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert re.search(r"^  research-check  Check research notes", listing.stdout, re.MULTILINE), listing.stdout


def test_the_verb_is_refused_while_the_stop_file_exists(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP").write_text("halted for the test", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    path = put(tmp_path, "research/telemetry/baseline.md", GOOD_NOTE)
    done = run_lab(str(path))
    assert done.returncode == 1
    assert json.loads(done.stdout)["outcome"] == "stopped"


# ---- the format document stays consistent with the checker ------------------------------------------------

EXAMPLE_BLOCK = re.compile(
    r"^```markdown (?P<kind>research-note|research-sources)(?P<attrs>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.S | re.M
)


def doc_examples():
    """The (kind, attributes, text) of every ```markdown research-note and research-sources block in the format document."""
    assert FORMAT_DOC.is_file(), f"expected {FORMAT_DOC.relative_to(REPO_ROOT).as_posix()} to hold the format"
    text = FORMAT_DOC.read_text(encoding="utf-8")
    examples = []
    for match in EXAMPLE_BLOCK.finditer(text):
        attrs = dict(re.findall(r"(\w+)=(\w+)", match["attrs"]))
        body = match["body"]
        if match["kind"] == "research-note" and not body.startswith("# "):
            body = "# Example\n\n## Example findings\n\n" + body  # a fragment of findings gets a note around it
        examples.append((match["kind"], attrs, body))
    return examples


def test_every_example_in_the_format_document_passes_the_checker(tmp_path):
    examples = doc_examples()
    assert {kind for kind, _, _ in examples} == {"research-note", "research-sources"}
    for number, (kind, attrs, body) in enumerate(examples):
        name = "SOURCES.md" if kind == "research-sources" else f"example-{number}.md"
        path = put(tmp_path, f"research/{attrs.get('track', 'telemetry')}/{name}", body)
        assert rc.check(path) == [], body


def test_the_format_document_has_one_example_for_each_label():
    shown = {}
    for kind, attrs, body in doc_examples():
        if kind == "research-note" and "label" in attrs:
            first_bullet = next(line for line in body.splitlines() if line.startswith("- "))
            shown.setdefault(attrs["label"], []).append(rc.label_of(first_bullet[2:]))
    assert shown == {label: [label] for label in rc.LABELS}


def test_the_documents_complete_note_and_its_sources_index_agree_with_each_other(tmp_path):
    notes = [(a, b) for kind, a, b in doc_examples() if kind == "research-note" and a.get("example") == "complete"]
    sources = [b for kind, a, b in doc_examples() if kind == "research-sources" and a.get("example") == "complete"]
    assert len(notes) == 1 and len(sources) == 1
    put(tmp_path, "research/telemetry/baseline.md", notes[0][1])
    put(tmp_path, "research/telemetry/SOURCES.md", sources[0])
    assert rc.check(tmp_path / "research" / "telemetry") == []
