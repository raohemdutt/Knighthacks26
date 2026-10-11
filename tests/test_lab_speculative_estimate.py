"""The offline n-gram speculative decoding estimate: vLLM's draft rule, the greedy step count, and the report."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from lab import speculative_estimate as se
from lab.__main__ import VERBS

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "fixtures" / "lab" / "speculative-tiny"


def _text(tokens):
    return "".join(map(chr, tokens))


def test_the_fast_draft_equals_the_port_of_vllms_search_on_random_tokens():
    rng = random.Random(54)
    for _ in range(3000):
        tokens = [rng.randrange(4) for _ in range(rng.randrange(0, 40))]
        min_n = rng.randrange(1, 4)
        max_n = rng.randrange(min_n, 6)
        k = rng.randrange(1, 9)
        expected = se.propose_reference(tokens, min_n, max_n, k)
        assert se.propose(_text(tokens), min_n, max_n, k) == _text(expected), (tokens, min_n, max_n, k)


def test_the_draft_is_what_followed_the_earliest_longest_match():
    # the suffix [1, 2] occurs at 0 and at 4; the earliest wins, so the draft is what followed it there
    tokens = [1, 2, 3, 9, 1, 2, 7, 5, 1, 2]
    assert se.propose_reference(tokens, 1, 3, 2) == [3, 9]
    # a longer match beats an earlier shorter one: [7, 1, 2] occurs once before the end
    tokens = [1, 2, 4, 7, 1, 2, 8, 7, 1, 2]
    assert se.propose_reference(tokens, 1, 3, 1) == [8]
    # no n-gram of at least min_n tokens recurs: no draft
    assert se.propose_reference([1, 2, 3, 4], 2, 3, 4) == []


def test_an_output_copied_from_the_prompt_is_accepted_in_one_step():
    tally = se.simulate([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], 1, 3, 4)
    assert (tally.tokens, tally.steps, tally.drafted, tally.accepted) == (5, 1, 4, 4)
    assert tally.summary()["tokens_per_step"] == 4.0


def test_an_output_with_nothing_to_look_up_takes_one_step_per_token():
    tally = se.simulate([1, 2, 3], [10, 11, 12, 13], 1, 3, 4)
    assert (tally.steps, tally.drafted, tally.accepted) == (3, 0, 0)
    assert tally.summary()["tokens_per_step"] == 1.0


def test_a_wrong_draft_still_emits_the_target_token():
    # prompt [1, 2, 3]; output [1, 2, 4]: after 1 the draft is [2, 3], 2 is accepted, 3 is not; the step emits 2, 4
    tally = se.simulate([1, 2, 3], [1, 2, 4], 1, 3, 2)
    assert (tally.steps, tally.drafted, tally.accepted) == (1, 2, 1)


def test_the_requests_are_the_assistant_turns_of_the_teacher_labels_and_the_template_cases():
    labels = se.label_requests(TINY)
    assert [(r.source, r.kind) for r in labels] == [("teacher_labels", "call"), ("teacher_labels", "answer")] * 2
    assert labels[0].prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert labels[0].output.startswith("<tool_call>\n<function=episodes_between>")
    assert labels[1].output.endswith("<|im_end|>") and "<tool_response>" in labels[1].prompt
    cases = se.template_requests(TINY / "rendered.jsonl")
    row = json.loads((TINY / "rendered.jsonl").read_text(encoding="utf-8"))
    assert len(cases) == 2
    # the rendered prompt plus the output is the real template's text up to the end of that turn
    assert row["text"].startswith(cases[1].prompt + cases[1].output)


def test_the_report_names_the_rule_the_inputs_and_every_setting(tmp_path):
    out = tmp_path / "estimate.json"
    summary = se.main(["--labels", str(TINY), "--template-cases", str(TINY / "rendered.jsonl"), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["format"] == "atlas.speculative-acceptance/1"
    assert report["rule"]["vllm_commit"].startswith("b1388b1")
    assert report["requests"] == summary["requests"] == 6
    assert len(report["settings"]) == len(se.LOOKUPS) * len(se.DRAFT_LENGTHS)
    for setting in report["settings"]:
        overall = setting["groups"]["all"]
        assert overall["tokens_per_step"] >= 1.0
        assert overall["decode_steps"] <= overall["baseline_decode_steps"]
        assert {"teacher_labels/call", "teacher_labels/answer", "template_cases"} <= set(setting["groups"])


def test_a_missing_input_is_an_error_that_names_the_file(tmp_path):
    with pytest.raises(se.EstimateError, match="manifest.json"):
        se.label_requests(tmp_path)


def test_the_verb_is_registered_and_runs():
    assert VERBS["speculative-estimate"] == "lab.speculative_estimate:main"
    done = subprocess.run([sys.executable, "-m", "lab", "speculative-estimate", "--labels", str(TINY),
                           "--template-cases", str(TINY / "rendered.jsonl")],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1])["requests"] == 6
