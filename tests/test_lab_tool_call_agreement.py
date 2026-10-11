"""Tool-call agreement (lab/tool_call_agreement.py) on fakes: the prompt is the trainer's, the marks follow the rule,
the two tests are the exact ones, and a run leaves its files and one ledger row. No server is called."""

import json
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from lab import format_traces as ft
from lab import tool_call_agreement as tca
from lab.dataset import load_split
from lab.eval_gate import sign_flip_p
from lab.fixture_dataset import load_expected

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "tool-calling-fixture"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "tool-call-agreement-test")


class World:
    """The committed fixture: its split, its references, and which example a prompt belongs to."""

    def __init__(self) -> None:
        self.split = load_split(FIXTURE, require_images=False)
        self.expected = load_expected(FIXTURE)
        self.system = self.split.data.manifest.spec.instruction
        self.by_prompt = {tca.prompt_for(self.system, e.context): e for e in self.split.data.examples}
        assert len(self.by_prompt) == len(self.split.data.examples)   # one prompt per example

    def gold(self, example_id: str) -> list[dict]:
        return [{"name": c.name, "arguments": c.arguments} for c in self.expected[example_id].label.tool_calls]


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def turn(calls: list[dict]) -> str:
    """An assistant turn as the template writes it after the empty thinking block, through <|im_end|>."""
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c} for c in calls]}]
    text, spans = ft.render_with_spans(messages, ft.TOOLS)
    return text[spans[0][0]:spans[0][1]]


class Student:
    """A fake model: it writes the reference call, except where `instead` gives other calls for the example."""

    def __init__(self, world: World, instead=lambda example: None) -> None:
        self.world, self.instead = world, instead

    def complete(self, prompt: str) -> dict:
        example = self.world.by_prompt[prompt]
        calls = self.instead(example)
        return {"text": turn(self.world.gold(example.id) if calls is None else calls)}


def test_the_prompt_is_a_training_row_up_to_its_first_assistant_span(world):
    example = world.split.held_out[0]
    reference = world.expected[example.id]
    calls = tuple(ft.as_call(call) for call in world.gold(example.id))
    trace = ft.Trace(example.id, world.system, example.context, (ft.Round(calls, tuple(reference.tool_results)),),
                     reference.label.answer, example.frame_ref)
    row = ft.build_row(trace)
    prompt = tca.prompt_for(world.system, example.context)
    assert prompt == row["text"][:row["spans"][0][0]]
    assert prompt.endswith(ft.GENERATION_PROMPT)


def test_an_output_is_right_only_when_its_calls_are_the_reference_calls(world):
    example = next(e for e in world.split.held_out if world.gold(e.id)[0]["name"] == "last_seen")
    reference, gold = world.expected[example.id], world.gold(example.id)

    def marks(output: str) -> tuple[bool, bool, bool]:
        scored = tca.score_output(output, reference)
        return scored["correct"], scored["names"], scored["parsed"]

    assert marks(turn(gold)) == (True, True, True)
    assert marks(turn(gold).removesuffix(ft.IM_END)) == (True, True, True)
    assert marks(turn(gold) + "\nwhatever came after the end of the turn") == (True, True, True)
    assert marks(turn([{"name": "last_seen", "arguments": {"label": "something else"}}])) == (False, True, True)
    assert marks(turn([*gold, {"name": "search_memory", "arguments": {"q": "keys"}}])) == (False, False, True)
    assert marks("I didn't see that.") == (False, False, False)
    unclosed = tca.score_output("<tool_call>\n<function=last_seen>\n", reference)
    assert (unclosed["correct"], unclosed["parsed"], unclosed["calls"]) == (False, False, None)
    assert unclosed["problem"]


def test_mcnemar_is_the_exact_two_sided_binomial_on_the_discordant_pairs():
    assert tca.mcnemar_exact(0, 8) == tca.mcnemar_exact(8, 0) == Fraction(1, 128)
    assert tca.mcnemar_exact(5, 1) == Fraction(7, 32)
    assert tca.mcnemar_exact(16, 15) == 1
    assert tca.mcnemar_exact(0, 0) == 1


def test_a_run_counts_the_flips_pairs_by_group_and_leaves_its_files(world, tmp_path):
    held_out = world.split.held_out
    groups = sorted({e.frame_ref for e in held_out})
    gains = [e for e in held_out if e.frame_ref == groups[0]]    # the before model misses this whole group
    other = [e for e in held_out if e.frame_ref == groups[1]]
    losses = {e.id for e in other[:2]}                           # the after model misses two of another
    wrong = [{"name": "search_memory", "arguments": {"q": "x"}}]
    before = Student(world, lambda e: wrong if e.frame_ref == groups[0] else None)
    after = Student(world, lambda e: wrong if e.id in losses else None)

    summary = tca.run(dataset=FIXTURE, completers={"before": before, "after": after}, out=tmp_path / "out",
                      run_id="run-fake", before_runs=2, after_runs=1)

    n = len(held_out)
    assert (summary["n_items"], summary["n_groups"]) == (n, 4)
    assert summary["correct"] == {"current": n - len(gains), "candidate": n - 2}
    assert summary["noise_runs"] == [n - len(gains)] * 2 and summary["noise"] == 0.0
    assert summary["delta"] == pytest.approx((len(gains) - 2) / n)
    assert set(summary["flips"]["gained_0_to_1"]) == {e.id for e in gains}
    assert set(summary["flips"]["lost_1_to_0"]) == losses
    assert summary["mcnemar"]["p_exact"] == str(tca.mcnemar_exact(len(gains), 2))
    # One group all gained and one group two lost: over 4 groups the gate's test cannot call that anything.
    assert summary["p_value"] == float(sign_flip_p([Fraction(1), Fraction(-2, len(other)), 0, 0])) == 1.0
    assert summary["sign_flip"]["smallest_possible_p"] == 0.125
    assert summary["identical_outputs"] == {"before_runs": n, "after_runs": None}
    assert summary["rule"] == tca.RULE and summary["alpha"] == 0.05

    saved = json.loads((tmp_path / "out" / "held_out.summary.json").read_text(encoding="utf-8"))
    assert {"correct", "noise_runs", "alpha", "test", "items", "n_items", "n_groups", "p_value", "split_sha256"} <= set(saved)
    item = next(i for i in saved["items"] if i["id"] == gains[0].id)
    assert set(item) == {"id", "group", "question", "expected", "before", "after"}
    assert (item["before"]["correct"], item["after"]["correct"], item["group"]) == (False, True, groups[0])
    assert json.loads(item["expected"]) == {"tool_calls": world.gold(gains[0].id)}
    raw = (tmp_path / "out" / "raw" / "held_out.before-2.jsonl").read_text(encoding="utf-8").splitlines()
    header, rows = json.loads(raw[0]), [json.loads(line) for line in raw[1:]]
    assert header["header"] is True and header["rule"] == tca.RULE and header["arm"] == "before-2"
    assert len(rows) == n and {"id", "prompt_sha256", "output", "calls", "expected", "correct", "names", "parsed"} <= set(rows[0])
    assert (tmp_path / "out" / "held_out.md").read_text(encoding="utf-8").startswith("# Tool-call agreement, held_out side")


def test_the_verb_refuses_a_server_that_is_not_greedy_and_writes_one_ledger_row(world, tmp_path):
    examples = list(world.split.held_out)
    index = {e.id: n for n, e in enumerate(examples)}

    def server(top_k: int):
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/props":
                return httpx.Response(200, json={"model_path": f"/models/{request.url.host}.gguf",
                                                 "default_generation_settings": {"n_ctx": 4096, "params": {"top_k": top_k}}})
            body = json.loads(request.content)
            if request.url.path == "/completion":
                assert (body["n_predict"], body["temperature"]) == (512, 0.2) and "top_k" not in body
                example = world.by_prompt[body["prompt"]]
                return httpx.Response(200, json={"content": "", "tokens": [index[example.id]], "stop_type": "eos",
                                                 "tokens_predicted": 26, "tokens_evaluated": 500, "truncated": False})
            assert request.url.path == "/detokenize"
            example = examples[body["tokens"][0]]
            missed = request.url.host == "after" and example.id == examples[0].id
            return httpx.Response(200, json={"content": "I do not know." + ft.IM_END if missed else turn(world.gold(example.id))})
        return httpx.MockTransport(handle)

    argv = ["--dataset", str(FIXTURE), "--run-id", "run-fake", "--before-url", "http://before", "--after-url",
            "http://after", "--out", str(tmp_path / "out"), "--limit", "6", "--before-runs", "2", "--after-runs", "2"]

    refused = tca.main(argv, transport=server(top_k=40))
    assert (refused["ok"], refused["outcome"]) == (False, "refused") and not (tmp_path / "out").exists()

    answer = tca.main(argv, transport=server(top_k=1))
    assert answer["ok"] is True and answer["n_items"] == 6
    assert answer["correct"] == {"current": 6, "candidate": 5} and answer["flips"]["lost_1_to_0"] == [examples[0].id]
    assert answer["parsed"]["after-1"] == 5 and answer["after_runs"] == [5, 5]
    rows = [json.loads(line) for line in (tmp_path / "state" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["verb"] for row in rows] == ["tool-call-agreement"]
    assert rows[0]["metrics"]["learning.tool_call_agreement.before"] == 1.0
    assert rows[0]["metrics"]["learning.tool_call_agreement.lost"] == 1
    assert rows[0]["refs"]["run_id"] == "run-fake" and rows[0]["conditions"]["side"] == "held_out"
