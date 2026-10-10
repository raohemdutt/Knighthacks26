from contexts.agent.eval import main, run, score
from runtime import FakeRuntime


def test_agent_retrieves_the_right_evidence_for_every_golden_question():
    results = run(FakeRuntime(), judge_answers=False)
    failed = [(r.q, r.cited) for r in results if not r.passed]
    assert failed == []
    assert score(results) == 1


def test_every_demo_question_and_paraphrase_takes_a_fixed_path():
    """A10: on stage, a 4B model must never be left to choose tools; every way of asking has a prefetch shape."""
    results = run(FakeRuntime(), judge_answers=False)
    assert len(results) >= 40
    assert [r.q for r in results if r.shape != "prefetch"] == []


def test_cli_writes_json(tmp_path):
    out = tmp_path / "eval.json"
    assert main(["--json", str(out)]) == 0
    assert '"score": 1.0' in out.read_text()
