from contexts.agent.eval import main, run, score
from runtime import FakeRuntime


def test_agent_retrieves_the_right_evidence_for_every_golden_question():
    results = run(FakeRuntime(), judge_answers=False)
    failed = [(r.q, r.cited) for r in results if not r.passed]
    assert failed == []
    assert score(results) == 1


def test_cli_writes_json(tmp_path):
    out = tmp_path / "eval.json"
    assert main(["--json", str(out)]) == 0
    assert '"score": 1.0' in out.read_text()
