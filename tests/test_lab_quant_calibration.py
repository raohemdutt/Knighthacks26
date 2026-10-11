"""Guards on the teacher quantization's calibration set (fixtures/lab/quant).

The committed calibration.jsonl is what the Colab run ships, so it has to match what its generator writes today,
come only from text the repository already holds, and look like serve-time traffic: Qwen3.5 chat turns with
thinking off and tool calls in the template's XML form.
"""

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT = REPO_ROOT / "fixtures" / "lab" / "quant"


def load_generator():
    spec = importlib.util.spec_from_file_location("make_calibration", QUANT / "make_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def committed_rows():
    return [json.loads(line) for line in (QUANT / "calibration.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_committed_set_is_what_the_generator_writes_today():
    gen = load_generator()
    assert gen.render(gen.build()) == (QUANT / "calibration.jsonl").read_text(encoding="utf-8")
    assert gen.main(["--check"]) == 0


def test_check_fails_on_a_stale_file(tmp_path, capsys):
    gen = load_generator()
    stale = tmp_path / "calibration.jsonl"
    stale.write_text('{"id": "old"}\n', encoding="utf-8")
    assert gen.main(["--check", "--out", str(stale)]) == 1
    assert "stale" in capsys.readouterr().err


def test_two_builds_are_byte_identical(tmp_path):
    gen = load_generator()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    gen.main(["--out", str(a)])
    gen.main(["--out", str(b)])
    assert a.read_bytes() == b.read_bytes()


def test_every_source_is_present_with_the_expected_counts():
    counts = Counter(r["source"] for r in committed_rows())
    assert counts == {"tool-calling": 118, "golden": 48, "qwen35-template": 16, "bench": 2}


def test_rows_are_unique_and_shaped():
    rows = committed_rows()
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    for r in rows:
        assert set(r) == {"id", "source", "text", "chars"}
        assert r["chars"] == len(r["text"]) > 0


def test_rows_are_chat_turns_with_thinking_off():
    for r in committed_rows():
        assert r["text"].startswith("<|im_start|>"), r["id"]
        assert "<think>\n\n</think>" in r["text"], r["id"]


def test_tool_calls_use_the_template_xml_form():
    tool_rows = [r for r in committed_rows() if r["source"] == "tool-calling"]
    with_calls = [r for r in tool_rows if "<tool_call>" in r["text"]]
    assert with_calls, "the tool-calling rows should carry tool calls"
    for r in with_calls:
        assert re.search(r"<tool_call>\n<function=\w+>\n", r["text"]), r["id"]
        assert "<tool_response>" in r["text"], r["id"]


def test_no_image_or_frame_content_enters_the_set():
    for r in committed_rows():
        assert "<|vision_start|>" not in r["text"] and "<|image_pad|>" not in r["text"], r["id"]
        assert not re.search(r"\.(png|jpe?g)\b", r["text"]), r["id"]


def test_the_set_is_large_enough_for_group_wise_calibration():
    total_chars = sum(r["chars"] for r in committed_rows())
    # about four characters per token: tens of thousands of tokens, enough for 128-wide groups
    assert total_chars // 4 > 20_000
