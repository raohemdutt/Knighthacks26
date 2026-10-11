"""lab/correctness.py, the correctness harness (R21): record a baseline of greedy outputs, compare a later run.

Every test talks to lab/bench_fake.py spawned as a real server process (port 0, named in a file), the same fake the
bench tests use. It answers the same prompt with the same token ids every time, and --diverge-at / --retoken-at make
a second server answer differently from a chosen position on, so a compare against it has a known first divergence."""
from __future__ import annotations

import contextlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from lab import __main__ as lab_main
from lab import bench, bench_fake, correctness
from lab.machine import ledger, metrics

TOKENS = 8
CHIP = "gfx1151"
FLAGS = ["--engine-commit", "b1388b1", "--quantization", "int4", "--chip", CHIP, "--rocm", "7.13",
         "--torch", "2.9.1+rocm7.13.0", "--triton", "3.5.1", "--max-tokens", str(TOKENS), "--node", "pod-a"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)


@contextlib.contextmanager
def server(tmp_path, name="fake", **options):
    """A fake server as a real process; yields its base URL."""
    port_file = tmp_path / f"{name}.port"
    argv = bench.fill_command(bench_fake.serve_command(root=str(tmp_path / f"{name}-pod"), **options),
                              {"port": 0, "port_file": port_file, "log": tmp_path / f"{name}.log"})
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30.0
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert port_file.exists(), "the fake never named its port"
        yield f"http://127.0.0.1:{port_file.read_text()}"
    finally:
        proc.terminate()
        proc.wait(10)


def run_main(*argv):
    return correctness.main([str(a) for a in argv])


def record(tmp_path, url, *extra, date="2026-10-10"):
    return run_main("record", "--url", url, "--out-dir", tmp_path / "fixtures", "--date", date, *FLAGS, *extra)


def compare(tmp_path, url, baseline, *extra, flags=FLAGS):
    return run_main("compare", "--url", url, "--baseline", baseline, "--state-dir", tmp_path / "runs", *flags, *extra)


def swap(flags, name, value):
    """FLAGS with one flag's value changed."""
    out = list(flags)
    out[out.index(name) + 1] = value
    return out


def last_row():
    return ledger.read()[-1]


# --- record -----------------------------------------------------------------------------------------------------

def test_record_writes_the_fixture_with_every_condition_and_one_output_per_prompt(tmp_path):
    with server(tmp_path) as url:
        result = record(tmp_path, url, "--patch", "lab/vllm-gfx1201.diff")
    fixture = tmp_path / "fixtures" / f"vllm-{CHIP}-2026-10-10.json"
    assert result["ok"] is True and result["fixture"] == str(fixture)
    doc = json.loads(fixture.read_text(encoding="utf-8"))
    prompts = bench.PromptSet.load()
    assert doc["schema"] == correctness.BASELINE_SCHEMA
    assert [o["prompt"] for o in doc["outputs"]] == [p["id"] for p in prompts.prompts]
    assert {o["kind"] for o in doc["outputs"]} == {"text", "image"}
    for out in doc["outputs"]:
        assert len(out["tokens"]) == TOKENS == len(out["logprobs"]) == out["completion_tokens"]
        assert all(isinstance(t, int) for t in out["tokens"]) and out["token_kind"] == "token_id"
        assert out["text"] == "".join(f"t{i} " for i in range(TOKENS)) and out["finish_reason"] == "length"
    c = doc["conditions"]
    assert c["engine"] == "vllm" and c["engine_commit"] == "b1388b1" and c["quantization"] == "int4"
    assert c["patch_hash"] == {"vllm-gfx1201.diff": bench.sha256_file(Path("lab/vllm-gfx1201.diff"))}
    assert c["checkpoint"] == "/home/jovyan/models/Qwen3.8-27B-INT4" and c["max_model_len"] == 16384  # /v1/models
    assert c["served_model"] == "teacher" and c["node"] == "pod-a" and c["chip"] == CHIP
    assert (c["rocm"], c["torch"], c["triton"]) == ("7.13", "2.9.1+rocm7.13.0", "3.5.1")
    assert (c["max_tokens"], c["temperature"], c["top_p"], c["seed"]) == (TOKENS, 0.0, 1.0, 0)
    assert c["ignore_eos"] is True and c["thinking"] is False and c["nonce"] is False
    assert c["prompt_set"] == prompts.describe() and c["unavailable"] == {}
    row = last_row()
    assert row["verb"] == "correctness" and row["outcome"] == "ok" and row["metrics"] == {"teacher.output_prompts": 4}


def test_record_sends_greedy_one_at_a_time_requests_that_ask_for_token_ids(tmp_path, monkeypatch):
    seen = []
    real = bench.DirectEndpoint.events

    def spy(self, body):
        seen.append(body)
        return real(self, body)

    monkeypatch.setattr(bench.DirectEndpoint, "events", spy)
    with server(tmp_path) as url:
        record(tmp_path, url)
    assert len(seen) == 4
    for body in seen:
        assert body["temperature"] == 0 and body["seed"] == 0 and body["max_tokens"] == TOKENS
        assert body["ignore_eos"] is True and body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["logprobs"] is True and body["return_tokens_as_token_ids"] is True
        assert "[run " not in json.dumps(body["messages"])             # no nonce: the prompt is the same every run


def test_record_refuses_to_overwrite_a_baseline_and_a_missing_chip_and_writes_nothing_on_a_failed_prompt(tmp_path):
    with server(tmp_path) as url:
        assert record(tmp_path, url)["ok"] is True
        again = record(tmp_path, url)
        assert again["ok"] is False and "--force" in again["error"]
        assert record(tmp_path, url, "--force")["ok"] is True
        no_chip = run_main("record", "--url", url, "--out-dir", tmp_path / "x", "--max-tokens", TOKENS)
        assert no_chip["ok"] is False and "--chip" in no_chip["error"]
        wrong = run_main("record", "--url", url, "--out-dir", tmp_path / "y", "--model", "nope", *FLAGS)
    assert wrong["ok"] is False and not (tmp_path / "y").exists()
    assert last_row()["outcome"] == "refused"


def test_a_record_names_the_server_one_way_only(tmp_path):
    result = run_main("record", "--out-dir", tmp_path / "fx", *FLAGS)
    assert result["ok"] is False and "--url or --account" in result["error"]


# --- compare ----------------------------------------------------------------------------------------------------

def test_compare_against_the_same_server_matches_every_prompt_and_writes_the_rows(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with server(tmp_path) as url:
        baseline = record(tmp_path, url)["fixture"]
        result = compare(tmp_path, url, baseline, "--metrics-file", metrics_path)
    assert result["ok"] is True and result["outcome"] == "ok"
    assert (result["prompts"], result["matches"], result["divergences"]) == (4, 4, 0)
    assert result["match_rate"] == 1.0 and result["normalised_text_matches"] == 4
    assert result["first_divergence_prompt"] is None and result["diverged"] == []
    assert result["condition_drift"] is False and result["unverified"] == [] and result["under_test"] is None
    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert [p["tokens_match"] for p in saved["prompts"]] == [True] * 4
    assert all(p["first_divergence"] is None for p in saved["prompts"])
    row = last_row()
    assert row["verb"] == "correctness" and row["outcome"] == "ok"
    assert row["metrics"] == {"teacher.output_matches": 4, "teacher.output_divergences": 0,
                              "teacher.output_prompts": 4, "teacher.output_match_rate": 1.0}
    assert row["refs"]["first_divergence_prompt"] is None and row["refs"]["run_id"] == result["run_id"]
    [mrow] = metrics.load(metrics_path)
    assert mrow["metric"] == "teacher.output_match_rate" and mrow["value"] == 1.0 and mrow["unit"] == "fraction"
    assert mrow["node"] == "pod-a" and mrow["conditions"]["batch_size"] == 1 and mrow["low_confidence"] is True
    assert mrow["conditions"]["chip"] == CHIP and mrow["conditions"]["matches"] == 4


def test_compare_with_a_divergent_server_reports_the_first_divergence_per_prompt(tmp_path):
    with server(tmp_path) as same, \
            server(tmp_path, "other", diverge_at=3, diverge_match="tidy a small kitchen") as other:
        baseline = record(tmp_path, same)["fixture"]
        result = compare(tmp_path, other, baseline)
    assert result["ok"] is False and result["outcome"] == "diverged" and "text-kitchen" in result["error"]
    assert (result["matches"], result["divergences"], result["match_rate"]) == (3, 1, 0.75)
    assert result["first_divergence_prompt"] == "text-kitchen" and result["first_divergence_index"] == 3
    [diverged] = result["diverged"]
    assert diverged["prompt"] == "text-kitchen" and diverged["first_divergence"] == 3
    assert diverged["logprob_difference"] == pytest.approx(-0.5)
    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    kitchen = saved["prompts"][0]
    assert kitchen["tokens_match"] is False and kitchen["baseline_token_at"] + 1 == kitchen["run_token_at"]
    assert kitchen["text_equal"] is False and kitchen["text_normalised_equal"] is False
    assert result["normalised_text_matches"] == 3
    row = last_row()
    assert row["outcome"] == "diverged" and row["metrics"]["teacher.output_divergences"] == 1
    assert row["metrics"]["teacher.output_match_rate"] == 0.75
    assert row["refs"]["first_divergence_prompt"] == "text-kitchen" and row["refs"]["first_divergence_index"] == 3


def test_a_different_token_with_the_same_text_is_a_token_divergence_but_a_normalised_text_match(tmp_path):
    with server(tmp_path) as same, server(tmp_path, "other", retoken_at=2) as other:
        baseline = record(tmp_path, same)["fixture"]
        result = compare(tmp_path, other, baseline)
    assert result["divergences"] == 4 and result["first_divergence_prompt"] == "text-kitchen"
    assert result["first_divergence_index"] == 2 and result["normalised_text_matches"] == 4


def test_a_baseline_whose_conditions_differ_is_refused_before_anything_is_sent(tmp_path, monkeypatch):
    with server(tmp_path) as url:
        baseline = record(tmp_path, url)["fixture"]
        sent = []
        real = bench.DirectEndpoint.events
        monkeypatch.setattr(bench.DirectEndpoint, "events", lambda self, body: sent.append(body) or real(self, body))
        for name, value in (("--quantization", "fp8"), ("--max-tokens", "4"), ("--engine-commit", "deadbeef"),
                            ("--rocm", "7.2"), ("--chip", "gfx1201")):
            result = compare(tmp_path, url, baseline, flags=swap(FLAGS, name, value))
            assert result["ok"] is False and result["outcome"] == "refused", name
            assert "conditions differ" in result["error"] and "--allow-condition-drift" in result["error"]
            assert name.lstrip("-").replace("-", "_") in result["error"] and "nothing was sent" in result["error"]
        assert sent == [] and not (tmp_path / "runs").exists()
        assert last_row()["outcome"] == "refused"
        drift = compare(tmp_path, url, baseline, "--allow-condition-drift", flags=swap(FLAGS, "--quantization", "fp8"))
    assert drift["ok"] is True and drift["matches"] == 4 and drift["condition_drift"] is True
    assert len(sent) == 4


def test_a_different_served_checkpoint_or_max_length_is_refused(tmp_path):
    with server(tmp_path) as url, server(tmp_path, "other", checkpoint="/models/other", max_len=8192) as other:
        baseline = record(tmp_path, url)["fixture"]
        result = compare(tmp_path, other, baseline)
    assert result["outcome"] == "refused" and "checkpoint" in result["error"] and "max_model_len" in result["error"]


def test_a_changed_patch_is_the_change_under_test_not_a_refusal(tmp_path):
    with server(tmp_path) as url:
        baseline = record(tmp_path, url)["fixture"]
        result = compare(tmp_path, url, baseline, "--patch", "lab/vllm-gfx1201.diff")
    assert result["ok"] is True and result["condition_drift"] is False
    assert list(result["under_test"]["patch_hash"]["run"]) == ["vllm-gfx1201.diff"]
    assert result["under_test"]["patch_hash"]["baseline"] == {}


def test_a_condition_the_run_does_not_know_is_unverified_not_a_difference(tmp_path):
    bare = list(FLAGS)
    i = bare.index("--engine-commit")
    del bare[i:i + 2]
    with server(tmp_path) as url:
        baseline = record(tmp_path, url)["fixture"]
        result = compare(tmp_path, url, baseline, flags=bare)
    assert result["ok"] is True and result["unverified"] == ["engine_commit"]


def test_extra_conditions_are_recorded_and_must_match_when_both_sides_know_them(tmp_path):
    with server(tmp_path) as url:
        baseline = record(tmp_path, url, "--set", "enforce_eager=true")["fixture"]
        assert json.loads(Path(baseline).read_text(encoding="utf-8"))["conditions"]["enforce_eager"] is True
        assert compare(tmp_path, url, baseline, "--set", "enforce_eager=true")["ok"] is True
        refused = compare(tmp_path, url, baseline, "--set", "enforce_eager=false")
    assert refused["outcome"] == "refused" and "enforce_eager" in refused["error"]


def test_compare_refuses_a_file_that_is_not_a_baseline(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    with server(tmp_path) as url:
        assert "not a lab.correctness.baseline/1 fixture" in compare(tmp_path, url, junk)["error"]
        assert "cannot read the baseline" in compare(tmp_path, url, tmp_path / "missing.json")["error"]


# --- the pieces -------------------------------------------------------------------------------------------------

def test_first_divergence_is_the_first_differing_position_or_where_one_output_ends():
    assert correctness.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert correctness.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert correctness.first_divergence([1, 2, 3], [1, 2]) == 2
    assert correctness.first_divergence([], [7]) == 0


def test_normalised_text_ignores_whitespace_runs_and_unicode_form_but_not_case():
    assert correctness.normalise("  a \n b\t c ") == "a b c"
    assert correctness.normalise("café") == correctness.normalise("café")
    assert correctness.normalise("Tidy") != correctness.normalise("tidy")


def test_a_server_that_sends_no_logprobs_is_recorded_by_its_chunks():
    class Plain:
        def events(self, body):
            for word in ("a ", "b ", "c "):
                yield 0.0, json.dumps({"choices": [{"delta": {"content": word}, "finish_reason": None}]})
            yield 0.0, json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
            yield 0.0, "[DONE]"

    prompts = bench.PromptSet.load()
    out = correctness.generate(Plain(), prompts, bench.Settings(model="teacher", prefix_caching=False), 0)
    assert out["token_kind"] == "chunk_text" and out["tokens"] == ["a ", "b ", "c "] and out["text"] == "a b c "
    assert out["logprobs"] == [None] * 3 and out["completion_tokens"] == 3 and out["error"] is None


def test_a_llama_server_token_id_beside_the_token_text_is_recorded_as_a_token_id():
    # llama-server ignores return_tokens_as_token_ids and sends logprobs only for top_logprobs >= 1, with an "id".
    seen = []

    class Llama:
        def events(self, body):
            seen.append(body)
            for tid, word in ((5170, "First"), (11, ",")):
                entry = {"id": tid, "token": word, "bytes": list(word.encode()), "logprob": -0.5, "top_logprobs": []}
                yield 0.0, json.dumps({"choices": [{"delta": {"content": word}, "finish_reason": None,
                                                    "logprobs": {"content": [entry]}}]})
            yield 0.0, json.dumps({"choices": [], "usage": {"prompt_tokens": 80, "completion_tokens": 2}})
            yield 0.0, "[DONE]"

    prompts = bench.PromptSet.load()
    out = correctness.generate(Llama(), prompts, bench.Settings(model="teacher", prefix_caching=False), 0,
                               top_logprobs=1)
    assert seen[0]["top_logprobs"] == 1 and seen[0]["logprobs"] is True
    assert out["token_kind"] == "token_id" and out["tokens"] == [5170, 11] and out["logprobs"] == [-0.5, -0.5]
    assert out["text"] == "First," and out["error"] is None


def test_top_logprobs_stays_zero_unless_asked(tmp_path, monkeypatch):
    seen = []
    real = bench.DirectEndpoint.events

    def spy(self, body):
        seen.append(body)
        return real(self, body)

    monkeypatch.setattr(bench.DirectEndpoint, "events", spy)
    with server(tmp_path) as url:
        record(tmp_path, url)
        compare(tmp_path, url, tmp_path / "fixtures" / f"vllm-{CHIP}-2026-10-10.json", "--top-logprobs", "1")
    assert [b["top_logprobs"] for b in seen] == [0] * 4 + [1] * 4


def test_the_verb_is_registered_once_and_listed():
    assert lab_main.VERBS["correctness"] == "lab.correctness:main"
    assert list(lab_main.VERBS) == sorted(lab_main.VERBS)
    assert "correctness" in lab_main.help_text()
