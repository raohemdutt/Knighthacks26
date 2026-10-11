"""Tests for the interference measurement (lab/interference.py, python -m lab interference).

The analysis runs on a hand-made raw record (fixtures/lab/interference/synthetic_raw.json, numbers chosen so every
rate can be worked out by hand); the pod run runs end to end against lab/bench_fake.py's server and power and
lab/train_fake.py's trainer, so the server spawn, the decode rounds, the trainer's files and the stop are real
processes and only the GPU is pretend. No pod, no hub and no GPU is reached; the ledger goes to a private state dir.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import bench, bench_fake
from lab import interference as itf
from lab.machine import ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX = REPO_ROOT / "fixtures" / "lab" / "interference"
RAW = FIX / "synthetic_raw.json"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "test")
    monkeypatch.setenv("LAB_ENV_FILES", "")


def raw() -> dict:
    return json.loads(RAW.read_text(encoding="utf-8"))


# --- the plan and the slot ----------------------------------------------------------------------------------------

def test_the_dry_run_plans_three_arms_in_one_30_minute_slot_at_the_share_that_leaves_room_for_the_lora():
    answer = itf.main(["--dry-run"])
    assert answer["ok"] is True and answer["outcome"] == "plan"
    assert answer["slot"]["slot_minutes"] == 30 and answer["slot"]["minutes"] <= 30
    assert answer["slot"]["parts_s"]["server_start_s"] == 523            # pod D's cold start
    assert answer["slot"]["train_examples"] == 80
    assert answer["memory"] == {"share": 0.63, "visible_gib": 64.0, "free_gib": 58.0, "teacher_gib": 40.32,
                                "lora_peak_gib": 11.1, "left_gib": 6.58, "fits": True}
    assert any(step.startswith("decode arm") for step in answer["order"])
    assert "--minutes 30" in answer["slot_request"]


def test_at_the_teacher_s_usual_share_the_lora_does_not_fit_and_the_plan_says_so():
    answer = itf.main(["--share", "0.9"])
    assert answer["ok"] is False and "leaves no room" in answer["error"]
    assert itf.memory_check(0.9)["left_gib"] < 0


def test_python_m_lab_interference_is_a_verb(capsys):
    assert "interference" in lab_cli.VERBS
    assert lab_cli.main(["interference", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "plan"


# --- the analysis -------------------------------------------------------------------------------------------------

def test_each_arm_s_rates_watts_and_joules_come_from_its_own_window():
    result = itf.build_result(raw())
    d, t, b = result["arms"]["decode"], result["arms"]["train"], result["arms"]["both"]
    # decode: two rounds of four 100-token answers in 20 s at 2000 J
    assert (d["answers"], d["tokens"], d["decode_tok_s"], d["answers_per_s"]) == (8, 800, 40.0, 0.4)
    assert d["joules_per_answer"] == 250.0
    # train: four 10 s steps of 4 examples, 4000 J
    assert (t["steps"], t["samples"], t["samples_per_s"], t["steps_per_s"]) == (4, 16, 0.4, 0.1)
    assert t["joules_per_step"] == 1000.0 and t["trainer_samples_per_s"] == 0.38
    # both: four 12.5 s steps; of four rounds only the two wholly inside the steps' window count
    assert b["train"]["window"][1] - b["train"]["window"][0] == 50.0
    assert (b["decode"]["rounds"], b["decode"]["answers"], b["decode"]["decode_tok_s"]) == (2, 8, 26.6667)
    assert (b["joules_per_answer_all"], b["joules_per_step_all"]) == (687.5, 1375.0)


def test_shared_work_is_compared_with_the_same_work_one_after_the_other():
    compare = itf.build_result(raw())["compare"]
    assert compare == {"decode_tok_s_change_pct": -33.33, "answers_per_s_change_pct": -33.33,
                       "samples_per_s_change_pct": -20.0, "serial_energy_j": 6000.0, "shared_energy_j": 5500.0,
                       "energy_change_pct": -8.33, "serial_seconds": 60.0, "shared_seconds": 50.0,
                       "time_change_pct": -16.67, "verdict": "share"}


def test_sharing_that_costs_time_or_energy_is_a_serial_verdict_and_a_missing_number_unknown():
    assert itf.verdict({"time_change_pct": 5.0, "energy_change_pct": -1.0}) == "serial"
    assert itf.verdict({"time_change_pct": -5.0, "energy_change_pct": 2.0}) == "serial"
    assert itf.verdict({"time_change_pct": None, "energy_change_pct": -2.0}) == "unknown"


def test_a_failed_arm_and_missing_watts_are_open_items_and_fail_the_checks():
    record = raw()
    record["arms"]["train"] = {"error": "TrainError: refused"}
    record["arms"]["decode"]["watts"] = {"available": False, "reason": "no power reader on this host"}
    result = itf.build_result(record)
    assert any(item.startswith("train arm: TrainError") for item in result["open"])
    assert any("no power reader" in item for item in result["open"])
    assert result["compare"]["verdict"] == "unknown"
    assert "the train arm has no step" in itf.check_result(result)
    assert itf.check_result(itf.build_result(raw())) == []


def test_the_committed_result_fixture_is_what_the_analysis_makes_of_the_raw_fixture():
    committed = json.loads((FIX / "interference-gfx1151-2026-10-11.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(itf.build_result(raw())))


def test_the_markdown_table_has_a_row_per_arm_and_the_comparison(tmp_path):
    result = itf.build_result(raw())
    json_path, md_path = itf.write_outputs(result, tmp_path)
    assert json_path.name == "interference-gfx1151-2026-10-11.json"
    text = md_path.read_text(encoding="utf-8")
    assert "| decode alone | 40.00 | 0.400 | - | - | 100.0 | 250.0 | - |" in text
    assert "| train alone | - | - | 0.400 | 0.1000 | 100.0 | - | 1000.0 |" in text
    assert "| both | 26.67 | 0.267 | 0.320 | 0.0800 | 110.0 | 687.5 (all) | 1375.0 (all) |" in text
    assert "| verdict | share |" in text
    with pytest.raises(itf.InterferenceError):
        itf.write_outputs(result, tmp_path)
    assert itf.newest_result(tmp_path) == json_path


def test_the_ledger_rows_are_one_per_arm_and_a_summary_on_pod_b():
    rows = itf.ledger_rows(itf.build_result(raw()), "lab/refine/research/kernels/data/x.json")
    assert [r["verb"] for r in rows] == ["interference.arm"] * 3 + ["interference.summary"]
    assert [r["conditions"].get("arm") for r in rows[:3]] == ["decode", "train", "both"]
    assert all(r["node"] == "pod-b" and r["low_confidence"] and r["outcome"] == "ok" for r in rows)
    assert rows[0]["metrics"]["interference.joules_per_answer"] == 250.0
    assert rows[1]["metrics"]["interference.joules_per_step"] == 1000.0
    assert rows[3]["metrics"] == {"interference.decode_tok_s_change_pct": -33.33,
                                  "interference.samples_per_s_change_pct": -20.0,
                                  "interference.time_change_pct": -16.67, "interference.energy_change_pct": -8.33}
    assert rows[3]["conditions"]["verdict"] == "share" and rows[3]["units"]["interference.time_change_pct"] == "%"


# --- the bundle and the script ------------------------------------------------------------------------------------

def test_the_result_block_survives_the_terminal_and_a_damaged_one_is_refused():
    files = {"result.json": b"{}", "raw.json": b"[]"}
    text = "prompt$ noise\r\n" + itf.bundle_text(itf.make_bundle(files)) + "### END rc=0\n"
    assert itf.read_bundle(text.encode()) == files
    lines = text.splitlines()
    lines[2] = lines[2][:-4] + "AAAA"
    with pytest.raises(itf.InterferenceError, match="damaged"):
        itf.read_bundle("\n".join(lines).encode())
    with pytest.raises(itf.InterferenceError, match="did not finish"):
        itf.read_bundle(b"nothing here")


def test_the_pod_script_carries_the_trainer_the_bench_and_the_rows_and_types_safely(tmp_path):
    answer = itf.main(["pod-script", "--expect-user-sha256", "ab" * 32, "--out", str(tmp_path / "run.sh")])
    assert answer["ok"] is True and answer["slot_minutes"] == 30
    assert answer["next"] == "python tooling/dispatcher/pod_exec.py D run <this script> r43_pod_run.sh 2700 > <output file>"
    script = (tmp_path / "run.sh").read_text(encoding="ascii")
    assert "\t" not in script and "!" not in script
    assert all(len(line) <= 76 for line in script.split("R43BUNDLE")[1].splitlines()[1:-1])
    assert f"python3 -m lab.interference pod-run --expect-user-sha256 {'ab' * 32} --share 0.63" in script
    assert script.rstrip().endswith("exit $rc") and "### END rc=$rc" in script
    names = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in itf.module_closure()}
    assert {"lab/interference.py", "lab/train.py", "lab/bench.py", "lab/pod_power.py", "lab/profile_decode_pod.py",
            "lab/dataset.py", "contracts/telemetry.py"} <= names
    assert "lab/placement.py" not in names
    data = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in itf.data_files()}
    assert {"fixtures/lab/bench/prompts.json", "lab/requirements-train.txt", "fixtures/learning/train-tiny/rows.jsonl",
            "fixtures/learning/train-tiny/tool-calling-tiny/split.json"} <= data


def test_the_pod_script_refuses_a_share_that_leaves_no_room():
    answer = itf.main(["pod-script", "--expect-user-sha256", "ab" * 32, "--share", "0.8", "--out", "unused.sh"])
    assert answer["ok"] is False and "above 0.63" in answer["error"]


def test_off_the_pod_the_run_is_refused_before_anything_starts(tmp_path, monkeypatch):
    monkeypatch.delenv("JUPYTERHUB_USER", raising=False)
    answer = itf.main(["pod-run", "--expect-user-sha256", "ab" * 32, "--out", str(tmp_path / "out"),
                       "--port", str(free_port())])
    assert answer["ok"] is False and answer["outcome"] == "refused"
    assert not (tmp_path / "out").exists()


def test_the_serve_line_gets_the_share_in_place_of_the_teacher_s_usual_one():
    argv = itf.serve_argv(bench.default_command("vllm", "gfx1151"), 0.6)
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.6" and "0.90" not in argv
    assert itf.serve_argv(["llama-server"], 0.5)[-2:] == ["--gpu-memory-utilization", "0.5"]


# --- a whole run on the fake pod, then collect --------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_whole_run_on_the_fake_pod_then_collect_writes_the_data_file_the_table_and_the_ledger(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    trainer = [bench_fake.spawn_python(), str(REPO_ROOT / "lab" / "train_fake.py"), "--dataset", "{dataset}",
               "--rows", "{rows}", "--model", "{model}", "--max-steps", "{steps}", "--batch", "{batch}",
               "--grad-accum", "{grad_accum}", "--out", "{out}", "--run-id", "{run_id}", "--step-s", "0.3"]
    args = itf.build_parser().parse_args([
        "pod-run", "--expect-user-sha256", "unused-here", "--port", str(free_port()), "--out", str(tmp_path / "out"),
        "--serve-dir", str(tmp_path / "serve"), "--max-tokens", "8", "--decode-s", "1.5", "--steps", "6",
        "--dataset", str(REPO_ROOT / itf.DEFAULT_DATASET), "--rows", str(REPO_ROOT / itf.DEFAULT_ROWS),
        "--serve-cmd", json.dumps(bench_fake.serve_command(root=str(root), stream_tok_s=200)),
        "--train-cmd", json.dumps(trainer)])
    run = itf.PodRun(args, power=bench_fake.FakePower(root))
    answer = itf.run_and_report(run)
    assert answer["ok"] is True, answer
    assert not bench.port_open(args.port)                                   # the server is gone
    raw_record = json.loads((run.dir / "raw.json").read_text(encoding="utf-8"))
    assert raw_record["serve"]["share"] == 0.63
    assert len(raw_record["arms"]["train"]["steps"]) == 6 and raw_record["arms"]["train"]["exit_code"] == 0
    assert raw_record["arms"]["both"]["rounds"]                             # decode ran while it trained
    assert (REPO_ROOT / itf.DEFAULT_DATASET / "split.json").is_file()       # the trainer deleted only its copy
    assert not (run.dir / "train" / "data" / "tool-calling-tiny").exists()
    result = json.loads((run.dir / "result.json").read_text(encoding="utf-8"))
    assert itf.check_result(result) == []
    assert result["arms"]["decode"]["answers"] > 0 and result["arms"]["decode"]["watts"]["available"] is True
    assert result["arms"]["train"]["steps"] == 6 and result["arms"]["train"]["joules_per_step"] > 0
    assert result["arms"]["both"]["decode"]["answers"] > 0
    assert result["compare"]["verdict"] in ("share", "serial")

    saved = tmp_path / "pod_output.txt"
    saved.write_text(capsys.readouterr().out, encoding="utf-8")
    collected = itf.main(["collect", str(saved), "--out", str(tmp_path / "data")])
    assert collected["ok"] is True and len(collected["ledger_seqs"]) == 4
    assert Path(collected["json"]).name == f"interference-gfx1151-{result['date']}.json"
    assert Path(collected["markdown"]).read_text(encoding="utf-8").startswith("# Interference on one gfx1151 chip")
    rows = ledger.read(state_dir=tmp_path / "state")
    assert [r["verb"] for r in rows] == ["interference.arm"] * 3 + ["interference.summary"]
    assert all(r["task"] == "R43" and r["node"] == "pod-b" for r in rows)


def test_the_fake_trainer_takes_the_real_trainer_s_arguments(tmp_path):
    from lab import train, train_fake

    data = tmp_path / "d"
    data.mkdir()
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}\n", encoding="utf-8")
    argv = ["--dataset", str(data), "--rows", str(rows), "--model", "m", "--max-steps", "3", "--batch", "2",
            "--grad-accum", "1", "--out", str(tmp_path / "out"), "--run-id", "r", "--step-s", "0"]
    real = train._parser().parse_args(argv[:-2])
    assert (real.max_steps, real.batch, real.grad_accum) == (3, 2, 1)
    assert train_fake.main(argv) == 0
    assert len(itf.read_steps(tmp_path / "out" / "telemetry.jsonl")) == 3
    assert not data.exists() and not rows.exists()
    assert json.loads((tmp_path / "out" / "run.json").read_text())["outcome"] == "trained"
    assert sys.executable


def test_the_research_note_states_the_plan_s_rules_with_the_data_file_s_format_and_is_clean():
    from lab.machine import research_check

    note = REPO_ROOT / "lab" / "refine" / "research" / "plumbing" / "placement.md"
    text = note.read_text(encoding="utf-8")
    assert "## The interference run and how the plan reads it" in text
    assert f"{itf.RESULT_PREFIX}-<date>.json" in text and f"schema `{itf.SCHEMA}`" in text
    for field in ("decode_tok_s_change_pct", "samples_per_s_change_pct", "time_change_pct", "energy_change_pct"):
        assert field in text
    assert research_check.main([str(note)])["ok"] is True
