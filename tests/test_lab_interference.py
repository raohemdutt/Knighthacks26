"""Tests for the interference measurement (lab/interference.py, python -m lab interference).

The analysis runs on a hand-made raw record (fixtures/lab/interference/synthetic_raw.json, numbers chosen so every
rate can be worked out by hand); the pod run runs end to end against lab/bench_fake.py's server and power and
lab/train_fake.py's trainer, so the server spawn, the decode rounds, the trainer's files and the stop are real
processes and only the GPU is pretend. No pod, no hub and no GPU is reached; the ledger goes to a private state dir.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import types
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
    # each run gets one batch's worth of the bundled rows, so every step is the same 4 examples: 20 measured steps
    # see 80, after 2 warm-up steps that are run and left out
    assert (answer["slot"]["train_rows"], answer["slot"]["warm_steps"], answer["slot"]["train_examples"]) == (4, 2, 80)
    assert answer["slot"]["parts_s"]["train_arm_s"] == 395                # 120 s to load, 88 examples at 0.32 a second
    assert any("~/venvs/atlas-train/bin/python" in step and "installs nothing" in step for step in answer["order"])
    train = next(step for step in answer["order"] if step.startswith("train arm"))
    assert "--max-steps 22" in train and "--no-install" in train and "the first 4 rows" in train
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
    # both: four 12.5 s steps, a 50 s window. Four 15 s rounds of four answers touch it: two whole, one with its
    # last 5 s inside (a third of its answers) and one with its first 10 s inside (two thirds): 4/3 + 8 + 8/3 = 12
    # answers and 1200 tokens over the 45 s the rounds spent inside.
    assert b["train"]["window"][1] - b["train"]["window"][0] == 50.0
    assert (b["decode"]["rounds"], b["decode"]["rounds_cut"], b["decode"]["busy_s"]) == (2, 2, 45.0)
    assert (b["decode"]["answers"], b["decode"]["tokens"], b["decode"]["decode_tok_s"]) == (12, 1200, 26.6667)
    assert (b["joules_per_answer_all"], b["joules_per_step_all"]) == (458.3333, 1375.0)


def test_a_round_the_window_cuts_counts_by_the_part_of_it_inside():
    def one(start, end, tokens=(100, 100)):
        return {"start": start, "end": end, "requests": [{"tokens": n, "decode_tok_s": 5.0, "error": None}
                                                         for n in tokens]}

    rounds = [one(0.0, 20.0), one(20.0, 40.0), one(40.0, 60.0), one(60.0, 80.0)]
    alone = itf.decode_summary(rounds)
    assert (alone["rounds"], alone["rounds_cut"], alone["answers"], alone["tokens"]) == (4, 0, 8, 800)
    assert (alone["decode_tok_s"], alone["window"]) == (10.0, [0.0, 80.0])
    # the window opens a quarter into the first round and closes halfway through the third; the fourth is outside
    cut = itf.decode_summary(rounds, (5.0, 50.0))
    assert (cut["rounds"], cut["rounds_cut"], cut["busy_s"], cut["window"]) == (1, 2, 45.0, [5.0, 50.0])
    assert (cut["answers"], cut["tokens"]) == (4.5, 450)                  # 2 x (0.75 + 1 + 0.5), 200 x the same
    assert (cut["decode_tok_s"], cut["answers_per_s"]) == (10.0, 0.1)     # the rate is the rounds' own
    # a failed request is an error and gives no answer; a round with no length is left out
    broken = [one(0.0, 20.0), {"start": 20.0, "end": 40.0, "requests": [{"tokens": 0, "error": "timeout"},
                                                                        {"tokens": 100, "error": None}]},
              one(40.0, 40.0)]
    part = itf.decode_summary(broken, (10.0, 30.0))
    assert (part["answers"], part["tokens"], part["errors"], part["rounds"]) == (1.5, 150, 1, 0)
    assert itf.decode_summary(rounds, (100.0, 200.0))["answers"] == 0


def test_a_trainer_run_s_warm_up_steps_are_recorded_and_left_out_of_its_window():
    plain = itf.build_result(raw())["arms"]
    record = raw()
    for name, first_end, slow_ms in (("train", "2026-10-11T06:33:20Z", 45000.0),
                                     ("both", "2026-10-11T06:50:00Z", 60000.0)):
        arm = record["arms"][name]
        arm["warm_steps"] = 1                      # one slow step before the four of the fixture
        arm["steps"] = [{"ts": first_end, "step": 0, "step_ms": slow_ms, "loss": 2.0, "tokens": 512}, *arm["steps"]]
    arms = itf.build_result(record)["arms"]
    for name, got, want in (("train", arms["train"], plain["train"]), ("both", arms["both"]["train"],
                                                                      plain["both"]["train"])):
        assert got["warm_steps"] == 1 and want["warm_steps"] == 0, name
        for key in ("steps", "window", "seconds", "samples", "samples_per_s", "steps_per_s"):
            assert got[key] == want[key], (name, key)
    assert arms["both"]["decode"] == plain["both"]["decode"]              # the rounds are cut by the same window
    # a run that died inside its warm-up has no measured step
    record["arms"]["train"]["warm_steps"] = 9
    dead = itf.build_result(record)
    assert dead["arms"]["train"]["steps"] == 0 and "the train arm has no step" in itf.check_result(dead)
    assert itf.slot_estimate(steps=20, train_batch=4, grad_accum=1, rows=4, warm_steps=2)["train_examples"] == 80
    assert itf.slot_estimate(steps=20, train_batch=4, grad_accum=1, rows=6, warm_steps=0)["train_examples"] == 60


def test_the_examples_a_trainer_saw_are_counted_epoch_by_epoch_not_multiplied_out():
    assert itf.examples_seen(18, 4, 4, 84) == 252          # L16: 18 steps over 84 rows at 4 x 4; the product says 288
    assert itf.examples_seen(20, 4, 1, 6) == 60            # the bundled rows: ten epochs of 4 + 2
    assert itf.examples_seen(3, 4, 1, 6) == 10             # 4 + 2, then the next epoch's first 4
    assert itf.examples_seen(20, 2, 1, 6) == 40            # the batch divides the rows: every step is full
    assert itf.examples_seen(5, 4, 4, 84) == 80            # inside the first epoch every step is four full batches
    assert itf.examples_seen(20, 4, 1) == 80 and itf.examples_seen(20, 4, 1, 0) == 80     # no row count: the product
    assert itf.examples_seen(0, 4, 1, 6) == 0
    record = raw()
    record["arms"]["train"]["rows"] = 6                    # four steps over six rows: 4 + 2 + 4 + 2
    train = itf.build_result(record)["arms"]["train"]
    assert (train["rows"], train["samples"], train["samples_per_s"]) == (6, 12, 0.3)


def test_shared_work_is_compared_with_the_same_work_one_after_the_other():
    compare = itf.build_result(raw())["compare"]
    # one after the other: 12 answers at 250 J and 2.5 s each, 4 steps at 1000 J and 10 s each
    assert compare == {"decode_tok_s_change_pct": -33.33, "answers_per_s_change_pct": -33.33,
                       "samples_per_s_change_pct": -20.0, "serial_energy_j": 7000.0, "shared_energy_j": 5500.0,
                       "energy_change_pct": -21.43, "serial_seconds": 70.0, "shared_seconds": 50.0,
                       "time_change_pct": -28.57, "verdict": "share"}


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
    assert "| both | 26.67 | 0.267 | 0.320 | 0.0800 | 110.0 | 458.3 (all) | 1375.0 (all) |" in text
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
                                  "interference.time_change_pct": -28.57, "interference.energy_change_pct": -21.43}
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
    here = (tmp_path / "run.sh").resolve().as_posix()
    saved = (tmp_path / "run.out").resolve().as_posix()
    assert answer["next"] == f"python tooling/dispatcher/pod_exec.py D run {here} r43_pod_run.sh 2700 > {saved}"
    assert answer["then"] == f"python -m lab interference collect {saved}"
    script = (tmp_path / "run.sh").read_text(encoding="ascii")
    assert "--python '~/venvs/atlas-train/bin/python'" in script          # the pod's training venv, never python3
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
    assert len(raw_record["arms"]["train"]["steps"]) == 8 and raw_record["arms"]["train"]["exit_code"] == 0
    assert raw_record["arms"]["train"]["warm_steps"] == 2                   # two steps before the six measured
    assert raw_record["workload"]["train_rows"] == 4 and raw_record["arms"]["both"]["rows"] == 4
    assert raw_record["arms"]["both"]["rounds"]                             # decode ran while it trained
    assert (REPO_ROOT / itf.DEFAULT_DATASET / "split.json").is_file()       # the trainer deleted only its copy
    assert not (run.dir / "train" / "data" / "tool-calling-tiny").exists()
    result = json.loads((run.dir / "result.json").read_text(encoding="utf-8"))
    assert itf.check_result(result) == []
    assert result["arms"]["decode"]["answers"] > 0 and result["arms"]["decode"]["watts"]["available"] is True
    assert result["arms"]["train"]["steps"] == 6 and result["arms"]["train"]["joules_per_step"] > 0
    assert result["arms"]["train"]["samples"] == 24                         # six steps of the same four rows
    assert result["arms"]["both"]["train"]["steps"] == 6
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


# --- the trainer: its Python, its environment, its end -------------------------------------------------------------

def pod_args(tmp_path, *extra: str):
    return itf.build_parser().parse_args([
        "pod-run", "--expect-user-sha256", "unused-here", "--port", str(free_port()), "--out", str(tmp_path / "out"),
        "--dataset", str(REPO_ROOT / itf.DEFAULT_DATASET), "--rows", str(REPO_ROOT / itf.DEFAULT_ROWS), *extra])


SLEEPER = ("import json, os, sys, time, pathlib; out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True); "
           "(out / 'env.json').write_text(json.dumps({k: os.environ.get(k) for k in "
           "('PYTHONNOUSERSITE', 'PYTHONUNBUFFERED', 'LAB_STATE_DIR', 'LAB_TASK')})); time.sleep(120)")


def wait_for(path: Path, seconds: float = 20.0) -> None:
    until = time.time() + seconds
    while not path.is_file() and time.time() < until:
        time.sleep(0.05)
    assert path.is_file(), f"{path} did not appear"


def test_the_trainer_runs_from_the_training_venv_and_never_installs(tmp_path):
    run = itf.PodRun(pod_args(tmp_path), power=bench_fake.FakePower(tmp_path))
    arm = run.dir / "train"
    argv = run.train_argv(arm, arm / "data" / "tool-calling-tiny", arm / "data" / "rows.jsonl")
    assert argv[0] == str(Path("~/venvs/atlas-train/bin/python").expanduser()) and argv[0] != "python3"
    assert argv[1:3] == ["-u", "lab/train.py"] and argv[-1] == "--no-install"
    assert argv[argv.index("--max-steps") + 1] == "22" and argv[argv.index("--batch") + 1] == "4"   # 2 warm + 20
    assert argv[argv.index("--model") + 1] == str(Path("~/models/Qwen3.5-4B").expanduser())
    assert itf.TRAIN_ENV["PYTHONNOUSERSITE"] == "1"
    from lab import train

    assert "--no-install" in train._parser().format_help()                 # the real trainer takes the flag


def test_the_trainer_gets_its_environment_and_is_stopped_when_the_run_ends_under_it(tmp_path):
    args = pod_args(tmp_path, "--train-cmd", json.dumps([sys.executable, "-c", SLEEPER, "{out}"]))
    run = itf.PodRun(args, power=bench_fake.FakePower(tmp_path))
    proc, arm_dir, log = run.start_train("train")
    try:
        wait_for(arm_dir / "run" / "env.json")
        env = json.loads((arm_dir / "run" / "env.json").read_text(encoding="utf-8"))
        assert env == {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "LAB_STATE_DIR": str(arm_dir / "state"),
                       "LAB_TASK": "R43"}
        # its rows are one batch's worth from the top of the rows file, so every step is the same batch
        source = (REPO_ROOT / itf.DEFAULT_ROWS).read_text(encoding="utf-8").splitlines()
        assert (arm_dir / "data" / "rows.jsonl").read_text(encoding="utf-8").splitlines() == source[:4]
        assert len(source) == 6 and run.train_rows == 4
        assert (arm_dir / "data" / "tool-calling-tiny" / "split.json").is_file()
        assert proc.poll() is None
        run.stop_trainer(timeout_s=20.0)
        assert proc.poll() is not None                                     # nothing of the run stays on the chip
        assert run.stop_trainer() == proc.poll()                           # a second call is harmless
    finally:
        log.close()
        if proc.poll() is None:
            proc.kill()


def test_the_rows_a_run_gets_can_be_set_and_are_never_more_than_the_file_holds(tmp_path):
    def rows(*extra: str) -> int:
        return itf.PodRun(pod_args(tmp_path, *extra), power=bench_fake.FakePower(tmp_path)).train_rows

    assert rows() == 4 and rows("--train-rows", "6") == 6 and rows("--train-rows", "50") == 6
    assert rows("--train-batch", "2") == 2 and rows("--train-batch", "2", "--grad-accum", "2") == 4


def test_a_decode_round_that_fails_in_the_both_arm_does_not_leave_the_trainer_running(tmp_path):
    args = pod_args(tmp_path, "--train-cmd", json.dumps([sys.executable, "-c", SLEEPER, "{out}"]))
    run = itf.PodRun(args, power=bench_fake.FakePower(tmp_path))

    def failing(endpoint, tag, stop):
        wait_for(run.dir / "both" / "run" / "env.json")
        raise RuntimeError("the server went away")

    run.decode_until = failing
    with pytest.raises(RuntimeError, match="went away"):
        run.both_arm(endpoint=None)
    assert run.trainer is not None and run.trainer.poll() is not None


def test_before_the_server_starts_the_run_checks_the_venv_the_pins_and_the_folders(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    args = pod_args(tmp_path, "--checkpoint", str(tmp_path / "teacher"), "--model", str(tmp_path / "student"),
                    "--python", str(python))
    calls = []

    def answering(stdout="[]", returncode=0, stderr=""):
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)
        return run

    # nothing there: every missing piece is named, and nothing is asked of a Python that does not exist
    problems = itf.preflight(args, run=answering())
    assert len(problems) == 3 and calls == []
    assert "no teacher checkpoint" in problems[0] and "no student model folder" in problems[1]
    assert "no trainer Python" in problems[2] and "installs nothing" in problems[2]
    (tmp_path / "teacher").mkdir()
    (tmp_path / "student").mkdir()
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    # the venv holds every pin: nothing in the way; the question went to that Python without the user site
    assert itf.preflight(args, run=answering("[]\n")) == []
    argv, kwargs = calls[-1]
    assert argv == [str(python), "-c", itf.PINS_CHECK]
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1" and kwargs["timeout"] == itf.PREFLIGHT_TIMEOUT_S
    # a pin is missing: refused, because a slot never runs pip
    problems = itf.preflight(args, run=answering('some warning\n["trl==1.15.0", "peft==0.21.2"]\n'))
    assert len(problems) == 1 and "trl==1.15.0, peft==0.21.2" in problems[0] and "installs nothing" in problems[0]
    # the Python cannot answer
    problems = itf.preflight(args, run=answering("", 1, "ModuleNotFoundError: No module named 'lab'"))
    assert len(problems) == 1 and "exit 1" in problems[0] and "No module named 'lab'" in problems[0]

    def hanging(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    assert "TimeoutExpired" in itf.preflight(args, run=hanging)[0]
    # a run with a stand-in trainer (the tests) asks nothing of a venv; a missing rows file is still named
    calls.clear()
    stand_in = pod_args(tmp_path, "--checkpoint", str(tmp_path / "teacher"), "--train-cmd", '["x"]',
                        "--rows", str(tmp_path / "no-rows.jsonl"))
    assert itf.preflight(stand_in, run=answering()) == [f"no rows file at {tmp_path / 'no-rows.jsonl'}"]
    assert calls == []


def test_the_question_preflight_asks_is_one_the_real_trainer_module_answers():
    done = subprocess.run([sys.executable, "-c", itf.PINS_CHECK], capture_output=True, text=True, cwd=str(REPO_ROOT),
                          env={**os.environ, **itf.TRAIN_ENV}, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    missing = json.loads(done.stdout.strip().splitlines()[-1])
    assert isinstance(missing, list) and all("==" in pin for pin in missing)


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


def test_the_research_note_states_the_plan_s_rules_with_the_data_file_s_format_and_is_clean():
    from lab.machine import research_check

    note = REPO_ROOT / "lab" / "refine" / "research" / "plumbing" / "placement.md"
    text = note.read_text(encoding="utf-8")
    assert "## The interference run and how the plan reads it" in text
    assert f"{itf.RESULT_PREFIX}-<date>.json" in text and f"schema `{itf.SCHEMA}`" in text
    for field in ("decode_tok_s_change_pct", "samples_per_s_change_pct", "time_change_pct", "energy_change_pct"):
        assert field in text
    assert research_check.main([str(note)])["ok"] is True
