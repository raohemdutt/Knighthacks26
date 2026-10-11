"""Tests for python -m lab experiment run (lab/machine/experiment_run.py): the speed-track experiment runner.

Nothing here reaches a pod, the hub or Notion. The pod is lab/hub_fake.py's FakeHub, whose file store is a folder on
disk standing for the pod's Jupyter root; the controller's terminal is replaced by one that runs the typed command
with bash in that folder, so the pod script the runner uploads really runs: it unpacks the harness, runs lab/bench.py
against lab/bench_fake.py's fake server (a real process on 127.0.0.1) and writes the rows the runner pulls. The GPU
queue is lab/machine/slots_fake.py's FakeNotion behind the real client and the real slot verbs. Every test has its own
state dir, no .env file and no NOTION_TOKEN.
"""
from __future__ import annotations

import collections.abc
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import bench_fake, placement
from lab import controller as controller_module
from lab.controller import Controller
from lab.hub import Hub
from lab.hub_fake import FakeClock, FakeHub
from lab.machine import experiment_run, experiments, ledger, slots, slots_fake, stop
from lab.machine.experiment_run import HubPod, Runner
from lab.machine.slots_fake import FakeNotion
from lab.machine.slots_notion import NotionClient
from lab.proxy import PodLink

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 10, 14, 0, 0, tzinfo=UTC)
HUB_TOKEN = "fake-hub-token-d-0123456789abcdef"
BASE = "https://hub.test"
BASH = shutil.which("bash") or "bash"     # on Windows a bare "bash" is System32's WSL launcher, not the one on PATH
HAS_BASH = shutil.which("bash") is not None and shutil.which("base64") is not None and shutil.which("tar") is not None
needs_bash = pytest.mark.skipif(not HAS_BASH, reason="the pod script needs bash, base64 and tar")
SERVER_WORDS = ("pod_a_teacher.sh", "kill", "pkill", "vllm serve", "systemctl", "/server")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    monkeypatch.setenv("LAB_ACTOR", "tester")
    for name in ("NOTION_TOKEN", "LAB_QUEUE_DATA_SOURCE", "LAB_NOTION_URL", "LAB_TIER", "LAB_TASK",
                 "LAB_PLACEMENT_FACTS", "LAB_POD_ROLES"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "state"


class DiskFiles(collections.abc.MutableMapping):
    """FakeHub's file store, (letter, path) -> text, kept as files under one folder per letter: the pod's disk."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        for root in roots.values():
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, key) -> Path:
        letter, path = key
        return self.roots[letter] / path

    def __getitem__(self, key):
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_text(encoding="utf-8", errors="surrogateescape")

    def __setitem__(self, key, value):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", errors="surrogateescape", newline="")

    def __delitem__(self, key):
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        path.unlink()

    def __iter__(self):
        for letter, root in self.roots.items():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    yield letter, path.relative_to(root).as_posix()

    def __len__(self):
        return sum(1 for _ in self)


def snapshot(root: Path) -> list[str]:
    """Every file and folder under the pod's root, which is what "as it was found" compares."""
    return sorted(p.relative_to(root).as_posix() + ("/" if p.is_dir() else "") for p in root.rglob("*"))


class World:
    """A fake hub with pod D (pod-b) on a disk folder, a terminal that runs bash there, the fake GPU queue, and the
    runner wired to them."""

    def __init__(self, tmp_path: Path, monkeypatch, *, letter: str = "D") -> None:
        self.tmp = tmp_path
        self.letter = letter
        self.clock = FakeClock("2026-10-10T14:00:00Z")
        self.hub = FakeHub(self.clock)
        self.pod_root = tmp_path / "pod"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.hub.files = DiskFiles({letter: self.pod_root})
        self.account = self.hub.add_account(letter, user=f"acct-{letter.lower()}", token=HUB_TOKEN,
                                            quota_enabled=False)
        self.hub.start_server(letter)
        self.hub.serve_model(letter, port=8000, models=("teacher",))
        self.pod_root.joinpath("notes.txt").write_text("already on the pod\n", encoding="utf-8")
        self.typed: list[str] = []
        self.terminal_env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.home),
                             "LAB_PYTHON": sys.executable, "LAB_STATE_DIR": str(tmp_path / "pod-ledger-unused")}
        world = self

        def terminal(controller, client, name, command, echo):
            """The pod's shell: run what was typed, in the pod's root, and return once it has finished."""
            world.typed.append(command)
            subprocess.run([BASH, "-c", command], cwd=world.pod_root, env=world.terminal_env, timeout=600,
                           check=False)

        monkeypatch.setattr(controller_module.Controller, "_type_into_terminal", terminal)
        self.notion = FakeNotion(clock=lambda: T0)
        self.notion_client = NotionClient(slots_fake.TOKEN, transport=self.notion.transport, sleep=lambda s: None)
        self.controller = Controller({letter: self.account}, base_url=BASE, transport=self.hub.transport,
                                     sleep=self.clock.sleep, clock=self.clock.time)
        self.link = PodLink(Hub(self.account, base_url=BASE, transport=self.hub.transport),
                            transport=self.hub.transport, sleep=self.clock.sleep, clock=self.clock.time)
        self.pod = HubPod(letter, self.controller, self.link)
        self.runner = Runner(self.pod, slot_client=self.notion_client, now=lambda: T0,
                             environ={"LAB_ACTOR": "tester", "LAB_ENV_FILES": ""})   # no .env above the clone

    def rows(self, verb=None):
        return ledger.read(verb=verb) if verb else ledger.read()

    def slot_rows(self):
        return [self.notion.values(row) for row in self.notion.row_ids()]

    def hub_calls(self, method=None, contains=""):
        return self.hub.calls_matching(method, contains)


@pytest.fixture
def world(tmp_path, monkeypatch):
    built = World(tmp_path, monkeypatch)
    yield built
    built.link.close()
    built.controller.close()


def fake_config(tmp_path: Path, *, batches=(1, 2), minutes=15, serve_cmd=None, **more) -> dict:
    """A run config whose arms are batch sizes of one shared fake spawn per repeat."""
    serve = serve_cmd if serve_cmd is not None else bench_fake.serve_command(engine="vllm")
    return {"bench": ["--engine", "vllm", "--serve-cmd", json.dumps(serve), "--checkpoint",
                      "/home/jovyan/models/Qwen3.8-27B-INT4", "--port", "0", "--batches",
                      ",".join(map(str, batches)), "--no-64", "--share-spawn", "--max-tokens", "8",
                      "--requests-per-stream", "1", "--warmup", "0", "--no-probe", "--idle-s", "0",
                      "--request-timeout-s", "60", "--python", sys.executable],   # a bare python3 on Windows is an installer stub
            "arms": {"A": {"batch": batches[0]}, "B": {"batch": batches[1]}}, "minutes": minutes, **more}


def add_experiment(eid="KX1", **fields):
    definition = {"id": eid, "title": "fake batch pair", "track": "kernels", "hypothesis": "batch 2 decodes more",
                  "lever": "batch size", "arms": {"A": "batch 1", "B": "batch 2"}, "metric": "teacher.decode_tok_s",
                  "better": "higher", "citation": "test", "min_n": 2, "node": "pod-b", **fields}
    return experiments.add(definition)


# --- prepare: the plan and the pod script ------------------------------------------------------------------------

def test_the_baseline_experiment_e3_plans_one_shared_spawn_per_repeat_on_pod_b_split_by_batch_size():
    experiments.seed()
    e3 = experiments.load()["E3"]

    plan = experiment_run.plan_run(e3, experiment_run.RUN_CONFIGS["E3"], run_id="E3-r1", node="pod-b")

    assert list(plan.invocations) == ["i1"]
    args = plan.invocations["i1"]
    assert args[0] == "run" and args[-4:] == ["--repeats", "3", "--node", "pod-b"]
    assert "--share-spawn" in args and args[args.index("--batches") + 1] == "1,4,16"
    assert plan.arms == {"A": {"invocation": "i1", "batch": 1}, "B": {"invocation": "i1", "batch": 4},
                         "C": {"invocation": "i1", "batch": 16}, "D": {"invocation": "i1", "batch": 64}}
    assert (plan.field, plan.prefix, plan.minutes) == ("joules_per_token", "teacher", 60)


def test_arms_with_different_arguments_get_one_invocation_each_and_repeats_can_be_given():
    experiments.seed()
    config = {"bench": ["--batches", "1"], "arms": {"A": {"args": ["--engine", "vllm"]},
                                                    "B": {"args": ["--engine", "llamacpp"]}}}
    plan = experiment_run.plan_run(experiments.load()["E2"], config, run_id="r", node="pod-b", repeats=7)

    assert list(plan.invocations) == ["i1", "i2"]
    assert plan.arms["B"] == {"invocation": "i2", "batch": None}
    assert all(args[-4:] == ["--repeats", "7", "--node", "pod-b"] for args in plan.invocations.values())


@pytest.mark.parametrize("change, problem", [
    ({"track": "telemetry"}, "on the telemetry track"),
    ({"metric": "teacher.samples_per_s"}, "not one the harness measures"),
    ({"node": "jetson"}, "runs on jetson"),
])
def test_an_experiment_the_harness_cannot_run_on_this_pod_is_refused(change, problem):
    experiment = add_experiment(**change)
    with pytest.raises(experiment_run.RunRefused) as caught:
        experiment_run.plan_run(experiment, fake_config(Path(".")), run_id="r", node="pod-b")
    assert problem in str(caught.value)


def test_a_config_that_misses_an_arm_or_sets_the_node_or_none_at_all_is_refused():
    experiment = add_experiment()
    with pytest.raises(experiment_run.RunRefused, match="no run config"):
        experiment_run.plan_run(experiment, None, run_id="r", node="pod-b")
    with pytest.raises(experiment_run.RunRefused, match="no arm B"):
        experiment_run.plan_run(experiment, {"bench": [], "arms": {"A": {}}}, run_id="r", node="pod-b")
    with pytest.raises(experiment_run.RunRefused, match="may not set --node"):
        experiment_run.plan_run(experiment, {"bench": ["--node", "pod-a"], "arms": {"A": {}, "B": {}}},
                                run_id="r", node="pod-b")


@needs_bash
def test_the_pod_script_is_valid_bash_carries_the_harness_and_never_touches_a_server():
    experiments.seed()
    plan = experiment_run.plan_run(experiments.load()["E3"], experiment_run.RUN_CONFIGS["E3"], run_id="E3-r1",
                                   node="pod-b")
    text = experiment_run.pod_script(plan)

    assert "\r" not in text                                         # LF only: the pod is Linux, whatever built the text
    assert subprocess.run([BASH, "-n"], input=text.encode("ascii"), capture_output=True).returncode == 0
    assert "RUN=lab_exp/E3-r1" in text
    assert '"$PY" "$REPO/lab/bench.py" run --engine vllm --checkpoint "$HOME"/models/Qwen3.8-27B-INT4' in text
    assert '--patch "$REPO"/lab/vllm-gfx1201.diff' in text and '--out "$PWD/$RUN/out/i1"' in text
    assert 'LAB_STATE_DIR="$PWD/$RUN/state"' in text
    assert not any(word in text for word in SERVER_WORDS)
    first = experiment_run.bundle()
    time.sleep(1.1)                                                 # a gzip header would carry the time
    assert experiment_run.bundle() == first                         # the same files make the same text
    assert all(ord(ch) < 128 for ch in text)


def test_the_bundle_holds_every_lab_module_the_harness_imports_on_the_pod():
    import re

    bundled = set(experiment_run.BUNDLE_FILES)
    imported = set()
    for name in bundled:
        if name.endswith(".py"):
            text = Path(experiment_run.REPO, name).read_text(encoding="utf-8")
            for package, names in re.findall(r"^\s*from (lab(?:\.\w+)*) import ([\w, ]+)", text, re.MULTILINE):
                if Path(experiment_run.REPO, *package.split(".")).with_suffix(".py").is_file():
                    imported.add(package.replace(".", "/") + ".py")
                    continue
                for name_ in (n.strip() for n in names.split(",")):
                    module = Path(experiment_run.REPO, *package.split("."), name_).with_suffix(".py")
                    if module.is_file():
                        imported.add(module.relative_to(experiment_run.REPO).as_posix())
    assert imported - bundled == set()


@needs_bash
def test_each_harness_call_is_interrupted_before_the_slot_ends_so_it_stops_its_own_server():
    experiments.seed()
    plan = experiment_run.plan_run(experiments.load()["E3"], experiment_run.RUN_CONFIGS["E3"], run_id="E3-r1",
                                   node="pod-b")
    text = experiment_run.pod_script(plan, bundle_text="")

    assert f"DEADLINE=$(( $(date +%s) + {60 * 60 - experiment_run.HARNESS_MARGIN_S} ))" in text
    assert 'timeout -s INT -k 60 "$LEFT" "$PY" "$REPO/lab/bench.py" run' in text
    assert experiment_run.HARNESS_MARGIN_S - experiment_run.HARNESS_KILL_S > experiment_run.SCRIPT_MARGIN_S


@needs_bash
def test_the_cleanup_interrupts_a_harness_of_this_run_still_going_and_no_other_process(tmp_path):
    if shutil.which("pgrep") is None:
        pytest.skip("the cleanup needs pgrep and pkill")
    experiments.seed()
    plan = experiment_run.plan_run(experiments.load()["E3"], experiment_run.RUN_CONFIGS["E3"], run_id="E3-r9",
                                   node="pod-b")
    root = tmp_path / "pod"
    harness = root / plan.run_dir / "repo" / "lab" / "bench.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("import time\ntry:\n    time.sleep(120)\nexcept KeyboardInterrupt:\n    pass\n")
    other = tmp_path / "other" / "lab" / "bench.py"            # another run's harness, which must be left alone
    other.parent.mkdir(parents=True)
    other.write_text(harness.read_text())
    (root / plan.run_dir / "made_root").touch()
    for name in (plan.script_name, plan.plan_name, "lab_run_E3-r9.sh", "lab_run_E3-r9.out"):
        (root / name).write_text("x")
    (root / "notes.txt").write_text("already on the pod")
    mine = subprocess.Popen([sys.executable, str(root / plan.run_dir / "repo" / "lab" / "bench.py")], cwd=root)
    theirs = subprocess.Popen([sys.executable, str(other)])
    try:
        time.sleep(0.5)
        done = subprocess.run([BASH, "-c", experiment_run.cleanup_script(plan, plan.run_id)], cwd=root,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stdout + done.stderr
        assert mine.wait(10) == 0 and theirs.poll() is None
    finally:
        for proc in (mine, theirs):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert snapshot(root) == ["notes.txt"]


def test_slot_arguments_are_checked_before_anything_is_sent():
    experiments.seed()
    plan = experiment_run.plan_run(experiments.load()["E3"], experiment_run.RUN_CONFIGS["E3"], run_id="r",
                                   node="pod-b")
    checked = Runner.check_slot(plan, task=None, row=None, reply=None, why=None, measured=None)
    assert checked["task"] == "T0" and len(checked["why"]) <= slots.MAX_WHY_CHARS

    plan.config.pop("why")
    with pytest.raises(experiment_run.RunRefused, match="needs --why"):
        Runner.check_slot(plan, task=None, row=None, reply=None, why=None, measured=None)
    assert Runner.check_slot(plan, task="O29", row="abc", reply=None, why=None, measured=None)["row"] == "abc"
    with pytest.raises(experiment_run.RunRefused, match="Task Board ID"):
        Runner.check_slot(plan, task="o-29", row="abc", reply=None, why=None, measured=None)
    with pytest.raises(experiment_run.RunRefused, match="--reply"):
        Runner.check_slot(plan, task=None, row="abc", reply="x" * 301, why=None, measured=None)


# --- the dry run against the fakes -------------------------------------------------------------------------------

@needs_bash
def test_a_run_uploads_takes_a_slot_runs_the_harness_pulls_records_releases_and_leaves_the_pod_as_found(world):
    add_experiment()
    before = snapshot(world.pod_root)
    puts_before_request = []
    world.notion.on_request(lambda fake, method, path, body: puts_before_request.append(
        len(world.hub_calls("PUT", "/api/contents/lab_exp_"))) if (method, path) == ("POST", "/v1/pages") else None)

    answer = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r1")

    assert answer["ok"] is True, json.dumps(answer, default=str)[:3000]
    assert answer["outcome"] == "recorded"
    assert answer["steps"] == ["check", "prepare", "pod", "upload", "slot_request", "slot_start", "run", "pull",
                               "ledger", "verdict", "release", "clean"]
    # uploaded before the slot was asked for: the script and its plan
    assert puts_before_request == [2]
    # the slot: requested on GPU B for bench, started, and released with the result line
    (slot,) = world.slot_rows()
    assert (slot["GPU"], slot["Status"], slot["Task ID"]) == ("B exclusive", "Done", "T0")
    assert slot["Request"] == "T0: KX1 bench run: fake batch pair" and slot["Started"] is not None
    assert slot["Result"].startswith("KX1 run KX1-r1: ")
    assert [r["verb"] for r in world.rows() if r["verb"].startswith("slot ")] == ["slot request", "slot start",
                                                                                   "slot done"]
    # the harness ran with the experiment's config: two spawns, each measuring batch 1 and batch 2
    assert len(answer["readings"]["A"]) == 2 and len(answer["readings"]["B"]) == 2
    pulled = Path(answer["work"]) / "results"
    rows = [json.loads(line) for line in (pulled / "i1.rows.jsonl").read_text().splitlines()]
    assert sorted(r["batch_size"] for r in rows) == [1, 1, 2, 2]
    assert {r["node"] for r in rows} == {"pod-b"} and {r["outcome"] for r in rows} == {"ok"}
    assert json.loads((pulled / "i1.answer.json").read_text())["ok"] is True
    # a ledger row per measured run, then the verdict, then the run's own row
    readings = world.rows("bench.run")
    assert len(readings) == 4
    assert all(r["refs"]["experiment"] == "KX1" and r["refs"]["slot"] == answer["row"] for r in readings)
    assert sorted(r["refs"]["arm"] for r in readings) == ["A", "A", "B", "B"]
    assert all(r["node"] == "pod-b" and "teacher.decode_tok_s" in r["metrics"] for r in readings)
    result = experiments.load()["KX1"]["result"]
    assert result["arms"]["A"]["n"] == 2 and result["arms"]["B"]["n"] == 2
    assert world.rows("experiment result")[0]["refs"]["run_id"] == "KX1-r1"
    assert answer["verdict"]["label"] in ("b_ahead", "a_ahead", "no_difference", "too_early")
    (run_row,) = world.rows("experiment run")
    assert (run_row["outcome"], run_row["refs"]["slot"], run_row["refs"]["run_id"]) == ("recorded", answer["row"],
                                                                                       "KX1-r1")
    # the pod as it was found: every file and folder, the teacher still listed, no pod launched or stopped
    assert snapshot(world.pod_root) == before
    assert answer["slot_released"] is True and answer["pod_cleaned"] is True
    assert world.hub._accounts["D"].models.get(8000) is not None
    assert not world.hub_calls("POST", "/server") and not world.hub_calls("DELETE", "/server")
    assert not any(word in command for command in world.typed for word in SERVER_WORDS)


@needs_bash
def test_runs_without_the_experiments_metric_are_still_ledger_rows_and_the_verdict_says_why_there_is_none(world):
    add_experiment(metric="teacher.joules_per_token", better="lower")

    answer = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r11")

    pulled = experiment_run.read_rows(Path(answer["work"]) / "results" / "i1.rows.jsonl")
    if any(row["joules_per_token"] is not None for row in pulled):
        pytest.skip("this machine reads power, so the harness measured joules per token")
    assert (answer["outcome"], answer["failed_step"]) == ("failed", "verdict")
    assert answer["error"] == "no verdict: no run with teacher.joules_per_token for arm(s) A, B (the control is A)"
    readings = world.rows("bench.run")
    assert len(readings) == 4 and all("teacher.decode_tok_s" in r["metrics"] for r in readings)
    assert answer["slot_released"] is True and answer["pod_cleaned"] is True


@needs_bash
def test_a_harness_that_fails_still_releases_the_slot_cleans_the_pod_and_writes_what_failed(world):
    add_experiment()
    before = snapshot(world.pod_root)
    broken = [sys.executable, "-c", "import sys; sys.exit(7)", "{port_file}"]

    answer = world.runner.run("KX1", config=fake_config(world.tmp, serve_cmd=broken), run_id="KX1-r2")

    assert (answer["ok"], answer["outcome"], answer["failed_step"]) == (False, "failed", "run")
    assert answer["slot_released"] is True and answer["pod_cleaned"] is True
    assert snapshot(world.pod_root) == before
    (slot,) = world.slot_rows()
    assert slot["Status"] == "Done" and slot["Result"].startswith("KX1 run KX1-r2 failed at run: ")
    (run_row,) = world.rows("experiment run")
    assert run_row["outcome"] == "failed" and run_row["note"].startswith("run: the pod script exited 1")
    assert run_row["conditions"]["slot_released"] is True and run_row["conditions"]["pod_cleaned"] is True
    assert experiments.load()["KX1"]["result"] is None


@needs_bash
def test_a_pull_that_fails_still_releases_the_slot_and_cleans_the_pod(world):
    add_experiment()
    before = snapshot(world.pod_root)
    world.hub.queue("GET", "/files/lab_exp/KX1-r3/results/i1.rows.jsonl", 500, text="boom", times=50)

    answer = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r3")

    assert (answer["outcome"], answer["failed_step"]) == ("failed", "pull")
    assert "lab_exp/KX1-r3/results/i1.rows.jsonl" in answer["error"]
    assert answer["slot_released"] is True and snapshot(world.pod_root) == before
    assert world.rows("experiment run")[0]["note"].startswith("pull: could not pull")


def test_an_interrupt_during_the_run_still_releases_the_slot_cleans_and_records_then_goes_on(world, monkeypatch):
    add_experiment()
    calls = []
    original = HubPod.run_script

    def interrupted(self, script, *, run_id, timeout_s):
        calls.append(run_id)
        if not run_id.endswith("-clean"):
            raise KeyboardInterrupt
        return original(self, script, run_id=run_id, timeout_s=timeout_s)

    monkeypatch.setattr(HubPod, "run_script", interrupted)
    if not HAS_BASH:
        monkeypatch.setattr(controller_module.Controller, "_type_into_terminal", lambda *a: None)
    before = snapshot(world.pod_root)

    with pytest.raises(KeyboardInterrupt):
        world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r4")

    assert calls == ["KX1-r4", "KX1-r4-clean"]
    (slot,) = world.slot_rows()
    assert slot["Status"] == "Done" and "failed at run: KeyboardInterrupt" in slot["Result"]
    (run_row,) = world.rows("experiment run")
    assert run_row["outcome"] == "failed" and run_row["conditions"]["slot_released"] is True
    if HAS_BASH:
        assert snapshot(world.pod_root) == before


def test_the_kill_switch_mid_run_still_releases_the_slot_and_removes_the_uploads(world, monkeypatch):
    add_experiment()

    def stopped_here(self, script, *, run_id, timeout_s):
        stop.stop_file().parent.mkdir(parents=True, exist_ok=True)
        stop.stop_file().write_text("owner stop", encoding="utf-8")
        return self.controller.run_script(self.letter, script, timeout_s=timeout_s, run_id=run_id)

    monkeypatch.setattr(HubPod, "run_script", stopped_here)
    before = snapshot(world.pod_root)

    with pytest.raises(stop.Stopped):
        world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r5")

    (slot,) = world.slot_rows()
    assert slot["Status"] == "Done"
    assert snapshot(world.pod_root) == before
    assert world.rows("experiment run")[0]["outcome"] == "failed"


def test_a_script_that_never_reports_back_fails_at_run_and_the_cleanup_still_runs(world, monkeypatch):
    add_experiment()
    calls = []

    def timed_out(self, script, *, run_id, timeout_s):
        calls.append((run_id, timeout_s))
        if not run_id.endswith("-clean"):
            return {"ok": False, "outcome": "timeout", "exit_code": None, "error": "no exit code after 780 s"}
        return {"ok": True, "outcome": "ok", "exit_code": 0}

    monkeypatch.setattr(HubPod, "run_script", timed_out)

    answer = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r10")

    assert (answer["outcome"], answer["failed_step"], answer["error"]) == ("failed", "run", "no exit code after 780 s")
    assert calls == [("KX1-r10", 15 * 60 - experiment_run.SCRIPT_MARGIN_S), ("KX1-r10-clean", 300.0)]
    assert answer["slot_released"] is True and "pull" not in answer["steps"]


# --- refusals and waits: nothing left behind --------------------------------------------------------------------

def test_a_pod_the_placement_plan_forbids_is_refused_before_the_queue_or_the_pod_is_asked(tmp_path, monkeypatch):
    built = World(tmp_path, monkeypatch, letter="C")       # account C fills pod-a, the teacher's pod
    add_experiment(node=None)
    try:
        answer = built.runner.run("KX1", config=fake_config(tmp_path), run_id="KX1-r6")
    finally:
        built.link.close()
        built.controller.close()

    assert (answer["ok"], answer["outcome"]) == (False, "placement_forbids")
    assert "the plan puts bench on GPU B (pod-b), not on pod-a" in answer["error"]
    assert built.notion.requests == [] and built.hub.calls == []
    assert built.rows("experiment run")[0]["outcome"] == "placement_forbids"


def test_the_plan_check_is_the_placement_plans_own(world, monkeypatch):
    add_experiment()
    seen = []
    monkeypatch.setattr(placement, "check_request", lambda **kw: seen.append(kw) or {
        "contradicts": True, "verdict": "contradicts", "reason": "the plan has no pod for bench now"})

    answer = world.runner.run("KX1", config=fake_config(world.tmp))

    assert answer["outcome"] == "placement_forbids" and seen[0]["gpu"] == "B" and seen[0]["kind"] == "bench"
    assert world.notion.requests == [] and world.hub.calls == []


def test_a_busy_card_ends_the_call_waiting_with_the_uploads_removed_and_a_rerun_finds_the_same_row(world):
    add_experiment()
    world.notion.add_row("R20: bench matrix", task_id="R20", status="Running", started=T0 - dt.timedelta(minutes=5),
                         requested=T0 - dt.timedelta(minutes=10))
    before = snapshot(world.pod_root)

    first = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r7")
    again = world.runner.run("KX1", config=fake_config(world.tmp), run_id="KX1-r8")

    assert (first["ok"], first["outcome"], first["slot"]["outcome"]) == (False, "waiting", "busy")
    assert first["next"] == f"python -m lab experiment run KX1 --row {first['row']}"
    assert again["row"] == first["row"]
    queued = [row for row in world.slot_rows() if row["Task ID"] == "T0"]
    assert len(queued) == 1 and queued[0]["Status"] == "Queued"
    assert snapshot(world.pod_root) == before and first["pod_cleaned"] is True
    assert [r["outcome"] for r in world.rows("slot request")] == ["queued", "already_queued"]


def test_without_a_notion_token_nothing_runs_and_the_uploads_are_removed(world):
    add_experiment()
    world.runner.slot_client = None
    before = snapshot(world.pod_root)

    answer = world.runner.run("KX1", config=fake_config(world.tmp))

    assert (answer["outcome"], answer["failed_step"]) == ("no_token", "slot_request")
    assert snapshot(world.pod_root) == before
    assert not world.hub_calls("POST", "/api/terminals")


def test_a_pod_that_is_not_up_is_reported_and_never_launched(world):
    add_experiment()
    world.hub.drop_server("D")

    answer = world.runner.run("KX1", config=fake_config(world.tmp))

    assert (answer["outcome"], answer["failed_step"]) == ("failed", "pod")
    assert world.notion.requests == []
    assert not world.hub_calls("POST", "/server") and not world.hub_calls("PUT")


def test_an_unknown_experiment_is_refused_with_a_ledger_row(world):
    answer = world.runner.run("KX9")
    assert (answer["outcome"], answer["problems"]) == ("refused", ["no experiment KX9 in the ledger (python -m lab "
                                                                   "experiment list)"])
    assert world.rows("experiment run")[0]["note"].startswith("refused: no experiment KX9")


# --- the verb ----------------------------------------------------------------------------------------------------

def test_the_verb_runs_through_the_experiment_dispatcher_with_a_dry_run(capsys, monkeypatch):
    experiments.seed()
    assert lab_cli.main(["experiment", "run", "E3", "--dry-run"]) == 1          # D fills no role by default
    assert json.loads(capsys.readouterr().out)["outcome"] == "no_pod_role"
    monkeypatch.setenv("LAB_POD_ROLES", "C:pod-a,D:pod-b")

    code = lab_cli.main(["experiment", "run", "E3", "--dry-run"])

    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and (answer["outcome"], answer["node"]) == ("dry_run", "pod-b")
    assert answer["what"] == "E3 bench run: Batching vs joules per token"
    script = Path(answer["script"])
    assert script.is_file() and script.name == f"lab_exp_{answer['run_id']}.sh"
    assert ledger.read(verb="experiment run") == []


def test_the_verbs_help_is_its_own(capsys):
    assert lab_cli.main(["experiment", "run", "-h"]) == 0
    assert "python -m lab experiment run" in capsys.readouterr().out


# --- the fake hub's file routes ----------------------------------------------------------------------------------

def test_the_fake_hub_serves_files_with_ranges_and_deletes_them():
    clock = FakeClock()
    hub = FakeHub(clock)
    account = hub.add_account("D", user="acct-d", token=HUB_TOKEN)
    hub.start_server("D")
    hub.files[("D", "lab_exp/x/a.txt")] = "hello pod"
    with PodLink(Hub(account, base_url=BASE, transport=hub.transport), transport=hub.transport) as link:
        part = link.request("GET", "/files/lab_exp/x/a.txt", headers={"Range": "bytes=0-4"}, accept="*/*")
        gone = link.request("DELETE", "/api/contents/lab_exp/x/a.txt", also_ok=(404,))
        again = link.request("DELETE", "/api/contents/lab_exp/x/a.txt", also_ok=(404,))

    assert (part.status, part.content) == (206, b"hello")
    assert (gone.status, again.status) == (204, 404) and ("D", "lab_exp/x/a.txt") not in hub.files
