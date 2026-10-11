"""Tests for the placement plan (lab/placement.py, python -m lab placement).

Everything runs on recorded facts: the pods' facts folder fixtures/lab/placement (the controller's status answer,
the facts step's output as the R29 and R30 pages recorded it, pod C's sampler file), the sample chip sheet under
lab/refine/research/kernels/data/ and the GPU Queue through lab/machine/slots_fake.FakeNotion. No pod, no hub and
no real Notion is reached: every test gets its own state dir and no .env file.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import placement
from lab.machine import slots, slots_fake
from lab.machine.slots_fake import FakeNotion
from lab.machine.slots_notion import NotionClient

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTS = REPO_ROOT / "fixtures" / "lab" / "placement"
SAMPLE = REPO_ROOT / "lab" / "refine" / "research" / "kernels" / "data" / "chipsheet-sample.json"
NOW = dt.datetime(2026, 10, 10, 22, 30, tzinfo=dt.timezone.utc)
B = "B exclusive"
A = "A teacher shared service"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private state dir, no .env files, no Notion token, no facts folder setting."""
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("NOTION_TOKEN", "LAB_QUEUE_DATA_SOURCE", "LAB_NOTION_URL", placement.ENV_FACTS):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def recorded_plan(**rules) -> placement.Plan:
    return placement.default_plan(facts=FACTS, rules=placement.Rules(**rules) if rules else None, now=NOW)


def copy_facts(tmp_path: Path) -> Path:
    """A writable copy of the recorded facts folder, with the sampler file it points at copied beside it."""
    folder = tmp_path / "facts"
    shutil.copytree(FACTS, folder)
    manifest = json.loads((folder / "pods.json").read_text(encoding="utf-8"))
    power = (FACTS / manifest["pods"]["pod-a"]["power"]).resolve()
    shutil.copy(power, folder / "pod-a.power.jsonl")
    manifest["pods"]["pod-a"]["power"] = "pod-a.power.jsonl"
    (folder / "pods.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def edit_status(folder: Path, letter: str, **fields) -> None:
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    status["accounts"][letter].update(fields)
    (folder / "status.json").write_text(json.dumps(status), encoding="utf-8")


# --- the kind of a request -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("L3: smoke LoRA 200 steps", "lora"),
    ("L3: train the adapter", "lora"),
    ("L3: eval the adapter", "lora"),
    ("O4: bench", "bench"),
    ("R20: baseline matrix of the teacher on pod D", "bench"),
    ("R47: decode profile of the teacher's 4-bit path", "bench"),
    ("R33: chip sheet on gfx1151", "bench"),
    ("R45: Ax trials of the serve settings", "ax"),
    ("R2: restart the teacher server", "teacher"),
    ("R30: serve the teacher once through the proxy", "teacher"),
    ("T6: pod power sampler under the teacher load", "sampler"),
    ("L3: x", None),
    ("", None),
])
def test_a_request_s_words_name_its_kind_with_the_more_specific_kind_first(text, kind):
    assert placement.classify(text) == kind


# --- the chip sheet ----------------------------------------------------------------------------------------------

def test_the_sample_chip_sheet_documents_every_field_the_plan_reads_and_marks_its_numbers_as_a_sample():
    sheet = placement.read_chipsheet(SAMPLE)
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    read = {f"{section}.{key}" for paths in placement.CHIP_FIELDS.values() for section, key in paths[:1]}
    assert read == set(raw["placement_reads"])
    assert sheet.sample is True and sheet.gfx == "gfx1151" and sheet.problem is None
    assert {name: num.value for name, num in sheet.fields.items()} == {
        "visible_gib": 64.0, "free_gib": 58.0, "idle_w": 9.0, "loaded_w": 120.3}
    assert {num.kind for num in sheet.fields.values()} == {"sample"}
    assert sheet.get("bandwidth_gb_s").known is False and sheet.get("matmul_tflops").known is False
    assert sheet.get("free_gib").source == "lab/refine/research/kernels/data/chipsheet-sample.json: memory.free_gib"


def test_a_measured_sheet_with_more_fields_and_rows_instead_of_peaks_is_read_and_the_rest_ignored(tmp_path):
    sheet_path = tmp_path / "chipsheet-gfx1151-2026-10-11.json"
    sheet_path.write_text(json.dumps({
        "format": "chipsheet/1", "chip": {"gfx": "gfx1151", "name": "Radeon 8060S"},
        "memory": {"visible_gib": 64.0, "free_gib": {"value": 61.2, "command": "torch.cuda.mem_get_info()"},
                   "lds_bytes": 65536},
        "bandwidth": {"copy_d2d": [{"bytes": 2 ** 20, "gb_s": 101.5}, {"bytes": 2 ** 30, "gb_s": 212.0},
                                   {"bytes": 2 ** 32, "gb_s": {"value": 208.4, "spread_pct": 1.1}}]},
        "matmul": {"shapes": [{"use": "lora-4b", "dtype": "bf16", "tflops": 31.5},
                              {"use": "decode-27b", "dtype": "bf16", "tflops": 4.2},
                              {"use": "lora-4b", "dtype": "fp16", "tflops": 40.0}]},
        "power": {"idle_w": 9, "loaded_w": {"value": 131.0}},
        "occupancy": [{"vgprs": 64, "waves": 16}], "versions": {"rocm": "7.13.0"},
    }), encoding="utf-8")
    sheet = placement.read_chipsheet(sheet_path)
    assert sheet.sample is False and sheet.problem is None
    assert {name: num.value for name, num in sheet.fields.items()} == {
        "visible_gib": 64.0, "free_gib": 61.2, "bandwidth_gb_s": 212.0, "matmul_tflops": 31.5, "idle_w": 9.0,
        "loaded_w": 131.0}
    assert {num.kind for num in sheet.fields.values()} == {"measured"}
    assert sheet.get("bandwidth_gb_s").source.endswith("the largest bandwidth.copy_d2d[].gb_s")


def test_the_newest_sheet_for_the_pod_s_chip_is_preferred_to_the_sample(tmp_path):
    for name in ("chipsheet-gfx1151-2026-10-10.json", "chipsheet-gfx1151-2026-10-12.json",
                 "chipsheet-gfx1201-2026-10-13.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert placement.default_chipsheet("gfx1151", tmp_path).name == "chipsheet-gfx1151-2026-10-12.json"
    assert placement.default_chipsheet("gfx9999", tmp_path).name == "chipsheet-sample.json"
    assert placement.default_chipsheet(None, tmp_path).name == "chipsheet-sample.json"


def test_a_broken_chip_sheet_leaves_its_numbers_unknown_and_the_plan_says_so(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    plan = placement.default_plan(facts=FACTS, chipsheets={"pod-b": broken}, now=NOW)
    assert plan.pods["pod-b"].chip.problem.startswith("cannot read ")
    assert any(note.startswith("pod-b: cannot read ") for note in plan.notes)
    assert plan.where["lora"] == ["pod-b"]          # the pod's facts still give its free memory


# --- the pods' facts ---------------------------------------------------------------------------------------------

def test_the_facts_step_output_is_read_line_by_line_and_notes_are_skipped():
    facts = placement.read_facts_text("# a note: with a colon\nmem: 62 GB total, 58 GB available\n\n"
                                      "listeners: no-8000 8099\nmem: 1 GB total, 1 GB available\n")
    assert facts == {"mem": "62 GB total, 58 GB available", "listeners": "no-8000 8099"}


def test_the_recorded_pods_are_read_through_the_status_answer_the_facts_and_the_sampler_file():
    pods, manifest = placement.load_pods(FACTS)
    a, b = pods["pod-a"], pods["pod-b"]
    assert manifest["recorded"] == "2026-10-10T22:30:00Z"
    assert (a.account, a.up, a.accelerator, a.started, a.serving) == ("C", True, "strix-halo", "2026-10-10T19:07:12Z",
                                                                     ("teacher",))
    assert (b.account, b.up, b.accelerator, b.started, b.serving) == ("D", True, "strix-halo", "2026-10-10T21:24:00Z",
                                                                     ())
    assert (a.available.value, a.visible.value, b.available.value, b.visible.value) == (58.0, 64.0, 58.0, 64.0)
    assert a.facts_server_running is False and a.listening == frozenset({8099, 8888})
    assert a.free_ports == frozenset({8000, 8790})
    assert b.holds_teacher_build is True and a.holds_teacher_build is False
    assert b.watts.value == 9.0 and "the last of 10 amd-smi reads" in b.watts.source
    assert a.watts.value == 92.377 and a.watts.source.startswith(
        "fixtures/telemetry/pod_power/strixhalo_2026-10-10_teacher_load_samples.jsonl: mean socket_w of the last 60 s")
    assert a.sampler_seen.endswith("samples from gpu-a up to 2026-10-10T20:53:15.150Z") and b.sampler_seen is None


def test_the_facts_folder_is_the_setting_then_a_live_snapshot_then_the_recorded_fixtures(tmp_path, monkeypatch):
    assert placement.facts_folder() == placement.FIXTURE_FACTS
    live = tmp_path / "state" / "placement"
    live.mkdir(parents=True)
    assert placement.facts_folder() == placement.FIXTURE_FACTS          # no pods.json: not a snapshot
    (live / "pods.json").write_text("{}", encoding="utf-8")
    assert placement.facts_folder() == live
    monkeypatch.setenv(placement.ENV_FACTS, str(tmp_path / "elsewhere"))
    assert placement.facts_folder() == tmp_path / "elsewhere"
    assert placement.facts_folder(tmp_path / "given") == tmp_path / "given"


# --- the plan for today's two pods -------------------------------------------------------------------------------

def test_today_the_teacher_stays_on_pod_a_and_every_exclusive_job_goes_to_pod_b():
    plan = recorded_plan()
    assert plan.where == {"teacher": ["pod-a"], "sampler": ["pod-a", "pod-b"], "bench": ["pod-b"], "ax": ["pod-b"],
                          "lora": ["pod-b"]}
    assert [(line.job, line.decision) for line in plan.lines] == [
        ("teacher", "pod-a (stays)"), ("sampler", "pod-a, pod-b"), ("bench", "pod-b (alone)"),
        ("ax", "pod-b (alone)"), ("lora", "pod-b")]
    teacher = plan.line("teacher").reason
    assert "it serves the teacher now (fixtures/lab/placement/status.json: account C serving on port 8000" in teacher
    assert "If pod-a goes down, pod-b holds the same build and checkpoint" in teacher
    assert "pod-b has none recorded: start it (bash lab/pod_power.sh)" in plan.line("sampler").reason


def test_the_teacher_and_a_lora_job_fit_one_pod_on_paper_but_not_at_today_s_share():
    plan = recorded_plan()
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    assert (share.allowed, share.on_paper) == (False, True)
    text = placement.plain(share.reason)
    assert ("the teacher's 57.6 GiB (share 0.9), a LoRA budget of 13.05 GiB (8.7 GiB peak plus a 50 % margin) and a "
            "4 GiB host reserve come to 74.65 GiB, more than the 58 GiB free") in text
    assert "the weights 18 GiB plus the rest are 35.05 GiB; with the teacher served at a share of at most 0.63" in text
    assert share.margin.value == pytest.approx(-16.65)
    lora = placement.plain(plan.line("lora").reason)
    assert "pod-b has 58 GiB free, leaving 40.95 GiB after the 4 GiB host reserve" in lora
    assert "Not pod-a: the teacher holds 57.6 GiB of its 58 GiB, leaving 0.4 GiB." in lora
    assert any("share of at most 0.63" in placement.plain(lever) for lever in plan.levers)


def test_served_at_a_smaller_share_the_teacher_s_pod_also_takes_lora_jobs():
    plan = recorded_plan(teacher_share=0.6)
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    assert (share.allowed, share.on_paper) == (True, True)
    assert "come to 55.45 GiB of 58 GiB free. Start the teacher first" in placement.plain(share.reason)
    assert plan.where["lora"] == ["pod-b", "pod-a"]
    assert not any("share of at most" in lever for lever in plan.levers)
    assert placement.check(plan, gpu="A", text="L3: smoke LoRA")["verdict"] == "consistent"


def test_a_larger_margin_shrinks_the_share_the_teacher_may_keep():
    assert placement.plain(recorded_plan(margin=1.0).line("lora").reason).startswith(
        "GPU B is the exclusive card: a LoRA job needs 17.4 GiB (8.7 GiB peak, measured on the R9700 smoke run, "
        "plus a 100 % margin)")
    share = next(s for s in recorded_plan(margin=1.0).sharing if s.jobs == ("teacher", "lora"))
    assert "a share of at most 0.57" in placement.plain(share.reason)


def test_measurements_never_share_a_chip_and_the_sampler_shares_any():
    plan = recorded_plan()
    allowed = {share.jobs: share.allowed for share in plan.sharing}
    assert allowed == {
        ("sampler", "teacher"): True, ("sampler", "lora"): True, ("sampler", "bench"): True, ("sampler", "ax"): True,
        ("teacher", "lora"): False, ("teacher", "bench"): False, ("teacher", "ax"): False, ("lora", "bench"): False,
        ("lora", "ax"): False, ("bench", "ax"): False, ("bench", "bench"): False, ("ax", "ax"): False,
        ("lora", "lora"): False, ("teacher", "teacher"): False}


def test_every_number_in_the_plan_is_traced_to_a_file_that_exists_or_to_a_formula_or_a_rule():
    plan = recorded_plan()
    text = placement.render(plan)
    numbers = plan.trace.views()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text.split("\nNumbers\n")[0])}
    assert cited == set(range(1, len(numbers) + 1))
    for number in numbers:
        if number["kind"] in ("recorded", "measured", "sample"):
            path = re.split(r"[:(]", number["source"], maxsplit=1)[0].strip()
            assert (REPO_ROOT / path).is_file(), number
        elif number["kind"] == "rule":
            assert "lab/refine/research/plumbing/placement.md#memory-margin" in number["source"], number
        else:
            assert number["kind"] == "derived" and number["source"], number
    assert {n["source"].split(" (")[0] for n in numbers if n["kind"] == "measured"} >= {
        "lab/refine/metrics.jsonl:14", "lab/refine/metrics.jsonl:24", "lab/refine/metrics.jsonl:29",
        "lab/refine/metrics.jsonl:34", "lab/refine/metrics.jsonl:35"}


def test_the_rule_anchor_the_plan_cites_exists_in_the_research_note():
    from lab.machine import research_check
    assert research_check.main(["--ref", f"{placement.NOTE}#memory-margin", "--root", str(REPO_ROOT)])["ok"] is True


def test_when_pod_a_is_down_the_teacher_moves_to_pod_b_and_measurements_wait(tmp_path):
    folder = copy_facts(tmp_path)
    edit_status(folder, "C", running=False, ready=False, serving={"port": 8000, "models": [], "health": 424})
    plan = placement.default_plan(facts=folder, now=NOW)
    assert plan.where == {"teacher": ["pod-b"], "sampler": ["pod-b"], "bench": [], "ax": [], "lora": []}
    assert plan.line("teacher").decision == "pod-b (start it)"
    assert plan.line("lora").decision == "wait"     # the teacher at 0.90 leaves pod-b no room for one
    assert "holds the build and the checkpoint (home_holds vllm, models)" in plan.line("teacher").reason
    assert plan.line("bench").decision == "wait"
    verdict = placement.check(plan, gpu="B", text="R20: baseline matrix")
    assert verdict["contradicts"] is True and verdict["reason"].startswith("the plan has no pod for bench now")


def test_with_no_pod_up_nothing_is_placed(tmp_path):
    folder = copy_facts(tmp_path)
    edit_status(folder, "C", running=False, ready=False)
    edit_status(folder, "D", running=False, ready=False)
    plan = placement.default_plan(facts=folder, now=NOW)
    assert plan.where == {"teacher": [], "sampler": [], "bench": [], "ax": [], "lora": []}
    assert plan.line("teacher").decision == "nowhere"


# --- checking a request ------------------------------------------------------------------------------------------

def test_a_lora_job_asked_for_on_the_teacher_s_card_contradicts_the_plan_and_says_where_to_ask():
    verdict = placement.check(recorded_plan(), gpu="A", text="L3: smoke LoRA 200 steps")
    assert (verdict["contradicts"], verdict["verdict"], verdict["kind"], verdict["kind_from"], verdict["node"]) == (
        True, "contradicts", "lora", "words", "pod-a")
    assert verdict["reason"].startswith("the plan puts lora on GPU B (pod-b), not on pod-a: GPU B is the exclusive")
    assert "[" not in verdict["reason"]
    assert verdict["next"] == 'ask for GPU B instead, or add --override "why it must run on pod-a"'


@pytest.mark.parametrize("gpu,text,kind,verdict", [
    ("B", "L3: smoke LoRA 200 steps", None, "consistent"),
    ("A", "R2: restart the teacher server", None, "consistent"),
    ("B", "R2: serve the teacher for the box", None, "contradicts"),
    ("B", "R20: baseline matrix", None, "consistent"),
    ("A", "R20: baseline matrix", None, "contradicts"),
    ("A", "R45: Ax trials", None, "contradicts"),
    ("A", "T6: power sampler", None, "consistent"),
    ("B", "L3: x", None, "no_line"),
    ("A", "L3: x", "bench", "contradicts"),
    ("A", "L3: smoke LoRA", "teacher", "consistent"),
])
def test_each_kind_is_checked_against_the_card_the_plan_gives_it(gpu, text, kind, verdict):
    assert placement.check(recorded_plan(), gpu=gpu, text=text, kind=kind)["verdict"] == verdict


def test_a_plan_that_cannot_be_built_never_blocks_a_request(tmp_path, monkeypatch):
    monkeypatch.setenv(placement.ENV_FACTS, str(tmp_path / "missing"))
    verdict = placement.check_request(gpu="A", text="L3: smoke LoRA")
    assert verdict["contradicts"] is False and verdict["verdict"] == "unavailable"
    assert verdict["reason"].startswith("no placement plan: FileNotFoundError")


# --- the queue ---------------------------------------------------------------------------------------------------

class Queue:
    def __init__(self) -> None:
        self.fake = FakeNotion(clock=lambda: NOW)
        self.client = NotionClient(slots_fake.TOKEN, transport=self.fake.transport, sleep=lambda s: None)

    def row(self, request: str, **fields) -> str:
        fields.setdefault("task_id", request.split(":")[0])
        fields.setdefault("requested", NOW - dt.timedelta(minutes=10))
        return self.fake.add_row(request, **fields)


def test_the_queue_is_read_through_the_slots_client_and_its_rows_are_checked_against_the_plan():
    queue = Queue()
    queue.row("R20: baseline matrix", gpu=B, status="Running", started=NOW - dt.timedelta(minutes=5))
    queue.row("L3: train the adapter", gpu=A, status="Queued")
    queue.row("L3: smoke LoRA", gpu=B, status="Queued")
    queue.row("O4: bench", gpu=B, status="Done", started=NOW - dt.timedelta(hours=2), ended=NOW - dt.timedelta(hours=1))
    text, answer, code = placement.run(["--json"], client=queue.client, now=NOW)
    assert code == 0 and answer["ok"] is True and answer["queue"]["read"] is True
    assert [(r["request"], r["status"], r["node"], r["kind"], r.get("consistent")) for r in answer["queue"]["rows"]] == [
        ("L3: train the adapter", "Queued", "pod-a", "lora", False),
        ("R20: baseline matrix", "Running", "pod-b", "bench", True),
        ("L3: smoke LoRA", "Queued", "pod-b", "lora", True)]
    bench = next(line for line in answer["lines"] if line["job"] == "bench")
    assert "It is busy now: queue row 'R20: baseline matrix' (Running); the queue keeps the order." in bench["reason"]
    assert "Queued on A teacher shared service: L3: train the adapter (lora, CONTRADICTS the plan)" in text


def test_a_running_slot_makes_its_card_busy_without_making_it_smaller():
    queue = Queue()
    queue.row("L3: smoke LoRA", gpu=B, status="Running", started=NOW - dt.timedelta(minutes=5))
    _, answer, code = placement.run(["--json"], client=queue.client, now=NOW)
    lines = {line["job"]: placement.plain(line["reason"]) for line in answer["lines"]}
    assert code == 0 and "pod-b has 58 GiB free, leaving 40.95 GiB" in lines["lora"]
    busy = "It is busy now: queue row 'L3: smoke LoRA' (Running); the queue keeps the order."
    assert busy in lines["lora"] and busy in lines["bench"]


def test_a_queue_file_stands_in_for_the_live_queue(tmp_path):
    queue = Queue()
    queue.row("L3: smoke LoRA", gpu=A, status="Queued")
    pages = queue.client.query_data_source(slots_fake.DATA_SOURCE_ID)
    (tmp_path / "queue.json").write_text(json.dumps(pages), encoding="utf-8")
    _, answer, code = placement.run(["--queue-file", str(tmp_path / "queue.json")], now=NOW)
    assert code == 0 and [r.get("consistent") for r in answer["queue"]["rows"]] == [False]


def test_without_a_token_the_plan_is_made_and_says_the_queue_was_not_read():
    text, answer, code = placement.run([], environ={"LAB_ENV_FILES": ""}, now=NOW)
    assert code == 0 and answer["queue"] == {"read": False, "rows": []}
    assert "the GPU Queue was not read: no NOTION_TOKEN" in answer["notes"]


# --- the command line --------------------------------------------------------------------------------------------

def test_python_m_lab_placement_prints_the_plan_as_text(capsys):
    assert lab_cli.main(["placement", "--no-queue", "--now", "2026-10-10T22:30:00Z"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Placement plan as of 2026-10-10T22:30:00Z; facts from fixtures/lab/placement")
    for heading in ("\nPods\n", "\nPlan\n", "\nMay share a chip\n", "\nQueue\n", "\nLevers\n", "\nNotes\n",
                    "\nNumbers\n"):
        assert heading in out
    assert "  teacher  -> pod-a (stays)" in out and "  lora     -> pod-b" in out


def test_json_is_one_object_on_one_line(capsys):
    assert lab_cli.main(["placement", "--no-queue", "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    answer = json.loads(lines[0])
    assert answer["outcome"] == "plan" and answer["where"]["teacher"] == ["pod-a"]


def test_check_exits_1_on_a_contradiction_and_0_when_consistent(capsys):
    assert lab_cli.main(["placement", "--no-queue", "--check", "L3: smoke LoRA", "--gpu", "A"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("CONTRADICTS the plan: the plan puts lora on GPU B (pod-b), not on pod-a")
    assert lab_cli.main(["placement", "--no-queue", "--check", "L3: smoke LoRA", "--gpu", "B", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["ok"], answer["outcome"], answer["kind"]) == (True, "consistent", "lora")


def test_out_writes_the_plan_as_json(tmp_path):
    target = tmp_path / "plan.json"
    _, answer, code = placement.run(["--no-queue", "--out", str(target)], now=NOW)
    assert code == 0 and json.loads(target.read_text(encoding="utf-8")) == answer


def test_a_facts_folder_without_a_manifest_is_an_error_not_a_crash(tmp_path):
    text, answer, code = placement.run(["--no-queue", "--facts", str(tmp_path)], now=NOW)
    assert code == 1 and answer["outcome"] == "error" and answer["error"].startswith("FileNotFoundError")


@pytest.mark.parametrize("argv", [
    ["--chipsheet", "pod-c=x.json"], ["--chipsheet", "pod-a"], ["--kind", "train"], ["--margin", "-1"],
    ["--queue-file", "q.json", "--no-queue"],
])
def test_bad_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        placement.run(argv)
    assert caught.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_the_status_fixture_is_what_its_generator_makes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_status", FACTS / "make_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.status() == json.loads((FACTS / "status.json").read_text(encoding="utf-8"))


def test_the_slot_verbs_import_the_placement_module_only_when_a_request_needs_it():
    import subprocess
    import sys
    code = ("import sys; from lab.machine import slots; "
            "print('lab.placement' in sys.modules)")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"
    assert slots.PLACEMENT_KINDS == placement.KINDS


def test_a_number_cited_by_file_and_line_is_on_that_line():
    plan = recorded_plan()
    placement.render(plan)
    checked = 0
    for number in plan.trace.views():
        cited = re.match(r"([\w./-]+):(\d+) ", number["source"])
        if not cited or number["value"] is None:
            continue
        line = (REPO_ROOT / cited.group(1)).read_text(encoding="utf-8").splitlines()[int(cited.group(2)) - 1]
        values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", line)]
        assert any(abs(value - number["value"]) < 1e-9 for value in values), (number, line)
        checked += 1
    assert checked >= 7


def test_every_sitting_may_ask_for_the_plan_and_check_a_request():
    from lab.machine import allowed
    for tier in ("fable", "opus", "sonnet"):
        for python in allowed.PYTHONS:
            assert allowed.allows(tier, "Bash", f"{python} -m lab placement")
            assert allowed.allows(tier, "Bash", f'{python} -m lab placement --check "L3: smoke LoRA" --gpu A --json')


def test_facts_read_with_the_teacher_up_are_not_charged_for_its_share_twice(tmp_path):
    folder = copy_facts(tmp_path)
    facts = (folder / "pod-a.facts.txt").read_text(encoding="utf-8")
    facts = facts.replace("mem: 62 GB total, 58 GB available", "mem: 62 GB total, 1 GB available")
    facts = facts.replace("server_running: no", "server_running: yes")
    (folder / "pod-a.facts.txt").write_text(facts, encoding="utf-8")
    plan = placement.default_plan(facts=folder, chipsheets={"pod-a": SAMPLE.parent / "none.json"}, now=NOW)
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    assert "more than the 58.6 GiB free" in placement.plain(share.reason)
    assert "leaving 1 GiB" in placement.plain(plan.line("lora").reason)


# --- the measured interference (lab/interference.py's data file) ------------------------------------------------

INTERFERENCE = REPO_ROOT / "fixtures" / "lab" / "interference" / "interference-gfx1151-2026-10-11.json"


def test_with_a_measured_interference_file_the_sharing_line_gives_the_measured_cost_instead_of_a_paper_estimate():
    plan = placement.default_plan(facts=FACTS, now=NOW, interference=INTERFERENCE)
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    text = placement.plain(share.reason)
    assert "unmeasured" not in text
    assert ("Measured on pod D with the teacher at a share of 0.63 "
            "(fixtures/lab/interference/interference-gfx1151-2026-10-11.json): beside the LoRA the teacher's decode "
            "went from 40 tok/s to 26.67 tok/s (-33.33 %) and the LoRA's samples per second from 0.4 samples/s to "
            "0.32 samples/s (-20 %); doing the work together changed its time by -16.67 % and its energy by "
            "-8.33 % against one after the other, so sharing pays.") in text
    assert share.view()["measured"]["file"] == "fixtures/lab/interference/interference-gfx1151-2026-10-11.json"
    assert share.view()["measured"]["decode_change"]["value"] == -33.33
    assert any("the decode slowdown measured beside a LoRA is -33.33 %" in placement.plain(lever)
               for lever in plan.levers)
    for number in plan.trace.views():
        if "interference-gfx1151" in number["source"]:
            assert number["kind"] == "measured"
            assert (REPO_ROOT / number["source"].split(":")[0]).is_file(), number


def test_a_measured_run_where_sharing_costs_more_says_to_run_them_one_after_the_other(tmp_path):
    data = json.loads(INTERFERENCE.read_text(encoding="utf-8"))
    data["compare"].update(time_change_pct=12.5, energy_change_pct=3.0)
    path = tmp_path / "interference-gfx1151-2026-10-12.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    plan = placement.default_plan(facts=FACTS, now=NOW, rules=placement.Rules(teacher_share=0.6), interference=path)
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    assert (share.allowed, share.on_paper) == (False, True)
    text = placement.plain(share.reason)
    assert text.startswith("not on pod-a: they fit (the teacher's")
    assert text.endswith("so sharing does not pay: run them one after the other.")
    assert plan.where["lora"] == ["pod-b"]
    assert placement.check(plan, gpu="A", text="L3: smoke LoRA")["verdict"] == "contradicts"
    paid = placement.default_plan(facts=FACTS, now=NOW, rules=placement.Rules(teacher_share=0.6),
                                  interference=INTERFERENCE)
    assert paid.where["lora"] == ["pod-b", "pod-a"]


def test_without_a_usable_interference_file_the_line_stays_a_paper_estimate(tmp_path):
    pods, manifest = placement.load_pods(FACTS)
    assert placement.newest_interference(tmp_path) is None
    plan = placement.build_plan(pods, manifest=manifest, now=NOW, interference_dir=tmp_path)
    share = next(s for s in plan.sharing if s.jobs == ("teacher", "lora"))
    assert "is unmeasured (R43)" in placement.plain(share.reason) and share.measured is None
    broken = tmp_path / "interference-gfx1151-2026-10-11.json"
    broken.write_text('{"schema": "something else"}', encoding="utf-8")
    assert placement.newest_interference(tmp_path) == broken
    assert placement.read_interference(broken) is None
    plan = placement.build_plan(pods, manifest=manifest, now=NOW, interference_dir=tmp_path)
    assert next(s for s in plan.sharing if s.jobs == ("teacher", "lora")).measured is None


def test_the_newest_interference_file_beside_the_chip_sheets_is_the_default(tmp_path):
    for day in ("2026-10-11", "2026-10-13"):
        shutil.copy(INTERFERENCE, tmp_path / f"interference-gfx1151-{day}.json")
    (tmp_path / "interference-gfx1151-2026-10-14.md").write_text("not json", encoding="utf-8")
    assert placement.newest_interference(tmp_path).name == "interference-gfx1151-2026-10-13.json"


def test_python_m_lab_placement_takes_an_interference_file(capsys):
    text, answer, code = placement.run(["--no-queue", "--now", NOW.isoformat(), "--facts", str(FACTS),
                                        "--interference", str(INTERFERENCE)])
    assert code == 0
    share = next(s for s in answer["sharing"] if s["jobs"] == ["teacher", "lora"])
    assert share["measured"]["time_change"]["value"] == -16.67
    assert "so sharing pays" in placement.plain(text)
