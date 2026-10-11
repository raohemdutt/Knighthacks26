"""lab/quark_compare.py (R23), one group of tests per clause of the task's Done when.

Nothing here reaches a pod, the hub, Notion or the network. The GPU Queue is lab/machine/slots_fake.FakeNotion, the
server and the pod are lab/bench_fake.py (a real process on 127.0.0.1 and a fake pod root), ~/vllm is a small git
repository whose file sits at the path of one loader file, and the checkpoint is a folder of sparse files at the
pinned revision's sizes with the real config.json. What only pod B can show is in lab/R23_REAL_RUN.md."""
from __future__ import annotations

import datetime as dt
import faulthandler
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lab import bench, bench_fake, quark_compare as qc, quark_keys
from lab.__main__ import VERBS
from lab.machine import experiments, ledger, slots, slots_fake, slots_notion
from lab.machine import verdict as vmath

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "kernels" / "quark_config_august_2026.json"
NOW = dt.datetime(2026, 10, 11, 6, 0, tzinfo=dt.timezone.utc)
WATCHDOG_S = 180.0
LOADER_FILE = "vllm/model_executor/layers/quantization/quark/quark.py"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


# --- the stand-ins ----------------------------------------------------------------------------------------------

def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def mark_sparse(fh) -> None:
    """Flag the open file as sparse before its last byte is written at the far offset. POSIX filesystems leave the gap
    unallocated by themselves; on Windows (NTFS) a plain file gets real zeros, 19.5 GB for the weights, which fills
    the disk (OSError 28) and takes minutes."""
    if os.name != "nt":
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                         wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                         wintypes.LPVOID]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    fsctl_set_sparse = 0x000900C4
    kernel32.DeviceIoControl(msvcrt.get_osfhandle(fh.fileno()), fsctl_set_sparse, None, 0, None, 0,
                             ctypes.byref(wintypes.DWORD(0)), None)


def make_checkpoint(folder: Path, *, skip=(), wrong=()) -> Path:
    """The checkpoint as copied from pod A: the real config.json and every other file sparse, at its size."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, size in qc.CHECKPOINT_FILES.items():
        if name in skip:
            continue
        if name == "config.json":
            shutil.copyfile(FIXTURE, folder / name)
            continue
        with open(folder / name, "wb") as fh:
            mark_sparse(fh)
            fh.seek(size + (1 if name in wrong else 0) - 1)     # truncate would still write the zeros on Windows
            fh.write(b"\0")
    return folder


def make_vllm(folder: Path) -> tuple[Path, Path]:
    """A git tree standing in for ~/vllm, and a copy of lab/vllm-gfx1201.diff whose loader part is a real hunk on it
    (the build part kept as it is)."""
    target = folder / LOADER_FILE
    target.parent.mkdir(parents=True)
    target.write_text("class QuarkConfig:\n    pass\n", encoding="utf-8", newline="\n")
    git(folder, "init", "-q")
    git(folder, "config", "core.autocrlf", "false")   # a Windows machine's global autocrlf would put CRLF back
    git(folder, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(folder, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "b1388b1 stand-in")
    target.write_text("class QuarkConfig:\n    pass\n\n\nclass QuarkW4A16Int4:\n    pass\n", encoding="utf-8", newline="\n")
    loader = git(folder, "diff")
    git(folder, "checkout", "--", ".")
    build = quark_keys.split_patch((REPO / "lab" / "vllm-gfx1201.diff").read_text(encoding="utf-8"))[0]
    patch = folder.parent / "vllm-gfx1201.diff"
    patch.write_text(build + loader, encoding="utf-8", newline="\n")
    return folder, patch


def control_rows(batches=(1, 4), n=5, *, tok_s=(2.9, 3.0, 2.8, 2.95, 2.85), jpt=(40.0, 41.0, 39.5, 40.5, 40.2),
                 conditions=None):
    """Arm A's bench rows (R20's rows.jsonl), one per cold spawn and batch size."""
    rows = []
    for b in batches:
        for i in range(n):
            rows.append({"run_id": f"a-{b}-{i}", "batch_size": b, "outcome": "ok", "engine": "vllm",
                         "decode_tok_s": tok_s[i % len(tok_s)] / (1 if b == 1 else 1.2),
                         "joules_per_token": jpt[i % len(jpt)] / b, "conditions": dict(conditions or {})})
    return rows


class Pod:
    """The fake pod B: its root (bench_fake's), ~/vllm, the copied checkpoint, the control checkpoint."""

    def __init__(self, tmp: Path) -> None:
        self.root = tmp / "pod"
        self.root.mkdir()
        self.vllm, self.patch = make_vllm(tmp / "home" / "vllm")
        self.checkpoint = make_checkpoint(tmp / "home" / "models" / "Qwen3.8-27B-Quark-AWQ-INT4-W4A16")
        self.control = tmp / "home" / "models" / "Qwen3.8-27B-INT4"
        self.control.mkdir(parents=True)
        self.host = bench_fake.FakeHost(self.root)
        self.power = bench_fake.FakePower(self.root)
        self.out = tmp / "out"

    def argv(self, sub, *more, reference=None, batches="1", repeats=2):
        serve = bench_fake.serve_command(engine="vllm", root=str(self.root), prefix_caching="off")
        return [sub, "--checkpoint", str(self.checkpoint), "--control-checkpoint", str(self.control),
                "--vllm-dir", str(self.vllm), "--patch", str(self.patch), "--out", str(self.out), "--port", "0",
                "--serve-cmd", json.dumps(serve), "--reference", str(reference or self.out / "no-reference.jsonl"),
                "--batches", batches, "--repeats", str(repeats), "--max-tokens", "6", "--correctness-max-tokens",
                "9", "--idle-s", "0.05", *more]

    def run(self, sub, *more, **kw):
        state = self.out.parent / "state"
        return qc.run(self.argv(sub, *more, **{k: v for k, v in kw.items() if k in ("reference", "batches",
                                                                                     "repeats")}),
                      host=self.host, power=self.power, state_dir=state, ready_timeout_s=30.0,
                      **{k: v for k, v in kw.items() if k in ("client", "environ", "now")})


@pytest.fixture
def pod(tmp_path):
    return Pod(tmp_path)


@pytest.fixture
def queue():
    fake = slots_fake.FakeNotion(clock=lambda: NOW)
    client = slots_notion.NotionClient(slots_fake.TOKEN, transport=fake.transport, sleep=lambda s: None)
    return fake, client


def critique(client, row):
    """What the referee writes on a long row: one Critique line after the justification."""
    page = slots.slot_from_page(client.retrieve_page(row))
    note = slots.add_entry(page.progress_note, slots.note_entry(slots.CRITIQUE, "referee", NOW,
                                                                "say why 5 spawns, not 3"))
    client.update_page(row, slots.rest_properties({"Progress note": note}))


# --- part 1: the loader patch and the checkpoint ----------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("VLLM_B1388B1"), reason="set VLLM_B1388B1 to a clean vLLM checkout at b1388b1")
def test_the_whole_diff_applies_to_a_clean_b1388b1():
    tree = Path(os.environ["VLLM_B1388B1"])
    assert git(tree, "rev-parse", "HEAD").startswith("b1388b1")
    subprocess.run(["git", "apply", "--check", str(REPO / "lab" / "vllm-gfx1201.diff")], cwd=tree, check=True)


def test_a_whole_copy_of_the_checkpoint_passes(tmp_path):
    check = qc.check_checkpoint(make_checkpoint(tmp_path / "ck"))
    assert check["ok"], check
    assert check["audit"]["scheme"] == "QuarkW4A16Int4"
    assert check["revision"] == quark_keys.CHECKPOINT_REVISION


def test_a_partial_copy_is_refused_and_nothing_is_downloaded(tmp_path, pod):
    make_checkpoint(tmp_path / "ck", skip=("model.safetensors",), wrong=("tokenizer.json",))
    check = qc.check_checkpoint(tmp_path / "ck")
    assert not check["ok"]
    assert check["missing"] == ["model.safetensors"]
    assert check["wrong_size"] == {"tokenizer.json": {"size": 12809321, "expected": 12809320}}
    (pod.checkpoint / "model.safetensors").unlink()
    answer = pod.run("run")
    assert answer["outcome"] == "checkpoint"
    assert not pod.out.exists()                      # refused before the run folder, the patch or a spawn


def test_the_runner_reads_the_checkpoint_size_list_from_the_hub_listing():
    assert sum(qc.CHECKPOINT_FILES.values()) == 19_535_854_318
    assert qc.CHECKPOINT_FILES["config.json"] == FIXTURE.stat().st_size


# --- the same settings sheet as the teacher ---------------------------------------------------------------------

def test_the_settings_sheet_is_the_pinned_baseline(tmp_path):
    sheet = qc.settings_sheet(run_dir=tmp_path)
    args = sheet["args"]
    for pair in (["--max-model-len", "16384"], ["--gpu-memory-utilization", "0.90"], ["--dtype", "bfloat16"],
                 ["--mm-encoder-attn-backend", "TRITON_ATTN"], ["--attention-backend", "ROCM_ATTN"],
                 ["--mamba-ssm-cache-dtype", "float32"], ["--max-num-seqs", "256"]):
        i = args.index(pair[0])
        assert args[i:i + 2] == pair
    assert "--enable-prefix-caching" in args and "--speculative-config" not in args
    assert sheet["env"]["GPU_MAX_HW_QUEUES"] == "4"
    assert sheet["env"]["VLLM_ROCM_USE_SKINNY_GEMM"] == "1"
    assert sheet["env"]["TRITON_CACHE_DIR"] == f"{tmp_path}/triton-cache"
    assert "VLLM_DISABLED_KERNELS" in sheet["unset"] and "VLLM_DISABLED_KERNELS" not in sheet["env"]


def test_both_arms_launch_the_same_line_but_for_the_checkpoint(tmp_path):
    template = qc.serve_template(qc.settings_sheet(run_dir=tmp_path))
    a = bench.fill_command(template, {"checkpoint": "/m/Qwen3.8-27B-INT4", "model": "teacher", "port": 8100})
    b = bench.fill_command(template, {"checkpoint": "/m/Qwen3.8-27B-Quark-AWQ-INT4-W4A16", "model": "teacher",
                                      "port": 8100})
    assert [x for x, y in zip(a, b) if x != y] == ["/m/Qwen3.8-27B-INT4"]
    assert a[:3] == ["env", "-u", "VLLM_DISABLED_KERNELS"]
    assert a[a.index("--served-model-name") + 1] == "teacher" and a.count("--served-model-name") == 1
    assert "vllm.entrypoints.openai.api_server" in a and "--host" in a


def test_the_correctness_pass_runs_eager():
    template = qc.serve_template(qc.settings_sheet(run_dir="/r"))
    argv, is_eager = qc.eager(template)
    assert is_eager and "--enforce-eager" in argv and "--compilation-config" not in argv
    assert qc.eager(["python3", "fake.py"]) == (["python3", "fake.py"], False)


# --- the time plan of a 60-minute slot --------------------------------------------------------------------------

def test_the_default_plan_fits_the_slot_with_its_extension():
    est = qc.estimate(qc.Plan(), prompts=4)
    assert est["fits_with_extension"], est
    assert est["spawns"] == vmath.MIN_RUNS_PER_ARM + 1


def test_the_whole_matrix_in_one_slot_is_over_budget_and_refused(pod):
    est = qc.estimate(qc.Plan(batches=[1, 4, 8, 16], with_64=True, control_reference=True), prompts=4)
    assert not est["fits_with_extension"]
    where = ["--checkpoint", str(pod.checkpoint), "--vllm-dir", str(pod.vllm), "--out", str(pod.out),
             "--batches", "1,4,8,16"]                 # the default tokens and idle reads, as on the pod
    answer = qc.run(["preflight", *where])
    assert answer["outcome"] == "over_budget" and answer["time"]["minutes"] > 75
    answer = qc.run(["run", *where])
    assert answer["outcome"] == "over_budget" and not pod.out.exists()
    assert qc.estimate(qc.Plan(batches=[8, 16]), prompts=4)["fits_with_extension"]   # the second slot


# --- serves on a pod B slot: the queue --------------------------------------------------------------------------

def test_the_slot_is_a_60_minute_pod_b_row_with_its_justification(queue):
    fake, client = queue
    answer = qc.take_slot(None, None, client=client, environ={}, now=lambda: NOW)
    assert answer["outcome"] == "waiting" and answer["slot"]["outcome"] == "critique_pending"
    page = slots.slot_from_page(client.retrieve_page(answer["row"]))
    assert (page.task_id, page.gpu, page.minutes, page.status) == ("R23", "B exclusive", 60, "Queued")
    assert slots.justification_state(page) == slots.AWAITING_CRITIQUE


def test_the_slot_starts_once_the_critique_is_answered(queue):
    fake, client = queue
    row = qc.take_slot(None, None, client=client, environ={}, now=lambda: NOW)["row"]
    critique(client, row)
    waiting = qc.take_slot(row, None, client=client, environ={}, now=lambda: NOW)
    assert waiting["outcome"] == "waiting" and "--reply" in waiting["next"]
    started = qc.take_slot(row, "five spawns is the verdict floor", client=client, environ={}, now=lambda: NOW)
    assert started["outcome"] == "started"
    assert slots.slot_from_page(client.retrieve_page(row)).status == "Running"


def test_all_waits_without_touching_the_pod_while_the_slot_cannot_start(pod, queue, tmp_path):
    fake, client = queue
    rows = tmp_path / "control.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in control_rows()), encoding="utf-8")
    answer = pod.run("all", "--control-rows", str(rows), client=client, environ={}, now=lambda: NOW)
    assert answer["outcome"] == "waiting"
    assert not pod.out.exists()
    assert git(pod.vllm, "status", "--porcelain") == ""


# --- the whole run: bench matrix, correctness, verdict, ledger, T0, release, pod as found -----------------------

@pytest.fixture
def finished(pod, queue, tmp_path):
    """One `all` call inside a started slot, with arm A's rows and a reference written from arm A's stand-in."""
    fake, client = queue
    row = qc.take_slot(None, None, client=client, environ={}, now=lambda: NOW)["row"]
    critique(client, row)
    rows = tmp_path / "control.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in control_rows(batches=(1,))), encoding="utf-8")
    before = git(pod.vllm, "status", "--porcelain"), git(pod.vllm, "diff")
    answer = pod.run("all", "--control-rows", str(rows), "--row", row, "--reply", "five spawns is the floor",
                     client=client, environ={}, now=lambda: NOW)
    return answer, row, client, before


def test_all_serves_benches_compares_and_releases(finished, pod):
    answer, row, client, before = finished
    assert answer["released"]["outcome"] == "done", answer
    page = slots.slot_from_page(client.retrieve_page(row))
    assert page.status == "Done" and page.result.startswith("R23 quark-")
    assert answer["pod_as_found"] is True
    assert (git(pod.vllm, "status", "--porcelain"), git(pod.vllm, "diff")) == before
    assert answer["patch"] == {"state_before": "absent", "applied_by_run": True, "reverted": True}


def test_the_baseline_matrix_ran_against_the_quark_checkpoint(finished):
    answer = finished[0]
    result = json.loads(Path(answer["result_path"]).read_text(encoding="utf-8"))
    rows = result["bench"]["rows"]
    assert [(r["batch_size"], r["outcome"]) for r in rows] == [(1, "ok"), (1, "ok")]   # 2 cold spawns at batch 1
    assert {r["conditions"]["checkpoint"] for r in rows} == {result["checkpoint_dir"]}
    assert {r["conditions"]["quantization"] for r in rows} == {qc.QUANTIZATION}
    assert {r["conditions"]["node"] for r in rows} == {"pod-b"}
    assert all(r["joules_per_token"] is not None and r["decode_tok_s"] for r in rows)
    assert result["settings_sheet"]["env"]["GPU_MAX_HW_QUEUES"] == "4"


def test_the_correctness_harness_ran_on_both_arms(finished):
    answer = finished[0]
    result = json.loads(Path(answer["result_path"]).read_text(encoding="utf-8"))
    for key in ("control_correctness", "correctness"):
        outputs = result[key]["outputs"]
        assert [o["prompt_id"] for o in outputs] == ["text-kitchen", "image-bars", "text-schedule", "image-shapes"]
        assert all(o["error"] is None and o["text"] for o in outputs)
        assert result[key]["batch"] == 1 and result[key]["stopped"]
    match = answer["verdict"]["exact_match"]
    assert (match["x"], match["n"], match["rate"]) == (4, 4, 1.0)   # the stand-in answers both arms alike


def test_the_verdict_ledger_row_and_e5(finished, tmp_path):
    answer = finished[0]
    state = tmp_path / "state"
    rows = ledger.read(state_dir=state)
    mine = [r for r in rows if r["verb"] == qc.VERB]
    assert len(mine) == 1
    row = mine[0]
    assert row["task"] == "R23" and row["node"] == "pod-b" and row["outcome"] in vmath.LABELS
    assert row["metrics"]["teacher.output_match_rate"] == 1.0
    assert set(row["metrics"]) >= {"quark.decode_tok_s.b1", "quark.joules_per_token.b1"}
    assert row["units"]["quark.joules_per_token.b1"] == "J/token"
    assert row["conditions"]["zero_point"].startswith("none (symmetric int4, group 128)")
    e5 = experiments.load(state)["E5"]
    assert e5["result"]["arms"]["A"]["n"] == 5 and e5["result"]["arms"]["B"]["n"] == 2
    assert e5["status"] == "running"                      # arm C has no reading yet
    assert {"slot request", "slot start", "slot done"} <= {r["verb"] for r in rows}
    assert any(r["verb"] == "bench.run" for r in rows)


def test_the_t0_text_is_written_for_the_dispatcher(finished):
    answer = finished[0]
    text = Path(answer["t0_path"]).read_text(encoding="utf-8")
    assert text.startswith("- [ ] E5 (kernels) Teacher checkpoint shootout")
    assert "R23, amd Quark AWQ INT4 (B) against RedHat INT4 (A) on pod B" in text
    assert "Exact match of B against A's greedy outputs: 4/4 (100.0%)" in text
    assert text.isascii()                                # no em dash, no emoji


def test_all_takes_the_extension_when_the_plan_needs_it(pod, queue, tmp_path, monkeypatch):
    fake, client = queue
    estimate = qc.estimate
    monkeypatch.setattr(qc, "estimate", lambda plan, prompts: {**estimate(plan, prompts), "needs_extension": True})
    row = qc.take_slot(None, None, client=client, environ={}, now=lambda: NOW)["row"]
    critique(client, row)
    rows = tmp_path / "control.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in control_rows(batches=(1,))), encoding="utf-8")
    answer = pod.run("all", "--control-rows", str(rows), "--row", row, "--reply", "ok", client=client, environ={},
                     now=lambda: NOW)
    result = json.loads(Path(answer["result_path"]).read_text(encoding="utf-8"))
    assert result["extension"]["outcome"] == "extended"
    page = slots.slot_from_page(client.retrieve_page(row))
    assert page.extended and "R23 bench done: 2 ok rows of 2" in page.progress_note
    assert page.status == "Done"


# --- the comparison, deterministic ------------------------------------------------------------------------------

def test_compare_arms_reads_each_batch_size():
    a = control_rows(batches=(1, 4))
    faster = [dict(r, decode_tok_s=r["decode_tok_s"] * 1.5, joules_per_token=r["joules_per_token"] * 0.6)
              for r in control_rows(batches=(1, 4))]
    out = qc.compare_arms(a, faster, [1, 4])
    assert out["b1"]["decode_tok_s"]["verdict"]["label"] == vmath.B_AHEAD
    assert out["b1"]["joules_per_token"]["verdict"]["label"] == vmath.B_AHEAD
    same = qc.compare_arms(a, control_rows(batches=(1, 4), tok_s=(2.95, 2.8, 3.0, 2.9, 2.85)), [1])
    assert same["b1"]["decode_tok_s"]["verdict"]["label"] == vmath.NO_DIFFERENCE
    few = qc.compare_arms(a, control_rows(batches=(1,), n=2), [1])
    assert few["b1"]["decode_tok_s"]["verdict"]["label"] == vmath.TOO_EARLY
    none = qc.compare_arms(a, control_rows(batches=(1,)), [8])
    assert none["b8"]["decode_tok_s"]["verdict"] is None and "arm A" in none["b8"]["decode_tok_s"]["why"]


def test_exact_match_and_the_self_match_floor(tmp_path):
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps(r) + "\n" for r in [
        {"prompt_id": "p1", "text": "the cat sat", "spawn": 1}, {"prompt_id": "p2", "text": "on the mat", "spawn": 1},
        {"prompt_id": "p1", "text": "the cat sat", "spawn": 2}, {"prompt_id": "p2", "text": "on the hat", "spawn": 2},
    ]), encoding="utf-8")
    spawns = qc.load_reference(ref)
    got = qc.exact_match(spawns[1], [{"prompt_id": "p1", "text": "the cat sat"},
                                     {"prompt_id": "p2", "text": "on a mat", "error": None}])
    assert (got["x"], got["n"], got["rate"]) == (1, 2, 0.5)
    assert got["mismatches"][0]["first_divergence_char"] == 3
    assert qc.first_divergence("abc", "abcd") == 3 and qc.first_divergence("x", "x") is None
    missing = qc.exact_match(spawns[1], [{"prompt_id": "p1", "text": "the cat sat"}])
    assert missing["mismatches"] == [{"prompt_id": "p2", "why": "no output"}]


def test_record_reads_a_reference_file_and_two_results(pod, tmp_path, queue):
    """The matrix across two slots: record takes both results; the reference file's two spawns give the floor."""
    first = pod.run("run", batches="1")
    second = pod.run("run", batches="4")
    assert first["ok"] and second["ok"]
    outputs = json.loads(Path(first["result_path"]).read_text())["correctness"]["outputs"]
    ref = tmp_path / "baseline_outputs.jsonl"
    lines = [{"prompt_id": o["prompt_id"], "text": o["text"], "spawn": s} for s in (1, 2) for o in outputs]
    lines[0]["text"] = "something else"                      # arm A's spawn 1 differs on one prompt
    ref.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    rows = tmp_path / "control.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in control_rows(batches=(1, 4))), encoding="utf-8")
    answer = qc.run(["record", "--result", first["result_path"], "--result", second["result_path"],
                     "--control-rows", str(rows), "--reference", str(ref), "--out", str(tmp_path / "rec")],
                    state_dir=tmp_path / "state")
    v = answer["verdict"]
    assert set(v["comparisons"]) == {"b1", "b4"}
    assert (v["exact_match"]["x"], v["exact_match"]["n"]) == (3, 4)
    assert (v["self_match_floor"]["x"], v["self_match_floor"]["n"]) == (3, 4)
    assert v["exact_match_verdict"]["label"] == vmath.TOO_EARLY      # 4 prompts is under the 50-trial floor
    assert not v["same_conditions"]["ok"]                            # R20's stand-in rows carry no conditions
    assert (tmp_path / "rec" / qc.T0_FILE).is_file()


# --- the pod is left as found -----------------------------------------------------------------------------------

def test_hunks_already_in_the_tree_are_left_there(pod, tmp_path):
    loader = tmp_path / "loader.diff"
    loader.write_text(quark_keys.split_patch(pod.patch.read_text(encoding="utf-8"))[1], encoding="utf-8", newline="\n")
    qc.apply_patch(pod.vllm, loader)
    before = git(pod.vllm, "diff")
    answer = pod.run("run", reference=None)
    assert answer["ok"] and answer["patch"] == {"state_before": "applied", "applied_by_run": False,
                                                 "reverted": False}
    assert git(pod.vllm, "diff") == before


def test_a_tree_the_hunks_do_not_fit_is_refused_before_any_spawn(pod):
    (pod.vllm / LOADER_FILE).write_text("something else entirely\n", encoding="utf-8", newline="\n")
    answer = pod.run("run")
    assert answer["outcome"] == "patch_conflict"
    assert not list(pod.out.glob("*/bench"))
    assert not (pod.root / "servers.jsonl").exists()


def test_the_hunks_come_off_even_when_the_bench_fails(pod, monkeypatch):
    def boom(*a, **k):
        raise bench.BenchError("the server died")
    monkeypatch.setattr(bench, "run_session", boom)
    with pytest.raises(bench.BenchError):
        pod.run("run")
    assert git(pod.vllm, "status", "--porcelain") == ""
    result = json.loads(next(pod.out.glob("*/result.json")).read_text())
    assert result["patch"]["reverted"] and result["pod_as_found"]["ok"]


def test_all_releases_the_slot_when_the_run_fails(pod, queue, tmp_path, monkeypatch):
    fake, client = queue
    row = qc.take_slot(None, None, client=client, environ={}, now=lambda: NOW)["row"]
    critique(client, row)
    rows = tmp_path / "control.jsonl"
    rows.write_text(json.dumps(control_rows()[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(bench, "run_session", lambda *a, **k: (_ for _ in ()).throw(bench.BenchError("died")))
    answer = pod.run("all", "--control-rows", str(rows), "--row", row, "--reply", "ok",
                     client=client, environ={}, now=lambda: NOW)
    assert answer["outcome"] == "error" and answer["released"]["outcome"] == "done"
    assert slots.slot_from_page(client.retrieve_page(row)).result.startswith("R23 failed: BenchError")
    assert git(pod.vllm, "status", "--porcelain") == ""


# --- the command line -------------------------------------------------------------------------------------------

def test_the_verb_is_registered_in_order():
    assert VERBS["quark-compare"] == "lab.quark_compare:main"
    assert list(VERBS) == sorted(VERBS)


def test_python_m_lab_quark_compare_preflight(tmp_path):
    ck = make_checkpoint(tmp_path / "ck")
    env = {**os.environ, "LAB_STATE_DIR": str(tmp_path / "state")}
    done = subprocess.run([sys.executable, "-m", "lab", "quark-compare", "preflight", "--checkpoint", str(ck)],
                          cwd=REPO, capture_output=True, text=True, env=env, timeout=60)
    answer = json.loads(done.stdout)
    assert done.returncode == 0, done.stderr
    assert answer["checkpoint"]["ok"] and answer["time"]["fits_with_extension"]
