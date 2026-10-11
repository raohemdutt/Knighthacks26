"""lab/sweep_e18.py, E18 (GPU_MAX_HW_QUEUES=1 against stock): the plan, the launches, a run of both arms on lab/bench_fake.py
spawned as a real server (and a divergent queue1 arm), the summary and the rows, and the pod script.

Nothing here touches a GPU. The servers bind port 0, the pod is lab.bench_fake.FakeHost and FakePower, and assertions are
on structure and on values the fake decides (counts, order, tokens, the variable read back), never on a rate measured on
this machine's clock."""
from __future__ import annotations

import faulthandler
import hashlib
import json
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import bench, bench_fake, correctness
from lab import sweep_e18 as e18
from lab import sweep_request as sr
from lab.machine import metrics
from tests.test_lab_sweep_request import tiny_data

WATCHDOG_S = 240.0
GPU_SHA = hashlib.sha256(b"roscon02").hexdigest()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


def grid_e18() -> sr.Grid:
    return sr.parse_grid(tiny_data(batches=[1, 4]))


def fake_launches(root: Path, **queue1_options) -> dict[str, list[sr.Launch]]:
    def launch(name: str, queues: str, **options) -> sr.Launch:
        argv = bench_fake.serve_command(root=str(root), set_env=[f"GPU_MAX_HW_QUEUES={queues}"], **options)
        return sr.Launch(name, tuple(argv), {}, ())

    return {"stock": [launch("fake", "4")], "queue1": [launch("fake", "1", **queue1_options)]}


def run(tmp_path: Path, arms=("stock", "queue1"), **options):
    root = tmp_path / "pod"
    root.mkdir(parents=True, exist_ok=True)
    host = sr.read_once(bench_fake.FakeHost(root))
    power = bench_fake.FakePower(root)
    grid = grid_e18()
    raw = e18.run_e18(grid, list(arms), host=host, power=power, out_dir=tmp_path / "runs",
                      checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4", engine_dir="/home/jovyan/vllm",
                      patches=[str(e18.REPO / "lab" / "vllm-gfx1201.diff")], port=0, hub="https://sjc.aupcloud.io",
                      account="D", idle_s=0.05, say=lambda message: None,
                      launches=options.pop("launches", fake_launches(root)), **options)
    return raw, grid


# --- the plan -----------------------------------------------------------------------------------------------------

def test_the_plan_is_the_base_cells_at_batch_1_and_4_with_the_grids_repeats_and_two_arms():
    grid = sr.load_grid()
    p = e18.plan(grid, ["stock", "queue1"])
    assert [c["id"] for c in p["cells"]] == ["base-b1", "base-b4"]
    assert p["repeats"] == 3 and p["order"] == "serpentine" and p["passes"] == {"stock": 2, "queue1": 1}
    est = p["estimate"]
    assert est["arms"]["stock"]["ready_s"] == 523.0 and est["arms"]["queue1"]["ready_s"] == e18.WARM_READY_S
    assert p["slot_minutes"] % 5 == 0 and p["slot_minutes"] * 60 >= est["total_s"] * 1.1
    assert p["slot_minutes"] > e18.plan(grid, ["queue1"])["slot_minutes"]
    assert e18.plan(grid, ["queue1"], deadline_min=21)["deadline_min"] == 21


def test_the_arms_differ_in_one_variable_only_and_the_stock_arm_keeps_the_pinned_value():
    grid = sr.load_grid()
    stock = {la.name: la for la in e18.arm_launches(grid, "stock", run_dir="/r")}
    queue1 = {la.name: la for la in e18.arm_launches(grid, "queue1", run_dir="/r")}
    assert stock["pinned"].env["GPU_MAX_HW_QUEUES"] == "4" and queue1["pinned"].env["GPU_MAX_HW_QUEUES"] == "1"
    assert {k: v for k, v in stock["pinned"].env.items() if k != "GPU_MAX_HW_QUEUES"} == {
        k: v for k, v in queue1["pinned"].env.items() if k != "GPU_MAX_HW_QUEUES"}
    assert stock["pinned"].argv == queue1["pinned"].argv
    assert "GPU_MAX_HW_QUEUES" not in stock["stock"].env and queue1["stock"].env == {"GPU_MAX_HW_QUEUES": "1"}


def test_arms_are_a_comma_list_of_known_arms_stock_first():
    assert e18.parse_arms("queue1,stock") == ["stock", "queue1"] and e18.parse_arms("queue1") == ["queue1"]
    for bad in ("", "stock,stock", "queue2"):
        with pytest.raises(sr.SweepError):
            e18.parse_arms(bad)


# --- a run on the fake --------------------------------------------------------------------------------------------

def test_both_arms_run_on_their_own_spawn_with_the_variable_read_back_and_the_outputs_compared(tmp_path):
    raw, grid = run(tmp_path)
    assert raw["error"] is None and list(raw["arms"]) == ["stock", "queue1"]
    stock, queue1 = raw["arms"]["stock"], raw["arms"]["queue1"]
    assert stock["spawn"]["pid"] != queue1["spawn"]["pid"] and stock["spawn"]["stopped"] and queue1["spawn"]["stopped"]
    for arm in (stock, queue1):
        assert len(arm["rows"]) == 6 and all(r["outcome"] == "ok" for r in arm["rows"])
        assert [r["cell"] for r in arm["rows"]] == ["base-b1", "base-b4", "base-b4", "base-b1", "base-b1", "base-b4"]
        assert arm["probe"]["decode_tok_s"] is not None
    assert len(stock["outputs"]) == 2 and len(queue1["outputs"]) == 1
    assert all(len(p["outputs"]) == 4 and not any(o["error"] for o in p["outputs"])
               for p in stock["outputs"] + queue1["outputs"])
    summary = e18.summarize(raw)
    assert summary["arms"]["stock"]["env_gpu_max_hw_queues"] == "4"
    assert summary["arms"]["queue1"]["env_gpu_max_hw_queues"] == "1"
    assert [c["cell"] for c in summary["compared"]] == ["base-b1", "base-b4"]
    out = summary["outputs"]
    assert out["stock_vs_stock"]["summary"]["matches"] == 4 and out["queue1_vs_stock"][0]["summary"]["matches"] == 4
    assert out["unchanged"] is True and out["identical"] is True
    assert summary["reading"]["reading"] in ("no_difference", "b_ahead", "a_ahead")
    assert summary["bands"]["classified"] is False and "not a finding" in summary["bands"]["note"]
    assert not [o for o in summary["open"] if "not valid" in o or "reads GPU_MAX_HW_QUEUES" in o]


def test_a_queue1_arm_whose_outputs_diverge_is_reported_against_the_stock_floor(tmp_path):
    root = tmp_path / "pod"
    root.mkdir(parents=True)
    raw, grid = run(tmp_path, launches=fake_launches(root, diverge_at=5))
    out = e18.summarize(raw)["outputs"]
    assert out["stock_vs_stock"]["summary"]["matches"] == 4
    first = out["queue1_vs_stock"][0]["summary"]
    assert first["divergences"] == 4 and first["first_divergence_index"] == 5
    assert out["unchanged"] is False and out["identical"] is False and out["best_match_rate"] == 0.0


def test_a_wrong_variable_in_the_server_environment_makes_the_arm_invalid(tmp_path):
    root = tmp_path / "pod"
    root.mkdir(parents=True)
    wrong = fake_launches(root)
    wrong["queue1"] = wrong["stock"]            # queue1's server is started with 4
    raw, _ = run(tmp_path, launches=wrong)
    assert any("not '1'" in item for item in e18.summarize(raw)["open"])


def test_queue1_alone_compares_with_the_stock_outputs_an_earlier_run_left(tmp_path):
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    (tmp_path / "pod").mkdir()
    raw1, _ = run(first_dir, arms=("stock",), outputs_dir=first_dir / "home")
    left = first_dir / "home" / "stock-outputs.json"
    assert left.is_file() and len(json.loads(left.read_text(encoding="utf-8"))["passes"]) == 2
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    raw2, _ = run(second_dir, arms=("queue1",), reference=left)
    assert list(raw2["arms"]) == ["queue1"] and raw2["reference"]["run_id"] == raw1["arms"]["stock"]["run_id"]
    out = e18.summarize(raw2)["outputs"]
    assert out["stock_source"]["from"] == "an earlier run's file"
    assert [c["against_stock_pass"] for c in out["queue1_vs_stock"]] == [1, 2] and out["unchanged"] is True
    assert e18.summarize(raw2)["compared"] == []     # no stock rows in this run: no speed comparison, said so
    assert e18.summarize(raw2)["reading"]["reading"] == "not_measured"


def test_an_arm_that_cannot_be_finished_before_the_deadline_is_skipped_with_the_reason(tmp_path):
    root = tmp_path / "pod"
    root.mkdir()
    host = sr.read_once(bench_fake.FakeHost(root))
    power = bench_fake.FakePower(root)
    raw = e18.run_e18(grid_e18(), ["stock", "queue1"], host=host, power=power, out_dir=tmp_path / "runs",
                      checkpoint="ckpt", port=0, idle_s=0.05, say=lambda m: None, launches=fake_launches(root),
                      deadline_s=40.0)    # the second arm's modelled seconds do not fit what is left
    assert "rows" in raw["arms"]["stock"] and "not started" in raw["arms"]["queue1"]["skipped"]
    assert any("queue1: not started" in item for item in e18.summarize(raw)["open"])


# --- the rows and the result block --------------------------------------------------------------------------------

def test_the_ledger_and_metrics_rows_name_the_arm_and_are_low_confidence_and_valid(tmp_path):
    raw, _ = run(tmp_path)
    summary = e18.summarize(raw)
    ledger = e18.ledger_rows(summary)
    assert [r["verb"] for r in ledger] == ["sweep_e18.cell"] * 4 + ["sweep_e18.outputs"]
    assert {(r["conditions"]["arm"], r["conditions"]["cell"]) for r in ledger[:4]} == {
        (a, c) for a in ("stock", "queue1") for c in ("base-b1", "base-b4")}
    assert all(r["node"] == "pod-b" and r["low_confidence"] for r in ledger)
    assert ledger[-1]["metrics"]["teacher.output_match_rate"] == 1.0
    rows = e18.metric_rows(summary)
    assert {r["conditions"].get("arm") for r in rows if r["metric"] == "teacher.decode_tok_s"} == {"stock", "queue1"}
    assert {"teacher.decode_tok_s_change_pct", "teacher.joules_per_token_change_pct",
            "teacher.output_match_rate"} <= {r["metric"] for r in rows}
    for row in rows:
        metrics.check_row(row)
    assert "Reading:" in e18.render_markdown(summary) and "| `base-b1` |" in e18.render_markdown(summary)


def test_the_result_block_round_trips_and_collect_writes_the_files_and_rows(tmp_path):
    raw, _ = run(tmp_path)
    out = tmp_path / "pod_run.out"
    out.write_text(e18.report(raw), encoding="utf-8")
    files = sr.read_bundle(out.read_bytes())
    assert json.loads(files["raw.json"])["run_id"] == raw["run_id"]
    assert any(name.startswith("stock-serve-") for name in files) and any(name.startswith("queue1-serve-") for name in files)
    data_dir, state_dir, mfile = tmp_path / "data", tmp_path / "keep", tmp_path / "metrics.jsonl"
    answer = e18.main(["collect", str(out), "--out-dir", str(data_dir), "--state-dir", str(state_dir),
                       "--metrics-file", str(mfile)])
    assert answer["ok"] and answer["outputs_identical"] is True and answer["ledger_rows"] == 5
    assert answer["metrics_rows"] == len(mfile.read_text(encoding="utf-8").splitlines())
    names = sorted(p.name for p in data_dir.iterdir())
    assert len(names) == 2 and names[0].startswith("e18-hwqueues-gfx1151-") and {Path(n).suffix for n in names} == {".json", ".md"}
    again = e18.main(["collect", str(out), "--out-dir", str(data_dir), "--state-dir", str(state_dir), "--no-ledger"])
    assert not again["ok"] and again["outcome"] == "already_collected"


# --- the pod script and the gate ---------------------------------------------------------------------------------

def test_the_pod_script_carries_this_module_the_correctness_harness_and_the_prompt_set_and_no_tab_or_bang():
    grid = sr.load_grid()
    script, info = e18.pod_script(grid, arms=["stock", "queue1"], expect_user_sha256=GPU_SHA, deadline_min=45,
                                  hub="https://sjc.aupcloud.io", account="D", node="pod-b")
    assert "\t" not in script and "!" not in script and script.isascii()
    assert "python3 -m lab.sweep_e18 pod-run --expect-user-sha256 " + GPU_SHA in script
    assert "--arms stock,queue1" in script and "--deadline-min 45" in script and script.rstrip().endswith("exit $rc")
    assert {"lab/sweep_e18.py", "lab/sweep_request.py", "lab/correctness.py", "lab/bench.py", "lab/vllm-gfx1201.diff",
            "fixtures/lab/bench/prompts.json", "fixtures/lab/bench/images/bars_1280x720.png",
            "fixtures/lab/bench/images/shapes_640x480.png"} <= set(info["files"])
    assert info["pod_args"][-2:] == ["--arms", "stock,queue1"] or "--arms" in info["pod_args"]


def test_the_pod_run_refuses_a_machine_that_is_not_pod_b_gfx1151_and_account_d(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "read_rocminfo", lambda: "Name: gfx1201")
    monkeypatch.setenv("JUPYTERHUB_USER", "someone-else")
    args = e18.build_parser().parse_args(["pod-run", "--expect-user-sha256", GPU_SHA])
    answer = e18.pod_run_main(args)
    assert answer["outcome"] == "refused" and any("gfx1201" in p for p in answer["problems"])
    assert any("another hub account" in p for p in answer["problems"])


def test_the_verb_is_registered():
    assert lab_cli.VERBS["sweep-e18"] == "lab.sweep_e18:main"
