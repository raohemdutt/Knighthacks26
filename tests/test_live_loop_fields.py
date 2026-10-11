"""The live loop relay posts the run card's missing fields and marks the stand-in gate (tooling/demo/live_loop.py):
samples per training step, energy per training example and the package's file size, through the existing import
kinds (run, manifest, compare) as added fields, and `gate: {stand_in: true, note}` on the stand-in gate's scores.
Run against the rehearsal's files (contexts/learning/tests/rehearsal.py) and the fake box of test_tooling_live_loop."""

import json
import os
import shutil
import subprocess

import pytest

from contexts.learning.tests import rehearsal
from tests import test_tooling_live_loop as base
from tests.test_tooling_live_loop_pod import POD, git_bash



@pytest.fixture(scope="module")
def live_loop():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("live_loop", base.ROOT / "tooling" / "demo" / "live_loop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_loop"] = module
    spec.loader.exec_module(module)
    return module


RUN_JSON = {"ok": True, "outcome": "trained",
            "refs": {"run_id": rehearsal.SOLO, "task": "desk-objects", "node": "pod-d", "base_model": "Qwen/Qwen3.5-4B"},
            "metrics": {"trainer.steps": 8, "trainer.batch": 1, "trainer.grad_accum": 4, "trainer.examples_seen": 32,
                        "trainer.examples": 75, "trainer.lora_rank": 16}}


class FakeSource:
    """A pod's files by path, read only."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def read(self, path: str) -> bytes | None:
        return self.files.get(path)


def solo_row() -> dict:
    return json.loads(rehearsal.text("comparison.json"))["acts"]["solo"]


def test_the_card_fields_come_from_the_run_json_the_training_window_and_the_done_line(live_loop):
    stages = [{"act": "solo", "stage": "done", "ts": "2026-10-11T08:06:30Z", "rc": 0, "package_bytes": 2_740_000_000},
              {"act": "duo", "stage": "done", "ts": "2026-10-11T08:07:30Z", "rc": 0, "package_bytes": None}]
    assert live_loop.package_bytes(stages, "solo") == 2_740_000_000
    assert live_loop.package_bytes(stages, "duo") is None and live_loop.package_bytes([], "solo") is None
    got = live_loop.card_fields(RUN_JSON, None, solo_row(), 2_740_000_000)
    assert got["batch"] == 1 and got["grad_accum"] == 4 and got["samples_per_step"] == 4
    assert got["train_joules"] == 2727.0 and got["train_examples"] == 32
    assert got["j_per_example"] == pytest.approx(2727.0 / 32, abs=0.01)
    assert got["package_bytes"] == 2_740_000_000
    # a two-pod act counts the examples both ranks saw; nothing in the files gives None, never a guess
    two = live_loop.card_fields(RUN_JSON, {"metrics": {"trainer.examples_seen": 32}}, solo_row(), None)
    assert two["train_examples"] == 64 and two["j_per_example"] == pytest.approx(2727.0 / 64, abs=0.01)
    empty = live_loop.card_fields(None, None, None, None)
    assert {k: empty[k] for k in ("samples_per_step", "train_joules", "j_per_example", "package_bytes")} == dict.fromkeys(
        ("samples_per_step", "train_joules", "j_per_example", "package_bytes"))


def test_the_stand_in_gate_is_marked_and_a_real_gate_is_untouched(live_loop):
    scores = json.loads(rehearsal.text("solo/scores.json"))
    assert live_loop.is_stand_in(scores)
    marked = live_loop.mark_stand_in(scores)
    assert marked["gate"]["stand_in"] is True and "not a real held-out check" in marked["gate"]["note"]
    assert {k: v for k, v in marked.items() if k != "gate"} == scores   # nothing else changes
    assert "gate" not in scores   # the input is not edited in place
    real = json.loads(json.dumps(scores))
    for item in real["items"]:
        item["before"]["answer"], item["after"]["answer"] = "a laptop", "a mug"
    assert not live_loop.is_stand_in(real) and live_loop.mark_stand_in(real) == real
    assert live_loop.mark_stand_in(real, known=True)["gate"]["stand_in"] is True   # the relay knows the pod ran GATE=fake


def test_the_relay_posts_the_fields_through_run_manifest_eval_and_compare(live_loop, tmp_path):
    client, transport = base.fake_box(tmp_path)
    box = live_loop.Box("http://box:8000", None, transport=transport)
    run_id, version = rehearsal.SOLO, f"atlas-group-{live_loop.TASK}-{rehearsal.SOLO}"
    lines = live_loop.act_lines(rehearsal.jsonl("stages.jsonl"), "solo")
    assert box.post("stage", "".join(json.dumps(x) + "\n" for x in lines).encode(), run_id, {"task": live_loop.TASK}).ok
    assert box.post("telemetry", rehearsal.raw("solo/telemetry.jsonl"), run_id).ok
    source = FakeSource({f"data/packages/{version}/manifest.json": rehearsal.raw("solo/manifest.json"),
                         f"group-runs/{run_id}/gate/ilan/scores.json": rehearsal.raw("solo/scores.json"),
                         f"group-runs/{run_id}/runs/{run_id}/run.json": json.dumps(RUN_JSON).encode()})

    said = live_loop.post_manifest(source, box, run_id, version, 2_740_000_000, tmp_path / "out")
    assert said.startswith("box 200"), said
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["bytes"] == 2_740_000_000
    said = live_loop.post_scores(source, box, run_id, tmp_path / "out", stand_in=True)
    assert said.startswith("box 200"), said
    posted = json.loads((tmp_path / "out" / "scores.json").read_text())
    assert posted["gate"]["stand_in"] is True and posted["p_value"] == json.loads(rehearsal.text("solo/scores.json"))["p_value"]

    run_json = live_loop.read_json(source, f"group-runs/{run_id}/runs/{run_id}/run.json")
    extras = live_loop.card_fields(run_json, None, solo_row(), 2_740_000_000)
    said = live_loop.post_run(box, run_id, run_json, extras, tmp_path / "out")
    assert said.startswith("box 200"), said
    sent = json.loads((tmp_path / "out" / "run.json").read_text())
    assert sent["live_loop"]["samples_per_step"] == 4 and sent["live_loop"]["package_bytes"] == 2_740_000_000
    assert sent["metrics"]["live_loop.j_per_example"] == extras["j_per_example"]
    assert {k: v for k, v in sent["metrics"].items() if not k.startswith("live_loop.")} == RUN_JSON["metrics"]

    # the box now has the trainer's batch (samples per step), so the card's samples and energy per sample are numbers
    story = client.get(f"/api/learning/lab/{run_id}").json()
    assert story["training"]["batch"] == 4
    assert story["training"]["samples"]["value"] == 32
    assert story["power"]["j_per_sample"]["value"] is not None

    doc = json.loads(rehearsal.text("comparison.json"))
    live_loop.add_card_fields(doc, {"solo": extras})
    assert doc["acts"]["solo"]["card"]["samples_per_step"] == 4 and "card" not in doc["acts"]["duo"]
    assert live_loop.post_comparison(box, doc).startswith("box 200")


@pytest.mark.skipif(git_bash() is None, reason="no bash here")
def test_the_pod_writes_the_package_size_on_the_done_line(tmp_path):
    shutil.copy(POD, tmp_path / "pod.sh")
    home = tmp_path / "home"
    verb = home / "l10" / "src" / "lab" / "pod_b_group_finetune.sh"
    verb.parent.mkdir(parents=True)
    verb.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (home / "l29" / "s").mkdir(parents=True)
    pkg = home / "data" / "packages" / "atlas-group-desk-objects-live-s-solo"
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(json.dumps({"version": pkg.name, "gguf": "m-Q4_K_M.gguf", "mmproj": "mmproj.gguf"},
                                                  indent=2), encoding="utf-8")
    (pkg / "m-Q4_K_M.gguf").write_bytes(b"g" * 12345)
    cmd = ('source ./pod.sh none > /dev/null; RUN=home/l29/s; STAGES="$RUN/stages.jsonl"; '
           'act solo l29-solo live-s-solo 1; act duo l29-duo live-s-duo 2')
    done = subprocess.run([git_bash(), "-c", cmd], env={**os.environ, "HOME": "home", "PEER_IP": "100.64.0.9"},
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(x) for x in (home / "l29" / "s" / "stages.jsonl").read_text(encoding="utf-8").splitlines()]
    ends = {x["act"]: x for x in lines if x["stage"] == "done"}
    assert ends["solo"]["package_bytes"] == 12345 and ends["solo"]["rc"] == 0
    assert ends["duo"]["package_bytes"] is None   # no package: null, never a guess
