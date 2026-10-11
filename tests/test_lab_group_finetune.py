"""lab/group_finetune.py (L10): a group fine-tune from the pooled opt-in exports of a Circle's members.

One test per clause of the Done when, on the two-member fixture pool under fixtures/learning/group-pool/ (copied to a
temporary folder first, because the run deletes the pooled data). Every outside call has a fake: the teacher is
lab/teacher_fake.py behind the real lab/label.py client, the trainer is lab/train_fake.py through
lab/group_finetune_fake.FakeTrainer, the packaging toolchain is lab/package_toolchain_fake.py, the gate's scorer is
lab/eval_gate_fake.py, the model servers and the boxes are lab/group_finetune_fake.py. No GPU, pod, hub or network.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import lab.__main__ as dispatcher
from lab import group_finetune as gf
from lab.dataset import load, load_split
from lab.eval_gate_fake import FakeClient
from lab.group_finetune_fake import FakeBox, FakeServers, FakeTrainer
from lab.label import Teacher
from lab.machine import ledger
from lab.machine import metrics as metrics_file
from lab.package_toolchain_fake import FakeToolchain
from lab.teacher_fake import FakeTeacher, text_policy

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "learning" / "group-pool"
CIRCLE, TASK = "circle-knights", "desk-objects"
UPLOADERS = ("ilan", "dom")
EVERYONE = ("ilan", "dom", "rao")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "group-finetune-test")
    monkeypatch.delenv("LAB_TASK", raising=False)
    for name in ("JUPYTERHUB_USER", "JUPYTERHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def pool_root(tmp_path) -> Path:
    root = tmp_path / "pool"
    shutil.copytree(FIXTURE, root)
    return root


@pytest.fixture
def pod(tmp_path):
    """The trainer pod's folders as packaging expects them: the bf16 base, its projector, llama.cpp."""
    base = tmp_path / "models" / "Qwen3.5-4B"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    mmproj = tmp_path / "models" / "mmproj-F16.gguf"
    mmproj.write_bytes(b"GGUF" + b"projector " * 50)
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    return {"base_dir": base, "mmproj": mmproj, "llama_dir": llama}


def everyone_passes_candidate(dataset: Path) -> dict[str, list[set[str]]]:
    held = {e.id for e in load_split(dataset, require_images=False, min_held_out=1).held_out}
    outcomes = {FakeServers.current_url(m): [set()] for m in EVERYONE}
    outcomes.update({FakeServers.CANDIDATE: [held], FakeServers.TEACHER: [held]})
    return outcomes


def run(pool_root, tmp_path, pod, **more):
    fake = FakeTeacher(text_policy("mug"))
    clients: list[FakeClient] = []

    def gate_client(task, dataset):
        client = FakeClient(dataset, everyone_passes_candidate(Path(dataset)), by_id=True)
        clients.append(client)
        return client

    trainer = more.pop("trainer", FakeTrainer())
    with Teacher(fake.base_url, transport=fake.transport, sleep=lambda s: None) as teacher:
        result = gf.run_group_finetune(
            pool_root=pool_root, circle=CIRCLE, task=TASK, run_id="g1", work=tmp_path / "work",
            packages=tmp_path / "packages", teacher=teacher, trainer=trainer, gate_client=gate_client,
            servers=FakeServers(), toolchain=FakeToolchain(), model=str(pod["base_dir"]), min_held_out=1,
            decode_attested=True, metrics_path=tmp_path / "metrics.jsonl", **pod, **more)
    return result, fake, trainer, clients


def rows(verb):
    return [r for r in ledger.read() if r["verb"] == verb]


# --- the pool: two members' approved exports for one task, and the Circle roster -------------------------------------

def test_the_pool_lists_every_member_and_counts_each_ones_approved_exports(pool_root):
    pool = gf.read_pool(pool_root, CIRCLE, TASK)
    assert [m.member for m in pool.members] == list(EVERYONE)
    counts = {m.member: m.examples for m in pool.members}
    assert counts == {"ilan": 8, "dom": 8, "rao": 0}
    assert all(e.action_id for m in pool.members for e in m.exports)
    assert pool.uploaders == ["ilan", "dom"]


def test_the_pool_refuses_an_export_of_another_task_and_a_folder_outside_the_roster(pool_root):
    folder = pool_root / CIRCLE / TASK
    manifest = folder / "dom" / "desk-objects-dom-0001" / "manifest.json"
    body = json.loads(manifest.read_text())
    body["spec"]["task"] = "spot-qa"
    manifest.write_text(json.dumps(body))
    shutil.copytree(folder / "ilan", folder / "stranger")
    with pytest.raises(gf.GroupError) as err:
        gf.read_pool(pool_root, CIRCLE, TASK)
    assert "spot-qa" in str(err.value) and "stranger" in str(err.value)


def test_add_export_copies_an_approved_export_into_the_pool_and_counts_it(pool_root, tmp_path):
    export = tmp_path / "data" / "exports" / "desk-objects-rao-0001"
    shutil.copytree(pool_root / CIRCLE / TASK / "ilan" / "desk-objects-ilan-0001", export)
    manifest = json.loads((export / "manifest.json").read_text())
    manifest["dataset_id"] = "desk-objects-rao-0001"
    (export / "manifest.json").write_text(json.dumps(manifest))
    gf.add_export(pool_root, CIRCLE, TASK, "rao", export)
    pool = gf.read_pool(pool_root, CIRCLE, TASK)
    assert {m.member: m.examples for m in pool.members}["rao"] == 8
    written = json.loads((pool_root / CIRCLE / TASK / gf.POOL_FILE).read_text())
    assert {m["member"]: m["exports"] for m in written["members"]}["rao"] == {"desk-objects-rao-0001": 8}


# --- pooling: one dataset, one frozen split with every uploader on the held-out side --------------------------------

def test_the_pooled_dataset_holds_every_uploaded_example_and_a_split_with_each_member_held_out(pool_root, tmp_path):
    pool = gf.read_pool(pool_root, CIRCLE, TASK)
    merged = gf.merge_pool(pool, tmp_path / "work" / "data" / "exports", run_id="g1")
    data = load(merged.root)
    assert data.manifest.examples == 16 and len(data.examples) == 16
    assert "ilan:approval-ilan-0001" in data.manifest.action_id and "dom:approval-dom-0001" in data.manifest.action_id
    split = load_split(merged.root, min_held_out=1)
    held_members = {merged.member_of[e.id] for e in split.held_out}
    assert held_members == set(UPLOADERS)
    episodes = json.loads(merged.episodes.read_text())
    assert set(episodes) == {e.frame_ref for e in data.examples}


# --- the whole run on fakes -------------------------------------------------------------------------------------------

def test_the_teacher_labels_the_pooled_set_and_one_lora_trains_on_its_train_side(pool_root, tmp_path, pod):
    result, fake, trainer, _ = run(pool_root, tmp_path, pod)
    assert result["ok"], result
    chats = [r for r in fake.requests if r.path.endswith("/chat/completions")]
    assert len(chats) == 16                       # every pooled example labeled, through lab/label.py
    assert len(trainer.calls) == 1                # one LoRA, on the pooled set
    call = trainer.calls[0]
    assert call["rows"] == result["counts"]["train_rows"] == 12
    assert call["config"] == gf.TrainConfig().fields()
    assert result["labels"]["accepted"] == 16


def test_the_gate_scores_the_group_model_against_each_members_current_model(pool_root, tmp_path, pod):
    result, _, _, clients = run(pool_root, tmp_path, pod)
    assert set(result["scores"]) == set(EVERYONE)
    scored = [url for client in clients for url in client.calls]
    for member in EVERYONE:
        assert FakeServers.current_url(member) in scored
        assert set(result["scores"][member]) >= {"current", "candidate", "teacher"}
    gate_rows = rows("eval-gate")
    assert len(gate_rows) == 3 and {r["refs"]["run_id"] for r in gate_rows} == {f"g1.{m}" for m in EVERYONE}


def test_the_package_carries_a_group_offer_for_every_member_with_rollback(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod)
    package = Path(result["package"])
    assert (package / "manifest.json").is_file()
    offer = json.loads((package / gf.OFFER_FILE).read_text())
    assert offer["model_version"] == result["model_version"] and offer["circle"] == CIRCLE
    members = {m["member"]: m for m in offer["members"]}
    assert set(members) == set(EVERYONE)
    assert members["rao"]["uploaded_examples"] == 0
    assert all(m["rollback_version"] == "atlas-base" for m in members.values())
    assert all(m["gate"]["decision"] in ("deploy", "reject") for m in members.values())


def test_ledger_and_metrics_rows_record_members_counts_run_and_scores(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod)
    row = rows("group-finetune.run")[-1]
    assert row["outcome"] == "ok" and row["refs"]["run_id"] == "g1"
    assert row["members"] == list(EVERYONE)
    assert row["counts"]["examples_by_member"] == {"ilan": 8, "dom": 8, "rao": 0}
    assert set(row["scores"]) == set(EVERYONE)
    assert row["metrics"]["group.examples"] == 16
    written = metrics_file.load(tmp_path / "metrics.jsonl")
    assert {r["conditions"].get("member") for r in written if r["metric"] == "group_finetune.candidate_delta"} \
        == set(EVERYONE)


def test_the_pooled_data_is_deleted_after_the_run_and_the_deletion_recorded(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod)
    folder = pool_root / CIRCLE / TASK
    assert sorted(p.name for p in folder.iterdir()) == [gf.POOL_FILE]
    assert not (tmp_path / "work" / "data").exists()
    deleted = rows("group-finetune.delete")[-1]
    assert deleted["outcome"] == "deleted" and deleted["refs"]["run_id"] == "g1"
    assert result["data_deleted"] == "yes"


def test_the_pooled_data_is_deleted_even_when_training_fails(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod, trainer=FakeTrainer(fail=True))
    assert not result["ok"] and result["outcome"] == "failed"
    assert sorted(p.name for p in (pool_root / CIRCLE / TASK).iterdir()) == [gf.POOL_FILE]
    assert rows("group-finetune.run")[-1]["outcome"] == "failed"
    assert rows("group-finetune.delete")[-1]["outcome"] == "deleted"


def test_keep_leaves_the_pool_for_debugging_and_records_that(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod, keep=True)
    assert result["ok"] and (pool_root / CIRCLE / TASK / "ilan").is_dir()
    assert rows("group-finetune.delete")[-1]["outcome"] == "kept"


def test_keep_is_refused_on_a_pod(pool_root, tmp_path, pod, monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_USER", "roscon02")
    with pytest.raises(gf.GroupError, match="pod"):
        run(pool_root, tmp_path, pod, keep=True)


# --- dry run ----------------------------------------------------------------------------------------------------------

def test_dry_run_prints_the_plan_and_writes_nothing(pool_root, tmp_path):
    before = sorted(str(p) for p in pool_root.rglob("*"))
    pool_json = (pool_root / CIRCLE / TASK / gf.POOL_FILE).read_text()
    answer = dispatcher.VERBS["group-finetune"]
    assert answer == "lab.group_finetune:main"
    out = gf.main(["--circle", CIRCLE, "--task", TASK, "--pool", str(pool_root), "--run-id", "g1",
                   "--work", str(tmp_path / "work"), "--dry-run"])
    assert out["ok"] and out["outcome"] == "dry-run"
    assert out["plan"]["members"] == {"ilan": 8, "dom": 8, "rao": 0}
    assert [s["step"] for s in out["plan"]["steps"]] == list(gf.STEPS)
    assert sorted(str(p) for p in pool_root.rglob("*")) == before
    assert (pool_root / CIRCLE / TASK / gf.POOL_FILE).read_text() == pool_json
    assert not (tmp_path / "work").exists() and ledger.read() == []


# --- the boxes take the offer -----------------------------------------------------------------------------------------

def test_a_member_who_uploaded_nothing_can_take_the_group_model_and_roll_back(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod)
    offer = Path(result["package"]) / gf.OFFER_FILE
    box = FakeBox()
    taken = gf.take_offer(offer, "rao", box)
    assert taken["ok"] and taken["version"] == result["model_version"]
    assert box.calls == [("pull", result["model_version"]), ("swap", result["model_version"])]
    back = gf.take_offer(offer, "rao", box, rollback=True)
    assert back["ok"] and box.calls[-1] == ("rollback", "atlas-base")


def test_a_failed_swap_rolls_back_and_a_stranger_is_refused(pool_root, tmp_path, pod):
    result, *_ = run(pool_root, tmp_path, pod)
    offer = Path(result["package"]) / gf.OFFER_FILE
    box = FakeBox(swap_fails=True)
    taken = gf.take_offer(offer, "dom", box)
    assert not taken["ok"] and box.calls[-1] == ("rollback", "atlas-base")
    with pytest.raises(gf.GroupError, match="not a member"):
        gf.take_offer(offer, "mallory", FakeBox())


def big_pool(root: Path, frames: int = 100) -> Path:
    """Two uploaders with `frames` one-example frames each (enough held out for a gate that may deploy), and rao."""
    from datetime import UTC, datetime

    from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
    folder = root / CIRCLE / TASK
    spec = TaskSpec(task=TASK, instruction="Name the one object on the desk in this picture.")
    now = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
    for member in UPLOADERS:
        export = folder / member / f"desk-objects-{member}-big"
        (export / "images").mkdir(parents=True)
        examples = []
        for k in range(frames):
            (export / "images" / f"{k:04d}.png").write_bytes(b"\x89PNG fake " + f"{member}{k}".encode())
            examples.append(DatasetExample(id=f"{member}-{k}", image=f"images/{k:04d}.png",
                                           frame_ref=f"frames:{member}:{k}", ts=now, context="An object."))
        manifest = DatasetManifest(dataset_id=export.name, spec=spec, created=now, examples=frames,
                                   action_id=f"approval-{member}")
        (export / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        (export / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples),
                                               encoding="utf-8", newline="\n")
    (folder / gf.POOL_FILE).write_text(json.dumps({"version": 1, "circle": CIRCLE, "task": TASK, "members": [
        {"member": m, "current_version": f"atlas-{m}-v1", "exports": {}} for m in EVERYONE]}), encoding="utf-8")
    return root


def test_the_gate_recommends_the_group_model_only_to_the_members_it_beats(tmp_path, pod):
    root = big_pool(tmp_path / "bigpool")
    fake = FakeTeacher(text_policy("stapler"))

    def gate_client(task, dataset):
        held = {e.id for e in load_split(dataset, require_images=False, min_held_out=1).held_out}
        outcomes = {FakeServers.current_url("ilan"): [set()], FakeServers.current_url("dom"): [held],
                    FakeServers.current_url("rao"): [set()], FakeServers.CANDIDATE: [held],
                    FakeServers.TEACHER: [held]}
        return FakeClient(dataset, outcomes, by_id=True)

    with Teacher(fake.base_url, transport=fake.transport, sleep=lambda s: None) as teacher:
        result = gf.run_group_finetune(
            pool_root=root, circle=CIRCLE, task=TASK, run_id="g2", work=tmp_path / "work",
            packages=tmp_path / "packages", teacher=teacher, trainer=FakeTrainer(), gate_client=gate_client,
            servers=FakeServers(), toolchain=FakeToolchain(), model=str(pod["base_dir"]), decode_attested=True,
            metrics_path=tmp_path / "metrics.jsonl", **pod)
    assert result["ok"], result
    offer = json.loads((Path(result["package"]) / gf.OFFER_FILE).read_text())
    status = {m["member"]: m["status"] for m in offer["members"]}
    assert status == {"ilan": "recommended", "dom": "available", "rao": "recommended"}
    assert {m["member"]: m["rollback_version"] for m in offer["members"]}["rao"] == "atlas-rao-v1"


def test_the_desk_gate_client_scores_held_out_pictures_and_sends_the_hub_token_to_the_teacher(pool_root, tmp_path):
    import httpx
    pool = gf.read_pool(pool_root, CIRCLE, TASK)
    merged = gf.merge_pool(pool, tmp_path / "data" / "exports", run_id="g3")
    examples = [json.loads(line) for line in (merged.root / "examples.jsonl").read_text().splitlines()]
    from lab.dataset import write_targets
    write_targets(merged.root, {e["id"]: "mug" for e in examples})
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "A mug."}}]})

    client = gf.DeskGateClient(merged.root, teacher_url="http://teacher.test", transport=httpx.MockTransport(handler),
                               teacher_headers={"Authorization": "token sekret"})
    out = client.score("http://teacher.test", model_version="teacher")
    held = load_split(merged.root, min_held_out=1).held_out
    assert [r["id"] for r in out["results"]] == [e.id for e in held]
    assert all(r["answer_ok"] for r in out["results"]) and out["score"] == 1.0
    assert all(auth == "token sekret" for _, auth in seen)
    seen.clear()
    client.score("http://current.test")
    assert all(auth is None for _, auth in seen)
    assert isinstance(gf.gate_client_for("desk-objects", merged.root), gf.DeskGateClient)


@pytest.mark.skipif(shutil.which("bash") is None or __import__("os").name == "nt", reason="needs bash on a POSIX host")
def test_the_pod_script_runs_only_inside_a_gpu_slot_and_with_the_teacher_token(tmp_path):
    import subprocess
    script = REPO / "lab" / "pod_b_group_finetune.sh"
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    no_slot = subprocess.run(["bash", str(script), "run", CIRCLE, TASK, "g9"], env=env, capture_output=True, text=True)
    assert no_slot.returncode != 0 and "GPU_SLOT" in no_slot.stderr
    no_token = subprocess.run(["bash", str(script), "run", CIRCLE, TASK, "g9"], env={**env, "GPU_SLOT": "row-1"},
                              capture_output=True, text=True)
    assert no_token.returncode != 0 and "TEACHER_TOKEN" in no_token.stderr
