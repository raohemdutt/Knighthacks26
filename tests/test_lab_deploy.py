"""python -m lab deploy (L6): the box pulls the packaged model from pod B through the hub proxy, verifies it, hot-swaps,
publishes learning.model.deployed, and goes back to the serving version when the new one does not answer.

What is real: the pod's file server (lab/package_server.py on 127.0.0.1, serving the fixture package from
fixtures/learning/package/), the Range pull (lab/package_pull.py), the event log (bus/bus.py), and in the last tests the
model store and jetson/atlas_swap.py's swap. What is faked: the hub and its proxy (lab/hub_fake.py,
lab/package_fake.py), the box's llama-server and app (an httpx MockTransport), the swap and restart (FakeBox).

Hashing once: the pull writes <version>/.verified.json after its sha256 pass, and the verify and the swap trust it
while each file keeps its recorded size and mtime. The tests count every full-file sha256 (the pull's and the box's)
with the record present, absent and stale, and with --no-trust."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import threading
from pathlib import Path

import httpx
import pytest

from contracts.events import parse_event
from contracts.learning import ModelDeployed, ModelManifest, RunProgress
from lab import deploy, package_pull, package_server
from lab.hub_fake import FakeClock, FakeHub
from lab.package_fake import FakePodProxy, write_package

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "learning" / "package" / "atlas-desk-objects-fixture"
VERSION = FIXTURE.name
OLD_ON_POD = "atlas-desk-objects-older"
TOKEN = "eeee5555ffff6666aaaa7777bbbb8888"
USER = "acct-d"
PART_MIB = str(16_384 / (1 << 20))          # 16 KiB parts: the 40 KB fixture model comes in three


class FakeBox:
    """data/models as the swap sees it, without symlinks: current and previous are attributes, verify is the store's
    rule (every manifest file present and matching its sha256, or vouched for by the pull's .verified.json when
    trust_verified and its size and mtime still match), swap checks twice as the real one does (the swap's check and
    the switch's), swap and swap_back are recorded. hashed lists every file the box sha256'd."""

    def __init__(self, models_dir: Path, current: str | None = "atlas-base", previous: str | None = "atlas-v1") -> None:
        self.models_dir = models_dir
        self._current, self._previous = current, previous
        self.calls: list[tuple] = []
        self.trusts: list[bool] = []
        self.hashed: list[str] = []
        self.swap_fails: str | None = None
        self.back_problem: str | None = None

    def current(self):
        return self._current

    def previous(self):
        return self._previous

    def verify(self, version, *, trust_verified=False):
        from runtime.modelstore import read_verified, record_vouches
        m = ModelManifest.model_validate_json((self.models_dir / version / "manifest.json").read_text())
        record = read_verified(self.models_dir / version) if trust_verified else {}
        problems = []
        for name in [m.gguf, *([m.mmproj] if m.mmproj else [])]:
            path = self.models_dir / version / name
            if path.is_file() and record_vouches(record.get(name), path, m.sha256.get(name, "")):
                continue
            if path.is_file():
                self.hashed.append(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != m.sha256.get(name):
                problems.append(f"{name} does not match")
        return problems

    def swap(self, version, *, trust_verified=False):
        self.calls.append(("swap", version))
        self.trusts.append(trust_verified)
        for _ in range(2):                 # atlas_swap.swap's check, then ModelStore.activate's
            problems = self.verify(version, trust_verified=trust_verified)
            if problems:
                raise deploy.DeployError("swap", f"{version} is not safe to switch to: " + "; ".join(problems))
        if self.swap_fails:
            raise deploy.DeployError("swap", self.swap_fails)
        self._current, self._previous = version, self._current

    def swap_back(self, version, previous, *, trust_verified=False):
        self.calls.append(("swap_back", version, previous))
        self._current, self._previous = version, previous
        return self.back_problem


class Local:
    """The box's own HTTP: llama-server on :8080 and the app on :8000, behind one MockTransport."""

    def __init__(self) -> None:
        self.chat: list[dict] = []
        self.deployed: list[dict] = []
        self.answer = "ok"
        self.app_status = 200
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            self.chat.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": self.answer}}]})
        if request.url.path == "/api/learning/deployed":
            self.deployed.append(json.loads(request.content))
            return httpx.Response(self.app_status, json={"ok": True})
        return httpx.Response(404)


class World:
    def __init__(self, tmp: Path) -> None:
        self.packages = tmp / "pod" / "data" / "packages"
        shutil.copytree(FIXTURE, self.packages / VERSION)
        write_package(self.packages, OLD_ON_POD, {"model.gguf": b"older weights", "mmproj-model.gguf": b"older proj"})
        self.log = io.StringIO()
        self.server = package_server.make_server(str(self.packages), "127.0.0.1", 0, self.log)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.clock = FakeClock("2026-10-11T06:00:00Z")
        self.hub = FakeHub(self.clock)
        self.account = self.hub.add_account("D", user=USER, token=TOKEN)
        self.hub.start_server("D")
        self.proxy = FakePodProxy(self.hub, self.account, self.server.server_address[1])
        self.env = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_D": USER, "AUP_HUB_TOKEN_D": TOKEN,
                    "AUP_HUB_URL_D": "https://hub.test"}
        self.models = tmp / "box" / "data" / "models"
        self.events = tmp / "box" / "data" / "events"
        self.box = FakeBox(self.models)
        self.local = Local()

    def run(self, *argv, box=None):
        return deploy.run([*argv, "--models-dir", str(self.models), "--events-dir", str(self.events),
                           "--part-mib", PART_MIB, "--quiet"],
                          environ=self.env, env_files=[], transport=self.proxy.transport,
                          local_transport=self.local.transport, box=box or self.box, sleep=self.clock.sleep)

    def logged(self):
        return [parse_event(line) for path in sorted(self.events.glob("*.jsonl"))
                for line in path.read_text(encoding="utf-8").splitlines() if line]

    def close(self):
        self.proxy.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    monkeypatch.delenv("ATLAS_TOKEN", raising=False)
    built = World(tmp_path)
    try:
        yield built
    finally:
        built.close()


def test_the_fixture_package_is_what_l5_writes_and_its_checksums_hold():
    m = ModelManifest.model_validate_json((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert (m.version, m.base, m.task) == (VERSION, "Qwen/Qwen3.5-4B", "desk-objects")
    for name in (m.gguf, m.mmproj):
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == m.sha256[name]


def test_a_deploy_pulls_the_newest_package_swaps_checks_and_tells_the_app(world):
    answer = world.run()

    assert (answer["ok"], answer["outcome"], answer["version"]) == (True, "deployed", VERSION)
    for name in ("manifest.json", f"{VERSION}-Q4_K_M.gguf", "mmproj-F16.gguf"):
        assert (world.models / VERSION / name).read_bytes() == (FIXTURE / name).read_bytes()
    assert world.box.calls == [("swap", VERSION)]
    assert (world.box.current(), world.box.previous()) == (VERSION, "atlas-base")
    assert answer["previous_version"] == "atlas-base" and answer["checksums_ok"] is True
    # every request went through the account's proxy route with the hub token, the model in Range parts
    assert all(item.headers["authorization"] == f"token {TOKEN}" for item in world.proxy.seen)
    assert [item.range for item in world.proxy.requests(f"/{VERSION}/{VERSION}-Q4_K_M.gguf")] == [
        "bytes=0-16383", "bytes=16384-32767", "bytes=32768-49151"]
    # the check is one chat request shaped as runtime/llama.py sends it
    assert world.local.chat == [{"messages": [{"role": "user", "content": deploy.CHECK_PROMPT}], "max_tokens": 16,
                                 "temperature": 0.2}]
    assert world.local.deployed == [{"model_version": VERSION, "previous_version": "atlas-base", "base": "Qwen/Qwen3.5-4B",
                                     "task": "desk-objects", "package_mb": 0.0, "pulled_seconds": answer["pulled_seconds"],
                                     "checksums_ok": True}]
    assert answer["published"] == "app" and answer["pull"]["bytes"] == 48_000 and answer["pull"]["outcome"] == "ok"
    assert world.logged() == []          # the app's bus writes the log; deploy does not write a second copy


def test_with_no_app_the_event_goes_straight_into_the_box_event_log(world):
    answer = world.run(VERSION, "--no-app")

    assert answer["ok"] and answer["published"] == "log" and world.local.deployed == []
    (event,) = world.logged()
    assert isinstance(event, ModelDeployed) and event.event_id == answer["event_id"]
    assert (event.model_version, event.previous_version, event.base, event.task, event.checksums_ok) == (
        VERSION, "atlas-base", "Qwen/Qwen3.5-4B", "desk-objects", True)
    assert event.package_mb == round(48_000 / 1e6, 1) and event.pulled_seconds is not None


def test_an_app_that_refuses_the_event_does_not_lose_it(world):
    world.local.app_status = 503
    answer = world.run(VERSION)

    assert answer["ok"] and answer["published"] == "log" and "503" in answer["app_note"]
    assert [e.type for e in world.logged()] == ["learning.model.deployed"]


def test_a_model_that_does_not_answer_the_gateway_is_swapped_back_and_the_failure_published(world):
    world.local.answer = ""
    answer = world.run(VERSION)

    assert (answer["ok"], answer["kind"], answer["rolled_back_to"]) == (False, "check", "atlas-base")
    assert world.box.calls == [("swap", VERSION), ("swap_back", "atlas-base", "atlas-v1")]
    assert (world.box.current(), world.box.previous()) == ("atlas-base", "atlas-v1")
    assert world.local.deployed == []
    (event,) = world.logged()
    assert isinstance(event, RunProgress) and (event.stage, event.task) == ("failed", "desk-objects")
    assert VERSION in event.message and "check" in event.message


def test_a_swap_that_fails_is_published_and_nothing_is_asked_of_the_model(world):
    world.box.swap_fails = "the model server did not become healthy"
    answer = world.run(VERSION)

    assert (answer["ok"], answer["kind"]) == (False, "swap") and world.local.chat == []
    assert [e.stage for e in world.logged()] == ["failed"]


def test_a_package_whose_checksum_is_wrong_is_never_swapped_to(world):
    write_package(world.packages, "atlas-bad", {"model.gguf": b"weights", "mmproj-model.gguf": b"proj"},
                  checksums={"model.gguf": "0" * 64})
    answer = world.run("atlas-bad")

    assert (answer["ok"], answer["kind"]) == (False, "pull") and "checksum_mismatch" in answer["error"]
    assert world.box.calls == [] and not (world.models / "atlas-bad" / "manifest.json").exists()
    assert [e.type for e in world.logged()] == ["learning.run.progress"]


def test_the_dry_run_prints_the_plan_and_touches_nothing(world):
    answer = world.run("--dry-run")

    assert (answer["ok"], answer["outcome"], answer["version"]) == (True, "dry_run", VERSION)
    plan = answer["plan"]
    assert [f["name"] for f in plan["files"]] == [f"{VERSION}-Q4_K_M.gguf", "mmproj-F16.gguf"]
    assert (plan["current"], plan["previous"], plan["account"]) == ("atlas-base", "atlas-v1", "D")
    assert plan["from"].endswith(f"/proxy/8300/{VERSION}") and plan["disk_free_mb"] > 0
    assert not world.models.exists() and not world.events.exists()
    assert world.box.calls == [] and world.local.chat == []


def test_a_version_already_serving_is_left_alone(world):
    world.box._current = VERSION
    answer = world.run(VERSION)

    assert (answer["ok"], answer["outcome"]) == (True, "already_serving")
    assert world.box.calls == [] and world.proxy.requests(".gguf") == []


def test_a_pod_with_no_complete_package_is_an_answer_not_a_crash(world):
    shutil.rmtree(world.packages)
    world.packages.mkdir(parents=True)
    answer = world.run()

    assert (answer["ok"], answer["kind"]) == (False, "no_package") and world.logged() == []


def test_a_pod_with_no_package_server_running_says_so(world):
    world.proxy.listening = False
    answer = world.run()

    assert (answer["ok"], answer["kind"]) == (False, "no_package") and "server_down" in answer["error"]
    assert world.box.calls == [] and world.logged() == []


def test_missing_hub_keys_are_named_and_never_printed(world):
    world.env = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_D": USER}
    answer = world.run()

    assert (answer["ok"], answer["kind"], answer["missing"]) == (False, "no_credentials", ["AUP_HUB_TOKEN_D"])


def test_the_app_publishes_what_deploy_sends():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from contexts.learning.api import learning_router
    from contexts.learning.application.board import LearningBoard

    published = []

    async def publish(event):
        published.append(event)

    app = FastAPI()
    app.include_router(learning_router(LearningBoard(), publish=publish))
    body = {"model_version": VERSION, "previous_version": "atlas-base", "base": "Qwen/Qwen3.5-4B", "task": "desk-objects",
            "package_mb": 3412.6, "pulled_seconds": 181.2, "checksums_ok": True}
    assert TestClient(app).post("/api/learning/deployed", json=body).status_code == 200
    assert published[0].model_dump(include=set(body)) == body


# --- hashing once: the pull's .verified.json --------------------------------------------------------------------

GGUF_NAME, MMPROJ_NAME = f"{VERSION}-Q4_K_M.gguf", "mmproj-F16.gguf"


@pytest.fixture
def pull_hashes(monkeypatch):
    """Every whole-file sha256 the pull runs (a file already on the box); a downloaded file is hashed as it streams."""
    seen: list[str] = []
    real = package_pull.sha256_file

    def counted(path):
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(package_pull, "sha256_file", counted)
    return seen


def _on_the_box_already(world):
    """The live demo's case: a first deploy put every file on the box; the box is back on atlas-base."""
    assert world.run(VERSION, "--no-app")["ok"]
    world.box._current, world.box._previous = "atlas-base", "atlas-v1"
    for seen in (world.box.calls, world.box.trusts, world.box.hashed):
        seen.clear()


def _after_the_pull(monkeypatch, change):
    """Run `change(folder)` right after the deploy's pull returns, before the verify and the swap."""
    real = package_pull.pull_package

    def pull_then_change(client, version, dest, **kwargs):
        answer = real(client, version, dest, **kwargs)
        change(Path(dest) / version)
        return answer

    monkeypatch.setattr(deploy.package_pull, "pull_package", pull_then_change)


def test_the_pull_records_what_it_hashed_beside_the_manifest(world):
    answer = world.run(VERSION, "--no-app")

    assert answer["ok"] and answer["trust_verified"] is True
    folder = world.models / VERSION
    record = json.loads((folder / ".verified.json").read_text(encoding="utf-8"))
    manifest = ModelManifest.model_validate_json((folder / "manifest.json").read_text(encoding="utf-8"))
    assert (record["written_by"], record["version"]) == ("lab/package_pull.py", VERSION)
    assert [(f["name"], f["size"], f["sha256"], f["mtime_ns"]) for f in record["files"]] == [
        (name, (folder / name).stat().st_size, manifest.sha256[name], (folder / name).stat().st_mtime_ns)
        for name in (GGUF_NAME, MMPROJ_NAME)]
    # the files were hashed as they streamed in; the box trusted the record and hashed nothing
    assert world.box.hashed == [] and world.box.trusts == [True]


def test_files_already_on_the_box_are_hashed_once_on_the_live_path(world, pull_hashes):
    _on_the_box_already(world)
    pull_hashes.clear()

    answer = world.run(VERSION, "--no-app")

    assert (answer["ok"], answer["outcome"], answer["pull"]["outcome"]) == (True, "deployed", "already_present")
    assert pull_hashes == [GGUF_NAME, MMPROJ_NAME]          # the one full pass: the pull's check of present files
    assert world.box.hashed == []                            # verify, the swap's check and the switch trust the record
    assert world.box.calls == [("swap", VERSION)] and world.box.trusts == [True]
    assert world.box.current() == VERSION


def test_no_trust_hashes_at_every_step_as_before(world, pull_hashes):
    _on_the_box_already(world)
    pull_hashes.clear()

    answer = world.run(VERSION, "--no-app", "--no-trust")

    assert answer["ok"] and answer["trust_verified"] is False and world.box.trusts == [False]
    assert pull_hashes == [GGUF_NAME, MMPROJ_NAME]
    assert world.box.hashed == [GGUF_NAME, MMPROJ_NAME] * 3    # the verify, the swap's check, the switch


def test_with_no_record_every_step_hashes(world, pull_hashes, monkeypatch):
    _on_the_box_already(world)
    pull_hashes.clear()
    _after_the_pull(monkeypatch, lambda folder: (folder / ".verified.json").unlink())

    answer = world.run(VERSION, "--no-app")

    assert answer["ok"] and answer["outcome"] == "deployed"
    assert pull_hashes == [GGUF_NAME, MMPROJ_NAME]
    assert world.box.hashed == [GGUF_NAME, MMPROJ_NAME] * 3


def test_a_record_someone_else_wrote_is_not_trusted(world, monkeypatch):
    def forge(folder):
        path = folder / ".verified.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "written_by": "someone else"}), encoding="utf-8")

    _after_the_pull(monkeypatch, forge)
    answer = world.run(VERSION, "--no-app")

    assert answer["ok"] and world.box.hashed == [GGUF_NAME, MMPROJ_NAME] * 3


def test_a_file_whose_mtime_moved_since_the_pull_is_hashed_again_and_the_rest_trusted(world, monkeypatch):
    def touch(folder):
        st = (folder / GGUF_NAME).stat()
        os.utime(folder / GGUF_NAME, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    _after_the_pull(monkeypatch, touch)
    answer = world.run(VERSION, "--no-app")

    assert answer["ok"] and answer["outcome"] == "deployed"
    assert world.box.hashed == [GGUF_NAME] * 3               # the bytes still match, so the deploy goes on


def test_a_file_changed_after_the_pull_with_the_same_size_is_caught_by_its_new_mtime(world, monkeypatch):
    def damage(folder):
        path = folder / GGUF_NAME
        st = path.stat()
        data = bytearray(path.read_bytes())
        data[100] ^= 0xFF
        path.write_bytes(bytes(data))
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    _after_the_pull(monkeypatch, damage)
    answer = world.run(VERSION, "--no-app")

    assert (answer["ok"], answer["kind"]) == (False, "pull") and GGUF_NAME in answer["error"]
    assert world.box.calls == [] and world.box.current() == "atlas-base"
    assert [e.type for e in world.logged()] == ["learning.run.progress"]


def test_a_file_whose_size_changed_after_the_pull_is_hashed_again_and_refused(world, monkeypatch):
    def grow(folder):
        path = folder / GGUF_NAME
        st = path.stat()
        with path.open("ab") as handle:
            handle.write(b"x")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime as recorded: the size alone gives it away

    _after_the_pull(monkeypatch, grow)
    answer = world.run(VERSION, "--no-app")

    assert (answer["ok"], answer["kind"]) == (False, "pull") and world.box.hashed == [GGUF_NAME]
    assert world.box.calls == []


def test_a_failed_pull_leaves_no_record_behind(world):
    _on_the_box_already(world)
    folder = world.models / VERSION
    assert (folder / ".verified.json").is_file()
    (folder / GGUF_NAME).write_bytes(b"damaged")
    world.proxy.fail(404, contains="Q4_K_M", times=1)       # the damaged file cannot be fetched again

    answer = world.run(VERSION, "--no-app")

    assert (answer["ok"], answer["kind"]) == (False, "pull") and not (folder / ".verified.json").exists()
    assert world.box.calls == []


# --- the real swap: runtime/modelstore.py and jetson/atlas_swap.py, with the restart and the wait faked ------------

@pytest.fixture
def symlinks(tmp_path):
    try:
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    (tmp_path / "probe").unlink()


@pytest.fixture
def store_hashes(monkeypatch):
    """Every sha256 runtime/modelstore.py runs (the real box's verify, the swap's check and the switch's)."""
    from runtime import modelstore
    seen: list[str] = []
    real = modelstore.sha256_file

    def counted(path):
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(modelstore, "sha256_file", counted)
    return seen


def _real_box(world, restarts):
    from tests.test_model_registry import make_version
    make_version(world.models, "atlas-base")
    box = deploy.JetsonBox(world.models, llm_url="http://127.0.0.1:8080", log=lambda line: None,
                           restart=lambda: restarts.append(box.current()), wait=lambda: None)
    box.store.activate("atlas-base")
    return box


def test_the_real_swap_switches_current_and_keeps_the_serving_version_for_rollback(world, symlinks):
    restarts = []
    box = _real_box(world, restarts)
    answer = world.run(VERSION, "--no-app", box=box)

    assert answer["ok"] and (box.current(), box.previous()) == (VERSION, "atlas-base") and restarts == [VERSION]


def test_the_real_box_hashes_once_on_the_live_path_and_three_more_times_with_no_trust(world, symlinks, pull_hashes,
                                                                                       store_hashes):
    restarts = []
    box = _real_box(world, restarts)
    assert world.run(VERSION, "--no-app", box=box)["ok"]
    box.store.activate("atlas-base")
    pull_hashes.clear(), store_hashes.clear()

    assert world.run(VERSION, "--no-app", box=box)["ok"] and box.current() == VERSION
    assert pull_hashes == [GGUF_NAME, MMPROJ_NAME] and store_hashes == []

    box.store.activate("atlas-base")
    pull_hashes.clear(), store_hashes.clear()
    assert world.run(VERSION, "--no-app", "--no-trust", box=box)["ok"] and box.current() == VERSION
    assert pull_hashes == [GGUF_NAME, MMPROJ_NAME] and store_hashes == [GGUF_NAME, MMPROJ_NAME] * 3


def test_the_real_swap_goes_back_when_the_gateway_check_fails(world, symlinks):
    restarts = []
    box = _real_box(world, restarts)
    world.local.answer = "  "
    answer = world.run(VERSION, "--no-app", box=box)

    assert answer["kind"] == "check" and box.current() == "atlas-base" and box.previous() is None
    assert restarts == [VERSION, "atlas-base"]
