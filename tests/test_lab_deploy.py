"""python -m lab deploy (L6): the box pulls the packaged model from pod B through the hub proxy, verifies it, hot-swaps,
publishes learning.model.deployed, and goes back to the serving version when the new one does not answer.

What is real: the pod's file server (lab/package_server.py on 127.0.0.1, serving the fixture package from
fixtures/learning/package/), the Range pull (lab/package_pull.py), the event log (bus/bus.py), and in the last tests the
model store and jetson/atlas_swap.py's swap. What is faked: the hub and its proxy (lab/hub_fake.py,
lab/package_fake.py), the box's llama-server and app (an httpx MockTransport), the swap and restart (FakeBox)."""
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
from lab import deploy, package_server
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
    rule (every manifest file present and matching its sha256), swap and swap_back are recorded."""

    def __init__(self, models_dir: Path, current: str | None = "atlas-base", previous: str | None = "atlas-v1") -> None:
        self.models_dir = models_dir
        self._current, self._previous = current, previous
        self.calls: list[tuple] = []
        self.swap_fails: str | None = None
        self.back_problem: str | None = None

    def current(self):
        return self._current

    def previous(self):
        return self._previous

    def verify(self, version):
        m = ModelManifest.model_validate_json((self.models_dir / version / "manifest.json").read_text())
        problems = []
        for name in [m.gguf, *([m.mmproj] if m.mmproj else [])]:
            path = self.models_dir / version / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != m.sha256.get(name):
                problems.append(f"{name} does not match")
        return problems

    def swap(self, version):
        self.calls.append(("swap", version))
        if self.swap_fails:
            raise deploy.DeployError("swap", self.swap_fails)
        self._current, self._previous = version, self._current

    def swap_back(self, version, previous):
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
    assert plan["from"].endswith(f"/user/{USER}/proxy/8300/{VERSION}")
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


# --- the real swap: runtime/modelstore.py and jetson/atlas_swap.py, with the restart and the wait faked ------------

@pytest.fixture
def symlinks(tmp_path):
    try:
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    (tmp_path / "probe").unlink()


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


def test_the_real_swap_goes_back_when_the_gateway_check_fails(world, symlinks):
    restarts = []
    box = _real_box(world, restarts)
    world.local.answer = "  "
    answer = world.run(VERSION, "--no-app", box=box)

    assert answer["kind"] == "check" and box.current() == "atlas-base" and box.previous() is None
    assert restarts == [VERSION, "atlas-base"]
