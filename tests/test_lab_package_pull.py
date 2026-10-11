"""lab.package_pull: pulling a model package from a pod through the hub proxy, in Range parts, and checking it.

What is real here: the pod's file server (lab.package_server, on 127.0.0.1 in a thread), the box-side client, the ledger
and the controller. What is faked is the hub and its proxy (lab.hub_fake.FakeHub, with lab.package_fake.FakePodProxy in
front of the file server, which refuses a response over `LIMIT` bytes as the live proxy refuses one over 100 MiB), the
websocket terminal of a pod, and time. Nothing launches a pod, calls the live hub or sleeps for real. Each test works
in its own pod folder, box folder and ledger."""
from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import http.server
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from lab import controller as ctl_module
from lab import package_pull, package_server
from lab.hub import Account
from lab.hub_fake import FakeClock, FakeHub
from lab.machine import ledger
from lab.machine import stop as kill_switch
from lab.package_fake import FakePodProxy, write_package

REPO = Path(__file__).resolve().parents[1]
MIB = 1 << 20
TOKEN = "eeee5555ffff6666aaaa7777bbbb8888"
WRONG_TOKEN = "0123456789abcdef0123456789abcdef"
USER = "acct-b"
ENV = {"AUP_HUB_USER_B": USER, "AUP_HUB_TOKEN_B": TOKEN, "LAB_HUB_URL": "https://hub.test"}
VERSION = "atlas-v3-desk"
LIMIT = 20_000            # stands for the proxy's 100 MiB in tests that use small files
PART = 16_384             # a part, under LIMIT as 64 MiB is under 100 MiB
_RANDOM = random.Random(7)
GGUF = _RANDOM.randbytes(50_000)
MMPROJ = _RANDOM.randbytes(7_000)


class StepClock:
    """A clock that moves by `step` each time it is read, so a transfer of N requests takes exactly N * step."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class World:
    """A pod's package folder with the real file server on it, a fake hub and proxy in front, and a box folder."""

    def __init__(self, tmp_path: Path, *, files=None, limit: int = LIMIT, letter: str = "B", user: str = USER) -> None:
        self.tmp = tmp_path
        self.letter = letter
        self.env = {"LAB_POD_ROLES": f"{letter}:{'pod-a' if letter in 'AC' else 'pod-b'}", f"AUP_HUB_USER_{letter}": user, f"AUP_HUB_TOKEN_{letter}": TOKEN, "LAB_HUB_URL": "https://hub.test"}
        self.packages = tmp_path / "pod" / "data" / "packages"
        self.on_pod = self.packages / VERSION
        write_package(self.packages, VERSION, files or {"model.gguf": GGUF, "mmproj-model.gguf": MMPROJ})
        self.server_log = io.StringIO()
        self.server = package_server.make_server(str(self.packages), "127.0.0.1", 0, self.server_log)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.clock = FakeClock("2026-10-10T09:27:00Z")
        self.hub = FakeHub(self.clock)
        self.account = self.hub.add_account(letter, user=user, token=TOKEN)
        self.hub.start_server(letter)
        self.proxy = FakePodProxy(self.hub, self.account, self.server.server_address[1], response_limit=limit)
        self.dest = tmp_path / "box" / "models"
        self.closers = [self.proxy.close]

    def client(self, account: Account | None = None, proxy: FakePodProxy | None = None) -> package_pull.PackageClient:
        proxy = proxy or self.proxy
        client = package_pull.PackageClient(account or self.account, proxy.port, base_url="https://hub.test",
                                       transport=proxy.transport, sleep=self.clock.sleep)
        self.closers.append(client.close)
        return client

    def pull(self, client=None, **kwargs):
        kwargs.setdefault("part_bytes", PART)
        return package_pull.pull_package(client or self.client(), VERSION, self.dest, sleep=self.clock.sleep, **kwargs)

    def probe(self, client=None, name=None, **kwargs):
        kwargs.setdefault("limit_bytes", LIMIT)
        return package_pull.probe_whole_file(client or self.client(), VERSION, name, sleep=self.clock.sleep, **kwargs)

    def run(self, *argv, environ=None, **kwargs):
        return package_pull.run(list(argv), environ=self.env if environ is None else environ, env_files=[], transport=self.proxy.transport,
                                sleep=self.clock.sleep, **kwargs)

    def ranges(self, name="model.gguf"):
        return [item.range for item in self.proxy.requests(f"/{name}")]

    def edit_manifest(self, **changes):
        path = self.on_pod / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")

    def rows(self, verb):
        return ledger.read(verb=verb)

    def only_row(self, verb):
        rows = self.rows(verb)
        assert len(rows) == 1, rows
        return rows[0]

    def server_lines(self, count: int, timeout: float = 10.0) -> list[str]:
        """The file server's log lines once there are `count` of them (a line is written just after its answer)."""
        deadline = time.monotonic() + timeout
        found: list[str] = []
        while time.monotonic() < deadline:
            found = [line for line in self.server_log.getvalue().splitlines() if line]
            if len(found) >= count:
                return found
            time.sleep(0.01)
        raise AssertionError(f"expected {count} server log lines, found {found}")

    def close(self) -> None:
        for closer in self.closers:
            closer()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    for name in ("LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    built = World(tmp_path)
    try:
        yield built
    finally:
        built.close()


# --- pulling a package -------------------------------------------------------------------------------------------

def test_a_pull_fetches_each_file_in_range_parts_and_checks_its_sha256(world):
    answer = world.pull()

    folder = world.dest / VERSION
    assert (answer["ok"], answer["outcome"]) == (True, "ok")
    assert (folder / "model.gguf").read_bytes() == GGUF
    assert (folder / "mmproj-model.gguf").read_bytes() == MMPROJ
    assert (folder / "manifest.json").read_bytes() == (world.on_pod / "manifest.json").read_bytes()
    assert sorted(item.name for item in folder.iterdir()) == [
        ".verified.json", "manifest.json", "mmproj-model.gguf", "model.gguf"]
    assert [(item.path, item.range, item.status) for item in world.proxy.seen] == [
        (f"/{VERSION}/manifest.json", None, 200),
        (f"/{VERSION}/model.gguf", "bytes=0-16383", 206),
        (f"/{VERSION}/model.gguf", "bytes=16384-32767", 206),
        (f"/{VERSION}/model.gguf", "bytes=32768-49151", 206),
        (f"/{VERSION}/model.gguf", "bytes=49152-65535", 206),
        (f"/{VERSION}/mmproj-model.gguf", "bytes=0-16383", 206)]
    assert max(item.body_bytes for item in world.proxy.seen) <= PART < LIMIT
    assert (answer["bytes"], answer["package_bytes"], answer["parts"], answer["retries"]) == (57_000, 57_000, 5, 0)
    assert [(f["name"], f["bytes"], f["transferred"], f["parts"], f["resumed_from"], f["outcome"], f["sha256"])
            for f in answer["files"]] == [("model.gguf", 50_000, 50_000, 4, 0, "pulled", "ok"),
                                          ("mmproj-model.gguf", 7_000, 7_000, 1, 0, "pulled", "ok")]


def test_every_request_carries_the_token_and_asks_for_a_buffered_identity_answer(world):
    world.pull()

    for item in world.proxy.seen:
        assert item.headers["authorization"] == f"token {TOKEN}"
        assert item.headers["user-agent"].startswith("Mozilla/5.0")
        assert item.headers["accept-encoding"] == "identity"
        assert item.headers["accept"] != "text/event-stream"
    assert {item.headers["accept"] for item in world.proxy.seen if item.range} == {"*/*"}
    assert all(call.account == "B" for call in world.hub.calls)


def test_the_server_that_was_sent_the_token_never_logs_it(world):
    world.pull()
    world.probe()

    sent_to_the_server = [item.headers["authorization"] for item in world.proxy.seen]
    log = "\n".join(world.server_lines(len(world.proxy.seen)))
    assert len(sent_to_the_server) == 9 and set(sent_to_the_server) == {f"token {TOKEN}"}
    assert TOKEN not in log and "authorization" not in log.lower()


def test_the_pull_row_holds_the_speed_bytes_seconds_part_size_node_and_account(world):
    answer = world.pull(clock=StepClock(0.25))

    assert (answer["seconds"], answer["wall_seconds"], answer["mb_s"]) == (1.25, 2.75, 0.046)
    row = world.only_row("link pull")
    assert (row["outcome"], row["node"], row["actor"]) == ("ok", "pod-b", "tester")
    assert row["metrics"] == {"link.pull_mb_s": 0.046}
    assert row["units"] == {"link.pull_mb_s": "MB/s"}
    assert row["refs"]["account"] == "B"
    assert row["refs"]["run_id"] == answer["run_id"]
    assert row["conditions"] == {"path": "proxy-range", "bytes": 57_000, "part_bytes": PART, "hub": "hub.test",
                                 "client": f"{platform.system()}-{platform.machine()}".lower()}
    assert (row["bytes"], row["seconds"], row["wall_seconds"], row["part_bytes"], row["parts"], row["retries"]) == (
        57_000, 1.25, 2.75, PART, 5, 0)
    assert (row["version"], row["files"]) == (VERSION, ["model.gguf", "mmproj-model.gguf"])
    assert "kind" not in row


def test_the_row_of_a_pull_that_sent_nothing_holds_no_speed(world):
    world.pull()
    seen_before = len(world.proxy.seen)

    again = world.pull()

    assert (again["ok"], again["outcome"], again["bytes"], again["mb_s"]) == (True, "already_present", 0, None)
    assert [item.range for item in world.proxy.seen[seen_before:]] == [None]          # the manifest, and not one part
    second = world.rows("link pull")[1]
    assert (second["outcome"], second["metrics"]) == ("already_present", {})
    assert [f["outcome"] for f in again["files"]] == ["present", "present"]


def test_each_pull_records_the_files_it_hashed_with_their_size_and_mtime(world):
    world.pull()
    folder = world.dest / VERSION
    st = (folder / "model.gguf").stat()
    os.utime(folder / "model.gguf", ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))

    again = world.pull()                       # present files: hashed again, and recorded as they are now

    assert again["outcome"] == "already_present"
    record = json.loads((folder / ".verified.json").read_text(encoding="utf-8"))
    assert (record["written_by"], record["version"]) == ("lab/package_pull.py", VERSION)
    assert [(f["name"], f["size"], f["sha256"], f["mtime_ns"]) for f in record["files"]] == [
        ("model.gguf", 50_000, hashlib.sha256(GGUF).hexdigest(), st.st_mtime_ns + 7_000_000_000),
        ("mmproj-model.gguf", 7_000, hashlib.sha256(MMPROJ).hexdigest(), (folder / "mmproj-model.gguf").stat().st_mtime_ns)]


def test_force_pulls_everything_again(world):
    world.pull()

    again = world.pull(force=True)

    assert (again["outcome"], again["bytes"], again["parts"]) == ("ok", 57_000, 5)
    assert (world.dest / VERSION / "model.gguf").read_bytes() == GGUF


def test_a_damaged_copy_already_on_disk_is_replaced_and_a_good_one_is_left_alone(world):
    folder = world.dest / VERSION
    folder.mkdir(parents=True)
    (folder / "model.gguf").write_bytes(bytes(50_000))
    (folder / "mmproj-model.gguf").write_bytes(MMPROJ)

    answer = world.pull()

    assert [(f["name"], f["outcome"], f["transferred"]) for f in answer["files"]] == [
        ("model.gguf", "pulled", 50_000), ("mmproj-model.gguf", "present", 0)]
    assert world.ranges("mmproj-model.gguf") == []
    assert (folder / "model.gguf").read_bytes() == GGUF


def test_the_manifest_is_written_last_so_a_half_pulled_package_is_not_taken_for_one(world):
    world.proxy.fail(404, contains="mmproj", times=1)

    failed = world.pull()

    folder = world.dest / VERSION
    assert (failed["ok"], failed["kind"], failed["status"]) == (False, "missing", 404)
    assert sorted(item.name for item in folder.iterdir()) == ["model.gguf"]     # and no .verified.json
    resumed = world.pull()
    assert (resumed["ok"], [f["outcome"] for f in resumed["files"]]) == (True, ["present", "pulled"])
    assert (folder / "manifest.json").is_file()


# --- stopping and starting again ---------------------------------------------------------------------------------

def test_a_pull_that_stopped_continues_from_the_bytes_already_on_disk(world):
    folder = world.dest / VERSION
    folder.mkdir(parents=True)
    (folder / "model.gguf.part").write_bytes(GGUF[:20_000])

    answer = world.pull()

    assert world.ranges() == ["bytes=20000-36383", "bytes=36384-52767"]
    assert (folder / "model.gguf").read_bytes() == GGUF
    assert not (folder / "model.gguf.part").exists()
    first = answer["files"][0]
    assert (first["resumed_from"], first["transferred"], first["parts"]) == (20_000, 30_000, 2)
    assert answer["bytes"] == 37_000


def test_a_resumed_copy_that_fails_its_checksum_is_fetched_again_from_the_start(world):
    folder = world.dest / VERSION
    folder.mkdir(parents=True)
    (folder / "model.gguf.part").write_bytes(bytes(20_000))

    answer = world.pull()

    assert world.ranges() == ["bytes=20000-36383", "bytes=36384-52767", "bytes=0-16383", "bytes=16384-32767",
                              "bytes=32768-49151", "bytes=49152-65535"]
    assert (folder / "model.gguf").read_bytes() == GGUF
    assert answer["ok"] is True
    assert answer["files"][0]["resumed_from"] == 0
    assert answer["bytes"] == 30_000 + 50_000 + 7_000


def test_a_file_that_does_not_match_the_manifest_after_a_clean_download_is_an_error_and_leaves_nothing(world):
    world.edit_manifest(sha256={"model.gguf": hashlib.sha256(b"another file").hexdigest(),
                                "mmproj-model.gguf": hashlib.sha256(MMPROJ).hexdigest()})

    answer = world.pull()

    folder = world.dest / VERSION
    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "checksum_mismatch")
    assert "model.gguf does not match the sha256 in the manifest" in answer["error"]
    assert list(folder.iterdir()) == []
    assert world.ranges("mmproj-model.gguf") == []                    # nothing after the first bad file
    row = world.only_row("link pull")
    assert (row["outcome"], row["kind"], row["metrics"]) == ("error", "checksum_mismatch", {})


def test_a_file_that_grows_while_it_is_pulled_is_reported(world):
    grown = []

    def grow(line):
        if not grown:
            grown.append(line)
            with (world.on_pod / "model.gguf").open("ab") as handle:
                handle.write(b"x" * 10)

    answer = world.pull(progress=grow)

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "file_changed", 206)
    assert "was 50000 bytes, now 50010" in answer["error"]
    assert (world.dest / VERSION / "model.gguf.part").stat().st_size == PART       # kept: a later pull continues from it


# --- the proxy answering badly -----------------------------------------------------------------------------------

def test_a_gateway_error_on_the_manifest_is_tried_again_after_growing_pauses(world):
    world.hub.queue("GET", f"/user/{USER}/proxy/8300/", 502, times=2)

    answer = world.pull()

    assert answer["ok"] is True
    assert world.clock.elapsed == 1.0 + 2.0


@pytest.mark.parametrize("status", [429, 502, 503, 504, 520, 521, 522, 523, 524, 599])
def test_a_part_the_proxy_fails_with_a_gateway_status_is_asked_for_again(world, status):
    world.proxy.fail(status, contains="mmproj", times=1)

    answer = world.pull()

    assert answer["ok"] is True
    assert (answer["retries"], world.clock.elapsed) == (1, 1.0)
    assert world.ranges("mmproj-model.gguf") == ["bytes=0-16383", "bytes=0-16383"]
    assert (world.dest / VERSION / "mmproj-model.gguf").read_bytes() == MMPROJ


def test_a_part_that_keeps_failing_gives_up_after_the_retries_and_says_so(world):
    world.proxy.fail(502, contains="/model.gguf", times=99)

    answer = world.pull()

    assert (answer["ok"], answer["kind"], answer["status"], answer["retries"]) == (False, "proxy_error", 502, 4)
    assert "gave up after 4 tries" in answer["error"]
    assert world.clock.elapsed == 1.0 + 2.0 + 4.0
    assert len(world.ranges()) == 4
    assert world.only_row("link pull")["outcome"] == "error"


def test_a_connection_dropped_in_the_middle_of_a_part_keeps_the_bytes_that_arrived(world):
    world.proxy.cut_body(5_000, contains="/model.gguf", times=1)

    answer = world.pull()

    assert world.ranges() == ["bytes=0-16383", "bytes=5000-21383", "bytes=21384-37767", "bytes=37768-54151"]
    assert [item.body_bytes for item in world.proxy.requests("/model.gguf")] == [5_000, 16_384, 16_384, 12_232]
    assert (world.dest / VERSION / "model.gguf").read_bytes() == GGUF
    assert (answer["ok"], answer["bytes"], answer["retries"], world.clock.elapsed) == (True, 57_000, 1, 0.0)


def test_a_connection_that_cannot_be_made_gives_up_and_names_neither_token_nor_account(world):
    world.hub.queue_dropped("GET", f"/user/{USER}/proxy/8300/", times=99)

    answer = world.pull()

    assert (answer["ok"], answer["kind"]) == (False, "unreachable")
    assert "<account B>" in answer["error"]
    everything = json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")
    assert USER not in everything and TOKEN not in everything


def test_a_wrong_token_is_a_login_redirect_that_is_reported_and_never_followed(world):
    stranger = Account("B", USER, WRONG_TOKEN)

    answer = world.pull(client=world.client(stranger))

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "auth", 302)
    assert world.proxy.seen == []
    assert [call.path for call in world.hub.calls] == [f"/user/{USER}/proxy/8300/health"]
    assert world.clock.elapsed == 0.0
    assert not world.dest.exists()


def test_a_stopped_pod_is_reported_at_once_without_retrying(world):
    world.hub.drop_server("B")

    answer = world.pull()

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "pod_stopped", 424)
    assert len(world.hub.calls) == 1 and world.clock.elapsed == 0.0


def test_a_port_with_nothing_on_it_is_reported_at_once_and_says_how_to_start_the_server(world):
    world.proxy.listening = False

    answer = world.pull()

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "server_down", 500)
    assert "serve-start" in answer["error"]
    assert world.clock.elapsed == 0.0


def test_a_version_the_pod_does_not_have_is_missing(world):
    answer = package_pull.pull_package(world.client(), "no-such-version", world.dest, part_bytes=PART, sleep=world.clock.sleep)

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "missing", 404)


@contextlib.contextmanager
def plain_http_server(folder: Path):
    """python's own file server (what `python3 -m http.server` is), which ignores Range."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_a_server_that_ignores_range_is_refused_before_it_sends_the_file(world):
    with plain_http_server(world.packages) as backend_port:
        plain = FakePodProxy(world.hub, world.account, backend_port, port=8301, response_limit=10 * MIB)
        try:
            answer = world.pull(client=world.client(proxy=plain))
        finally:
            plain.close()

    assert (answer["ok"], answer["kind"], answer["status"], answer["bytes"]) == (False, "range_ignored", 200, 0)
    assert "package_server.py" in answer["error"]
    assert [item.range for item in plain.seen] == [None, "bytes=0-16383"]


# --- what the manifest can say -----------------------------------------------------------------------------------

@pytest.mark.parametrize("changes, wanted", [
    ({"sha256": {"model.gguf": hashlib.sha256(GGUF).hexdigest()}}, "no usable sha256 for mmproj-model.gguf"),
    ({"sha256": {"model.gguf": "abc", "mmproj-model.gguf": hashlib.sha256(MMPROJ).hexdigest()}}, "no usable sha256 for model.gguf"),
    ({"gguf": "../evil.gguf", "sha256": {"../evil.gguf": "0" * 64, "mmproj-model.gguf": "0" * 64}}, "does not serve: '../evil.gguf'"),
    ({"gguf": "model.gguf.part", "sha256": {"model.gguf.part": "0" * 64, "mmproj-model.gguf": "0" * 64}}, "does not serve: 'model.gguf.part'"),
    ({"gguf": None}, "is not a ModelManifest"),
], ids=["no-checksum-for-the-projector", "checksum-not-hex", "name-that-leaves-the-folder", "name-of-an-unfinished-file",
        "no-gguf"])
def test_a_manifest_that_cannot_be_trusted_is_refused_before_any_file_is_pulled(world, changes, wanted):
    world.edit_manifest(**changes)

    answer = world.pull()

    assert (answer["ok"], answer["kind"]) == (False, "bad_manifest")
    assert wanted in answer["error"]
    assert not any(item.range for item in world.proxy.seen)
    assert not (world.tmp / "box" / "evil.gguf").exists() and not (world.dest / "evil.gguf").exists()


def test_a_manifest_that_is_not_json_is_refused(world):
    (world.on_pod / "manifest.json").write_text("<html>not json</html>", encoding="utf-8")

    answer = world.pull()

    assert (answer["ok"], answer["kind"]) == (False, "bad_manifest")


def test_files_the_manifest_lists_beyond_the_model_and_projector_are_pulled_and_checked_too(world):
    extra = _RANDOM.randbytes(3_000)
    (world.on_pod / "adapter.bin").write_bytes(extra)
    world.edit_manifest(sha256={"model.gguf": hashlib.sha256(GGUF).hexdigest(),
                                "mmproj-model.gguf": hashlib.sha256(MMPROJ).hexdigest(),
                                "adapter.bin": hashlib.sha256(extra).hexdigest()})

    answer = world.pull()

    assert [f["name"] for f in answer["files"]] == ["model.gguf", "mmproj-model.gguf", "adapter.bin"]
    assert (world.dest / VERSION / "adapter.bin").read_bytes() == extra


def test_an_empty_file_in_a_package_is_pulled(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    built = World(tmp_path, files={"model.gguf": GGUF, "mmproj-model.gguf": b""})
    try:
        answer = built.pull()
    finally:
        built.close()

    assert answer["ok"] is True
    assert (built.dest / VERSION / "mmproj-model.gguf").read_bytes() == b""


# --- the box's own side ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("part", [100 * MIB, 100 * MIB - 1, 96 * MIB + 1, 0, 1023, 1.5, True, "64"])
def test_a_part_the_proxy_would_refuse_or_that_is_not_a_size_is_refused_before_any_request(world, part):
    with pytest.raises(ValueError):
        world.pull(part_bytes=part)

    assert world.hub.calls == []


def test_the_largest_part_is_under_the_proxys_limit_and_the_default_is_64_mib(world):
    assert package_pull.check_part_bytes(96 * MIB) == 96 * MIB < 100 * MIB
    assert package_pull.DEFAULT_PART_BYTES == 64 * MIB


def test_a_disk_without_room_for_the_file_stops_the_pull_before_it_writes_a_byte(world, monkeypatch):
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(package_pull.shutil, "disk_usage", lambda path: usage(10**9, 10**9 - 1000, 1000))

    answer = world.pull()

    assert (answer["ok"], answer["kind"]) == (False, "no_space")
    assert "needs 50000 more bytes and the disk has 1000 free" in answer["error"]
    assert list((world.dest / VERSION).iterdir()) == []


def test_a_destination_that_cannot_be_written_is_an_answer_not_a_crash(world):
    (world.tmp / "box").mkdir()
    (world.tmp / "box" / "models").write_text("a file where the folder should be", encoding="utf-8")

    answer = world.pull()

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "disk")


def test_a_ledger_that_cannot_be_written_does_not_undo_the_pull(world, monkeypatch):
    not_a_folder = world.tmp / "state-is-a-file"
    not_a_folder.write_text("a file where the state folder should be", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(not_a_folder))

    answer = world.pull()

    assert answer["ok"] is True
    assert "ledger_error" in answer
    assert (world.dest / VERSION / "model.gguf").read_bytes() == GGUF


def test_nothing_is_pulled_while_the_stop_file_exists(world):
    state = world.tmp / "state"
    state.mkdir(parents=True)
    (state / "STOP").write_text("hold everything", encoding="utf-8")

    with pytest.raises(kill_switch.Stopped):
        world.pull()
    with pytest.raises(kill_switch.Stopped):
        world.probe()

    assert world.hub.calls == []


# --- the one whole-file GET --------------------------------------------------------------------------------------

def test_the_whole_file_is_asked_for_once_and_the_status_the_proxy_answers_is_recorded(world):
    answer = world.probe()

    assert (answer["ok"], answer["outcome"], answer["status"], answer["file"], answer["file_bytes"]) == (
        True, "whole_file_refused", 599, "model.gguf", 50_000)
    assert answer["bytes_read"] == 0 and answer["content_length"] is None
    assert "answered 599 to the whole file (50000 bytes on the pod)" in answer["note"]
    assert world.ranges() == [None]                                   # one GET, no Range, and not tried again
    assert world.clock.elapsed == 0.0
    row = world.only_row("link probe")
    assert (row["outcome"], row["node"], row["status"], row["file"], row["file_bytes"], row["bytes_read"]) == (
        "whole_file_refused", "pod-b", 599, "model.gguf", 50_000, 0)
    assert row["conditions"] == {"path": "proxy-whole-file", "hub": "hub.test",
                                 "client": f"{platform.system()}-{platform.machine()}".lower()}
    assert row["refs"]["account"] == "B" and row["metrics"] == {}


def test_a_file_the_proxy_lets_through_whole_is_reported_as_served(world):
    world.proxy.response_limit = 10**9

    answer = world.probe(max_read_bytes=10_000)

    assert (answer["ok"], answer["outcome"], answer["status"]) == (True, "whole_file_served", 200)
    assert answer["bytes_read"] >= 10_000
    assert answer["note"] == ("50000 bytes of one response came through the proxy (the file is 50000 bytes): the "
                              "20000 bytes limit did not hold here")
    assert world.only_row("link probe")["outcome"] == "whole_file_served"


def test_a_file_not_larger_than_the_limit_says_nothing_and_the_probe_says_so(world):
    world.proxy.response_limit = 10**9

    answer = world.probe(name="mmproj-model.gguf")

    assert (answer["ok"], answer["outcome"], answer["kind"], answer["status"], answer["bytes_read"]) == (
        False, "error", "file_too_small", 200, 7_000)
    assert world.only_row("link probe")["outcome"] == "error"


def test_a_whole_file_the_proxy_cuts_off_part_way_is_reported_as_refused(world):
    world.proxy.response_limit = 10**9
    world.proxy.cut_body(5_000, contains="/model.gguf")

    answer = world.probe(max_read_bytes=40_000)

    assert (answer["ok"], answer["outcome"], answer["status"], answer["bytes_read"]) == (True, "whole_file_refused", 200, 5_000)
    assert "cut the whole file off after 5000 of 50000 bytes" in answer["note"]


def test_a_refusal_for_a_file_under_the_limit_is_recorded_but_not_taken_for_the_size_limit(world):
    world.proxy.response_limit = 5_000                                # refuses the 7000-byte projector, far under 100 MiB

    answer = world.probe(name="mmproj-model.gguf", limit_bytes=package_pull.PROXY_LIMIT_BYTES)

    assert (answer["outcome"], answer["status"]) == ("whole_file_refused", 599)
    assert "not over the 100 MiB limit, so the refusal is not that limit" in answer["note"]


@pytest.mark.parametrize("arrange, kind, status", [
    (lambda world: world.hub.drop_server("B"), "pod_stopped", 424),
    (lambda world: setattr(world.proxy, "listening", False), "server_down", 500),
])
def test_a_probe_that_never_reached_the_server_is_an_error_and_not_a_refusal(world, arrange, kind, status):
    arrange(world)

    answer = world.probe(name="model.gguf")

    assert (answer["ok"], answer["outcome"], answer["kind"], answer["status"]) == (False, "error", kind, status)


def test_a_listing_the_proxy_fails_does_not_hold_up_the_one_whole_file_try(world):
    world.proxy.fail(502, times=1)                                   # the first request: the listing of the version

    answer = world.probe(name="model.gguf")

    assert (answer["ok"], answer["outcome"], answer["file_bytes"]) == (True, "whole_file_refused", None)
    assert "(an unknown size on the pod)" in answer["note"]
    assert world.clock.elapsed == 0.0
    assert [(item.path, item.range) for item in world.proxy.seen] == [(f"/{VERSION}", None), (f"/{VERSION}/model.gguf", None)]


def test_a_manifest_that_names_a_file_the_server_does_not_serve_is_refused_by_the_probe_too(world):
    world.edit_manifest(gguf="../evil.gguf")

    answer = world.probe()

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "bad_manifest")
    assert [item.range for item in world.proxy.seen] == [None]       # the manifest, and not a file


def test_a_probe_with_a_wrong_token_is_an_auth_error_and_follows_nothing(world):
    answer = world.probe(client=world.client(Account("B", USER, WRONG_TOKEN)), name="model.gguf")

    assert (answer["ok"], answer["kind"], answer["status"]) == (False, "auth", 302)
    assert [call.path for call in world.hub.calls] == [f"/user/{USER}/proxy/8300/health"] * 2    # the listing and the file


# --- the command line --------------------------------------------------------------------------------------------

def test_the_pull_command_answers_one_json_object_and_leaves_a_row_for_the_call_beside_the_measurement(world):
    answer, code = world.run("pull", "--account", "b", "--version", VERSION, "--dest", str(world.dest),
                             "--part-mib", "0.015625", "--quiet")

    assert code == 0
    assert (answer["ok"], answer["outcome"], answer["account"], answer["node"], answer["bytes"]) == (True, "ok", "B", "pod-b", 57_000)
    assert "refs" not in answer
    json.dumps(answer)
    call = world.only_row("package pull")
    assert (call["outcome"], call["metrics"], call["refs"]["account"], call["refs"]["run_id"]) == (
        "ok", {}, "B", answer["run_id"])
    assert "node" not in call and call["duration_s"] >= 0
    assert world.only_row("link pull")["refs"]["run_id"] == answer["run_id"]


def test_the_hub_is_the_address_the_account_carries(world):
    own = Account("B", USER, TOKEN, "https://sjc.test")
    client = package_pull.PackageClient(own, world.proxy.port, transport=world.proxy.transport, sleep=world.clock.sleep, environ={})
    try:
        answer = world.pull(client=client)
    finally:
        client.close()

    assert (answer["ok"], answer["hub"]) == (True, "sjc.test")
    assert world.only_row("link pull")["conditions"]["hub"] == "sjc.test"


def test_the_command_takes_the_hub_from_the_accounts_own_address_key(world):
    environ = {"AUP_HUB_USER_B": USER, "AUP_HUB_TOKEN_B": TOKEN, "AUP_HUB_URL_B": "https://sjc.test"}

    answer, code = world.run("pull", "--account", "B", "--version", VERSION, "--dest", str(world.dest),
                             "--part-mib", "0.015625", "--quiet", environ=environ)
    lines, _ = world.run("curl", "--account", "B", "--version", VERSION, environ=environ)

    assert (code, answer["hub"]) == (0, "sjc.test")
    assert lines["commands"][0] == 'curl -sS -H "Authorization: token ${AUP_HUB_TOKEN_B}" "https://sjc.test/user/${AUP_HUB_USER_B}/proxy/8300/"'


def test_an_account_beyond_a_and_b_takes_its_role_in_the_answer_and_the_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    built = World(tmp_path, letter="C", user="acct-c")
    try:
        answer, code = built.run("pull", "--account", "C", "--version", VERSION, "--dest", str(built.dest),
                                 "--part-mib", "0.015625", "--quiet")
    finally:
        built.close()

    assert (code, answer["ok"], answer["account"], answer["node"]) == (0, True, "C", "pod-a")
    row = ledger.read(verb="link pull")[0]
    assert (row["node"], row["refs"]["account"]) == ("pod-a", "C")


def test_a_failed_pull_exits_1_with_the_reason_and_both_rows_say_error(world):
    world.hub.drop_server("B")

    answer, code = world.run("pull", "--account", "B", "--version", VERSION, "--dest", str(world.dest),
                             "--part-mib", "0.015625", "--quiet")

    assert code == 1
    assert (answer["outcome"], answer["kind"]) == ("error", "pod_stopped")
    assert world.only_row("package pull")["outcome"] == "error"
    assert world.only_row("link pull")["kind"] == "pod_stopped"


@pytest.mark.parametrize("part", ["100", "96.5", "0", "0.0005", "many"])
def test_a_part_size_the_proxy_would_refuse_is_a_usage_error(world, capsys, part):
    with pytest.raises(SystemExit) as caught:
        world.run("pull", "--account", "B", "--version", VERSION, "--dest", str(world.dest), "--part-mib", part)

    assert caught.value.code == 2
    assert "--part-mib" in capsys.readouterr().err
    assert world.hub.calls == []


def test_missing_keys_are_named_and_exit_2(world):
    answer, code = world.run("pull", "--account", "B", "--version", VERSION, "--dest", str(world.dest), environ={})

    assert code == 2
    assert (answer["outcome"], answer["missing"]) == ("no_credentials", ["AUP_HUB_USER_B", "AUP_HUB_TOKEN_B"])
    assert world.hub.calls == []


def test_the_kill_switch_stops_a_pull_command_and_the_stop_is_recorded(world):
    state = world.tmp / "state"
    state.mkdir(parents=True)
    (state / "STOP").write_text("hold everything", encoding="utf-8")

    answer, code = world.run("pull", "--account", "B", "--version", VERSION, "--dest", str(world.dest), "--quiet")

    assert code == 1
    assert (answer["outcome"], answer["reason"]) == ("stopped", "hold everything")
    assert world.only_row("package pull")["outcome"] == "stopped"
    assert world.hub.calls == []


def test_the_probe_command_records_the_status_and_exits_0(world):
    answer, code = world.run("probe", "--account", "B", "--version", VERSION, "--max-read-mib", "0.01")

    assert code == 0
    assert (answer["outcome"], answer["status"]) == ("whole_file_refused", 599)
    assert world.only_row("package probe")["outcome"] == "whole_file_refused"
    assert world.only_row("link probe")["status"] == 599


def test_the_curl_lines_name_the_address_and_leave_the_account_and_token_as_variables(world):
    answer, code = world.run("curl", "--account", "B", "--version", VERSION, "--file", "model.gguf")

    assert code == 0
    base = "https://hub.test/user/${AUP_HUB_USER_B}/proxy/8300"
    assert answer["commands"] == [
        f'curl -sS -H "Authorization: token ${{AUP_HUB_TOKEN_B}}" "{base}/"',
        f'curl -sS -H "Authorization: token ${{AUP_HUB_TOKEN_B}}" "{base}/{VERSION}/manifest.json"',
        f'curl -sS -H "Authorization: token ${{AUP_HUB_TOKEN_B}}" -D - -o /dev/null -r 0-1048575 "{base}/{VERSION}/model.gguf"',
        f'curl -sS -H "Authorization: token ${{AUP_HUB_TOKEN_B}}" -D - -o /dev/null -r -1048576 "{base}/{VERSION}/model.gguf"',
        f'curl -sS -H "Authorization: token ${{AUP_HUB_TOKEN_B}}" -D - -o /dev/null -m 30 "{base}/{VERSION}/model.gguf"']
    assert TOKEN not in json.dumps(answer) and USER not in json.dumps(answer)
    assert world.hub.calls == [] and ledger.read() == []


def test_as_a_program_it_prints_one_json_line_and_exits_2_without_keys(tmp_path):
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("", encoding="utf-8")
    environ = {name: value for name, value in os.environ.items() if not name.startswith(("AUP_HUB_", "LAB_"))}
    environ.update({"LAB_ENV_FILES": str(empty_env_file), "LAB_STATE_DIR": str(tmp_path / "state")})

    done = subprocess.run([sys.executable, "-m", "lab.package_pull", "pull", "--account", "B", "--version", VERSION,
                           "--dest", str(tmp_path / "dest")], cwd=REPO, env=environ, capture_output=True, text=True, timeout=120)

    assert done.returncode == 2, done.stderr
    lines = [line for line in done.stdout.splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["outcome"] == "no_credentials"


# --- the real limit ----------------------------------------------------------------------------------------------

def test_a_101_mib_gguf_is_refused_whole_at_the_real_limit_and_pulled_in_64_mib_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    big = b"".join(os.urandom(MIB) for _ in range(101))
    built = World(tmp_path, files={"model.gguf": big, "mmproj-model.gguf": MMPROJ}, limit=package_pull.PROXY_LIMIT_BYTES)
    try:
        probe = package_pull.probe_whole_file(built.client(), VERSION, sleep=built.clock.sleep)
        answer = package_pull.pull_package(built.client(), VERSION, built.dest, sleep=built.clock.sleep)
        ranges = built.ranges()
    finally:
        built.close()

    assert (probe["ok"], probe["outcome"], probe["status"], probe["file_bytes"]) == (True, "whole_file_refused", 599, 101 * MIB)
    assert "as predicted for a response over 100 MiB" in probe["note"]
    assert ranges == [None, f"bytes=0-{64 * MIB - 1}", f"bytes={64 * MIB}-{128 * MIB - 1}"]
    assert answer["ok"] is True and answer["part_bytes"] == 64 * MIB
    assert (built.dest / VERSION / "model.gguf").read_bytes() == big
    assert answer["mb_s"] > 0


# --- starting and stopping the server on the pod -----------------------------------------------------------------

@contextlib.contextmanager
def fake_terminal(world, *, exit_code=0, output="package server up on port 8300\n"):
    """A Jupyter terminal on localhost: the pod that runs a command is this handler writing the command's output file
    into the fake hub's file store, as the shell on a real pod would."""
    from websockets.sync.server import serve

    typed = []

    def handler(connection):
        connection.send(json.dumps(["setup", {}]))
        for raw in connection:
            kind, payload = json.loads(raw)
            if kind == "stdin":
                typed.append(payload)
                match = re.search(r"> \./(\S+\.out)", payload)
                if match:
                    world.hub.files[("B", match.group(1))] = output + f"### END rc={exit_code}\n"
                connection.send(json.dumps(["stdout", payload]))

    with serve(handler, "127.0.0.1", 0) as server, mock.patch.object(ctl_module, "TERMINAL_SETTLE_S", 60.0):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1], typed
        finally:
            server.shutdown()
            thread.join(timeout=5)


def uploaded_script(world):
    names = [name for (letter, name) in world.hub.files if letter == "B" and name.endswith(".sh")]
    assert len(names) == 1, names
    return world.hub.files[("B", names[0])]


def test_serve_start_runs_a_script_that_carries_the_server_and_then_asks_its_listing(world):
    with fake_terminal(world) as (port, typed):
        answer, code = world.run("serve-start", "--account", "B", environ={**ENV, "LAB_HUB_URL": f"http://127.0.0.1:{port}"},
                                 clock=world.clock.time)

    assert code == 0, answer
    assert (answer["ok"], answer["outcome"], answer["versions"], answer["root"]) == (True, "ok", [VERSION], "$HOME/data/packages")
    assert (answer["on_pod"]["outcome"], answer["on_pod"]["exit_code"]) == ("ok", 0)
    script = uploaded_script(world)
    head, heredoc, rest = script.partition("<<'PACKAGE_SERVER_SOURCE_END'\n")
    source, terminator, tail = rest.partition("\nPACKAGE_SERVER_SOURCE_END\n")
    assert heredoc and terminator
    assert source == Path(package_server.__file__).read_text(encoding="utf-8").rstrip("\n")
    assert "PORT=8300\n" in head and 'ROOT="$HOME/data/packages"\n' in head
    assert "@" not in head + tail
    assert re.search(r'nohup \$LAUNCH python3 "\$DIR/package_server.py" --root "\$ROOT" --port "\$PORT" --log "\$LOG"', tail)
    assert re.fullmatch(r"bash \./lab_run_\S+\.sh > \./lab_run_\S+\.out 2>&1; echo \"### END rc=\$\?\" >> \./lab_run_\S+\.out\n",
                        typed[0])
    assert world.only_row("package serve-start")["outcome"] == "ok"
    assert world.only_row("pod run-script")["node"] == "pod-b"


def test_serve_start_takes_the_port_and_root_it_is_given(world):
    with fake_terminal(world) as (port, _):
        world.run("serve-start", "--account", "B", "--port", "8411", "--root", "~/models/packages",
                  environ={**ENV, "LAB_HUB_URL": f"http://127.0.0.1:{port}"}, clock=world.clock.time)

    head = uploaded_script(world).partition("<<'PACKAGE_SERVER_SOURCE_END'")[0]
    assert "PORT=8411\n" in head and 'ROOT="~/models/packages"\n' in head


def test_serve_start_says_so_when_the_script_fails_on_the_pod(world):
    with fake_terminal(world, exit_code=1, output="package server did not answer on port 8300\n") as (port, _):
        answer, code = world.run("serve-start", "--account", "B", environ={**ENV, "LAB_HUB_URL": f"http://127.0.0.1:{port}"},
                                 clock=world.clock.time)

    assert code == 1
    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "fail")
    assert "did not answer on port 8300" in answer["on_pod"]["output_tail"]
    assert world.only_row("package serve-start")["outcome"] == "error"


def test_serve_start_says_when_the_proxy_has_no_route_to_the_port(world):
    world.hub.queue("GET", f"/user/{USER}/proxy/8300/", 404)
    with fake_terminal(world) as (port, _):
        answer, code = world.run("serve-start", "--account", "B", environ={**ENV, "LAB_HUB_URL": f"http://127.0.0.1:{port}"},
                                 clock=world.clock.time)

    assert code == 1
    assert (answer["outcome"], answer["kind"]) == ("error", "missing")
    assert "jupyter-server-proxy is not loaded on this pod" in answer["hint"]


def test_serve_start_on_a_stopped_pod_uploads_nothing(world):
    world.hub.drop_server("B")
    answer, code = world.run("serve-start", "--account", "B", clock=world.clock.time)

    assert code == 1
    assert (answer["ok"], answer["kind"]) == (False, "not_running")
    assert world.hub.files == {}


def test_serve_stop_runs_a_script_that_stops_the_server_and_nothing_else(world):
    with fake_terminal(world, output="package server stopped\n") as (port, _):
        answer, code = world.run("serve-stop", "--account", "B", environ={**ENV, "LAB_HUB_URL": f"http://127.0.0.1:{port}"},
                                 clock=world.clock.time)

    assert code == 0
    assert (answer["ok"], answer["outcome"]) == (True, "ok")
    assert uploaded_script(world) == package_pull.STOP_SCRIPT
    assert "pkill -u \"$(id -u)\" -f 'package_server.py --root'" in package_pull.STOP_SCRIPT
    assert world.only_row("package serve-stop")["outcome"] == "ok"


def test_serve_commands_are_refused_while_the_stop_file_exists(world):
    state = world.tmp / "state"
    state.mkdir(parents=True)
    (state / "STOP").write_text("hold everything", encoding="utf-8")

    started, _ = world.run("serve-start", "--account", "B")
    stopped, _ = world.run("serve-stop", "--account", "B")

    assert (started["outcome"], stopped["outcome"]) == ("stopped", "stopped")
    assert world.hub.calls == [] and world.hub.files == {}


@pytest.mark.parametrize("root", ['"; rm -rf ~; echo "', "$(id)", "`id`", "../packages", "packages", "", "~/a b", "$HOMEx/y;z"])
def test_a_server_root_that_could_run_a_command_is_refused(root):
    with pytest.raises(ValueError):
        package_pull.start_script(8300, root)


def test_the_start_script_keeps_the_source_it_carries_intact(world):
    text = package_pull.start_script(8300)

    assert text.startswith("#!/bin/bash\n")
    assert text.count("<<'PACKAGE_SERVER_SOURCE_END'") == 1 and text.count("\nPACKAGE_SERVER_SOURCE_END\n") == 1
    compile(text.split("<<'PACKAGE_SERVER_SOURCE_END'\n")[1].split("\nPACKAGE_SERVER_SOURCE_END\n")[0], "package_server.py", "exec")
