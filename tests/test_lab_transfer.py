"""lab.transfer: pushing a dataset up and pulling a model package down through the hub.

Every test drives the real PodLink and PodFiles and the real push and pull functions, with the real ModelStore and the
real contract models checking what lands. The only double is the network between the box and the pod,
lab.proxy_fake.FakePodProxy, which keeps a Jupyter root in memory and answers the contents API and the /files/ route the
way Jupyter's does (research/plumbing/transfers.md), so nothing here opens a socket, sleeps for real or calls a pod."""
from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from contracts.learning import DatasetManifest, ModelManifest, TaskSpec
from lab import transfer
from lab.hub import Hub
from lab.hub_fake import FakeClock
from lab.proxy import BadToken, NotFound, PodLink, PodUnresponsive, ProxyTimeout, Unreachable
from lab.proxy_fake import FakePodProxy
from lab.transfer import (ChecksumMismatch, NotApproved, PodFiles, RangeUnsupported, TransferError, UnsafePath,
                          clean_name, clean_remote, pull_file, pull_package, push_dataset, push_file)
from runtime.modelstore import ModelStore

TOKEN = "0123456789abcdef0123456789abcdef"
USER = "roscon02"
URL = "https://hub.test"
ENV = {"AUP_HUB_USER_D": USER, "AUP_HUB_TOKEN_D": TOKEN, "LAB_HUB_URL": URL}
CONTENTS = f"/user/{USER}/api/contents/"
HTML = {"content-type": "text/html"}
GGUF, MMPROJ = "model-Q4_K_M.gguf", "mmproj-F16.gguf"
WHEN = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)


def blob(size: int, seed: int = 0) -> bytes:
    """Bytes that are not all alike, so a part in the wrong place shows."""
    return bytes((i * 7 + seed + i // 251) % 256 for i in range(size))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class World:
    """One account's pod on a fake proxy and a PodFiles wired to it."""

    def __init__(self, **fake_options):
        self.clock = FakeClock()
        self.fake = FakePodProxy(self.clock, **fake_options)
        self.account = self.fake.add_account("D", user=USER, token=TOKEN)
        hub = Hub(self.account, base_url=self.fake.url, transport=self.fake.transport, sleep=self.clock.sleep, environ={})
        self.link = PodLink(hub, transport=self.fake.transport, sleep=self.clock.sleep, clock=self.clock.time)
        self.files = PodFiles(self.link)

    def put_calls(self, contains=""):
        return self.fake.calls_matching("PUT", contains)

    def remote(self, path):
        return self.fake.file("D", path)


def make_dataset(root: Path, dataset_id="ds-0007", *, action_id="approval-123", images=2) -> Path:
    folder = root / dataset_id
    (folder / "images").mkdir(parents=True)
    manifest = DatasetManifest(dataset_id=dataset_id, spec=TaskSpec(task="desk-objects", instruction="What is on the desk?"),
                               created=WHEN, examples=images, action_id=action_id)
    (folder / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (folder / "examples.jsonl").write_text("\n".join(json.dumps({"id": f"e{i}"}) for i in range(images)) + "\n",
                                           encoding="utf-8")
    for i in range(images):
        (folder / "images" / f"{i:04d}.jpg").write_bytes(blob(300 + i, i))
    (folder / "thumbs").mkdir()                       # a name that sorts after manifest.json
    (folder / "thumbs" / "0000.jpg").write_bytes(blob(120, 7))
    return folder


def make_package(w: World, remote_dir="packages/atlas-v3", *, gguf_size=3000, mmproj_size=1200, **overrides):
    """A package in the pod's Jupyter root; overrides replace manifest fields. Returns the manifest bytes as put and the
    two files' bytes."""
    gguf_bytes, mmproj_bytes = blob(gguf_size, 1), blob(mmproj_size, 2)
    fields = {"version": "atlas-v3", "base": "Qwen/Qwen3.5-4B", "task": "desk-objects", "gguf": GGUF, "mmproj": MMPROJ,
              "sha256": {GGUF: sha(gguf_bytes), MMPROJ: sha(mmproj_bytes)}, "scores": {"candidate": 0.78}, "created": WHEN}
    manifest = ModelManifest(**{**fields, **overrides}).model_dump_json().encode("utf-8")
    w.fake.put_file("D", f"{remote_dir}/manifest.json", manifest)
    w.fake.put_file("D", f"{remote_dir}/{GGUF}", gguf_bytes)
    w.fake.put_file("D", f"{remote_dir}/{MMPROJ}", mmproj_bytes)
    return manifest, gguf_bytes, mmproj_bytes


# --- remote paths and names ------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["", "/abs", "a/../b", "a//b", "a\\b", "a/./b", "a/b/", "..", ".", "a\x00b", "a\nb"])
def test_a_remote_path_that_could_leave_its_folder_is_refused(path):
    with pytest.raises(UnsafePath):
        clean_remote(path)


def test_a_plain_remote_path_is_kept_as_it_is():
    assert clean_remote("datasets/ds-0007/images/0001.jpg") == "datasets/ds-0007/images/0001.jpg"
    assert clean_remote("with space/a.bin") == "with space/a.bin"


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b", "a\\b", "C:evil", "-lead", ".hidden", "trail.", "a b",
                                  "x" * 201])
def test_a_name_must_be_one_plain_name(name):
    with pytest.raises(UnsafePath):
        clean_name(name)


def test_the_names_a_package_and_a_dataset_really_use_are_plain():
    for name in ("atlas-v3-desk-objects", "Qwen3.5-4B-Q4_K_M.gguf", "mmproj-F16.gguf", "ds-0007", "atlas_v3+x"):
        assert clean_name(name) == name


# --- pushing a file -------------------------------------------------------------------------------------------

def test_a_small_file_goes_in_one_put_after_its_folders_and_its_size_is_checked(tmp_path):
    w = World()
    data = blob(1000)
    local = tmp_path / "small.bin"
    local.write_bytes(data)

    stats = push_file(w.files, local, "a/b/small.bin")

    assert w.remote("a/b/small.bin") == data
    puts = w.put_calls()
    assert [(c.path.removeprefix(CONTENTS), c.body["type"]) for c in puts] == [
        ("a", "directory"), ("a/b", "directory"), ("a/b/small.bin", "file")]
    assert puts[-1].body == {"type": "file", "format": "base64", "content": 1336}      # 1000 bytes are 1336 base64 characters
    assert (stats.bytes, stats.requests, stats.retries, stats.verified, stats.route) == (1000, 1, 0, True, "contents")


def test_a_big_file_goes_in_numbered_chunks_the_last_one_minus_one(tmp_path):
    w = World()
    data = blob(2500)
    local = tmp_path / "big.bin"
    local.write_bytes(data)
    seen = []

    stats = push_file(w.files, local, "big.bin", chunk_bytes=1000, progress=lambda done, total: seen.append((done, total)))

    puts = w.put_calls("big.bin")
    assert [c.body["chunk"] for c in puts] == [1, 2, -1]
    assert [c.body["content"] for c in puts] == [1336, 1336, 668]
    assert all(c.body["format"] == "base64" and c.body["type"] == "file" for c in puts)
    assert w.remote("big.bin") == data
    assert seen == [(1000, 2500), (2000, 2500), (2500, 2500)]
    assert (stats.bytes, stats.requests, stats.part_bytes, stats.verified) == (2500, 3, 1000, True)


def test_a_file_of_exactly_two_chunks_is_chunk_one_then_minus_one(tmp_path):
    w = World()
    data = blob(2000)
    local = tmp_path / "two.bin"
    local.write_bytes(data)

    push_file(w.files, local, "two.bin", chunk_bytes=1000)

    assert [c.body["chunk"] for c in w.put_calls("two.bin")] == [1, -1]
    assert w.remote("two.bin") == data


def test_an_empty_file_is_pushed_as_an_empty_file(tmp_path):
    w = World()
    local = tmp_path / "empty.bin"
    local.write_bytes(b"")

    stats = push_file(w.files, local, "empty.bin")

    assert w.remote("empty.bin") == b""
    assert (stats.bytes, stats.verified) == (0, True)


@pytest.mark.parametrize("chunk", [0, -5, transfer.MAX_CHUNK_BYTES + 1])
def test_a_chunk_size_outside_the_limits_is_refused_before_anything_is_sent(tmp_path, chunk):
    w = World()
    local = tmp_path / "x.bin"
    local.write_bytes(b"x")

    with pytest.raises(ValueError):
        push_file(w.files, local, "x.bin", chunk_bytes=chunk)

    assert w.fake.calls == []


def test_the_default_chunk_is_8_mib_and_the_ceiling_keeps_a_chunk_under_the_cloudflare_cap():
    assert transfer.DEFAULT_CHUNK_BYTES == 8 * 1024 * 1024
    assert transfer.MAX_CHUNK_BYTES * 4 / 3 < 100_000_000


def test_folders_are_made_once_however_many_files_go_into_them(tmp_path):
    w = World()
    for name in ("one.bin", "two.bin"):
        (tmp_path / name).write_bytes(blob(10))
        push_file(w.files, tmp_path / name, f"x/y/{name}")

    directories = [c.path.removeprefix(CONTENTS) for c in w.put_calls() if c.body["type"] == "directory"]
    assert directories == ["x", "x/y"]


def test_a_folder_that_is_a_file_on_the_pod_stops_the_push(tmp_path):
    w = World()
    w.fake.put_file("D", "x", b"i am a file")
    (tmp_path / "f.bin").write_bytes(b"1")

    with pytest.raises(TransferError, match="x is a file on the pod"):
        push_file(w.files, tmp_path / "f.bin", "x/f.bin")


def test_a_chunked_push_replaces_a_file_that_is_already_there(tmp_path):
    w = World()
    w.fake.put_file("D", "big.bin", blob(5000, 9))
    data = blob(2500)
    (tmp_path / "big.bin").write_bytes(data)

    push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert [c.method for c in w.fake.calls_matching(None, "big.bin")][:2] == ["GET", "DELETE"]
    assert w.remote("big.bin") == data


def test_a_chunk_whose_answer_was_lost_after_it_landed_is_not_sent_again(tmp_path):
    w = World()
    data = blob(3000)
    (tmp_path / "big.bin").write_bytes(data)
    w.fake.queue("PUT", "/api/contents/big.bin", 502, text="<html>Bad gateway</html>", headers=HTML, apply=True, after=1)

    stats = push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert w.remote("big.bin") == data                  # a blind second send would have appended the chunk twice
    assert len(w.put_calls("big.bin")) == 3
    assert (stats.retries, stats.verified) == (0, True)


def test_a_chunk_that_did_not_land_is_sent_again_after_a_pause(tmp_path):
    w = World()
    data = blob(3000)
    (tmp_path / "big.bin").write_bytes(data)
    w.fake.queue("PUT", "/api/contents/big.bin", 503, text="Service Unavailable", after=1)

    stats = push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert w.remote("big.bin") == data
    assert [c.body["chunk"] for c in w.put_calls("big.bin")] == [1, 2, 2, -1]
    assert (stats.retries, stats.requests) == (1, 4)
    assert w.clock.elapsed == pytest.approx(1.0, abs=1e-3)


def test_a_dropped_connection_on_the_first_chunk_is_settled_by_looking_for_the_file(tmp_path):
    w = World()
    data = blob(2500)
    (tmp_path / "big.bin").write_bytes(data)
    w.fake.queue_dropped("PUT", "/api/contents/big.bin")

    stats = push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert w.remote("big.bin") == data
    assert [c.body["chunk"] for c in w.put_calls("big.bin")] == [1, 1, 2, -1]
    assert stats.retries == 1


def test_a_push_stops_when_the_size_after_a_failed_chunk_matches_neither_boundary(tmp_path):
    w = World()
    (tmp_path / "big.bin").write_bytes(blob(3000))
    w.fake.queue("PUT", "/api/contents/big.bin", 503, text="Service Unavailable", after=1)
    w.fake.queue("GET", "/api/contents/big.bin", 200, body={"type": "file", "size": 1500}, after=1)

    with pytest.raises(TransferError, match="holds 1500 bytes after a failed chunk"):
        push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert len(w.put_calls("big.bin")) == 2


def test_a_chunk_that_keeps_failing_gives_up_after_its_retries(tmp_path):
    w = World()
    (tmp_path / "big.bin").write_bytes(blob(3000))
    w.fake.queue("PUT", "/api/contents/big.bin", 503, text="Service Unavailable", after=1, times=20)

    with pytest.raises(PodUnresponsive):
        push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert len(w.put_calls("big.bin")) == 1 + 4           # the first chunk, then the second one and three more tries
    assert w.clock.elapsed == pytest.approx(7.0, abs=1e-3)


def test_a_token_problem_in_the_middle_of_a_push_stops_it_at_once(tmp_path):
    w = World()
    (tmp_path / "big.bin").write_bytes(blob(3000))
    w.fake.queue("PUT", "/api/contents/big.bin", 403, body={"message": "Forbidden"}, after=1)

    with pytest.raises(BadToken):
        push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1000)

    assert len(w.put_calls("big.bin")) == 2
    assert w.clock.elapsed == 0


def test_the_size_on_the_pod_is_checked_when_the_last_byte_is_in(tmp_path):
    w = World()
    (tmp_path / "small.bin").write_bytes(blob(1000))
    w.fake.queue("GET", "/api/contents/small.bin", 200, body={"type": "file", "size": 5})

    with pytest.raises(TransferError, match="5 bytes on the pod and 1000 here"):
        push_file(w.files, tmp_path / "small.bin", "small.bin")


def test_a_push_to_an_unsafe_path_sends_nothing(tmp_path):
    w = World()
    (tmp_path / "x.bin").write_bytes(b"x")

    with pytest.raises(UnsafePath):
        push_file(w.files, tmp_path / "x.bin", "../x.bin")

    assert w.fake.calls == []


def test_the_push_speed_is_reported_in_megabytes_a_second_and_named_as_the_link_metric(tmp_path):
    w = World(bandwidth=1_000_000)
    (tmp_path / "big.bin").write_bytes(blob(2_000_000))

    stats = push_file(w.files, tmp_path / "big.bin", "big.bin", chunk_bytes=1_000_000)

    assert stats.seconds == pytest.approx(2.667, abs=0.01)         # two bodies of about 1.33 MB of base64
    assert stats.mb_s == pytest.approx(0.75, abs=0.01)
    assert stats.metrics() == {"link.push_mb_s": stats.mb_s}


# --- pushing a dataset -----------------------------------------------------------------------------------------

def test_an_approved_export_goes_to_datasets_with_the_manifest_last(tmp_path):
    w = World()
    folder = make_dataset(tmp_path)

    done = push_dataset(w.files, folder)

    names = ["examples.jsonl", "images/0000.jpg", "images/0001.jpg", "thumbs/0000.jpg", "manifest.json"]
    sent = [c.path.removeprefix(CONTENTS) for c in w.put_calls() if c.body["type"] == "file"]
    assert sent == [f"datasets/ds-0007/{name}" for name in names]
    for name in names:
        assert w.remote(f"datasets/ds-0007/{name}") == (folder / name).read_bytes()
    assert (done.dataset_id, done.remote_dir, done.files, done.skipped, done.persistent) == (
        "ds-0007", "datasets/ds-0007", 5, [], False)
    assert done.bytes == sum((folder / name).stat().st_size for name in names)
    assert done.stats.verified is True


def test_the_report_says_when_the_pods_root_has_a_home_folder(tmp_path):
    w = World()
    w.fake.pod("D").root_has_home = True

    done = push_dataset(w.files, make_dataset(tmp_path))

    assert done.persistent is True
    assert w.files.layout().has_home is True and "home" in w.files.layout().names


def test_the_destination_folder_can_be_chosen(tmp_path):
    w = World()
    folder = make_dataset(tmp_path, "ds-0008")

    done = push_dataset(w.files, folder, remote_root="exports/today")
    at_root = push_dataset(w.files, folder, remote_root="")

    assert done.remote_dir == "exports/today/ds-0008" and at_root.remote_dir == "ds-0008"
    assert w.remote("exports/today/ds-0008/manifest.json") == w.remote("ds-0008/manifest.json")


def test_an_export_with_no_approval_id_is_not_pushed(tmp_path):
    w = World()
    folder = make_dataset(tmp_path, action_id="  ")

    with pytest.raises(NotApproved, match="no approval id"):
        push_dataset(w.files, folder)

    assert w.fake.calls == []


def test_a_folder_that_is_not_an_export_is_not_pushed(tmp_path):
    w = World()
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "notes.txt").write_text("hello")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"dataset_id": 3}))

    with pytest.raises(NotApproved, match="no manifest.json"):
        push_dataset(w.files, tmp_path / "plain")
    with pytest.raises(NotApproved, match="not a dataset manifest"):
        push_dataset(w.files, bad)

    assert w.fake.calls == []


def test_a_dataset_id_that_could_leave_its_folder_is_refused(tmp_path):
    w = World()
    folder = make_dataset(tmp_path, "ds-0007")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["dataset_id"] = "../evil"
    (folder / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(UnsafePath):
        push_dataset(w.files, folder)

    assert w.fake.calls == []


def test_symbolic_links_inside_an_export_are_neither_followed_nor_sent(tmp_path):
    folder = make_dataset(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "inner.txt").write_text("SECRET")
    try:
        os.symlink(secret, folder / "link.txt")
        os.symlink(elsewhere, folder / "linkdir", target_is_directory=True)
    except OSError:
        pytest.skip("this machine cannot create symlinks")
    w = World()

    done = push_dataset(w.files, folder)

    assert done.skipped == ["link.txt", "linkdir"]
    assert done.files == 5
    assert all(b"SECRET" not in bytes(data) for data in w.fake.pod("D").files.values())


# --- pulling a file ---------------------------------------------------------------------------------------------

def test_a_file_comes_in_byte_range_parts_and_is_checked_against_its_checksum(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a/b.bin", data)
    out = tmp_path / "out" / "b.bin"

    stats = pull_file(w.files, "a/b.bin", out, part_bytes=1000, sha256=sha(data))

    assert out.read_bytes() == data
    assert not (tmp_path / "out" / "b.bin.part").exists()
    calls = w.fake.calls_matching("GET", "/files/")
    assert [c.headers["range"] for c in calls] == ["bytes=0-999", "bytes=1000-1999", "bytes=2000-2499"]
    assert all(c.path == f"/user/{USER}/files/a/b.bin" and c.headers["authorization"] == f"token {TOKEN}" for c in calls)
    assert (stats.bytes, stats.requests, stats.retries, stats.verified, stats.route) == (2500, 3, 0, True, "files")


def test_a_file_server_on_a_pod_port_is_reached_through_the_proxy_in_ranges(tmp_path):
    w = World()
    data = blob(2500)
    server = w.fake.serve_files("D", 8001, root="packages")
    w.fake.put_file("D", "packages/x/model.bin", data)
    out = tmp_path / "model.bin"

    stats = pull_file(w.files, "x/model.bin", out, via="proxy", port=8001, part_bytes=1000, sha256=sha(data))

    assert out.read_bytes() == data
    assert server.requests == [("x/model.bin", "bytes=0-999"), ("x/model.bin", "bytes=1000-1999"),
                               ("x/model.bin", "bytes=2000-2499")]
    assert server.authorizations == [f"token {TOKEN}"] * 3
    assert w.fake.calls[0].path == f"/user/{USER}/proxy/8001/x/model.bin"
    assert stats.route == "proxy:8001"


def test_a_server_that_ignores_range_is_taken_whole_as_the_first_request_but_cannot_continue_a_part(tmp_path):
    w = World()
    data = blob(700)
    server = w.fake.serve_files("D", 8001, ranges=False)
    w.fake.put_file("D", "f.bin", data)
    out = tmp_path / "f.bin"

    stats = pull_file(w.files, "f.bin", out, via="proxy", port=8001, part_bytes=1000)

    assert out.read_bytes() == data and stats.requests == 1 and len(server.requests) == 1

    other = tmp_path / "g.bin"
    (tmp_path / "g.bin.part").write_bytes(data[:100])
    with pytest.raises(RangeUnsupported, match="remove the .part file"):
        pull_file(w.files, "f.bin", other, via="proxy", port=8001, part_bytes=1000)
    assert (tmp_path / "g.bin.part").read_bytes() == data[:100] and not other.exists()


def test_a_whole_file_over_the_proxys_limit_is_refused_while_ranges_get_it_there(tmp_path):
    w = World(proxy_response_cap=1500)
    data = blob(2500)
    w.fake.put_file("D", "big.bin", data)
    w.fake.serve_files("D", 8001, ranges=False)

    with pytest.raises(ProxyTimeout) as caught:
        pull_file(w.files, "big.bin", tmp_path / "whole.bin", via="proxy", port=8001, part_bytes=1000)
    assert caught.value.status == 599

    w.fake.serve_files("D", 8002, ranges=True)
    pull_file(w.files, "big.bin", tmp_path / "parts.bin", via="proxy", port=8002, part_bytes=1000, sha256=sha(data))
    assert (tmp_path / "parts.bin").read_bytes() == data


def test_a_part_that_breaks_off_is_continued_from_the_byte_it_reached(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a.bin", data)
    w.fake.break_next_download_after = 400
    out = tmp_path / "a.bin"

    stats = pull_file(w.files, "a.bin", out, part_bytes=1000, sha256=sha(data))

    assert out.read_bytes() == data
    assert [c.headers["range"] for c in w.fake.calls_matching("GET", "/files/")] == [
        "bytes=0-999", "bytes=400-1399", "bytes=1400-2399", "bytes=2400-2499"]
    assert (stats.retries, stats.requests, stats.bytes, stats.verified) == (1, 4, 2500, True)
    assert w.clock.elapsed == pytest.approx(1.0, abs=1e-3)


def test_a_part_that_keeps_breaking_gives_up_and_leaves_what_arrived_for_the_next_run(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a.bin", data)
    out = tmp_path / "a.bin"
    w.fake.queue_dropped("GET", "/files/a.bin", times=10)

    with pytest.raises(Unreachable):
        pull_file(w.files, "a.bin", out, part_bytes=1000)

    assert len(w.fake.calls_matching("GET", "/files/a.bin")) == 4         # the first try and three more
    assert not out.exists() and (tmp_path / "a.bin.part").exists()


def test_a_part_file_from_an_earlier_run_is_continued(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a.bin", data)
    out = tmp_path / "a.bin"
    (tmp_path / "a.bin.part").write_bytes(data[:1200])

    stats = pull_file(w.files, "a.bin", out, part_bytes=1000, sha256=sha(data))

    assert out.read_bytes() == data
    assert [c.headers["range"] for c in w.fake.calls_matching("GET", "/files/")] == ["bytes=1200-2199", "bytes=2200-2499"]
    assert stats.bytes == 1300


def test_a_copy_that_does_not_match_its_checksum_is_removed_and_reported(tmp_path):
    w = World()
    w.fake.put_file("D", "a.bin", blob(2500))
    out = tmp_path / "a.bin"

    with pytest.raises(ChecksumMismatch):
        pull_file(w.files, "a.bin", out, part_bytes=1000, sha256=sha(b"something else"))

    assert not out.exists() and not (tmp_path / "a.bin.part").exists()


def test_a_continued_copy_with_a_bad_start_is_pulled_again_from_the_first_byte(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a.bin", data)
    out = tmp_path / "a.bin"
    (tmp_path / "a.bin.part").write_bytes(b"\xff" * 500)

    stats = pull_file(w.files, "a.bin", out, part_bytes=1000, sha256=sha(data))

    assert out.read_bytes() == data
    assert w.fake.calls_matching("GET", "/files/")[0].headers["range"] == "bytes=500-1499"
    assert any(c.headers["range"] == "bytes=0-999" for c in w.fake.calls_matching("GET", "/files/"))
    assert (stats.bytes, stats.verified) == (2000 + 2500, True)


def test_a_file_already_there_with_the_right_checksum_is_not_fetched_again(tmp_path):
    w = World()
    data = blob(900)
    w.fake.put_file("D", "a.bin", data)
    out = tmp_path / "a.bin"
    out.write_bytes(data)

    stats = pull_file(w.files, "a.bin", out, sha256=sha(data))

    assert (stats.skipped, stats.bytes, stats.verified) == (1, 0, True)
    assert w.fake.calls == []


def test_an_empty_file_is_pulled(tmp_path):
    w = World()
    w.fake.put_file("D", "empty.bin", b"")
    out = tmp_path / "empty.bin"

    stats = pull_file(w.files, "empty.bin", out, sha256=sha(b""))

    assert out.read_bytes() == b"" and stats.verified is True


def test_a_file_the_pod_does_not_have_is_not_found(tmp_path):
    w = World()

    with pytest.raises(NotFound):
        pull_file(w.files, "nope.bin", tmp_path / "nope.bin")

    assert not (tmp_path / "nope.bin").exists()


def test_a_range_answer_for_the_wrong_bytes_is_an_error(tmp_path):
    w = World()
    w.fake.put_file("D", "a.bin", blob(2500))
    w.fake.queue("GET", "/files/a.bin", 206, text="xxx", headers={"content-range": "bytes 5-7/2500"})

    with pytest.raises(TransferError, match="wrong range"):
        pull_file(w.files, "a.bin", tmp_path / "a.bin", part_bytes=1000)


def test_a_part_that_arrives_short_is_interrupted_and_tried_again(tmp_path):
    w = World()
    data = blob(2500)
    w.fake.put_file("D", "a.bin", data)
    w.fake.queue("GET", "/files/a.bin", 206, content=data[:300], headers={"content-range": "bytes 0-999/2500"})

    stats = pull_file(w.files, "a.bin", tmp_path / "a.bin", part_bytes=1000, sha256=sha(data))

    assert (tmp_path / "a.bin").read_bytes() == data
    assert [c.headers["range"] for c in w.fake.calls_matching("GET", "/files/")] == [
        "bytes=0-999", "bytes=300-1299", "bytes=1300-2299", "bytes=2300-2499"]
    assert stats.retries == 1


@pytest.mark.parametrize("part", [0, -1, transfer.MAX_PART_BYTES + 1])
def test_a_part_size_outside_the_limits_is_refused(tmp_path, part):
    w = World()

    with pytest.raises(ValueError):
        pull_file(w.files, "a.bin", tmp_path / "a.bin", part_bytes=part)

    assert w.fake.calls == []


def test_the_route_must_be_files_or_a_proxy_with_a_port(tmp_path):
    w = World()

    with pytest.raises(ValueError, match="needs a port"):
        pull_file(w.files, "a.bin", tmp_path / "a.bin", via="proxy")
    with pytest.raises(ValueError, match="unknown route"):
        pull_file(w.files, "a.bin", tmp_path / "a.bin", via="ssh")


def test_the_largest_part_is_under_the_proxys_limit():
    assert transfer.MAX_PART_BYTES < 100 * 1024 * 1024
    assert transfer.DEFAULT_PART_BYTES == 64 * 1024 * 1024


def test_the_pull_speed_is_reported_in_megabytes_a_second_and_named_as_the_link_metric(tmp_path):
    w = World(bandwidth=1_000_000)
    w.fake.put_file("D", "big.bin", blob(2_000_000))

    stats = pull_file(w.files, "big.bin", tmp_path / "big.bin", part_bytes=1_000_000)

    assert stats.seconds == pytest.approx(2.0, abs=0.01)
    assert stats.mb_s == pytest.approx(1.0, abs=0.01)
    assert stats.metrics() == {"link.pull_mb_s": stats.mb_s}


# --- pulling a package -----------------------------------------------------------------------------------------

def test_a_package_is_installed_as_a_version_the_model_switch_accepts(tmp_path):
    w = World()
    manifest, gguf, mmproj = make_package(w)
    models = tmp_path / "models"

    got = pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)

    assert (got.version, got.directory, got.already_installed) == ("atlas-v3", models / "atlas-v3", False)
    assert (got.directory / GGUF).read_bytes() == gguf and (got.directory / MMPROJ).read_bytes() == mmproj
    assert (got.directory / "manifest.json").read_bytes() == manifest
    assert got.files == {GGUF: 3000, MMPROJ: 1200}
    store = ModelStore(models)
    assert store.verify("atlas-v3") == []
    assert [m.version for m in store.versions()] == ["atlas-v3"]
    assert store.current() is None                      # it is installed, not switched to
    assert not (models / ".atlas-v3.part").exists()
    assert (got.stats.files, got.stats.bytes, got.stats.verified) == (2, 4200, True)


def test_an_installed_package_can_be_switched_to(tmp_path):
    w = World()
    make_package(w)
    models = tmp_path / "models"
    pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    store = ModelStore(models)

    try:
        store.activate("atlas-v3")
    except OSError:
        pytest.skip("this machine cannot create symlinks")

    assert store.current() == "atlas-v3"


def test_pulling_a_version_that_is_installed_and_intact_fetches_only_its_manifest(tmp_path):
    w = World()
    make_package(w)
    models = tmp_path / "models"
    pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    before = len(w.fake.calls)

    again = pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)

    assert again.already_installed is True
    assert again.files == {GGUF: 3000, MMPROJ: 1200}
    assert [c.path for c in w.fake.calls[before:]] == [f"/user/{USER}/files/packages/atlas-v3/manifest.json"]


@pytest.mark.parametrize("override", [
    {"gguf": "../evil.gguf"}, {"gguf": "sub/dir.gguf"}, {"mmproj": "C:evil.gguf"}, {"gguf": "back\\slash.gguf"},
    {"version": "current"}, {"version": "previous"}, {"version": "../x"}, {"version": ""}, {"version": ".hidden"}])
def test_a_manifest_naming_something_that_is_not_a_plain_file_or_version_is_refused(tmp_path, override):
    w = World()
    make_package(w, **override)
    models = tmp_path / "models"

    with pytest.raises(UnsafePath):
        pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)

    assert not models.exists()
    assert [c.path.rsplit("/", 1)[-1] for c in w.fake.calls] == ["manifest.json"]


def test_a_file_the_manifest_has_no_checksum_for_is_not_accepted(tmp_path):
    w = World()
    make_package(w, sha256={GGUF: "ab" * 32})

    with pytest.raises(TransferError, match=f"no checksum for {MMPROJ}"):
        pull_package(w.files, "packages/atlas-v3", tmp_path / "models", part_bytes=1000)

    assert [c.path.rsplit("/", 1)[-1] for c in w.fake.calls] == ["manifest.json"]


def test_a_manifest_that_is_not_a_manifest_or_is_far_too_big_is_refused(tmp_path):
    w = World()
    w.fake.put_file("D", "packages/bad/manifest.json", b'{"version": "x"}')
    w.fake.put_file("D", "packages/huge/manifest.json", b"x" * (transfer.MANIFEST_LIMIT_BYTES + 1))

    with pytest.raises(TransferError, match="not a model manifest"):
        pull_package(w.files, "packages/bad", tmp_path / "models")
    with pytest.raises(TransferError, match="larger than a manifest should be"):
        pull_package(w.files, "packages/huge", tmp_path / "models")


def test_a_file_that_does_not_match_the_manifest_leaves_no_version_behind(tmp_path):
    w = World()
    make_package(w)
    w.fake.put_file("D", f"packages/atlas-v3/{GGUF}", blob(3000, 99))        # the pod's copy is not the one the manifest names
    models = tmp_path / "models"

    with pytest.raises(ChecksumMismatch):
        pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)

    assert not (models / "atlas-v3").exists()
    assert ModelStore(models).versions() == []
    assert not (models / ".atlas-v3.part" / GGUF).exists()


def test_a_pull_that_failed_part_way_picks_up_where_it_stopped(tmp_path):
    w = World()
    make_package(w)
    models = tmp_path / "models"
    w.fake.queue("GET", f"/files/packages/atlas-v3/{MMPROJ}", 403, body={"message": "Forbidden"})

    with pytest.raises(BadToken):
        pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    assert not (models / "atlas-v3").exists() and (models / ".atlas-v3.part" / GGUF).is_file()

    got = pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)

    assert (got.stats.skipped, got.stats.bytes) == (1, 1200)       # the model file was kept, only the projector was fetched
    assert ModelStore(models).verify("atlas-v3") == []


def test_a_damaged_installed_version_is_replaced_only_when_asked(tmp_path):
    w = World()
    _, gguf, _ = make_package(w)
    models = tmp_path / "models"
    pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    (models / "atlas-v3" / GGUF).write_bytes(b"damaged")

    with pytest.raises(TransferError, match="installed but damaged"):
        pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    assert (models / "atlas-v3" / GGUF).read_bytes() == b"damaged"

    got = pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000, replace=True)

    assert got.already_installed is False
    assert (models / "atlas-v3" / GGUF).read_bytes() == gguf


def test_the_version_that_is_serving_is_never_replaced(tmp_path):
    w = World()
    make_package(w)
    models = tmp_path / "models"
    pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000)
    store = ModelStore(models)
    try:
        store.activate("atlas-v3")
    except OSError:
        pytest.skip("this machine cannot create symlinks")
    (models / "atlas-v3" / GGUF).write_bytes(b"damaged")

    with pytest.raises(TransferError, match="is the version that is serving"):
        pull_package(w.files, "packages/atlas-v3", models, part_bytes=1000, replace=True)

    assert (models / "atlas-v3" / GGUF).read_bytes() == b"damaged"


def test_a_package_can_be_pulled_through_a_file_server_port(tmp_path):
    w = World()
    manifest, gguf, _ = make_package(w, "packages/atlas-v3")
    w.fake.serve_files("D", 8001, root="packages")

    got = pull_package(w.files, "atlas-v3", tmp_path / "models", via="proxy", port=8001, part_bytes=1000)

    assert (got.directory / GGUF).read_bytes() == gguf
    assert got.stats.route == "proxy:8001"


# --- the command line --------------------------------------------------------------------------------------------

def run_cli(w, *argv, environ=None):
    out = io.StringIO()
    code = transfer.main(list(argv), environ=ENV if environ is None else environ, env_files=[],
                         transport=w.fake.transport, sleep=w.clock.sleep, clock=w.clock.time, stdout=out)
    return code, out.getvalue()


def test_push_dataset_prints_one_json_line_with_the_report(tmp_path):
    w = World()
    folder = make_dataset(tmp_path)

    code, out = run_cli(w, "push-dataset", "--account", "d", str(folder), "--to", "incoming", "--chunk-mib", "1")

    report = json.loads(out)
    assert code == 0
    assert (report["ok"], report["dataset_id"], report["remote_dir"], report["files"], report["persistent"]) == (
        True, "ds-0007", "incoming/ds-0007", 5, False)
    assert report["stats"]["part_bytes"] == 1024 * 1024
    assert TOKEN not in out and USER not in out


def test_push_file_and_pull_file_report_their_speed_as_a_metric(tmp_path):
    w = World(bandwidth=1_000_000)
    (tmp_path / "up.bin").write_bytes(blob(1_000_000))

    code, out = run_cli(w, "push-file", "--account", "D", str(tmp_path / "up.bin"), "up.bin", "--chunk-mib", "1")
    pushed = json.loads(out)
    code2, out2 = run_cli(w, "pull-file", "--account", "D", "up.bin", str(tmp_path / "down.bin"), "--sha256",
                          sha(blob(1_000_000)), "--part-mib", "1")
    pulled = json.loads(out2)

    assert (code, code2) == (0, 0)
    assert (tmp_path / "down.bin").read_bytes() == blob(1_000_000)
    assert pushed["metrics"] == {"link.push_mb_s": pushed["mb_s"]} and pushed["mb_s"] > 0
    assert pulled["metrics"] == {"link.pull_mb_s": pulled["mb_s"]} and pulled["verified"] is True


def test_pull_package_prints_the_version_to_switch_to(tmp_path):
    w = World()
    make_package(w)

    code, out = run_cli(w, "pull-package", "--account", "D", "packages/atlas-v3", "--models-dir", str(tmp_path / "models"),
                        "--part-mib", "1")

    report = json.loads(out)
    assert code == 0
    assert (report["version"], report["already_installed"], report["files"]) == ("atlas-v3", False, {GGUF: 3000, MMPROJ: 1200})
    assert report["directory"] == str(tmp_path / "models" / "atlas-v3")


def test_a_failed_call_is_exit_1_with_its_kind(tmp_path):
    w = World()

    code, out = run_cli(w, "pull-file", "--account", "D", "nope.bin", str(tmp_path / "nope.bin"))

    report = json.loads(out)
    assert code == 1 and (report["kind"], report["status"], report["retryable"]) == ("not_found", 404, False)


def test_a_refused_push_is_exit_1_and_names_why(tmp_path):
    w = World()
    folder = make_dataset(tmp_path, action_id="")

    code, out = run_cli(w, "push-dataset", "--account", "D", str(folder))

    report = json.loads(out)
    assert code == 1 and report["kind"] == "NotApproved" and "approval id" in report["error"]
    assert w.fake.calls == []


def test_missing_keys_and_a_bad_account_are_exit_2(tmp_path):
    w = World()

    code, out = run_cli(w, "pull-file", "--account", "D", "a", str(tmp_path / "a"), environ={"AUP_HUB_USER_D": USER})
    code2, out2 = run_cli(w, "pull-file", "--account", "dd", "a", str(tmp_path / "a"))

    assert code == 2 and json.loads(out)["missing"] == ["AUP_HUB_TOKEN_D"]
    assert code2 == 2 and json.loads(out2)["kind"] == "usage"
    assert w.fake.calls == []
