"""lab.link_speed: the push and pull speed measurement of a synthetic package, rehearsed with small sizes.

What is real here: the measurement code, lab.transfer's chunked push through the contents API, lab.package_pull's pull
in Range parts, lab.package_server (in one test), the ledger and the report. What is faked is the network and the pod:
lab.proxy_fake.FakePodProxy stands for the hub, the Jupyter contents API and a file server on a pod port (it refuses a
response over `RESPONSE_CAP` bytes the way the live proxy refuses one over 100 MiB and a request body over its edge cap
with 413), and a fake clock moves only when something sleeps for a number of bytes over the fake link's bandwidth. Each
test works in its own folders and ledger; nothing opens the live hub."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import zlib
from pathlib import Path

import pytest

from lab import __main__ as lab_main
from lab import link_speed, package_server
from lab.hub_fake import FakeClock, FakeHub
from lab.link_speed import LiveTarget
from lab.machine import ledger
from lab.package_fake import FakePodProxy as PackageProxy
from lab.proxy_fake import FakePodProxy

MB = 1_000_000
KIB = 1 << 10
TOKEN = "eeee5555ffff6666aaaa7777bbbb8888"
USER = "roscon02"
URL = "https://hub.test"
PORT = 8301
SCRATCH = "r14-link-speed"
CHUNK = 256 * KIB
PART = 256 * KIB
RESPONSE_CAP = 400_000             # stands for the live proxy's 100 MiB
ENV = {"AUP_HUB_USER_D": USER, "AUP_HUB_TOKEN_D": TOKEN, "LAB_HUB_URL": URL, "LAB_POD_ROLES": "C:pod-a,D:pod-b"}


class StepClock:
    """A clock that moves by `step` each time it is read."""

    def __init__(self, step: float = 0.01) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class World:
    """One pod D on the fake proxy with a file server on PORT, and the target that measures it."""

    def __init__(self, tmp_path: Path, *, edge_body_cap: int = 100_000_000, bandwidth: float | None = 2_000_000) -> None:
        self.tmp = tmp_path
        self.clock = FakeClock()
        self.fake = FakePodProxy(self.clock, edge_body_cap=edge_body_cap, proxy_response_cap=RESPONSE_CAP, bandwidth=bandwidth)
        self.account = self.fake.add_account("D", user=USER, token=TOKEN)
        self.state = tmp_path / "state"
        self.workdir = tmp_path / "work"
        self.listener = None
        self.served = 0
        self.stopped = 0
        self.target = LiveTarget(self.account, port=PORT, base_url=URL, transport=self.fake.transport,
                                 sleep=self.clock.sleep, clock=self.clock.time, serve=self.serve, unserve=self.unserve)

    def serve(self, scratch: str, version: str) -> None:
        self.served += 1
        self.listener = self.fake.serve_files("D", PORT, root=scratch)

    def unserve(self) -> None:
        self.stopped += 1
        self.fake.pod("D").listeners.pop(PORT, None)

    def measure(self, **options):
        options.setdefault("sizes", [MB, 3 * MB])
        options.setdefault("repeats", 3)
        options.setdefault("chunk_bytes", CHUNK)
        options.setdefault("part_bytes", PART)
        options.setdefault("probe", False)
        return link_speed.measure(self.target, scratch=SCRATCH, workdir=self.workdir, state_dir=self.state, **options)

    def rows(self, verb: str | None = None):
        rows = ledger.read(state_dir=self.state)
        return [row for row in rows if verb is None or row["verb"] == verb]

    def close(self) -> None:
        self.target.close()


@pytest.fixture
def world(tmp_path):
    made = World(tmp_path)
    yield made
    made.close()


# --- the synthetic file -------------------------------------------------------------------------------------

def test_payload_is_seeded_exact_in_size_and_does_not_compress(tmp_path):
    first, second, other = tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "c.bin"
    size = 2 * MB + 12_345                                        # not a multiple of the block it is written in
    sha_a = link_speed.make_payload(first, size, seed=14)
    sha_b = link_speed.make_payload(second, size, seed=14)
    sha_c = link_speed.make_payload(other, size, seed=15)
    assert first.stat().st_size == size
    assert sha_a == sha_b == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()
    assert sha_c != sha_a
    sample = first.read_bytes()[:MB]
    assert len(zlib.compress(sample, 9)) > 0.99 * len(sample)     # random bytes: nothing for a compressing link to save


def test_a_size_has_a_plain_label():
    assert link_speed.size_label(10**9) == "1GB"
    assert link_speed.size_label(3 * 10**9) == "3GB"
    assert link_speed.size_label(MB) == "1MB"
    assert link_speed.size_label(1_500_000) == "1500KB"
    assert link_speed.size_label(1234) == "1234B"


def test_summary_is_mean_sample_spread_and_range():
    summary = link_speed.summarize([10.0, 12.0, 14.0])
    assert (summary.n, summary.mean, summary.stdev, summary.minimum, summary.maximum) == (3, 12.0, 2.0, 10.0, 14.0)
    single = link_speed.summarize([7.5])
    assert (single.n, single.mean, single.stdev) == (1, 7.5, 0.0)
    assert link_speed.summarize([]) is None


# --- the measurement ----------------------------------------------------------------------------------------

def test_three_repeats_each_way_of_both_sizes_are_pushed_pulled_checked_and_recorded(world):
    report = world.measure()
    assert report.ok
    assert [item.size_bytes for item in report.sizes] == [MB, 3 * MB]
    for item in report.sizes:
        pushes = [run for run in item.runs if run.direction == "push"]
        pulls = [run for run in item.runs if run.direction == "pull"]
        assert (len(pushes), len(pulls)) == (3, 3)
        assert all(run.ok and run.bytes == item.size_bytes and run.mb_s > 0 for run in pushes + pulls)
        assert all(run.sha256_ok for run in pulls)
        assert item.pass_rate == (3, 3)
        assert all(run.size_bytes == run.bytes for run in pushes)

    push_rows, pull_rows = world.rows("link push"), world.rows("link pull")
    assert (len(push_rows), len(pull_rows)) == (6, 6)
    for row in push_rows + pull_rows:
        assert row["outcome"] == "ok"
        assert row["node"] == "pod-b"
        assert row["refs"]["account"] == "D"
        assert row["hub"] == "hub.test" and row["conditions"]["hub"] == "hub.test"
        assert row["bytes"] in (MB, 3 * MB) and row["seconds"] > 0
        assert row["package"] in ("1MB", "3MB") and row["repeat"] in (1, 2, 3)
        assert row["refs"]["run_id"] == report.run_id
    for row in push_rows:
        assert set(row["metrics"]) == {"link.push_mb_s"} and row["units"] == {"link.push_mb_s": "MB/s"}
        assert row["chunk_bytes"] == CHUNK and "part_bytes" not in row
        assert row["metrics"]["link.push_mb_s"] == pytest.approx(row["bytes"] / 1e6 / row["seconds"], rel=0.01)
    for row in pull_rows:
        assert set(row["metrics"]) == {"link.pull_mb_s"} and row["units"] == {"link.pull_mb_s": "MB/s"}
        assert row["part_bytes"] == PART and row["sha256_ok"] is True
    assert report.ledger_errors == []


def test_the_push_goes_in_chunks_and_the_pull_in_range_parts_never_a_whole_file(world):
    world.measure(sizes=[MB], repeats=1)
    model_puts = [call for call in world.fake.calls_matching("PUT", "model.gguf")]
    assert len(model_puts) == 4                                    # 1 MB in chunks of 256 KiB
    assert [call.body["chunk"] for call in model_puts] == [1, 2, 3, -1]
    ranges = [rng for path, rng in world.listener.requests if path.endswith("model.gguf")]
    assert ranges and all(rng and rng.startswith("bytes=") for rng in ranges)
    assert len(ranges) == 4                                        # four parts of at most 256 KiB
    assert all(call.status in (200, 206) for call in world.fake.calls_matching("GET", "/proxy/"))


def test_the_pod_and_the_box_are_left_as_found(world):
    world.measure()
    pod = world.fake.pod("D")
    assert pod.files == {} and pod.dirs == {""}                    # the scratch folder and every file in it are gone
    assert list(world.workdir.glob("*")) == [] if world.workdir.exists() else True
    assert world.stopped == 1 and world.served == 1                # one file server for the whole run, stopped at the end


def test_a_failed_push_is_a_row_without_a_speed_and_its_pull_is_skipped(world):
    world.fake.queue("PUT", "model.gguf", 413, text="too big")
    report = world.measure(sizes=[MB])
    item = report.sizes[0]
    pushes = [run for run in item.runs if run.direction == "push"]
    pulls = [run for run in item.runs if run.direction == "pull"]
    assert [run.ok for run in pushes] == [False, True, True]
    assert pushes[0].error and pushes[0].mb_s is None
    assert [run.repeat for run in pulls] == [2, 3]
    assert not report.ok
    failed = [row for row in world.rows("link push") if row["outcome"] != "ok"]
    assert len(failed) == 1 and failed[0]["metrics"] == {} and failed[0]["repeat"] == 1
    assert failed[0]["note"]


def test_a_pulled_copy_that_does_not_match_is_a_checksum_failure_and_has_no_speed(world):
    real_push = world.target.push
    pushes = []

    def damaging_push(local, remote, chunk_bytes):
        stats = real_push(local, remote, chunk_bytes)
        if remote.endswith("model.gguf"):
            pushes.append(remote)
            if len(pushes) == 2:
                world.fake.pod("D").files[remote][100] ^= 0xFF      # the copy on the pod went bad after the push
        return stats

    world.target.push = damaging_push
    report = world.measure(sizes=[MB])
    item = report.sizes[0]
    pulls = [run for run in item.runs if run.direction == "pull"]
    assert [run.sha256_ok for run in pulls[:2]] == [True, False]
    assert item.pass_rate == (2, 3)                                # repeat 3 pushes the file afresh, so it matches again
    assert not report.ok
    bad = [row for row in world.rows("link pull") if row["outcome"] != "ok"]
    assert bad and all(row["metrics"] == {} and row["sha256_ok"] is False for row in bad)
    assert "mismatch" in bad[0]["note"] or "sha256" in bad[0]["note"]


def test_a_crash_midway_still_removes_the_scratch_folder(world):
    def explode(version, dest, part_bytes):
        raise RuntimeError("the machine went away")

    world.target.pull = explode
    with pytest.raises(RuntimeError):
        world.measure(sizes=[MB], repeats=1)
    assert world.fake.pod("D").files == {}
    assert world.stopped == 1


# --- the largest part ---------------------------------------------------------------------------------------

def test_the_probe_doubles_the_part_until_the_api_refuses_and_keeps_the_last_accepted(tmp_path):
    world = World(tmp_path, edge_body_cap=1_000_000)               # base64 makes 512 KiB 699 KB and 1 MiB 1.4 MB on the wire
    try:
        probe = link_speed.probe_part_limit(world.target, SCRATCH, start_bytes=64 * KIB, ceiling_bytes=4 * 1024 * KIB,
                                            seed=14, state_dir=world.state, run_id="r14-test")
        assert [step.bytes for step in probe.steps] == [64 * KIB, 128 * KIB, 256 * KIB, 512 * KIB, 1024 * KIB]
        assert [step.accepted for step in probe.steps] == [True, True, True, True, False]
        assert probe.largest_accepted_bytes == 512 * KIB
        assert probe.first_refused_bytes == 1024 * KIB
        assert probe.steps[-1].status == 413 and not probe.ceiling_reached
        assert world.fake.pod("D").files == {}                     # the probe file is removed
        row = world.rows("link part-probe")[0]
        assert row["node"] == "pod-b" and row["refs"]["account"] == "D"
        assert row["metrics"] == {"link.max_part_mb": pytest.approx(512 * KIB / 1e6, rel=0.001)}
        assert row["first_refused_bytes"] == 1024 * KIB
    finally:
        world.close()


def test_a_probe_that_is_never_refused_stops_at_its_ceiling_and_says_so(world):
    probe = link_speed.probe_part_limit(world.target, SCRATCH, start_bytes=64 * KIB, ceiling_bytes=256 * KIB, seed=14,
                                        state_dir=world.state, run_id="r14-test")
    assert probe.largest_accepted_bytes == 256 * KIB
    assert probe.first_refused_bytes is None and probe.ceiling_reached


def test_the_full_run_probes_first_and_reports_the_largest_file_and_part(tmp_path):
    world = World(tmp_path, edge_body_cap=1_000_000)
    try:
        report = world.measure(sizes=[MB], repeats=1, probe=True, probe_start_bytes=64 * KIB, probe_ceiling_bytes=4 * 1024 * KIB)
        assert report.probe.largest_accepted_bytes == 512 * KIB
        assert report.largest_file_bytes == MB                     # the biggest file that was pushed whole, in chunks
        assert "512 KiB" in report.markdown()
    finally:
        world.close()


# --- the report ---------------------------------------------------------------------------------------------

def test_the_table_has_one_line_per_package_and_direction_with_spread_and_checks(world):
    report = world.measure(probe=False)
    text = report.markdown()
    assert text.startswith("# R14")
    header = next(line for line in text.splitlines() if line.startswith("| Package"))
    for column in ("Direction", "Chunk or part", "Runs ok", "MB/s mean", "stdev", "min", "max", "sha256"):
        assert column in header
    table = text.split("| Package", 1)[1].split("

", 1)[0].splitlines()
    lines = [line for line in table if line.startswith("| 1MB") or line.startswith("| 3MB")]
    assert len(lines) == 4
    assert all("3 of 3" in line for line in lines)
    assert "pod-b" in text and "hub.test" in text and "account D" in text
    assert TOKEN not in text and USER not in text
    as_json = json.dumps(report.as_dict())
    assert TOKEN not in as_json and USER not in as_json


def test_a_package_with_no_successful_run_shows_dashes_not_a_made_up_speed(world):
    world.fake.queue("PUT", "model.gguf", 413, text="too big", times=100)
    report = world.measure(sizes=[MB], repeats=2)
    text = report.markdown()
    row = next(line for line in text.splitlines() if line.startswith("| 1MB") and "push" in line)
    assert "0 of 2" in row and "n/a" in row


# --- the real file server -----------------------------------------------------------------------------------

def test_the_pull_works_against_the_real_package_server_through_the_package_fake(tmp_path):
    pod_files = tmp_path / "podroot"
    pushed = FakePodProxy(FakeClock(), edge_body_cap=100_000_000)
    account = pushed.add_account("D", user=USER, token=TOKEN)
    hub = FakeHub(FakeClock())
    hub_account = hub.add_account("D", user=USER, token=TOKEN)
    hub.start_server("D")
    server = package_server.make_server(str(pod_files), "127.0.0.1", 0, None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = PackageProxy(hub, hub_account, server.server_address[1], port=PORT, response_limit=RESPONSE_CAP)
    clock = StepClock()

    def serve(scratch, version):
        for path, data in pushed.pod("D").files.items():           # the pod's Jupyter root is what the server serves
            target = pod_files / path.removeprefix(scratch + "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(data))

    target = LiveTarget(account, port=PORT, base_url=URL, transport=pushed.transport, pull_transport=proxy.transport,
                        sleep=lambda s: None, clock=clock, serve=serve, unserve=lambda: None)
    try:
        report = link_speed.measure(target, sizes=[MB], repeats=2, chunk_bytes=CHUNK, part_bytes=PART, scratch=SCRATCH,
                                    workdir=tmp_path / "work", probe=False, state_dir=tmp_path / "state")
        assert report.ok
        assert report.sizes[0].pass_rate == (2, 2)
        assert [run.parts for run in report.sizes[0].runs if run.direction == "pull"] == [4, 4]
        assert [seen.range for seen in proxy.requests("model.gguf")][:2] == ["bytes=0-262143", "bytes=262144-524287"]
    finally:
        target.close()
        server.shutdown()
        server.server_close()
        proxy.close()


# --- the pod-side scripts -----------------------------------------------------------------------------------

def test_the_start_script_has_its_own_file_name_and_never_touches_the_standing_package_server():
    script = link_speed.serve_script(8301, "r14-link-speed", "r14-1gb")
    assert "8301" in script and "r14-link-speed/r14-1gb/manifest.json" in script
    assert "r14_file_server.py --root" in script
    assert "pkill -u \"$(id -u)\" -f 'package_server.py" not in script       # R10's server on 8300 is not killed
    assert "$HOME/lab" not in script and "package_server.log" not in script
    assert TOKEN not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash here")
def test_both_scripts_are_valid_shell(tmp_path):
    for name, text in (("start.sh", link_speed.serve_script(8301, "r14-link-speed", "r14-1gb")), ("stop.sh", link_speed.stop_script())):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8", newline="\n")
        done = subprocess.run(["bash", "-n", path.name], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


def test_the_stop_script_stops_only_its_own_server_and_removes_only_its_own_files():
    script = link_speed.stop_script()
    assert "r14_file_server.py --root" in script
    assert "package_server.py" not in script
    assert "rm -rf \"$HOME/.r14_link_speed\"" in script


# --- the verb -----------------------------------------------------------------------------------------------

def test_the_verb_is_registered_in_order():
    assert lab_main.VERBS["link-speed"] == "lab.link_speed:main"
    assert list(lab_main.VERBS) == sorted(lab_main.VERBS)


def test_main_runs_the_measurement_writes_the_report_and_answers_with_one_dict(world, tmp_path):
    report_path = tmp_path / "R14_REPORT.md"
    answer = link_speed.main(["--account", "D", "--sizes-mb", "1,3", "--repeats", "2", "--chunk-mib", "0.25", "--part-mib", "0.25",
                              "--no-probe", "--scratch", SCRATCH, "--workdir", str(world.workdir), "--report", str(report_path)],
                             target=world.target, state_dir=world.state)
    assert answer["ok"] is True and answer["outcome"] == "ok"
    assert answer["runs"] == 8 and answer["report"] == str(report_path)
    assert report_path.read_text(encoding="utf-8").startswith("# R14")
    assert TOKEN not in json.dumps(answer) and USER not in json.dumps(answer)


def test_main_without_keys_names_the_missing_keys_and_not_any_value(tmp_path):
    answer = link_speed.main(["--account", "D", "--report", str(tmp_path / "r.md")], environ={}, env_files=[])
    assert answer["ok"] is False and answer["outcome"] == "no_credentials"
    assert answer["missing"] == ["AUP_HUB_USER_D", "AUP_HUB_TOKEN_D"]


def test_main_refuses_a_size_or_a_repeat_that_makes_no_sense(tmp_path):
    for bad in (["--sizes-mb", "0"], ["--sizes-mb", "x"], ["--repeats", "0"], ["--chunk-mib", "65"], ["--part-mib", "96"]):
        with pytest.raises(SystemExit) as stop:
            link_speed.main(["--account", "D", *bad], environ={}, env_files=[])
        assert stop.value.code == 2
