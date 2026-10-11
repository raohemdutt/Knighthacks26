"""lab.lab_push: a training run's files read from a pod (or a folder) and posted to the box's Learning lab.

Every test runs the real verb. The pod is lab.proxy_fake.FakePodProxy (the lab's stand-in for the hub and a pod's
Jupyter server, on lab.hub_fake.FakeClock, so a minute of waiting takes no time), and the box is FakeBox below: a real
HTTP server on this machine that answers the import route with the box's own parsing and dedupe
(contexts/learning/lab_import.py), the gateway's token rule and its size cap. The step lines are written by the trainer's
own StepLog and the ledger by the lab's own experiments verb. Nothing here reaches a live pod or the live box."""
from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

import lab.__main__ as lab_cli
from contexts.learning import lab_import
from contracts.learning import ModelManifest, RunEvaluated
from contracts.telemetry import TrainStep
from lab import lab_push, train
from lab.hub_fake import FakeClock, FakeHub
from lab.machine import experiments
from lab.proxy_fake import FakePodProxy

HUB_TOKEN = "0123456789abcdef0123456789abcdef"
BOX_TOKEN = "box-token-for-the-relay-tests"
USER = "roscon02"
RUN = "train-20261011T032149Z-7dd098"
RUN_DIR = f"adapters/l16-{RUN}"
ENV = {"AUP_HUB_USER_D": USER, "AUP_HUB_TOKEN_D": HUB_TOKEN, "LAB_HUB_URL": "https://hub.test"}
IMPORT = "/api/learning/lab/import"
T0 = datetime(2026, 10, 11, 3, 22, tzinfo=timezone.utc)
RUN_JSON = json.dumps({"ok": True, "outcome": "trained", "refs": {"run_id": RUN}, "metrics": {"steps": 18}}).encode() + b"\n"


class Tick:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def step_lines(folder: Path, count: int, run_id: str = RUN) -> list[bytes]:
    """The lines of a telemetry.jsonl as lab/train.py's own StepLog writes them, one step every 40 s."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"steps-{run_id}.jsonl"
    path.unlink(missing_ok=True)
    wall, timer = Tick(T0.timestamp()), Tick(10.0)
    log = train.StepLog(path, run_id, train.POD_NODE, None, clock=wall, timer=timer)
    for step in range(1, count + 1):
        log.begin()
        wall.now, timer.now = wall.now + 40.0, timer.now + 40.0
        log.end(step)
        log.logged(step, {"loss": round(2.0 / step, 4), "num_tokens": 900.0 * step})
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) == count
    return lines


def scores_json(run_id: str = RUN) -> bytes:
    """scores.json as lab/eval_gate.py writes it: its result keys and the gate's own event under "event"."""
    event = RunEvaluated(event_id=f"gate-{run_id}", ts=datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc), task="tool-calling",
                         run_id=run_id, decision="deploy", model_version="atlas-v3-tool-calling",
                         scores={"current": 0.4, "candidate": 0.8, "teacher": 0.9, "gate.noise": 0.02})
    return json.dumps({"decision": "deploy", "reasons": [], "run_id": run_id, "task": "tool-calling", "scores": event.scores,
                       "noise": 0.02, "p_value": 0.001, "n_items": 60, "n_groups": 20, "split_sha256": "ab" * 32,
                       "decode": {"thinking": "off", "temperature": 0}, "event": json.loads(event.model_dump_json())}).encode()


def manifest_json() -> bytes:
    return ModelManifest(version="atlas-v3-tool-calling", base="Qwen/Qwen3.5-4B", task="tool-calling", gguf="atlas-v3.gguf",
                         sha256={"atlas-v3.gguf": "cd" * 32}, created=datetime(2026, 10, 11, 6, 10, tzinfo=timezone.utc)
                         ).model_dump_json().encode()


class FakeBox:
    """The box's import route on a real socket. token: the gateway's ATLAS_TOKEN (None: writes are open). route=False:
    a box that was not redeployed and has no such route; ledger=False: one whose import is older than kind=ledger.
    fail_next: statuses answered before anything else. data_dir is where the box keeps a pushed ledger."""

    def __init__(self, token: str | None = None, route: bool = True, ledger: bool = True) -> None:
        self.token, self.route, self.ledger = token, route, ledger
        self._data = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._data.name)
        self.seen: set[tuple] = set()
        self.events: list = []
        self.posts: list[dict] = []
        self.fail_next: list[int] = []
        box = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, status: int, payload: object) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                known = box.route and urlsplit(self.path).path == IMPORT
                self._send(405 if known else 404, {"detail": "Method Not Allowed" if known else "Not Found"})

            def do_POST(self) -> None:
                parts = urlsplit(self.path)
                query = dict(parse_qsl(parts.query))
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                authorization = self.headers.get("Authorization")
                box.posts.append({"path": parts.path, "query": query, "body": body, "authorization": authorization})
                self._send(*box.answer(parts.path, query, body, authorization))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def answer(self, path: str, query: dict, body: bytes, authorization: str | None) -> tuple[int, object]:
        if self.fail_next:
            return self.fail_next.pop(0), {"detail": "Service Unavailable"}
        if path != IMPORT or not self.route:
            return 404, {"detail": "Not Found"}
        if self.token and authorization != f"Bearer {self.token}":
            return 401, {"detail": "missing or wrong token"}
        if len(body) > lab_import.MAX_BYTES:
            return 413, {"detail": f"the file is over {lab_import.MAX_BYTES} bytes"}
        kind = query.get("kind", "telemetry")
        try:
            if kind == lab_import.LEDGER and self.ledger:
                return 200, lab_import.store_ledger(body, self.data_dir)
            if kind == lab_import.LEDGER:
                raise lab_import.ImportRefused([f"kind must be one of {list(lab_import.KINDS)}"])   # the answer before that kind
            events = lab_import.events_of(kind, body.decode("utf-8"), query.get("run_id"))
        except lab_import.ImportRefused as err:
            return 400, {"detail": {"refused": err.problems[:50], "problems": len(err.problems)}}

        async def publish(event: object) -> None:
            self.events.append(event)

        return 200, {"kind": kind, **asyncio.run(lab_import.publish_new(events, self.seen, publish))}

    def steps(self, run_id: str = RUN) -> list[int]:
        return [e.step for e in self.events if isinstance(e, TrainStep) and e.run_id == run_id]

    def types(self) -> list[str]:
        return [e.type for e in self.events if not isinstance(e, TrainStep)]

    def kept_ledger(self) -> bytes | None:
        path = self.data_dir / "lab_state" / "ledger.jsonl"
        return path.read_bytes() if path.is_file() else None

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self._data.cleanup()


@pytest.fixture
def box():
    fake = FakeBox()
    yield fake
    fake.close()


class Pod:
    """Account D's pod on the fake proxy, with the run folder a trainer would be writing."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.fake = FakePodProxy(self.clock)
        self.fake.add_account("D", user=USER, token=HUB_TOKEN)

    def put(self, name: str, data: bytes, folder: str = RUN_DIR) -> None:
        self.fake.put_file("D", f"{folder}/{name}", data)


def push(argv: list[str], *, pod: Pod | None = None, box: FakeBox | None = None, environ: dict | None = None):
    """Run the verb. Returns its answer and what it wrote to stderr."""
    pod = pod or Pod()
    err = io.StringIO()
    argv = list(argv) + ([] if box is None or "--box" in argv else ["--box", box.url])
    result = lab_push.main(argv, environ={**ENV, **(environ or {})}, env_files=[], transport=pod.fake.transport,
                           sleep=pod.clock.sleep, clock=pod.clock.time, stderr=err)
    return result, err.getvalue()


# --- one read, one post ---------------------------------------------------------------------------

def test_a_run_on_the_pod_reaches_the_box_with_reads_only(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 11)
    pod.put("telemetry.jsonl", b"".join(lines))
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert result["ok"] is True and result["outcome"] == "pushed"
    assert result["run_id"] == RUN and result["source"] == "pod D" and result["finished"] is False and result["rounds"] == 1
    assert result["telemetry"] == {"bytes": len(b"".join(lines)), "lines": 11, "box": {"status": 200, "answer": {
        "kind": "telemetry", "published": 11, "already_there": 0, "events": 11, "runs": [RUN], "last_step": 11}}}
    assert box.steps() == list(range(1, 12))
    assert [(p["path"], p["query"], p["body"]) for p in box.posts] == [(IMPORT, {"kind": "telemetry", "run_id": RUN}, b"".join(lines))]
    # the pod was only read: every call a GET of /files/, each with a query the edge has never seen
    calls = pod.fake.calls
    assert [(c.method, c.path) for c in calls] == [("GET", f"/user/{USER}/files/{RUN_DIR}/run.json"),
                                                  ("GET", f"/user/{USER}/files/{RUN_DIR}/telemetry.jsonl")]
    assert all(c.query.startswith("_=") for c in calls) and len({c.query for c in calls}) == len(calls)
    assert pod.fake.file("D", f"{RUN_DIR}/telemetry.jsonl") == b"".join(lines)


def test_posting_the_same_file_again_adds_nothing(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 5)))
    push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    again, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert again["ok"] is True and again["telemetry"]["box"]["answer"]["published"] == 0
    assert again["telemetry"]["box"]["answer"]["already_there"] == 5 and box.steps() == [1, 2, 3, 4, 5]


def test_a_line_still_being_written_waits_for_the_next_read(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 4)
    pod.put("telemetry.jsonl", b"".join(lines[:3]) + lines[3][:40])
    first, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert first["ok"] is True and first["telemetry"]["lines"] == 3 and box.steps() == [1, 2, 3]
    assert box.posts[0]["body"] == b"".join(lines[:3])
    pod.put("telemetry.jsonl", b"".join(lines[:3]) + lines[3].rstrip(b"\n"))   # whole, its newline not yet written
    second, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert second["telemetry"]["lines"] == 4 and box.steps() == [1, 2, 3, 4]


def test_a_run_with_no_step_yet_posts_nothing(box):
    pod = Pod()
    absent, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert absent["ok"] is False and absent["outcome"] == "no_telemetry" and absent["telemetry"] == {"state": "absent"}
    pod.put("telemetry.jsonl", b"")
    empty, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert empty["ok"] is False and empty["outcome"] == "no_telemetry" and empty["telemetry"] == {"state": "empty"}
    assert box.posts == []


def test_a_file_over_the_cap_of_the_box_is_not_sent(tmp_path, box):
    pod, line = Pod(), step_lines(tmp_path, 1)[0]
    pod.put("telemetry.jsonl", line * (lab_import.MAX_BYTES // len(line) + 1))
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert result["ok"] is False and result["outcome"] == "too_large" and box.posts == []
    assert lab_push.BOX_MAX_BYTES == lab_import.MAX_BYTES


# --- following a run that is training -----------------------------------------------------------------

def test_follow_reposts_the_growing_file_until_run_json_appears_then_stops(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 18)
    pod.put("telemetry.jsonl", b"".join(lines[:11]))
    pod.clock.call_later(70, lambda: pod.put("telemetry.jsonl", b"".join(lines[:14])))

    def finish() -> None:
        pod.put("telemetry.jsonl", b"".join(lines))
        pod.put("run.json", RUN_JSON)

    pod.clock.call_later(130, finish)
    kept = tmp_path / "kept"
    result, said = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60", "--save", str(kept)], pod=pod, box=box)
    assert result["ok"] is True and result["outcome"] == "pushed" and result["finished"] is True
    assert result["rounds"] == 4 and pod.clock.elapsed == 180          # reads at 0, 60, 120 and 180 s, and no wait after the last
    assert [len(p["body"].splitlines()) for p in box.posts] == [11, 11, 14, 18]
    assert result["telemetry"]["box"]["answer"] == {"kind": "telemetry", "published": 4, "already_there": 14, "events": 18,
                                                    "runs": [RUN], "last_step": 18}
    assert box.steps() == list(range(1, 19))
    assert (kept / "telemetry.jsonl").read_bytes() == b"".join(lines) and (kept / "run.json").read_bytes() == RUN_JSON
    assert result["saved"] == {"run.json": len(RUN_JSON), "telemetry.jsonl": len(b"".join(lines))}
    assert said.count("\n") == 4 and "18 lines" in said.splitlines()[-1]
    assert all(c.method == "GET" for c in pod.fake.calls)


def test_follow_gives_up_at_its_bound_when_the_run_never_ends(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 3)))
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60", "--max-minutes", "5"], pod=pod, box=box)
    assert result["ok"] is False and result["outcome"] == "bound_reached" and result["finished"] is False
    assert result["rounds"] == 6 and pod.clock.elapsed == 300          # 0 to 300 s, never past the bound
    assert box.steps() == [1, 2, 3] and result["telemetry"]["box"]["status"] == 200


def test_follow_waits_for_the_first_step_and_rides_out_a_box_that_drops_one_post(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 2)
    pod.clock.call_later(30, lambda: pod.put("telemetry.jsonl", lines[0]))

    def finish() -> None:
        pod.put("telemetry.jsonl", b"".join(lines))
        pod.put("run.json", RUN_JSON)

    pod.clock.call_later(90, finish)
    box.fail_next = [503]
    result, said = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=box)
    # 0 s: no file. 60 s: one step, the box answers 503. 120 s: run.json and both steps, the box takes them.
    assert result["ok"] is True and result["outcome"] == "pushed" and result["rounds"] == 3
    assert [len(p["body"].splitlines()) for p in box.posts] == [1, 2] and box.steps() == [1, 2]
    assert "503" in said


def test_follow_goes_round_again_when_the_last_post_fails_after_the_run_ended(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 2)
    pod.put("telemetry.jsonl", b"".join(lines))
    pod.put("run.json", RUN_JSON)
    box.fail_next = [502]
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=box)
    assert result["ok"] is True and result["outcome"] == "pushed" and result["rounds"] == 2 and box.steps() == [1, 2]


# --- what the box and the pod can answer ------------------------------------------------------------------

def test_a_box_that_wants_a_token_stops_the_relay_and_the_token_from_the_environment_opens_it(tmp_path):
    pod, lines, guarded = Pod(), step_lines(tmp_path, 3), FakeBox(token=BOX_TOKEN)
    pod.put("telemetry.jsonl", b"".join(lines))
    try:
        refused, said = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=guarded)
        assert refused["ok"] is False and refused["outcome"] == "box_wants_token" and refused["rounds"] == 1
        assert refused["telemetry"]["box"] == {"status": 401, "answer": {"detail": "missing or wrong token"}}
        assert refused["token"] == "none" and guarded.steps() == [] and pod.clock.elapsed == 0
        opened, said = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=guarded, environ={"ATLAS_TOKEN": BOX_TOKEN})
        assert opened["ok"] is True and opened["token"] == "sent" and guarded.steps() == [1, 2, 3]
        assert guarded.posts[-1]["authorization"] == f"Bearer {BOX_TOKEN}"
        assert BOX_TOKEN not in json.dumps(opened) + said and HUB_TOKEN not in json.dumps(opened) + said
        assert USER not in json.dumps(opened) + said
    finally:
        guarded.close()


def test_a_box_without_the_route_is_named_and_not_retried(tmp_path):
    pod, old = Pod(), FakeBox(route=False)
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 3)))
    try:
        result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=old)
        assert result["ok"] is False and result["outcome"] == "route_missing" and result["rounds"] == 1
        assert result["telemetry"]["box"]["status"] == 404 and len(old.posts) == 1
    finally:
        old.close()


def test_a_box_nobody_answers_at_is_reported_without_a_traceback(tmp_path):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 2)))
    gone = FakeBox()
    gone.close()
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--box", gone.url], pod=pod)
    assert result["ok"] is False and result["outcome"] == "box_unreachable" and result["telemetry"]["box"]["status"] == 0


def test_steps_of_another_run_are_refused_by_the_box_and_the_reason_comes_back(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 2, run_id="some-other-run")))
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--run-id", RUN, "--follow", "--every", "60"], pod=pod, box=box)
    assert result["ok"] is False and result["outcome"] == "box_refused_file" and result["rounds"] == 1
    assert result["telemetry"]["box"] == {"status": 400, "answer": {"detail": {"problems": 2, "refused": [
        f"line 1: run 'some-other-run', not '{RUN}'", f"line 2: run 'some-other-run', not '{RUN}'"]}}}
    assert box.steps("some-other-run") == []


def test_a_wrong_hub_token_stops_the_relay_and_no_key_is_shown(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 2)))
    wrong = "f" * 32
    result, said = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=box,
                        environ={"AUP_HUB_TOKEN_D": wrong})
    assert result["ok"] is False and result["outcome"] == "pod_error" and result["kind"] == "bad_token" and result["rounds"] == 1
    assert box.posts == [] and wrong not in json.dumps(result) + said and USER not in json.dumps(result) + said


def test_follow_rides_out_a_pod_read_that_times_out_at_the_edge(tmp_path, box):
    pod, lines = Pod(), step_lines(tmp_path, 3)
    pod.put("telemetry.jsonl", b"".join(lines))
    pod.put("run.json", RUN_JSON)
    pod.fake.queue("GET", "telemetry.jsonl", 524, text="error code: 524")     # Cloudflare gave up on the origin, once
    result, said = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=box)
    assert result["ok"] is True and result["outcome"] == "pushed" and result["rounds"] == 2 and box.steps() == [1, 2, 3]
    assert "round 1: the pod could not be read" in said and len(box.posts) == 1
    # without --follow the same answer ends the call and is named
    pod.fake.queue("GET", "telemetry.jsonl", 524, text="error code: 524")
    once, _ = push(["--account", "D", "--run-dir", RUN_DIR], pod=pod, box=box)
    assert once["ok"] is False and once["outcome"] == "pod_error" and once["rounds"] == 1 and len(box.posts) == 1


def test_a_pod_that_has_stopped_ends_the_follow(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 3)))
    pod.fake.stop_pod("D")
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"], pod=pod, box=box)
    assert result["ok"] is False and result["outcome"] == "pod_error" and result["kind"] == "pod_stopped"
    assert result["rounds"] == 1 and box.posts == []


def test_missing_hub_keys_are_named_and_nothing_is_called(box):
    pod = Pod()
    err = io.StringIO()
    result = lab_push.main(["--account", "D", "--run-dir", RUN_DIR, "--box", box.url], environ={}, env_files=[],
                           transport=pod.fake.transport, sleep=pod.clock.sleep, clock=pod.clock.time, stderr=err)
    assert result == {"ok": False, "outcome": "no_credentials", "missing": ["AUP_HUB_USER_D", "AUP_HUB_TOKEN_D"],
                      "error": "keys missing from the environment and the .env files"}
    assert pod.fake.calls == [] and box.posts == []


@pytest.mark.parametrize("path", ["/home/jovyan/adapters/run", "adapters/../../etc", "adapters//run", "adapters\\run"])
def test_a_pod_path_that_is_not_below_the_jupyter_root_is_refused(path, box):
    pod = Pod()
    result, _ = push(["--account", "D", "--run-dir", path], pod=pod, box=box)
    assert result["ok"] is False and result["outcome"] == "unsafe_path" and pod.fake.calls == [] and box.posts == []


# --- the gate's scores and the package's manifest -----------------------------------------------------------

def test_scores_and_manifest_from_the_pod_follow_the_steps(tmp_path, box):
    pod = Pod()
    pod.put("telemetry.jsonl", b"".join(step_lines(tmp_path, 3)))
    pod.put("run.json", RUN_JSON)
    pod.put("scores.json", scores_json(), folder="l4/gate")
    pod.put("manifest.json", manifest_json(), folder="data/packages/atlas-v3-tool-calling")
    result, _ = push(["--account", "D", "--run-dir", RUN_DIR, "--eval", "l4/gate/scores.json",
                      "--manifest", "data/packages/atlas-v3-tool-calling/manifest.json"], pod=pod, box=box)
    assert result["ok"] is True and result["outcome"] == "pushed" and result["finished"] is True
    assert [p["query"] for p in box.posts] == [{"kind": "telemetry", "run_id": RUN}, {"kind": "eval", "run_id": RUN},
                                               {"kind": "manifest", "run_id": RUN}]
    # since L29 the box adds the learning.run.progress that moves the Teach screen's stage (eval, then packaging)
    assert result["eval"]["box"] == {"status": 200, "answer": {"kind": "eval", "published": 3, "already_there": 0, "events": 3,
                                                              "runs": [RUN], "last_step": None}}
    assert result["manifest"]["box"]["answer"]["published"] == 2
    assert box.types() == ["learning.run.evaluated", "learning.eval.reported", "learning.run.progress",
                           "learning.package.built", "learning.run.progress"]
    assert box.posts[1]["body"] == scores_json() and box.posts[2]["body"] == manifest_json()


def test_local_files_are_posted_without_any_hub_key(tmp_path):
    lines = step_lines(tmp_path, 4)
    run = tmp_path / "runs" / RUN
    run.mkdir(parents=True)
    (run / "telemetry.jsonl").write_bytes(b"".join(lines))
    (tmp_path / "scores.json").write_bytes(scores_json())
    pod, local_box = Pod(), FakeBox()
    try:
        result = lab_push.main(["--box", local_box.url, "--run-dir", str(run), "--eval", str(tmp_path / "scores.json")],
                               environ={}, env_files=[], transport=pod.fake.transport, stderr=io.StringIO())
        assert result["ok"] is True and result["source"] == "local" and result["run_id"] == RUN
        assert local_box.steps() == [1, 2, 3, 4] and local_box.types() == ["learning.run.evaluated", "learning.eval.reported",
                                                                           "learning.run.progress"]
        assert pod.fake.calls == []
    finally:
        local_box.close()


def test_an_eval_for_another_run_is_refused_by_the_box(tmp_path, box):
    (tmp_path / "scores.json").write_bytes(scores_json("run-of-yesterday"))
    result, _ = push(["--eval", str(tmp_path / "scores.json"), "--run-id", RUN], box=box)
    assert result["ok"] is False and result["outcome"] == "box_refused_file"
    assert result["eval"]["box"]["answer"] == {"detail": {"problems": 1, "refused": [f"the eval is for run 'run-of-yesterday', not '{RUN}'"]}}
    assert box.events == []


def test_a_manifest_needs_a_run_id_and_a_missing_file_is_named(tmp_path, box):
    (tmp_path / "manifest.json").write_bytes(manifest_json())
    nameless, _ = push(["--manifest", str(tmp_path / "manifest.json")], box=box)
    assert nameless["ok"] is False and nameless["outcome"] == "no_run_id" and box.posts == []
    named, _ = push(["--manifest", str(tmp_path / "manifest.json"), "--run-id", RUN], box=box)
    assert named["ok"] is True and box.types() == ["learning.package.built", "learning.run.progress"] and box.events[0].run_id == RUN
    absent, _ = push(["--eval", str(tmp_path / "no-such-scores.json"), "--run-id", RUN], box=box)
    assert absent["ok"] is False and absent["outcome"] == "missing_file" and absent["eval"] == {"state": "absent"}


# --- the lab's ledger ---------------------------------------------------------------------------------------

def seeded_ledger(tmp_path: Path) -> Path:
    """A ledger.jsonl written by the lab's own experiments verb: one `experiment add` row per seeded experiment."""
    state = tmp_path / "state"
    added = experiments.seed(state_dir=state)["added"]
    assert added and (state / "ledger.jsonl").read_bytes().count(b"\n") == len(added)
    return state / "ledger.jsonl"


def test_the_ledger_is_posted_whole_and_the_box_keeps_it(tmp_path, box):
    ledger = seeded_ledger(tmp_path)
    rows = ledger.read_bytes().count(b"\n")
    first, said = push(["--ledger", str(ledger)], box=box)
    assert first["ok"] is True and first["outcome"] == "pushed" and first["run_id"] is None
    assert [(p["query"], p["body"]) for p in box.posts] == [({"kind": "ledger"}, ledger.read_bytes())]
    answer = first["ledger"]["box"]["answer"]
    assert first["ledger"]["box"]["status"] == 200 and first["ledger"]["bytes"] == len(ledger.read_bytes())
    assert (answer["kind"], answer["changed"], answer["rows"], answer["experiments"]) == ("ledger", True, rows, rows)
    assert box.kept_ledger() == ledger.read_bytes()
    assert f"{rows} rows" in said and "replaced the copy there" in said
    again, _ = push(["--ledger", str(ledger)], box=box)
    assert again["ok"] is True and again["ledger"]["box"]["answer"]["changed"] is False


def test_a_ledger_row_still_being_appended_is_left_out_and_a_run_id_never_rides_with_a_ledger(tmp_path, box):
    ledger = seeded_ledger(tmp_path)
    whole = ledger.read_bytes()
    ledger.write_bytes(whole + b'{"seq": 99, "verb": "experiment ad')
    result, _ = push(["--ledger", str(ledger), "--run-id", RUN], box=box)
    assert result["ok"] is True and box.posts[0]["query"] == {"kind": "ledger"} and box.kept_ledger() == whole


def test_a_box_older_than_the_ledger_import_says_which_kinds_it_knows(tmp_path):
    ledger, old = seeded_ledger(tmp_path), FakeBox(ledger=False)
    try:
        result, _ = push(["--ledger", str(ledger)], box=old)
        assert result["ok"] is False and result["outcome"] == "box_refused_file"
        assert result["ledger"]["box"] == {"status": 400, "answer": {"detail": {"problems": 1, "refused": [
            "kind must be one of ['telemetry', 'eval', 'manifest', 'run']"]}}}   # the box's own kinds, run since L29
        assert old.kept_ledger() is None
    finally:
        old.close()


def test_an_empty_ledger_is_not_posted(tmp_path, box):
    (tmp_path / "ledger.jsonl").write_bytes(b"")
    result, _ = push(["--ledger", str(tmp_path / "ledger.jsonl")], box=box)
    assert result["ok"] is False and result["outcome"] == "empty_file" and box.posts == []


# --- the same verb against the lab's fake hub ---------------------------------------------------------------

def test_the_run_is_read_through_the_fake_hub_with_gets_only(tmp_path, box):
    clock = FakeClock()
    hub = FakeHub(clock)
    hub.add_account("D", user=USER, token=HUB_TOKEN)
    hub.start_server("D")
    lines = step_lines(tmp_path, 6)
    hub.files[("D", f"{RUN_DIR}/telemetry.jsonl")] = b"".join(lines[:4]).decode()

    def finish() -> None:
        hub.files[("D", f"{RUN_DIR}/telemetry.jsonl")] = b"".join(lines).decode()
        hub.files[("D", f"{RUN_DIR}/run.json")] = RUN_JSON.decode()

    clock.call_later(90, finish)
    result = lab_push.main(["--box", box.url, "--account", "D", "--run-dir", RUN_DIR, "--follow", "--every", "60"],
                           environ=dict(ENV), env_files=[], transport=hub.transport, sleep=clock.sleep,
                           clock=clock.time, stderr=io.StringIO())
    assert result["ok"] is True and result["outcome"] == "pushed" and result["rounds"] == 3 and result["finished"] is True
    assert box.steps() == [1, 2, 3, 4, 5, 6] and [len(p["body"].splitlines()) for p in box.posts] == [4, 4, 6]
    pod_calls = hub.calls_matching(contains=f"/user/{USER}/")
    assert len(pod_calls) == 6 and all(c.method == "GET" and "/files/" in c.path for c in pod_calls)
    assert hub.running("D") is True


def test_a_path_that_is_a_folder_is_an_answer_and_not_a_traceback(tmp_path, box):
    for extra in ([], ["--dry-run"]):
        result, _ = push(["--eval", str(tmp_path), "--run-id", RUN, *extra], box=box)
        assert result["ok"] is False and result["outcome"] == "read_error", extra
    assert box.posts == []


# --- the dry run and the command line ---------------------------------------------------------------------

def test_dry_run_names_every_read_and_post_and_sends_nothing(box):
    pod = Pod()
    result, said = push(["--account", "D", "--run-dir", RUN_DIR, "--run-id", RUN, "--eval", "l4/gate/scores.json", "--follow",
                         "--every", "60", "--max-minutes", "90", "--dry-run"], pod=pod, box=box, environ={"ATLAS_TOKEN": BOX_TOKEN})
    assert result["ok"] is True and result["outcome"] == "dry_run" and result["source"] == "pod D"
    assert result["would_read"] == [f"pod D: GET /files/{RUN_DIR}/run.json", f"pod D: GET /files/{RUN_DIR}/telemetry.jsonl",
                                    "pod D: GET /files/l4/gate/scores.json"]
    assert result["would_post"] == [f"POST {box.url}{IMPORT}?kind=telemetry&run_id={RUN}", f"POST {box.url}{IMPORT}?kind=eval&run_id={RUN}"]
    assert result["follow"] == {"every_s": 60.0, "max_minutes": 90.0, "until": "run.json appears in the run folder"}
    assert result["token"] == "sent"
    assert pod.fake.calls == [] and box.posts == [] and pod.clock.elapsed == 0
    assert BOX_TOKEN not in json.dumps(result) + said and HUB_TOKEN not in json.dumps(result) + said


def test_dry_run_of_a_local_folder_says_what_is_there(tmp_path, box):
    run = tmp_path / "run"
    run.mkdir()
    lines = step_lines(tmp_path, 3)
    (run / "telemetry.jsonl").write_bytes(b"".join(lines))
    result, _ = push(["--run-dir", str(run), "--dry-run"], box=box)
    assert result["ok"] is True and result["token"] == "none" and box.posts == []
    assert result["would_post"] == [f"POST {box.url}{IMPORT}?kind=telemetry&run_id={RUN}"]
    assert result["local"] == {"telemetry.jsonl": {"bytes": len(b"".join(lines)), "lines": 3}, "run.json": None}


def test_the_box_address_comes_from_the_flag_or_the_environment(tmp_path, box):
    (tmp_path / "telemetry.jsonl").write_bytes(b"".join(step_lines(tmp_path / "made", 2)))
    none, _ = push(["--run-dir", str(tmp_path)])
    assert none["ok"] is False and none["outcome"] == "no_box"
    bad, _ = push(["--run-dir", str(tmp_path), "--box", "100.124.34.36:8000"])
    assert bad["ok"] is False and bad["outcome"] == "no_box"
    result, _ = push(["--run-dir", str(tmp_path)], environ={"ATLAS_URL": box.url + "/"})
    assert result["ok"] is True and result["box"] == box.url and box.steps() == [1, 2]


@pytest.mark.parametrize("argv", [[], ["--follow", "--eval", "scores.json"], ["--run-dir", "x", "--every", "1"],
                                  ["--run-dir", "x", "--max-minutes", "0"], ["--run-dir", "x", "--account", "DD"]])
def test_bad_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        lab_push.main([*argv, "--box", "http://127.0.0.1:9"], environ=dict(ENV), env_files=[])
    assert exit_.value.code == 2
    assert "usage: python -m lab lab-push" in capsys.readouterr().err


def test_the_verb_is_registered_alphabetically_and_answers_help(capsys):
    assert lab_cli.VERBS["lab-push"] == "lab.lab_push:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)
    assert lab_cli.main(["lab-push", "-h"]) == 0
    assert "usage: python -m lab lab-push" in capsys.readouterr().out
