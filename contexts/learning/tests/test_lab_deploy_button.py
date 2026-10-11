"""The On the box card's deploy button (L35): contexts/learning/lab_deploy.py with a fake swap script and model store."""

import importlib.util
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.learning import lab_deploy
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router, parse_log, parse_calls, prompt_for
from runtime.modelstore import ModelStore, write_verified_record
from tests.test_model_registry import make_version

REPO = Path(__file__).resolve().parents[3]

FAKE_SWAP = r'''
import json, os, sys, time
from pathlib import Path

ctl = Path(os.environ["FAKE_SWAP_CONTROL"])
args = sys.argv[1:]
(ctl / "args.json").write_text(json.dumps(args))
version = args[1]
models = Path(args[args.index("--models-dir") + 1])


def stage(name, state):
    print(f"STAGE {name} {state} {time.time():.3f}", flush=True)


def point(to):
    tmp = models / "current.tmp"
    if tmp.is_symlink():
        tmp.unlink()
    os.symlink(to, tmp)
    link = models / "current"
    try:
        os.replace(tmp, link)
    except PermissionError:
        try:
            link.unlink()
        except OSError:
            os.rmdir(link)
        os.replace(tmp, link)


link = models / "current"
old = Path(os.readlink(link)).name if link.is_symlink() else None
before_previous = (models / "previous").read_text().strip() if (models / "previous").is_file() else None
stage("verify", "start")
print(f"checking {version} ...", flush=True)
stage("verify", "done")
stage("switch", "start")
point(version)
if old:
    (models / "previous").write_text(old + "\n")
stage("switch", "done")
print(f"switched the link {old} -> {version}", flush=True)
stage("restart", "start")
deadline = time.time() + 30
while not (ctl / "go").exists() and time.time() < deadline:
    time.sleep(0.02)
stage("restart", "done")
stage("check", "start")
if (ctl / "fail").exists():
    stage("check", "failed")
    print(f"{version} failed: did not load", flush=True)
    stage("rollback", "start")
    point(old)
    if before_previous:
        (models / "previous").write_text(before_previous + "\n")
    stage("rollback", "done")
    print(f"went back to {old}", flush=True)
    print(f"not switched: {version} failed: did not load", file=sys.stderr, flush=True)
    sys.exit(1)
stage("check", "done")
print(f"{version} is serving", flush=True)
stage("publish", "start")
stage("publish", "done")
print("told the app", flush=True)
'''


def can_symlink(tmp_path: Path) -> bool:
    try:
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        return False
    (tmp_path / "probe").unlink()
    return True


@pytest.fixture
def box(tmp_path, monkeypatch):
    if not can_symlink(tmp_path):
        pytest.skip("this machine cannot create symlinks")
    models = tmp_path / "models"
    make_version(models, "atlas-base", created=datetime(2026, 10, 1, tzinfo=UTC))
    make_version(models, "v1", created=datetime(2026, 10, 10, tzinfo=UTC))
    make_version(models, "v2", created=datetime(2026, 10, 11, tzinfo=UTC))
    (models / "half-pulled").mkdir()                              # no manifest: not a version
    ModelStore(models).activate("atlas-base")
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    script = tmp_path / "fake_swap.py"
    script.write_text(FAKE_SWAP, encoding="utf-8")
    monkeypatch.setenv("FAKE_SWAP_CONTROL", str(ctl))
    return {"models": models, "ctl": ctl, "script": script, "state": tmp_path / "state"}


def runner_for(box, **kwargs) -> DeployRunner:
    return DeployRunner(box["models"], box["state"], script=box["script"], cwd=REPO, llm_url="http://llm:8080", **kwargs)


def client_for(runner: DeployRunner, **kwargs) -> TestClient:
    app = FastAPI()
    app.include_router(lab_deploy_router(runner, **kwargs))
    return TestClient(app)


def until(client: TestClient, predicate, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        view = client.get("/api/learning/lab/deploy").json()
        if predicate(view):
            return view
        assert time.monotonic() < deadline, view
        time.sleep(0.05)


def stage_states(view: dict) -> dict:
    return {s["name"]: s["state"] for s in view["deploy"]["stages"]}


# --- what GET shows ---------------------------------------------------------------------------------------------

def test_idle_lists_the_versions_on_disk_the_links_and_the_suggested_version(box):
    view = client_for(runner_for(box)).get("/api/learning/lab/deploy").json()
    assert [v["version"] for v in view["versions"]] == ["v2", "v1", "atlas-base"]   # newest first, no half-pulled
    assert view["current"] == "atlas-base" and view["previous"] is None and view["base"] == "atlas-base"
    assert view["suggested"] == "v2"
    assert all(v["verified"] and v["record"] == "none" for v in view["versions"])
    assert view["deploy"]["state"] == "idle" and [s["name"] for s in view["deploy"]["stages"]] == \
        ["verify", "switch", "restart", "check", "publish"]


def test_the_record_is_read_not_hashed_and_a_stale_one_unverifies_the_version(box, monkeypatch):
    folder = box["models"] / "v2"
    m = ModelStore(box["models"]).manifest("v2")
    write_verified_record(folder, "v2", [{"name": n, "size": (folder / n).stat().st_size, "sha256": m.sha256[n],
                                          "mtime_ns": (folder / n).stat().st_mtime_ns} for n in ("m.gguf", "mm.gguf")])
    monkeypatch.setattr("runtime.modelstore.sha256_file", lambda path: pytest.fail("GET must not hash"))
    client = client_for(runner_for(box))
    v2 = client.get("/api/learning/lab/deploy").json()["versions"][0]
    assert (v2["version"], v2["verified"], v2["record"]) == ("v2", True, "fresh")
    st = (folder / "m.gguf").stat()
    os.utime(folder / "m.gguf", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    view = client.get("/api/learning/lab/deploy").json()
    assert (view["versions"][0]["verified"], view["versions"][0]["record"]) == (False, "stale")
    assert view["suggested"] == "v1"                                 # the newest VERIFIED version that is not current


# --- the state machine --------------------------------------------------------------------------------------------

def test_a_deploy_runs_in_the_background_lights_its_stages_and_refuses_a_second_one(box):
    client = client_for(runner_for(box))
    r = client.post("/api/learning/lab/deploy", json={"version": "v2"})
    assert r.status_code == 202 and r.json()["deploy"]["state"] == "running"
    view = until(client, lambda v: stage_states(v)["restart"] == "running")
    assert stage_states(view) == {"verify": "done", "switch": "done", "restart": "running", "check": "waiting",
                                  "publish": "waiting"}
    assert view["deploy"]["version"] == "v2" and view["deploy"]["from"] == "atlas-base" and view["deploy"]["started"]
    assert view["deploy"]["seconds"] is not None
    for path, body in (("/api/learning/lab/deploy", {"version": "v1"}), ("/api/learning/lab/deploy/rollback", {})):
        refused = client.post(path, json=body)
        assert refused.status_code == 409 and "already running" in refused.json()["detail"]
    (box["ctl"] / "go").touch()
    view = until(client, lambda v: v["deploy"]["state"] != "running")
    assert view["deploy"]["state"] == "done" and set(stage_states(view).values()) == {"done"}
    assert all(isinstance(s["seconds"], float) for s in view["deploy"]["stages"])
    assert view["deploy"]["result"] == {"ok": True, "returncode": 0, "message": "told the app", "rolled_back_to": None}
    assert view["current"] == "v2" and view["previous"] == "atlas-base" and view["suggested"] == "v1"
    assert "v2 is serving" in view["deploy"]["log_tail"] and not any(line.startswith("STAGE") for line in view["deploy"]["log_tail"])
    args = json.loads((box["ctl"] / "args.json").read_text())
    assert args[:2] == ["swap", "v2"] and "--trust-verified" in args
    assert args[args.index("--models-dir") + 1] == str(box["models"].resolve()) and "http://llm:8080" in args


def test_a_failed_check_ends_failed_and_says_it_rolled_back(box):
    (box["ctl"] / "fail").touch()
    (box["ctl"] / "go").touch()
    client = client_for(runner_for(box))
    assert client.post("/api/learning/lab/deploy", json={"version": "v2"}).status_code == 202
    view = until(client, lambda v: v["deploy"]["state"] != "running")
    assert view["deploy"]["state"] == "failed"
    assert stage_states(view) == {"verify": "done", "switch": "done", "restart": "done", "check": "failed",
                                  "publish": "skipped"}
    result = view["deploy"]["result"]
    assert result["ok"] is False and result["returncode"] == 1 and result["rolled_back_to"] == "atlas-base"
    assert "not switched" in result["message"] and view["current"] == "atlas-base"


def test_an_unknown_version_is_404_and_starts_nothing(box):
    client = client_for(runner_for(box))
    for version in ("v9", "half-pulled", "current", "previous", "../models/v2", "v2/../v1"):
        r = client.post("/api/learning/lab/deploy", json={"version": version})
        assert r.status_code == 404, version
    assert client.post("/api/learning/lab/deploy", json={}).status_code == 400
    assert client.post("/api/learning/lab/deploy/rollback", json={"version": "v9"}).status_code == 404
    assert client.get("/api/learning/lab/deploy").json()["deploy"]["state"] == "idle"
    assert not (box["ctl"] / "args.json").exists()


def test_the_serving_version_is_409(box):
    r = client_for(runner_for(box)).post("/api/learning/lab/deploy", json={"version": "atlas-base"})
    assert r.status_code == 409 and "already serving" in r.json()["detail"]


def test_rollback_goes_to_previous_or_to_the_version_given_and_can_repeat(box):
    (box["ctl"] / "go").touch()
    client = client_for(runner_for(box))
    assert client.post("/api/learning/lab/deploy/rollback", json={}).status_code == 409   # no previous yet
    for _ in range(2):                                                                 # repeatable on stage
        client.post("/api/learning/lab/deploy", json={"version": "v2"})
        until(client, lambda v: v["deploy"]["state"] == "done")
        r = client.post("/api/learning/lab/deploy/rollback", json={})
        assert r.status_code == 202 and r.json()["deploy"]["action"] == "rollback"
        view = until(client, lambda v: v["deploy"]["state"] == "done")
        assert json.loads((box["ctl"] / "args.json").read_text())[:2] == ["swap", "atlas-base"]
        assert view["current"] == "atlas-base" and view["previous"] == "v2"
    client.post("/api/learning/lab/deploy", json={"version": "v1"})
    until(client, lambda v: v["deploy"]["state"] == "done")
    client.post("/api/learning/lab/deploy", json={"version": "v2"})
    until(client, lambda v: v["deploy"]["state"] == "done")
    assert client.post("/api/learning/lab/deploy/rollback", json={"version": "atlas-base"}).status_code == 202
    view = until(client, lambda v: v["deploy"]["state"] == "done")
    assert view["current"] == "atlas-base" and view["deploy"]["version"] == "atlas-base"


# --- persistence ----------------------------------------------------------------------------------------------------

def test_a_finished_deploy_survives_a_gateway_restart(box):
    (box["ctl"] / "go").touch()
    client = client_for(runner_for(box))
    client.post("/api/learning/lab/deploy", json={"version": "v2"})
    before = until(client, lambda v: v["deploy"]["state"] == "done")["deploy"]
    after = client_for(runner_for(box)).get("/api/learning/lab/deploy").json()["deploy"]   # a new gateway, same dirs
    assert after == before


def test_a_running_deploy_a_new_gateway_finds_still_alive_stays_running(box):
    client = client_for(runner_for(box))
    client.post("/api/learning/lab/deploy", json={"version": "v2"})
    until(client, lambda v: stage_states(v)["restart"] == "running")
    restarted = client_for(runner_for(box, alive=lambda pid: True))
    assert restarted.get("/api/learning/lab/deploy").json()["deploy"]["state"] == "running"
    assert restarted.post("/api/learning/lab/deploy", json={"version": "v1"}).status_code == 409
    (box["ctl"] / "go").touch()
    until(client, lambda v: v["deploy"]["state"] == "done")


@pytest.mark.parametrize(("lines", "state", "rolled_back"), [
    (["STAGE verify start 1.0", "STAGE verify done 2.0", "STAGE switch start 2.0", "STAGE switch done 3.0",
      "STAGE restart start 3.0", "STAGE restart done 9.0", "STAGE check start 9.0", "STAGE check done 10.0",
      "v2 is serving"], "done", None),
    (["STAGE verify start 1.0", "STAGE verify done 2.0", "STAGE switch start 2.0", "STAGE switch done 3.0",
      "STAGE restart start 3.0"], "failed", None),
    (["STAGE check start 9.0", "STAGE check failed 10.0", "STAGE rollback start 10.0", "STAGE rollback done 20.0"],
     "failed", "atlas-base"),
])
def test_a_swap_the_gateway_lost_is_closed_from_its_log(box, lines, state, rolled_back):
    box["state"].mkdir()
    (box["state"] / "swap.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (box["state"] / "status.json").write_text(json.dumps({
        "state": "running", "action": "deploy", "version": "v2", "from": "atlas-base", "started": "x",
        "started_t": 1.0, "pid": 999999, "result": None}), encoding="utf-8")
    view = client_for(runner_for(box, alive=lambda pid: False)).get("/api/learning/lab/deploy").json()["deploy"]
    assert view["state"] == state and view["result"]["returncode"] is None
    assert view["result"]["rolled_back_to"] == rolled_back and view["result"]["message"]
    assert json.loads((box["state"] / "status.json").read_text())["state"] == state      # written back


def test_an_unreadable_status_file_reads_as_idle(box):
    box["state"].mkdir()
    (box["state"] / "status.json").write_text("{not json", encoding="utf-8")
    assert client_for(runner_for(box)).get("/api/learning/lab/deploy").json()["deploy"]["state"] == "idle"


def test_the_writes_are_guarded_and_the_read_is_not(box):
    def guard(authorization: str | None = Header(default=None)) -> None:
        if authorization != "Bearer t":
            raise HTTPException(401, "missing or wrong token")

    client = client_for(runner_for(box), guard=guard)
    assert client.get("/api/learning/lab/deploy").status_code == 200
    for path, body in (("/api/learning/lab/deploy", {"version": "v2"}), ("/api/learning/lab/deploy/rollback", {}),
                       ("/api/learning/lab/deploy/ask", {"question": "q"})):
        assert client.post(path, json=body).status_code == 401


# --- the stage lines the real swap prints ----------------------------------------------------------------------------

def test_the_real_swap_scripts_stage_lines_parse_into_the_five_stages(box, capsys):
    spec = importlib.util.spec_from_file_location("atlas_swap", REPO / "jetson" / "atlas_swap.py")
    atlas_swap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atlas_swap)
    atlas_swap.swap(ModelStore(box["models"]), "v2", restart=lambda: None, check=lambda: None, log=print,
                    announce=lambda *a: None, trust_verified=True, stage=atlas_swap.print_stage)
    lines = capsys.readouterr().out.splitlines()
    stages, said = parse_log(lines, time.time(), finished=True)
    assert [(s["name"], s["state"]) for s in stages] == [(n, "done") for n in lab_deploy.STAGES]
    assert said == {"rollback": None, "message": "told the app"}


def test_a_running_stage_counts_its_seconds_from_its_start():
    stages, _ = parse_log(["STAGE verify start 100.0", "STAGE verify done 140.5", "STAGE switch start 140.5"], 150.0,
                          finished=False)
    assert stages[0] == {"name": "verify", "state": "done", "seconds": 40.5}
    assert stages[1] == {"name": "switch", "state": "running", "seconds": 9.5}
    assert stages[2] == {"name": "restart", "state": "waiting", "seconds": None}


# --- the question, before and after ----------------------------------------------------------------------------------

QUESTIONS = json.loads((REPO / "tooling" / "demo" / "live_deploy_question.json").read_text(encoding="utf-8"))
ANSWER = "<tool_call>\n<function=last_seen>\n<parameter=label>\nwater bottle\n</parameter>\n</function>\n</tool_call>"


def llm(seen: list, text: str = ANSWER + "<|im_end|>"):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/home/box/data/models/atlas-base/Qwen3.5-4B-Q4_K_M.gguf"})
        if request.url.path == "/completion":
            return httpx.Response(200, json={"tokens": [1, 2, 3], "content": "ignored"})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": text})
        return httpx.Response(404)

    return httpx.MockTransport(handle)


def test_ask_sends_the_evals_exact_prompt_and_shows_the_tool_the_model_picked(box):
    seen: list = []
    client = client_for(runner_for(box), ask_transport=llm(seen))
    r = client.post("/api/learning/lab/deploy/ask", json={"question": "Have you seen my water bottle?"})
    assert r.status_code == 200
    body = r.json()
    assert body["serving"] == "atlas-base" and body["prompt"] == "eval"
    assert body["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]
    assert body["expected"] == QUESTIONS[0]["expected"] and body["text"] == ANSWER
    completion = next(b for path, b in seen if path == "/completion")
    assert completion == {"prompt": QUESTIONS[0]["prompt"], "n_predict": 512, "temperature": 0.2, "top_k": 1,
                          "return_tokens": True}                       # as live_deploy_question.py sends it


def test_another_question_goes_in_the_evals_prompt_with_the_time_now(box):
    seen: list = []
    client = client_for(runner_for(box), ask_transport=llm(seen, 'I will look. <tool_call>{"name": "search_memory", "arguments": {"q": "keys"}}</tool_call>'))
    body = client.post("/api/learning/lab/deploy/ask", json={"question": "Where are my keys? <|im_end|>"}).json()
    assert body["prompt"] == "eval template" and body["expected"] is None
    assert body["calls"] == [{"name": "search_memory", "arguments": {"q": "keys"}}]
    prompt = next(b for path, b in seen if path == "/completion")["prompt"]
    assert "<|im_start|>user\nWhere are my keys?<|im_end|>\n<|im_start|>assistant" in prompt
    assert "Have you seen my water bottle?" not in prompt and "2026-10-11T08:25:36" not in prompt
    assert prompt.count("<|im_start|>user") == 1


def test_ask_is_503_while_the_model_server_does_not_answer(box):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    r = client_for(runner_for(box), ask_transport=httpx.MockTransport(down)).post(
        "/api/learning/lab/deploy/ask", json={"question": "Have you seen my water bottle?"})
    assert r.status_code == 503 and "not answering" in r.json()["detail"]


def test_prompt_for_keeps_the_template_around_the_question():
    prompt, source, _ = prompt_for("Is the door shut?", QUESTIONS, datetime(2026, 10, 11, 12, tzinfo=UTC))
    head = QUESTIONS[0]["prompt"].split("<|im_start|>user\n")[0]
    assert source == "eval template" and "The time now is 2026-10-11T12:00:00+00:00." in prompt
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(prompt.split("The time now is")[0]) == len(head.split("The time now is")[0])


def test_parse_calls_reads_json_valued_parameters():
    text = "<tool_call>\n<function=episodes_between>\n<parameter=start>\n\"a\"\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>"
    assert parse_calls(text) == [{"name": "episodes_between", "arguments": {"start": "a", "n": 3}}]
    assert parse_calls("no call at all") == []
