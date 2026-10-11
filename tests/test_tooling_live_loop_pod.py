"""L29: the pods' side of the live loop (tooling/demo/live_loop_pod.sh), offline: the pool trim, act 2's handoff to
pod C and its fetch through a fake hub proxy, rank 1's arguments, and the start line act 2 writes. No pod is touched."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POD = ROOT / "tooling" / "demo" / "live_loop_pod.sh"


def heredoc(name: str) -> str:
    """The text of one helper the pod script writes (cat > "$1/<name>" <<'PY' ... PY)."""
    text = POD.read_text(encoding="utf-8")
    start = text.index(f'cat > "$1/{name}" <<\'PY\'\n') + len(f'cat > "$1/{name}" <<\'PY\'\n')
    return text[start:text.index("\nPY\n", start)] + "\n"


@pytest.fixture()
def split(tmp_path):
    path = tmp_path / "l29_split.py"
    path.write_text(heredoc("l29_split.py"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("l29_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rank_helper_sets_the_floor_and_runs_r65():
    text = heredoc("l29_rank.py")
    assert 'min_held_out"] = int(sys.argv[1])' in text and 'runpy.run_path("lab/r65_local_sgd.py"' in text
    compile(text, "l29_rank.py", "exec")


def test_trim_keeps_the_first_examples_and_the_count_and_drops_a_split(tmp_path, split):
    ex = tmp_path / "pool" / "ilan" / "desk-objects-ilan-smoke"
    ex.mkdir(parents=True)
    (ex / "examples.jsonl").write_text("".join(json.dumps({"id": f"e{i}"}) + "\n" for i in range(10)), encoding="utf-8")
    (ex / "manifest.json").write_text(json.dumps({"dataset_id": ex.name, "examples": 10}), encoding="utf-8")
    (ex / "split.json").write_text("{}", encoding="utf-8")
    split.trim(tmp_path / "pool", "4")
    assert [json.loads(x)["id"] for x in (ex / "examples.jsonl").read_text().splitlines()] == ["e0", "e1", "e2", "e3"]
    assert json.loads((ex / "manifest.json").read_text())["examples"] == 4 and not (ex / "split.json").exists()
    split.trim(tmp_path / "pool", "50")   # fewer than asked: left as it is
    assert len((ex / "examples.jsonl").read_text().splitlines()) == 4


def handoff(tmp_path, split):
    home_d = tmp_path / "home-d"
    data = home_d / "group-runs" / "r" / "data" / "pooled"
    (data / "images").mkdir(parents=True)
    (data / "images" / "a.jpg").write_bytes(b"\xff\xd8 blurred")
    (data / "split.json").write_text("{}", encoding="utf-8")
    rows = home_d / "group-runs" / "r" / "rows.jsonl"
    rows.write_text("".join(json.dumps({"n": i}) + "\n" for i in range(9)), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    args = ["--dataset", str(data), "--rows", str(rows), "--model", str(home_d / "models" / "Qwen3.5-4B"),
            "--out", str(home_d / "group-runs" / "r" / "runs" / "r"), "--run-id", "live-s-duo", "--max-steps", "8"]
    os.environ["HOME"] = str(home_d)
    split.handoff(str(run), *args)
    return run, args, home_d


def test_handoff_splits_the_rows_and_packs_pod_cs_half_with_the_dataset(tmp_path, split, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    run, args, _ = handoff(tmp_path, split)
    assert [json.loads(x)["n"] for x in (run / "rows.rank0.jsonl").read_text().splitlines()] == [0, 2, 4, 6, 8]
    rank0 = (run / "rank0.args").read_text().splitlines()
    assert rank0[rank0.index("--rows") + 1] == str(run / "rows.rank0.jsonl") and "--max-steps" in rank0
    assert not (run / "rows.rank1.jsonl").exists()   # it lives only in the handoff
    tar_path = run / "handoff" / "handoff.tar"
    assert (run / "handoff" / "ready").read_text().strip() == hashlib.sha256(tar_path.read_bytes()).hexdigest()
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
        rows1 = tar.extractfile("rows.rank1.jsonl").read().decode().splitlines()
    assert {"dataset/pooled/images/a.jpg", "dataset/pooled/split.json", "args.json"} <= names
    assert [json.loads(x)["n"] for x in rows1] == [1, 3, 5, 7]


class Proxy(BaseHTTPRequestHandler):
    """The hub proxy in front of pod D's handoff server: /user/<user>/proxy/<port>/<name> with the account's token."""
    root: Path
    user = "d-user"
    token = "the-d-token"

    def log_message(self, *args):
        return

    def do_GET(self):
        prefix = f"/user/{self.user}/proxy/29610/"
        if self.headers.get("Authorization") != f"token {self.token}" or not self.path.startswith(prefix):
            self.send_response(403)
            self.end_headers()
            return
        path = self.root / self.path[len(prefix):]
        if not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_pod_c_fetches_the_handoff_through_the_proxy_and_gets_its_own_arguments(tmp_path, split, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    run, args, home_d = handoff(tmp_path, split)
    Proxy.root = run / "handoff"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("PROXY_HOST", f"127.0.0.1:{server.server_address[1]}")
        token = tmp_path / "tokd"
        token.write_text("the-d-token\n", encoding="utf-8")
        data = tmp_path / "home-c" / "l29" / "s" / "data"
        assert split.fetch("d-user", "29610", str(token), str(data), "5") == 0
        assert (data / "dataset" / "pooled" / "images" / "a.jpg").read_bytes() == b"\xff\xd8 blurred"
        assert not (data / "handoff.tar").exists()
        token.write_text("wrong", encoding="utf-8")
        assert split.fetch("d-user", "29610", str(token), str(tmp_path / "other"), "1") == 1   # the proxy refuses it
    finally:
        server.shutdown()
    home_c = tmp_path / "home-c"
    monkeypatch.setenv("HOME", str(home_c))
    out = home_c / "l29" / "s" / "rank1"
    import contextlib
    import io
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        split.rank1_args(str(data), str(out), "live-s-duo.pod-c", "gpu-a")
    got = printed.getvalue().splitlines()
    value = lambda name: got[got.index(name) + 1]  # noqa: E731
    assert value("--dataset") == str(data / "dataset" / "pooled") and value("--rows") == str(data / "rows.rank1.jsonl")
    assert value("--out") == str(out) and value("--run-id") == "live-s-duo.pod-c" and value("--power-node") == "gpu-a"
    assert value("--model") == str(home_c / "models" / "Qwen3.5-4B")   # pod D's home moved to pod C's
    assert value("--max-steps") == "8"


def git_bash() -> str | None:
    """bash, but on Windows only Git Bash (never System32's, which would start WSL)."""
    if os.name != "nt":
        return shutil.which("bash")
    found = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "bin" / "bash.exe"
    return str(found) if found.is_file() else None


@pytest.mark.skipif(git_bash() is None, reason="no bash here")
def test_act_two_writes_its_members_pods_and_part_on_its_start_line(tmp_path):
    shutil.copy(POD, tmp_path / "pod.sh")   # relative paths only, so any bash reads them
    home = tmp_path / "home"
    verb = home / "l10" / "src" / "lab" / "pod_b_group_finetune.sh"
    verb.parent.mkdir(parents=True)
    verb.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    run = home / "l29" / "s"
    run.mkdir(parents=True)
    cmd = ('source ./pod.sh none > /dev/null; RUN=home/l29/s; STAGES="$RUN/stages.jsonl"; '
           'DUO_PODS=2; act duo l29-duo live-s-duo 2; DUO_PODS=1; act solo l29-solo live-s-solo 1')
    done = subprocess.run([git_bash(), "-c", cmd], env={**os.environ, "HOME": "home", "PEER_IP": "100.64.0.9"},
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(x) for x in (run / "stages.jsonl").read_text(encoding="utf-8").splitlines()]
    start = [x for x in lines if x["stage"] == "start"]
    assert start[0]["members"] == 2 and start[0]["pods"] == ["pod-d", "pod-c"]
    assert start[0]["parts"] == {"pod-c": "live-s-duo.pod-c"}
    assert start[1]["members"] == 1 and start[1]["pods"] == ["pod-d"] and start[1]["parts"] == {}
    assert [x["rc"] for x in lines if x["stage"] == "done"] == [0, 0]
