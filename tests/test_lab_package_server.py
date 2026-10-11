"""lab.package_server: the pod's package file server, driven over a real socket.

Every test starts the real server (lab/package_server.py) on 127.0.0.1 in a thread, on a package folder of its own, and
talks to it over HTTP; nothing here is faked. The server is the pod-side half of the package pull: it must answer HTTP Range requests
(python3 -m http.server does not) and must never write a request header, the hub token that jupyter-server-proxy
forwards included, into its log."""
from __future__ import annotations

import io
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from lab import package_server

REPO = Path(__file__).resolve().parents[1]
DATA = bytes(range(256)) * 41                       # 10496 bytes holding every byte value, so a shifted range shows
SIZE = len(DATA)
SECRET = "feedface0123456789abcdef01234567"        # stands for the hub token
LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?P<method>[A-Z]+) (?P<path>\S+) (?P<status>\d{3}) (?P<sent>\d+)"
                      r"(?: (?P<span>\d+-\d+/\d+))?$")


def send_raw(port: int, request: bytes) -> bytes:
    """Send bytes as they are and return everything the server answers before it closes the connection."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        try:
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass                          # the server has already answered and closed: its answer is read below
        chunks = []
        while True:
            try:
                block = sock.recv(65536)
            except OSError:
                break
            if not block:
                break
            chunks.append(block)
    return b"".join(chunks)


class Pod:
    """A package folder, the server on it and a client."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "packages"
        folder = self.root / "v1"
        folder.mkdir(parents=True)
        (folder / "model.gguf").write_bytes(DATA)
        (folder / "manifest.json").write_bytes(b'{"version": "v1"}\n')
        (folder / "empty.bin").write_bytes(b"")
        (folder / "big.gguf.part").write_bytes(b"still being written")
        (folder / "scratch.tmp").write_bytes(b"still being written")
        (folder / ".hidden").write_bytes(b"x")
        (self.root / "v2").mkdir()
        (self.root / "v2" / "model.gguf").write_bytes(b"no manifest yet")
        (self.root / "notes.txt").write_text("a file beside the versions, not a version\n")
        self.log = io.StringIO()
        self.server = package_server.make_server(str(self.root), "127.0.0.1", 0, self.log)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", trust_env=False, timeout=30.0)

    def close(self) -> None:
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def raw(self, request: bytes) -> bytes:
        return send_raw(self.port, request)

    def status_of(self, target: str) -> int:
        answer = self.raw(f"GET {target} HTTP/1.1\r\nHost: pod\r\nConnection: close\r\n\r\n".encode("ascii"))
        return int(answer.split(b" ", 2)[1])

    def lines(self, count: int, timeout: float = 10.0) -> list[str]:
        """The log lines once there are `count` of them (a line is written just after the answer is sent)."""
        deadline = time.monotonic() + timeout
        found: list[str] = []
        while time.monotonic() < deadline:
            found = [line for line in self.log.getvalue().splitlines() if line]
            if len(found) >= count:
                return found
            time.sleep(0.01)
        raise AssertionError(f"expected {count} log lines, found {found}")


@pytest.fixture
def pod(tmp_path):
    pod = Pod(tmp_path)
    try:
        yield pod
    finally:
        pod.close()


# --- Range ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("asked, first, last", [
    ("bytes=0-0", 0, 0),
    ("bytes=0-99", 0, 99),
    ("bytes=100-199", 100, 199),
    ("bytes=10400-", 10400, SIZE - 1),
    ("bytes=-96", SIZE - 96, SIZE - 1),
    ("bytes=-99999", 0, SIZE - 1),
    ("bytes=10495-10495", SIZE - 1, SIZE - 1),
    ("bytes=10000-99999999999999999999", 10000, SIZE - 1),
    ("BYTES=5-9", 5, 9),
])
def test_a_range_is_answered_with_206_and_exactly_those_bytes(pod, asked, first, last):
    answer = pod.client.get("/v1/model.gguf", headers={"Range": asked})

    assert answer.status_code == 206
    assert answer.content == DATA[first:last + 1]
    assert answer.headers["content-range"] == f"bytes {first}-{last}/{SIZE}"
    assert answer.headers["content-length"] == str(last - first + 1)
    assert answer.headers["accept-ranges"] == "bytes"
    assert answer.headers["content-type"] == "application/octet-stream"
    assert answer.headers["etag"].startswith('"')
    assert answer.headers["last-modified"].endswith("GMT")


@pytest.mark.parametrize("asked", ["bytes=10496-", "bytes=10496-10500", "bytes=20000-30000", "bytes=-0", "bytes=0-1,5-6"])
def test_a_range_that_cannot_be_satisfied_is_416_with_the_size(pod, asked):
    answer = pod.client.get("/v1/model.gguf", headers={"Range": asked})

    assert answer.status_code == 416
    assert answer.headers["content-range"] == f"bytes */{SIZE}"
    assert len(answer.content) < 100          # a short message, never the file


@pytest.mark.parametrize("asked", ["items=0-5", "bytes=5-2", "bytes=a-b", "bytes=", "bytes=-", "bytes=0-5-7", "bytes=--5",
                                   "0-5", "bytes=+5-6"])
def test_a_range_that_is_not_one_well_formed_byte_range_is_400(pod, asked):
    answer = pod.client.get("/v1/model.gguf", headers={"Range": asked})

    assert answer.status_code == 400
    assert "content-range" not in answer.headers


def test_a_whole_file_get_is_200_with_the_full_body_and_says_it_does_ranges(pod):
    answer = pod.client.get("/v1/model.gguf")

    assert answer.status_code == 200
    assert answer.content == DATA
    assert answer.headers["content-length"] == str(SIZE)
    assert answer.headers["accept-ranges"] == "bytes"


def test_the_manifest_is_served_as_json(pod):
    answer = pod.client.get("/v1/manifest.json")

    assert answer.status_code == 200
    assert answer.headers["content-type"] == "application/json"
    assert answer.json() == {"version": "v1"}


def test_head_answers_the_headers_of_a_get_and_no_body(pod):
    whole = pod.client.head("/v1/model.gguf")
    ranged = pod.client.head("/v1/model.gguf", headers={"Range": "bytes=10-19"})

    assert (whole.status_code, whole.headers["content-length"], whole.content) == (200, str(SIZE), b"")
    assert (ranged.status_code, ranged.headers["content-length"], ranged.content) == (206, "10", b"")
    assert ranged.headers["content-range"] == f"bytes 10-19/{SIZE}"


def test_an_empty_file_is_served_whole_and_any_range_of_it_is_416(pod):
    whole = pod.client.get("/v1/empty.bin")
    ranged = pod.client.get("/v1/empty.bin", headers={"Range": "bytes=0-0"})

    assert (whole.status_code, whole.content) == (200, b"")
    assert ranged.status_code == 416
    assert ranged.headers["content-range"] == "bytes */0"


def test_if_range_with_the_current_etag_or_date_gets_the_range_and_any_other_value_gets_the_whole_file(pod):
    first = pod.client.get("/v1/model.gguf")
    etag, modified = first.headers["etag"], first.headers["last-modified"]
    ask = {"Range": "bytes=10-19"}

    by_etag = pod.client.get("/v1/model.gguf", headers={**ask, "If-Range": etag})
    by_date = pod.client.get("/v1/model.gguf", headers={**ask, "If-Range": modified})
    other_etag = pod.client.get("/v1/model.gguf", headers={**ask, "If-Range": '"changed"'})
    weak = pod.client.get("/v1/model.gguf", headers={**ask, "If-Range": "W/" + etag})
    other_date = pod.client.get("/v1/model.gguf", headers={**ask, "If-Range": "Mon, 01 Jan 2001 00:00:00 GMT"})

    assert [(a.status_code, len(a.content)) for a in (by_etag, by_date)] == [(206, 10), (206, 10)]
    assert [(a.status_code, a.content) for a in (other_etag, weak, other_date)] == [(200, DATA)] * 3


def test_two_ranges_in_flight_at_once_each_get_their_own_bytes(pod):
    big = bytes(range(256)) * 4096                    # 1 MiB
    (pod.root / "v1" / "big.gguf").write_bytes(big)
    pieces: dict[int, bytes] = {}

    def fetch(index: int) -> None:
        start = index * len(big) // 8
        end = (index + 1) * len(big) // 8 - 1
        with httpx.Client(base_url=f"http://127.0.0.1:{pod.port}", trust_env=False, timeout=30.0) as client:
            pieces[index] = client.get("/v1/big.gguf", headers={"Range": f"bytes={start}-{end}"}).content

    threads = [threading.Thread(target=fetch, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert b"".join(pieces[index] for index in range(8)) == big


# --- what is served and what is not -----------------------------------------------------------------------

def test_the_root_lists_the_versions_and_a_version_lists_its_files(pod):
    root = pod.client.get("/")
    version = pod.client.get("/v1/")
    bare = pod.client.get("/v1")
    unfinished = pod.client.get("/v2/")

    assert root.json() == {"versions": ["v1", "v2"]}
    assert version.json() == {"version": "v1", "complete": True, "files": [
        {"name": "empty.bin", "size": 0}, {"name": "manifest.json", "size": 18}, {"name": "model.gguf", "size": SIZE}]}
    assert bare.json() == version.json()
    assert unfinished.json() == {"version": "v2", "complete": False, "files": [{"name": "model.gguf", "size": 15}]}


@pytest.mark.parametrize("target", [
    "/v3/", "/v3/model.gguf", "/v1/missing.gguf", "/v1/big.gguf.part", "/v1/scratch.tmp", "/v1/.hidden",
    "/v1/../notes.txt", "/v1/%2e%2e/notes.txt", "/v1/..%2fnotes.txt", "/..%2fnotes.txt", "/%2e%2e/packages/v1/model.gguf",
    "/v1%5cmodel.gguf", "/v1/model.gguf%00", "/v1//model.gguf", "/v1/model.gguf/extra",
    "/notes.txt", "/v1/" + "a" * 129,
])
def test_a_path_that_is_not_a_served_file_or_version_is_404(pod, target):
    assert pod.status_of(target) == 404


def test_a_link_that_leads_out_of_the_root_is_not_served(pod):
    outside = pod.tmp / "outside"
    outside.mkdir()
    (outside / "secret.gguf").write_bytes(b"outside the root")
    try:
        (pod.root / "v1" / "leak.gguf").symlink_to(outside / "secret.gguf")
    except OSError:     # Windows without Developer Mode or admin rights cannot make one (WinError 1314); pods can
        pytest.skip("this machine cannot create symlinks")
    (pod.root / "escape").symlink_to(outside, target_is_directory=True)
    (pod.root / "v1" / "inside.gguf").symlink_to(pod.root / "v1" / "model.gguf")

    assert pod.status_of("/v1/leak.gguf") == 404
    assert pod.status_of("/escape/secret.gguf") == 404
    assert pod.client.get("/escape/").json()["files"] == []
    assert pod.client.get("/v1/inside.gguf").content == DATA


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_only_get_and_head_are_served(pod, method):
    answer = pod.client.request(method, "/v1/model.gguf", content=b"x")

    assert answer.status_code == 405
    assert answer.headers["allow"] == "GET, HEAD"


def test_the_root_is_created_when_it_is_missing(tmp_path):
    server = package_server.make_server(str(tmp_path / "new" / "packages"), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        answer = httpx.get(f"http://127.0.0.1:{server.server_address[1]}/", trust_env=False, timeout=30.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    assert (tmp_path / "new" / "packages").is_dir()
    assert answer.json() == {"versions": []}


# --- the parser on its own --------------------------------------------------------------------------------

@pytest.mark.parametrize("header, size, expected", [
    ("bytes=0-0", 1, (0, 0)),
    ("bytes=0-", 5, (0, 4)),
    ("bytes=2-99", 5, (2, 4)),
    ("bytes=-2", 5, (3, 4)),
    ("bytes=-9", 5, (0, 4)),
    (" bytes=1-2 ", 5, (1, 2)),
])
def test_parse_range_gives_first_and_last_byte_inclusive(header, size, expected):
    assert package_server.parse_range(header, size) == expected


@pytest.mark.parametrize("header, size, status", [
    ("bytes=5-", 5, 416), ("bytes=0-0", 0, 416), ("bytes=-1", 0, 416), ("bytes=-0", 5, 416), ("bytes=0-1,3-4", 5, 416),
    ("bytes=3-1", 5, 400), ("bytes=x-1", 5, 400), ("pages=1-2", 5, 400), ("bytes", 5, 400), ("bytes=-", 5, 400),
])
def test_parse_range_refuses_what_it_cannot_serve_with_the_status_to_send(header, size, status):
    with pytest.raises(package_server.RangeError) as caught:
        package_server.parse_range(header, size)
    assert caught.value.status == status


# --- the log: no request header, no query ----------------------------------------------------------------

def test_the_log_is_one_line_per_request_and_holds_nothing_the_caller_sent_but_the_path(pod):
    headers = {"Authorization": f"token {SECRET}", "Cookie": f"_xsrf={SECRET}", "X-Forwarded-For": "203.0.113.9",
               "User-Agent": f"Mozilla/5.0 {SECRET}", "Referer": f"https://hub.example/{SECRET}", "X-Api-Key": SECRET,
               "Range": "bytes=0-3"}
    pod.client.get(f"/v1/model.gguf?token={SECRET}&other={SECRET}", headers=headers)
    pod.lines(1)                                  # a line is written just after its answer: keep the order
    pod.client.get("/v1/missing.gguf", headers=headers)
    pod.lines(2)
    pod.client.head("/v1/model.gguf", headers=headers)
    pod.lines(3)
    pod.client.post("/v1/model.gguf", headers=headers, content=SECRET.encode())

    lines = pod.lines(4)
    parsed = [LOG_LINE.match(line) for line in lines]

    assert all(parsed), lines
    assert [(m["method"], m["path"], m["status"], m["sent"], m["span"]) for m in parsed] == [
        ("GET", "/v1/model.gguf", "206", "4", "0-3/10496"),
        ("GET", "/v1/missing.gguf", "404", "10", None),
        ("HEAD", "/v1/model.gguf", "206", "0", "0-3/10496"),
        ("POST", "/v1/model.gguf", "405", "29", None)]
    text = pod.log.getvalue().lower()
    for leaked in (SECRET, "authorization", "cookie", "x-forwarded", "mozilla", "referer", "x-api-key", "203.0.113", "token", "range"):
        assert leaked not in text


@pytest.mark.parametrize("request_bytes, expected_line", [
    (f"GET /v1/model.gguf?token={SECRET} HTTP/1.1 garbage\r\nAuthorization: token {SECRET}\r\n\r\n".encode(), "error 400"),
    (f"GET /v1/model.gguf?token={SECRET} HTTP/9.9\r\nAuthorization: token {SECRET}\r\n\r\n".encode(), "error 505"),
    (f"BREW /v1/model.gguf?token={SECRET} HTTP/1.1\r\nAuthorization: token {SECRET}\r\n\r\n".encode(), "error 501"),
    (b"GET /" + b"a" * 70000 + f"?token={SECRET} HTTP/1.1\r\n\r\n".encode(), "error 414"),
    (f"GET /v1/model.gguf HTTP/1.1\r\nAuthorization: token {SECRET}\r\n".encode() + b"X-Pad: 1\r\n" * 150 + b"\r\n", "error 431"),
    (f"GET http://[::1/v1/model.gguf?token={SECRET} HTTP/1.1\r\nAuthorization: token {SECRET}\r\n\r\n".encode(), "GET ? 404 10"),
], ids=["garbage-after-the-version", "unknown-http-version", "unknown-method", "request-line-too-long", "too-many-headers",
        "target-that-cannot-be-split"])
def test_a_request_the_server_cannot_parse_is_logged_by_its_status_alone(pod, capsys, request_bytes, expected_line):
    pod.raw(request_bytes)
    pod.raw(b"GET / HTTP/1.1\r\nHost: pod\r\nConnection: close\r\n\r\n")        # a sentinel: the log has moved past the bad one

    lines = pod.lines(2)

    assert capsys.readouterr().err == ""              # the standard library's own logging goes to stderr: it must be silent

    assert lines[0].split(" ", 1)[1] == expected_line
    assert lines[1].split(" ", 1)[1] == "GET / 200 27"
    text = "\n".join(lines).lower()
    for leaked in (SECRET, "authorization", "garbage", "token", "9.9", "brew", "aaaa", "x-pad"):
        assert leaked not in text


def test_a_client_that_leaves_in_the_middle_of_a_body_does_not_stop_the_server(pod):
    (pod.root / "v1" / "huge.gguf").write_bytes(bytes(32 * 1024 * 1024))
    with socket.create_connection(("127.0.0.1", pod.port), timeout=10) as sock:
        sock.sendall(b"GET /v1/huge.gguf HTTP/1.1\r\nHost: pod\r\nConnection: close\r\n\r\n")
        assert sock.recv(1024).startswith(b"HTTP/1.1 200")
    again = pod.client.get("/v1/model.gguf", headers={"Range": "bytes=0-9"})

    assert (again.status_code, again.content) == (206, DATA[:10])
    cut = [LOG_LINE.match(line) for line in pod.lines(2)]
    huge = next(m for m in cut if m and m["path"] == "/v1/huge.gguf")
    assert int(huge["sent"]) < 32 * 1024 * 1024


# --- run as the pod runs it --------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_it_runs_from_a_bare_python_with_nothing_else_of_the_repo(tmp_path):
    root = tmp_path / "pod-home" / "data" / "packages"
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "model.gguf").write_bytes(DATA)
    log_file = tmp_path / "package_server.log"
    port = _free_port()
    # -I: no script folder, no user site and no PYTHON* variables on the path, so a stray `import lab` would fail
    process = subprocess.Popen([sys.executable, "-I", str(REPO / "lab" / "package_server.py"), "--root", str(root),
                                "--port", str(port), "--log", str(log_file)], cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        answer = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            try:
                answer = httpx.get(f"http://127.0.0.1:{port}/v1/model.gguf", headers={"Range": "bytes=256-511",
                                   "Authorization": f"token {SECRET}"}, trust_env=False, timeout=5.0)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        if answer is not None:
            send_raw(port, f"GET /v1/model.gguf?token={SECRET} HTTP/1.1 garbage\r\nAuthorization: token {SECRET}\r\n\r\n".encode())
            send_raw(port, b"GET / HTTP/1.1\r\nHost: pod\r\nConnection: close\r\n\r\n")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and log_file.read_text(encoding="utf-8").count("\n") < 4:
                time.sleep(0.05)
    finally:
        process.terminate()
        _, errors = process.communicate(timeout=30)

    assert answer is not None, errors.decode(errors="replace")
    assert (answer.status_code, answer.content) == (206, DATA[256:512])
    written = log_file.read_text(encoding="utf-8")
    lines = written.splitlines()
    assert f"listening on 127.0.0.1:{port} serving" in lines[0]
    assert [line.split(" ", 1)[1] for line in lines[1:]] == ["GET /v1/model.gguf 206 256 256-511/10496", "error 400",
                                                              "GET / 200 21"]
    assert SECRET not in written + errors.decode(errors="replace")
    assert errors == b""                              # nothing went to stderr: the log file holds everything the server says


def test_main_refuses_to_start_where_it_cannot_bind(tmp_path, capsys):
    code = package_server.main(["--root", str(tmp_path / "packages"), "--host", "203.0.113.77", "--port", "0"])

    assert code == 1
    assert "package_server: cannot start" in capsys.readouterr().err
