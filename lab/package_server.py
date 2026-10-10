"""package_server: the pod's file server for model packages, with HTTP Range support and no header logging.

    python3 package_server.py [--root ~/data/packages] [--port 8300] [--host 127.0.0.1] [--log FILE]

It serves the packages under ROOT, one folder per version, so the box can pull them through the hub's proxy
(https://<hub>/user/<account>/proxy/<port>/...):

    GET /                    {"versions": ["atlas-v3", ...]}
    GET /<version>/          {"version": "atlas-v3", "files": [{"name": ..., "size": ...}], "complete": true}
    GET /<version>/<file>    the file; HEAD answers the same headers without the body

The root is the persistent home (~/data/packages), not the Jupyter root, which a relaunch wipes.

Why not `python3 -m http.server`. CPython's handler has no Range support: SimpleHTTPRequestHandler.send_head
answers every GET for a file with 200 and the whole body. jupyter-server-proxy refuses a response body over
100 MiB (Tornado's default max_body_size in its HTTP client), so a multi-GB GGUF can only cross the proxy in
Range parts. This server answers `Range: bytes=a-b`, `bytes=a-` and `bytes=-n` with 206 and a Content-Range,
honors `If-Range` against its ETag, and answers a range past the end of the file with 416. A request names one
range; two or more are refused with 416, and a range that is not bytes or not well formed with 400.

It never logs a request header. jupyter-server-proxy copies every header of the caller's request to the backend,
the hub token in `Authorization: token ...` included, so the log is one line per request (time, method, path
without its query, status, bytes sent, and the byte span answered) and nothing else the caller sent. A request
the server cannot parse is logged by its status code alone.

It listens on 127.0.0.1 unless told otherwise: the proxy on the same pod reaches it there and nothing else can.
Only plain file names are served (letters, digits, dot, dash and underscore, not starting with a dot), only from
inside the root, and never a name that ends in .part or .tmp, which is how a package script marks a file it is
still writing. Standard library only and Python 3.8 syntax, so it runs from a bare `python3` on the pod.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8300
DEFAULT_ROOT = "~/data/packages"
MANIFEST = "manifest.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
UNFINISHED_SUFFIXES = (".part", ".tmp")
CHUNK_BYTES = 1 << 20            # bytes read from disk and written to the socket at a time
CLIENT_TIMEOUT_S = 300           # a client that stays silent this long is dropped
DRAIN_BYTES = 1 << 20            # the largest unused request body that is read before a 405 is sent
LOG_PATH_CHARS = 200
JSON_TYPE = "application/json"
BINARY_TYPE = "application/octet-stream"
TEXT_TYPE = "text/plain; charset=utf-8"


class RangeError(Exception):
    """A Range header the server refuses: status 400 (not one well formed byte range) or 416 (cannot be satisfied)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def parse_range(header: str, size: int) -> tuple[int, int]:
    """(first, last) byte, both inclusive, of the one range `header` asks of a file of `size` bytes.

    A last byte past the end is cut to the end, and a suffix longer than the file is the whole file. RangeError
    (400) for text that is not a single bytes range, RangeError (416) for several ranges, for a range that starts
    at or past the end, and for any range of an empty file."""
    unit, equals, spec = header.strip().partition("=")
    if not equals or unit.strip().lower() != "bytes":
        raise RangeError(400, "only a bytes range is served")
    spec = spec.strip()
    if "," in spec:
        raise RangeError(416, "one range per request")
    first, dash, last = (part.strip() for part in spec.partition("-"))
    if not dash or (first and not re.fullmatch(r"[0-9]+", first)) or (last and not re.fullmatch(r"[0-9]+", last)) \
            or not (first or last):
        raise RangeError(400, "not a byte range")
    if not first:
        count = int(last)
        if count == 0 or size == 0:
            raise RangeError(416, "range not satisfiable")
        return max(0, size - count), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        raise RangeError(400, "the last byte is before the first")
    if start >= size:
        raise RangeError(416, "range not satisfiable")
    return start, min(end, size - 1)


def range_applies(if_range: str | None, etag: str, last_modified: str) -> bool:
    """Whether a Range header is honored: always without If-Range, and with one only when it names this file's
    ETag or Last-Modified exactly (a changed file is sent whole, as RFC 9110 asks)."""
    if if_range is None:
        return True
    value = if_range.strip()
    return value == etag or value == last_modified


def path_of(raw: str) -> str:
    """The path of a request target without its query or fragment; "?" for a target that cannot be split."""
    try:
        return urllib.parse.urlsplit(raw).path
    except ValueError:
        return "?"


def split_path(raw: str) -> list[str] | None:
    """The names in a request path: [] for the root, [version] or [version, file]; None for anything else."""
    path = urllib.parse.unquote(path_of(raw))
    if "\x00" in path or "\\" in path or not path.startswith("/"):
        return None
    parts = path[1:].split("/")
    if parts and parts[-1] == "":
        parts.pop()                      # one trailing slash is the folder's own address
    if len(parts) > 2 or not all(NAME_RE.match(part) for part in parts):
        return None
    return parts


def served_name(name: str) -> bool:
    return bool(NAME_RE.match(name)) and not name.endswith(UNFINISHED_SUFFIXES)


def resolve(root: str, version: str, name: str) -> str | None:
    """The real path of root/version/name when it is a plain file that lives inside root, else None."""
    if not served_name(name) or not NAME_RE.match(version):
        return None
    base = os.path.realpath(root)
    real = os.path.realpath(os.path.join(base, version, name))
    try:
        inside = os.path.commonpath([base, real]) == base
    except ValueError:                   # another drive
        return None
    return real if inside and os.path.isfile(real) else None


def versions_in(root: str) -> list[str]:
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(name for name in names if NAME_RE.match(name) and os.path.isdir(os.path.join(root, name)))


def files_in(root: str, version: str) -> list[dict] | None:
    """The served files of a version with their sizes; None when the version is not a folder of the root."""
    folder = os.path.join(root, version)
    if not NAME_RE.match(version) or not os.path.isdir(folder):
        return None
    found = []
    for name in sorted(os.listdir(folder)):
        path = resolve(root, version, name)
        if path is not None:
            found.append({"name": name, "size": os.path.getsize(path)})
    return found


def _printable(text: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)[:LOG_PATH_CHARS]


def _now_text() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PackageHandler(BaseHTTPRequestHandler):
    server_version = "package-server"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = CLIENT_TIMEOUT_S

    def version_string(self) -> str:
        return self.server_version

    # --- the log: the only text that leaves this class. It is built from the method, the path without its
    # query, the status and the byte counts, never from self.headers and never from the request line. ---

    def log_message(self, format: str, *args: object) -> None:
        return None

    def log_request(self, code: object = "-", size: object = "-") -> None:
        return None

    def log_error(self, format: str, *args: object) -> None:
        """Called by the standard library for a request it cannot parse; its text can carry the request line,
        so only the status code is kept."""
        code = args[0] if args and isinstance(args[0], int) else "-"
        self.server.log_line("%s error %s" % (_now_text(), code))

    def _log_access(self) -> None:
        path = _printable(path_of(self.path))
        self.server.log_line("%s %s %s %s %d%s" % (_now_text(), self.command, path, self._status, self._sent, self._span))

    # --- methods ---

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def _refuse_method(self) -> None:
        self._begin()
        try:
            self._drain_body()
            self._text(405, "only GET and HEAD are served", extra={"Allow": "GET, HEAD"}, send_body=True)
        except OSError:
            self.close_connection = True
        finally:
            self._log_access()

    def _drain_body(self) -> None:
        """Read a small request body that will not be used, so a client that sent one gets the 405 and not a reset."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return
        if 0 < length <= DRAIN_BYTES:
            self.rfile.read(length)

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse_method

    def _begin(self) -> None:
        self._status = "-"
        self._sent = 0
        self._span = ""

    def _serve(self, send_body: bool) -> None:
        self._begin()
        try:
            self._route(send_body)
        except OSError:
            self.close_connection = True       # the client went away, or the disk failed: the log line says how far it got
        finally:
            self._log_access()

    def _route(self, send_body: bool) -> None:
        names = split_path(self.path)
        root = self.server.root
        if names is None:
            self._text(404, "not found", send_body=send_body)
        elif not names:
            self._json(200, {"versions": versions_in(root)}, send_body)
        elif len(names) == 1:
            files = files_in(root, names[0])
            if files is None:
                self._text(404, "not found", send_body=send_body)
            else:
                complete = any(item["name"] == MANIFEST for item in files)
                self._json(200, {"version": names[0], "files": files, "complete": complete}, send_body)
        else:
            path = resolve(root, names[0], names[1])
            if path is None:
                self._text(404, "not found", send_body=send_body)
            else:
                self._file(path, names[1], send_body)

    # --- answers ---

    def _head(self, status: int, length: int, content_type: str, extra: dict | None = None) -> None:
        self._status = status
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True

    def _text(self, status: int, message: str, extra: dict | None = None, send_body: bool = True) -> None:
        body = (message + "\n").encode("utf-8")
        self._head(status, len(body), TEXT_TYPE, extra)
        if send_body:
            self.wfile.write(body)
            self._sent += len(body)

    def _json(self, status: int, payload: object, send_body: bool) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self._head(status, len(body), JSON_TYPE)
        if send_body:
            self.wfile.write(body)
            self._sent += len(body)

    def _file(self, path: str, name: str, send_body: bool) -> None:
        with open(path, "rb") as handle:
            info = os.fstat(handle.fileno())
            size = info.st_size
            etag = '"%x-%x"' % (size, info.st_mtime_ns)
            modified = email.utils.formatdate(info.st_mtime, usegmt=True)
            common = {"Accept-Ranges": "bytes", "ETag": etag, "Last-Modified": modified}
            content_type = JSON_TYPE if name == MANIFEST else BINARY_TYPE
            asked = self.headers.get("Range")
            span = None
            if asked is not None and range_applies(self.headers.get("If-Range"), etag, modified):
                try:
                    span = parse_range(asked, size)
                except RangeError as err:
                    extra = dict(common)
                    if err.status == 416:
                        extra["Content-Range"] = "bytes */%d" % size
                    self._text(err.status, err.message, extra, send_body)
                    return
            if span is None:
                self._head(200, size, content_type, common)
                start, length = 0, size
            else:
                start, length = span[0], span[1] - span[0] + 1
                self._span = " %d-%d/%d" % (span[0], span[1], size)
                self._head(206, length, content_type, {**common, "Content-Range": "bytes %d-%d/%d" % (span[0], span[1], size)})
            if send_body and length > 0:
                handle.seek(start)
                left = length
                while left > 0:
                    chunk = handle.read(min(CHUNK_BYTES, left))
                    if not chunk:
                        break                  # the file shrank while it was sent: the client sees a short body
                    self.wfile.write(chunk)
                    self._sent += len(chunk)
                    left -= len(chunk)


class PackageServer(ThreadingHTTPServer):
    """The HTTP server: one thread per request, the root folder, and the log."""

    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address: tuple[str, int], root: str, log: object = None) -> None:
        super().__init__(address, PackageHandler)
        self.root = root
        self.log = log
        self._log_lock = threading.Lock()

    def log_line(self, text: str) -> None:
        if self.log is None:
            return
        with self._log_lock:
            try:
                self.log.write(text + "\n")
                self.log.flush()
            except (OSError, ValueError):
                pass                           # a log that cannot be written must not stop a transfer

    def handle_error(self, request: object, client_address: object) -> None:
        """An exception that escaped a request: the default prints its traceback, which can quote request text."""
        self.log_line("%s error %s" % (_now_text(), type(sys.exc_info()[1]).__name__))


def make_server(root: str = DEFAULT_ROOT, host: str = DEFAULT_HOST, port: int = 0, log: object = None) -> PackageServer:
    """A server bound to host:port (port 0 picks a free one; server.server_address[1] says which) serving root.
    The root is created when it is missing. Nothing is served until serve_forever() runs."""
    folder = os.path.abspath(os.path.expanduser(root))
    os.makedirs(folder, exist_ok=True)
    return PackageServer((host, port), folder, log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="package_server.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=DEFAULT_ROOT, help="folder holding one sub-folder per package version")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on (default %d)" % DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on (default %s)" % DEFAULT_HOST)
    parser.add_argument("--log", help="append the log to this file (default: standard error)")
    args = parser.parse_args(argv)
    stream = open(args.log, "a", buffering=1, encoding="utf-8", errors="replace") if args.log else sys.stderr
    try:
        server = make_server(args.root, args.host, args.port, stream)
    except OSError as exc:
        print("package_server: cannot start: %s" % exc, file=sys.stderr)
        return 1
    server.log_line("%s listening on %s:%d serving %s" % (_now_text(), args.host, server.server_address[1], server.root))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
