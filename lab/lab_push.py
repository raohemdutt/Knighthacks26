"""Push a training run's files to the box's Learning lab: `python -m lab lab-push`.

    python -m lab lab-push --box http://<box>:8000 --account D --run-dir adapters/l16-<run_id> --follow
    python -m lab lab-push --box http://<box>:8000 --run-dir runs/<run_id>
    python -m lab lab-push --box http://<box>:8000 --eval lab/state/eval-gate/<run_id>/scores.json --run-id <run_id>
    python -m lab lab-push --box http://<box>:8000 --account D --manifest data/packages/<version>/manifest.json --run-id <run_id>
    python -m lab lab-push --box http://<box>:8000 --ledger lab/state/ledger.jsonl

lab/train.py writes one telemetry.train.step line per optimizer step to <out>/telemetry.jsonl and run.json when it
ends; nothing sends them anywhere. The box's Learning lab screen shows a run from POST /api/learning/lab/import
(contexts/learning/lab_import.py): the steps file as it is, the gate's scores.json with kind=eval, a package's
manifest.json with kind=manifest and the run's id, and the lab's ledger with kind=ledger (the Experiments tab reads
the copy the box keeps). This verb is the relay between the two.

Where the files are read. With --account the paths are on that account's pod, relative to its Jupyter root (the home
folder on the SJC pods), and each is read with one GET of /files/<path> through lab.proxy.PodLink, the same route
lab/transfer.py pulls a package's manifest by. Nothing on the pod is written, started or stopped. Every read carries a
query parameter that is never repeated: the hub's front end caches some GETs for hours, and a cached answer would hide
the lines a running trainer has added since (lab/link_speed.py FreshLink has the measurement). Without --account the
paths are files on this machine.

What is posted. --run-dir is the trainer's --out folder: its telemetry.jsonl is posted as it stands, cut back to its
last whole line (a line the trainer is still writing would refuse the whole file on the box), with the run's id taken
from --run-id or else from the lines themselves. Posting the same lines again adds nothing on the box, which knows a
step by its run and number, so a file that has grown adds only its new steps. --eval and --manifest are posted after
the steps, each once, as they are. A manifest does not name its run, so it needs a run id from --run-id or from the
steps posted in the same call. --ledger is the lab's ledger.jsonl, posted last and whole (cut back to its last whole
line, as the steps are): the box checks every row, keeps the file in place of the copy it had, and answers with the
rows, experiments and verdicts it now holds; a box older than that import answers 400 and names the kinds it knows.
run.json is read only to learn that the run has ended (and kept by --save); the box has no import for it.

--follow keeps doing that for a run that is still training: every --every seconds (60) it looks for run.json, then
reads and posts the steps, and it stops after the first post that follows run.json, which holds the whole file. It
also stops at --max-minutes (120), and at once when the answer will not change by waiting: the box asks for a token
(401), has no such route (404, 405: it runs code older than the import), refuses the file (400, 413), the hub refuses
the account, or the pod has stopped. A box that does not answer or answers 5xx, and a pod read that times out or meets
a gateway error, are tried again on the next round.

--save DIR keeps a copy of every file read, under its own name; a steps file never replaces a longer copy.

--dry-run sends nothing to the pod or the box: it names every read and every post the same call would make, and for
files on this machine their sizes.

The box's address is --box, else $ATLAS_URL. A box with ATLAS_TOKEN set wants it on writes: it is read from the
process environment, then from the .env files lab/machine/ready.py searches, sent as `Authorization: Bearer`, and
never printed (the answer says only whether one was sent). The hub keys of --account are read the same way.

One JSON object on stdout, one progress line per round on stderr (--quiet drops them). Exit 0 when everything asked
for was posted, 1 when not, 2 for bad arguments.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

import httpx

from lab.hub import MissingCredentials, load_accounts
from lab.machine import ready
from lab.proxy import NotFound, PodLink, ProxyError
from lab.transfer import PodFiles, UnsafePath, clean_remote

__all__ = ["BOX_ENV", "TOKEN_ENV", "IMPORT_PATH", "BOX_MAX_BYTES", "Box", "BoxAnswer", "LocalSource", "PodSource", "Relay",
           "complete_lines", "run_ids_in", "build_parser", "main"]

BOX_ENV = "ATLAS_URL"                 # the app's address, as jetson/atlas_swap.py reads it
TOKEN_ENV = "ATLAS_TOKEN"             # the gateway's write token (contexts/gateway/bootstrap.py)
IMPORT_PATH = "/api/learning/lab/import"
BOX_MAX_BYTES = 8 * 1024 * 1024       # the box's cap on one imported file (contexts/learning/lab_import.py MAX_BYTES)
TELEMETRY_FILE = "telemetry.jsonl"
RUN_FILE = "run.json"
DEFAULT_EVERY_S = 60.0
MIN_EVERY_S = 5.0                     # a pod with a run on its card is read gently
DEFAULT_MAX_MINUTES = 120.0
MAX_MAX_MINUTES = 24 * 60.0
BOX_CONNECT_TIMEOUT_S = 10.0
BOX_READ_TIMEOUT_S = 30.0
POD_READ_TIMEOUT_S = 60.0
ANSWER_CHARS = 2000                   # how much of a box answer that is not JSON, or is very long, is kept
MIN_HIDDEN_CHARS = 4
# Box answers that waiting will not change, and what each is called.
FINAL_ANSWERS = {400: "box_refused_file", 401: "box_wants_token", 403: "box_wants_token", 404: "route_missing",
                 405: "route_missing", 413: "too_large"}
# Pod answers that waiting will not change (lab/proxy.py's kinds): the hub refuses the account, or the pod is gone.
FINAL_POD_KINDS = frozenset({"bad_token", "unauthorized", "blocked", "pod_stopped"})
CONTENT_TYPES = {"telemetry": "application/x-ndjson", "eval": "application/json", "manifest": "application/json",
                 "ledger": "application/x-ndjson"}
LINE_KINDS = ("ledger",)              # JSON-lines files that may be read while a line is being appended


# --- the files ---------------------------------------------------------------------------------------

def complete_lines(data: bytes) -> bytes:
    """A JSON-lines file cut back to its last whole line. What follows the last newline is kept only when it is a whole
    JSON object whose newline has not been written yet; a line the writer is in the middle of is left for the next read."""
    head, newline, tail = data.rpartition(b"\n")
    body = head + newline
    if tail.strip():
        try:
            whole = isinstance(json.loads(tail), dict)
        except ValueError:
            whole = False
        if whole:
            body += tail + b"\n"
    return body


def run_ids_in(lines: bytes) -> list[str]:
    """The run ids the lines of a steps file carry, in the order they first appear."""
    found: list[str] = []
    for line in lines.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        run_id = row.get("run_id") if isinstance(row, dict) else None
        if isinstance(run_id, str) and run_id and run_id not in found:
            found.append(run_id)
    return found


class Source(Protocol):
    """Where a run's files are read from."""

    label: str

    def join(self, folder: str, name: str) -> str: ...
    def read(self, path: str) -> bytes | None: ...
    def where(self, path: str) -> str: ...
    def close(self) -> None: ...


class LocalSource:
    """Files on this machine."""

    label = "local"

    def join(self, folder: str, name: str) -> str:
        return str(Path(folder).expanduser() / name)

    def read(self, path: str) -> bytes | None:
        """The file's bytes, or None when there is no such file."""
        try:
            return Path(path).expanduser().read_bytes()
        except FileNotFoundError:
            return None

    def where(self, path: str) -> str:
        return f"read {Path(path).expanduser()}"

    def close(self) -> None:
        return None


class PodSource:
    """Files in a pod's Jupyter root, read through the hub and never written."""

    _tick = itertools.count(1)

    def __init__(self, link: PodLink, *, timeout_s: float = POD_READ_TIMEOUT_S) -> None:
        self.link = link
        self.label = f"pod {link.letter}"
        self.timeout_s = timeout_s
        self._salt = secrets.token_hex(3)

    def join(self, folder: str, name: str) -> str:
        return f"{folder.rstrip('/')}/{name}"

    def read(self, path: str) -> bytes | None:
        """The file's bytes, or None when the pod has no such file. One GET; the query is never the same twice."""
        rel = PodFiles.read_rel(clean_remote(path), "files", None)
        try:
            reply = self.link.request("GET", rel, accept="*/*", params={"_": f"{self._salt}{next(PodSource._tick)}"},
                                      timeout=self.timeout_s)
        except NotFound:
            return None
        return reply.content

    def where(self, path: str) -> str:
        return f"{self.label}: GET /files/{clean_remote(path)}"

    def close(self) -> None:
        self.link.close()


# --- the box -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxAnswer:
    """What the box said to one post. status 0: no answer arrived."""

    status: int
    answer: Any

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def final(self) -> str | None:
        """What to call an answer that the next round would only get again."""
        return FINAL_ANSWERS.get(self.status)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "answer": self.answer}

    def brief(self) -> str:
        if self.ok and isinstance(self.answer, Mapping) and "published" in self.answer:
            return f"box 200, {self.answer.get('published')} new, {self.answer.get('already_there')} already there"
        if self.ok and isinstance(self.answer, Mapping) and "rows" in self.answer:
            return (f"box 200, {self.answer.get('rows')} rows, {self.answer.get('experiments')} experiments, "
                    f"{'replaced the copy there' if self.answer.get('changed') else 'the same copy was already there'}")
        return f"box {self.status or 'did not answer'}: {_clip(json.dumps(self.answer), 200)}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class Box:
    """The box's import route. The token rides as a Bearer header and is cut out of anything handed back."""

    def __init__(self, url: str, token: str | None = None, *, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url
        self._token = token or None
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        self._client = httpx.Client(base_url=url, transport=transport, follow_redirects=False, headers=headers,
                                    timeout=httpx.Timeout(BOX_READ_TIMEOUT_S, connect=BOX_CONNECT_TIMEOUT_S))

    @property
    def sends_token(self) -> bool:
        return self._token is not None

    def close(self) -> None:
        self._client.close()

    def _scrub(self, text: str) -> str:
        if self._token and len(self._token) >= MIN_HIDDEN_CHARS:
            text = text.replace(self._token, "<token>")
        return text

    @staticmethod
    def query(kind: str, run_id: str | None) -> dict[str, str]:
        return {"kind": kind, **({"run_id": run_id} if run_id else {})}

    def address(self, kind: str, run_id: str | None) -> str:
        return f"{self.url}{IMPORT_PATH}?{httpx.QueryParams(self.query(kind, run_id))}"

    def post(self, kind: str, body: bytes, run_id: str | None) -> BoxAnswer:
        """One post of one file. Never raises for what the box or the network does: that is the answer."""
        try:
            response = self._client.post(IMPORT_PATH, params=self.query(kind, run_id), content=body,
                                         headers={"Content-Type": CONTENT_TYPES[kind]})
        except httpx.HTTPError as exc:
            return BoxAnswer(0, self._scrub(_clip(f"{type(exc).__name__}: {exc}", ANSWER_CHARS)))
        try:
            answer: Any = response.json()
        except ValueError:
            answer = _clip(response.text, ANSWER_CHARS)
        if len(json.dumps(answer)) > ANSWER_CHARS:
            answer = _clip(json.dumps(answer), ANSWER_CHARS)
        if isinstance(answer, str):
            answer = self._scrub(answer)
        return BoxAnswer(response.status_code, answer)


def box_url(text: str | None) -> str | None:
    """The box's address without a trailing slash, or None when it is not a plain http(s)://host[:port] address."""
    if not text:
        return None
    try:
        url = httpx.URL(text.strip())
    except httpx.InvalidURL:
        return None
    if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
        return None
    if url.path not in ("", "/"):
        return None
    return str(url).rstrip("/")


# --- the relay ---------------------------------------------------------------------------------------

@dataclass
class Relay:
    """One source, one box, and what has been read and posted so far."""

    source: Source
    box: Box
    run_id: str | None = None
    save_dir: Path | None = None
    saved: dict[str, int] = field(default_factory=dict)

    def keep(self, path: str, data: bytes) -> None:
        """--save: a copy under the file's own name. A steps file only grows, so a shorter read never replaces a longer copy."""
        if self.save_dir is None:
            return
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        target = self.save_dir / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if name == TELEMETRY_FILE and target.is_file() and target.stat().st_size > len(data):
            return
        part = target.with_name(name + ".part")
        part.write_bytes(data)
        os.replace(part, target)
        self.saved[name] = len(data)

    def steps(self, path: str) -> dict[str, Any]:
        """Read the steps file and post its whole lines. `box` is in the answer only when a post was made."""
        data = self.source.read(path)
        if data is None:
            return {"state": "absent"}
        self.keep(path, data)
        body = complete_lines(data)
        lines = body.count(b"\n")
        if not lines:
            return {"state": "empty"}
        if len(body) > BOX_MAX_BYTES:
            return {"state": "too_large", "bytes": len(body), "lines": lines}
        if self.run_id is None:
            ids = run_ids_in(body)
            self.run_id = ids[0] if len(ids) == 1 else None
        return {"bytes": len(body), "lines": lines, "box": self.box.post("telemetry", body, self.run_id)}

    def one_file(self, kind: str, path: str) -> dict[str, Any]:
        """Read a scores.json, a manifest.json or a ledger and post it as it is (a ledger up to its last whole line). The
        run id rides with the first two; a ledger belongs to no run."""
        data = self.source.read(path)
        if data is None:
            return {"state": "absent"}
        self.keep(path, data)
        if kind in LINE_KINDS:
            data = complete_lines(data)
            if not data.strip():
                return {"state": "empty"}
        if len(data) > BOX_MAX_BYTES:
            return {"state": "too_large", "bytes": len(data)}
        if kind == "manifest" and not self.run_id:
            return {"state": "no_run_id", "bytes": len(data)}
        return {"bytes": len(data), "box": self.box.post(kind, data, None if kind in LINE_KINDS else self.run_id)}


def _shown(part: Mapping[str, Any]) -> dict[str, Any]:
    """A read-and-post record as the answer prints it."""
    return {key: value.as_dict() if isinstance(value, BoxAnswer) else value for key, value in part.items()}


def _verdict(part: Mapping[str, Any], *, one_shot: bool) -> str | None:
    """Why a read-and-post record ends the call, or None when it went through (or may yet, on a later round)."""
    state = part.get("state")
    if state == "too_large":
        return "too_large"
    if state == "no_run_id":
        return "no_run_id"
    box: BoxAnswer | None = part.get("box")
    if box is not None:
        if box.final:
            return box.final
        if not box.ok and one_shot:
            return "box_unreachable" if box.status == 0 else "box_error"
    return None


def follow_steps(relay: Relay, run_dir: str, *, follow: bool, every_s: float, max_s: float,
                 sleep: Callable[[float], None], clock: Callable[[], float], say: Callable[[str], None]) -> dict[str, Any]:
    """Post the run's steps: once, or (follow) round after round until the post that follows run.json, the bound, or an
    answer that waiting will not change. Returns outcome (None: posted), rounds, finished, telemetry and any error."""
    started = clock()
    steps_path, run_path = relay.source.join(run_dir, TELEMETRY_FILE), relay.source.join(run_dir, RUN_FILE)
    out: dict[str, Any] = {"outcome": None, "rounds": 0, "finished": False, "telemetry": {"state": "absent"}}
    while True:
        out["rounds"] += 1
        problem: str | None = None
        ended = False
        try:
            # run.json first: when it is there the trainer has written its last step, so the read below is the whole file
            run_json = relay.source.read(run_path)
            ended = run_json is not None
            if run_json is not None:
                relay.keep(run_path, run_json)
            part = relay.steps(steps_path)
            out["telemetry"] = part
        except ProxyError as exc:
            if not follow or exc.kind in FINAL_POD_KINDS:
                return {**out, "outcome": "pod_error", "kind": exc.kind, "error": str(exc)}
            part, problem = {}, f"{exc.kind}: {exc}"
        except OSError as exc:
            return {**out, "outcome": "read_error", "error": f"{type(exc).__name__}: {exc}"}
        out["finished"] = out["finished"] or ended
        box: BoxAnswer | None = part.get("box")
        if problem is not None:
            say(f"round {out['rounds']}: the pod could not be read ({problem})")
        elif box is not None:
            say(f"round {out['rounds']}: {part['lines']} lines, {box.brief()}" + (", run.json is there" if ended else ""))
        else:
            say(f"round {out['rounds']}: {TELEMETRY_FILE} is {part.get('state')}" + (", run.json is there" if ended else ""))
        verdict = _verdict(part, one_shot=not follow)
        if verdict is not None:
            return {**out, "outcome": verdict}
        if box is not None and box.ok and (ended or not follow):
            return out
        if box is None and problem is None and (ended or not follow):
            return {**out, "outcome": "no_telemetry"}        # a run that ended, or a single look, with no step to post
        if clock() - started + every_s > max_s:
            return {**out, "outcome": "bound_reached"}
        sleep(every_s)


# --- the command line --------------------------------------------------------------------------------

def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from None
    if not MIN_EVERY_S <= value <= 3600:
        raise argparse.ArgumentTypeError(f"--every is {MIN_EVERY_S:g} to 3600 seconds")
    return value


def _minutes(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of minutes") from None
    if not 0 < value <= MAX_MAX_MINUTES:
        raise argparse.ArgumentTypeError(f"--max-minutes is above 0 and at most {MAX_MAX_MINUTES:g}")
    return value


def _letter(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z]", text):
        raise argparse.ArgumentTypeError("--account is one letter, e.g. D")
    return text.upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab lab-push", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--box", default=None, help=f"the box's address, e.g. http://100.x.y.z:8000 (default ${BOX_ENV})")
    parser.add_argument("--account", type=_letter, default=None,
                        help="read from this account's pod through the hub (paths relative to its Jupyter root); "
                             "without it the paths are files on this machine")
    parser.add_argument("--run-dir", default=None, help=f"the trainer's --out folder: its {TELEMETRY_FILE} is posted")
    parser.add_argument("--eval", default=None, help="a gate's scores.json, posted with kind=eval")
    parser.add_argument("--manifest", default=None, help="a package's manifest.json, posted with kind=manifest (needs a run id)")
    parser.add_argument("--ledger", default=None, help="the lab's ledger.jsonl, posted whole with kind=ledger")
    parser.add_argument("--run-id", default=None, help="the run the files belong to (default: the one the steps name)")
    parser.add_argument("--follow", action="store_true", help=f"keep posting the steps until {RUN_FILE} appears")
    parser.add_argument("--every", type=_seconds, default=DEFAULT_EVERY_S, help="seconds between rounds of --follow (60)")
    parser.add_argument("--max-minutes", type=_minutes, default=DEFAULT_MAX_MINUTES, help="when --follow gives up (120)")
    parser.add_argument("--save", default=None, help="keep a copy of every file read in this folder")
    parser.add_argument("--dry-run", action="store_true", help="name every read and post; send nothing")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return parser


def _single_files(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The files posted once each after the steps, in order: (kind, path)."""
    return [(kind, path) for kind, path in (("eval", args.eval), ("manifest", args.manifest), ("ledger", args.ledger)) if path]


def _plan(args: argparse.Namespace, source: Source, box: Box, run_id: str | None) -> dict[str, Any]:
    """--dry-run: every read and post the call would make. Files on this machine are looked at; nothing is sent."""
    reads: list[str] = []
    posts: list[str] = []
    local: dict[str, Any] = {}
    is_local = isinstance(source, LocalSource)
    if args.run_dir:
        for name in (RUN_FILE, TELEMETRY_FILE):
            path = source.join(args.run_dir, name)
            reads.append(source.where(path))
            if is_local:
                data = source.read(path)
                if data is None:
                    local[name] = None
                elif name == TELEMETRY_FILE:
                    body = complete_lines(data)
                    local[name] = {"bytes": len(body), "lines": body.count(b"\n")}
                    ids = run_ids_in(body)
                    run_id = run_id or (ids[0] if len(ids) == 1 else None)
                else:
                    local[name] = {"bytes": len(data)}
        posts.append(f"POST {box.address('telemetry', run_id)}")
    for kind, path in _single_files(args):
        reads.append(source.where(path))
        if is_local:
            data = source.read(path)
            local[Path(path).name] = None if data is None else {"bytes": len(data)}
        posts.append(f"POST {box.address(kind, None if kind in LINE_KINDS else run_id)}")
    plan: dict[str, Any] = {"ok": True, "outcome": "dry_run", "box": box.url, "source": source.label, "run_id": run_id,
                            "would_read": reads, "would_post": posts, "token": "sent" if box.sends_token else "none"}
    if is_local:
        plan["local"] = local
    if args.follow:
        plan["follow"] = {"every_s": args.every, "max_minutes": args.max_minutes,
                          "until": f"{RUN_FILE} appears in the run folder"}
    if run_id is None and (args.run_dir or args.eval or args.manifest):
        plan["note"] = ("no run id is known before the steps are read: it is taken from them then; "
                        "a manifest posted without steps needs --run-id")
    return plan


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_files: Sequence[Path] | None = None, transport: httpx.BaseTransport | None = None,
         box_transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic, stderr: TextIO | None = None) -> dict[str, Any]:
    """Post a training run's steps, a gate's scores, a package's manifest and the lab's ledger to the box's Learning lab."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not (args.run_dir or args.eval or args.manifest or args.ledger):
        parser.error("nothing to push: give --run-dir, --eval, --manifest or --ledger")
    if args.follow and not args.run_dir:
        parser.error("--follow needs --run-dir")
    environ = os.environ if environ is None else environ
    files = ready.default_env_files(environ) if env_files is None else list(env_files)
    found, _ = ready.collect_env([TOKEN_ENV, BOX_ENV], environ, files)
    url = box_url(args.box or found.get(BOX_ENV))
    if url is None:
        return {"ok": False, "outcome": "no_box",
                "error": f"the box's address is --box or ${BOX_ENV}, as http://host:port with nothing after it"}
    err = stderr if stderr is not None else sys.stderr
    say: Callable[[str], None] = (lambda line: None) if args.quiet else (lambda line: print(f"lab-push: {line}", file=err, flush=True))

    box = Box(url, (found.get(TOKEN_ENV) or "").strip() or None, transport=box_transport)
    source: Source = LocalSource()
    try:
        if args.account:
            try:
                for path in (args.run_dir, args.eval, args.manifest, args.ledger):
                    if path:
                        clean_remote(path)
            except UnsafePath as exc:
                return {"ok": False, "outcome": "unsafe_path",
                        "error": f"{exc}; a path on the pod is relative to its Jupyter root, e.g. adapters/l16-<run_id>"}
            try:
                account = load_accounts([args.account], environ=environ, env_files=files)[args.account]
            except MissingCredentials as exc:
                return {"ok": False, "outcome": "no_credentials", "missing": exc.missing,
                        "error": "keys missing from the environment and the .env files"}
            source = PodSource(PodLink.for_account(account, transport=transport, environ=environ, sleep=sleep, clock=clock))
        relay = Relay(source, box, run_id=args.run_id, save_dir=Path(args.save).expanduser() if args.save else None)
        if args.dry_run:
            try:
                return _plan(args, source, box, args.run_id)
            except OSError as exc:
                return {"ok": False, "outcome": "read_error", "error": f"{type(exc).__name__}: {exc}"}
        return _push(args, relay, sleep=sleep, clock=clock, say=say)
    finally:
        source.close()
        box.close()


def _push(args: argparse.Namespace, relay: Relay, *, sleep: Callable[[float], None], clock: Callable[[], float],
          say: Callable[[str], None]) -> dict[str, Any]:
    started = clock()
    result: dict[str, Any] = {"ok": False, "outcome": "pushed", "box": relay.box.url, "source": relay.source.label,
                              "run_id": relay.run_id, "token": "sent" if relay.box.sends_token else "none"}
    outcome: str | None = None
    if args.run_dir:
        steps = follow_steps(relay, args.run_dir, follow=args.follow, every_s=args.every, max_s=args.max_minutes * 60.0,
                             sleep=sleep, clock=clock, say=say)
        outcome = steps.pop("outcome")
        result.update(steps, telemetry=_shown(steps["telemetry"]))
    for kind, path in _single_files(args):
        if outcome is not None:
            break
        try:
            part = relay.one_file(kind, path)
        except ProxyError as exc:
            outcome = "pod_error"
            result.update(kind=exc.kind, error=str(exc))
            break
        except OSError as exc:
            outcome = "read_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            break
        result[kind] = _shown(part)
        box: BoxAnswer | None = part.get("box")
        say(f"{kind}: " + (box.brief() if box is not None else str(part.get("state"))))
        outcome = {"absent": "missing_file", "empty": "empty_file"}.get(part.get("state")) or _verdict(part, one_shot=True)
    result["run_id"] = relay.run_id
    if relay.save_dir is not None:
        result["saved"] = dict(sorted(relay.saved.items()))
    result["seconds"] = round(clock() - started, 3)
    if outcome is None:
        result["ok"] = True
    else:
        result["outcome"] = outcome
    return result


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok") else 1)
