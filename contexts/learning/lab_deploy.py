"""The deploy button on the Learning lab's On the box card: the swap runs on the box itself, from the web app (L35).

    app.include_router(lab_deploy_router(DeployRunner(data_dir / "models", data_dir / "lab_deploy",
                                                      llm_url=settings.llm_url), guard=authorized))
    # BEFORE lab_router: its /api/learning/lab/{run_id} would take these paths

GET  /api/learning/lab/deploy
    {"versions": [{version, base, task, created, current, previous, verified, record}],   newest first
     "current": str | null, "previous": str | null,          data/models/current and data/models/previous
     "base": "atlas-base" | null,                             when that version is on disk
     "suggested": str | null,                                 the newest verified version that is not current
     "featured": str | null,                                  what the card's "Put v2 on the box" deploys: FEATURED
                                                              (ATLAS_LAB_DEPLOY_VERSION) when it is on disk and
                                                              verified, else the newest verified version; never the
                                                              base. It may be current: the card then disables it
     "deploy": {state: idle|running|done|failed, action: deploy|rollback, version, from, started, ended, seconds,
                stages: [{name, state, seconds}],             verify, switch, restart, check, publish
                log_tail: [str], result: {ok, returncode, message, rolled_back_to} | null,
                interrupted: bool, message: str | null}}
    `interrupted` is true for a failed swap a restart cut short after it had already switched: returncode -15 or
    none, the check stage never done, and the current link is the swap's version. `state` stays failed (older cards
    read it); `message` then says the box is serving that version. Both are false and null otherwise.
    `verified` is true when the version's manifest.json reads and either no .verified.json is there or the record
    still vouches for every file the manifest names (same sha256, size and mtime); `record` says which: "fresh",
    "stale" (the record no longer matches a file: the next swap hashes it) or "none". Nothing is hashed here.
POST /api/learning/lab/deploy             {version}
    starts `jetson/atlas_swap.py swap <version> --trust-verified` in a background subprocess (202, the GET shape).
    409 while a swap runs or when the version is already serving; 404 for a version with no manifest.json on disk.
    Nothing is pulled from the network: only versions already on the box can be deployed.
POST /api/learning/lab/deploy/rollback    {} or {version}
    the same swap to `previous` (409 when there is none), or to the version given, e.g. {"version": "atlas-base"}.
POST /api/learning/lab/deploy/ask         {question}
    asks the box's model server (llama-server, not the gateway's /api/ask, which answers "have you seen my X" in code
    before the model picks a tool) the way tooling/demo/live_deploy_question.py does: the eval's exact prompt when the
    question is one of tooling/demo/live_deploy_question.json, else that prompt with the question and the time put in.
    {question, serving, text, calls: [{name, arguments}], expected | null, seconds, prompt: "eval" | "eval template",
     energy: {joules, idle_w, seconds, sampler, samples, above_idle_j, low_confidence} | null}.
    `energy` is what the answer cost on the box (G6, contexts/learning/answer_energy.py): the Jetson's power sampler
    integrated over the answer's window, gross, the box's idle beside it; null when the sampler is not running.
    Since the comparison (ask-compare) the same response also carries both answers, side by side:
    {..., "box": {answer, calls, serving, seconds, energy, prompt, expected} | {error, seconds},
          "brain": {answer, calls, serving, seconds, energy: {joules, watts_avg, seconds, pods: {C, D}} | null}
                   | {error, seconds} | null}
    The box: the eval's exact prompt for a held-out question (the tool it picks is what the eval scores), any other
    question in the normal chat prompt with the tools offered and not forced, so "how are you" gets a plain answer.
    Either way the tools it calls run against the box's own memory with the agent's tool executor (QA._run_tool,
    the one the Ask tab's tool loop uses) and the model answers from the results, at most ROUNDS rounds. The brain:
    the same question in the same chat prompt to the 122B pair (the head on pod C, served name "teacher", through the
    hub proxy with contexts/circle/brain_pair.py's keys), its tool calls run against the same memory, so both answer
    from the same facts; its joules are both pods' samplers over the request window (gross, idle beside). The two
    run in parallel, each with a SIDE_TIMEOUT_S timeout and an honest {error} when one fails.
    ATLAS_ASK_COMPARE=off leaves the brain out ("brain": null).
POST /api/learning/lab/live-chat          {question, target: "pair" | "box"}
    the same question answered as Server-Sent Events, token by token, then {answer, tokens, seconds, tok_s, ttft_s,
    energy}: the Speed and energy tab's live chat beside the power graph (contexts/learning/live_chat.py).
    503 when the model server does not answer (it does not for about a minute of every swap).

The stages come from the `STAGE <name> <start|done|failed> <unix seconds>` lines atlas_swap.py prints. The status is
kept in <state_dir>/status.json and the swap's output in <state_dir>/swap.log, so a gateway restart does not lose
the last deploy; a swap the restarted gateway finds no longer running is closed from its log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from runtime.modelstore import VERIFIED_RECORD, ModelStore, read_verified, record_vouches

log = logging.getLogger("atlas.learning.lab_deploy")

REPO = Path(__file__).resolve().parents[2]
SWAP_SCRIPT = REPO / "jetson" / "atlas_swap.py"
QUESTIONS = REPO / "tooling" / "demo" / "live_deploy_question.json"
BASE_VERSION = "atlas-base"
# The package the stage demo puts on the box (tooling/demo/LIVE_DEPLOY_RUN_OF_SHOW.md). Other v2 packages sit beside
# it on the box's disk (r01), so the newest-by-manifest rule alone could pick the wrong one.
FEATURED_VERSION = os.environ.get("ATLAS_LAB_DEPLOY_VERSION", "atlas-tool-calling-v2-a100-f2r4")
STAGES = ("verify", "switch", "restart", "check", "publish")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RESERVED = {"current", "previous", "current.tmp"}
STAGE_RE = re.compile(r"^STAGE (\w+) (start|done|failed) (\d+(?:\.\d+)?)\s*$")
TAIL_LINES = 12
DEFAULT_LLM_URL = "http://127.0.0.1:8080"
IM_END = "<|im_end|>"
USER_TURN = "<|im_start|>user\n"
ASSISTANT_TURN = "<|im_end|>\n<|im_start|>assistant"
SPECIAL_RE = re.compile(r"<\|[^|<>]*\|>")   # the chat template's own tokens never come from a typed question
TIME_RE = re.compile(r"The time now is [^ ]+?\.(?= )")


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()


def pid_alive(pid: int | None) -> bool:
    """Whether a process with this id still runs (POSIX). Elsewhere a process this gateway did not start is taken
    as gone."""
    if not pid or os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def parse_log(lines: Sequence[str], now: float, *, finished: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(the five stages, what else the log says: rollback state and the last plain line)."""
    seen: dict[str, dict[str, float | str]] = {}
    plain: list[str] = []
    for line in lines:
        m = STAGE_RE.match(line.strip())
        if not m:
            if line.strip():
                plain.append(line.rstrip())
            continue
        name, state, t = m.group(1), m.group(2), float(m.group(3))
        entry = seen.setdefault(name, {})
        if state == "start":
            entry.update(start=t, state="running")
        else:
            entry.update(end=t, state=state)
    stages = []
    for name in STAGES:
        entry = seen.get(name)
        if not entry:
            stages.append({"name": name, "state": "skipped" if finished else "waiting", "seconds": None})
            continue
        state = str(entry["state"])
        if state == "running" and finished:
            state = "failed"  # started and never ended: the swap stopped inside it
        start = entry.get("start")
        end = entry.get("end", None if state == "running" else start)
        seconds = None if start is None else round(float(end if end is not None else now) - float(start), 1)
        stages.append({"name": name, "state": state, "seconds": seconds})
    rollback = seen.get("rollback")
    return stages, {"rollback": None if not rollback else rollback.get("state"),
                    "message": plain[-1] if plain else None}


class DeployRunner:
    """Runs one swap at a time as a subprocess and keeps its state on disk."""

    def __init__(self, models_dir: str | Path, state_dir: str | Path, *, llm_url: str | None = None,
                 app_url: str | None = None, python: str = sys.executable, script: str | Path = SWAP_SCRIPT,
                 cwd: str | Path = REPO, trust_verified: bool = True, clock: Callable[[], float] = time.time,
                 alive: Callable[[int | None], bool] = pid_alive, featured: str | None = FEATURED_VERSION) -> None:
        self.models_dir = Path(models_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.store = ModelStore(self.models_dir)
        self.llm_url = llm_url
        self.app_url = app_url
        self.python = python
        self.script = Path(script)
        self.cwd = Path(cwd)
        self.trust_verified = trust_verified
        self.clock = clock
        self.alive = alive
        self.featured = featured
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    # --- files ---------------------------------------------------------------------------------------------------

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "swap.log"

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"state": "idle"}
        except (OSError, ValueError):
            return {"state": "idle"}

    def _save(self, status: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        scratch = self.status_path.with_suffix(".tmp")
        scratch.write_text(json.dumps(status, indent=1), encoding="utf-8")
        os.replace(scratch, self.status_path)

    def _log_lines(self) -> list[str]:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    # --- the model store -----------------------------------------------------------------------------------------

    def versions(self) -> list[dict[str, Any]]:
        current, previous = self.store.current(), self.store.previous()
        out = []
        for m in self.store.versions():
            folder = self.models_dir / m.version
            names = [m.gguf, *([m.mmproj] if m.mmproj else [])]
            if (folder / VERIFIED_RECORD).is_file():
                vouched = read_verified(folder)
                fresh = all(record_vouches(vouched.get(n), folder / n, m.sha256.get(n, "")) for n in names)
                record = "fresh" if fresh else "stale"
            else:
                record = "none"
            present = all((folder / n).is_file() for n in names)
            out.append({"version": m.version, "base": m.base, "task": m.task, "created": m.created.isoformat(),
                        "current": m.version == current, "previous": m.version == previous,
                        "verified": present and record != "stale", "record": record})
        return out

    def on_disk(self, version: str) -> bool:
        if not NAME_RE.match(version) or version in RESERVED:
            return False
        folder = self.models_dir / version
        return folder.is_dir() and not folder.is_symlink() and (folder / "manifest.json").is_file()

    # --- the swap ------------------------------------------------------------------------------------------------

    def _finish(self, status: dict[str, Any], returncode: int | None) -> dict[str, Any]:
        """Close a swap that is no longer running: its state, timings and result from the return code and the log."""
        lines = self._log_lines()
        ended = self.clock()
        stages, said = parse_log(lines, ended, finished=True)
        if returncode is None:  # a swap the gateway lost (it restarted meanwhile): judge by what the log says
            states = {s["name"]: s["state"] for s in stages}
            ok = states.get("check") == "done" and said["rollback"] is None
        else:
            ok = returncode == 0
        rolled_back_to = status.get("from") if said["rollback"] else None
        message = said["message"]
        if returncode is None and not ok and not message:
            message = "the gateway restarted while the swap ran; the box serves " + str(self.store.current())
        status.update(state="done" if ok else "failed", ended=_iso(ended), pid=None,
                      seconds=round(ended - float(status.get("started_t") or ended), 1),
                      result={"ok": ok, "returncode": returncode, "message": message, "rolled_back_to": rolled_back_to})
        self._save(status)
        return status

    def _refresh(self) -> dict[str, Any]:
        status = self._load()
        if status.get("state") != "running":
            return status
        proc = self._proc
        if proc is not None and proc.pid == status.get("pid"):
            rc = proc.poll()
            if rc is None:
                return status
            self._proc = None
            return self._finish(status, rc)
        if self.alive(status.get("pid")):
            return status  # a swap a previous gateway started, still going
        return self._finish(status, None)

    def _wait(self, proc: subprocess.Popen) -> None:
        proc.wait()
        with self._lock:
            self._refresh()

    def start(self, version: str | None, action: str = "deploy") -> dict[str, Any]:
        with self._lock:
            status = self._refresh()
            if status.get("state") == "running":
                raise Refused(409, f"a swap to {status.get('version')} is already running")
            current = self.store.current()
            if action == "rollback" and not version:
                version = self.store.previous()
                if not version:
                    raise Refused(409, "there is no earlier version to go back to")
            if not version or not self.on_disk(version):
                raise Refused(404, f"no model version {version!r} on this box (only versions already on disk deploy)")
            if version == current:
                raise Refused(409, f"{version} is already serving")
            cmd = [self.python, "-u", str(self.script), "swap", version, "--models-dir", str(self.models_dir)]
            if self.trust_verified:
                cmd.append("--trust-verified")
            if self.llm_url:
                cmd += ["--url", self.llm_url]
            if self.app_url:
                cmd += ["--app-url", self.app_url]
            env = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": os.pathsep.join(p for p in (str(self.cwd), os.environ.get("PYTHONPATH", "")) if p)}
            self.state_dir.mkdir(parents=True, exist_ok=True)
            started = self.clock()
            with self.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=self.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=os.name == "posix")
            self._proc = proc
            status = {"state": "running", "action": action, "version": version, "from": current,
                      "started": _iso(started), "started_t": started, "ended": None, "seconds": None,
                      "pid": proc.pid, "result": None}
            self._save(status)
        threading.Thread(target=self._wait, args=(proc,), daemon=True, name="lab-deploy-wait").start()
        return self.view()

    def deploy_state(self) -> dict[str, Any]:
        with self._lock:
            status = self._refresh()
        state = status.get("state", "idle")
        if state == "idle":
            return {"state": "idle", "action": None, "version": None, "from": None, "started": None, "ended": None,
                    "seconds": None, "stages": [{"name": n, "state": "waiting", "seconds": None} for n in STAGES],
                    "log_tail": [], "result": None, "interrupted": False, "message": None}
        lines = self._log_lines()
        now = self.clock()
        stages, _ = parse_log(lines, now, finished=state != "running")
        seconds = status.get("seconds")
        if state == "running" and status.get("started_t"):
            seconds = round(now - float(status["started_t"]), 1)
        return {"state": state, "action": status.get("action"), "version": status.get("version"),
                "from": status.get("from"), "started": status.get("started"), "ended": status.get("ended"),
                "seconds": seconds, "stages": stages,
                "log_tail": [line for line in lines if not STAGE_RE.match(line.strip())][-TAIL_LINES:],
                "result": status.get("result"), **self._interrupted(status, stages)}

    def _interrupted(self, status: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
        """A failed swap a restart cut short (SIGTERM, or lost with the gateway) after its switch: the box serves it."""
        result = status.get("result") if isinstance(status.get("result"), dict) else {}
        version = status.get("version")
        check = next((s["state"] for s in stages if s["name"] == "check"), None)
        if (status.get("state") == "failed" and result.get("returncode") in (-15, None) and check != "done"
                and version and self.store.current() == version):
            return {"interrupted": True,
                    "message": f"the last swap was interrupted by a restart; the box is serving {version}"}
        return {"interrupted": False, "message": None}

    def view(self) -> dict[str, Any]:
        versions = self.versions()
        current = self.store.current()
        candidates = [v for v in versions if v["verified"] and v["version"] != BASE_VERSION]
        suggested = next((v["version"] for v in candidates if not v["current"]), None)
        featured = next((v["version"] for v in candidates if v["version"] == self.featured), None)
        return {"versions": versions, "current": current, "previous": self.store.previous(),
                "base": BASE_VERSION if any(v["version"] == BASE_VERSION for v in versions) else None,
                "suggested": suggested, "featured": featured or (candidates[0]["version"] if candidates else None),
                "deploy": self.deploy_state()}


# --- the question, before and after -----------------------------------------------------------------------------

def load_questions(path: str | Path = QUESTIONS) -> list[dict[str, Any]]:
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [i for i in items if isinstance(i, dict) and isinstance(i.get("prompt"), str)] if isinstance(items, list) else []


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def prompt_for(question: str, items: list[dict[str, Any]], now: datetime | None = None) -> tuple[str, str, Any]:
    """(prompt, "eval" | "eval template", expected): the eval's own prompt for a question it holds, else the first
    one with this question and the time now put in its place."""
    for item in items:
        if _norm(str(item.get("question", ""))) == _norm(question):
            return item["prompt"], "eval", item.get("expected")
    if not items:
        raise Refused(503, "tooling/demo/live_deploy_question.json is not on this box")
    template = items[0]["prompt"]
    head, sep, rest = template.partition(USER_TURN)
    _, sep2, tail = rest.partition(ASSISTANT_TURN)
    if not sep or not sep2:
        raise Refused(503, "the eval prompt does not have the shape this route expects")
    stamp = (now or datetime.now(UTC)).isoformat()
    head = TIME_RE.sub(f"The time now is {stamp}.", head)
    return head + USER_TURN + question + ASSISTANT_TURN + tail, "eval template", None


CALL_RE = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
JSON_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_calls(text: str) -> list[dict[str, Any]]:
    """The tool calls in the model's text: the Qwen3.5 template form (<function=..><parameter=..>), or JSON."""
    calls = []
    for name, body in CALL_RE.findall(text):
        args: dict[str, Any] = {}
        for key, value in PARAM_RE.findall(body):
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value.strip()
        calls.append({"name": name, "arguments": args})
    if calls:
        return calls
    for raw in JSON_CALL_RE.findall(text):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            calls.append({"name": data["name"], "arguments": data.get("arguments") or {}})
    return calls


async def ask_model(question: str, llm_url: str, *, items: list[dict[str, Any]],
                    transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    prompt, source, expected = prompt_for(question, items)
    began = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=llm_url.rstrip("/"), transport=transport, timeout=120) as http:
            serving = None
            try:
                props = (await http.get("/props", timeout=5)).json()
                path = str(props.get("model_path") or "")
                serving = Path(path).parent.name if path else None
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
            r = await http.post("/completion", json={"prompt": prompt, "n_predict": 512, "temperature": 0.2,
                                                     "top_k": 1, "return_tokens": True})
            r.raise_for_status()
            body = r.json()
            tokens = body.get("tokens") or []
            if tokens:
                d = await http.post("/detokenize", json={"tokens": tokens})
                d.raise_for_status()
                text = d.json().get("content", "")
            else:
                text = body.get("content", "")
    except (httpx.HTTPError, ValueError) as exc:
        raise Refused(503, f"the model server is not answering ({type(exc).__name__}); it is down for about a "
                           "minute of every swap") from None
    text = str(text).split(IM_END)[0].strip()
    return {"question": question, "serving": serving, "text": text, "calls": parse_calls(text),
            "expected": expected, "seconds": round(time.perf_counter() - began, 1), "prompt": source}


# --- the box and the network brain, side by side (ask-compare) --------------------------------------------------

COMPARE_ENV = "ATLAS_ASK_COMPARE"
ROUNDS = 3  # model turns that may call tools, as the agent's tool loop (qa.MAX_TOOL_CALLS)
SIDE_TIMEOUT_S = 60.0
ANSWER_TOKENS = 256
BRAIN_MODEL = "teacher"
BRAIN_SERVING = "Qwen3.5-122B across pods C and D"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CALL_MARKUP_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def chat_system(now: datetime) -> str:
    return ("You are Atlas, a small home box with cameras and a memory of what they saw. "
            f"The time now is {now.isoformat(timespec='seconds')}. When the question is about what the cameras saw "
            "(where something is, what happened, when something was last seen), call a tool and answer from what it "
            "returns, in one or two short sentences with the times. When the question needs no tool, answer it "
            "directly in one or two short sentences.")


def _memory_tools() -> list[dict[str, Any]]:
    from contexts.agent.application.qa import MEMORY_TOOLS

    return MEMORY_TOOLS


def _plain(text: str) -> str:
    """The words of a model turn: no thinking block, no tool-call markup, no chat-template tokens."""
    text = CALL_MARKUP_RE.sub("", THINK_RE.sub("", str(text or "")))
    return " ".join(SPECIAL_RE.sub("", text.split(IM_END)[0]).split())


def _calls_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI tool_calls, or the same calls written as text (a server without a tool parser), as {id, name, arguments}."""
    out = []
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {}
        if fn.get("name"):
            out.append({"id": call.get("id") or f"call-{i}", "name": fn["name"], "arguments": args or {}})
    if not out:
        out = [{"id": f"call-{i}", **c} for i, c in enumerate(parse_calls(str(message.get("content") or "")))]
    return out


async def run_tool(qa: Any, name: str, arguments: dict[str, Any]) -> str:
    """One tool call against the box's memory with the agent's own executor (QA._run_tool): what it found, as text."""
    if qa is None:
        return "the box's memory is not available here"
    from contexts.agent.application.qa import Evidence
    from contracts.types import ToolCall

    ev = Evidence()
    await qa._run_tool(ToolCall(id="ask", name=name, arguments=dict(arguments or {})), ev)
    return "\n".join(ev.lines) or "nothing found"


async def chat_loop(post: Callable[[dict[str, Any]], Any], messages: list[dict[str, Any]], qa: Any,
                    rounds: int = ROUNDS, calls: list[dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """The agent's tool loop over an OpenAI chat endpoint: tools offered, not forced; each call run against the box's
    memory; at most `rounds` turns that call tools, then one turn without tools for the answer."""
    calls = [] if calls is None else calls
    tools = _memory_tools()
    for _ in range(rounds):
        message = await post({"messages": messages, "tools": tools, "tool_choice": "auto"})
        found = _calls_of(message)
        if not found:
            return _plain(message.get("content")), calls
        messages.append({"role": "assistant", "content": _plain(message.get("content")),
                         "tool_calls": [{"id": c["id"], "type": "function", "function": {
                             "name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in found]})
        for c in found:
            calls.append({"name": c["name"], "arguments": c["arguments"]})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": await run_tool(qa, c["name"], c["arguments"])})
    message = await post({"messages": messages})
    return _plain(message.get("content")), calls


def _message(reply: dict[str, Any]) -> dict[str, Any]:
    choices = reply.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("the model server sent no answer")
    return choices[0].get("message") or {}


async def ask_box(question: str, llm_url: str, *, items: list[dict[str, Any]], qa: Any, meter: Any | None = None,
                  transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """The box's answer: the eval's prompt for a held-out question, the chat prompt otherwise; tools run on memory."""
    began = time.perf_counter()
    started = datetime.now(UTC)
    _, source, expected = prompt_for(question, items)
    held_out = source == "eval"
    async with httpx.AsyncClient(base_url=llm_url.rstrip("/"), transport=transport, timeout=SIDE_TIMEOUT_S) as http:
        async def post(body: dict[str, Any]) -> dict[str, Any]:
            r = await http.post("/v1/chat/completions", json={**body, "temperature": 0.2, "max_tokens": ANSWER_TOKENS})
            r.raise_for_status()
            return _message(r.json())

        messages = [{"role": "system", "content": chat_system(datetime.now().astimezone())},
                    {"role": "user", "content": question}]
        first = None
        if held_out:  # the eval's exact prompt picks the tool, as the eval scores it; the answer continues in chat
            first = await ask_model(question, llm_url, items=items, transport=transport)
            calls: list[dict[str, Any]] = []
            if first["calls"]:
                found = [{"id": f"call-{i}", **c} for i, c in enumerate(first["calls"])]
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in found]})
                for c in found:
                    calls.append({"name": c["name"], "arguments": c["arguments"]})
                    messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": await run_tool(qa, c["name"], c["arguments"])})
                try:
                    answer, calls = await chat_loop(post, messages, qa, rounds=ROUNDS - 1, calls=calls)
                except (httpx.HTTPError, ValueError) as exc:  # the tool it picked still shows; say why no words came
                    answer = f"(picked the tool, but no answer came back: {type(exc).__name__})"
            else:
                answer = _plain(first["text"])
            serving = first["serving"]
        else:
            answer, calls = await chat_loop(post, messages, qa)
            serving = None
            try:
                props = (await http.get("/props", timeout=5)).json()
                serving = Path(str(props.get("model_path") or "")).parent.name or None
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
    ended = datetime.now(UTC)
    energy = None
    if meter is not None:
        try:
            energy = await meter.measure(started, ended)
        except Exception:
            log.exception("measuring the box's answer failed")
    return {"answer": answer or "(no answer)", "calls": calls, "serving": serving,
            "seconds": round(time.perf_counter() - began, 1), "energy": energy,
            "prompt": source if held_out else "chat", "expected": expected if held_out else None,
            "text": first["text"] if first else answer}


async def ask_brain(question: str, pair: Any, qa: Any) -> dict[str, Any]:
    """The 122B pair's answer to the same question, in the same chat prompt, its tools run on the box's memory."""
    began = time.perf_counter()
    start = time.time()

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return _message(await pair.chat({**body, "model": BRAIN_MODEL, "temperature": 0.2, "max_tokens": ANSWER_TOKENS,
                                         "chat_template_kwargs": {"enable_thinking": False}}, timeout_s=SIDE_TIMEOUT_S))

    messages = [{"role": "system", "content": chat_system(datetime.now().astimezone())},
                {"role": "user", "content": question}]
    answer, calls = await chat_loop(post, messages, qa)
    end = time.time()
    try:
        energy = await pair.window_energy(start, end)
    except Exception:
        log.exception("measuring the brain's answer failed")
        energy = None
    return {"answer": answer or "(no answer)", "calls": calls, "serving": BRAIN_SERVING,
            "seconds": round(time.perf_counter() - began, 1), "energy": energy}


async def _side(coro, began: float) -> dict[str, Any]:
    """One side of the comparison with its timeout; a failure is said, never raised."""
    try:
        return await asyncio.wait_for(coro, SIDE_TIMEOUT_S + 5)
    except (asyncio.TimeoutError, TimeoutError):
        return {"error": f"no answer in {SIDE_TIMEOUT_S:.0f} s", "seconds": round(time.perf_counter() - began, 1)}
    except Refused as err:
        return {"error": err.message, "seconds": round(time.perf_counter() - began, 1)}
    except Exception as exc:  # the other side still answers
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "seconds": round(time.perf_counter() - began, 1)}


_PAIR: list[Any] = []


def default_pair() -> Any | None:
    """The 122B pair from the box's .env (contexts/circle/brain_pair.py), built once; None when not configured."""
    if not _PAIR:
        try:
            from contexts.circle.brain_pair import build_brain_pair

            _PAIR.append(build_brain_pair())
        except Exception:
            log.exception("the network brain could not be set up")
            _PAIR.append(None)
    pair = _PAIR[0]
    return pair if pair is not None and pair.configured else None


# --- the routes -------------------------------------------------------------------------------------------------

def lab_deploy_router(runner: DeployRunner, *, guard: Callable | None = None, questions: str | Path = QUESTIONS,
                      ask_transport: httpx.AsyncBaseTransport | None = None, meter: Any | None = None,
                      qa: Any | None = None, pair: Any | None = None, compare: bool | None = None) -> APIRouter:
    """`meter` (contexts/learning/answer_energy.AnswerMeter) measures each answer; without one `energy` is null.
    `qa` is the agent's QA (its tool executor and the box's memory; default the app's, app.state.atlas.qa), `pair`
    the 122B pair (default contexts/circle/brain_pair.py from .env), `compare` False leaves the brain out (default:
    on unless ATLAS_ASK_COMPARE=off)."""
    router = APIRouter()
    guarded = [Depends(guard)] if guard else []

    def refused(err: Refused) -> HTTPException:
        return HTTPException(err.status, err.message)

    @router.get("/api/learning/lab/deploy")
    def deploy_view() -> dict:
        return runner.view()

    @router.post("/api/learning/lab/deploy", status_code=202, dependencies=guarded)
    def deploy(body: dict[str, Any] | None = Body(default=None)) -> dict:
        version = (body or {}).get("version")
        if not isinstance(version, str) or not version:
            raise HTTPException(400, "give the version to deploy: {\"version\": \"...\"}")
        try:
            return runner.start(version, "deploy")
        except Refused as err:
            raise refused(err) from None

    @router.post("/api/learning/lab/deploy/rollback", status_code=202, dependencies=guarded)
    def rollback(body: dict[str, Any] | None = Body(default=None)) -> dict:
        version = (body or {}).get("version")
        if version is not None and not isinstance(version, str):
            raise HTTPException(400, "version must be a string")
        try:
            return runner.start(version or None, "rollback")
        except Refused as err:
            raise refused(err) from None

    @router.post("/api/learning/lab/deploy/ask", dependencies=guarded)
    async def ask(request: Request, body: dict[str, Any] | None = Body(default=None)) -> dict:
        question = (body or {}).get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(400, "give the question: {\"question\": \"...\"}")
        question = " ".join(SPECIAL_RE.sub("", question).replace("<|", "").replace("|>", "").split())[:300]
        if not question:
            raise HTTPException(400, "give the question: {\"question\": \"...\"}")
        tools = qa if qa is not None else getattr(getattr(request.app.state, "atlas", None), "qa", None)
        on = compare if compare is not None else os.environ.get(COMPARE_ENV, "").strip().lower() not in ("off", "0", "false", "no")
        in_app = getattr(request.app.state, "atlas", None) is not None  # the box's .env is read only in the gateway
        brain_pair = (pair if pair is not None else default_pair() if in_app else None) if on else None
        items = load_questions(questions)
        began = time.perf_counter()
        box_task = _side(ask_box(question, runner.llm_url or DEFAULT_LLM_URL, items=items, qa=tools, meter=meter,
                                 transport=ask_transport), began)
        if on:
            brain_task = (_side(ask_brain(question, brain_pair, tools), began) if brain_pair is not None else
                          asyncio.sleep(0, {"error": "this box has no keys for the pods (see contexts/circle/README.md)",
                                            "seconds": 0.0}))
            box, brain = await asyncio.gather(box_task, brain_task)
        else:
            box, brain = await box_task, None
        if "error" in box and (brain is None or "error" in brain):
            raise HTTPException(503, box["error"])  # nothing answered: as before, the model server is down
        # the fields the card read before the comparison, from the box's side
        old = {"question": question, "serving": box.get("serving"), "text": box.get("text", box.get("answer", "")),
               "calls": box.get("calls", []), "expected": box.get("expected"), "seconds": box.get("seconds", 0.0),
               "prompt": box.get("prompt", "chat"), "energy": box.get("energy")}
        return {**old, "box": box, "brain": brain}

    @router.post("/api/learning/lab/live-chat", dependencies=guarded)
    async def live_chat(request: Request, body: dict[str, Any] | None = Body(default=None)) -> StreamingResponse:
        # the live chat beside the power graph: tokens as Server-Sent Events, then the joules (live_chat.py)
        from contexts.learning import live_chat as chat

        question = (body or {}).get("question")
        target = (body or {}).get("target", "pair")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(400, "give the question: {\"question\": \"...\", \"target\": \"pair\" | \"box\"}")
        if target not in chat.TARGETS:
            raise HTTPException(400, "target is \"pair\" or \"box\"")
        question = " ".join(SPECIAL_RE.sub("", question).replace("<|", "").replace("|>", "").split())[:300]
        if not question:
            raise HTTPException(400, "give the question: {\"question\": \"...\"}")
        tools = qa if qa is not None else getattr(getattr(request.app.state, "atlas", None), "qa", None)
        in_app = getattr(request.app.state, "atlas", None) is not None
        brain_pair = (pair if pair is not None else default_pair() if in_app else None) if target == "pair" else None
        stream = chat.event_stream(question, target, qa=tools, pair=brain_pair,
                                   llm_url=runner.llm_url or DEFAULT_LLM_URL, transport=ask_transport, meter=meter)
        return StreamingResponse(stream, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router
