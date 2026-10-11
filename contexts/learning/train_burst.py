"""The "Train for a minute" button on the live power graph: a real, short LoRA training burst on the AMD pods, run
beside the serving 122B pair, so the watts rise on the graph, then the joules it took (L37).

    POST /api/learning/lab/train-burst   {pods: ["C", "D"] | ["D"] | ["C"], seconds: 60}     (login, like deploy)
        202 the view below, state "running"
        400 a bad body; 409 a burst already runs; 503 the controller cannot run from this gateway (the reason)
    GET  /api/learning/lab/train-burst
        {"available": bool, "reason": str | null,
         "state": "idle" | "running" | "done" | "failed", "pods": [str], "seconds_asked": int | null,
         "started": iso | null, "ended": iso | null, "started_ms": int | null, "ended_ms": int | null,
         "per_pod": {"C": {"ok", "steps", "seconds", "train_s", "loss", "peak_gb", "error"} | null, "D": ...},
         "energy": {"joules", "watts_avg", "seconds", "pods": {"C": {...}, "D": {...}}} | null,
         "message": str | null}

Each named pod runs lab/train_burst.sh through the lab's own controller (lab.controller run-script, the hub's Jupyter
terminal with that account's token from the box's .env), all pods in parallel threads, so a burst on both pods starts
on both at once. The script trains Qwen3.5-4B with the R66 proof's LoRA settings for about `seconds` and prints one
`BURST_RESULT {...}` line, which per_pod repeats. `started_ms`/`ended_ms` are this gateway's clock (epoch ms), the
window the graph shades.

`energy` is both pods' samplers over the burst window (contexts/circle/brain_pair.py BrainPair.window_energy): gross
joules, nothing subtracted, each pod's idle_w beside it, measured once the burst ends; null when the meters do not
cover the window. The burst never touches the pair's processes: the script only starts its own trainer.

When lab.controller cannot be imported or the pods' keys are missing, POST answers 503 with the reason: never a
pretend burst.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("atlas.learning.train_burst")

PODS = ("C", "D")
DEFAULT_SECONDS = 60
MIN_SECONDS, MAX_SECONDS = 30, 300
AVAILABLE_CACHE_S = 30.0  # GET is polled; the .env is read at most this often
RESULT_RE = re.compile(r"^BURST_RESULT (\{.*\})\s*$", re.MULTILINE)
SCRIPT = Path(__file__).resolve().parents[2] / "lab" / "train_burst.sh"


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z") if t is not None else None


def parse_result(answer: Mapping[str, Any]) -> dict[str, Any]:
    """One pod's per_pod entry from the controller's run-script answer: the script's BURST_RESULT line, or the
    controller's own failure when there is none."""
    tail = str(answer.get("output_tail") or "")
    found = RESULT_RE.findall(tail)
    if found:
        try:
            got = json.loads(found[-1])
        except ValueError:
            got = None
        if isinstance(got, dict):
            return {"ok": bool(got.get("ok")), "steps": got.get("steps"), "seconds": got.get("seconds"),
                    "train_s": got.get("train_s"), "loss": got.get("loss"), "peak_gb": got.get("peak_gb"),
                    "error": got.get("error")}
    error = answer.get("error") or f"the pod script ended with {answer.get('outcome') or 'no answer'}"
    return {"ok": False, "steps": None, "seconds": None, "train_s": None, "loss": None, "peak_gb": None,
            "error": str(error)[:200]}


def controller_available(letters: Sequence[str] = PODS) -> str | None:
    """None when this process can run lab.controller run-script on those pods, else why not (key names only)."""
    try:
        from lab import hub as lab_hub
        from lab import controller  # noqa: F401  (the import is the check)
    except Exception as exc:
        return f"the lab controller is not importable here ({type(exc).__name__}: {str(exc)[:120]})"
    try:
        lab_hub.load_accounts(list(letters), require_role=True)
    except lab_hub.MissingCredentials as exc:
        return "missing keys for the pods: " + ", ".join(exc.missing)
    except Exception as exc:  # PodRoleError and friends
        return f"the pods are not set up on this box: {str(exc)[:160]}"
    if not SCRIPT.is_file():
        return f"{SCRIPT.name} is not in this copy of the repo"
    return None


def controller_run_pod(letter: str, seconds: int) -> dict[str, Any]:
    """Runs lab/train_burst.sh on that pod through the lab's controller and returns its answer."""
    from lab import controller

    timeout = seconds * 4 + 300
    answer, _code = controller.run(["run-script", "--account", letter, "--timeout", str(timeout),
                                    str(SCRIPT), letter, str(seconds)])
    return answer


def pair_energy(start: float, end: float) -> dict[str, Any] | None:
    """Both pods' gross joules over [start, end] from their samplers, with a pair built for this one measurement
    (its async client lives and dies on this thread's loop)."""
    from contexts.circle.brain_pair import build_brain_pair

    async def measure() -> dict[str, Any] | None:
        pair = build_brain_pair()
        try:
            if not pair.configured:
                return None
            return await pair.window_energy(start, end)
        finally:
            await pair.aclose()

    return asyncio.run(measure())


class BurstRunner:
    """At most one burst at a time. run_pod, energy, available and clock replace the pods, the samplers, the
    controller check and the time in tests."""

    def __init__(self, *, run_pod: Callable[[str, int], Mapping[str, Any]] = controller_run_pod,
                 energy: Callable[[float, float], dict[str, Any] | None] = pair_energy,
                 available: Callable[[Sequence[str]], str | None] = controller_available,
                 clock: Callable[[], float] = time.time) -> None:
        self.run_pod = run_pod
        self.energy = energy
        self.available = available
        self.clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._idle()
        self._reason: tuple[float, str | None] | None = None  # the controller check, cached AVAILABLE_CACHE_S

    def _available(self) -> str | None:
        now = time.monotonic()
        if self._reason is None or now - self._reason[0] > AVAILABLE_CACHE_S:
            self._reason = (now, self.available(PODS))
        return self._reason[1]

    @staticmethod
    def _idle() -> dict[str, Any]:
        return {"state": "idle", "pods": [], "seconds_asked": None, "started": None, "ended": None,
                "per_pod": {}, "energy": None, "message": None}

    def view(self) -> dict[str, Any]:
        with self._lock:
            s = json.loads(json.dumps(self._state))
        reason = None if s["state"] == "running" else self._available()
        started, ended = s.pop("_t0", None), s.pop("_t1", None)
        return {"available": reason is None, "reason": reason, **s,
                "started_ms": round(started * 1000) if started is not None else None,
                "ended_ms": round(ended * 1000) if ended is not None else None}

    def start(self, pods: Any, seconds: Any) -> dict[str, Any]:
        if pods is None:
            pods = list(PODS)
        if not isinstance(pods, list) or not pods or any(p not in PODS for p in pods) or len(set(pods)) != len(pods):
            raise Refused(400, "pods is a list of \"C\" and \"D\", e.g. [\"C\", \"D\"]")
        if seconds is None:
            seconds = DEFAULT_SECONDS
        if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise Refused(400, f"seconds is a number from {MIN_SECONDS} to {MAX_SECONDS}")
        seconds = int(seconds)
        pods = [p for p in PODS if p in pods]
        with self._lock:
            if self._state["state"] == "running":
                raise Refused(409, "a training burst is already running on " + " and ".join(f"pod {p}" for p in self._state["pods"]))
        reason = self.available(pods)
        self._reason = (time.monotonic(), reason) if list(pods) == list(PODS) else self._reason
        if reason:
            raise Refused(503, reason)
        with self._lock:
            if self._state["state"] == "running":
                raise Refused(409, "a training burst is already running")
            t0 = self.clock()
            self._state = {"state": "running", "pods": pods, "seconds_asked": seconds, "started": _iso(t0),
                           "ended": None, "per_pod": {p: None for p in pods}, "energy": None, "message": None,
                           "_t0": t0, "_t1": None}
            self._thread = threading.Thread(target=self._run, args=(pods, seconds, t0), daemon=True,
                                            name="train-burst")
            self._thread.start()
        return self.view()

    def _one(self, letter: str, seconds: int, results: dict[str, Any]) -> None:
        try:
            results[letter] = parse_result(self.run_pod(letter, seconds))
        except Exception as exc:
            log.exception("the training burst on pod %s failed", letter)
            results[letter] = parse_result({"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        with self._lock:
            self._state["per_pod"][letter] = results[letter]

    def _run(self, pods: list[str], seconds: int, t0: float) -> None:
        results: dict[str, Any] = {}
        threads = [threading.Thread(target=self._one, args=(p, seconds, results), daemon=True) for p in pods]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t1 = self.clock()
        energy = None
        try:
            energy = self.energy(t0, t1)
        except Exception:
            log.exception("measuring the training burst failed")
        trained = [p for p in pods if (results.get(p) or {}).get("ok")]
        failed = [p for p in pods if p not in trained]
        message = None
        if failed:
            message = "; ".join(f"pod {p}: {(results.get(p) or {}).get('error') or 'failed'}" for p in failed)
        with self._lock:
            self._state.update({"state": "done" if trained else "failed", "ended": _iso(t1), "_t1": t1,
                                "per_pod": {p: results.get(p) for p in pods}, "energy": energy, "message": message})

    def wait(self, timeout: float | None = None) -> None:
        """For tests: blocks until the running burst has ended."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
