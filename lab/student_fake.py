"""A stand-in for the student on the Jetson, and for the power samples of both nodes, so the routing hook's
closed-loop test (lab/routing_closed_loop.py, T11) runs with no Jetson, no pod and no network.

    clock = FakeClock()
    student = fake_student(load_expected(folder), miss_share=0.3, clock=clock)    # a FakeTeacher, as llama-server
    feed = FakePowerFeed({"jetson": 25.0, "pod-a": 112.0}, clock=clock)
    feed.note("jetson", started, ended, "r1")      # the node was answering over that window
    events = feed.poll()                           # power samples up to the clock's now, as telemetry events

The student. The Jetson serves the Atlas model with llama-server (config/models.yaml: atlas_model), whose chat API
is OpenAI-compatible, so the student fake is lab/teacher_fake.py's FakeTeacher under the student's model name and
checkpoint, at the student's pace, with a policy that answers like the fixture's reference except on a fixed share
of questions (student_policy). Which questions it misses is decided by a hash of the question and a seed, so a run
is the same every time. On a missed question it makes no tool call and says "I didn't see that.", which the label
agreement scores as a miss.

The power samples. FakePowerFeed stands for the two samplers (the Jetson's tegrastats reader, telemetry.jetson.power
with VDD_IN in mW, and lab/pod_power.py on pod A, telemetry.gpu.power with socket_w in W). Each node draws its idle
watts, and its busy watts while a window noted on it is open; each window's busy watts carry their own jitter (a
seeded normal draw times `jitter`), so joules per answer have a spread. poll() returns every sample from the last
one returned up to the clock's now, one every `period_s`, stamped on the one clock both nodes share.

The default numbers are placeholders that set the fakes' pace, nothing more: the teacher's 112 W, 0.52 s to the
first token and 2.83 tok/s are the live smoke on pod C in lab/README.md; the Jetson's 15.76 W idle is T13's
measurement there; the student's busy watts and its decode rate are not measured and are guesses.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from contracts.events import Event
from contracts.telemetry import GpuPower, JetsonPower
from lab.hub_fake import FakeClock
from lab.teacher_fake import NO_EVIDENCE, FakeTeacher, Policy, Turn, reference_policy

__all__ = ["STUDENT_MODEL", "STUDENT_ROOT", "STUDENT_TTFT_S", "STUDENT_DECODE_TOK_S", "STUDENT_BUSY_W",
           "JETSON_IDLE_W", "TEACHER_BUSY_W", "POD_IDLE_W", "student_policy", "fake_student", "FakePowerFeed"]

STUDENT_MODEL = "atlas-base"             # config/models.yaml atlas_model.version
STUDENT_ROOT = "Qwen/Qwen3.5-4B"         # config/models.yaml atlas_model.base
STUDENT_TTFT_S = 0.3                     # a guess: not measured on the Jetson
STUDENT_DECODE_TOK_S = 12.0              # a guess: not measured on the Jetson
STUDENT_BUSY_W = 25.0                    # a guess: VDD_IN while the student answers is not measured
JETSON_IDLE_W = 15.76                    # T13: VDD_IN mean with the gateway and model loaded, no request (README)
TEACHER_BUSY_W = 112.0                   # the live smoke on pod C: average socket_w over 4 requests (README)
POD_IDLE_W = 22.0                        # a guess near a fifth of the teacher's draw (telemetry rubric, TT2 item 7)
PERIOD_S = 0.5


def _share(seed: int, question: str) -> float:
    digest = hashlib.sha1(f"{seed}:{question}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def student_policy(expected: Mapping[str, Any], *, miss_share: float = 0.3, seed: int = 0) -> Policy:
    """Answers like the fixture's reference (teacher_fake.reference_policy), except on about miss_share of the
    questions, picked by a hash of the question and the seed: there it calls no tool and says it saw nothing."""
    if not 0.0 <= miss_share <= 1.0:
        raise ValueError(f"miss_share must be between 0 and 1, not {miss_share!r}")
    reference = reference_policy(expected)

    def answer(body: Mapping[str, Any]) -> Turn:
        question = next((m.get("content") for m in body.get("messages") or []
                         if isinstance(m, Mapping) and m.get("role") == "user"), "")
        if isinstance(question, str) and _share(seed, question) < miss_share:
            return Turn(content=NO_EVIDENCE)
        return reference(body)

    return answer


def fake_student(expected: Mapping[str, Any], *, miss_share: float = 0.3, seed: int = 0, clock: Any = None,
                 ttft_s: float = STUDENT_TTFT_S, decode_tok_s: float = STUDENT_DECODE_TOK_S) -> FakeTeacher:
    """The student as llama-server on the Jetson would answer: a FakeTeacher with the student's name and pace."""
    return FakeTeacher(student_policy(expected, miss_share=miss_share, seed=seed), model=STUDENT_MODEL,
                       root=STUDENT_ROOT, host="student.test", port=8080, ttft_s=ttft_s, decode_tok_s=decode_tok_s,
                       clock=clock)


def _dt(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=UTC)


class FakePowerFeed:
    """Power samples of the Jetson and pod A on one clock. busy: node -> watts while answering; idle: node ->
    watts otherwise; jitter: the relative spread of each window's busy watts."""

    def __init__(self, busy: Mapping[str, float], *, idle: Mapping[str, float] | None = None, jitter: float = 0.05,
                 period_s: float = PERIOD_S, seed: int = 0, clock: FakeClock | None = None) -> None:
        self.busy = dict(busy)
        self.idle = {node: (idle or {}).get(node, JETSON_IDLE_W if node == "jetson" else POD_IDLE_W)
                     for node in self.busy}
        self.jitter = jitter
        self.period_s = period_s
        self.clock = clock or FakeClock()
        self._rng = random.Random(seed)
        self._windows: dict[str, list[tuple[float, float, float]]] = {node: [] for node in self.busy}
        self._next: dict[str, float] = {node: self.clock.time() for node in self.busy}

    def note(self, node: str, started: float, ended: float, request_id: str = "") -> float:
        """The node answered over [started, ended] (clock seconds); returns the busy watts drawn for it."""
        watts = self.busy[node] * max(0.0, 1.0 + self.jitter * self._rng.gauss(0.0, 1.0))
        self._windows[node].append((started, ended, watts))
        return watts

    def watts(self, node: str, t: float) -> float:
        for started, ended, watts in reversed(self._windows[node]):
            if started <= t <= ended:
                return watts
        return self.idle[node]

    def poll(self) -> list[Event]:
        """Every sample from the last one returned up to now, both nodes, oldest first."""
        now = self.clock.time()
        out: list[tuple[float, Event]] = []
        for node in self.busy:
            t = self._next[node]
            while t <= now:
                out.append((t, self._sample(node, t)))
                t += self.period_s
            self._next[node] = t
        return [event for _, event in sorted(out, key=lambda pair: pair[0])]

    def _sample(self, node: str, t: float) -> Event:
        watts = self.watts(node, t)
        if node == "jetson":
            return JetsonPower(node=node, ts=_dt(t), source="fake", period_ms=self.period_s * 1000.0,
                               rails={"VDD_IN": watts * 1000.0}, nvpmodel="MAXN_SUPER")
        return GpuPower(node=node, ts=_dt(t), source="fake", period_ms=self.period_s * 1000.0, socket_w=watts,
                        gfx_mhz=None, mem_mhz=None, throttled=False, sampler="lab/student_fake.py",
                        under_load=watts > self.idle[node])
