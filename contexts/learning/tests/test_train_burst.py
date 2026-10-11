"""The "Train for a minute" button's routes (contexts/learning/train_burst.py), with a fake controller and fake
samplers: nothing here reaches a pod."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.learning import train_burst as tb
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router

ENERGY = {"joules": 21000.0, "watts_avg": 175.0, "seconds": 120.0,
          "pods": {"C": {"joules": 11000.0, "watts_avg": 91.7, "idle_w": 10.27, "samples": 118},
                   "D": {"joules": 10000.0, "watts_avg": 83.3, "idle_w": 10.27, "samples": 119}}}


def answer_for(letter: str, *, ok: bool = True, steps: int = 1) -> dict:
    line = {"pod": letter, "ok": ok, "rc": 0 if ok else 1, "steps": steps, "seconds": 82.5, "train_s": 64.1,
            "loss": 0.1594, "peak_gb": 13.1, "vram_before_gb": 42.4, "vram_peak_gb": 55.5,
            "started": "2026-10-11T14:12:00.000Z", "ended": "2026-10-11T14:13:22.000Z",
            "error": None if ok else "the trainer exited with code 1"}
    tail = f"burst burst-{letter}: 1 step(s)\n{{'loss': '0.1594'}}\nBURST_RESULT {json.dumps(line)}"
    return {"ok": ok, "outcome": "ok" if ok else "fail", "output_tail": tail, "exit_code": 0 if ok else 1}


class FakePods:
    """Records which pods ran, holds them until released, and answers like lab.controller run-script."""

    def __init__(self, answers: dict | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.release = threading.Event()
        self.both_in = threading.Barrier(2, timeout=5)
        self.answers = answers or {}

    def __call__(self, letter: str, seconds: int) -> dict:
        self.calls.append((letter, seconds))
        self.release.wait(5)
        got = self.answers.get(letter, answer_for(letter))
        if isinstance(got, Exception):
            raise got
        return got


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_760_000_000.0

    def __call__(self) -> float:
        self.t += 60.0
        return self.t


def build(pods: FakePods, *, energy=None, available=lambda letters: None, guard=None):
    windows: list[tuple[float, float]] = []

    def measure(start: float, end: float):
        windows.append((start, end))
        if isinstance(energy, Exception):
            raise energy
        return ENERGY if energy is None else energy

    runner = tb.BurstRunner(run_pod=pods, energy=measure, available=available, clock=FakeClock())
    app = FastAPI()
    app.include_router(lab_deploy_router(DeployRunner("models", "state"), burst=runner, guard=guard))
    return TestClient(app), runner, windows


def test_idle_view_says_whether_the_controller_can_run():
    c, _, _ = build(FakePods())
    body = c.get("/api/learning/lab/train-burst").json()
    assert body["state"] == "idle" and body["available"] is True and body["reason"] is None
    assert body["per_pod"] == {} and body["energy"] is None and body["started_ms"] is None
    c2, _, _ = build(FakePods(), available=lambda letters: "missing keys for the pods: AUP_HUB_TOKEN_C")
    view = c2.get("/api/learning/lab/train-burst").json()
    assert view["available"] is False and "AUP_HUB_TOKEN_C" in view["reason"]


def test_a_burst_on_both_pods_runs_both_at_once_and_reports_steps_and_joules():
    pods = FakePods()
    c, runner, windows = build(pods)
    r = c.post("/api/learning/lab/train-burst", json={"pods": ["C", "D"], "seconds": 60})
    assert r.status_code == 202
    running = r.json()
    assert running["state"] == "running" and running["pods"] == ["C", "D"] and running["seconds_asked"] == 60
    assert running["started_ms"] is not None and running["ended_ms"] is None
    # a second press while it runs is refused
    again = c.post("/api/learning/lab/train-burst", json={"pods": ["D"]})
    assert again.status_code == 409 and "already running" in again.json()["detail"]
    pods.release.set()
    runner.wait(5)
    done = c.get("/api/learning/lab/train-burst").json()
    assert sorted(l for l, _ in pods.calls) == ["C", "D"] and all(s == 60 for _, s in pods.calls)
    assert done["state"] == "done" and done["message"] is None
    assert done["per_pod"]["C"] == {"ok": True, "steps": 1, "seconds": 82.5, "train_s": 64.1, "loss": 0.1594,
                                    "peak_gb": 13.1, "error": None}
    assert done["per_pod"]["D"]["steps"] == 1
    assert done["energy"] == ENERGY
    assert windows == [(done["started_ms"] / 1000, done["ended_ms"] / 1000)]
    assert done["ended_ms"] > done["started_ms"]


def test_the_two_pods_start_in_parallel():
    pods = FakePods()
    seen = []

    def run_pod(letter: str, seconds: int) -> dict:
        seen.append(letter)
        pods.both_in.wait()  # deadlocks (BrokenBarrierError) unless both pods run at the same time
        return answer_for(letter)

    c, runner, _ = build(run_pod)  # type: ignore[arg-type]
    assert c.post("/api/learning/lab/train-burst", json={}).status_code == 202  # default: both pods, 60 s
    runner.wait(5)
    view = c.get("/api/learning/lab/train-burst").json()
    assert view["state"] == "done" and sorted(seen) == ["C", "D"] and view["seconds_asked"] == 60


def test_one_pod_failing_is_said_and_the_other_still_counts():
    pods = FakePods({"C": answer_for("C", ok=False)})
    pods.release.set()
    c, runner, _ = build(pods)
    c.post("/api/learning/lab/train-burst", json={"pods": ["C", "D"], "seconds": 60})
    runner.wait(5)
    view = c.get("/api/learning/lab/train-burst").json()
    assert view["state"] == "done" and view["per_pod"]["C"]["ok"] is False
    assert "pod C" in view["message"] and "code 1" in view["message"]


def test_every_pod_failing_is_failed_with_the_controller_reason():
    pods = FakePods({"D": {"ok": False, "outcome": "not_running", "error": "the pod is not ready", "output_tail": ""}})
    pods.release.set()
    c, runner, _ = build(pods, energy=None)
    c.post("/api/learning/lab/train-burst", json={"pods": ["D"], "seconds": 60})
    runner.wait(5)
    view = c.get("/api/learning/lab/train-burst").json()
    assert view["state"] == "failed" and view["per_pod"]["D"]["error"] == "the pod is not ready"
    assert list(view["per_pod"]) == ["D"]


def test_a_controller_exception_is_a_failure_not_a_crash():
    pods = FakePods({"D": RuntimeError("hub said no")})
    pods.release.set()
    c, runner, _ = build(pods, energy=RuntimeError("samplers down"))
    c.post("/api/learning/lab/train-burst", json={"pods": ["D"]})
    runner.wait(5)
    view = c.get("/api/learning/lab/train-burst").json()
    assert view["state"] == "failed" and "hub said no" in view["per_pod"]["D"]["error"] and view["energy"] is None


def test_meters_that_do_not_cover_the_window_give_null_energy():
    pods = FakePods()
    pods.release.set()
    c, runner, _ = build(pods, energy=False)
    runner.energy = lambda a, b: None
    c.post("/api/learning/lab/train-burst", json={"pods": ["D"]})
    runner.wait(5)
    assert c.get("/api/learning/lab/train-burst").json()["energy"] is None


def test_no_controller_is_503_with_the_reason_and_never_a_fake():
    pods = FakePods()
    c, _, _ = build(pods, available=lambda letters: "the lab controller is not importable here (ImportError: x)")
    r = c.post("/api/learning/lab/train-burst", json={"pods": ["C", "D"], "seconds": 60})
    assert r.status_code == 503 and "not importable" in r.json()["detail"]
    assert pods.calls == [] and c.get("/api/learning/lab/train-burst").json()["state"] == "idle"


@pytest.mark.parametrize("body", [{"pods": ["E"]}, {"pods": []}, {"pods": "D"}, {"pods": ["D", "D"]},
                                  {"seconds": 5}, {"seconds": 9999}, {"seconds": "60"}, {"seconds": True}])
def test_bad_bodies_are_400(body):
    pods = FakePods()
    c, _, _ = build(pods)
    assert c.post("/api/learning/lab/train-burst", json=body).status_code == 400
    assert pods.calls == []


def test_the_post_is_guarded_and_the_get_is_not():
    def guard(authorization: str | None = Header(default=None)) -> None:
        if authorization != "Bearer ok":
            raise HTTPException(401, "log in")

    pods = FakePods()
    pods.release.set()
    c, runner, _ = build(pods, guard=guard)
    assert c.post("/api/learning/lab/train-burst", json={"pods": ["D"]}).status_code == 401
    assert c.get("/api/learning/lab/train-burst").status_code == 200
    assert c.post("/api/learning/lab/train-burst", json={"pods": ["D"]},
                  headers={"Authorization": "Bearer ok"}).status_code == 202
    runner.wait(5)


def test_parse_result_reads_the_last_burst_line():
    got = tb.parse_result(answer_for("D"))
    assert got["ok"] is True and got["peak_gb"] == 13.1 and got["loss"] == 0.1594
    assert tb.parse_result({"outcome": "timeout", "error": "no exit code after 540 s"})["error"].startswith("no exit")
    assert tb.parse_result({"outcome": "fail", "output_tail": "BURST_RESULT {not json"})["ok"] is False


def test_the_real_check_names_missing_keys_only(monkeypatch, tmp_path):
    from lab import hub as lab_hub

    def missing(*a, **k):
        raise lab_hub.MissingCredentials(["AUP_HUB_TOKEN_C"])

    monkeypatch.setattr(lab_hub, "load_accounts", missing)
    assert tb.controller_available(["C"]) == "missing keys for the pods: AUP_HUB_TOKEN_C"


def test_the_pod_script_is_in_the_repo():
    text = tb.SCRIPT.read_text(encoding="utf-8")
    assert "BURST_RESULT" in text and "ATLAS_TRAIN_WORKERS=0" in text and "--rank 16" in text
    assert "pkill" not in text and "kill -9" not in text  # it never touches the pair's processes
