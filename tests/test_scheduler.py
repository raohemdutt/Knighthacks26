"""R4: the device GPU scheduler (runtime/scheduler.py) and its use by LlamaServerRuntime."""

import json
import threading
import time

import httpx
import pytest

from contracts.types import ModelVersion
from runtime.llama import LlamaServerRuntime, PriorityGate
from runtime.scheduler import GpuScheduler


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.005)


def waiting(s: GpuScheduler) -> int:
    return sum(s.stats()["waiting"].values())


class Holder:
    """Holds one slot until released, so the test controls when the queue drains."""

    def __init__(self, s: GpuScheduler, priority="caption", vlm=False):
        self.go, self.started = threading.Event(), threading.Event()

        def run():
            with s.slot(priority, vlm=vlm):
                self.started.set()
                self.go.wait(5)

        self.thread = threading.Thread(target=run)
        self.thread.start()
        self.started.wait(5)

    def release(self):
        self.go.set()
        self.thread.join(5)


def queue(s: GpuScheduler, order: list, name: str, priority: str, vlm=False) -> threading.Thread:
    before = waiting(s)

    def run():
        with s.slot(priority, vlm=vlm):
            order.append(name)

    t = threading.Thread(target=run)
    t.start()
    wait_until(lambda: waiting(s) == before + 1)  # queued in this order, deterministically
    return t


def test_queue_order_is_detection_alerts_chat_captions_summaries_backfill():
    s, order = GpuScheduler(), []
    busy = Holder(s)
    threads = [queue(s, order, name, name) for name in ["backfill", "summary", "caption", "chat", "alert", "detection"]]
    busy.release()
    for t in threads:
        t.join(5)
    assert order == ["detection", "alert", "chat", "caption", "summary", "backfill"]


def test_first_come_first_served_within_a_rank():
    s, order = GpuScheduler(), []
    busy = Holder(s)
    threads = [queue(s, order, f"caption {i}", "caption") for i in range(4)]
    busy.release()
    for t in threads:
        t.join(5)
    assert order == ["caption 0", "caption 1", "caption 2", "caption 3"]


def test_unknown_priority_is_refused():
    with pytest.raises(ValueError):
        with GpuScheduler().slot("urgent-please"):
            pass


def test_vlm_cap_holds_background_images_but_lets_text_and_chat_through():
    now = [0.0]
    s, order = GpuScheduler(vlm_max_per_min=2, clock=lambda: now[0]), []
    for _ in range(2):  # two captions use the minute's budget
        with s.slot("caption", vlm=True):
            pass
    capped = queue(s, order, "caption 3", "caption", vlm=True)
    time.sleep(0.05)
    assert order == []  # over the cap: waits, without holding the slot
    for name, prio, vlm in [("summary", "summary", False), ("chat image", "chat", True)]:
        t = threading.Thread(target=_enter_and_record, args=(s, order, name, prio, vlm))
        t.start()
        t.join(5)
    assert order == ["summary", "chat image"] and s.stats()["capped_waits"] == 1
    now[0] = 60.0  # the first caption leaves the rolling minute
    capped.join(5)
    assert order[-1] == "caption 3"


def _enter_and_record(s, order, name, prio, vlm):
    with s.slot(prio, vlm=vlm):
        order.append(name)


def test_with_two_slots_background_leaves_one_free_for_chat():
    s = GpuScheduler(slots=2)
    caption = Holder(s, "caption")
    order: list = []
    second_caption = queue(s, order, "caption 2", "caption")
    time.sleep(0.05)
    assert order == []  # background may hold only slots-1 = 1
    chat = threading.Thread(target=_enter_and_record, args=(s, order, "chat", "chat", False))
    chat.start()
    chat.join(5)
    assert order == ["chat"]  # ran beside the caption: batched by llama-server's second slot
    caption.release()
    second_caption.join(5)
    assert order == ["chat", "caption 2"]


def test_coalesce_runs_identical_work_once():
    s, calls, results = GpuScheduler(), [], []
    gate = threading.Event()

    def work():
        calls.append(1)
        gate.wait(5)
        return "the answer"

    threads = [threading.Thread(target=lambda: results.append(s.coalesce("k", work))) for _ in range(3)]
    for t in threads:
        t.start()
    wait_until(lambda: s.counts["coalesced"] == 2)
    gate.set()
    for t in threads:
        t.join(5)
    assert calls == [1] and results == ["the answer"] * 3
    assert s.coalesce("k", lambda: "fresh") == "fresh"  # nothing cached once it finished


def test_coalesce_shares_the_error_too():
    s, gate, errors = GpuScheduler(), threading.Event(), []

    def boom():
        gate.wait(5)
        raise RuntimeError("server down")

    def call():
        try:
            s.coalesce("k", boom)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    wait_until(lambda: s.counts["coalesced"] == 1)
    gate.set()
    for t in threads:
        t.join(5)
    assert errors == ["server down", "server down"]


def test_from_config():
    s = GpuScheduler.from_config({"slots": 2, "vlm_max_per_min": 10, "coalesce": False})
    assert (s.slots, s.vlm_max, s.coalesce_enabled) == (2, 10, False)
    d = GpuScheduler.from_config(None)
    assert (d.slots, d.vlm_max, d.coalesce_enabled) == (1, None, True)


def test_old_gate_interface_still_maps_urgent_to_chat():
    gate, order = PriorityGate(), []
    busy = Holder(gate, "caption")
    t1 = threading.Thread(target=lambda: _old(gate, order, "caption", False))
    t1.start()
    wait_until(lambda: waiting(gate) == 1)
    t2 = threading.Thread(target=lambda: _old(gate, order, "question", True))
    t2.start()
    wait_until(lambda: waiting(gate) == 2)
    busy.release()
    t1.join(5)
    t2.join(5)
    assert order == ["question", "caption"]


def _old(gate, order, name, urgent):
    with gate.slot(urgent):
        order.append(name)


def _blocking_server(release: threading.Event):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        release.wait(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Someone came and went."}}]})

    return httpx.MockTransport(handle), seen


def test_runtime_coalesces_identical_summaries_into_one_request():
    release = threading.Event()
    transport, seen = _blocking_server(release)
    rt = LlamaServerRuntime(ModelVersion(version="v", base="b"), transport=transport)
    out: list[str] = []
    threads = [threading.Thread(target=lambda: out.append(rt.summarize(["a person sits", "the person leaves"])))
               for _ in range(2)]
    for t in threads:
        t.start()
    wait_until(lambda: rt.gate.counts["coalesced"] == 1)
    release.set()
    for t in threads:
        t.join(5)
    assert len(seen) == 1 and out == ["Someone came and went."] * 2


def test_runtime_passes_each_call_its_rank():
    ranks: list[tuple[str, bool]] = []

    class Spy(GpuScheduler):
        def slot(self, priority="caption", vlm=False):
            ranks.append((priority, vlm))
            return super().slot(priority, vlm)

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]}))
    rt = LlamaServerRuntime(ModelVersion(version="v", base="b"), transport=transport, scheduler=Spy())
    rt.caption(b"jpeg")
    rt.find_objects(b"jpeg")
    rt.summarize(["x"])
    rt.caption(b"jpeg", "what is on the desk?", priority="chat")  # e.g. a keyframe looked at to answer a question
    from contracts.types import ChatMessage
    rt.chat([ChatMessage(role="user", content="hi")], priority="backfill")
    assert ranks == [("caption", True), ("caption", True), ("summary", False), ("chat", True), ("backfill", False)]


def test_registry_builds_the_configured_scheduler(tmp_path):
    from runtime.registry import build_runtime

    cfg = tmp_path / "models.yaml"
    cfg.write_text("atlas_model: {backend: llama-server}\nscheduler: {slots: 1, vlm_max_per_min: 24}\n")
    rt = build_runtime(cfg)
    assert rt.gate.vlm_max == 24 and rt.gate.slots == 1
