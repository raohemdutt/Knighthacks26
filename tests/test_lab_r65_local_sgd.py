"""R65: the two-pod LoRA's exchange pieces (lab/r65_local_sgd.py): the int8 code, the mailbox and the fetcher."""
from __future__ import annotations

import socket
import threading

import pytest

from lab import r65_local_sgd as m

torch = pytest.importorskip("torch")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_int8_code_round_trips_with_a_small_error_and_error_feedback_keeps_the_rest():
    torch.manual_seed(0)
    x = torch.randn(3 * m.BLOCK + 17) * 1e-3
    blob, back = m.quantize(torch, x)
    assert len(blob) == 4 * 4 + x.numel()                      # four float32 scales, one int8 per value
    again = m.dequantize(torch, blob, x.numel())
    assert torch.equal(again, back)                            # the peer decodes exactly what the sender kept
    assert float((x - back).norm() / x.norm()) < 0.01
    residual = x - back
    blob2, back2 = m.quantize(torch, x + residual)             # what is lost once is sent the next round
    assert float((2 * x - back - back2).norm() / (2 * x).norm()) < 0.01


def test_zero_delta_codes_to_zero():
    blob, back = m.quantize(torch, torch.zeros(10))
    assert torch.count_nonzero(back) == 0
    assert torch.count_nonzero(m.dequantize(torch, blob, 10)) == 0


def test_mailbox_serves_chunks_and_the_relay_fetcher_reads_them_in_parallel():
    port = free_port()
    box = m.Mailbox(port)
    try:
        box.put("1/0", b"a" * 1000)
        box.put("1/1", b"b" * 10)
        fetcher = m.Fetcher("relay", "nobody", port, "", "127.0.0.1", "", parallel=2)
        assert fetcher.fetch(["1/0", "1/1"], timeout=5) == [b"a" * 1000, b"b" * 10]
    finally:
        box.close()


def test_fetcher_waits_for_a_chunk_that_is_published_late():
    port = free_port()
    box = m.Mailbox(port)
    try:
        threading.Timer(1.0, box.put, args=("2/0", b"late")).start()
        fetcher = m.Fetcher("relay", "nobody", port, "", "127.0.0.1", "", parallel=1)
        assert fetcher.fetch(["2/0"], timeout=10) == [b"late"]
    finally:
        box.close()


def test_fetcher_gives_up_after_its_deadline():
    port = free_port()
    box = m.Mailbox(port)
    try:
        fetcher = m.Fetcher("relay", "nobody", port, "", "127.0.0.1", "", parallel=1)
        with pytest.raises(TimeoutError):
            fetcher.fetch(["3/0"], timeout=1)
    finally:
        box.close()


def test_swap_exchanges_both_ways_at_once():
    a, b = socket.socketpair()
    got: dict[str, bytes] = {}
    thread = threading.Thread(target=lambda: got.update(b=m.swap(b, b"from b")))
    thread.start()
    got["a"] = m.swap(a, b"from a" * 100000)
    thread.join()
    assert got == {"a": b"from b", "b": b"from a" * 100000}
