"""R65's two-pod LoRA without the relay timing (lab/r65_local_sgd.py --no-relay-timing), for the live loop's budget.
No torch needed: only the control channel's agreement is exercised."""
from __future__ import annotations

import socket
import threading

from lab import r65_local_sgd as m



def test_no_relay_timing_skips_the_relay_fetch_and_still_agrees_with_the_peer():
    """The live loop's budget (L29): the final fetch over the Tailscale relay (about 15 s) is left out, the done
    agreement stays, so neither rank closes its mailbox early."""
    a, b = socket.socketpair()

    class NoFetch:
        def fetch(self, *args, **kwargs):
            raise AssertionError("the relay must not be asked")

    class Link:
        sock = a

    got: dict[str, object] = {}
    thread = threading.Thread(target=lambda: got.update(peer=m.swap_json(b, {"done": True})))
    thread.start()
    record = {"final_peer": {"keys": ["k"], "sha": "x"}}
    m.time_relay(Link(), NoFetch(), record, timed=False)
    thread.join()
    assert record["relay_exchange"] == {"skipped": "--no-relay-timing"} and got["peer"] == {"done": True}
