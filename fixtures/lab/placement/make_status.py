"""Write status.json: the answer `python -m lab.controller status --probe-port 8000` gives for accounts C and D,
made by the real controller against lab/hub_fake.py set up as the two SJC pods were recorded on 2026-10-10.

    python fixtures/lab/placement/make_status.py        # from the repo root; rewrites status.json beside this file

What comes from the record and what from the fake:

- Pod C (account C, pod-a's role): server ready since 19:07:12Z, runtime 2400 minutes, accelerator strix-halo
  (lab/refine/research/plumbing/measurements-2026-10-10.md, M24); the teacher answers as model "teacher" on port
  8000 through the proxy (R29 task page, /v1/models 200 at 20:09 UTC; it ran on through the T6 sampler run).
- Pod D (account D, pod-b's role): the session R13 launched at 21:24 UTC with 2400 minutes and strix-halo
  (lab/README.md, controller section; R30 task page, line of Sat 6:10 PM), no model server (R30 page, line of
  Sat 6:40 PM: nothing listens on port 8000 after 22:21 UTC).
- Both: quota disabled on SJC (balance 0, unlimited, enabled false; M19 and M24); the account names are
  acct-c and acct-d, never the real ones.
- From the fake, not the record: the last activity times (pod C as just active, pod D as of 22:21 UTC) and the
  moment of the snapshot, 22:30 UTC on the hub's clock. The fake's clock offset is set to 0 so the times above
  stay the hub's own (M22: the 46 s offset seen that day was the workstation's slow clock, M17).

The controller writes ledger rows for the call; they go to a temporary state folder that is deleted afterwards.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from lab.controller import Controller  # noqa: E402
from lab.hub_fake import FakeClock, FakeHub, _epoch  # noqa: E402

SNAPSHOT = "2026-10-10T22:30:00Z"
SJC = "https://sjc.aupcloud.io"
TOKENS = {"C": "cccc9999dddd0000eeee1111ffff2222", "D": "dddd3333eeee4444ffff5555aaaa6666"}  # fake tokens


def status() -> dict:
    clock = FakeClock(SNAPSHOT)
    hub = FakeHub(clock, hub_clock_ahead_s=0.0)
    accounts = {letter: hub.add_account(letter, user=f"acct-{letter.lower()}", token=TOKENS[letter],
                                        quota_enabled=False, url=SJC) for letter in ("C", "D")}
    now = _epoch(SNAPSHOT)
    hub.start_server("C", runtime_minutes=2400, accelerator="strix-halo",
                     started_ago_s=now - _epoch("2026-10-10T19:07:12Z"), idle_s=0)
    hub.serve_model("C", port=8000, models=("teacher",))
    hub.start_server("D", runtime_minutes=2400, accelerator="strix-halo",
                     started_ago_s=now - _epoch("2026-10-10T21:24:00Z"), idle_s=now - _epoch("2026-10-10T22:21:00Z"))
    before = os.environ.get("LAB_STATE_DIR")
    with tempfile.TemporaryDirectory() as state:
        os.environ["LAB_STATE_DIR"] = state
        try:
            with Controller(accounts, transport=hub.transport, sleep=clock.sleep, clock=clock.time,
                            environ={}) as controller:
                answer = controller.status(["C", "D"], probe_port=8000)
        finally:
            if before is None:
                os.environ.pop("LAB_STATE_DIR", None)
            else:
                os.environ["LAB_STATE_DIR"] = before
    answer.pop("ledger_error", None)
    return answer


if __name__ == "__main__":
    (HERE / "status.json").write_text(json.dumps(status(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {HERE / 'status.json'}")
