"""R64 sync energy: lab/link_speed.py's measure() (the R14 verb's own code) on account C, with every push and pull
bracketed by a read of pod C's power sampler (/proxy/8790/status: last_seq and the pod's clock), so each transfer's
window is a range of sampler sequence numbers and no clock has to be aligned. Prints no secret.

    sync_energy.py OUT_DIR [--repeats N] [--idle-s S]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from lab import hub, link_speed
from lab.proxy import PodLink

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--idle-s", type=float, default=90.0)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

acct = hub.load_accounts(["C"], require_role=True)["C"]
link = PodLink.for_account(acct)
marks: list[dict] = []


def mark(what: str, **extra) -> dict:
    t0 = time.time()
    body = link.request("GET", link.proxy_rel(8790, "status")).json()
    t1 = time.time()
    row = {"what": what, "local_before": round(t0, 3), "local_after": round(t1, 3), "last_seq": body.get("last_seq"),
           "pod_sent": body.get("pod_sent"), **extra}
    marks.append(row)
    print(json.dumps(row), file=sys.stderr, flush=True)
    return row


class Timed:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def push(self, local, remote, chunk_bytes):
        kind = "manifest" if str(remote).endswith("manifest.json") else "payload"
        mark(f"push-start-{kind}", remote=str(remote))
        t = time.perf_counter()
        try:
            return self.inner.push(local, remote, chunk_bytes)
        finally:
            mark(f"push-end-{kind}", seconds=round(time.perf_counter() - t, 3))

    def pull(self, version, dest, part_bytes):
        mark("pull-start")
        t = time.perf_counter()
        try:
            answer = self.inner.pull(version, dest, part_bytes)
            return answer
        finally:
            mark("pull-end", seconds=round(time.perf_counter() - t, 3))


mark("idle-start")
time.sleep(args.idle_s)
mark("idle-end")
target = link_speed.LiveTarget(acct)
try:
    report = link_speed.measure(Timed(target), sizes=[1_000_000_000], repeats=args.repeats, probe=False,
                                progress=lambda line: print(f"link-speed: {line}", file=sys.stderr, flush=True))
finally:
    target.close()
mark("after-end")
time.sleep(min(args.idle_s, 60.0))
mark("idle2-end")
(out / "R64_link_speed_report.md").write_text(report.markdown(), encoding="utf-8", newline="\n")
(out / "marks.json").write_text(json.dumps(marks, indent=1) + "\n", encoding="utf-8")
(out / "report.json").write_text(json.dumps(report.as_dict(), indent=1, default=str) + "\n", encoding="utf-8")
print(json.dumps({"ok": report.ok, "run_id": report.run_id, "cleanup_problems": report.cleanup_problems,
                  "ledger_errors": report.ledger_errors}))
link.close()
