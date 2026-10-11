"""Pod C sampler through the hub proxy. Prints no secret.

    samples.py status                  -> last_seq and the pod's clock
    samples.py fetch AFTER OUT.jsonl   -> every sample with seq > AFTER, one JSON object a line
"""
import json
import sys

from lab import hub
from lab.proxy import PodLink


def main() -> int:
    acct = hub.load_accounts(["C"])["C"]
    with PodLink.for_account(acct) as link:
        if sys.argv[1] == "status":
            body = link.request("GET", link.proxy_rel(8790, "status")).json()
            print(json.dumps({k: body.get(k) for k in ("ok", "sampler", "period_ms", "measured_period_ms",
                                                       "last_seq", "pod_sent", "errors", "stale_seen")}))
            return 0
        after = int(sys.argv[2])
        out = open(sys.argv[3], "w", encoding="utf-8")
        n = 0
        while True:
            body = link.request("GET", link.proxy_rel(8790, f"samples?after={after}")).json()
            batch = body.get("samples") or []
            if not batch:
                break
            for s in batch:
                out.write(json.dumps(dict(s.get("event", s), seq=s.get("seq"))) + "\n")
                n += 1
            after = body.get("last_seq") or batch[-1].get("seq")
            if after is None or len(batch) < 2:
                break
        out.close()
        print(json.dumps({"samples": n, "last_seq": after}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
