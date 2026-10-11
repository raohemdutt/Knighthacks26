"""A pod's clock against this machine's, from the four timestamps of an exchange with its sampler through the hub proxy.

Every answer of lab/pod_power.py's endpoint carries `pod_received` and `pod_sent`, the pod's clock when the request
arrived and when the reply left (milliseconds). With t1 (sent) and t4 (received) on this machine's clock, one GET is
one NTP-style exchange (domain/energy.py clock_offset). The best of several, the one with the smallest round trip,
has the smallest error, so a reader keeps the last few and reports that one.

    python -m contexts.telemetry.infrastructure.clock C [--port 8790] [--exchanges 10]
        measures account C's pod through the hub proxy (keys AUP_HUB_USER_C, AUP_HUB_TOKEN_C, AUP_HUB_URL_C, read from
        the environment or the nearest .env up from here; the token is sent only in the Authorization header and
        printed nowhere) and prints the telemetry.clock.offset event as JSON. It only reads. Off the Jetson the
        reference is this machine's clock (reference "local:<hostname>"), not the Jetson's.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contexts.telemetry.domain.energy import clock_offset
from contexts.telemetry.domain.nodes import JETSON
from contracts.telemetry import ClockOffset

VIA = "hub proxy /samples"


@dataclass(frozen=True)
class Exchange:
    t1: datetime  # request sent, this machine's clock
    t2: datetime  # request received, the pod's clock
    t3: datetime  # reply sent, the pod's clock
    t4: datetime  # reply received, this machine's clock

    @property
    def measured(self) -> tuple[float, float, float]:
        """(offset_s, round_trip_s, error_s): the pod's clock minus this one's, within error_s."""
        return clock_offset(*(_s(t) for t in (self.t1, self.t2, self.t3, self.t4)))


def _s(t: datetime) -> float:
    return t.timestamp()


def _parse(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo else out.replace(tzinfo=UTC)


def exchange_of(t1: datetime, body: Mapping[str, Any], t4: datetime) -> Exchange | None:
    """The exchange in one answer, or None when it carries no pod timestamps or they run backwards."""
    t2, t3 = _parse(body.get("pod_received")), _parse(body.get("pod_sent"))
    if t2 is None or t3 is None:
        return None
    exchange = Exchange(t1, t2, t3, t4)
    try:
        exchange.measured
    except ValueError:
        return None
    return exchange


def best_offset(node: str, exchanges: Sequence[Exchange], reference: str = JETSON, via: str = VIA) -> ClockOffset:
    """The telemetry.clock.offset of the exchange with the smallest round trip."""
    if not exchanges:
        raise ValueError("no exchanges to measure the offset from")
    best = min(exchanges, key=lambda e: e.measured[1])
    offset, round_trip, error = best.measured
    return ClockOffset(node=node, ts=best.t4, reference=reference, via=via, t1=best.t1, t2=best.t2, t3=best.t3,
                       t4=best.t4, offset_s=offset, round_trip_s=round_trip, error_s=error, exchanges=len(exchanges))


# --- the one-off measurement from this machine -------------------------------------------------------------------

def _env_file(start: Path) -> dict[str, str]:
    for folder in (start, *start.parents):
        path = folder / ".env"
        if path.is_file():
            out = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition("=")
                if sep and not key.lstrip().startswith("#"):
                    out[key.strip()] = value.strip().strip('"').strip("'")
            return out
    return {}


def measure(letter: str, *, port: int = 8790, count: int = 10, node: str | None = None,
            env: Mapping[str, str] | None = None, transport: Any = None, reference: str = JETSON) -> ClockOffset:
    """`count` GETs of the account's pod sampler through the hub proxy, read only; the best one as an event."""
    import httpx

    from lab import hub as lab_hub

    env = {**_env_file(Path.cwd()), **os.environ} if env is None else env
    user_key, token_key = lab_hub.key_names(letter)
    user, token = env.get(user_key, "").strip(), env.get(token_key, "").strip()
    if not user or not token:
        raise SystemExit(f"set {user_key} and {token_key}")
    hub = (env.get(lab_hub.url_key(letter)) or env.get(lab_hub.ENV_URL) or lab_hub.DEFAULT_URL).rstrip("/")
    url = f"{hub}/user/{user}/proxy/{port}/samples"
    headers = {"Authorization": f"token {token}",
               "User-Agent": (env.get(lab_hub.ENV_USER_AGENT) or "").strip() or lab_hub.USER_AGENT}
    exchanges = []
    with httpx.Client(timeout=20, transport=transport, follow_redirects=False) as client:
        for _ in range(count):
            t1 = datetime.now(UTC)
            reply = client.get(url, params={"after": 2**31, "limit": 1}, headers=headers)
            t4 = datetime.now(UTC)
            if reply.status_code != 200:
                raise SystemExit(f"the {letter} sampler answered HTTP {reply.status_code}")
            body = reply.json()
            exchange = exchange_of(t1, body, t4)
            if exchange is not None:
                exchanges.append(exchange)
            node = node or body.get("node")
    if not exchanges:
        raise SystemExit("no answer carried pod_received and pod_sent")
    from contexts.telemetry.domain.nodes import canonical

    return best_offset(canonical(node or f"account-{letter}"), exchanges, reference=reference)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.telemetry.infrastructure.clock")
    parser.add_argument("account", help="the hub account letter, e.g. C")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--exchanges", type=int, default=10)
    args = parser.parse_args(argv)
    from contexts.telemetry.infrastructure import host

    # off the Jetson the offset is against this machine's clock, and says so
    reference = JETSON if host.on_jetson() else f"local:{socket.gethostname()}"
    event = measure(args.account.upper(), port=args.port, count=args.exchanges, reference=reference)
    print(json.dumps(event.model_dump(mode="json"), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
