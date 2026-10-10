"""The pod reader against the hub as main models it (lab.hub_fake.FakeHub), and the hub settings main uses.

main's lab/hub.py sends a browser User-Agent because the hub's front end answers 403 (Cloudflare error 1010) to a
bare script agent, and lab/hub_fake.py reproduces that. The reader must therefore get through the same front end,
and must read its hub URL from the one setting the rest of the lab reads (lab.hub.ENV_URL).
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from contexts.telemetry.infrastructure.pods import PodEndpoint, PodReader, pod_endpoints
from contracts.telemetry import GpuPower
from lab.hub_fake import FakeClock, FakeHub

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "pod_power"
TOKEN = "hub-token-for-tests"
NOW = datetime(2026, 10, 10, 18, 2, tzinfo=UTC)


def recorded(name: str = "samples_gpu-a.jsonl") -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def test_the_reader_gets_through_the_hub_front_end_that_refuses_a_script_user_agent():
    events = recorded()[:3]
    fake = FakeHub(FakeClock())
    fake.add_account("A", user="acct-a", token=TOKEN)
    fake.queue("GET", "/proxy/8790/samples", 200, body={
        "ok": True, "node": "gpu-a", "sampler_started": "2026-10-10T18:00:00.000Z", "first_seq": 1, "last_seq": 3,
        "samples": [{"seq": seq, "event": event} for seq, event in enumerate(events, 1)]})
    reader = PodReader([PodEndpoint("gpu-a", "https://dub.aupcloud.io/user/acct-a/proxy/8790", TOKEN)],
                       transport=fake.transport, clock=lambda: NOW, sleep=lambda _: asyncio.sleep(0), rounds=1)

    async def run():
        return [e async for e in reader.samples()]

    got = asyncio.run(run())

    assert reader.status()["gpu-a"]["error"] is None
    assert [e.event_id for e in got] == [e["event_id"] for e in events]
    assert all(isinstance(e, GpuPower) for e in got)
    assert fake.calls, "the reader made no request to the hub"
    agents = [call.headers.get("user-agent", "") for call in fake.calls]
    assert all(agent.startswith("Mozilla/5.0") for agent in agents), agents


def test_the_lab_hub_url_setting_moves_the_pods_and_an_accounts_own_hub_url_still_wins():
    env = {"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": "t", "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": "u",
           "LAB_HUB_URL": "https://sjc.hub.test/", "AUP_HUB_URL_B": "https://other.hub.test"}

    pods = pod_endpoints(env)

    assert [(p.node, p.base_url) for p in pods] == [
        ("gpu-a", "https://sjc.hub.test/user/acct-a/proxy/8790"),
        ("gpu-b", "https://other.hub.test/user/acct-b/proxy/8790")]
