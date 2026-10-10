"""ntfy on the Jetson: messages (W2 send_message), alerts and approval cards (X7).

Adapted from the home Atlas runner's NtfyNotifier: ntfy's JSON publish API,
with Approve / Deny as `http` action buttons that POST straight back to the
gateway, so the phone can decide without opening the app.
"""

from collections.abc import Awaitable, Callable

import httpx

from contexts.workspace.domain.approvals import ActionRequest
from contracts.agent import AlertRaised, PlanCreated
from contracts.habits import NudgeRaised
from contracts.workspace import ApprovalRequested

PRIORITY = {"low": 2, "default": 3, "high": 4, "urgent": 5}


class Ntfy:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        token: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self.transport = transport

    async def publish(
        self,
        topic: str,
        message: str,
        title: str | None = None,
        priority: str = "default",
        tags: list[str] | None = None,
        click: str | None = None,
        actions: list[dict] | None = None,
    ) -> None:
        payload = {"topic": topic, "message": message, "priority": PRIORITY[priority]}
        if title:
            payload["title"] = title
        if tags:
            payload["tags"] = tags
        if click:
            payload["click"] = click
        if actions:
            payload["actions"] = actions
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            r = await client.post(self.base_url, json=payload, headers=headers)
            r.raise_for_status()


def send_message_executor(ntfy: Ntfy, topics: dict[str, str]) -> Callable[[ActionRequest], Awaitable[str]]:
    """Workspace executor for `send_message`: target is a person, params.text the message.

    `topics` maps people to their ntfy topics (config, not code), so the agent can
    only message people the household listed."""

    async def run(req: ActionRequest) -> str:
        topic = topics.get(req.target)
        if topic is None:
            raise ValueError(f"no ntfy topic for {req.target}")
        await ntfy.publish(topic, req.params.get("text", req.reason), title="Message from Atlas", tags=["speech_balloon"])
        return f"sent to {req.target}"

    return run


class PhoneNotifier:
    """Bus handlers that push alerts, nudges and approval cards to the phones."""

    def __init__(self, ntfy: Ntfy, topic: str = "atlas-alerts", gateway_url: str = "http://127.0.0.1:8000") -> None:
        self.ntfy = ntfy
        self.topic = topic
        self.gateway_url = gateway_url.rstrip("/")

    async def on_alert(self, e: AlertRaised) -> None:
        await self.ntfy.publish(self.topic, e.message, title="Atlas", priority="high", tags=["rotating_light"])

    async def on_nudge(self, e: NudgeRaised) -> None:
        tag = "tada" if e.tone == "celebrate" else "seedling"
        await self.ntfy.publish(self.topic, e.message, title="Atlas", tags=[tag])

    async def on_plan(self, e: PlanCreated) -> None:
        link = next((c.url for c in e.citations if c.url), None)
        await self.ntfy.publish(self.topic, e.text, title=e.subject, tags=["clipboard"], click=link)

    async def on_approval(self, e: ApprovalRequested) -> None:
        url = f"{self.gateway_url}/api/approvals/{e.action_id}"
        actions = [
            {"action": "http", "label": label, "url": url, "method": "POST",
             "headers": {"Content-Type": "application/json"}, "body": f'{{"decision": "{decision}"}}', "clear": True}
            for label, decision in (("Approve", "approve"), ("Deny", "deny"))
        ]
        await self.ntfy.publish(
            self.topic,
            f"{e.reason}\n({e.kind}: {e.target})",
            title="Atlas wants to act",
            priority="high",
            tags=["raised_hand"],
            actions=actions,
        )
