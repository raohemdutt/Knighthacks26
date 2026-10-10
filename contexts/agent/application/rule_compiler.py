"""Plain English to an alert rule (A3).

"Tell me if a package sits at the door for more than 10 seconds" becomes
{label: [package, box], zone: door, kind: dwell, seconds: 10}. The on-device
model drafts the JSON under structured output (the Rule schema, with the zone
limited to the known zones, so the model cannot produce anything else), and
code still validates it, so a bad draft is rejected instead of silently never firing.
"""

import asyncio
import json
import re

from pydantic import ValidationError

from contexts.agent.domain.rules import Rule
from contracts.types import ChatMessage
from runtime.interface import Runtime

SYSTEM = """Turn the user's request into one alert rule as JSON, and reply with only the JSON.
Schema: {{"id": "kebab-case-name", "when": {{"label": ["detector labels"], "zone": "one of {zones}",
"kind": "enter" | "exit" | "dwell", "seconds": number (only for dwell)}}, "notify": "short message",
"cooldown_s": number}}
Detector labels are COCO names such as person, backpack, suitcase, bottle, cup, laptop, cell phone;
a package is usually detected as "suitcase", "backpack" or "box", so list several."""


class RuleCompileError(ValueError):
    pass


def _json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuleCompileError("the model did not return JSON")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise RuleCompileError(f"bad JSON: {exc}") from exc


async def compile_rule(text: str, runtime: Runtime, zones: list[str]) -> Rule:
    system = SYSTEM.format(zones=", ".join(zones))
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=text)]
    schema = Rule.model_json_schema()
    schema["$defs"]["Condition"]["properties"]["zone"] = {"type": "string", "enum": zones}
    reply = await asyncio.to_thread(lambda: runtime.chat(messages, None, json_schema=schema))
    try:
        rule = Rule.model_validate(_json_object(reply.content))
    except ValidationError as exc:
        raise RuleCompileError(f"not a valid rule: {exc.errors()[0]['msg']}") from exc
    if rule.when.zone not in zones:
        raise RuleCompileError(f"unknown zone {rule.when.zone!r}; known zones: {', '.join(zones)}")
    return rule
