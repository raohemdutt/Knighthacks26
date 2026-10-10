"""GroupMe (K4) as an MCP connector: python -m contexts.connectors.servers.groupme

Each group in `options.groups` has one bot, and a bot can only post into the
group it was created in.

Write: `post(text, group, speaker)` sends one message through that group's
bot. With a speaker it is sent as "Maya: ...", since every bot post shows the
bot's name in GroupMe.

Read: `list_messages()` returns recent messages from every group as items.
With `GROUPME_ACCESS_TOKEN` it reads the live groups. Without it, it reads the
local log of what our own bots posted, since bots cannot read; that covers a
demo group where every speaker is posted by our bot.

Seed a demo conversation (a real post, so run it on purpose):
    python -m contexts.connectors.servers.groupme seed fixtures/groupme/class_chat.yaml

options:
  state_dir: data/connectors/groupme
  limit: 50
  groups:
    - {key: class, name: "COP4533 group chat", bot_env: GROUPME_BOT_CLASS, group_id: "118084819"}
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

API = "https://api.groupme.com/v3"
MAX_TEXT = 1000  # GroupMe's limit for a bot post
TITLE_LEN = 80
LOG_NAME = "posted.jsonl"
DEFAULT_STATE_DIR = "data/connectors/groupme"


def find_group(options: dict, key: str | None) -> dict:
    groups = options.get("groups", [])
    if not groups:
        raise RuntimeError("no groups in connectors.yaml under groupme.options.groups")
    if key is None:
        return groups[0]
    for group in groups:
        if group["key"] == key:
            return group
    raise RuntimeError(f"unknown group {key!r}; known: {[g['key'] for g in groups]}")


def post_message(
    options: dict, text: str, group: str | None, speaker: str | None, send, now: datetime, env=os.environ
) -> dict:
    target = find_group(options, group)
    bot_id = env.get(target["bot_env"])
    if not bot_id:
        raise RuntimeError(f"set {target['bot_env']} in .env")
    send(bot_id, (f"{speaker}: {text}" if speaker else text)[:MAX_TEXT])
    entry = {
        "id": f"local-{now.timestamp():.3f}",
        "group": target["key"],
        "name": speaker or "Atlas",
        "text": text,
        "created_at": int(now.timestamp()),
    }
    state_dir = Path(options.get("state_dir", DEFAULT_STATE_DIR))
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / LOG_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def split_speaker(name: str, text: str, bot: bool) -> tuple[str, str]:
    """Our bot posts "Maya: hi"; read it back as Maya saying "hi"."""
    speaker, sep, rest = text.partition(": ")
    if bot and sep and rest and len(speaker) <= 30 and " " not in speaker.strip():
        return speaker.strip(), rest
    return name, text


def to_items(messages: list[dict], group: dict) -> list[dict]:
    """GroupMe messages as connector items, oldest first. System and empty messages are dropped."""
    items = []
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text or m.get("system"):
            continue
        name, text = split_speaker(m.get("name") or "someone", text, m.get("sender_type") == "bot")
        items.append({
            "external_id": f"{group['key']}:{m['id']}",
            "kind": "message",
            "title": f"{name}: {text}"[:TITLE_LEN],
            "when": datetime.fromtimestamp(m["created_at"], UTC).isoformat(),
            "summary": f"{name} in {group.get('name', group['key'])}: {text}",
        })
    items.sort(key=lambda i: i["when"])
    return items


def read_log(state_dir: Path, group_key: str, limit: int) -> list[dict]:
    log = state_dir / LOG_NAME
    if not log.exists():
        return []
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    return [e for e in entries if e.get("group") == group_key][-limit:]


def read_messages(options: dict, fetch, env=os.environ) -> list[dict]:
    limit = options.get("limit", 50)
    token = env.get("GROUPME_ACCESS_TOKEN")
    state_dir = Path(options.get("state_dir", DEFAULT_STATE_DIR))
    items: list[dict] = []
    for group in options.get("groups", []):
        if token and group.get("group_id"):
            items += to_items(fetch(group["group_id"], token, limit), group)
        else:
            items += to_items(read_log(state_dir, group["key"], limit), group)
    return items


def seed(options: dict, script_path: Path, send, sleep=time.sleep, env=os.environ) -> int:
    """Post a scripted conversation through a group's bot. Returns how many messages were sent."""
    script = yaml.safe_load(script_path.read_text(encoding="utf-8"))
    gap = script.get("gap_s", 1.5)
    for i, line in enumerate(script["messages"]):
        if i:
            sleep(gap)  # keeps GroupMe from reordering or rate-limiting the burst
        post_message(options, line["text"], script.get("group"), line.get("speaker"), send, datetime.now(UTC), env)
    return len(script["messages"])


def _send(bot_id: str, text: str) -> None:
    r = httpx.post(f"{API}/bots/post", json={"bot_id": bot_id, "text": text}, timeout=20)
    r.raise_for_status()


def _fetch(group_id: str, token: str, limit: int) -> list[dict]:
    r = httpx.get(f"{API}/groups/{group_id}/messages", params={"token": token, "limit": min(limit, 100)}, timeout=20)
    r.raise_for_status()
    return r.json()["response"]["messages"]


mcp = FastMCP("atlas-groupme")


@mcp.tool()
def list_messages() -> list[dict]:
    """Recent messages from every configured group chat, oldest first per group."""
    return read_messages(server_options(), _fetch)


@mcp.tool()
def post(text: str, group: str | None = None, speaker: str | None = None) -> str:
    """Send a message to a group chat. A write: only ever run after an approval."""
    entry = post_message(server_options(), text, group, speaker, _send, datetime.now(UTC))
    return f"sent to {entry['group']}: {text}"


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        from contexts.connectors.domain.spec import load_specs

        spec = next(s for s in load_specs() if s.name == "groupme")
        env = dict(os.environ)
        if Path(".env").exists():  # run by hand, outside the app that normally loads .env
            for line in Path(".env").read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and not key.startswith("#"):
                    env.setdefault(key.strip(), value.strip())
        print(f"posted {seed(spec.options, Path(sys.argv[2]), _send, env=env)} messages")
    else:
        mcp.run()
