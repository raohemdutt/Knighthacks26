"""The group brain from a terminal: which name the teacher answers to, and one question asked the way the box asks it.

    python -m contexts.circle.brain status
    python -m contexts.circle.brain ask --log-dir /tmp/brain-log --circle team --member ilan --opt-in \\
        --fact "Maya is at the Library until 9:00 PM" "Where should we meet tonight?"

For proving the path and rehearsing; on the box the same GroupBrain runs in the gateway. The pod is the one in the
pod-a role of LAB_POD_ROLES (--account names it, and must fill that role), its keys and hub read from the environment
and the .env files as lab/hub.py reads them.

status asks the teacher's server what it serves (GET /v1/models, no member text) and prints the name a question would
use, "group-brain" or "teacher". ask runs GroupBrain.ask: it writes the opt-in and every send to --log-dir as events
(circle.brain.asked before the request leaves, circle.brain.answered after), so that folder is the record of what was
sent. --log-dir is a folder of its own, never the box's live data/events, which the gateway writes. --opt-in records
the member's opt-in there (source cli) when the folder holds none that is on; without it, a member who has not opted
in is refused and nothing is sent.

One JSON line on stdout; exit 0 when the answer came (or status was read), 1 when the question was refused or not
answered, 2 for bad arguments, missing keys or an account outside the teacher's role. No token and no account name is
printed. Nothing here starts, stops or restarts a pod.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

from bus import EventBus
from contexts.circle.application.group_brain import BrainRefused, BrainUnavailable, GroupBrain
from contexts.circle.infrastructure.teacher import TeacherLink
from lab.hub import MissingCredentials, PodRoleError

REPO = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m contexts.circle.brain", description=(__doc__ or "").split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="which served name a question would use (no member text is sent)")
    ask = commands.add_parser("ask", help="ask one planning question, logged to --log-dir")
    for command in (status, ask):
        command.add_argument("--account", default=None, help="the account letter; it must fill the pod-a role")
    ask.add_argument("--log-dir", required=True, help="a folder of its own for the events; never the box's data/events")
    ask.add_argument("--circle", required=True, help="the circle id, e.g. team")
    ask.add_argument("--member", required=True, help="the member asking, e.g. ilan")
    ask.add_argument("--fact", action="append", default=[], help="one fact the plan rests on; repeat for more")
    ask.add_argument("--purpose", default="plan", help="one short word: plan, where_to_go")
    ask.add_argument("--opt-in", action="store_true", help="record the member's opt-in first if the log has none")
    ask.add_argument("question")
    return parser


def _emit(out: TextIO, payload: Mapping[str, Any]) -> None:
    out.write(json.dumps(payload, sort_keys=True) + "\n")
    out.flush()


def _is_live_log(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == (REPO / "data" / "events").resolve() or (resolved.name == "events"
                                                                 and resolved.parent.name == "data")


async def _ask(args: argparse.Namespace, link: TeacherLink, out: TextIO) -> int:
    bus = EventBus(Path(args.log_dir))
    brain = GroupBrain(args.circle, args.member, link, bus.publish)
    bus.subscribe("circle.brain.consent", brain.on_consent, replay=True)
    await bus.replay_into()
    if brain.opt_in_ref is None and args.opt_in:
        await brain.set_consent(True, "cli")
    try:
        answer = await brain.ask(args.question, args.fact, purpose=args.purpose)
    except BrainRefused as exc:
        _emit(out, {"ok": False, "kind": exc.reason, "error": str(exc)})
        return 1
    except BrainUnavailable as exc:
        _emit(out, {"ok": False, "kind": exc.kind, "error": str(exc), "ask_id": exc.ask_id})
        return 1
    except ValueError as exc:
        _emit(out, {"ok": False, "kind": "usage", "error": str(exc)})
        return 2
    _emit(out, {"ok": True, "answer": answer.text, "badge": answer.badge, "ask_id": answer.ask_id,
                "opt_in_ref": answer.opt_in_ref, "model": answer.model, "latency_ms": answer.latency_ms,
                "log_dir": str(args.log_dir)})
    return 0


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_files: Sequence[Path] | None = None, transport: httpx.BaseTransport | None = None,
         stdout: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    if args.command == "ask" and _is_live_log(Path(args.log_dir)):
        _emit(out, {"ok": False, "kind": "usage",
                    "error": "--log-dir is a folder of its own, never the box's data/events (the gateway writes it)"})
        return 2
    try:
        link = TeacherLink.from_env(letter=args.account, environ=environ, env_files=env_files, transport=transport)
    except MissingCredentials as exc:
        _emit(out, {"ok": False, "kind": "missing_keys", "error": str(exc), "missing": exc.missing})
        return 2
    except PodRoleError as exc:
        _emit(out, {"ok": False, "kind": "pod_role", "error": str(exc)})
        return 2
    try:
        if args.command == "status":
            try:
                model = link.model()
            except Exception as exc:
                _emit(out, {"ok": False, "kind": getattr(exc, "kind", type(exc).__name__), "error": str(exc)})
                return 1
            _emit(out, {"ok": True, "model": model, "node": link.node, "host": link.host, "priority": link.priority,
                        "max_tokens": link.max_tokens})
            return 0
        return asyncio.run(_ask(args, link, out))
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
