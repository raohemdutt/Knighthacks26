"""The demo event feed.

    python -m contexts.demo seed [--data-dir data] [--now | --end ISO] [--force]
        The whole demo night, published through the real app: every projection fills from the log.
        Then run the app on it:
            ATLAS_DATA_DIR=data ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway

    python -m contexts.demo fake <context> [--data-dir data] [--now]
        A few realistic events of every type one context publishes, appended to the log.

    python -m contexts.demo preflight --gateway http://<box>:8000 [--token $ATLAS_TOKEN] [--reset]
        The T-15 checklist (D9): PASS/FAIL with the fix for each, plus a warm-up question and its time.
        --reset first clears alert cooldowns and leftover approval cards (memory is untouched).

    python -m contexts.demo fake perception --gateway http://<box>:8000 [--token $ATLAS_TOKEN]
        The camera scene pushed into a running box (POST /api/events), timed from now: its own zone
        engine, rules, Sort It and Waste Watch react live, as if the camera saw it.
        Contexts: perception connectors memory habits impact agent circle vault play day

    python -m contexts.demo video check <file> | chapters | cards
        The submission video's 10:00 number-key layout (D10): check an export's length and where each
        key lands, print the YouTube description, or write the title cards (HTML).

    python -m contexts.demo totals [--data-dir data] [--since ISO] [--json]
        The hackathon in numbers from the box's own event log (D11), each with the events it was
        counted from, for the video's closing card and the Devpost.
"""

import argparse
import os
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from contexts.demo import fakes
from contexts.demo.seed import SeedRefused, seed, summary
from contexts.demo.story import load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m contexts.demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed", help="publish the demo night through the real app")
    s.add_argument("--data-dir", type=Path, default=Path("data"))
    s.add_argument("--now", action="store_true", help="move the story so it ends at the current time")
    s.add_argument("--end", type=datetime.fromisoformat, help="move the story so it ends at this time (with a UTC offset)")
    s.add_argument("--force", action="store_true", help="add to a data directory that already has an event log")
    f = sub.add_parser("fake", help="publish a small sample of one context's events")
    f.add_argument("context", choices=fakes.CONTEXTS)
    f.add_argument("--data-dir", type=Path, default=Path("data"))
    f.add_argument("--now", action="store_true", help="start at the current time instead of the demo night's")
    f.add_argument("--gateway", help="push into a running box instead of writing the log, e.g. http://192.168.1.20:8000")
    f.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"), help="the box's ATLAS_TOKEN, if it has one")
    pf = sub.add_parser("preflight", help="check the box is ready for the demo (D9)")
    pf.add_argument("--gateway", default="http://127.0.0.1:8000")
    pf.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"))
    pf.add_argument("--reset", action="store_true", help="clear cooldowns and leftover approval cards first")
    pf.add_argument("--question", default="Where are my keys?", help="the warm-up question")
    v = sub.add_parser("video", help="the submission video's number-key layout (D10)")
    v.add_argument("action", choices=["check", "chapters", "cards"])
    v.add_argument("file", nargs="?", type=Path, help="check: the exported video; cards: where to write the HTML")
    t = sub.add_parser("totals", help="the hackathon in numbers, from the event log (D11)")
    t.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR", "data")))
    t.add_argument("--since", type=datetime.fromisoformat, help="count only from this time (with a UTC offset)")
    t.add_argument("--json", action="store_true", help="JSON instead of a markdown table")
    args = p.parse_args(argv)

    if args.command == "totals":
        from contexts.demo import totals

        if args.since and args.since.tzinfo is None:
            p.error("--since needs a UTC offset, e.g. 2026-10-10T09:00:00-04:00")
        rows = totals.totals(totals.read_events(args.data_dir / "events"), since=args.since)
        print(totals.as_json(rows) if args.json else totals.table(rows))
        return 0

    if args.command == "video":
        return video_command(args.action, args.file)

    if args.command == "preflight":
        return preflight(args.gateway, args.token, args.reset, args.question)

    if args.command == "seed":
        if args.end and args.end.tzinfo is None:
            p.error("--end needs a UTC offset, e.g. 2026-10-11T09:30:00-04:00")
        end = datetime.now(UTC).replace(microsecond=0) if args.now else args.end
        try:
            seeded = asyncio.run(seed(args.data_dir, now=end, force=args.force))
        except SeedRefused as refused:
            print(refused, file=sys.stderr)
            return 1
        print(summary(seeded))
        moved = seeded.now - load().now
        if moved.total_seconds() % 86400:
            print("\nNote: the story was moved by a part of a day, so its nights and meals no longer fall on the same local "
                  "days and the habit numbers differ from the script's. Use --end with a 09:30 local time to keep them.")
        return 0

    if args.gateway:
        return push(args.context, args.gateway, args.token)
    start = datetime.now(UTC).replace(microsecond=0) if args.now else load().now
    published = asyncio.run(fakes.publish(args.context, args.data_dir / "events", start))
    for e in published:
        print(f"{e.ts.isoformat()}  {e.type}")
    print(f"published {len(published)} {args.context} events into {args.data_dir / 'events'}")
    return 0


WARM_ANSWER_S = 5.0  # a warmed on-device model answers a prefetch question well under this


def preflight(gateway: str, token: str | None, do_reset: bool, question: str) -> int:
    import time

    import httpx

    base, headers = gateway.rstrip("/"), ({"Authorization": f"Bearer {token}"} if token else {})
    try:
        if do_reset:
            r = httpx.post(f"{base}/api/preflight/reset", headers=headers, timeout=10).raise_for_status().json()
            print(f"reset: {r['cooldowns_cleared']} cooldowns cleared, {r['cards_denied']} cards denied")
        checks = httpx.get(f"{base}/api/preflight", timeout=30).raise_for_status().json()["checks"]
        start = time.monotonic()
        a = httpx.post(f"{base}/api/ask", json={"text": question}, headers=headers, timeout=60).raise_for_status().json()
        took = time.monotonic() - start
    except httpx.HTTPError as exc:
        print(f"FAIL  gateway                    {base} did not answer: {exc}")
        return 1
    warm = took <= WARM_ANSWER_S and bool(a.get("answer"))
    checks.append({"name": "warm-up answer", "ok": warm,
                   "detail": f"{took:.1f} s: {' '.join(a.get('answer', '').split())[:70]}",
                   "fix": "" if warm else "ask it again to warm the model; if still slow, check `journalctl -u atlas-llm`"})
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<30} {c['detail']}" + (f"\n      fix: {c['fix']}" if c["fix"] else ""))
    failed = [c for c in checks if not c["ok"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} ready" + (" - go." if not failed else ""))
    return 1 if failed else 0


def video_command(action: str, file: Path | None) -> int:
    from contexts.demo import video

    if action == "chapters":
        print(video.chapters())
        return 0
    if action == "cards":
        out = file or Path("video_cards.html")
        out.write_text(video.cards(), encoding="utf-8")
        print(f"wrote {len(video.SLOTS)} title cards to {out}")
        return 0
    if file is None:
        print("video check needs the exported file", file=sys.stderr)
        return 2
    ok, text = video.report(video.probe(file))
    print(text)
    return 0 if ok else 1


def push(context: str, gateway: str, token: str | None) -> int:
    """The context's fake, timed from now, POSTed to a running gateway. Only input the box accepts is sent;
    zone events and the rest are what the box makes of it."""
    import tempfile

    import httpx

    from contexts.gateway.app import INPUT_EVENTS

    with tempfile.TemporaryDirectory() as tmp:
        events = asyncio.run(fakes.publish(context, tmp, datetime.now(UTC).replace(microsecond=0)))
    sent = [e for e in events if e.type.startswith(INPUT_EVENTS)]
    if not sent:
        print(f"{context} has no camera or connector input to push", file=sys.stderr)
        return 1
    body = [e.model_dump(mode="json", exclude={"event_id"}) for e in sent]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(gateway.rstrip("/") + "/api/events", json=body, headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"{r.status_code}: {r.text}", file=sys.stderr)
        return 1
    print(f"pushed {r.json()['published']} events into {gateway}: {', '.join(r.json()['types'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
