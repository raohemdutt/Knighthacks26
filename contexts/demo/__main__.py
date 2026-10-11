"""The demo event feed.

    python -m contexts.demo seed [--data-dir data] [--now | --end ISO] [--force] [--history] [--people]
        The whole demo night, published through the real app: every projection fills from the log.
        --history first lives the twelve days before it through the Day loop (plans, re-plans, reviews), so
        Today's Plan has earlier days on its date strip, three of them the old presenter presets.
        --people adds demo people for the People screen (made-up vectors marked as demo data, never a face) and a
        week of the owner's looks; nothing of the story changes.
        Then run the app on it:
            ATLAS_DATA_DIR=data ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway

    python -m contexts.demo circle --data-dir <dir> [--config-dir <dir>] [--end ISO] [--force]
        The same night plus a Circle of three on this one box, no network: names, places shared, two plans
        agreed (one after a counter-offer) and one still open, and three demo people for the People screen (made-up
        vectors marked as demo data: contexts/demo/people.py). It writes its own config directory (default
        <data-dir>/config; the committed config never names a member) and prints the command to run the app on it.

    python -m contexts.demo neon --circle <id> [--name "<name>"] [--members a,b] [--days 7] [--end YYYY-MM-DD]
                                 [--dry-run | --apply] [--remove] [--into-existing]
        Demo members in the SHARED database (Neon), written through the classes a box writes with: a week of
        habit summaries, Waste Watch counts and Sort It scores each, two agreed plans, the League's last week.
        The default is a dry run that prints every row; --remove deletes exactly what --apply wrote
        (contexts/demo/neon_seed.py).

    python -m contexts.demo checkin --data-dir <empty dir> [--end ISO] [--onto]
        The check-in (D4): one invented afternoon (a short night, resting heart rate up, strained looks, a bottle
        not picked up, three earlier days like it) published through the real app, so the Day loop's own rule
        raises it and the narrator says it. An earlier strained afternoon is in its log too, so the look memory
        calls back to it ("That's the same look you had <that weekday> afternoon, when ..."). Prints the sentence
        and the command to run the app on it. --onto adds the afternoon to a directory already seeded with
        `python -m contexts.demo circle`, so one box shows the night, the Circle, the people and the check-in.

    python -m contexts.demo optional --data-dir <empty dir> [--day YYYY-MM-DD]
        An optional event ("Near you"), suggested and said yes to: one invented day published through the real
        app. The morning plan is approved locally, Atlas says when the afternoon's study block starts, suggests
        the friends' dinner, the yes raises a re-plan and its card, that card is approved locally, the dinner is
        a block of the day and Atlas says when to leave. Prints each line (the same for the same day, whatever
        the directory) and, on stderr, the command to run the app on it.

    python -m contexts.demo fake <context> [--data-dir data] [--now]
        A few realistic events of every type one context publishes, appended to the log.

    python -m contexts.demo preflight --gateway http://<box>:8000 [--token $ATLAS_TOKEN] [--reset]
        The T-15 checklist (D9): PASS/FAIL with the fix for each, plus a warm-up question and its time.
        --reset first clears alert cooldowns and leftover approval cards (memory is untouched).

    python -m contexts.demo kit --gateway http://<box>:8000 [--ntfy URL] [--topic atlas] [--devpost URL] [--out data/kit]
        The demo staging kit (D6): the physical checklist with the beat each item serves, then the software half
        checked against the box (app served and installable, ntfy and its topic, preflight); --devpost writes a
        printable QR card to --out.

    python -m contexts.demo fake perception --gateway http://<box>:8000 [--token $ATLAS_TOKEN] [--realtime]
        The camera scene pushed into a running box (POST /api/events), timed from now: its own zone
        engine, rules, Sort It and Waste Watch react live, as if the camera saw it.
        Contexts: perception connectors memory habits impact agent circle vault play day

    python -m contexts.demo video check <file> | chapters | cards | json [out]
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
from datetime import UTC, date, datetime
from pathlib import Path

from contexts.demo import fakes, neon_seed
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
    s.add_argument("--history", action="store_true", help="also the twelve days before the story, lived through the Day loop")
    s.add_argument("--people", action="store_true", help="also demo people for the People screen (made-up vectors, marked as demo data)")
    c = sub.add_parser("circle", help="the demo night plus a Circle of three, on one box")
    c.add_argument("--data-dir", type=Path, required=True)
    c.add_argument("--config-dir", type=Path, help="where to write the demo's config (default: <data-dir>/config)")
    c.add_argument("--end", type=datetime.fromisoformat, help="end the story at this time (with a UTC offset) instead of now")
    c.add_argument("--force", action="store_true", help="add to a data directory that already has an event log")
    n = sub.add_parser("neon", help="demo members in the shared database: leaderboards, plans, the League")
    neon_seed.add_arguments(n)
    k = sub.add_parser("checkin", help="an invented afternoon that ends in a check-in, through the real app (D4)")
    k.add_argument("--data-dir", type=Path, required=True)
    k.add_argument("--end", type=datetime.fromisoformat, help="when the check-in is raised (with a UTC offset); default: now in "
                                                             "the afternoon, else today 15:40 local")
    k.add_argument("--onto", action="store_true", help="add the afternoon to a directory seeded with `python -m contexts.demo circle`")
    o = sub.add_parser("optional", help="an invented afternoon: an optional event suggested, said yes to and approved")
    o.add_argument("--data-dir", type=Path, required=True)
    o.add_argument("--day", type=date.fromisoformat, help="the invented day (default: today, local)")
    f = sub.add_parser("fake", help="publish a small sample of one context's events")
    f.add_argument("context", choices=fakes.CONTEXTS)
    f.add_argument("--data-dir", type=Path, default=Path("data"))
    f.add_argument("--now", action="store_true", help="start at the current time instead of the demo night's")
    f.add_argument("--gateway", help="push into a running box instead of writing the log, e.g. http://192.168.1.20:8000")
    f.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"), help="the box's ATLAS_TOKEN, if it has one")
    f.add_argument("--realtime", action="store_true", help="with --gateway: push each second of the scene as it happens")
    pf = sub.add_parser("preflight", help="check the box is ready for the demo (D9)")
    pf.add_argument("--gateway", default="http://127.0.0.1:8000")
    pf.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"))
    pf.add_argument("--reset", action="store_true", help="clear cooldowns and leftover approval cards first")
    pf.add_argument("--question", default="Where are my keys?", help="the warm-up question")
    kt = sub.add_parser("kit", help="the demo staging kit: physical checklist and software checks (D6)")
    kt.add_argument("--gateway", default="http://127.0.0.1:8000")
    kt.add_argument("--ntfy", default=os.environ.get("NTFY_URL"), help="default: NTFY_URL, else the gateway's host on :8090")
    kt.add_argument("--topic", default=os.environ.get("NTFY_TOPIC", "atlas"))
    kt.add_argument("--devpost", default=os.environ.get("DEVPOST_URL"), help="the project page; writes the printable QR card")
    kt.add_argument("--out", type=Path, default=Path("data/kit"), help="where the QR card goes (gitignored)")
    kt.add_argument("--list", action="store_true", help="only print the physical checklist")
    v = sub.add_parser("video", help="the submission video's number-key layout (D10)")
    v.add_argument("action", choices=["check", "chapters", "cards", "json"])
    v.add_argument("file", nargs="?", type=Path, help="check: the exported video; cards: where to write the HTML")
    t = sub.add_parser("totals", help="the hackathon in numbers, from the event log (D11)")
    t.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR", "data")))
    t.add_argument("--since", type=datetime.fromisoformat, help="count only from this time (with a UTC offset)")
    t.add_argument("--json", action="store_true", help="JSON instead of a markdown table")
    args = p.parse_args(argv)

    if args.command == "neon":
        return neon_seed.main(args)

    if args.command == "totals":
        from contexts.demo import totals

        if args.since and args.since.tzinfo is None:
            p.error("--since needs a UTC offset, e.g. 2026-10-10T09:00:00-04:00")
        rows = totals.totals(totals.read_events(args.data_dir / "events"), since=args.since)
        print(totals.as_json(rows) if args.json else totals.table(rows))
        return 0

    if args.command == "video":
        return video_command(args.action, args.file)

    if args.command == "kit":
        from contexts.demo import kit

        print(kit.checklist())
        if args.list:
            return 0
        checks = kit.software_checks(args.gateway, args.ntfy, args.topic, args.devpost, args.out)
        print()
        print(kit.report(checks))
        return 0 if all(c.ok for c in checks) else 1

    if args.command == "preflight":
        return preflight(args.gateway, args.token, args.reset, args.question)

    if args.command == "checkin":
        from contexts.demo import checkin

        if args.end and args.end.tzinfo is None:
            p.error("--end needs a UTC offset, e.g. 2026-10-10T15:40:00-04:00")
        try:
            now = datetime.now(UTC)
            end = args.end or (checkin.onto_end(args.data_dir, now) if args.onto else checkin.default_end(now))
            shown = asyncio.run(checkin.show(args.data_dir, end, onto=args.onto))
        except SeedRefused as refused:
            print(refused, file=sys.stderr)
            return 1
        print(checkin.summary(shown))
        return 0 if shown.checkins else 1

    if args.command == "optional":
        from contexts.demo import optional

        try:
            shown = asyncio.run(optional.show(args.data_dir, args.day or optional.default_day(datetime.now(UTC))))
        except SeedRefused as refused:
            print(refused, file=sys.stderr)
            return 1
        print("\n".join(optional.lines(shown)))  # the same lines for the same day, whatever the directory
        print("\n" + optional.run_it(shown), file=sys.stderr)  # the one part that names the directory
        return 0 if shown.ok else 1

    if args.command == "circle":
        from contexts.demo import circle

        if args.end and args.end.tzinfo is None:
            p.error("--end needs a UTC offset, e.g. 2026-10-11T09:30:00-04:00")
        # the Circle is about what is coming up, so the story ends now unless told otherwise
        end = args.end or datetime.now(UTC).replace(microsecond=0)
        cfg = circle.config_dir(args.config_dir or args.data_dir / "config")
        try:
            seeded = asyncio.run(seed(args.data_dir, now=end, force=args.force, config_dir=cfg, extra=circle.circle_events,
                                     history=True, people=True))
        except SeedRefused as refused:
            print(refused, file=sys.stderr)
            return 1
        print(summary(seeded))
        print(f"\nCircle {circle.CIRCLE_ID}: {', '.join(circle.NAMES.values())}; this box is {circle.NAMES[circle.ME]}'s.")
        print("Run the app on it (no network, no second machine), then open /circle/:")
        print(f"  ATLAS_DATA_DIR={args.data_dir} ATLAS_CONFIG_DIR={cfg} ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 "
              "python -m contexts.gateway")
        return 0

    if args.command == "seed":
        if args.end and args.end.tzinfo is None:
            p.error("--end needs a UTC offset, e.g. 2026-10-11T09:30:00-04:00")
        end = datetime.now(UTC).replace(microsecond=0) if args.now else args.end
        try:
            seeded = asyncio.run(seed(args.data_dir, now=end, force=args.force, history=args.history, people=args.people))
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
        return push(args.context, args.gateway, args.token, realtime=args.realtime)
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
    if action == "json":
        import json

        text = json.dumps(video.timeline(), indent=2)
        if file:
            file.write_text(text + "\n", encoding="utf-8")
            print(f"wrote the timeline to {file}")
        else:
            print(text)
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


def push(context: str, gateway: str, token: str | None, realtime: bool = False) -> int:
    """The context's fake, timed from now, POSTed to a running gateway. Only input the box accepts is sent;
    zone events and the rest are what the box makes of it. `realtime` sends each second when its time comes,
    so the room view and Sort It play out live instead of arriving all at once."""
    import tempfile
    import time

    import httpx

    from contexts.gateway.app import INPUT_EVENTS

    with tempfile.TemporaryDirectory() as tmp:
        events = asyncio.run(fakes.publish(context, tmp, datetime.now(UTC).replace(microsecond=0)))
    sent = [e for e in events if e.type.startswith(INPUT_EVENTS)]
    if not sent:
        print(f"{context} has no camera or connector input to push", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    sent.sort(key=lambda e: e.ts)
    batches = [sent]
    if realtime:  # one request per second of the scene
        by_second: dict = {}
        for e in sent:
            by_second.setdefault(e.ts.replace(microsecond=0), []).append(e)
        batches = [by_second[s] for s in sorted(by_second)]
    published, types = 0, set()
    for batch in batches:
        if realtime:
            time.sleep(max(0.0, (batch[0].ts - datetime.now(UTC)).total_seconds()))
        body = [e.model_dump(mode="json", exclude={"event_id"}) for e in batch]
        r = httpx.post(gateway.rstrip("/") + "/api/events", json=body, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"{r.status_code}: {r.text}", file=sys.stderr)
            return 1
        published += r.json()["published"]
        types |= set(r.json()["types"])
    print(f"pushed {published} events into {gateway}: {', '.join(sorted(types))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
