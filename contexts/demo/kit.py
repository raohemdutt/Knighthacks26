"""The demo staging kit (D6): what has to be on the table, and a check of the parts software can see.

    python -m contexts.demo kit --gateway http://<box>:8000 [--ntfy http://<box>:8090] [--topic atlas]
                                [--devpost https://devpost.com/software/...] [--out kit/]

The physical half (keychain, package, tape, mounts, cables, power, hotspot, phones, printed QR) is a list a
person ticks; it prints with the beat each item serves. The software half is checked against the running box:
the phone app is served and installable, ntfy answers and the alert topic can be read, and the box's own
preflight (D9) passes. With --devpost it writes a printable QR card (HTML and SVG) to --out.

Nothing here sends a notification or changes the box: every check is a GET.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class Item:
    name: str
    beat: str  # the Demo Plan beat that needs it
    check: str  # what "ready" looks like at T-15
    required: bool = True  # False: in the Demo Plan, not in D6's "Done when"


KIT: tuple[Item, ...] = (
    Item("Bright keychain", "Where are my keys? (1:05)",
         "On the desk in view of cam-desk at least 2 minutes before the run, not moved since."),
    Item("Cardboard 'package'", "It watches for you (1:35) and Unplug the internet (3:15)",
         "Out of frame, within the Stagehand's reach; a box the detector calls a package (try it once at T-15)."),
    Item("Tape-marked door zone", "It watches for you (1:35)",
         "Tape on the floor matches the Door zone drawn on cam-door in /live/; the package fits inside it."),
    Item("Camera mounts / tripods", "Hook (0:00) and every camera beat",
         "cam-desk sees the desk, keys and laptop; cam-door sees the taped zone; both tiles live in /live/."),
    Item("Power strip", "everything",
         "Jetson, both cameras, speaker and phone chargers on one strip, plugged into a live outlet."),
    Item("Long USB cables", "Hook (0:00) and every camera beat",
         "Both cameras reach the Jetson without pulling on the mounts; cables taped down where people walk."),
    Item("Phone hotspot", "Unplug the internet (3:15), and the backup if venue Wi-Fi fails",
         "Hotspot on, the Jetson joins it, the box answers on its LAN address from the demo phone."),
    Item("2 charged demo phones (ntfy + PWA)", "It watches for you (1:35), It learns (2:05), Unplug (3:15)",
         "Above 80%, Do Not Disturb off, ntfy subscribed to the alert topic on the box, Atlas on the home screen, "
         "Tailscale and LAN addresses bookmarked."),
    Item("Printed QR to the Devpost", "Close",
         "Printed from `kit --devpost <url>`, taped to the table, scans to the project page on a phone."),
    Item("Table sign: 'Saved pictures are always blurred. Nothing with a face leaves this box. Only your own signed-in screen shows you.'",
         "the whole table", "Visible from the aisle.",
         required=False),
    Item("Empty can (Waste Watch)", "v2: Waste Watch", "Within reach; the Recycling and Trash zones drawn.", required=False),
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def checklist(items: tuple[Item, ...] = KIT) -> str:
    """The physical half, as a markdown checklist a person can print or paste into Notion."""
    lines = ["Physical kit (tick each at T-15):"]
    for it in items:
        tag = "" if it.required else " (Demo Plan extra)"
        lines.append(f"  [ ] {it.name}{tag} - {it.beat}\n        ready: {it.check}")
    return "\n".join(lines)


def default_ntfy(gateway: str) -> str:
    """ntfy runs on the box beside the gateway, on port 8090 (X7)."""
    parts = urlsplit(gateway)
    host = parts.hostname or "127.0.0.1"
    return f"http://{host}:8090"


def check_pwa(client: httpx.Client, base: str) -> list[Check]:
    out: list[Check] = []
    try:
        r = client.get(f"{base}/")
        page = r.status_code == 200 and "<html" in r.text.lower()
        out.append(Check("phone app served", page, f"GET / -> {r.status_code}",
                         "" if page else "build the app (`cd web && npm run build`) or set ATLAS_WEB_DIR to web/out"))
    except httpx.HTTPError as exc:
        return [Check("phone app served", False, f"{base} did not answer: {exc}", "start the gateway: python -m contexts.gateway")]
    try:
        m = client.get(f"{base}/manifest.webmanifest")
        name = m.json().get("name") if m.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        name = None
    out.append(Check("app installable (manifest)", bool(name), f"manifest name: {name!r}" if name else "no manifest.webmanifest",
                     "" if name else "rebuild web/ so out/manifest.webmanifest exists"))
    secure = base.startswith("https://")
    out.append(Check("https for install and mic", secure, base,
                     "" if secure else "Add to Home Screen and the mic need https on a phone: run the gateway with ATLAS_HTTPS=1"))
    return out


def check_ntfy(client: httpx.Client, ntfy: str, topic: str) -> list[Check]:
    try:
        h = client.get(f"{ntfy}/v1/health")
        healthy = h.status_code == 200 and bool(h.json().get("healthy"))
    except (httpx.HTTPError, ValueError) as exc:
        return [Check("ntfy answers", False, f"{ntfy} did not answer: {exc}", "start it on the box: sudo systemctl start ntfy")]
    out = [Check("ntfy answers", healthy, f"{ntfy}/v1/health", "" if healthy else "ntfy is up but unhealthy: journalctl -u ntfy")]
    try:  # poll=1 returns the cached messages and closes: a read, nothing is sent
        t = client.get(f"{ntfy}/{topic}/json", params={"poll": "1", "since": "10m"})
        readable = t.status_code == 200
        detail = f"topic {topic!r}: {t.status_code}"
    except httpx.HTTPError as exc:
        readable, detail = False, f"topic {topic!r}: {exc}"
    out.append(Check("alert topic readable", readable, detail,
                     "" if readable else f"check NTFY_TOPIC on the box and subscribe both phones to {topic!r}"))
    return out


def check_preflight(client: httpx.Client, base: str) -> Check:
    try:
        checks = client.get(f"{base}/api/preflight").raise_for_status().json()["checks"]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return Check("box preflight (D9)", False, f"no /api/preflight: {exc}", "update the box to current main")
    failed = [c["name"] for c in checks if not c.get("ok")]
    return Check("box preflight (D9)", not failed, f"{len(checks) - len(failed)}/{len(checks)} pass" +
                 (f"; failing: {', '.join(failed)}" if failed else ""),
                 "" if not failed else "run `python -m contexts.demo preflight --gateway <box> --reset` for each fix")


DEVPOST = re.compile(r"^https://([a-z0-9-]+\.)?devpost\.com/\S+$")


def check_devpost(url: str | None) -> Check:
    if not url:
        return Check("Devpost link", False, "no --devpost URL given", "pass --devpost <project page URL> once the page exists")
    ok = bool(DEVPOST.match(url))
    return Check("Devpost link", ok, url, "" if ok else "expected https://devpost.com/software/<project> or a *.devpost.com page")


def qr_svg(url: str) -> str:
    """An SVG QR code for the URL. Needs the `qrcode` package (pure Python; no Pillow for SVG)."""
    import io

    import qrcode
    import qrcode.image.svg

    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=20, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")


def qr_card(url: str, svg: str, title: str = "Atlas") -> str:
    """A one-page printable card: the QR, the project name and the URL under it."""
    body = svg[svg.index("<svg"):]  # drop the XML prolog so it inlines in HTML
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)} on Devpost</title>
<style>
  @page {{ size: letter; margin: 0.6in; }}
  body {{ font-family: system-ui, sans-serif; text-align: center; margin: 0; }}
  .qr svg {{ width: 5.5in; height: 5.5in; }}
  h1 {{ font-size: 44pt; margin: 0.2in 0 0; }}
  p {{ font-size: 16pt; margin: 0.1in 0; word-break: break-all; }}
</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Scan for the Devpost: the story, the video and the code.</p>
  <div class="qr">{body}</div>
  <p>{html.escape(url)}</p>
</body></html>
"""


def write_qr(url: str, out: Path) -> Check:
    try:
        svg = qr_svg(url)
    except ImportError:
        return Check("Devpost QR printed", False, "the qrcode package is not installed", "pip install qrcode, then run again")
    out.mkdir(parents=True, exist_ok=True)
    (out / "devpost-qr.svg").write_text(svg, encoding="utf-8")
    card = out / "devpost-qr.html"
    card.write_text(qr_card(url, svg), encoding="utf-8")
    return Check("Devpost QR printed", True, f"wrote {card} (open it and print one page)")


def software_checks(gateway: str, ntfy: str | None, topic: str, devpost: str | None, out: Path,
                    client: httpx.Client | None = None) -> list[Check]:
    base = gateway.rstrip("/")
    own = client is None
    client = client or httpx.Client(timeout=10, verify=False)  # the box's https is self-signed
    try:
        checks = check_pwa(client, base)
        checks += check_ntfy(client, (ntfy or default_ntfy(base)).rstrip("/"), topic)
        checks.append(check_preflight(client, base))
    finally:
        if own:
            client.close()
    link = check_devpost(devpost)
    checks.append(link)
    if link.ok and devpost:
        checks.append(write_qr(devpost, out))
    return checks


def report(checks: list[Check]) -> str:
    lines = ["Software (checked against the box):"]
    for c in checks:
        lines.append(f"  {'PASS' if c.ok else 'FAIL'}  {c.name:<28} {c.detail}" + (f"\n        fix: {c.fix}" if c.fix else ""))
    failed = sum(not c.ok for c in checks)
    lines.append(f"\n{len(checks) - failed}/{len(checks)} software checks pass; tick the physical list by hand.")
    return "\n".join(lines)
