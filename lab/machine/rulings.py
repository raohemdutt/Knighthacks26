"""Owner rulings: list the open ones, record the answered ones as owner rules, mirror the waiting ones to Notion.

    python -m lab rulings list   [--all] [--track NAME ...]
    python -m lab rulings apply  [--dry-run] [--track NAME ...] [--date YYYY-MM-DD]
    python -m lab rulings mirror [--print]

Each refinement track keeps an OWNER_RULINGS.md: numbered questions the passes could not decide, each
with the default the plan follows now, the recommendation, the source of the tension and an Answer
line that starts blank (lab/refine/TEMPLATES.md). A ruling is open until the owner writes anything but
underscores after "Answer:". Every call answers with one JSON object.

list     Every open ruling of every track with its default (now), its recommendation and its tension.
         --all adds the answered ones with their answer and where they stand: new (answered, not yet
         in the log), reanswered (the answer was rewritten after the log recorded it), recorded (in
         the log, and no pass entry has followed) or done (a pass entry followed it, or the
         correction itself carried the numbered changes); unknown when the track's log cannot be read.

apply    Records each new or reanswered ruling as an owner rule by appending an owner correction to the
         track's REFINE_LOG.md: a "Rulings applied: <n>" line, the answer as written and the default
         it replaces. The correction changes no counter and no plan text. It returns, under
         next_pass_must_change, every ruling the next pass still has to make the plan follow, with
         the exact source line to cite. A ruling already in the log is not recorded twice. The
         OWNER_RULINGS.md files are never written: only the owner writes an Answer.

mirror   Posts the open rulings to Notion: one segment on the "Needs a human (Ilan)" line of Ilan's
         section on the Overnight Status page, and one signed FYI block on Ilan's Team Comms page,
         each carrying a short list id. A page that already shows the current list id is left alone;
         a changed list replaces the earlier post of this flow in place; an empty list closes it.
         Only text this flow wrote is ever replaced. It needs NOTION_TOKEN (environment or .env).
         With --print it touches nothing and returns the exact texts and where they go, for a
         session to paste through its own Notion tools; without a token it returns the same texts
         and exits 1.

The Notion boundary is NotionPages (read a page's markdown, replace an exact piece of it), which
Notion's page-markdown endpoints implement one to one. RestNotionClient is that implementation;
the tests drive the same code through a fake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from lab.machine import ready, stop
from lab.machine.lock import FileLock, LockTimeout
from lab.machine.refine_formats import FormatError, Log, OwnerCorrection, PassEntry, Ruling, parse_log, parse_rulings

__all__ = [
    "NotionError",
    "NotionPages",
    "RestNotionClient",
    "Standing",
    "apply_rulings",
    "build_posts",
    "collect",
    "main",
    "mirror_rulings",
    "notion_text",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
REFINE_DIR = REPO_ROOT / "lab" / "refine"
RULINGS_NAME = "OWNER_RULINGS.md"
LOG_NAME = "REFINE_LOG.md"
TRACK_ORDER = ("telemetry", "learning", "plumbing", "kernels")
LOCK_NAME = "rulings.lock"
LOCK_TIMEOUT_S = 10.0

# The two pages the mirror writes (Notion page ids and their links).
OVERNIGHT_PAGE = "3f596ef5-fec5-8172-a2d3-e0db86fc09a3"
OVERNIGHT_URL = "https://app.notion.com/p/3f596ef5fec58172a2d3e0db86fc09a3"
TEAM_COMMS_PAGE = "3f596ef5-fec5-8163-bbef-d0e7ea5acea7"
TEAM_COMMS_URL = "https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7"

TOKEN_ENV = "NOTION_TOKEN"
NOTION_API = "https://api.notion.com"
NOTION_VERSION = "2026-03-11"
NOTION_TIMEOUT_S = 20.0
MAX_BODY_BYTES = 8_000_000

# States of a ruling (see Standing).
OPEN, NEW, REANSWERED, RECORDED, DONE, UNKNOWN = "open", "new", "reanswered", "recorded", "done", "unknown"
MUST_CHANGE = (NEW, REANSWERED, RECORDED)

SIGNER = "\U0001F916 Ilan's agent \u00b7"
PHRASE = "Lab rulings waiting on you"
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MAX_TITLES = 6  # questions named on the Overnight Status line before "and N more"
TITLE_CHARS = 80
FIELD_CHARS = 700  # longest default, recommendation or tension quoted in the FYI block

_STAMP = re.escape(SIGNER) + r" [A-Z][a-z]{2} [0-9]{1,2}:[0-9]{2} [AP]M ET:"
_LIST_ID = r"\(list (?P<id>[0-9a-f]{8})\)"
# What this flow wrote earlier: the segment on the Needs a human line, and the first line of its FYI block.
_SEGMENT = re.compile(_STAMP + " " + re.escape(PHRASE) + r".*?" + _LIST_ID)
_FYI_FIRST = re.compile(r"^[ \t]*- " + _STAMP + " " + re.escape(PHRASE) + r"\b.*" + _LIST_ID + r"\s*$")
_NEEDS_LINE = re.compile(r"^(?P<head>[ \t]*- \*\*Needs a human \(Ilan\):\*\*)(?P<body>.*)$")
_FYI_HEADING = re.compile(r"^##\s+FYI\b.*$")
_EMPTY_BODY = frozenset({"", "-", "\u2014", "\u2013", "none", "nothing"})

_OWNER_DATE = re.compile(r"\(\s*owner\s*,\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\s*\)")
_CODE_WORD = re.compile(r"\.(?:py|md|txt|jsonl?|ya?ml|toml|sh|csv|log|cfg|ini)$")
_EDGE_PUNCTUATION = ".,;:!?()\"'"
_NOTION_MARKUP = "\\*~`$[]<>{}|^"
_ESCAPES = str.maketrans({char: "\\" + char for char in _NOTION_MARKUP})


class NotionError(Exception):
    """Notion could not be read or written; the message never carries the token."""


class NotionPages(Protocol):
    """What the mirror needs from Notion: a page's markdown, and one exact replacement in it.

    replace_text swaps `old` for `new` and fails (NotionError) unless `old` occurs in the page exactly
    once, which is how Notion's own content update behaves.
    """

    def read_page(self, page_id: str) -> str: ...

    def replace_text(self, page_id: str, old: str, new: str) -> None: ...


# (method, url, headers, body, timeout) -> (HTTP status, decoded JSON or text)
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, Any]]


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read(MAX_BODY_BYTES)
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read(MAX_BODY_BYTES)
    text = raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


class RestNotionClient:
    """NotionPages over Notion's page-markdown endpoints with an integration token.

    read_page is GET /v1/pages/{id}/markdown; replace_text is PATCH on the same path with one
    update_content operation. A page the API reports as truncated is refused, because text the mirror
    cannot see could hide an earlier post.
    """

    def __init__(self, token: str, *, transport: Transport | None = None, timeout: float = NOTION_TIMEOUT_S) -> None:
        if not token:
            raise ValueError("a Notion token is required")
        self._token = token
        self._transport = transport or urllib_transport
        self._timeout = timeout

    def __repr__(self) -> str:
        return "RestNotionClient(token=***)"

    def read_page(self, page_id: str) -> str:
        payload = self._call("GET", page_id)
        if not isinstance(payload, dict) or not isinstance(payload.get("markdown"), str):
            raise NotionError("Notion answered a page read without the page's markdown")
        if payload.get("truncated") or payload.get("unknown_block_ids"):
            raise NotionError("Notion returned only part of the page, so it cannot be edited safely")
        return payload["markdown"]

    def replace_text(self, page_id: str, old: str, new: str) -> None:
        body = {"type": "update_content", "update_content": {"content_updates": [{"old_str": old, "new_str": new}]}}
        self._call("PATCH", page_id, body)

    def _call(self, method: str, page_id: str, body: Mapping[str, Any] | None = None) -> Any:
        url = f"{NOTION_API}/v1/pages/{urllib.parse.quote(page_id, safe='')}/markdown"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
            "User-Agent": "atlas-lab-rulings",
        }
        data = json.dumps(body).encode("utf-8") if body is not None else None
        try:
            status, payload = self._transport(method, url, headers, data, self._timeout)
        except Exception as exc:
            raise NotionError(ready.scrub(f"cannot reach Notion: {type(exc).__name__}: {exc}", [self._token])) from None
        if not 200 <= status < 300:
            detail = (payload.get("message") or payload.get("code") or "") if isinstance(payload, dict) else str(payload)
            raise NotionError(ready.scrub(f"Notion answered {status} to {method} on page {page_id}: {str(detail)[:300]}", [self._token]))
        return payload


# ---------------------------------------------------------------------------------------------
# Reading the tracks


@dataclass(frozen=True)
class Track:
    """One track's rulings and parsed log. log is None when the track has no REFINE_LOG.md or it cannot
    be read; log_error then says why in the second case."""

    name: str
    folder: Path
    rulings: tuple[Ruling, ...]
    log: Log | None
    log_error: str = ""

    @property
    def log_path(self) -> Path:
        return self.folder / LOG_NAME


@dataclass(frozen=True)
class Standing:
    """A ruling and where it stands.

    open        the Answer line is blank; the plan keeps the default (now)
    new         answered, and the log does not say so yet
    reanswered  answered again since the log recorded the earlier answer
    recorded    the log records the answer, and no pass entry has followed it: the next pass must act
    done        the log records the answer, and a pass entry followed it or the correction carried the changes
    unknown     answered, but the track's log cannot be read, so it is not known whether the log records it
    """

    track: str
    number: int
    question: str
    origin: str
    now: str
    recommend: str
    tension: str
    answer: str
    state: str

    @property
    def answered(self) -> bool:
        return self.state != OPEN

    @property
    def must_change(self) -> bool:
        return self.state in MUST_CHANGE

    def source(self, today: str) -> str:
        """The citation a change made for this ruling puts in its SOURCE line."""
        found = _OWNER_DATE.search(self.answer)
        return f"lab/refine/{self.track}/{RULINGS_NAME} {self.number} (owner, {found['date'] if found else today})"

    def as_dict(self, *, answer: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "track": self.track,
            "number": self.number,
            "question": self.question,
            "origin": self.origin,
            "now": self.now,
            "recommend": self.recommend,
            "tension": self.tension,
        }
        if answer:
            row["answer"] = self.answer
            row["state"] = self.state
        return row


def discover(refine_dir: Path) -> list[str]:
    """Names of the folders under refine_dir that hold an OWNER_RULINGS.md: the known tracks in their usual order, then the rest."""
    if not refine_dir.is_dir():
        return []
    found = [folder.name for folder in refine_dir.iterdir() if (folder / RULINGS_NAME).is_file()]
    return [name for name in TRACK_ORDER if name in found] + sorted(name for name in found if name not in TRACK_ORDER)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FormatError(f"cannot read it ({type(exc).__name__}: {exc})") from None


def load_track(refine_dir: Path, name: str) -> Track:
    """Parse a track's rulings and log. A rulings file that breaks the format raises FormatError naming it;
    a log that breaks it is kept in log_error, because the open rulings can still be listed without it."""
    folder = refine_dir / name
    try:
        rulings = parse_rulings(_read(folder / RULINGS_NAME))
    except FormatError as exc:
        raise FormatError(f"{folder / RULINGS_NAME}: {exc}") from None
    log, log_error = None, ""
    if (folder / LOG_NAME).is_file():
        try:
            log = parse_log(_read(folder / LOG_NAME))
        except FormatError as exc:
            log_error = f"{folder / LOG_NAME}: {exc}"
    return Track(name, folder, rulings, log, log_error)


def digest(answer: str) -> str:
    """Eight hex digits that change when an answer's wording changes."""
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:8]


def standings(track: Track) -> list[Standing]:
    """Where each ruling of the track stands, in file order, judged from the track's log."""
    marks: dict[int, tuple[int, OwnerCorrection]] = {}
    last_pass = -1
    for index, entry in enumerate(track.log.entries if track.log else ()):
        if isinstance(entry, OwnerCorrection):
            for number in entry.rulings:
                marks[number] = (index, entry)
        elif isinstance(entry, PassEntry) and entry.kind == "pass":
            last_pass = index
    out: list[Standing] = []
    for ruling in track.rulings:
        if not ruling.answered:
            state = OPEN
        elif track.log_error:
            state = UNKNOWN
        elif ruling.number not in marks:
            state = NEW
        else:
            index, correction = marks[ruling.number]
            recorded = re.search(rf"answer digest (?P<digest>[0-9a-f]{{8}}) for ruling {ruling.number}\b", correction.summary)
            if recorded and recorded["digest"] != digest(ruling.answer):
                state = REANSWERED
            elif correction.changes or index < last_pass:
                state = DONE
            else:
                state = RECORDED
        out.append(
            Standing(track.name, ruling.number, ruling.question, ruling.origin, ruling.now, ruling.recommend, ruling.tension, ruling.answer, state)
        )
    return out


def collect(refine_dir: Path, only: Sequence[str] = ()) -> tuple[list[Standing], list[dict[str, str]], list[dict[str, str]]]:
    """Every ruling of every track (or of the tracks named in only), the tracks whose rulings file could not
    be read (their rulings are missing), and the tracks whose log could not be read (their answered rulings
    stand as unknown)."""
    rows: list[Standing] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for name in discover(refine_dir):
        if only and name not in only:
            continue
        try:
            track = load_track(refine_dir, name)
        except FormatError as exc:
            errors.append({"track": name, "error": str(exc)})
            continue
        if track.log_error:
            warnings.append({"track": name, "error": track.log_error})
        rows.extend(standings(track))
    return rows, errors, warnings


# ---------------------------------------------------------------------------------------------
# apply: record answered rulings in the track log


def _paragraph(item: Standing, today: str) -> str:
    if item.state == REANSWERED:
        lead = f"Ruling {item.number} ({item.question}) was answered again by the owner, and this answer replaces the earlier one."
    else:
        lead = f"Ruling {item.number} ({item.question}) was answered by the owner and is now an owner rule at the top of the authority order."
    return (
        f'{lead} Answer, as written: "{item.answer}" (answer digest {digest(item.answer)} for ruling {item.number}). '
        f'Until this answer the plan followed the default: "{item.now}" '
        "No plan text changed here. The next pass makes the plan follow the answer, keeps it that way, "
        f"and cites {item.source(today)} in the SOURCE line of each change it makes for it."
    )


def render_correction(items: Sequence[Standing], today: str) -> str:
    """The owner correction entry (lab/refine/TEMPLATES.md, 1.2) that records these rulings, ending with its rule line."""
    numbers = ", ".join(str(item.number) for item in items)
    paragraphs = "\n\n".join(_paragraph(item, today) for item in items)
    return f"## Owner correction, {today} (no increment)\n\nRulings applied: {numbers}\n\n{paragraphs}\n\n---\n"


def _addition(raw: bytes, entry: str) -> str:
    """What to append to a log's bytes so the entry follows a blank line, in the file's own line endings."""
    newline = "\r\n" if b"\r\n" in raw else "\n"
    lead = newline if raw and not raw.endswith(b"\n") else ""
    return lead + newline + entry.replace("\n", newline)


def _must_change_row(item: Standing, today: str) -> dict[str, Any]:
    source = item.source(today)
    return {
        "track": item.track,
        "ruling": item.number,
        "question": item.question,
        "state": item.state,
        "was": item.now,
        "owner_rule": item.answer,
        "recommend": item.recommend,
        "source": source,
        "instruction": (
            f"Make the plan follow the owner's answer wherever it still follows the default in 'was', and cite {source} "
            "in the SOURCE line of each change. A change to RUBRIC.md waits for the merge pass, which cites the same source."
        ),
    }


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def apply_rulings(
    refine_dir: Path, only: Sequence[str] = (), *, today: str, dry_run: bool = False, lock_timeout: float = LOCK_TIMEOUT_S
) -> dict[str, Any]:
    """Record every new or reanswered ruling in its track's REFINE_LOG.md and list what the next pass must change.

    The log is only appended to, under a lock, and only after the log with the new entry has been
    parsed back; a track whose files do not parse is reported and left untouched.
    """
    names = [name for name in discover(refine_dir) if not only or name in only]
    errors: list[dict[str, str]] = []
    written: list[str] = []
    previews: dict[str, str] = {}
    recorded_now: list[dict[str, Any]] = []
    already: list[dict[str, Any]] = []
    must_change: list[dict[str, Any]] = []

    def run() -> None:
        for name in names:
            try:
                track = load_track(refine_dir, name)
            except FormatError as exc:
                errors.append({"track": name, "error": str(exc)})
                continue
            rows = standings(track)
            if any(row.state == UNKNOWN for row in rows):
                errors.append({"track": name, "error": f"{track.log_error}; nothing was written"})
                continue
            marking = [row for row in rows if row.state in (NEW, REANSWERED)]
            if marking:
                if track.log is None:
                    errors.append({"track": name, "error": f"{track.log_path}: there is no log to record the answers in"})
                    continue
                entry = render_correction(marking, today)
                try:
                    raw = track.log_path.read_bytes()
                    addition = _addition(raw, entry)
                    reread = parse_log(raw.decode("utf-8") + addition)
                except (OSError, UnicodeDecodeError, FormatError) as exc:
                    errors.append({"track": name, "error": f"{track.log_path}: nothing was written ({type(exc).__name__}: {exc})"})
                    continue
                last = reread.entries[-1] if reread.entries else None
                if not isinstance(last, OwnerCorrection) or last.rulings != tuple(row.number for row in marking):
                    errors.append({"track": name, "error": f"{track.log_path}: the correction does not read back as written; nothing was written"})
                    continue
                if dry_run:
                    previews[name] = entry
                else:
                    with open(track.log_path, "ab") as handle:
                        handle.write(addition.encode("utf-8"))
                    written.append(_shown(track.log_path))
                recorded_now.extend({"track": name, "ruling": row.number, "state": row.state} for row in marking)
            already.extend({"track": name, "ruling": row.number} for row in rows if row.state in (RECORDED, DONE))
            must_change.extend(_must_change_row(row, today) for row in rows if row.must_change)

    if dry_run:
        run()
    else:
        try:
            with FileLock(stop.state_dir() / LOCK_NAME, timeout=lock_timeout):
                run()
        except LockTimeout as exc:
            return {"ok": False, "outcome": "busy", "error": str(exc)}

    result: dict[str, Any] = {
        "ok": not errors,
        "dry_run": dry_run,
        "recorded": recorded_now,
        "already_recorded": already,
        "written": written,
        "next_pass_must_change": must_change,
        "errors": errors,
    }
    if dry_run:
        result["preview"] = previews
    return result


# ---------------------------------------------------------------------------------------------
# mirror: post the open rulings to Notion


def notion_text(text: str) -> str:
    """Text that Notion renders as written: file names and identifiers go in code, inline code stays,
    and the characters Notion reads as markup are escaped. Whitespace becomes single spaces."""
    parts = re.split(r"(`[^`\n]+`)", " ".join(text.split()))
    return "".join(part if index % 2 else _plain(part) for index, part in enumerate(parts))


def _plain(part: str) -> str:
    def word(match: re.Match[str]) -> str:
        raw = match.group(0)
        core = raw.strip(_EDGE_PUNCTUATION)
        if core and "`" not in core and ("_" in core or _CODE_WORD.search(core)):
            start = raw.index(core)
            return raw[:start].translate(_ESCAPES) + f"`{core}`" + raw[start + len(core):].translate(_ESCAPES)
        return raw.translate(_ESCAPES)

    return re.sub(r"\S+", word, part)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ") + " ..."


def eastern(moment: datetime) -> datetime:
    """The moment on Eastern time. Without a time zone database the machine's own clock is used,
    which is Eastern on the machine this runs on."""
    try:
        from zoneinfo import ZoneInfo

        return moment.astimezone(ZoneInfo("America/New_York"))
    except (ImportError, KeyError):
        return moment.astimezone()


def stamp(moment: datetime) -> str:
    """The signature every post starts with: the robot sign, Ilan's agent, a middle dot, then the day and Eastern time."""
    et = eastern(moment)
    return f"{SIGNER} {DAYS[et.weekday()]} {et.hour % 12 or 12}:{et.minute:02d} {'AM' if et.hour < 12 else 'PM'} ET:"


def list_id(items: Sequence[Standing]) -> str:
    """Eight hex digits naming this exact list of open rulings, their wording included."""
    canonical = json.dumps([[i.track, i.number, i.question, i.now, i.recommend, i.tension] for i in items], ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:8]


@dataclass(frozen=True)
class Posts:
    """The texts of one mirror run. segment is None, and block is the closing line, when nothing is waiting."""

    list_id: str
    count: int
    segment: str | None
    block: str


def build_posts(items: Sequence[Standing], moment: datetime) -> Posts:
    """The Overnight Status segment and the Team Comms FYI block for these open rulings."""
    ident = list_id(items)
    sign = stamp(moment)
    if not items:
        closing = f"- {sign} {PHRASE}: none. Every ruling in the lab's `{RULINGS_NAME}` files now has an answer. (list {ident})"
        return Posts(ident, 0, None, closing)

    titles = [f'{i.track} {i.number} "{notion_text(_clip(i.question, TITLE_CHARS))}"' for i in items[:MAX_TITLES]]
    more = len(items) - len(titles)
    tail = f", and {more} more" if more else ""
    segment = (
        f"{sign} {PHRASE} ({len(items)}): {'; '.join(titles)}{tail}"
        f" - the FYI of the same name on your Team Comms page has each one with the plan's default and the recommendation (list {ident})"
    )
    lines = [
        f"- {sign} {PHRASE} ({len(items)}). These are questions the lab's refinement passes could not decide by themselves; "
        "until you answer one, the plan keeps the default listed with it. To answer, tell your agent the track, the number and your "
        f"decision, or write it on that ruling's Answer line in the file named. (list {ident})"
    ]
    for i in items:
        pieces = [f"{i.track} question {i.number}: {notion_text(i.question)}"]
        for label, text in (("Until you answer, the plan does this:", i.now), ("Recommendation:", i.recommend), ("Why it is open:", i.tension)):
            if text:
                pieces.append(f"{label} {notion_text(_clip(text, FIELD_CHARS))}")
        pieces.append(f"File: `lab/refine/{i.track}/{RULINGS_NAME}`.")
        lines.append("\t- " + " ".join(pieces))
    return Posts(ident, len(items), segment, "\n".join(lines))


def print_plan(posts: Posts) -> dict[str, Any]:
    """Where each text goes and how a session pastes it, for use when no Notion client is available."""
    waiting = posts.segment is not None
    earlier = f"an earlier post of this flow: it starts with '{SIGNER}', a time and '{PHRASE}' and ends at '(list <8 hex digits>)'"
    return {
        "unanswered": posts.count,
        "list_id": posts.list_id,
        "posts": [
            {
                "target": "overnight_status",
                "action": "add_or_replace" if waiting else "remove_if_present",
                "page_url": OVERNIGHT_URL,
                "where": "the bullet that starts '- **Needs a human (Ilan):**' under '## Ilan'; the text goes at the end of that bullet after '; '",
                "text": posts.segment or "",
                "skip_if_page_shows": f"(list {posts.list_id})" if waiting else "",
                "replace_if_present": earlier + ("" if waiting else "; remove it from the bullet, and write 'none' there if nothing else is left on it"),
            },
            {
                "target": "team_comms_fyi",
                "action": "add_or_replace" if waiting else "replace_if_present",
                "page_url": TEAM_COMMS_URL,
                "where": "directly under the '## FYI (no reply needed)' heading, as its first item; the lines with a tab in front are children of the first line",
                "text": posts.block,
                "skip_if_page_shows": f"(list {posts.list_id})",
                "replace_if_present": earlier + ", together with its indented children",
            },
        ],
        "how": (
            "Fetch each page first. If it already shows the list id, post nothing there. If it shows an earlier post of this flow, "
            "replace that post with the text instead of adding a second one. Edit nothing else on either page."
        ),
    }


def _rewrite_needs_line(line: str, posts: Posts) -> tuple[str, str]:
    """The Needs a human line with this run's segment in place, and what was done: posted, updated, removed or unchanged."""
    found_line = _NEEDS_LINE.match(line)
    assert found_line is not None
    head, body = found_line["head"], found_line["body"]
    found = _SEGMENT.search(body)
    if posts.segment is None:
        if not found:
            return line, "unchanged"
        before = body[: found.start()].strip().rstrip(";").rstrip()
        after = body[found.end():].strip().lstrip(";").lstrip()
        return f"{head} {'; '.join(side for side in (before, after) if side) or 'none'}", "removed"
    if found:
        if found["id"] == posts.list_id:
            return line, "unchanged"
        return head + body[: found.start()] + posts.segment + body[found.end():], "updated"
    rest = body.strip()
    if rest.lower() in _EMPTY_BODY:
        return f"{head} {posts.segment}", "posted"
    glue = " " if rest.endswith((";", ".")) else "; "
    return f"{head} {rest}{glue}{posts.segment}", "posted"


def _mirror_overnight(client: NotionPages, posts: Posts) -> dict[str, Any]:
    try:
        lines = client.read_page(OVERNIGHT_PAGE).split("\n")
        hits = [i for i, text in enumerate(lines) if _NEEDS_LINE.match(text)]
        if len(hits) != 1:
            return {"action": "failed", "error": f"expected one '- **Needs a human (Ilan):**' line on the Overnight Status page, found {len(hits)}"}
        old = lines[hits[0]]
        new, action = _rewrite_needs_line(old, posts)
        if action == "unchanged":
            return {"action": "unchanged"}
        client.replace_text(OVERNIGHT_PAGE, old, new)
        after = client.read_page(OVERNIGHT_PAGE)
        shown = f"(list {posts.list_id})" in after if posts.segment else not _SEGMENT.search(after)
        if not shown:
            return {"action": "failed", "error": "the Overnight Status page does not show the change after the write"}
        return {"action": action}
    except NotionError as exc:
        return {"action": "failed", "error": str(exc)}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _mirror_fyi(client: NotionPages, posts: Posts) -> dict[str, Any]:
    try:
        lines = client.read_page(TEAM_COMMS_PAGE).split("\n")
        heading = next((i for i, text in enumerate(lines) if _FYI_HEADING.match(text)), None)
        if heading is None:
            return {"action": "failed", "error": "Ilan's Team Comms page has no '## FYI' heading"}
        mine = {i: found["id"] for i, text in enumerate(lines) if (found := _FYI_FIRST.match(text))}
        if posts.list_id in mine.values():
            return {"action": "unchanged"}
        if mine:
            start = min(mine)
            end = start + 1
            while end < len(lines) and lines[end].strip() and _indent(lines[end]) > _indent(lines[start]):
                end += 1
            old, new = "\n".join(lines[start:end]), posts.block
            action = "updated" if posts.count else "closed"
        elif posts.count:
            old, new = lines[heading], lines[heading] + "\n" + posts.block
            action = "posted"
        else:
            return {"action": "unchanged"}
        client.replace_text(TEAM_COMMS_PAGE, old, new)
        if f"(list {posts.list_id})" not in client.read_page(TEAM_COMMS_PAGE):
            return {"action": "failed", "error": "Ilan's Team Comms page does not show the post after the write"}
        return {"action": action}
    except NotionError as exc:
        return {"action": "failed", "error": str(exc)}


def mirror_rulings(items: Sequence[Standing], client: NotionPages, moment: datetime) -> dict[str, Any]:
    """Bring both Notion posts in line with the open rulings; each target is judged on its own."""
    posts = build_posts(items, moment)
    targets = {"overnight_status": _mirror_overnight(client, posts), "team_comms_fyi": _mirror_fyi(client, posts)}
    return {
        "ok": all(result["action"] != "failed" for result in targets.values()),
        "mode": "live",
        "unanswered": posts.count,
        "list_id": posts.list_id,
        "targets": targets,
    }


# ---------------------------------------------------------------------------------------------
# The verb


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--refine-dir", type=Path, default=REFINE_DIR, help="folder that holds the track folders (default: lab/refine)")
    parser = argparse.ArgumentParser(prog="python -m lab rulings", description="Owner rulings of the refinement tracks.")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", parents=[common], help="the open rulings with their defaults and recommendations")
    listing.add_argument("--all", action="store_true", help="include the answered rulings, with their answer and where they stand")
    listing.add_argument("--track", action="append", default=[], help="only this track (repeatable)")

    apply = commands.add_parser("apply", parents=[common], help="record answered rulings as owner rules in the track logs")
    apply.add_argument("--dry-run", action="store_true", help="write nothing; return the entries that would be appended")
    apply.add_argument("--track", action="append", default=[], help="only this track (repeatable)")
    apply.add_argument("--date", type=_date.fromisoformat, default=None, help="date of the correction entries (default: today, Eastern)")

    mirror = commands.add_parser("mirror", parents=[common], help="post the open rulings to Notion")
    mirror.add_argument("--print", dest="print_only", action="store_true", help="touch nothing; return the texts and where they go")
    return parser


def main(
    argv: Sequence[str],
    *,
    client: NotionPages | None = None,
    transport: Transport | None = None,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
    env_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Owner rulings: list the open ones, record the answered ones as owner rules, mirror the waiting ones to Notion.

    A rulings file that cannot be read stops the call that needs the whole list (mirror) and is an error in the
    others; a log that cannot be read is a warning, because the open rulings are complete without it.

    client, transport, now, environ and env_files replace the Notion client, its HTTP transport, the clock, the
    process environment and the .env search; the dispatcher passes none of them.
    """
    args = _parser().parse_args(list(argv))
    moment = now or datetime.now(timezone.utc)
    refine_dir: Path = args.refine_dir
    tracks = discover(refine_dir)
    if not tracks:
        return {"ok": False, "outcome": "no_tracks", "error": f"no {RULINGS_NAME} under {refine_dir}"}

    wanted: list[str] = getattr(args, "track", [])
    unknown = [name for name in wanted if name not in tracks]
    if unknown:
        return {"ok": False, "outcome": "unknown_track", "error": f"no such track: {', '.join(unknown)}", "tracks": tracks}

    if args.command == "apply":
        today = (args.date or eastern(moment).date()).isoformat()
        return apply_rulings(refine_dir, wanted, today=today, dry_run=args.dry_run)

    rows, errors, warnings = collect(refine_dir, wanted)
    if args.command == "list":
        shown = rows if args.all else [row for row in rows if row.state == OPEN]
        return {
            "ok": not errors,
            "unanswered": sum(1 for row in rows if row.state == OPEN),
            "answered": sum(1 for row in rows if row.answered),
            "rulings": [row.as_dict(answer=args.all) for row in shown],
            "errors": errors,
            "warnings": warnings,
        }

    # mirror: a list built from some of the tracks would tell Ilan less than is waiting, so any unreadable track stops it
    if errors:
        return {"ok": False, "outcome": "format_error", "error": "a track's files cannot be read, so the list would be incomplete", "errors": errors}
    open_rows = [row for row in rows if row.state == OPEN]
    posts = build_posts(open_rows, moment)
    extra = {"warnings": warnings} if warnings else {}
    if args.print_only:
        return {"ok": True, "mode": "print", **print_plan(posts), **extra}
    if client is None:
        environment = os.environ if environ is None else environ
        files = ready.default_env_files(environment) if env_files is None else env_files
        # Stripped as the mirror verb strips it: a CRLF .env sourced into a shell leaves a CR on the value, which
        # http.client refuses in a header with a message that spells the token in a form ready.scrub cannot see.
        token = ready.collect_env((TOKEN_ENV,), environment, files)[0].get(TOKEN_ENV, "").strip()
        if not token:
            return {
                "ok": False,
                "outcome": "no_token",
                "error": f"{TOKEN_ENV} is not set, so nothing was posted; paste the posts below through the Notion tools, or run with --print",
                "mode": "print",
                **print_plan(posts),
                **extra,
            }
        client = RestNotionClient(token, transport=transport)
    return {**mirror_rulings(open_rows, client, moment), **extra}
