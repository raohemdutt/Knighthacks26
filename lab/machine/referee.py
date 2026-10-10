"""referee: end the GPU slots that ran past their cap, report who waits, and critique long requests, for the Fable
sitting that posts it.

    python -m lab referee [--gpu A|B|all] [--dry-run]
    python -m lab referee --critique <row id or url> --text "one line on how it could be done more efficiently"

The GPU queue (lab/machine/slots.py) lets every agent work only on its own row, so a slot that nobody finishes
would hold the card for good. The referee is the one caller that may write another agent's row: it ends one that
ran past its cap, and it adds the one critique line to a long request. A Fable sitting runs it first each time
it sits and posts what it prints. One call reads the Queued, Running and finished rows of the Notion database
"GPU Queue (R9700 pods)" (--gpu B, A or both, the default) and

  ends      every Running row past its cap, which is its Minutes plus 15 once Extended is checked
            (slots.is_overrun): Status "Ended by referee", Ended now, and a note in Result that says how long it
            ran against its cap. Nothing else on the row changes, and a Result already there stays in front of
            the note. The row is read again just before the update, and one that its agent finished or extended
            in the meantime is left alone (listed under skipped). A row inside its cap is never touched, and
            neither is a Queued row or a Running row with no Started, whose cap cannot be judged (anomalies).
  lists     the Queued rows with a Complaint, with how long they have waited and what they wait behind; the
            Queued rows that have waited more than twice their Minutes and wrote no Complaint (slots.
            complaint_due: the lab pool calls a referee sitting for these too); the Queued rows that could
            start now, which is the first row of the line of a card that has no Running row left once the slots
            above are ended (slots.queue_order: a starved row first, then shortest first); what still runs
            inside its cap; the Queued rows that ask for more than 30 minutes and wait for a critique
            (critique_due, with the justification to read) or for their agent's reply (awaiting_reply); and
            anything odd, such as two Running rows on one card or a long request with no justification.
  prints    one JSON line (below) whose summary is plain text for the sitting to paste under its own signature.
            post says whether there is anything to post; a quiet check is not worth a post.

The critique. A Queued row that asks for more than 30 minutes carries a justification in its Progress note (see
slots.py). The sitting reads each one from critique_due once and writes one critique with --critique ROW --text
"...": a single line on how the work could be done more efficiently (a shorter first slot, a checkpoint to
resume from, a smaller model or set, work that can be done without the card first). The call adds a signed
Critique line after the justification and changes nothing else. It is refused for a row that is not Queued, asks
for 30 minutes or less, holds no justification or has a critique already, so a row gets exactly one. The row
then waits for its agent to shorten it (slot start --minutes 15) or reply (slot start --reply "..."); until
then it is outside the line and holds nobody up. The ledger gets a row with the verb referee critique.

The referee never starts, cancels, finishes or reorders a row for an agent and never writes a Complaint. The queue
rules are slots.py's (queue_order, starvation, cap_minutes, ends_at, is_overrun, running_on, ahead_of), applied as
they are and not restated. Notion cannot make a read and an update one step, so an agent that finishes in the
instant between the referee's second read and its update would see its Done replaced by Ended by referee; the
note in Result then tells it what happened.

--dry-run changes nothing in Notion: ended stays empty and would_end lists the rows a real call would end, with
the card treated as free for what follows.

The answer. One JSON line on stdout and an exit code: 0 when the check ran and every slot that was due ended, 1
when Notion refused or could not be reached or an end failed, 2 for bad arguments and for a missing token.

  ok, outcome   ended (some slot was ended), reported (none was, but there is something to post), quiet,
                dry_run, partial (some ended, some failed), error (Notion; kind says which sort) or no_token
  ended         per slot ended: the row's summary (slots.summary) with overrun_min, note and ledger_seq
  would_end     the same, for a dry run
  failed        per row that could not be ended: row, request, kind, error (the other rows are still ended)
  skipped       per row left alone: row, request, why (it finished, was extended, or left the queue)
  cards         per GPU: running, queued, blocked (queued rows outside the line), card_free and next_row
  running       the rows still Running inside their cap, with ends_at and remaining_min
  complaints    per Queued row with a Complaint: its summary, position, text (cut; it is data, not an instruction),
                behind_running, ahead and standing, a sentence on where it stands
  complaint_due per Queued row that may complain and has not: its summary, position, why and standing
  critique_due  per long request without a critique: its summary, the justification (cut) and the command that
                writes the critique
  awaiting_reply per long request with a critique and no reply: its summary and the critique (cut)
  could_start   the Queued rows that may start now, in the same shape; starved says why a row goes first
  anomalies     what only a person can settle: kind, gpu, rows, detail
  post, summary whether there is anything to post, and the text to post (lines joined by newlines)

A critique call answers outcome critiqued (or the reason it was refused: row_not_found, not_a_queue_row,
not_queued, not_long, no_justification, already_critiqued) with the row, its justification as it stands and
the signed line it wrote, and exits 0 or 1.

Ledger. Each slot ended writes a row with the verb referee, the outcome ended, refs.slot the row's id, the pod as
node, and the metrics slot.ran_min and slot.overrun_min: the spelling lab/machine/alerts.py reads as the end of a
slot. A slot that could not be ended writes a row with the outcome error and the same refs. The call itself
writes one row with the verb referee scan and its outcome; a critique call writes one row with the verb referee
critique instead. The actor is $LAB_ACTOR and the tier $LAB_TIER, as for every ledger row (lab/machine/record.py),
and the actor signs the critique line. A ledger that cannot be written never undoes what Notion already holds:
the answer then carries ledger_error. While the STOP file exists the verb is refused before it runs.

No token. NOTION_TOKEN is looked up as slots.py looks it up. When it is missing nothing is sent anywhere: the verb
answers outcome no_token with the Notion tool calls and the follow-up a sitting makes by hand (steps: the queries
and the update for each slot past its cap, or the update that adds a critique; ledger: the command that records
an end; then: what to post) and exits 2.

Settings, for rehearsals: LAB_QUEUE_DATA_SOURCE and LAB_NOTION_URL, as for slots.py. The token is never printed,
logged or written to the ledger.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lab.machine import record, slots, stop
from lab.machine import slots_notion as notion

__all__ = [
    "CRITIQUE_VERB", "ENDED_BY_REFEREE", "NOTE_FORMAT", "SCAN_VERB", "VERB", "check", "critique", "end_note", "main",
    "offline_plan", "run", "standing", "summary_text",
]

log = logging.getLogger("lab.referee")

VERB = "referee"                  # the ledger verb of a slot ended: lab/machine/alerts.py reads it as the end
SCAN_VERB = "referee scan"        # the ledger verb of the call itself
CRITIQUE_VERB = "referee critique"  # the ledger verb of a critique written
ENDED_BY_REFEREE = slots.ENDED_BY_REFEREE

CHOICES = ("A", "B", "ALL")
CARD_ORDER = ("B", "A")           # the exclusive card first
CARDS_BY_NAME = {name: letter for letter, name in slots.GPUS.items()}

NOTE_FORMAT = ("Ended by the referee at {when}Z: ran {ran} min against a cap of {cap} ({asked} asked{extended}), "
               "{over} min over, and was not finished with slot done.")

MAX_LISTED = 8                    # entries of one kind that the summary names before "and N more"
TITLE_CHARS = 80
COMPLAINT_CHARS = 300             # of a complaint, in the answer
SUMMARY_COMPLAINT_CHARS = 120     # of a complaint, in the summary text
JUSTIFICATION_CHARS = 400         # of a justification or a critique, in the answer
SUMMARY_JUSTIFICATION_CHARS = 240  # of a justification, in the summary text
ERROR_CHARS = 160

RULE_END = ("A Running row is past its cap when Started plus Minutes (plus 15 when Extended is checked) is earlier "
            "than now. End each such row, and only such rows, with one update: Status 'Ended by referee', Ended "
            "now, and Result holding the note (keep any text Result already has, in front of it). Read the row "
            "again just before the update and leave it if its agent has finished or extended it meanwhile. "
            "Never start, cancel or finish a row for its agent, and write no Complaint. Then collect every Queued "
            "row with a Complaint (how long it has waited, what it waits behind) and the Queued rows that could "
            "start: for each GPU with no Running row left, the first row of the line. Which row is first follows "
            "the start rule: " + slots.RULE_START)
RULE_DRY = "Read only: write nothing to Notion, only report which rows you would end. "
RULE_CRITIQUE = ("One critique per row: only on a Queued row that asks for more than 30 minutes, holds a Justification "
                 "line in its Progress note and no Critique line yet. Write one line on how the work could be done "
                 "more efficiently and add it after the text the Progress note already holds, after ' | '. Change "
                 "nothing else on the row.")


# --- small helpers --------------------------------------------------------------------------

def flat(text: Any, limit: int) -> str:
    """text on one line, cut with ... at limit characters."""
    one = " ".join(str(text).split())
    return one if len(one) <= limit else one[: max(limit - 3, 0)] + "..."


def hhmm(moment: str) -> str:
    """The clock time of a slots.iso text, as 14:30Z."""
    return moment[11:16] + "Z"


def letters_of(choice: str) -> tuple[str, ...]:
    """The GPU letters a --gpu choice names, the exclusive card first."""
    return CARD_ORDER if choice == "ALL" else (choice,)


def _timing(slot: slots.Slot, now: dt.datetime) -> tuple[float, float, float]:
    """(minutes run, the cap, minutes past the cap) of a started slot at now."""
    ran = slots.minutes_between(now, slot.started) if slot.started else 0.0
    cap = slots.cap_minutes(slot)
    return ran, cap, max(0.0, ran - cap)


def end_note(slot: slots.Slot, now: dt.datetime) -> str:
    """The note a slot ended by the referee carries: how long it ran against its cap."""
    ran, cap, over = _timing(slot, now)
    extended = f", extended by {slots.EXTENSION_MINUTES}" if slot.extended else ""
    return NOTE_FORMAT.format(when=slots.iso(now)[:16], ran=slots.tidy(ran), cap=slots.tidy(cap),
                              asked=slots.tidy(slot.minutes_or_default), extended=extended, over=slots.tidy(over))


def with_note(existing: str, note: str) -> str:
    """The Result column with note added after what it already holds, the old text cut to fit one Notion value."""
    if not existing:
        return note
    room = max(0, slots.MAX_COMPLAINT_CHARS - len(note) - len(slots.COMPLAINT_JOIN))
    return existing[:room] + slots.COMPLAINT_JOIN + note


def _title(slot: slots.Slot) -> str:
    return flat(slot.request, TITLE_CHARS)


def _overrun_min(slot: slots.Slot, now: dt.datetime) -> int | float:
    """Minutes a started slot is past its cap at now; 0 while it is inside it."""
    return slots.tidy(_timing(slot, now)[2])


# --- what is standing ------------------------------------------------------------------------

def _standing(running: Sequence[slots.Slot], ahead: Sequence[slots.Slot], may_start: bool,
              now: dt.datetime, hungry: slots.Starvation | None = None) -> str:
    """One clause on where a Queued row stands: next in line, or what it waits behind, and why it goes first when
    the starvation guard put it there."""
    if may_start:
        text = "it is next in line and the card is free, so its agent can run slot start now"
    else:
        parts = []
        for other in running:
            end = slots.ends_at(other)
            if end is None:
                left = "no Started"
            elif now > end:
                left = "past its cap"
            else:
                left = f"{slots.tidy(slots.minutes_between(end, now))} min left"
            parts.append(f'the Running slot "{_title(other)}" ({left})')
        if ahead:
            parts.append(f"{len(ahead)} Queued row(s) that go first (fewest minutes, then earliest request)")
        text = "it waits behind " + " and ".join(parts)
    return text + (f"; it goes first, whatever its length, because it {hungry.why}" if hungry else "")


def standing(boards: Mapping[str, Sequence[slots.Slot]], now: dt.datetime) -> dict[str, Any]:
    """What stands on each card, given its rows (GPU name -> the Queued and Running rows and the ones that started
    and are over): the cards, the rows still Running, the complaints, the Queued rows that could start, the long
    requests waiting for a critique or a reply, and the anomalies. The slot rules are slots.py's."""
    cards: list[dict[str, Any]] = []
    still_running: list[dict[str, Any]] = []
    complaints: list[dict[str, Any]] = []
    complaint_due: list[dict[str, Any]] = []
    critique_due: list[dict[str, Any]] = []
    awaiting_reply: list[dict[str, Any]] = []
    could_start: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    for gpu_name, rows in boards.items():
        running = slots.running_on(rows, gpu_name)
        line = slots.queue_order(rows, now)
        in_line = {row.id for row in line}
        held = sorted((row for row in rows if row.status == slots.QUEUED and row.id not in in_line),
                      key=slots.queue_key)
        free = not running
        cards.append({"gpu": gpu_name, "running": len(running), "queued": len(line) + len(held), "blocked": len(held),
                      "card_free": free, "next_row": line[0].id if line else None})
        still_running += [slots.summary(row, now) for row in running]
        if len(running) > 1:
            anomalies.append({"kind": "several_running", "gpu": gpu_name, "rows": [row.id for row in running],
                              "detail": f"{len(running)} rows are Running on {gpu_name}, where the queue allows one; "
                                        "the referee ends only slots past their cap, so settle this by hand"})
        for row in running:
            if row.started is None:
                anomalies.append({"kind": "running_without_started", "gpu": gpu_name, "rows": [row.id],
                                  "detail": f'"{_title(row)}" is Running with no Started, so its cap cannot be '
                                            "judged; set Started or finish it by hand"})
        waiting: list[tuple[slots.Slot, dict[str, Any], str, list[slots.Slot]]] = []
        for position, row in enumerate(line, start=1):
            may_start = free and position == 1
            entry = {**slots.summary(row, now), "url": row.url, "position": position, "may_start": may_start}
            hungry = slots.starvation(row, rows, now) if slots.jumped(row, rows, now) else None
            if hungry is not None:
                entry["starved"] = hungry.why
            if may_start:
                could_start.append(entry)
            ahead = slots.ahead_of(row, rows, now)
            waiting.append((row, entry, _standing(running, ahead, may_start, now, hungry), ahead))
        for row in held:
            state = slots.justification_state(row)
            entry = {**slots.summary(row, now), "url": row.url, "position": None, "may_start": False,
                     "waiting_for": slots.WAITING_FOR[state]}
            waiting.append((row, entry, f"it cannot start yet: it waits for {slots.WAITING_FOR[state]}", []))
            thread = slots.read_thread(row.progress_note)
            if state == slots.AWAITING_CRITIQUE:
                critique_due.append({**entry, "justification": flat(thread.justification, JUSTIFICATION_CHARS),
                                     "write": f'python -m lab referee --critique {row.id} --text "<one line>"'})
            elif state == slots.AWAITING_REPLY:
                awaiting_reply.append({**entry, "critique": flat(thread.critique, JUSTIFICATION_CHARS)})
            else:
                anomalies.append({"kind": "long_without_justification", "gpu": gpu_name, "rows": [row.id],
                                  "detail": f'"{_title(row)}" asks {slots.tidy(row.minutes_or_default)} minutes and '
                                            "holds no justification, so it cannot start; its agent shortens it "
                                            "(slot start --minutes 15) or cancels it and asks again with --why and "
                                            "--measured"})
        for row, entry, where, ahead in waiting:
            if row.complaint:
                complaints.append({**entry, "text": flat(row.complaint, COMPLAINT_CHARS),
                                   "behind_running": [slots.summary(other, now) for other in running],
                                   "ahead": [slots.summary(other, now) for other in ahead], "standing": where})
                continue
            due, why, _ = slots.complaint_due(row, rows, now)
            if due:
                complaint_due.append({**entry, "why": why, "standing": where})
    return {"cards": cards, "running": still_running, "complaints": complaints, "complaint_due": complaint_due,
            "critique_due": critique_due, "awaiting_reply": awaiting_reply, "could_start": could_start,
            "anomalies": anomalies}


# --- the summary a sitting posts -------------------------------------------------------------

def _listed(heading: str, items: Sequence[str]) -> list[str]:
    """A heading and up to MAX_LISTED bullet lines; nothing at all when there are no items."""
    if not items:
        return []
    lines = [heading, *(f"- {item}" for item in items[:MAX_LISTED])]
    if len(items) > MAX_LISTED:
        lines.append(f"- and {len(items) - MAX_LISTED} more (see the JSON).")
    return lines


def _card_line(card: Mapping[str, Any], running: Sequence[Mapping[str, Any]]) -> str:
    """One card on one line: what runs on it (and when that ends) and how many wait."""
    mine = [row for row in running if row["gpu"] == card["gpu"]]
    if not card["running"]:
        state = "free"
    else:
        state = f"{card['running']} running"
        if mine and "ends_at" in mine[0]:
            state += f" (ends {hhmm(mine[0]['ends_at'])})"
    return f"{card['gpu']}: {state}, {card['queued']} waiting."


def _agent(row: Mapping[str, Any]) -> str:
    return row["agent"] or "no agent"


def summary_text(answer: Mapping[str, Any], now: dt.datetime) -> str:
    """The text to post, from an answer's lists. Names, numbers and clipped row text only; one line per item."""
    dry = bool(answer["dry_run"])
    lines = [f"Referee check at {slots.iso(now)[:16]}Z"
             + (" (dry run: nothing was changed, and the cards below are as they would be after the ends)." if dry
                else ".")]
    lines += [_card_line(card, answer["running"]) for card in answer["cards"]]
    ended = answer["would_end"] if dry else answer["ended"]
    lines += _listed(
        f"{'Would end' if dry else 'Ended'} {len(ended)} slot(s) past the cap:",
        [f'"{flat(row["request"], TITLE_CHARS)}" ({_agent(row)}) on {row["gpu"]} ran {row["ran_min"]} min against a '
         f'cap of {row["cap_min"]}, {row["overrun_min"]} min over.' for row in ended])
    lines += _listed(
        f"Could not end {len(answer['failed'])} slot(s):",
        [f'"{flat(row["request"], TITLE_CHARS)}": {row["kind"]}, {flat(row["error"], ERROR_CHARS)}'
         for row in answer["failed"]])
    lines += _listed(
        f"Complaints on waiting requests ({len(answer['complaints'])}):",
        [f'"{flat(row["request"], TITLE_CHARS)}" ({_agent(row)}) has waited {row.get("waited_min", "?")} min for '
         f'{row["minutes"]} asked; {row["standing"]}. It says: {flat(row["text"], SUMMARY_COMPLAINT_CHARS)}'
         for row in answer["complaints"]])
    lines += _listed(
        f"Complaint due, none written ({len(answer['complaint_due'])}):",
        [f'"{flat(row["request"], TITLE_CHARS)}" ({_agent(row)}) {row["why"]}; {row["standing"]}.'
         for row in answer["complaint_due"]])
    lines += _listed(
        f"Long requests waiting for a critique ({len(answer['critique_due'])}); write one per row with python -m "
        'lab referee --critique ROW --text "one line on how it could be done more efficiently":',
        [f'row {row["row"]}: "{flat(row["request"], TITLE_CHARS)}" ({_agent(row)}) asks {row["minutes"]} min and has '
         f'waited {row.get("waited_min", "?")} min. Justification: '
         f'{flat(row["justification"], SUMMARY_JUSTIFICATION_CHARS)}' for row in answer["critique_due"]])
    lines += _listed(
        f"Could start now ({len(answer['could_start'])}):",
        [f'{row["gpu"]}: "{flat(row["request"], TITLE_CHARS)}" ({_agent(row)}, {row["minutes"]} min, waiting '
         f'{row.get("waited_min", "?")} min); its agent should run python -m lab slot start --row {row["row"]}.'
         + (f' It goes first, whatever its length, because it {row["starved"]}.' if row.get("starved") else "")
         for row in answer["could_start"]])
    lines += _listed(f"Needs a look by hand ({len(answer['anomalies'])}):", [row["detail"] for row in answer["anomalies"]])
    if not answer["post"]:
        lines.append("Nothing is past its cap, nobody has complained, and no request is waiting to start.")
    return "\n".join(lines)


# --- the check -------------------------------------------------------------------------------

def _end_one(queue: slots.SlotQueue, clock: Callable[[], dt.datetime],
             slot: slots.Slot) -> tuple[str, dict[str, Any], slots.Slot | None]:
    """End one slot past its cap and say what came of it: ("ended", "skipped" or "failed"), the entry for the
    answer, and the row as it stands for the rest of the check (None when the page is gone). The row is read
    again first, so one that was finished or extended meanwhile is left alone. A NotionError is the row's failure
    and is not raised."""
    try:
        fresh, refusal = queue.load(slot.id, "", VERB, own=False)
        if fresh is None:
            return "skipped", {"row": slot.id, "request": slot.request, "why": refusal["reason"]}, None
        if fresh.status != slots.RUNNING:
            return "skipped", {"row": slot.id, "request": fresh.request,
                               "why": f"the row is {fresh.status} now, not Running"}, fresh
        now = clock()
        if not slots.is_overrun(fresh, now):
            return "skipped", {"row": slot.id, "request": fresh.request,
                               "why": "the slot is inside its cap now (it was extended)"}, fresh
        note = end_note(fresh, now)
        change = {"Status": ENDED_BY_REFEREE, "Ended": now, "Result": with_note(fresh.result, note)}
        updated = slots.slot_from_page(queue.client.update_page(fresh.id, slots.rest_properties(change)))
        if updated.status != ENDED_BY_REFEREE:
            raise notion.NotionError(f"Notion answered the update with Status {updated.status!r}", kind="bad_response")
        return "ended", {**slots.summary(updated, now), "overrun_min": _overrun_min(fresh, now), "note": note}, updated
    except notion.NotionError as exc:
        return "failed", {"row": slot.id, "request": slot.request, "kind": exc.kind,
                          "error": flat(exc, ERROR_CHARS)}, slot


def check(queue: slots.SlotQueue, clock: Callable[[], dt.datetime], letters: Sequence[str], *,
          dry_run: bool = False, on_ended: Callable[[slots.Slot, dict[str, Any]], int | None] | None = None,
          on_failed: Callable[[slots.Slot, dict[str, Any]], int | None] | None = None) -> dict[str, Any]:
    """One round over the cards named by letters (A, B): end what ran past its cap, then say what stands.

    on_ended(row before, its entry) runs after each end and on_failed(row, its entry) after each failure; they
    return a ledger seq, and an exception from one is kept as ledger_error and changes nothing. A NotionError
    from reading the queue propagates, before anything is written; one from ending a row is that row's failure
    and the round goes on. A slot that was ended stays on the board as a row that started, because its start is
    one of those that passed a waiting row over."""
    seen = clock()
    boards = {slots.GPUS[letter]: queue.board(slots.GPUS[letter]) for letter in letters}
    ended: list[dict[str, Any]] = []
    would_end: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    buckets = {"ended": ended, "failed": failed, "skipped": skipped}
    callbacks = {"ended": on_ended, "failed": on_failed}
    for gpu_name, rows in list(boards.items()):
        stands = {row.id: row for row in rows}
        for slot in sorted((row for row in rows if slots.is_overrun(row, seen)), key=lambda row: (row.started, row.id)):
            if dry_run:
                would_end.append({**slots.summary(slot, seen), "overrun_min": _overrun_min(slot, seen),
                                  "note": end_note(slot, seen)})
                stands[slot.id] = dataclasses.replace(slot, status=ENDED_BY_REFEREE, ended=seen)
                continue
            what, entry, current = _end_one(queue, clock, slot)
            callback = callbacks.get(what)
            if callback is not None:
                try:
                    entry["ledger_seq"] = callback(slot, entry)
                except Exception as exc:
                    entry["ledger_seq"] = None
                    entry["ledger_error"] = f"{type(exc).__name__}: {exc}"
                    log.error("referee: no ledger row written for slot %s (%s): %s", slot.id, what, exc)
            buckets[what].append(entry)
            if current is None:
                del stands[slot.id]
            else:
                stands[slot.id] = current
        boards[gpu_name] = list(stands.values())
    report = standing(boards, seen)
    post = bool(ended or would_end or failed or report["complaints"] or report["complaint_due"]
                or report["critique_due"] or report["could_start"] or report["anomalies"])
    if dry_run:
        outcome = "dry_run"
    elif failed:
        outcome = "partial" if ended else "error"
    else:
        outcome = "ended" if ended else ("reported" if post else "quiet")
    answer: dict[str, Any] = {"ok": not failed, "outcome": outcome, "verb": VERB, "now": slots.iso(seen),
                              "dry_run": dry_run, "ended": ended, "would_end": would_end, "failed": failed,
                              "skipped": skipped, **report, "post": post}
    answer["summary"] = summary_text(answer, seen)
    return answer


# --- the critique ----------------------------------------------------------------------------

def critique(queue: slots.SlotQueue, clock: Callable[[], dt.datetime], row_id: str, text: str, *,
             actor: str = record.UNKNOWN_ACTOR) -> dict[str, Any]:
    """Add the one critique to the justification of a long request: the answer, outcome critiqued, or the refusal.
    The row is read first and must be Queued, ask for more than 30 minutes, hold a justification and no critique;
    only the Progress note is written, with the signed line after the text it holds. A NotionError propagates."""
    verb = CRITIQUE_VERB
    slot, bad = queue.load(row_id, "", verb, own=False)
    if bad:
        return bad
    now = clock()
    state = slots.justification_state(slot)
    if slot.status != slots.QUEUED:
        return slots.refusal(verb, "not_queued", f"only a Queued row is critiqued; this one is {slot.status}",
                             row=slot.id, slot=slots.summary(slot, now))
    if state == slots.NOT_REQUIRED:
        return slots.refusal(verb, "not_long", f"the row asks for {slots.tidy(slot.minutes_or_default)} minutes, no "
                             f"more than {slots.LONG_ABOVE_MINUTES}, so it carries no justification to critique",
                             row=slot.id, slot=slots.summary(slot, now))
    if state == slots.MISSING:
        return slots.refusal(verb, "no_justification", "the row asks for more than "
                             f"{slots.LONG_ABOVE_MINUTES} minutes but holds no justification line to critique",
                             row=slot.id, slot=slots.summary(slot, now))
    if state != slots.AWAITING_CRITIQUE:
        return slots.refusal(verb, "already_critiqued", "the row has its one critique already",
                             row=slot.id, slot=slots.summary(slot, now),
                             justification=slots.justification_view(slot))
    line = slots.note_entry(slots.CRITIQUE, actor, now, text)
    change = {"Progress note": slots.add_entry(slot.progress_note, line)}
    updated = slots.slot_from_page(queue.client.update_page(slot.id, slots.rest_properties(change)))
    if slots.justification_state(updated) != slots.AWAITING_REPLY:
        raise notion.NotionError("Notion answered the update without the critique line", kind="bad_response")
    return {"ok": True, "outcome": "critiqued", "verb": verb, "row": updated.id, "line": line,
            "slot": slots.summary(updated, now), "justification": slots.justification_view(updated),
            "next": f"the agent of the row answers with python -m lab slot start --row {updated.id} --reply \"...\" "
                    f"or shortens it with python -m lab slot start --row {updated.id} --minutes "
                    f"{slots.DEFAULT_MINUTES}"}


# --- no token: what a sitting does by hand ---------------------------------------------------

def offline_plan(args: argparse.Namespace, now: dt.datetime, data_source: str) -> dict[str, Any]:
    """What a sitting does with its Notion tools when no token lets the verb do it: the queries, the update to
    apply to each slot past its cap and what to do after, as an answer with outcome no_token."""
    names = [slots.GPUS[letter] for letter in letters_of(args.gpu)]
    note = NOTE_FORMAT.format(when=slots.iso(now)[:16], ran="<minutes since Started>",
                              cap="<Minutes, plus 15 if Extended>", asked="<Minutes>",
                              extended="<, extended by 15 - only when Extended is checked>", over="<ran minus cap>")
    change = {"Status": ENDED_BY_REFEREE, "Ended": now, "Result": f"<the Result as it is>{slots.COMPLAINT_JOIN}{note}"}
    plan: dict[str, Any] = {
        "ok": False, "outcome": "no_token", "verb": VERB, "dry_run": args.dry_run,
        "error": f"{slots.ENV_TOKEN} is not set (process environment or a .env file), so nothing was sent to "
                 "Notion; make the calls below with your Notion tools instead",
        "steps": [slots._query_step(name, data_source, f"read the Queued, Running, Done and Ended by referee rows of "
                                    f"{name}") for name in names],
    }
    if not args.dry_run:
        plan["write"] = slots.mcp_properties(change)
        plan["steps"].append(slots._update_step(
            "<page id of a Running row past its cap>", change,
            "once for each Running row past its cap, after reading the row again; fill in the <> parts of Result"))
        plan["ledger"] = {
            "command": "python -m lab ledger add referee ended --ref slot=<page id> --task <Task ID> --node "
                       "<pod-a or pod-b> --metric slot.ran_min=<minutes it ran> --metric slot.overrun_min=<minutes "
                       "over the cap>",
            "why": "once for each row you ended, with the <> parts filled in, so the alerts and the review see "
                   "the slot end"}
    plan["then"] = ["Post what you found as a signed FYI on Ilan's Team Comms page: the rows you ended, each "
                    "Complaint you read (how long it has waited and what it waits behind), the Queued rows that "
                    "could start, and anything odd. Leave out the parts that are empty, and post nothing when all "
                    "of them are.",
                    "For each Queued row that asks for more than 30 minutes, holds a Justification line in its "
                    "Progress note and has no Critique line yet, write one critique (see rule_critique) and list "
                    "the row in your post."]
    plan["rule"] = (RULE_DRY if args.dry_run else "") + RULE_END
    plan["rule_critique"] = RULE_CRITIQUE
    return plan


def offline_critique_plan(args: argparse.Namespace, now: dt.datetime, data_source: str,
                          actor: str = record.UNKNOWN_ACTOR) -> dict[str, Any]:
    """What a sitting does with its Notion tools to add one critique when no token lets the verb do it."""
    line = slots.note_entry(slots.CRITIQUE, actor, now, args.text)
    change = {"Progress note": f"<the Progress note as it is>{slots.COMPLAINT_JOIN}{line}"}
    return {
        "ok": False, "outcome": "no_token", "verb": CRITIQUE_VERB,
        "error": f"{slots.ENV_TOKEN} is not set (process environment or a .env file), so nothing was sent to "
                 "Notion; make the calls below with your Notion tools instead",
        "write": slots.mcp_properties(change),
        "steps": [slots._fetch_step(args.critique, "read the row: it must be Queued, ask for more than 30 minutes, "
                                    "hold a Justification line in its Progress note and have no Critique line yet"),
                  slots._update_step(args.critique, change, "add the critique line after the text the Progress note "
                                     "already holds")],
        "rule": RULE_CRITIQUE,
    }


# --- the command line ------------------------------------------------------------------------

def _gpu_choice(value: str) -> str:
    choice = value.strip().upper()
    if choice not in CHOICES:
        raise argparse.ArgumentTypeError(f"{value!r} is not a GPU choice; use A, B or all")
    return choice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab referee", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", type=_gpu_choice, default=None, metavar="{A,B,all}",
                        help="which card to referee: B (the exclusive card), A (the shared teacher pod) or both (default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write nothing to Notion; report the slots a real call would end and what stands after")
    parser.add_argument("--critique", type=slots._row_type, metavar="ID_OR_URL",
                        help="instead of the check, write the one critique of the justification on this row "
                             "(needs --text)")
    parser.add_argument("--text", type=slots._text_type(slots.MAX_CRITIQUE_CHARS, "--text"), metavar="TEXT",
                        help="the critique: one line on how the work could be done more efficiently")
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """The arguments of a referee call; SystemExit(2), argparse's usage message on stderr, for bad ones. gpu is
    ALL, A or B for a check and None for a critique."""
    parser = build_parser()
    args = parser.parse_args(list(argv))
    if args.critique is None:
        if args.text is not None:
            parser.error("--text goes with --critique")
        args.gpu = args.gpu or "ALL"
    else:
        if args.text is None:
            parser.error("--critique needs --text: one line on how the work could be done more efficiently")
        if args.gpu is not None or args.dry_run:
            parser.error("--critique writes one critique on one row; --gpu and --dry-run belong to the check")
    return args


def _failure(exc: notion.NotionError) -> dict[str, Any]:
    """The answer to Notion trouble: the one the queue verbs give (with its hint on what to do), under this verb."""
    return {**slots._failure(argparse.Namespace(sub=VERB), exc), "verb": VERB}


def _who(environ: Mapping[str, str]) -> dict[str, Any]:
    return {"actor": record.current_actor(environ), "tier": record.current_tier(environ),
            "task": record.current_task(environ)}


def _node_of(slot: slots.Slot) -> str | None:
    letter = CARDS_BY_NAME.get(slot.gpu)
    return slots.NODES[letter] if letter else None


def _record_end(slot: slots.Slot, entry: Mapping[str, Any], environ: Mapping[str, str],
                state_dir: str | os.PathLike[str] | None) -> int:
    """The ledger row of a slot ended: the spelling alerts.py reads as the end of a slot."""
    who = _who(environ)
    who["task"] = slot.task_id or who["task"]
    return record.record(
        VERB, "ended", metrics={"slot.ran_min": entry["ran_min"], "slot.overrun_min": entry["overrun_min"]},
        refs={"slot": slot.id}, state_dir=state_dir, node=_node_of(slot), gpu=slot.gpu,
        minutes=slots.tidy(slot.minutes_or_default),
        note=f"ran {entry['ran_min']} min against a cap of {entry['cap_min']}, {entry['overrun_min']} min over",
        **who)


def _record_failure(slot: slots.Slot, entry: Mapping[str, Any], environ: Mapping[str, str],
                    state_dir: str | os.PathLike[str] | None) -> int:
    """The ledger row of a slot that could not be ended (alerts.py ignores a failed outcome)."""
    who = _who(environ)
    who["task"] = slot.task_id or who["task"]
    return record.record(VERB, "error", refs={"slot": slot.id}, state_dir=state_dir, node=_node_of(slot),
                         gpu=slot.gpu, kind=entry["kind"], note=flat(entry["error"], record.NOTE_CHARS), **who)


#: The counts of an answer that the call's own ledger row keeps: field name -> the answer's list.
_COUNTS = {"slots_ended": "ended", "slots_would_end": "would_end", "slots_failed": "failed",
           "slots_skipped": "skipped", "complaints": "complaints", "complaint_due": "complaint_due",
           "critique_due": "critique_due", "awaiting_reply": "awaiting_reply", "could_start": "could_start",
           "slots_running": "running", "anomalies": "anomalies"}


def _record_scan(answer: Mapping[str, Any], environ: Mapping[str, str],
                 state_dir: str | os.PathLike[str] | None) -> int:
    """The ledger row of the call itself: its outcome and what it found, as counts."""
    extras: dict[str, Any] = {name: len(answer[key]) for name, key in _COUNTS.items()
                              if isinstance(answer.get(key), list)}
    extras["dry_run"] = bool(answer.get("dry_run"))
    if isinstance(answer.get("kind"), str):
        extras["kind"] = answer["kind"]
    if answer.get("error"):
        extras["note"] = flat(answer["error"], record.NOTE_CHARS)
    return record.record(SCAN_VERB, answer["outcome"], refs={}, state_dir=state_dir, **_who(environ), **extras)


def _record_critique(answer: Mapping[str, Any], row_id: str, environ: Mapping[str, str],
                     state_dir: str | os.PathLike[str] | None) -> int:
    """The ledger row of a critique call: the decision (critiqued, or why it was refused) on the row's slot."""
    who = _who(environ)
    shown = answer.get("slot") if isinstance(answer.get("slot"), Mapping) else {}
    who["task"] = shown.get("task") or who["task"]
    letter = next((key for key, name in slots.GPUS.items() if name == shown.get("gpu")), None)
    extras: dict[str, Any] = {name: shown[name] for name in ("gpu", "minutes") if shown.get(name) is not None}
    for name in ("reason", "kind"):
        if isinstance(answer.get(name), str):
            extras[name] = answer[name]
    if isinstance(answer.get("line"), str):
        extras["note"] = flat(answer["line"], record.NOTE_CHARS)
    return record.record(CRITIQUE_VERB, answer["outcome"], refs={"slot": answer.get("row") or row_id},
                         state_dir=state_dir, node=slots.NODES[letter] if letter else None, **extras, **who)


def run(argv: Sequence[str], *, client: notion.NotionClient | None = None, environ: Mapping[str, str] | None = None,
        env_files: Sequence[Any] | None = None, now: Callable[[], dt.datetime] | None = None,
        state_dir: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], int]:
    """One referee call: (the answer, the exit code). client replaces the Notion client the token would make, and
    now the clock; environ and env_files say where NOTION_TOKEN is looked up. The ledger rows are written here."""
    environ = os.environ if environ is None else environ
    clock = now or slots.utc_now
    args = parse_args(argv)
    data_source = slots.data_source_of(environ)
    client = client or slots.connect(environ, env_files)
    if args.critique is not None:
        return _run_critique(args, client, environ, clock, data_source, state_dir)
    if client is None:
        answer, code = offline_plan(args, clock(), data_source), 2
    else:
        queue = slots.SlotQueue(client, data_source=data_source, now=clock)
        try:
            answer = check(queue, clock, letters_of(args.gpu), dry_run=args.dry_run,
                           on_ended=lambda slot, entry: _record_end(slot, entry, environ, state_dir),
                           on_failed=lambda slot, entry: _record_failure(slot, entry, environ, state_dir))
        except notion.NotionError as exc:
            answer = _failure(exc)
        code = 0 if answer["ok"] else 1
    try:
        answer["ledger_seq"] = _record_scan(answer, environ, state_dir)
    except Exception as exc:
        answer["ledger_seq"] = None
        answer["ledger_error"] = f"{type(exc).__name__}: {exc}"
        log.error("referee: no ledger row written for the call (%s): %s", answer["outcome"], exc)
    return answer, code


def _run_critique(args: argparse.Namespace, client: notion.NotionClient | None, environ: Mapping[str, str],
                  clock: Callable[[], dt.datetime], data_source: str,
                  state_dir: str | os.PathLike[str] | None) -> tuple[dict[str, Any], int]:
    """A call with --critique: the one critique on one row, and its ledger row."""
    actor = record.current_actor(environ)
    if client is None:
        answer, code = offline_critique_plan(args, clock(), data_source, actor), 2
    else:
        queue = slots.SlotQueue(client, data_source=data_source, now=clock)
        try:
            answer = critique(queue, clock, args.critique, args.text, actor=actor)
        except notion.NotionError as exc:
            answer = {**slots._failure(argparse.Namespace(sub="critique"), exc), "verb": CRITIQUE_VERB}
        code = 0 if answer["ok"] else 1
    try:
        answer["ledger_seq"] = _record_critique(answer, args.critique, environ, state_dir)
    except Exception as exc:
        answer["ledger_seq"] = None
        answer["ledger_error"] = f"{type(exc).__name__}: {exc}"
        log.error("referee: no ledger row written for the critique (%s): %s", answer["outcome"], exc)
    return answer, code


@stop.guarded
def main(argv: Sequence[str]) -> int:
    """Referee of the GPU queue: end slots past their cap, list complaints and rows that could start."""
    answer, code = run(argv)
    print(json.dumps(answer))
    return code
