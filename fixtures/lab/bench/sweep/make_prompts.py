"""Writes the two text prompt sets of the request-side settings sweep (lab/sweep_request.py) beside this file.

    python fixtures/lab/bench/sweep/make_prompts.py            # write short.json and long.json
    python fixtures/lab/bench/sweep/make_prompts.py --check    # exit 1 when a written file differs from what it makes

short.json holds four requests of a few sentences each, two of them the text prompts of the benchmark's own set
(fixtures/lab/bench/prompts.json), about the length of the text request the teacher's baseline was measured on.
long.json holds four requests that each carry a household log of sixty lines (about a thousand tokens, the prompt
length of a 1280x720 caption request) and a question about it. Text only, never camera frames. The files come out
the same byte for byte on every run (no clock, no random numbers), so a set's sha256 names it: a change here is a
new prompt set and a new grid run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = "You are a careful assistant in a home. Answer in plain sentences."

SHORT = [
    ("text-kitchen", "Describe, step by step, how to tidy a small kitchen after cooking dinner for four people. Cover "
                     "the counters, the stove, the sink, the dishes and the floor, in that order, and say why each step "
                     "comes where it does."),
    ("text-schedule", "A family of three shares one car. Write a weekday plan that gets one parent to work by 8:30, a "
                      "child to school by 8:00 and the other parent to a 9:15 appointment, and explain each choice."),
    ("text-laundry", "Explain how to sort, wash and dry a week of laundry for a family of four, and say which steps "
                     "save the most energy and why."),
    ("text-plants", "Give a weekly routine for keeping six houseplants healthy in a flat with one sunny window, and "
                    "explain the reason for each step."),
]

ROOMS = ["Kitchen", "Hallway", "Living room", "Bathroom", "Bedroom", "Study", "Laundry", "Garden"]
EVENTS = [
    "the kettle was switched on and left to boil",
    "a window was opened for fresh air",
    "the recycling bin was emptied into the outdoor container",
    "two mugs and a plate were left on the counter",
    "the floor was swept and the dust pan emptied",
    "a delivery box was opened and flattened for recycling",
    "the heating was turned down by one degree",
    "the plants on the sill were watered",
    "the dishwasher was started on the eco programme",
    "a towel was hung on the radiator to dry",
    "the lights were switched off when the room was empty",
    "a shopping list was written on the fridge board",
    "the laundry basket was carried to the washing machine",
    "the door was closed to keep the warmth in",
    "a book was put back on the shelf",
    "the bed was made and the curtains were opened",
    "the bin bag was tied and taken out",
    "the table was wiped after breakfast",
    "a glass of water was poured and left half full",
    "the vacuum cleaner was put on its charger",
    "the washing machine finished and beeped twice",
    "wet clothes were moved to the drying rack",
    "the oven was preheated and then switched off unused",
    "a tap was left dripping for a few minutes",
    "the fruit bowl was refilled with apples and pears",
    "a phone was left charging on the side table",
]
LINES = 60
QUESTIONS = [
    "Read the whole log, then go through it room by room: say what happened in each room, in time order, and what "
    "should be done next there.",
    "List every task in the log that was left unfinished, say which room it was in and when it was last mentioned, "
    "and suggest who in the family should finish it.",
    "Write a short report of the morning for someone who was away: what changed in the house, what used energy, and "
    "what to check on arrival.",
    "Group the events by kind (cleaning, energy, food, laundry, rubbish) and for each group say how often it happened "
    "and whether the pattern looks healthy.",
]


def log_text(offset: int) -> str:
    """Sixty log lines, six minutes apart from 07:00, the rooms and events walked with a fixed stride."""
    lines = []
    for k in range(LINES):
        minutes = 7 * 60 + 6 * k
        room = ROOMS[(k + offset) % len(ROOMS)]
        event = EVENTS[(5 * k + 3 * offset) % len(EVENTS)]
        lines.append(f"{minutes // 60:02d}:{minutes % 60:02d} {room}: {event}.")
    return "\n".join(lines)


def short_set() -> dict:
    return {
        "id": "sweep-short-v1",
        "about": "The short prompt class of the request-side settings sweep (lab/sweep_request.py): four text requests "
                 "of a few sentences, two of them the text prompts of fixtures/lab/bench/prompts.json. Text only, "
                 "never camera frames. Written by make_prompts.py; a change is a new prompt set.",
        "prompts": [{"id": pid, "kind": "text", "system": SYSTEM, "user": user} for pid, user in SHORT],
    }


def long_set() -> dict:
    prompts = []
    for offset, question in enumerate(QUESTIONS):
        user = ("Here is the household log for one morning, one line per event.\n\n" + log_text(offset)
                + "\n\n" + question)
        prompts.append({"id": f"log-{offset + 1}", "kind": "text", "system": SYSTEM, "user": user})
    return {
        "id": "sweep-long-v1",
        "about": "The long prompt class of the request-side settings sweep (lab/sweep_request.py): four text requests, "
                 "each a household log of sixty lines (about a thousand tokens) and a question about it. Text only, "
                 "never camera frames. Written by make_prompts.py; a change is a new prompt set.",
        "prompts": prompts,
    }


def render(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def main(argv: list[str]) -> int:
    outputs = {"short.json": render(short_set()), "long.json": render(long_set())}
    if "--check" in argv:
        stale = [name for name, data in outputs.items() if (HERE / name).read_bytes() != data]
        print(json.dumps({"ok": not stale, "stale": stale}))
        return 1 if stale else 0
    for name, data in outputs.items():
        (HERE / name).write_bytes(data)
    print(json.dumps({"ok": True, "written": sorted(outputs),
                      "chars": {name: max(len(p["system"]) + len(p["user"]) for p in json.loads(data)["prompts"])
                                for name, data in outputs.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
