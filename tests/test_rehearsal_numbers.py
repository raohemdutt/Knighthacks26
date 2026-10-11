"""The People screen's "Your group" quotes the recorded rehearsal where a box has no comparison of its own.

web/lib/rehearsal.ts copies those numbers from the committed record; this fails if the two ever disagree, so a
figure on that screen is never one somebody typed.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "fixtures" / "learning" / "live-loop-2026-10-11" / "comparison.json"
COPY = ROOT / "web" / "lib" / "rehearsal.ts"


def test_the_rehearsal_numbers_on_the_people_screen_are_the_recorded_ones():
    record, copy = json.loads(RECORD.read_text()), COPY.read_text()
    assert RECORD.relative_to(ROOT).as_posix() in copy
    for act in ("solo", "duo"):
        got = record["acts"][act]
        line = re.search(rf"{act}: \{{ examples: ([\d.]+), pods_joules: ([\d.]+), j_per_example: ([\d.]+) \}}", copy)
        assert line, act
        assert [float(x) for x in line.groups()] == [got["examples"], got["pods_joules"], got["j_per_example"]]
    ratio = re.search(r"ratio: \{ examples: ([\d.]+), pods_joules: ([\d.]+) \}", copy)
    assert [float(x) for x in ratio.groups()] == [record["duo_over_solo"]["examples"], record["duo_over_solo"]["pods_joules"]]
    assert f'teacher: "{record["acts"]["duo"]["labeling"]["node"]}"' in copy
    assert f'trainer: "{record["acts"]["duo"]["training"]["node"]}"' in copy
