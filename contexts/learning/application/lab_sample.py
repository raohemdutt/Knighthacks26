"""The committed sample run, for the Learning lab when the box has no real run (or `?sample=1`).

    view = load_sample()          # a LabView(sample=True) folded from fixtures/learning/sample_run/events.jsonl

The sample is NOT measured on our hardware. It never goes on the event bus, so no other projection
(totals, the League, slides) can count it; it lives only in this separate view, and everything that
view returns is marked `sample: true`. Every line of the file must say `"sample": true` itself, so a
real log can never be loaded as a sample by mistake, nor a sample line pass as a real event.
`fixtures/learning/make_sample_run.py` writes the file.
"""

import json
from functools import lru_cache
from pathlib import Path

from contexts.learning.application.lab_view import LabView, fold
from contracts import parse_event
from contracts.events import Event

SAMPLE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "learning" / "sample_run"


class SampleError(ValueError):
    pass


def read_sample_events(path: Path) -> list[Event]:
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.pop("sample", None) is not True:
            raise SampleError(f"{path.name}:{n} does not say it is a sample")
        events.append(parse_event(raw))
    return events


def load_sample(sample_dir: str | Path = SAMPLE_DIR) -> LabView:
    root = Path(sample_dir)
    return fold(LabView(exports_dir=root / "dataset", sample=True), read_sample_events(root / "events.jsonl"))


@lru_cache(maxsize=4)
def cached_sample(sample_dir: str) -> LabView | None:
    """None when the sample is not on this machine (a box installed without fixtures)."""
    try:
        return load_sample(sample_dir)
    except (OSError, ValueError):
        return None
