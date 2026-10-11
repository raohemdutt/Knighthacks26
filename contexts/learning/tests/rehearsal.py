"""The live-loop rehearsal (fixtures/learning/live-loop-2026-10-11) as test data with its stamp and run ids changed.

The box never serves a run a committed rehearsal names (lab_loops.fixture_run_ids), so tests that post the rehearsal
the way the relay posts a live run give it a stamp of its own: the shapes and the numbers stay the night's, the ids
are not the fixture's.
"""

import json
from pathlib import Path

REHEARSAL = Path(__file__).resolve().parents[3] / "fixtures" / "learning" / "live-loop-2026-10-11"
FIXTURE_STAMP, STAMP = "1011-040342", "1011-990001"
SOLO, DUO = f"live-{STAMP}-solo", f"live-{STAMP}-duo"
FIXTURE_SOLO, FIXTURE_DUO = f"live-{FIXTURE_STAMP}-solo", f"live-{FIXTURE_STAMP}-duo"


def text(name: str) -> str:
    return (REHEARSAL / name).read_text(encoding="utf-8").replace(FIXTURE_STAMP, STAMP)


def raw(name: str) -> bytes:
    return text(name).encode("utf-8")


def copy_to(folder: Path) -> Path:
    """The rehearsal folder's files, relabelled, under `folder` (a relay --out folder copied onto a box by hand)."""
    for path in REHEARSAL.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
            target = folder / path.relative_to(REHEARSAL)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text(str(path.relative_to(REHEARSAL))), encoding="utf-8")
    return folder


def jsonl(name: str) -> list[dict]:
    return [json.loads(x) for x in text(name).splitlines() if x.strip()]
