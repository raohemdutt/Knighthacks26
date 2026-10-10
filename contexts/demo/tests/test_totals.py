import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from contexts.demo import totals
from contexts.demo.__main__ import main
from contexts.demo.seed import seed

T0 = datetime(2026, 10, 10, 1, 0, tzinfo=UTC)


def ev(kind: str, at: float, **fields) -> dict:
    return {"type": kind, "ts": (T0 + timedelta(seconds=at)).isoformat(), **fields}


def by_name(rows):
    return {r.name: r for r in rows}


def test_counts_come_from_the_events_and_say_where_from():
    events = [
        ev("experience.question.asked", 0),
        ev("agent.answer.generated", 2),
        ev("experience.question.asked", 60),
        ev("agent.alert.raised", 120),
        ev("experience.approval.decided", 130, decision="approve"),
        ev("experience.approval.decided", 140, decision="deny"),
        ev("memory.episode.closed", 300, start=(T0).isoformat(), end=(T0 + timedelta(minutes=30)).isoformat(),
           keyframe_refs=["a", "b"], captions=["x"]),
    ]
    rows = by_name(totals.totals(events))
    assert rows["Questions asked"].value == "2"
    assert rows["Questions asked"].source == "`experience.question.asked` x 2"
    assert rows["Questions asked"].first == T0
    assert rows["Answers given, with citations"].value == "1"
    assert rows["Approval decisions (approve / deny)"].value == "1 / 1"
    assert rows["Keyframes kept (faces blurred)"].value == "2"
    assert "0.5 h of activity" in rows["Episodes remembered"].source


def test_missing_numbers_say_so_instead_of_zero():
    rows = by_name(totals.totals([ev("experience.question.asked", 0)]))
    assert rows["Alerts raised"].value == "none in the log"
    assert rows["Energy used by the Jetson"].value == "none in the log"
    assert rows["Frames seen by the cameras"].value == "none in the log"


def test_running_hours_skip_the_box_being_off():
    events = [ev("agent.note.written", s) for s in (0, 300, 600)] + [ev("agent.note.written", 600 + 7200)]
    rows = by_name(totals.totals(events))
    assert rows["Hours Atlas was running"].value == "0.2 h"  # 10 minutes; the two-hour gap is off


def test_frames_and_joules_are_integrated_and_marked_estimated():
    events = [ev("capture.camera.health", s, camera_id="cam-desk", status="ok", fps_actual=15.0) for s in (0, 30, 60)]
    events += [ev("telemetry.jetson.power", s, node="jetson", rails={"VDD_IN": 7200.0}) for s in (0, 1800, 3600)]
    events += [ev("telemetry.gpu.power", s, node="gpu-a", socket_w=200.0) for s in range(0, 3601, 60)]
    rows = by_name(totals.totals(events))
    assert rows["Frames seen by the cameras"].value == "~900"
    assert rows["Frames seen by the cameras"].source.startswith("estimated")
    # 30-minute power gaps are longer than one sample covers, so they add nothing
    assert rows["Energy used by the Jetson"].value == "none in the log"
    assert rows["Energy used by the R9700s"].value == "200.0 Wh"
    assert "(gpu-a)" in rows["Energy used by the R9700s"].source


def test_since_drops_earlier_events_and_bad_lines_are_skipped(tmp_path):
    (tmp_path / "2026-10-10.jsonl").write_text(
        json.dumps(ev("experience.question.asked", 0)) + "\nnot json\n" + json.dumps(ev("experience.question.asked", 3600)) + "\n",
        encoding="utf-8")
    rows = by_name(totals.totals(totals.read_events(tmp_path), since=T0 + timedelta(minutes=30)))
    assert rows["Questions asked"].value == "1"


def test_the_seeded_demo_night_prints_a_full_card(tmp_path, capsys):
    asyncio.run(seed(tmp_path, now=datetime.now(UTC).replace(microsecond=0)))
    assert main(["totals", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| Number | Value | Source | Events from |")
    rows = by_name(totals.totals(totals.read_events(tmp_path / "events")))
    assert rows["Episodes remembered"].count > 0
    assert rows["Hours Atlas was running"].value != "none in the log"
    assert main(["totals", "--data-dir", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["name"] == "Hours Atlas was running"


def test_since_needs_an_offset():
    with pytest.raises(SystemExit):
        main(["totals", "--since", "2026-10-10T09:00:00"])


def test_only_actions_that_ran_count_as_carried_out():
    events = [ev("workspace.action.completed", n, action_id=str(n), status=s)
              for n, s in enumerate(["expired", "denied", "done", "expired"])]
    row = by_name(totals.totals(events))["Outside actions carried out, each after an approval"]
    assert row.value == "1"
    assert row.source == "`workspace.action.completed` with status done, of 4 (1 denied, 2 expired)"
