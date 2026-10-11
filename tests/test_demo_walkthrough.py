"""The pure parts of scripts/demo_walkthrough.py: the results table, the counts and exit code, and the guard
that decides which approval cards a script may approve. The guard matters most: a mistake there posts to a
group chat or writes someone's calendar from an unattended run."""

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contracts.agent import ActionKind

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("demo_walkthrough", ROOT / "scripts" / "demo_walkthrough.py")
dw = importlib.util.module_from_spec(spec)
sys.modules["demo_walkthrough"] = dw  # dataclasses look their module up by name
spec.loader.exec_module(dw)


# ---------- the card guard ----------

def card(kind="connector_write", target="calendar.create_event", reason="", params=None):
    return {"id": "a1", "kind": kind, "target": target, "reason": reason, "params": params or {}, "status": "proposed"}


def test_a_plan_kept_on_a_box_without_a_calendar_is_local():
    reason = ("Replace today's Day Run blocks with 2 block(s) for Sat Oct 10: Eat 8 PM, Sleep 3 AM (Win, 2 stars). "
              "No calendar connector on this box: approving keeps this as today's plan, nothing is written")
    ok, why = dw.local_only_card(card(reason=reason, params={"events": []}))
    assert ok and "nothing is written" in why


def test_the_same_calendar_card_with_a_calendar_connected_is_refused():
    # what ConnectorHub.propose_write raises when a calendar can be written: same kind and target, no such sentence
    ok, why = dw.local_only_card(card(reason="Add 2 block(s) for Sat Oct 10: Eat 8 PM, Sleep 3 AM (Win, 2 stars)"))
    assert not ok and "real calendar" in why


def test_the_planner_still_writes_the_sentences_the_guard_reads():
    """The guard recognises the local plan card by the planner's own words: if they change, this fails first."""
    from contexts.day.application import planner

    source = Path(planner.__file__).read_text(encoding="utf-8")
    assert dw.NO_CALENDAR in source and dw.NOTHING_WRITTEN in source
    assert planner.CALENDAR_WRITE == "calendar.create_event"


@pytest.mark.parametrize("target", ["spot.reinspect", "spot.work_order"])
def test_an_inspection_work_order_is_local(target):
    ok, _ = dw.local_only_card(card(target=target, reason="Re-inspect HX-202 ...", params={"asset_id": "HX-202", "stays_on_this_box": True}))
    assert ok


def test_the_real_work_order_cards_pass_the_guard():
    """Built by the product code, not by hand: both operator actions on the sponsor's kit."""
    from contexts.spot.application import workorders

    assert set(workorders.TARGETS.values()) == {"spot.reinspect", "spot.work_order"}
    assert workorders.KIND == "connector_write"
    source = Path(workorders.__file__).read_text(encoding="utf-8")
    assert source.count('"stays_on_this_box": True') == 2


@pytest.mark.parametrize("params", [{}, {"stays_on_this_box": False}, {"stays_on_this_box": "yes"}, {"stays_on_this_box": 1}])
def test_a_spot_card_without_the_flag_is_refused(params):
    assert not dw.local_only_card(card(target="spot.reinspect", params=params))[0]


@pytest.mark.parametrize("kind,target", [
    ("send_message", "groupme"),
    ("send_message", "ntfy:dominick"),
    ("circle_share", "demo-circle"),
    ("export_training_data", "desk-objects"),
    ("browse", "https://example.com"),
    ("connector_write", "groupme.post_message"),
    ("connector_write", "canvas.submit"),
    ("connector_write", "spot.reinspect.elsewhere"),
    ("connector_write", "calendar.delete_event"),
    ("something_new", "anything"),
])
def test_everything_else_counts_as_leaving_the_box(kind, target):
    # even when it claims to be local in its text or its params
    lying = card(kind=kind, target=target, params={"stays_on_this_box": True},
                 reason="No calendar connector on this box: approving keeps this as today's plan, nothing is written")
    ok, why = dw.local_only_card(lying)
    assert not ok and why


def test_every_action_kind_in_the_contract_is_decided_on_purpose():
    """A kind added to contracts.agent.ActionKind is refused until someone lists it here."""
    kinds = set(ActionKind.__args__)
    assert kinds == set(dw.OUTSIDE_KINDS) | {"connector_write"}
    for kind in dw.OUTSIDE_KINDS:
        assert not dw.local_only_card(card(kind=kind, target="spot.reinspect", params={"stays_on_this_box": True}))[0]


@pytest.mark.parametrize("bad", [None, "card", [], {}, {"kind": "connector_write"}, {"kind": None, "target": "spot.reinspect"},
                                 {"kind": "connector_write", "target": "spot.reinspect", "params": "stays_on_this_box"},
                                 {"kind": "connector_write", "target": "calendar.create_event", "reason": None}])
def test_a_malformed_card_is_refused(bad):
    assert dw.local_only_card(bad)[0] is False


class FakeApi:
    """Records every call; serves one pending card."""
    proxy = None

    def __init__(self, cards):
        self.cards, self.calls = cards, []

    def get(self, path, timeout=30):
        self.calls.append(("GET", path, None))
        return dw.Resp(200, self.cards, content_type="application/json")

    def post(self, path, body=None, timeout=30):
        self.calls.append(("POST", path, body))
        return dw.Resp(200, {"status": "approved" if body["decision"] == "approve" else "denied"}, content_type="application/json")


def decisions(api):
    return [body["decision"] for method, _, body in api.calls if method == "POST"]


def test_settle_card_never_approves_an_outside_card_even_when_asked_to_approve(capsys):
    api = FakeApi([card(kind="send_message", target="groupme") | {"id": "x"}])
    run = dw.Run(api, approve_local=True)
    assert run.settle_card("a message", "x", expect_local=False) is False
    assert decisions(api) == ["deny"]
    assert run.rows[-1].status == dw.PASS and "denied" in run.rows[-1].note


def test_settle_card_denies_and_fails_when_a_card_expected_local_is_not(capsys):
    api = FakeApi([card(reason="Add 1 block(s)") | {"id": "x"}])  # a calendar is connected
    run = dw.Run(api, approve_local=True)
    assert run.settle_card("the plan card", "x", expect_local=True) is False
    assert decisions(api) == ["deny"]
    assert run.rows[-1].status == dw.FAIL


def test_settle_card_approves_a_local_card_only_with_the_flag(capsys):
    local = card(target="spot.reinspect", params={"stays_on_this_box": True}) | {"id": "x"}
    api = FakeApi([local])
    assert dw.Run(api, approve_local=False).settle_card("work order", "x", expect_local=True) is False
    assert decisions(api) == ["deny"]
    api = FakeApi([local])
    assert dw.Run(api, approve_local=True).settle_card("work order", "x", expect_local=True) is True
    assert decisions(api) == ["approve"]


def test_settle_card_fails_without_deciding_when_the_card_is_not_listed(capsys):
    api = FakeApi([])
    run = dw.Run(api, approve_local=True)
    assert run.settle_card("work order", "gone", expect_local=True) is False
    assert decisions(api) == [] and run.rows[-1].status == dw.FAIL


# ---------- the table ----------

ROWS = [
    dw.Row("1 Ask", "the box answers", dw.PASS, 0.004, "model fake-0"),
    dw.Row("1 Ask", "a long step name here", dw.FAIL, 12.3456, "HTTP 500"),
    dw.Row("12 Voice", "hands-free", dw.SKIP, None, "needs a person: speak"),
]


def test_the_table_has_the_five_columns_aligned():
    lines = dw.format_table(ROWS).splitlines()
    assert [c.strip() for c in lines[0].split("|")] == ["use case", "step", "result", "seconds", "note"]
    assert set(lines[1]) == {"-", "+"}
    bars = [[i for i, ch in enumerate(line) if ch == "|"] for line in (lines[0], *lines[2:])]
    assert all(b[:4] == bars[0][:4] for b in bars)  # the first four separators line up on every row
    assert [c.strip() for c in lines[3].split("|")] == ["1 Ask", "a long step name here", "FAIL", "12.35", "HTTP 500"]
    assert [c.strip() for c in lines[4].split("|")][3] == ""  # no time for a step that was not run
    assert all(line == line.rstrip() for line in lines)


def test_an_empty_table_is_just_the_header():
    assert len(dw.format_table([]).splitlines()) == 2


def test_the_markdown_table_has_one_row_per_step():
    lines = dw.markdown_table(ROWS).splitlines()
    assert lines[1] == "| --- | --- | --- | --- | --- |"
    assert lines[2] == "| 1 Ask | the box answers | PASS | 0.00 | model fake-0 |"
    assert len(lines) == 2 + len(ROWS)


def test_one_line_folds_whitespace_keeps_the_table_whole_and_cuts_long_text():
    assert dw.one_line("a\n  b\tc") == "a b c"
    assert "|" not in dw.one_line("x | y")
    cut = dw.one_line("word " * 100, 40)
    assert len(cut) <= 40 and cut.endswith("…")


def test_counts_and_exit_code():
    assert dw.counts(ROWS) == {"PASS": 1, "FAIL": 1, "SKIP": 1}
    assert dw.by_use_case(ROWS) == {"1 Ask": {"PASS": 1, "FAIL": 1, "SKIP": 0}, "12 Voice": {"PASS": 0, "FAIL": 0, "SKIP": 1}}
    assert dw.exit_code(ROWS) == 1
    assert dw.exit_code([r for r in ROWS if r.status != dw.FAIL]) == 0  # a SKIP alone does not fail the run
    assert dw.exit_code([]) == 0


def test_every_use_case_has_a_step_function_and_the_run_sheet_lists_it():
    keys = [k for k, _ in dw.USE_CASES]
    assert len(keys) == len(set(keys)) == 15
    sheet = (ROOT / "docs" / "demo-run-sheet.md").read_text(encoding="utf-8")
    for n in range(1, 16):
        assert f"\n## {n}. " in sheet, f"use case {n} has no section in docs/demo-run-sheet.md"


# ---------- flags ----------

def test_read_only_asks_nothing_and_writes_nothing(capsys):
    class ReadOnlyApi(FakeApi):
        def call(self, method, path, body=None, timeout=30):
            raise AssertionError(f"{method} {path} in a read-only run")

        def post(self, path, body=None, timeout=30):
            raise AssertionError(f"POST {path} in a read-only run")

        def get(self, path, timeout=30):
            self.calls.append(("GET", path, None))
            return dw.Resp(404, {"detail": "not here"}, content_type="application/json")

    run = dw.Run(ReadOnlyApi([]), read_only=True, ask=False)
    dw.walk(run, None, scene_pushed=False, expect_profile=None)
    assert run.rows and all(method == "GET" for method, _, _ in run.api.calls)


def test_the_scene_is_never_pushed_read_only_or_through_a_proxy(monkeypatch, capsys):
    monkeypatch.setattr(dw.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pushed")))
    for run in (dw.Run(FakeApi([]), read_only=True), dw.Run(type("P", (FakeApi,), {"proxy": "socks5h://x"})([]))):
        dw.push_scene(run, "http://box:8000", None)
        assert run.rows[-1].status == dw.SKIP


def test_the_token_is_not_on_the_curl_command_line(monkeypatch):
    seen = {}

    def fake_run(cmd, input=None, **kwargs):
        seen["cmd"], seen["stdin"] = cmd, input.decode()
        return type("P", (), {"returncode": 0, "stdout": b'{"ok": true}\n200\tapplication/json', "stderr": b""})()

    monkeypatch.setattr(dw.subprocess, "run", fake_run)
    r = dw.Client("http://box:8000/", token="s3cret", proxy="socks5h://127.0.0.1:1055").post("/api/ask", {"text": 'say "hi"'})
    assert r.status == 200 and r.data == {"ok": True}
    assert not any("s3cret" in part for part in seen["cmd"])
    assert "Authorization: Bearer s3cret" in seen["stdin"]
    assert json.loads(json.loads(seen["stdin"].split("data-raw = ", 1)[1])) == {"text": 'say "hi"'}
    assert seen["cmd"][-1] == "http://box:8000/api/ask" and "--proxy" in seen["cmd"]


def test_the_json_output_is_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dw.Client, "call", lambda self, method, path, body=None, timeout=30: dw.Resp(404, {"detail": "nothing here"}, content_type="application/json")
                        if path != "/health" else dw.Resp(200, {"ok": True, "model": "fake-0"}, content_type="application/json"))
    out = tmp_path / "r.json"
    code = dw.main(["--gateway", "http://127.0.0.1:1", "--read-only", "--only", "voice", "--json", str(out)])
    body = json.loads(out.read_text())
    assert code == 0 and body["counts"]["FAIL"] == 0 and body["counts"]["SKIP"] == 4
    assert {r["status"] for r in body["rows"]} == {"SKIP"} and body["model"] == "fake-0"
    assert datetime.fromisoformat(body["started"]) <= datetime.now(UTC)
