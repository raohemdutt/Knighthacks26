"""The operator's words (application/plain.py): every triage item and every finding becomes a sentence a person on shift can read,
with no raw enum, no bare id, no statistics vocabulary, and the same numbers as the analysis behind it."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application import plain
from contexts.spot.application.assistant import Assistant
from contexts.spot.application.service import Spot
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.helpers import small_kit
from runtime import FakeRuntime

ENUM = re.compile(r"\b[A-Z]{2,}(?:_[A-Z]+)+\b|\b(?:CRITICAL|HIGH|MEDIUM|LOW|COMPLETED|FAILED|SUCCESS|GOOD|QUESTIONABLE|BASELINE|IMPROVED|NAVIGATION)\b")
ROW_ID = re.compile(r"\b(?:AN\d{3}|M\d{3}|A\d{4}|R\d{4}|V\d)\b")  # row ids and version codes never reach an operator sentence
TAG = re.compile(r"\b(?:[A-Z]{2,6}-\d+|WP\d{2})\b")  # equipment tags and waypoint codes: only in brackets, after a name
JARGON = ("p=", "p<", "interval", "anomal", "significan", "n=", "wilson", "percentile", "odds", "unverified", "criticality")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@pytest.fixture(scope="module")
def spot():
    return Spot(load_kit(Path("fixtures/spot/kit")))


@pytest.fixture(scope="module")
def view(spot):
    return plain.shift(spot)


def readable(text: str) -> None:
    assert text and (text[0].isupper() or text[0].isdigit() or text[0] == "\x27") and text.rstrip()[-1] in ".?!", text
    assert not ENUM.search(text), f"raw enum in: {text}"
    assert not ROW_ID.search(text), f"row id in: {text}"
    assert not any(j in text.lower() for j in JARGON), f"statistics word in: {text}"
    for m in TAG.finditer(text):  # "Heat exchanger 2 (HX-202)": a tag is always in brackets after a name
        assert text[m.start() - 1] == "(" and text[m.end()] == ")" and text[m.start() - 2] == " " and text[m.start() - 3].isalnum(), f"bare tag in: {text}"


def numbers(text: str) -> set[float]:
    return {float(n) for n in NUMBER.findall(TAG.sub(" ", text))}


def test_every_triage_item_yields_a_readable_sentence_with_a_level_a_why_and_a_button(spot, view):
    assert len(view["triage"]) == len(spot.triage) > 10
    for raw, it in zip(spot.triage, view["triage"]):
        assert it["id"] == raw["id"] and it["rank"] == raw["rank"]
        readable(it["sentence"])
        readable(it["headline"] + ".")
        assert it["level"] in ("act", "watch", "ok") and it["level_words"]
        assert it["why_ref"] == {"kind": "asset" if raw["kind"] == "asset" else "location", "id": raw["id"]}
        if raw["kind"] == "asset":
            want = "work_order" if raw["next_step"]["action"] == "dispatch" else "reinspect"
            assert it["action"]["kind"] == want and it["action"]["asset_id"] == raw["id"]
            assert (it["action"]["anomaly_id"] == raw["anomaly_id"]) if want == "work_order" else it["action"]["anomaly_id"] is None


def test_the_top_of_the_queue_reads_like_the_brief(view):
    first = view["do_first"][0]
    assert first["headline"] == "Send SPOT back to Heat exchanger 2 (HX-202)" and first["level"] == "act"
    assert first["sentence"] == ("SPOT flagged strong vibration in the last round, but that check failed (the reading was not reliable), so the alarm "
                                 "is not confirmed. This is critical equipment. Send SPOT back before sending a person.")
    second = view["do_first"][1]
    assert second["action"] == {"kind": "work_order", "label": "Send a technician", "asset_id": "HX-204", "anomaly_id": "AN014"}
    assert "the alarm stands" in second["sentence"]
    assert len(view["do_first"]) == 3 and view["do_first_total"] == view["status"]["act"] == sum(a["level"] == "act" for a in view["assets"])


def test_every_finding_kind_yields_a_sentence_whose_numbers_are_the_findings(spot, view):
    assert {f["id"] for f in view["findings"]} == set(spot.findings) and set(spot.findings) <= set(plain.FINDING_WORDS)
    for f in view["findings"]:
        readable(f["sentence"])
        readable(f["headline"] + ".")
        if f["note"]:
            readable(f["note"])
        assert f["why_ref"]["kind"] in ("finding", "location", "asset")
        stated = numbers(f["sentence"])
        for name, value in f["facts"].items():  # every fact is in the sentence, as written
            assert float(value) in stated, f"{f['id']}: {name}={value} not in: {f['sentence']}"
    f = {x["id"]: x for x in view["findings"]}
    raw = spot.findings
    low = [r for r in raw["battery"].rows if r.key in ("0-20", "20-40")]
    assert f["battery"]["facts"] == {"k": sum(r.k for r in low), "n": sum(r.n for r in low), "k_rest": 24, "n_rest": 199, "threshold": 40}
    assert f["battery"]["facts"]["n"] == raw["battery"].n and "about 3 times" in f["battery"]["sentence"]
    worst = raw["reliability_by_type"].rows[0]
    assert (f["reliability_by_type"]["facts"]["k"], f["reliability_by_type"]["facts"]["n"]) == (worst.k, worst.n) == (13, 43)
    assert "about 1 in 3" in f["reliability_by_type"]["sentence"] and "(acoustic)" in f["reliability_by_type"]["sentence"]
    assert f["reliability_by_location"]["facts"]["n"] == raw["reliability_by_location"].n == 22
    assert f["hidden_failures"]["facts"] == {"k": round(raw["hidden_failures"].value * raw["hidden_failures"].n), "n": raw["hidden_failures"].n}
    assert f["anomalies_hidden"]["facts"]["unsure"] == len(raw["anomalies_hidden"].extra["unverified"]) == 32
    assert "hint" in f["navigation_hotspot"]["sentence"] and f["navigation_hotspot"]["level"] == "ok"  # not established: said so, not pushed
    assert "cannot say the update is the reason" in f["version_comparison"]["sentence"]
    assert [w["id"] for w in view["watch"]] == ["battery", "reliability_by_type", "reliability_by_location"]
    assert all(w["visual"]["kind"] == "compare" and len(w["visual"]["bars"]) == 2 for w in view["watch"])


def test_every_asset_has_a_level_a_last_check_a_reading_and_a_sample(spot, view):
    assert len(view["assets"]) == len(spot.kit.assets) == 20
    for a in view["assets"]:
        for text in (a["sentence"], a["what"], a["last_check"]["sentence"], a["reading"]["sentence"], a["alarm"]["sentence"]):
            readable(text)
        assert a["label"].endswith(f"({a['asset_id']})") and not a["label"].startswith("(")
        assert (a["level"] == "ok") == (not a["reasons"])
        assert a["sample"] is None or "not a picture of this equipment" in a["sample"]["label"]
    by = {a["asset_id"]: a for a in view["assets"]}
    hx = by["HX-202"]
    assert hx["level"] == "act" and hx["last_check"]["worked"] is False and "4 days ago" in hx["last_check"]["sentence"]
    assert hx["reading"]["value"] == 75.5 and hx["reading"]["unit"] == "°C" and hx["reading"]["in_range"] is True
    assert "our own band" in hx["reading"]["range"]["words"]  # the kit has no engineering limits: said where the range is shown
    assert by["VALVE-207"]["level"] == "ok" and by["VALVE-207"]["alarm"]["sentence"] == "No alarm on record."
    assert by["GAUGE-209"]["level"] == "act" and "No check here has worked for 2 days" in by["GAUGE-209"]["sentence"]
    s = view["status"]
    assert s["act"] + s["watch"] + s["ok"] == 20 and s["sentence"].startswith("SPOT has run 27 rounds. 4 things need you.")
    assert view["fine"]["count"] == s["ok"] and str(s["ok"]) in view["fine"]["sentence"]
    assert view["trust"]["sentence"].startswith("21 of 26 completed rounds") and view["trust"]["why_ref"] == {"kind": "finding", "id": "hidden_failures"}


def test_rounds_have_plain_labels_and_every_step_is_a_sentence(spot, view):
    assert view["rounds"][-1]["label"] == "Round 27 · today 10:00 AM · 1 check failed" and view["latest_round"] == "M027"
    assert view["rounds"][16]["label"] == "Round 17 · Mon 7 Sep 2:00 AM · did not finish, 3 checks failed"
    assert view["rounds"][25]["result"] == "all checks worked"
    for m in spot.kit.missions:
        r = plain.round_view(spot, m.mission_id)
        readable(r["summary"])
        assert len(r["steps"]) == len(spot.kit.actions_of[m.mission_id])
        for st, a in zip(r["steps"], spot.kit.actions_of[m.mission_id]):
            readable(st["sentence"] + ".")
            assert st["worked"] == (not a.failed) and st["alarm"] == bool(spot.kit.anomalies_of.get(a.action_id)) and st["battery_pct"] == a.battery_pct
    r = plain.round_view(spot, "M027")
    assert r["steps"][4]["sentence"] == "10:12 AM · Listening check at Heat exchanger 2 · failed: the reading was not reliable · alarm: strong vibration (not confirmed)"
    assert r["steps"][3]["sentence"] == "10:10 AM · Heat check at Electrical panel 11 · worked · 57.8 °C"
    assert "9 of 10 checks worked" in r["summary"] and "marked completed even though a check failed" in r["summary"]
    assert plain.round_view(spot, "M999") is None
    assert view["map"]["route"][0] == spot.kit.actions_of["M027"][0].location_id and len(view["map"]["places"]) == 15


def test_words_for_time_names_and_shares():
    from datetime import datetime
    now = datetime(2026, 9, 10, 10, 25)
    assert plain.ago(now, datetime(2026, 9, 8, 10, 13)) == "2 days ago" and plain.ago(now, datetime(2026, 9, 10, 10, 5)) == "20 minutes ago"
    assert plain.ago(now, datetime(2026, 9, 9, 2, 6)) == "32 hours ago" and plain.span(1) == "1 hour"
    assert plain.when(now, datetime(2026, 9, 9, 18, 0)) == "yesterday 6:00 PM" and plain.clock(datetime(2026, 9, 10, 0, 5)) == "12:05 AM"
    assert plain.tidy("Heat Exchanger 02") == "Heat exchanger 2" and plain.article("Electrical Panel") == "an"
    assert plain.one_in(13, 43) == "about 1 in 3" and plain.one_in(8, 11) == "more than half" and plain.one_in(0, 5) == "none"
    assert plain.times((25, 71), (24, 199)) == "about 3 times" and plain.times((8, 22), (41, 248)) == "about twice" and plain.times((2, 10), (2, 10)) is None
    assert plain.soft("SOME_NEW_ENUM") == "some new enum" and plain.failure_word("NEW_FAULT") == "new fault" and "new kind" in plain.alarm_word("NEW_KIND")


def test_a_tiny_kit_with_other_names_still_reads(spot):
    small = Spot(small_kit())
    v = plain.shift(small)
    assert v["configured"] and len(v["assets"]) == 2 and v["status"]["rounds"] == 3
    for a in v["assets"]:
        readable(a["sentence"])
    for f in v["findings"]:
        readable(f["sentence"])
    assert plain.shift(Spot(None))["configured"] is False and plain.round_view(Spot(None), "M001") is None
    assert plain.shift(spot) == plain.shift(spot)  # the same data gives the same words


def test_ask_in_operator_words_keeps_the_exact_draft(spot, view):
    ask = Assistant(spot, FakeRuntime())
    for q, must in ((view["ask"][0], "Start here, in this order."), (view["ask"][1], "Failed checks per round went from 2.3 to 0.9"),
                    (view["ask"][2], "At Pipe Gallery (WP05), 11 of 43 checks have failed."), (view["ask"][3], "9 of 10 checks worked")):
        d = ask.draft(q)
        words = plain.answer(spot, d)
        assert words and must in words, q
        for line in words.split("\n"):
            assert not ENUM.search(line) and not ROW_ID.search(line) and not any(j in line.lower() for j in JARGON), line
    assert ask.draft(view["ask"][2])["intent"] == "waypoint" and ask.draft(view["ask"][3])["intent"] == "mission"
    assert plain.answer(spot, {"intent": "replay", "target": None}) is None


def test_the_shift_endpoints(tmp_path, monkeypatch):
    with TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))) as c:
        v = c.get("/api/spot/shift").json()
        assert v["configured"] is True and len(v["do_first"]) == 3 and len(v["watch"]) == 3 and len(v["assets"]) == 20 and len(v["rounds"]) == 27
        assert v["map"]["plan"]["image_url"] == "/api/spot/plan/image" and len(v["glossary"]) >= 4 and len(v["ask"]) == 4
        r = c.get("/api/spot/shift/rounds/m027").json()
        assert r["configured"] is True and len(r["steps"]) == 10 and r["label"].startswith("Round 27")
        assert c.get("/api/spot/shift/rounds/M999").status_code == 404
        exact = c.post("/api/spot/ask", json={"text": "What should I check first?"}).json()
        words = c.post("/api/spot/ask", json={"text": "What should I check first?", "plain": True}).json()
        assert words["plain"] is True and words["text"].startswith("Start here") and words["draft"] == exact["text"] and words["citations"] == exact["citations"]
    monkeypatch.setenv("ATLAS_SPOT_KIT", str(tmp_path / "no-kit-here"))
    with TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))) as c:
        assert c.get("/api/spot/shift").json()["configured"] is False and c.get("/api/spot/shift/rounds/M001").json()["configured"] is False
