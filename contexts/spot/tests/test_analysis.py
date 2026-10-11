"""Every analysis function on a hand-computed kit, the minimum-n rule, the real kit's headline numbers, and triage order."""

from datetime import timedelta
from pathlib import Path

import pytest

from contexts.spot.domain import analysis, triage
from contexts.spot.domain.stats import MIN_N
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.helpers import action, anomaly, make_kit, mission, reading, small_kit


@pytest.fixture(scope="module")
def real():
    return load_kit(Path("fixtures/spot/kit"))


# --- hand-computed ---


def test_hidden_failures_counts_completed_missions_with_failed_actions_inside():
    f = analysis.hidden_failures(small_kit())
    assert f.value == 0.5 and f.n == 2  # M001 hides failures, M003 does not; M002 is not COMPLETED
    assert [b.key for b in f.rows] == ["M001"] and f.rows[0].k == 2
    assert f.evidence.missions == ["M001"] and f.evidence.actions == ["A0009", "A0010"]
    assert f.extra == {"failed_inside_completed": 2, "failed_total": 3}
    assert "M002" in f.caveat


def test_kpis_carry_their_denominators():
    k = {x["key"]: x for x in analysis.kpis(small_kit())}
    assert (k["mission_completion"]["k"], k["mission_completion"]["n"]) == (2, 3)
    assert (k["action_success"]["k"], k["action_success"]["n"]) == (27, 30)
    assert (k["hidden_failures"]["k"], k["hidden_failures"]["n"]) == (1, 2)
    assert k["action_success"]["lo"] < 0.9 < k["action_success"]["hi"]


def test_reliability_by_location_ranks_and_keeps_the_failed_rows():
    f = analysis.reliability_by_location(small_kit())
    rows = {b.key: b for b in f.rows}
    assert (rows["WP02"].k, rows["WP02"].n) == (2, 5) and (rows["WP01"].k, rows["WP01"].n) == (1, 25)
    assert not rows["WP02"].enough and "not enough data" in rows["WP02"].note  # 5 actions is under the minimum
    assert not f.enough and "Not enough data" in f.sentence


def test_minimum_n_rule_a_group_under_eight_rows_gets_no_conclusion():
    m = mission(1)
    acts = [action(m, i, i, failed=i <= 3, kind="GAUGE_READING" if i <= 4 else "VISUAL_INSPECTION") for i in range(1, 11)]
    f = analysis.reliability_by_type(make_kit([m], acts))
    assert MIN_N == 8 and not f.enough  # 4 gauge readings and 6 visual: neither reaches 8
    assert all(not b.enough for b in f.rows)


def test_version_comparison_on_the_small_kit_says_not_enough_missions_but_gives_the_counts():
    f = analysis.version_comparison(small_kit())
    by = {m["key"]: m for m in f.extra["metrics"]}
    assert (by["action_success"]["a"]["k"], by["action_success"]["a"]["n"]) == (17, 20)
    assert (by["action_success"]["b"]["k"], by["action_success"]["b"]["n"]) == (10, 10)
    assert by["mission_completion"]["verdict"] == "not enough data"  # 2 missions against 1
    assert by["action_success"]["evidence"]["a"] == ["A0009", "A0010", "A0011"]


def test_one_version_only_is_not_a_comparison():
    m = mission(1)
    f = analysis.version_comparison(make_kit([m], [action(m, 1, 1)]))
    assert not f.enough and "exactly two" in f.sentence


def test_battery_bins_use_the_action_battery():
    m = mission(1)
    acts = [action(m, i, i, failed=i > 8, battery=100 - 10 * i) for i in range(1, 11)]  # 90..0; the two lowest fail
    f = analysis.battery_vs_reliability(make_kit([m, mission(2)], acts + [action(mission(2), i, 20 + i, battery=80) for i in range(1, 9)]))
    rows = {b.key: (b.k, b.n) for b in f.rows}
    assert rows["0-20"] == (2, 2) and rows["20-40"] == (0, 2) and rows["80-100"] == (0, 10)


def test_anomaly_findings_separate_confirmed_from_unverified():
    kit = small_kit()
    f = analysis.anomalies_in_successful_missions(kit)
    assert f.extra["confirmed"] == ["AN002"] and f.extra["unverified"] == ["AN001"]
    assert f.extra["in_clean_missions"] == ["AN002"]  # M003 has no failed action
    m = analysis.anomaly_matrix(kit)
    cell = next(c for c in m.extra["matrix"] if c["severity"] == "HIGH" and c["criticality"] == "CRITICAL")
    assert cell["anomalies"] == ["AN001"] and m.value == 1


def test_repeat_anomalies_need_two_on_one_asset():
    m = mission(1)
    acts = [action(m, i, i) for i in range(1, 4)]
    kit = make_kit([m], acts, anomalies=[anomaly(acts[0], 1, "LOW"), anomaly(acts[1], 2, "HIGH"), ])
    f = analysis.repeat_anomalies(kit)
    assert [b.key for b in f.rows] == ["PUMP-1"] and f.rows[0].extra["same_type"] == {"VISUAL_DEFECT": 2}
    assert not analysis.repeat_anomalies(small_kit()).rows  # one anomaly each on two assets


def test_reading_trend_needs_five_good_readings_and_uses_only_good_ones():
    ms = [mission(i) for i in range(1, 10)]
    acts = [action(m, 1, i, failed=i == 9) for i, m in enumerate(ms, 1)]
    readings = [reading(a, i, value=10 * i) for i, a in enumerate(acts, 1)]  # +10 per mission, missions 8 h apart
    f = analysis.reading_trends(make_kit(ms, acts, readings))
    s = f.extra["series"][0]
    assert s["n"] == 9 and s["n_good"] == 8 and s["enough"]
    assert s["slope_per_day"] == pytest.approx(30.0)  # 10 per 8 hours
    short = analysis.reading_trends(make_kit(ms[:3], acts[:3], readings[:3])).extra["series"][0]
    assert not short["enough"] and "not enough data" in short["note"] and short["slope_per_day"] is None


# --- the real kit ---


def test_every_finding_on_the_real_kit_has_a_sentence_an_n_and_evidence(real):
    findings = analysis.all_findings(real)
    assert len(findings) == len(analysis.FINDERS) and len({f.id for f in findings}) == len(findings)
    for f in findings:
        assert f.sentence and f.n > 0, f.id
        assert f.evidence.count > 0 or f.id == "reading_trends", f.id
        for id_ in f.evidence.actions:
            assert id_ in real.action
        for id_ in f.evidence.anomalies:
            assert id_ in real.anomaly


def test_our_numbers_agree_with_the_kits_validation_results(real):
    v = analysis.validation(real)
    assert v["their_status"] == "PASS" and v["differences"] == []
    assert v["agree"] == v["compared"] == 12
    by = {r["name"]: r for r in v["rows"]}
    assert by["baseline_action_success_pct"]["ours"] == 77.2 and by["improved_action_success_pct"]["ours"] == 91.1
    assert by["baseline_mission_completion_pct"]["ours"] == 94.4 and by["wp05_navigation_failure_pct"]["ours"] == 28.0


def test_headline_numbers_on_the_real_kit(real):
    f = {x.id: x for x in analysis.all_findings(real)}
    assert f["hidden_failures"].n == 26 and len(f["hidden_failures"].rows) == 21
    vc = {m["key"]: m for m in f["version_comparison"].extra["metrics"]}
    assert (vc["action_success"]["a"]["k"], vc["action_success"]["a"]["n"], vc["action_success"]["b"]["k"], vc["action_success"]["b"]["n"]) == (139, 180, 82, 90)
    assert vc["action_success"]["verdict"] == "changed" and vc["mission_completion"]["verdict"] == "no clear change"
    assert vc["action_success_matched_battery"]["verdict"] == "no clear change"  # the confound the caveat spells out
    assert "cannot separate" in f["version_comparison"].caveat
    assert f["reliability_by_type"].extra["worst"] == "ACOUSTIC_INSPECTION"
    nav = f["navigation_hotspot"]
    assert nav.extra["location_id"] == "WP05" and (nav.rows[0].k, nav.rows[0].n) == (7, 25) and not nav.extra["clear"]
    assert "suggestive" in nav.sentence and "first stop of 25 of 27" in nav.caveat
    assert f["battery"].extra["clear"] and f["battery"].rows[0].key == "0-20" and (f["battery"].rows[0].k, f["battery"].rows[0].n) == (8, 11)


def test_assumptions_list_is_generated_and_says_now_is_the_data(real):
    items = analysis.assumptions(real)
    assert len(items) >= 12 and all(a["assumption"] and a["why"] for a in items)
    assert "2026-09-10 10:25" in items[0]["assumption"]


# --- triage ---


def test_triage_orders_by_the_transparent_score():
    ms = [mission(i) for i in range(1, 8)]  # 8 hours apart; "now" is the end of M007
    acts = []
    for i, m in enumerate(ms):
        acts.append(action(m, 1, 10 * i + 1, asset="PUMP-1", loc="WP01"))
        acts.append(action(m, 2, 10 * i + 2, asset="HX-9", loc="WP02", failed=i == 6))
    old_low = anomaly(acts[0], 1, "LOW")  # PUMP-1 (LOW asset), two days ago, confirmed, seen fine since
    new_high = anomaly(acts[13], 2, "HIGH")  # HX-9 (CRITICAL asset), latest mission, on the failed action
    items = triage.triage(make_kit(ms, acts, anomalies=[old_low, new_high]))
    assert [i["id"] for i in items][:2] == ["HX-9", "PUMP-1"]
    top = items[0]
    parts = {p["label"]: p["points"] for p in top["parts"]}
    assert parts == {"severity": 25.0, "criticality": 25, "not inspected": 15}  # HIGH x 1.0 recency + CRITICAL asset + failed latest attempt
    assert top["score"] == 65.0 and top["next_step"]["action"] == "reinspect" and not top["confirmed"]
    low = {p["label"]: p["points"] for p in items[1]["parts"]}
    assert low["severity"] == pytest.approx(3 * 0.75, abs=0.06) and low["seen since"] < 0  # shown to one decimal
    assert items[1]["confirmed"] and [i["rank"] for i in items] == list(range(1, len(items) + 1))


def test_confirmed_serious_anomaly_means_dispatch_and_route_anomaly_means_check_the_path():
    m = mission(1)
    a1, a2 = action(m, 1, 1, asset="HX-9", loc="WP02"), action(m, 2, 2, kind="NAVIGATION", failed=True)
    items = {i["id"]: i for i in triage.triage(make_kit([m], [a1, a2], anomalies=[anomaly(a1, 1, "CRITICAL"), anomaly(a2, 2, "MEDIUM", "NAVIGATION_OBSTRUCTION")]))}
    assert items["HX-9"]["next_step"]["action"] == "dispatch"
    assert items["PUMP-1"]["next_step"]["action"] == "check_path"


def test_uninspected_risk_lists_important_assets_whose_last_good_inspection_is_old():
    ms = [mission(i) for i in range(1, 6)]
    acts = [action(ms[0], 1, 1, asset="HX-9", loc="WP02")] + [action(m, 2, 10 + i, asset="HX-9", loc="WP02", failed=True) for i, m in enumerate(ms[1:])]
    acts += [action(ms[-1], 3, 99)]
    rows = triage.uninspected_risk(make_kit(ms, acts))
    top = rows[0]
    assert top["asset_id"] == "HX-9" and top["at_risk"] and top["last_good_action"] == "A0001"
    assert top["failed_since"] == ["A0010", "A0011", "A0012", "A0013"]
    assert top["hours_since_good"] == pytest.approx((ms[-1].end_time - acts[0].timestamp) / timedelta(hours=1))
    assert not rows[-1]["at_risk"]  # PUMP-1 is a LOW asset


def test_triage_and_next_steps_on_the_real_kit(real):
    items = triage.triage(real)
    assert items[0]["id"] == "HX-202" and items[0]["criticality"] == "CRITICAL" and items[0]["anomaly_id"] == "AN005"
    assert all(a["score"] >= b["score"] for a, b in zip(items, items[1:]))
    nxt = triage.investigate_next(real)
    assert len(nxt) == 3 and [n["key"] for n in nxt] == ["asset", "pattern", "coverage"]
    assert nxt[0]["target"] == {"kind": "asset", "id": "HX-202"} and nxt[1]["finding_id"] == "battery"
    assert all(sum(len(v) for v in n["evidence"].values()) > 0 for n in nxt)
