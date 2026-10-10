"""The anomaly scorer trained on the kit: reproducible weights, honest evaluation, causal features, and how the app uses it."""

import json
from pathlib import Path

import pytest

from contexts.spot import train as train_cli
from contexts.spot.application.service import Spot
from contexts.spot.domain import scorer
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.helpers import small_kit

WEIGHTS = Path("fixtures/spot/anomaly_scorer.json")


@pytest.fixture(scope="module")
def kit():
    return load_kit(Path("fixtures/spot/kit"))


@pytest.fixture(scope="module")
def retrained(kit):
    return scorer.train(kit)


def test_committed_weights_match_a_retrain(retrained):
    assert json.loads(WEIGHTS.read_text()) == json.loads(json.dumps(retrained))
    assert WEIGHTS.read_text() == json.dumps(retrained, indent=1) + "\n"  # what `python -m contexts.spot.train --check` compares


def test_train_command_check_mode(tmp_path, capsys):
    other = tmp_path / "w.json"
    other.write_text("{}")
    assert train_cli.main(["--check", "--out", str(other)]) == 1
    assert train_cli.main(["--kit", str(tmp_path / "none")]) == 2


def test_evaluation_is_split_by_mission_in_time_order_and_reports_counts(retrained, kit):
    p = retrained["evaluation"]["primary"]
    ids = [m.mission_id for m in kit.missions]
    assert p["train_missions"] + p["test_missions"] == ids and p["train_missions"][-1] < p["test_missions"][0]
    assert p["n"] == 10 * len(p["test_missions"]) and p["positives"] == sum(1 for a in kit.anomalies if a.mission_id in p["test_missions"])
    for rule in ("model", "status_rule", "threshold_rule"):
        r = p[rule]
        assert r["tp"] + r["fn"] == p["positives"] and r["flagged"] == r["tp"] + r["fp"] and 0 <= r["pr_auc"] <= 1
    # the status rule's counts can be checked by hand: failed actions in the held-out missions, and how many carry an anomaly
    failed = [a for a in kit.actions if a.mission_id in p["test_missions"] and a.failed]
    assert p["status_rule"]["flagged"] == len(failed) and p["status_rule"]["tp"] == sum(1 for a in failed if kit.anomalies_of.get(a.action_id))
    for fold in retrained["evaluation"]["rolling"]["folds"]:
        assert fold["train_upto"] < fold["test"][0]


def test_the_verdict_says_plainly_what_the_model_does_not_beat(retrained):
    v = retrained["verdict"]
    p = retrained["evaluation"]["primary"]
    assert "reading-outside-band threshold rule" in v
    if p["model"]["tp"] <= p["status_rule"]["tp"] and p["model"]["pr_auc"] - p["status_rule"]["pr_auc"] < 0.10:
        assert "does NOT clearly beat the simpler rule" in v
    assert "never to raise one on its own" in v


def test_features_only_look_backwards():
    kit = small_kit()
    fleet = scorer.fleet_stats(kit, {m.mission_id for m in kit.missions})
    rows = scorer.build(kit, fleet)
    assert [r["action_id"] for r in rows] == [a.action_id for a in kit.actions] and len(rows[0]["x"]) == len(scorer.FEATURES)
    i = scorer.FEATURES.index("waypoint_history")
    assert rows[0]["x"][i] == pytest.approx(fleet["base_rate"])  # nothing seen yet: the smoothed prior, not this row's label
    j = scorer.FEATURES.index("delta_from_previous")
    assert rows[0]["x"][j] == 0 and rows[1]["x"][j] > 0  # the first reading has no previous one
    # dropping every later mission must not change an earlier row's features
    from contexts.spot.tests.helpers import make_kit
    first = make_kit(kit.missions[:1], kit.actions_of["M001"], [r for r in kit.readings if r.mission_id == "M001"], [a for a in kit.anomalies if a.mission_id == "M001"])
    assert [r["x"] for r in scorer.build(first, fleet)] == [r["x"] for r in rows[:10]]


def test_fit_learns_an_obvious_signal_and_explains_it():
    rows = [{"x": [float(i % 2)] + [0.0] * (len(scorer.FEATURES) - 1), "y": i % 2} for i in range(40)]
    model = scorer.fit(rows)
    assert model["weights"][0] > 1 and scorer.score(model, rows[1]["x"]) > 0.8 > 0.2 > scorer.score(model, rows[0]["x"])
    assert scorer.explain(model, rows[1]["x"])[0]["feature"] == "failed"
    assert scorer.average_precision([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))
    assert scorer.average_precision([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)  # ties are taken together
    assert scorer.average_precision([0.5], [0]) is None
    assert scorer.confusion([True, True, False], [1, 0, 1]) == {"tp": 1, "fp": 1, "fn": 1, "flagged": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_the_app_uses_the_model_to_explain_and_break_ties_never_to_change_a_score(kit):
    from contexts.spot.domain import triage

    spot = Spot(kit)
    plain = triage.triage(kit)
    assert [(i["id"], i["score"]) for i in spot.triage] == [(i["id"], i["score"]) for i in plain] or \
        sorted(i["score"] for i in spot.triage) == sorted(i["score"] for i in plain)
    assert all(a["score"] >= b["score"] for a, b in zip(spot.triage, spot.triage[1:]))
    top = spot.triage[0]
    assert top["id"] == "HX-202" and top["model"]["action_id"] == "A0265" and top["model"]["flagged"] and top["model"]["why"][0]["feature"] == "failed"
    view = spot.model_view()
    assert view["available"] and view["quiet_total"] >= view["quiet_without_anomaly_row"] > 0
    for f in view["quiet_flags"]:
        assert f["score"] >= view["threshold"] and f["why"] and f["action_id"] in kit.action
    assert Spot(None).model_view()["configured"] is False
