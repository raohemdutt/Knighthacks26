import asyncio
import json

from contexts.vault import Policy
from contexts.vault.tests.helpers import Box
from contracts.vault import PolicyChanged, RetentionChanged

DEFAULTS = {"blur_faces": True, "paused_cameras": [], "offline": False, "keep_days": None}


def test_defaults_are_private_and_nothing_is_written_until_something_changes(tmp_path):
    box = Box(tmp_path)
    assert box.http.get("/vault/policy").json() == DEFAULTS
    assert box.policy.blur_faces and not box.policy.is_paused("cam-door")
    assert not box.policy.path.exists()


def test_put_changes_the_policy_and_publishes_the_contract_events(tmp_path):
    box = Box(tmp_path)
    r = box.http.put("/vault/policy", json={"blur_faces": False, "paused_cameras": ["cam-door", "cam-door"], "keep_days": 7})
    assert r.status_code == 200
    assert r.json() == {"blur_faces": False, "paused_cameras": ["cam-door"], "offline": False, "keep_days": 7}
    assert box.http.get("/vault/policy").json() == r.json()
    assert not box.policy.blur_faces and box.policy.keep_days == 7
    assert box.policy.is_paused("cam-door") and not box.policy.is_paused("cam-desk")

    changed, retention = box.seen
    assert isinstance(changed, PolicyChanged) and isinstance(retention, RetentionChanged)
    assert (changed.blur_faces, changed.paused_cameras, changed.offline) == (False, ["cam-door"], False)
    assert retention.keep_days == 7
    assert json.loads(box.policy.path.read_text()) == r.json()  # snapshot on disk follows


def test_put_keeps_what_it_does_not_mention_and_retention_event_only_on_change(tmp_path):
    box = Box(tmp_path)
    box.http.put("/vault/policy", json={"paused_cameras": ["cam-desk"], "keep_days": 30})
    box.seen.clear()
    r = box.http.put("/vault/policy", json={"offline": True, "keep_days": 30})  # same retention
    assert r.json() == {"blur_faces": True, "paused_cameras": ["cam-desk"], "offline": True, "keep_days": 30}
    assert [e.type for e in box.seen] == ["vault.policy.changed"]

    box.seen.clear()
    r = box.http.put("/vault/policy", json={"keep_days": None})  # null = keep forever
    assert r.json()["keep_days"] is None
    assert [e.type for e in box.seen] == ["vault.policy.changed", "vault.retention.changed"]
    assert box.seen[1].keep_days is None


def test_put_rejects_bad_bodies_and_needs_the_token(tmp_path):
    box = Box(tmp_path, token="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    assert box.http.put("/vault/policy", json={"blur_faces": False}).status_code == 401
    assert box.http.put("/vault/policy", json={"keep_days": 0}, headers=auth).status_code == 422
    assert box.http.put("/vault/policy", json={"blur": False}, headers=auth).status_code == 422
    assert box.seen == [] and box.policy.blur_faces
    assert box.http.get("/vault/policy").status_code == 200  # reads are open on the tailnet
    assert box.http.put("/vault/policy", json={"blur_faces": False}, headers=auth).status_code == 200


def test_policy_survives_a_restart_from_the_file_alone(tmp_path):
    Box(tmp_path).http.put("/vault/policy", json={"blur_faces": False, "paused_cameras": ["cam-door"], "keep_days": 14})
    restarted = Policy(tmp_path)  # before any replay
    assert not restarted.blur_faces and restarted.is_paused("cam-door") and restarted.keep_days == 14


def test_the_log_wins_over_the_file_on_replay(tmp_path):
    box = Box(tmp_path)
    box.http.put("/vault/policy", json={"blur_faces": False, "keep_days": 14})
    box.http.put("/vault/policy", json={"paused_cameras": ["cam-desk"]})
    truth = box.http.get("/vault/policy").json()

    box.policy.path.unlink()  # file lost: replay rebuilds it
    lost = Box(tmp_path)
    assert lost.policy.snapshot() == DEFAULTS
    assert asyncio.run(lost.bus.replay_into()) == 3
    assert lost.policy.snapshot() == truth
    assert json.loads(lost.policy.path.read_text()) == truth
    assert lost.seen == []  # replay reaches only replay=True handlers

    box.policy.path.write_text(json.dumps(DEFAULTS | {"keep_days": 999}))  # file edited by hand: the log corrects it
    edited = Box(tmp_path)
    assert edited.policy.keep_days == 999
    asyncio.run(edited.bus.replay_into())
    assert edited.policy.snapshot() == truth
    assert json.loads(edited.policy.path.read_text()) == truth

    box.policy.path.write_text("{not json")  # unreadable: private defaults until replay
    broken = Box(tmp_path)
    assert broken.policy.snapshot() == DEFAULTS
    asyncio.run(broken.bus.replay_into())
    assert broken.policy.snapshot() == truth


def test_with_no_policy_events_in_the_log_the_file_stands(tmp_path):
    box = Box(tmp_path)
    box.http.put("/vault/policy", json={"blur_faces": False})
    for f in (tmp_path / "events").glob("*.jsonl"):  # what a wipe of the log leaves behind
        f.unlink()
    after = Box(tmp_path)
    assert asyncio.run(after.bus.replay_into()) == 0
    assert not after.policy.blur_faces


def test_egress_is_an_honest_placeholder(tmp_path):
    assert Box(tmp_path).http.get("/vault/egress").json() == {"allowed": [], "blocked": 0, "enforced": False}


def test_a_damaged_egress_snapshot_counts_as_none(tmp_path):
    box = Box(tmp_path)
    snap = tmp_path / "vault" / "egress.json"
    snap.parent.mkdir(parents=True)
    for text in ('{"allowed": [], "blocked": 0}', '{"updated": "yesterday"}', '{"updated": 5}', "[]", '"x"', "null", "{"):
        snap.write_text(text)
        r = box.http.get("/vault/egress")
        assert r.status_code == 200 and r.json() == {"allowed": [], "blocked": 0, "enforced": False}, text
    snap.write_text('{"updated": "2026-10-11T11:00:00Z", "allowed": [], "blocked": 2, "enforced": true}')
    assert box.http.get("/vault/egress").json()["stale"] is True  # written an hour before the box's clock
