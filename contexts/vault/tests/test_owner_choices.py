"""The two owner-only choices ("Only me clear" on my screen, my own face unblurred for training) need a scanned owner."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.vault import Policy
from contexts.vault.api import SCAN_FIRST, vault_router
from contexts.vault.application.policy import PolicyState, effective_live_faces, effective_train_faces
from contracts.events import parse_event
from contracts.vault import PolicyChanged


def box(tmp_path, owner):
    bus, policy = EventBus(tmp_path / "events"), Policy(tmp_path)
    policy.subscribe(bus)
    app = FastAPI()
    app.include_router(vault_router(policy, bus, owner=owner))
    return policy, TestClient(app)


def test_only_me_and_the_training_opt_in_are_refused_until_the_owner_has_scanned(tmp_path):
    me = {"scanned": False, "ready": False, "reason": SCAN_FIRST, "demo": False}
    policy, http = box(tmp_path, lambda: me)
    got = http.get("/vault/policy").json()
    assert got["owner"] == me and got["live_faces"] == got["live_faces_now"] == "clear" and got["faces_note"] is None
    assert got["train_faces"] == "blurred"  # off by default
    for body in ({"live_faces": "owner"}, {"train_faces": "owner_clear"}):
        r = http.put("/vault/policy", json=body)
        assert (r.status_code, r.json()["detail"]) == (409, SCAN_FIRST)
    assert policy.live_faces == "clear" and policy.train_faces == "blurred"

    me.update(scanned=True, ready=True, reason=None)
    got = http.put("/vault/policy", json={"live_faces": "owner", "train_faces": "owner_clear"}).json()
    assert (got["live_faces"], got["live_faces_now"], got["train_faces"], got["faces_note"]) == ("owner", "owner", "owner_clear", None)
    assert http.put("/vault/policy", json={"paused_cameras": ["cam-door"]}).json()["train_faces"] == "owner_clear"  # carried
    assert Policy(tmp_path).live_owner_only and Policy(tmp_path).train_faces == "owner_clear"  # and after a restart

    me.update(ready=False, reason="Recognition is switched off.")  # switched off since: it behaves as blurred, and says why
    got = http.get("/vault/policy").json()
    assert (got["live_faces"], got["live_faces_now"], got["faces_note"]) == ("owner", "blurred", "Recognition is switched off.")
    assert http.put("/vault/policy", json={"live_faces": "blurred", "train_faces": "blurred"}).status_code == 200  # off is always allowed


def test_without_an_identity_service_the_answer_is_the_policy_alone_and_nobody_can_be_the_owner(tmp_path):
    policy, http = box(tmp_path, None)
    assert "owner" not in http.get("/vault/policy").json()
    assert http.put("/vault/policy", json={"live_faces": "owner"}).status_code == 409
    assert http.put("/vault/policy", json={"live_faces": "nobody"}).status_code == 422


def test_a_value_this_version_does_not_know_is_read_as_blurred_and_never_crashes(tmp_path):
    assert effective_live_faces(None) == "clear" and effective_live_faces("friends") == "blurred"
    assert effective_train_faces(None) == effective_train_faces("everyone_clear") == "blurred"
    assert PolicyState.model_validate({"live_faces": "friends", "train_faces": "everyone_clear"}).model_dump()["live_faces"] == "blurred"
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "policy.json").write_text(json.dumps({"blur_faces": True, "live_faces": "friends", "train_faces": "all"}))
    policy = Policy(tmp_path)  # a newer box wrote the file
    assert (policy.live_faces, policy.train_faces, policy.live_clear, policy.live_owner_only) == ("blurred", "blurred", False, False)


def test_events_from_before_these_fields_replay_unchanged(tmp_path):
    old = parse_event({"type": "vault.policy.changed", "blur_faces": True, "paused_cameras": [], "offline": False, "live_faces": "blurred"})
    assert isinstance(old, PolicyChanged) and old.train_faces is None
    policy = Policy(tmp_path)
    policy.on_policy_changed(old)
    assert (policy.live_faces, policy.train_faces) == ("blurred", "blurred")
    asyncio.run(EventBus(persist=False).publish(PolicyChanged(blur_faces=True, live_faces="owner", train_faces="owner_clear")))
