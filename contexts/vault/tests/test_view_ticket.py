"""The view ticket and the `live_faces` policy field: who may ask for the clear live picture, and what an older box
does after the upgrade (clear on the owner's signed-in screen, like every box, until its owner switches that off)."""

import asyncio
import json
from datetime import timedelta

from contexts.vault import Policy, ViewTickets
from contexts.vault.tests.helpers import T0, Box
from contracts import parse_event
from contracts.vault import PolicyChanged

OLD_EVENT = {"type": "vault.policy.changed", "ts": "2026-10-01T09:00:00+00:00", "blur_faces": True,
             "paused_cameras": ["cam-door"], "offline": False}  # as written before `live_faces` existed
OLD_FILE = {"blur_faces": True, "paused_cameras": [], "offline": False, "keep_days": 30}


# --- the ticket itself ---


def test_a_ticket_is_good_until_it_expires_and_holds_nothing_about_anyone():
    tickets = ViewTickets()
    ticket, expires_at = tickets.issue(T0)
    assert expires_at == T0 + timedelta(minutes=10)
    assert tickets.valid(ticket, T0) and tickets.valid(ticket, T0 + timedelta(minutes=9, seconds=59))
    assert not tickets.valid(ticket, T0 + timedelta(minutes=10))  # expired: simply not a ticket
    expiry, random_id, signature = ticket.split(".")
    assert expiry == str(int(expires_at.timestamp())) and len(random_id) >= 16 and len(signature) == 64
    assert tickets.issue(T0)[0] != ticket  # a random id in each


def test_anything_that_is_not_our_ticket_is_no_ticket_and_never_an_error():
    tickets = ViewTickets()
    ticket, _ = tickets.issue(T0)
    expiry, random_id, signature = ticket.split(".")
    later = f"{int(expiry) + 3600}.{random_id}.{signature}"  # the expiry pushed out without the secret
    for bad in ("", "x", "..", ticket[:-1], ticket + "0", later, f"{expiry}.{random_id}.", f".{signature}", ticket.upper(),
                "9" * 500, "é" * 20, None, 7, b"bytes", ViewTickets().issue(T0)[0]):  # the last: another process's secret
        assert tickets.valid(bad, T0) is False, bad
    assert not ViewTickets(secret=b"k" * 32).valid(ticket, T0)
    same = ViewTickets(secret=b"k" * 32)
    assert ViewTickets(secret=b"k" * 32).valid(same.issue(T0)[0], T0)


# --- POST /vault/view-ticket ---


def test_the_ticket_route_needs_the_token_like_every_write(tmp_path):
    box = Box(tmp_path, token="s3cret")
    assert box.http.post("/vault/view-ticket").status_code == 401
    assert box.http.post("/vault/view-ticket", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = box.http.post("/vault/view-ticket", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and set(r.json()) == {"ticket", "expires_at", "expires_in"}
    assert r.json()["expires_in"] == 600 and r.json()["expires_at"] == (T0 + timedelta(minutes=10)).isoformat()
    assert box.tickets.valid(r.json()["ticket"], box.clock())
    box.clock.tick(601)
    assert not box.tickets.valid(r.json()["ticket"], box.clock())
    assert box.seen == []  # asking for a ticket is not an event: nothing about it is logged


def test_with_no_token_set_only_the_box_itself_gets_a_ticket(tmp_path):
    """The gateway lets every write through when ATLAS_TOKEN is unset. That must not hand faces to the whole Wi-Fi."""
    wifi = Box(tmp_path / "a")  # no token; the request comes from another machine
    r = wifi.http.post("/vault/view-ticket")
    assert r.status_code == 403 and "ATLAS_TOKEN" in r.json()["detail"]
    assert wifi.http.put("/vault/policy", json={"offline": True}).status_code == 200  # the other writes are as open as before

    for peer in ("127.0.0.1", "::1"):
        local = Box(tmp_path / peer.replace(":", "_"), peer=(peer, 50000))
        assert local.http.post("/vault/view-ticket").status_code == 200
        # ... but not when a proxy on the box passed the request on for someone else
        assert local.http.post("/vault/view-ticket", headers={"X-Forwarded-For": "192.168.1.20"}).status_code == 403
        assert local.http.post("/vault/view-ticket", headers={"Forwarded": "for=192.168.1.20"}).status_code == 403
        assert local.http.post("/vault/view-ticket", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 200  # `next dev`


# --- the policy field ---


def test_a_new_box_starts_clear_and_the_switch_round_trips(tmp_path):
    box = Box(tmp_path)
    assert box.policy.live_faces == "clear" and box.http.get("/vault/policy").json()["live_faces"] == "clear"
    r = box.http.put("/vault/policy", json={"live_faces": "blurred"})
    assert r.status_code == 200 and r.json()["live_faces"] == "blurred" and box.policy.live_faces == "blurred"
    assert isinstance(box.seen[-1], PolicyChanged) and box.seen[-1].live_faces == "blurred"
    assert box.seen[-1].blur_faces is True  # saved pictures: untouched by this switch
    assert json.loads(box.policy.path.read_text())["live_faces"] == "blurred"
    assert box.http.put("/vault/policy", json={"live_faces": "clear"}).json()["live_faces"] == "clear"
    assert box.http.put("/vault/policy", json={"live_faces": "sharp"}).status_code == 422
    assert Policy(tmp_path).live_faces == "clear"  # after a restart, from the file alone
    assert box.policy.live_clear and Policy(tmp_path).live_clear


def test_an_explicit_blurred_survives_a_pause_a_restart_and_a_replay(tmp_path):
    """The switch is the way to turn the clear view off: once off, nothing but the owner turns it back on."""
    box = Box(tmp_path)
    box.http.put("/vault/policy", json={"live_faces": "blurred"})
    box.http.put("/vault/policy", json={"paused_cameras": ["cam-door"]})  # a pause carries the choice
    assert box.seen[-1].live_faces == "blurred" and not box.policy.live_clear
    assert json.loads(box.policy.path.read_text())["live_faces"] == "blurred"

    restarted = Policy(tmp_path)  # from the file alone
    assert restarted.live_faces == "blurred" and not restarted.live_clear

    box.policy.path.unlink()  # from the log alone
    replayed = Box(tmp_path)
    assert replayed.policy.live_clear  # nothing applied yet: the default
    assert asyncio.run(replayed.bus.replay_into()) == 2
    assert replayed.policy.live_faces == "blurred" and not replayed.policy.live_clear
    assert replayed.http.get("/vault/policy").json()["live_faces"] == "blurred"


def test_pausing_a_camera_keeps_the_live_view_choice(tmp_path):
    box = Box(tmp_path)
    box.http.put("/vault/policy", json={"live_faces": "blurred"})
    r = box.http.put("/vault/policy", json={"paused_cameras": ["cam-door"]})
    assert r.json()["live_faces"] == "blurred" and box.seen[-1].live_faces == "blurred"
    r = box.http.put("/vault/policy", json={"offline": True, "keep_days": 7})
    assert r.json()["live_faces"] == "blurred" and r.json()["paused_cameras"] == ["cam-door"]
    box.http.put("/vault/policy", json={"live_faces": "clear"})
    r = box.http.put("/vault/policy", json={"paused_cameras": []})
    assert r.json()["live_faces"] == "clear" and box.seen[-1].live_faces == "clear"


def test_an_event_from_before_the_field_still_parses_and_replays_to_clear(tmp_path):
    old = parse_event(OLD_EVENT)
    assert isinstance(old, PolicyChanged) and old.live_faces is None
    events = tmp_path / "events"
    events.mkdir()
    (events / "2026-10-01.jsonl").write_text(json.dumps(OLD_EVENT) + "\n")
    box = Box(tmp_path)
    assert box.policy.live_faces == "clear"  # nothing applied yet
    assert asyncio.run(box.bus.replay_into()) == 1
    assert box.policy.snapshot() == {"blur_faces": True, "paused_cameras": ["cam-door"], "offline": False,
                                     "keep_days": None, "live_faces": "clear", "train_faces": "blurred"}  # nothing recorded: clear, no switch touched
    assert box.policy.live_clear and box.http.get("/vault/policy").json()["live_faces"] == "clear"
    assert json.loads((events / "2026-10-01.jsonl").read_text().splitlines()[0]) == OLD_EVENT  # the old event is not rewritten

    # The owner switches it off; from then on every policy event carries the choice, so a replay ends up there too.
    assert box.http.put("/vault/policy", json={"live_faces": "blurred"}).json()["live_faces"] == "blurred"
    box.http.put("/vault/policy", json={"paused_cameras": []})
    box.policy.path.unlink()
    again = Box(tmp_path)
    assert asyncio.run(again.bus.replay_into()) == 3
    assert again.policy.live_faces == "blurred" and again.policy.snapshot()["paused_cameras"] == []


def test_a_policy_file_from_before_the_field_loads_as_clear(tmp_path):
    (tmp_path / "vault").mkdir()
    for name, old in (("absent", OLD_FILE), ("null", OLD_FILE | {"live_faces": None})):
        (tmp_path / "vault" / "policy.json").write_text(json.dumps(old))
        policy = Policy(tmp_path)
        assert policy.live_faces == "clear" and policy.live_clear and policy.keep_days == 30 and policy.blur_faces, name
    box = Box(tmp_path)
    assert box.http.get("/vault/policy").json()["live_faces"] == "clear"  # the Privacy screen's switch shows on
    r = box.http.put("/vault/policy", json={"paused_cameras": ["cam-desk"]})  # an unrelated change does not flip it
    assert r.json()["live_faces"] == "clear"
    assert box.http.put("/vault/policy", json={"live_faces": "blurred"}).json()["live_faces"] == "blurred"
    assert Policy(tmp_path).live_faces == "blurred"  # and the file now records the owner's choice
