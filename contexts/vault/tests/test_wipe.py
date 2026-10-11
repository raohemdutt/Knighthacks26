import asyncio
import os
from datetime import timedelta
from pathlib import Path

import pytest

from contexts.vault import KeyRing, Wiper, WipeTokens
from contexts.vault.tests.helpers import T0, Box
from contracts.vault import WipeRequested


def fill(data: Path) -> dict[str, Path]:
    """A data dir with something of everything, the way the box lays it out."""
    files = {
        "frame": data / "frames" / "cam-door" / "1000.jpg",
        "frame2": data / "frames" / "cam-desk" / "2000.jpg",
        "db": data / "memory.sqlite",
        "wal": data / "memory.sqlite-wal",
        "shm": data / "memory.sqlite-shm",
        "export": data / "exports" / "desk-objects" / "train.jsonl",
        "events": data / "events" / "2020-01-01.jsonl",  # an old day, so it never collides with today's log
        "invite": data / "invites" / "invite-20261010T180000Z-abc.ics",
        "plan": data / "dayrun" / "plans" / "2026-10-10.json",
        "place_pack": data / "places" / "home.yaml",
        "inbox_file": data / "files" / "0123456789abcdef0123456789abcdef.bin",  # W1: a file shared from the phone
        "dayrun_other": data / "dayrun" / "forecasts.json",  # not on the wipe list: only the committed plans go
        "circle": data / "circle.sqlite",  # not on the wipe list
        "browser": data / "browser" / "profile" / "cookies",  # not on the wipe list
    }
    for f in files.values():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    KeyRing(data).key_for("memory")
    files["key"] = data / "vault" / "master.key"
    return files


def left(files: dict[str, Path]) -> set[str]:
    return {name for name, f in files.items() if f.exists()}


# --- confirmation token ---


# Both tests below build symbolic links, which Windows only allows with a privilege most accounts lack.
needs_symlinks = pytest.mark.skipif(os.name == "nt", reason="needs symbolic links")


def test_token_works_once_inside_its_lifetime():
    tokens = WipeTokens(ttl=timedelta(seconds=60))
    token, expires_at = tokens.request("keyframes", T0)
    assert expires_at == T0 + timedelta(seconds=60)
    assert tokens.redeem("guess", T0) is None  # a wrong guess does not burn the real one
    assert tokens.redeem(token, T0 + timedelta(seconds=59)) == "keyframes"
    assert tokens.redeem(token, T0 + timedelta(seconds=59)) is None  # single use


def test_token_expires_and_a_new_request_replaces_the_old_one():
    tokens = WipeTokens(ttl=timedelta(seconds=60))
    token, _ = tokens.request("all", T0)
    assert tokens.redeem(token, T0 + timedelta(seconds=60)) is None  # expired
    assert tokens.redeem(token, T0) is None  # and spent by the attempt
    old, _ = tokens.request("all", T0)
    new, _ = tokens.request("memory", T0)
    assert old != new
    assert tokens.redeem(old, T0) is None
    assert tokens.redeem(new, T0) == "memory"
    assert WipeTokens().redeem("", T0) is None  # nothing pending


# --- the endpoints ---


def test_wipe_needs_a_fresh_token_and_publishes_the_event(tmp_path):
    files = fill(tmp_path)
    box = Box(tmp_path)
    assert box.http.post("/vault/wipe", json={"token": "nope"}).status_code == 403
    assert box.http.post("/vault/wipe", json={}).status_code == 422

    asked = box.http.post("/vault/wipe/request", json={"scope": "keyframes"}).json()
    assert asked["scope"] == "keyframes" and asked["expires_at"] == (T0 + timedelta(minutes=2)).isoformat()
    box.clock.tick(121)
    assert box.http.post("/vault/wipe", json={"token": asked["token"]}).status_code == 403  # expired
    assert box.seen == [] and left(files) == set(files)

    token = box.http.post("/vault/wipe/request", json={"scope": "keyframes"}).json()["token"]
    box.clock.tick(119)
    r = box.http.post("/vault/wipe", json={"token": token})
    assert r.status_code == 200
    assert r.json() == {"scope": "keyframes", "done": True, "removed": {"frames": 2}, "refused": [],
                        "identity_removed": 0, "live_memory_cleared": False, "summary": "Removed 2 files from frames."}
    [event] = box.seen
    assert isinstance(event, WipeRequested) and event.scope == "keyframes"
    assert left(files) == set(files) - {"frame", "frame2"}
    assert box.http.post("/vault/wipe", json={"token": token}).status_code == 403  # single use
    assert len(box.seen) == 1


def test_wipe_endpoints_need_the_gateway_token(tmp_path):
    box = Box(tmp_path, token="s3cret")
    assert box.http.post("/vault/wipe/request", json={}).status_code == 401
    assert box.http.post("/vault/wipe", json={"token": "x"}).status_code == 401
    r = box.http.post("/vault/wipe/request", json={}, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and r.json()["scope"] == "all"


def test_wipe_is_not_run_again_on_replay(tmp_path):
    box = Box(tmp_path)
    asyncio.run(box.bus.publish(WipeRequested(scope="memory")))  # logged, db not there yet
    files = fill(tmp_path)
    restarted = Box(tmp_path)
    assert asyncio.run(restarted.bus.replay_into()) == 1
    assert restarted.wiper.last is None and left(files) == set(files)


# --- the wiper ---


def test_each_scope_removes_only_what_it_lists(tmp_path):
    files = fill(tmp_path)
    wiper = Wiper(tmp_path)
    assert wiper.wipe("memory").removed == {"memory": 3}
    assert left(files) == set(files) - {"db", "wal", "shm"}

    report = wiper.wipe("all")
    assert report.removed == {"frames": 2, "memory": 0, "exports": 1, "events": 1, "invites": 1, "dayrun/plans": 1,
                              "places": 1, "files": 1}
    assert report.refused == [] and not report.master_key_removed and wiper.last is report
    assert left(files) == {"circle", "browser", "key", "dayrun_other"}  # the key and everything unlisted stay
    for kept in ("frames", "exports", "events", "invites", "dayrun/plans", "places", "files"):  # emptied, not removed
        assert (tmp_path / kept).is_dir() and not any((tmp_path / kept).iterdir())
    assert report.summary() == ("Removed 2 files from frames, 1 file from exports, 1 file from events, 1 file from invites, "
                                "1 file from dayrun/plans, 1 file from places, 1 file from files.")
    assert set(wiper.wipe("all").removed.values()) == {0}  # nothing left: no error
    assert wiper.last.summary() == "Nothing was stored for this scope."


class LiveStore:
    """Stands in for the open Memory store: records what the wiper asked of it."""

    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.append("clear")

    def close(self):
        self.calls.append("close")

    def reopen(self):
        self.calls.append("reopen")


def test_a_memory_wipe_empties_the_open_store_then_reopens_it(tmp_path):
    files = fill(tmp_path)
    live = LiveStore()
    wiper = Wiper(tmp_path, live_memory=live)
    assert not wiper.wipe("keyframes").live_memory_cleared and live.calls == []  # only memory and all touch it
    report = wiper.wipe("memory")
    assert live.calls == ["clear", "close", "reopen"] and report.live_memory_cleared  # closed before the file goes: Windows cannot remove an open file
    assert "db" not in left(files)
    assert report.summary() == "Removed 3 files from memory, the running Memory store (emptied)."
    assert wiper.wipe("all").live_memory_cleared and live.calls == ["clear", "close", "reopen"] * 2


@needs_symlinks
def test_a_refused_memory_file_is_not_emptied_through_the_open_store_either(tmp_path):
    data, outside = tmp_path / "data", tmp_path / "outside"
    fill(data)
    outside.mkdir()
    (outside / "other.sqlite").write_bytes(b"keep me")
    live = LiveStore()
    report = Wiper(data, memory_db=outside / "other.sqlite", live_memory=live).wipe("memory")
    assert live.calls == [] and not report.live_memory_cleared and len(report.refused) == 3
    (data / "memory.sqlite").unlink()
    (data / "memory.sqlite").symlink_to(outside / "other.sqlite")
    report = Wiper(data, live_memory=live).wipe("all")
    assert live.calls == [] and str(data / "memory.sqlite") in report.refused
    assert (outside / "other.sqlite").read_bytes() == b"keep me"


def test_master_key_goes_only_when_asked_for_by_name(tmp_path):
    files = fill(tmp_path)
    wiper = Wiper(tmp_path)
    wiper.on_wipe_requested(WipeRequested(scope="all"))  # the event cannot reach the key
    assert files["key"].exists()
    report = wiper.wipe("keyframes", include_master_key=True)
    assert report.master_key_removed and not files["key"].exists()
    assert not wiper.wipe("keyframes", include_master_key=True).master_key_removed  # already gone


@needs_symlinks
def test_refuses_targets_that_leave_the_data_dir(tmp_path):
    data, outside = tmp_path / "data", tmp_path / "outside"
    files = fill(data)
    (outside / "photos").mkdir(parents=True)
    precious = [outside / "photos" / "a.jpg", outside / "other.sqlite", outside / "notes.txt"]
    for f in precious:
        f.write_bytes(b"keep me")

    (data / "exports" / "desk-objects" / "train.jsonl").unlink()
    (data / "exports" / "desk-objects").rmdir()
    (data / "exports").rmdir()
    (data / "exports").symlink_to(outside / "photos")  # a target that is a link out
    (data / "frames" / "cam-door" / "link.jpg").symlink_to(outside / "notes.txt")  # links inside a wiped dir
    (data / "frames" / "album").symlink_to(outside / "photos")

    wiper = Wiper(data, memory_db=outside / "other.sqlite")  # a db configured somewhere else
    report = wiper.wipe("all")
    assert all(f.read_bytes() == b"keep me" for f in precious)
    assert sorted(report.refused) == sorted(
        [str(data / "exports"), str(outside / "other.sqlite"), str(outside / "other.sqlite-wal"), str(outside / "other.sqlite-shm")]
    )
    assert report.removed == {"frames": 4, "memory": 0, "exports": 0, "events": 1, "invites": 1, "dayrun/plans": 1,
                              "places": 1, "files": 1}  # the links themselves went
    assert not any((data / "frames").iterdir())
    assert files["db"].exists()  # not the configured db, so not touched


@needs_symlinks
def test_refuses_dot_dot_paths_links_back_inside_and_the_vault_folder(tmp_path):
    data = tmp_path / "data"
    files = fill(data)
    (tmp_path / "other.sqlite").write_bytes(b"keep me")
    assert Wiper(data, memory_db=data / ".." / "other.sqlite").wipe("memory").refused  # climbs out with ..
    assert (tmp_path / "other.sqlite").exists()
    assert Wiper(data, memory_db=data / "vault" / "master.key").wipe("memory").refused  # the vault's own folder
    assert Wiper(data, memory_db=data).wipe("memory").refused  # the data dir itself
    assert files["key"].exists() and files["frame"].exists()

    files["frame"].unlink()
    files["frame2"].unlink()
    for d in ("cam-door", "cam-desk"):
        (data / "frames" / d).rmdir()
    (data / "frames").rmdir()
    (data / "frames").symlink_to(data / "vault")  # a link that stays inside but points at the key
    report = Wiper(data).wipe("keyframes")
    assert report.refused == [str(data / "frames")] and files["key"].exists()
