import json
from datetime import timedelta

from contexts.vault.tests.helpers import Box


def snapshot(updated):
    return {"enforced": True, "mode": "online", "allowed": [{"name": "hub", "hosts": ["dub.aupcloud.io"], "packets": 3, "bytes": 900}],
            "blocked": 2, "blocked_bytes": 120, "recent_blocks": [], "updated": updated}


def test_the_egress_screen_shows_the_firewalls_numbers(tmp_path):
    box = Box(tmp_path)
    (tmp_path / "vault").mkdir(exist_ok=True)
    now = box.clock()  # the vault test box runs on a fixed clock
    (tmp_path / "vault" / "egress.json").write_text(json.dumps(snapshot(now.strftime("%Y-%m-%dT%H:%M:%SZ"))))
    body = box.http.get("/vault/egress").json()
    assert body["enforced"] is True and body["blocked"] == 2 and body["allowed"][0]["name"] == "hub"
    assert body["allowed"][0]["bytes"] == 900 and body["stale"] is False


def test_an_old_snapshot_is_marked_stale_and_no_snapshot_means_not_enforced(tmp_path):
    box = Box(tmp_path)
    assert box.http.get("/vault/egress").json() == {"allowed": [], "blocked": 0, "enforced": False}
    (tmp_path / "vault").mkdir(exist_ok=True)
    old = box.clock() - timedelta(minutes=10)
    (tmp_path / "vault" / "egress.json").write_text(json.dumps(snapshot(old.strftime("%Y-%m-%dT%H:%M:%SZ"))))
    assert box.http.get("/vault/egress").json()["stale"] is True
    (tmp_path / "vault" / "egress.json").write_text("{half a fi")
    assert box.http.get("/vault/egress").json()["enforced"] is False  # a torn read is not an error
