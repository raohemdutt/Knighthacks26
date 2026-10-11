import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.vault import PictureLedger, Policy
from contexts.vault.api import vault_router
from contexts.vault.tests.helpers import T0
from contracts.learning import DatasetExported


def exported(data_dir, dataset_id="desk-objects-1a2b3c4d", sizes=(1000, 2500)):
    images = data_dir / "exports" / dataset_id / "images"
    images.mkdir(parents=True)
    for i, n in enumerate(sizes):
        (images / f"{i:04d}.jpg").write_bytes(b"x" * n)
    return DatasetExported(ts=T0, task="desk-objects", dataset_id=dataset_id, examples=len(sizes), action_id="card-7")


def test_nothing_approved_means_no_picture_was_cleared_to_leave(tmp_path):
    snap = PictureLedger(tmp_path).snapshot()
    assert (snap["video_bytes"], snap["pictures"], snap["picture_bytes"], snap["approved_exports"]) == (0, 0, 0, 0)


def test_an_approved_export_counts_its_pictures_bytes_and_the_approval_behind_it(tmp_path):
    ledger = PictureLedger(tmp_path)
    ledger.on_exported(exported(tmp_path))
    snap = ledger.snapshot()
    assert (snap["pictures"], snap["picture_bytes"], snap["approved_exports"]) == (2, 3500, 1)
    assert snap["batches"][0]["approval"] == "card-7" and snap["video_bytes"] == 0


def test_the_ledger_rebuilds_from_the_log(tmp_path):
    bus = EventBus(tmp_path / "events")
    asyncio.run(bus.publish(exported(tmp_path)))
    again = PictureLedger(tmp_path)
    fresh = EventBus(tmp_path / "events")
    again.subscribe(fresh)
    asyncio.run(fresh.replay_into())
    assert again.snapshot()["pictures"] == 2


def test_egress_shows_the_pictures_beside_the_firewall_numbers(tmp_path):
    bus, ledger = EventBus(tmp_path / "events"), PictureLedger(tmp_path)
    ledger.on_exported(exported(tmp_path))
    app = FastAPI()
    app.include_router(vault_router(Policy(tmp_path), bus, pictures=ledger))
    body = TestClient(app).get("/vault/egress").json()
    assert body["enforced"] is False and body["pictures"]["picture_bytes"] == 3500
