import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.perception.__main__ import main
from contexts.perception.application.backfill import backfill
from contexts.perception.application.captioner import Captioner
from contexts.perception.domain.scenes import scene
from contexts.perception.infrastructure.frames import keyframe_loader, stored_frames
from contracts.perception import SceneDescribed, ZoneEvent
from contracts.types import frame_ref
from runtime import FakeRuntime

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def ms(s: float) -> int:
    return int((T0 + timedelta(seconds=s)).timestamp() * 1000)


def save_frames(data_dir, camera: str, *seconds: float) -> None:
    (data_dir / "frames" / camera).mkdir(parents=True, exist_ok=True)
    for s in seconds:
        (data_dir / "frames" / camera / f"{ms(s)}.jpg").write_bytes(b"jpeg")


def run_backfill(data_dir, rt: FakeRuntime, limit: int | None = None) -> int:
    bus = EventBus(data_dir / "events")
    return asyncio.run(backfill(bus, rt, stored_frames(data_dir), keyframe_loader(data_dir), limit=limit))


def logged(data_dir) -> list[SceneDescribed]:
    return [e for e in EventBus(data_dir / "events").read_log() if isinstance(e, SceneDescribed)]


def test_backfill_captions_only_frames_without_one_oldest_first(tmp_path):
    save_frames(tmp_path, "cam-desk", 20, 0, 10)
    save_frames(tmp_path, "cam-door", 5)
    bus = EventBus(tmp_path / "events")
    live = scene("cam-desk", frame_ref("cam-desk", ms(10)), T0, "Captioned live.", "fake-0")
    desk_crop = scene("cam-desk", frame_ref("cam-desk", ms(20)), T0, "On the desk: keys.", "fake-0", zone_id="desk")
    asyncio.run(bus.publish_all([live, desk_crop]))  # an object list is not a caption: frame 20 still needs one

    rt = FakeRuntime(caption_text="Backfilled.")
    assert run_backfill(tmp_path, rt) == 3
    new = [e for e in logged(tmp_path) if e.caption == "Backfilled."]
    assert [(e.camera_id, e.ts) for e in new] == [("cam-desk", T0), ("cam-door", T0 + timedelta(seconds=5)), ("cam-desk", T0 + timedelta(seconds=20))]
    assert all(e.zone_id is None and e.model_version == "fake-0" and e.frame_ref == frame_ref(e.camera_id, int(e.ts.timestamp() * 1000)) for e in new)


def test_backfill_is_idempotent_and_resumes_after_a_limit(tmp_path):
    save_frames(tmp_path, "cam-desk", 0, 10, 20)
    rt = FakeRuntime()
    assert run_backfill(tmp_path, rt, limit=2) == 2
    assert run_backfill(tmp_path, rt) == 1
    assert run_backfill(tmp_path, rt) == 0
    assert [name for name, _ in rt.calls].count("caption") == 3
    assert [e.ts for e in logged(tmp_path)] == [T0 + timedelta(seconds=s) for s in (0, 10, 20)]


def test_backfill_and_the_live_path_agree_on_the_event(tmp_path):
    save_frames(tmp_path, "cam-desk", 0)
    ref = frame_ref("cam-desk", ms(0))
    enter = ZoneEvent(ts=T0, camera_id="cam-desk", track_id=1, label="person", zone_id="desk", kind="enter", frame_ref=ref)
    live = asyncio.run(Captioner(FakeRuntime(), keyframe_loader(tmp_path)).on_zone_event(enter))
    assert run_backfill(tmp_path, FakeRuntime()) == 1
    assert logged(tmp_path) == live  # the handler returns the scene events of the calls it started


def test_cli_refuses_to_write_fake_captions_unless_asked(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ATLAS_LLM_URL", raising=False)
    save_frames(tmp_path, "cam-desk", 0)
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text("atlas_model: {backend: fake}\n")
    args = ["backfill", "--data-dir", str(tmp_path), "--config-dir", str(config)]
    assert main(args) == 1 and logged(tmp_path) == []
    assert main([*args, "--allow-fake"]) == 0 and len(logged(tmp_path)) == 1
    assert "captioned 1 keyframes with fake-0" in capsys.readouterr().out
