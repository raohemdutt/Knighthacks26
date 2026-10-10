from datetime import UTC, datetime, timedelta

from contexts.capture.application.retention import prune_frames

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def ms(days_ago):
    return int((NOW - timedelta(days=days_ago)).timestamp() * 1000)


def put(root, camera, name):
    path = root / "frames" / camera / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpeg")
    return path


def test_pictures_older_than_the_limit_go_and_newer_ones_stay(tmp_path):
    old, edge, new = put(tmp_path, "cam-desk", f"{ms(8)}.jpg"), put(tmp_path, "cam-door", f"{ms(7.2)}.jpg"), put(tmp_path, "cam-desk", f"{ms(1)}.jpg")
    assert prune_frames(tmp_path, 7, NOW) == 2
    assert not old.exists() and not edge.exists() and new.exists()


def test_forever_removes_nothing_and_only_timestamped_jpegs_are_touched(tmp_path):
    ancient = put(tmp_path, "cam-desk", f"{ms(400)}.jpg")
    assert prune_frames(tmp_path, None, NOW) == 0 and prune_frames(tmp_path, 0, NOW) == 0 and ancient.exists()
    other = put(tmp_path, "cam-desk", "notes.jpg")
    elsewhere = tmp_path / "events" / f"{ms(400)}.jpg"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"x")
    assert prune_frames(tmp_path, 7, NOW) == 1
    assert other.exists() and elsewhere.exists() and not ancient.exists()  # a name that is not a time, and other folders: untouched


def test_a_missing_frames_folder_is_fine(tmp_path):
    assert prune_frames(tmp_path, 7, NOW) == 0


def test_the_gateway_tick_prunes_once_an_hour_when_the_policy_has_a_limit(tmp_path):
    import asyncio

    from contexts.gateway.bootstrap import Atlas

    class Policy:
        keep_days = 7

    class Settings:
        data_dir = tmp_path

    atlas = Atlas.__new__(Atlas)  # only the retention job is under test
    atlas.policy, atlas.settings, atlas._last_retention = Policy(), Settings(), None
    old = put(tmp_path, "cam-desk", f"{ms(30)}.jpg")
    job = atlas._retention(NOW)
    assert asyncio.run(job()) == [] and not old.exists()
    assert atlas._retention(NOW + timedelta(minutes=30)) is None  # not again within the hour
    assert atlas._retention(NOW + timedelta(hours=1, minutes=1)) is not None
    atlas.policy.keep_days = None
    assert atlas._retention(NOW + timedelta(hours=5)) is None  # forever: nothing to do
