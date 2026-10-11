import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lab.package_pull import FileResult, write_verified
from runtime import modelstore
from runtime.modelstore import ModelStore, ModelStoreError
from tests.test_model_registry import make_version

spec = importlib.util.spec_from_file_location("atlas_swap", Path(__file__).parent.parent / "jetson" / "atlas_swap.py")
atlas_swap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas_swap)


@pytest.fixture
def store(tmp_path):
    try:
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    (tmp_path / "probe").unlink()
    make_version(tmp_path, "v1")
    make_version(tmp_path, "v2", created=datetime(2026, 10, 11, tzinfo=UTC))
    s = ModelStore(tmp_path)
    s.activate("v1")
    return s


def test_a_good_swap_restarts_once_and_keeps_the_new_version(store):
    restarts = []
    atlas_swap.swap(store, "v2", restart=lambda: restarts.append(store.current()), check=lambda: None, log=lambda m: None)
    assert store.current() == "v2" and restarts == ["v2"]  # restarted after the link moved


def test_a_model_that_will_not_serve_is_rolled_back(store):
    restarts, answers = [], iter(["did not load", None])
    with pytest.raises(ModelStoreError, match="did not load"):
        atlas_swap.swap(store, "v2", restart=lambda: restarts.append(store.current()), check=lambda: next(answers),
                        log=lambda m: None)
    assert store.current() == "v1" and restarts == ["v2", "v1"]  # new one tried, then the old one back
    assert store.previous() is None  # the failed attempt is not something to "roll back" to


def test_a_damaged_package_is_refused_before_anything_is_touched(store, tmp_path):
    (tmp_path / "v2" / "m.gguf").write_bytes(b"truncated")
    restarts = []
    with pytest.raises(ModelStoreError, match="not safe"):
        atlas_swap.swap(store, "v2", restart=lambda: restarts.append(1), check=lambda: None, log=lambda m: None)
    assert store.current() == "v1" and restarts == []


def test_a_good_swap_tells_the_app_which_model_replaced_which(store):
    told = []
    atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: None, log=lambda m: None,
                    announce=lambda version, previous: told.append((version, previous)))
    assert told == [("v2", "v1")]


def test_a_swap_that_fails_announces_nothing_and_an_unreachable_app_does_not_undo_a_swap(store):
    told = []
    with pytest.raises(ModelStoreError):
        atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: "did not load", log=lambda m: None,
                        announce=lambda *a: told.append(a))
    assert told == []
    logged = []
    atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: None, log=logged.append,
                    announce=lambda *a: "the app could not be reached")
    assert store.current() == "v2" and any("could not tell the app" in m for m in logged)


# --- --trust-verified: the pull's .verified.json stands in for a second sha256 ------------------------------------

@pytest.fixture
def hashes(monkeypatch):
    seen = []
    real = modelstore.sha256_file

    def counted(path):
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(modelstore, "sha256_file", counted)
    return seen


def record(root, version):
    """What lab/package_pull.py writes after its sha256 pass over a version's files."""
    folder = root / version
    m = ModelStore(root).manifest(version)
    files = [FileResult(name, (folder / name).stat().st_size, sha256=m.sha256[name],
                        mtime_ns=(folder / name).stat().st_mtime_ns) for name in (m.gguf, m.mmproj)]
    return write_verified(folder, version, files)


def quiet_swap(store, version, **kwargs):
    restarts = []
    atlas_swap.swap(store, version, restart=lambda: restarts.append(store.current()), check=lambda: None,
                    log=lambda m: None, **kwargs)
    return restarts


def test_with_the_record_present_a_trusting_swap_hashes_nothing(store, tmp_path, hashes):
    record(tmp_path, "v2")
    assert quiet_swap(store, "v2", trust_verified=True) == ["v2"] and hashes == []


def test_without_the_flag_the_record_is_ignored_and_both_checks_hash(store, tmp_path, hashes):
    record(tmp_path, "v2")
    quiet_swap(store, "v2")
    assert store.current() == "v2" and hashes == ["m.gguf", "mm.gguf"] * 2   # the swap's check, then the switch's


def test_with_no_record_a_trusting_swap_hashes_once_and_writes_the_record_itself(store, tmp_path, hashes):
    quiet_swap(store, "v2", trust_verified=True)
    assert store.current() == "v2" and hashes == ["m.gguf", "mm.gguf"]      # the check hashed, the switch trusted it
    written = json.loads((tmp_path / "v2" / modelstore.VERIFIED_RECORD).read_text(encoding="utf-8"))
    m = store.manifest("v2")
    assert written["written_by"] == modelstore.VERIFIED_BY == "lab/package_pull.py"   # the pull trusts it too
    assert written["hashed_by"] == "jetson/atlas_swap.py" and written["version"] == "v2"
    assert [(f["name"], f["size"], f["sha256"], f["mtime_ns"]) for f in written["files"]] == [
        (n, (tmp_path / "v2" / n).stat().st_size, m.sha256[n], (tmp_path / "v2" / n).stat().st_mtime_ns)
        for n in ("m.gguf", "mm.gguf")]


def test_every_later_press_hashes_nothing_and_a_never_pulled_base_gets_its_record_on_the_way_back(store, tmp_path, hashes):
    quiet_swap(store, "v2", trust_verified=True)                     # first press: one pass over v2
    quiet_swap(store, "v1", trust_verified=True)                     # back to base: one pass over v1, never pulled
    assert hashes == ["m.gguf", "mm.gguf"] * 2 and (tmp_path / "v1" / modelstore.VERIFIED_RECORD).is_file()
    hashes.clear()
    for _ in range(3):                                               # every later press: nothing hashed
        quiet_swap(store, "v2", trust_verified=True)
        quiet_swap(store, "v1", trust_verified=True)
    assert hashes == [] and store.current() == "v1"


def test_the_record_the_swap_wrote_goes_stale_like_the_pulls(store, tmp_path, hashes):
    quiet_swap(store, "v2", trust_verified=True)
    quiet_swap(store, "v1", trust_verified=True)
    hashes.clear()
    path = tmp_path / "v2" / "m.gguf"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # mtime moved: m.gguf is hashed again
    quiet_swap(store, "v2", trust_verified=True)
    assert hashes == ["m.gguf"]
    quiet_swap(store, "v1", trust_verified=True)
    hashes.clear()
    path.write_bytes(b"weights, but longer")                                # size moved and content wrong: refused
    with pytest.raises(ModelStoreError, match="not safe"):
        quiet_swap(store, "v2", trust_verified=True)
    assert hashes == ["m.gguf"] and store.current() == "v1"


def test_a_trusting_swap_over_the_pulls_record_leaves_that_record_alone(store, tmp_path, hashes):
    path = record(tmp_path, "v2")
    before = path.read_bytes()
    quiet_swap(store, "v2", trust_verified=True)
    assert hashes == [] and path.read_bytes() == before                   # nothing hashed, so nothing rewritten


def test_the_swap_names_each_stage_as_it_starts_and_ends(store):
    marks = []
    atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: None, log=lambda m: None,
                    announce=lambda *a: None, stage=lambda name, state: marks.append(f"{name} {state}"))
    assert marks == ["verify start", "verify done", "switch start", "switch done", "restart start", "restart done",
                     "check start", "check done", "publish start", "publish done"]


def test_a_failed_check_marks_check_failed_and_the_rollback(store):
    marks, answers = [], iter(["did not load", None])
    with pytest.raises(ModelStoreError):
        atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: next(answers), log=lambda m: None,
                        stage=lambda name, state: marks.append(f"{name} {state}"))
    assert marks[-3:] == ["check failed", "rollback start", "rollback done"] and store.current() == "v1"


def test_a_damaged_package_marks_verify_failed_and_nothing_after(store, tmp_path):
    (tmp_path / "v2" / "m.gguf").write_bytes(b"truncated")
    marks = []
    with pytest.raises(ModelStoreError):
        atlas_swap.swap(store, "v2", restart=lambda: None, check=lambda: None, log=lambda m: None,
                        stage=lambda name, state: marks.append(f"{name} {state}"))
    assert marks == ["verify start", "verify failed"]


def test_the_command_line_prints_machine_readable_stage_lines(capsys):
    atlas_swap.print_stage("verify", "start")
    name_state_time = capsys.readouterr().out.split()
    assert name_state_time[:3] == ["STAGE", "verify", "start"] and float(name_state_time[3]) > 0


def test_a_file_whose_mtime_moved_is_hashed_again_and_the_other_trusted(store, tmp_path, hashes):
    record(tmp_path, "v2")
    path = tmp_path / "v2" / "m.gguf"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    quiet_swap(store, "v2", trust_verified=True)
    assert store.current() == "v2" and hashes == ["m.gguf"]   # the check hashes it and records it; the switch trusts that


def test_a_file_whose_size_changed_is_hashed_and_refused_before_anything_is_touched(store, tmp_path, hashes):
    record(tmp_path, "v2")
    path = tmp_path / "v2" / "m.gguf"
    st = path.stat()
    path.write_bytes(b"weights, but longer")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))     # the recorded mtime: the size alone must give it away
    restarts = []
    with pytest.raises(ModelStoreError, match="not safe"):
        atlas_swap.swap(store, "v2", restart=lambda: restarts.append(1), check=lambda: None, log=lambda m: None,
                        trust_verified=True)
    assert store.current() == "v1" and restarts == [] and hashes == ["m.gguf"]


def test_a_record_the_pull_did_not_write_or_with_another_checksum_is_not_trusted(store, tmp_path, hashes):
    path = record(tmp_path, "v2")
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "written_by": "someone else"}), encoding="utf-8")
    assert store.verify("v2", trust_verified=True) == [] and hashes == ["m.gguf", "mm.gguf"]
    hashes.clear()
    data["files"][0]["sha256"] = "0" * 64                    # the manifest's checksum is not the one recorded
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.verify("v2", trust_verified=True) == [] and hashes == ["m.gguf"]
    hashes.clear()
    path.write_text("{not json", encoding="utf-8")
    assert store.verify("v2", trust_verified=True) == [] and hashes == ["m.gguf", "mm.gguf"]


def test_a_failed_trusting_swap_goes_back_without_hashing_a_recorded_old_version(store, tmp_path, hashes):
    record(tmp_path, "v1")
    record(tmp_path, "v2")
    restarts, answers = [], iter(["did not load", None])
    with pytest.raises(ModelStoreError, match="did not load"):
        atlas_swap.swap(store, "v2", restart=lambda: restarts.append(store.current()), check=lambda: next(answers),
                        log=lambda m: None, trust_verified=True)
    assert store.current() == "v1" and restarts == ["v2", "v1"] and hashes == []


@pytest.mark.parametrize(("argv", "trusted"), [([], False), (["--trust-verified"], True)])
def test_the_command_line_passes_trust_verified_only_when_asked(tmp_path, monkeypatch, argv, trusted):
    seen = {}
    monkeypatch.setattr(atlas_swap, "swap", lambda store, version, **kwargs: seen.update(kwargs, version=version))
    assert atlas_swap.main(["swap", "v2", "--models-dir", str(tmp_path), *argv]) == 0
    assert (seen["version"], seen["trust_verified"]) == ("v2", trusted)
