import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
