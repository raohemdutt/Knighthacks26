import hashlib
import importlib.util
from pathlib import Path

import pytest

from contracts.learning import ModelManifest

spec = importlib.util.spec_from_file_location("make_manifest", Path(__file__).parent.parent / "jetson" / "make_manifest.py")
make_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_manifest)


def test_manifest_matches_the_contract_and_checksums_files(tmp_path):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"projector")
    raw = make_manifest.build_manifest(tmp_path, base="Qwen/Qwen3.5-4B", version="atlas-base")
    manifest = ModelManifest.model_validate(raw)
    assert (manifest.version, manifest.gguf, manifest.mmproj) == ("atlas-base", "model-Q4_K_M.gguf", "mmproj-F16.gguf")
    assert manifest.sha256["model-Q4_K_M.gguf"] == hashlib.sha256(b"weights").hexdigest()
    assert manifest.sha256["mmproj-F16.gguf"] == hashlib.sha256(b"projector").hexdigest()


def test_text_only_package_has_no_mmproj(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    raw = make_manifest.build_manifest(tmp_path, base="b")
    assert raw["mmproj"] is None and raw["version"] == tmp_path.name


def test_ambiguous_folder_is_refused(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "b.gguf").write_bytes(b"y")
    with pytest.raises(SystemExit):
        make_manifest.build_manifest(tmp_path, base="b")
