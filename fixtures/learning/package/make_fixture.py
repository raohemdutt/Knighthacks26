"""Rebuild the fixture model package `python -m lab deploy` (L6) is tested on, with L5's own manifest code.

    python fixtures/learning/package/make_fixture.py

Writes fixtures/learning/package/atlas-desk-objects-fixture/: a placeholder Q4_K_M GGUF (40 KB), a placeholder vision
projector (8 KB) and manifest.json from lab/package.py's folder_manifest (jetson/make_manifest.py's folder rule and a
sha256 per file), with a fixed `created` and the gate's scores, as lab/package.py build leaves a package on pod B."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lab.package import _validated, folder_manifest  # noqa: E402
from lab.package_toolchain_fake import GGUF_MAGIC, placeholder  # noqa: E402

VERSION = "atlas-desk-objects-fixture"
CREATED = "2026-10-11T05:00:00+00:00"
SCORES = {"current": 0.41, "candidate": 0.78, "teacher": 0.86}


def main() -> Path:
    folder = Path(__file__).resolve().parent / VERSION
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{VERSION}-Q4_K_M.gguf").write_bytes(GGUF_MAGIC + placeholder(b"fixture model", 40_000 - len(GGUF_MAGIC)))
    (folder / "mmproj-F16.gguf").write_bytes(GGUF_MAGIC + placeholder(b"fixture mmproj", 8_000 - len(GGUF_MAGIC)))
    manifest = folder_manifest(folder, base="Qwen/Qwen3.5-4B", version=VERSION, task="desk-objects")
    manifest.update(created=CREATED, scores=SCORES)
    (folder / "manifest.json").write_text(json.dumps(_validated(manifest), indent=2) + "\n", encoding="utf-8", newline="\n")
    return folder


if __name__ == "__main__":
    print(main())
