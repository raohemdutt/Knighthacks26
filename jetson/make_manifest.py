"""Write manifest.json (with sha256s) for a model package folder under data/models/ (R7/R8).

    python3 jetson/make_manifest.py data/models/atlas-base --base Qwen/Qwen3.5-4B

Stdlib only, so it runs on the Jetson without the repo's dependencies. The output
follows ModelManifest in contracts/learning.py (tests/test_make_manifest.py checks it).
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(folder: Path, base: str, version: str | None = None, task: str | None = None) -> dict:
    mmprojs = sorted(p for p in folder.glob("*.gguf") if p.name.startswith("mmproj"))
    ggufs = sorted(p for p in folder.glob("*.gguf") if not p.name.startswith("mmproj"))
    if len(ggufs) != 1 or len(mmprojs) > 1:
        raise SystemExit(f"{folder}: expected exactly one model .gguf and at most one mmproj*.gguf")
    files = ggufs + mmprojs
    return {
        "version": version or folder.name,
        "base": base,
        "task": task,
        "gguf": ggufs[0].name,
        "mmproj": mmprojs[0].name if mmprojs else None,
        "sha256": {p.name: sha256_of(p) for p in files},
        "scores": {},
        "created": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--version")
    parser.add_argument("--task")
    args = parser.parse_args()
    manifest = build_manifest(args.folder, args.base, args.version, args.task)
    (args.folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.folder / 'manifest.json'}")


if __name__ == "__main__":
    main()
