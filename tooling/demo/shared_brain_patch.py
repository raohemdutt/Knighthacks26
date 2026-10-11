"""Put the Shared brain wiring (G4) into a repo copy whose gateway files differ from main, such as the box's.

    python3 tooling/demo/shared_brain_patch.py <repo root>            # edits in place, prints what it did
    python3 tooling/demo/shared_brain_patch.py <repo root> --check    # says what it would do, changes nothing

It inserts the same three additive blocks the PR adds to contexts/gateway/bootstrap.py and the two to
contexts/gateway/app.py, each anchored on a line that exists both on main and on the box's copy, and skips a block
that is already there, so running it twice changes nothing. contexts/circle/brain_power.py must be copied beside it
first. It never touches .env. Stdlib only (the box's system python3 runs it).
"""

import sys
from pathlib import Path

BOOT = [
    ("    face_share: object | None = None  # FaceShare (contexts/circle): a friend's box recognises you; off by default\n",
     "    shared_brain: object | None = None  # SharedBrain (contexts/circle/brain_power.py): the group brain and its watts\n",
     "after"),
    ("    # Place packs (S5): shipped packs in config/places, uploads in data/places (later wins).\n",
     "    # The Circle's shared brain (G4): the group brain (opt-in questions to the teacher pod) and that pod's live watts\n"
     "    # for the Circle screen. None without a Circle; each part is None when the teacher pod's account is not in .env.\n"
     "    from contexts.circle.brain_power import build_shared_brain\n"
     "\n"
     "    shared_brain = build_shared_brain(circle_cfg, bus) if circle else None\n"
     "\n",
     "before"),
    ("                 face_share=face_share,\n", "                 face_share=face_share, shared_brain=shared_brain,\n", "replace"),
]
APP = [
    ("from contexts.circle.face_share_api import face_share_router\n",
     "from contexts.circle.brain_api import group_brain_router\nfrom contexts.circle.brain_power import brain_power_router\n",
     "after"),
    ("    app.include_router(face_share_router(atlas.face_share, atlas.bus, guard=authorized))\n",
     "    # GET /api/circle/brain, POST /api/circle/brain/consent|ask (G4); GET /api/circle/brain/power: the brain's watts\n"
     "    shared = atlas.shared_brain\n"
     "    app.include_router(group_brain_router(getattr(shared, \"brain\", None), guard=authorized))\n"
     "    app.include_router(brain_power_router(getattr(shared, \"power\", None), getattr(shared, \"reason\", None)))\n",
     "after"),
]


def apply(path: Path, edits, check: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    said = []
    for anchor, block, how in edits:
        marker = block.strip().splitlines()[0].strip()
        if marker in text:
            said.append(f"{path.name}: already has {marker[:60]!r}")
            continue
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: the anchor {anchor.strip()[:60]!r} is there {text.count(anchor)} times, not once; "
                             "nothing was changed")
        new = {"after": anchor + block, "before": block + anchor, "replace": block}[how]
        text = text.replace(anchor, new)
        said.append(f"{path.name}: {'would add' if check else 'added'} {marker[:60]!r}")
    if not check:
        path.write_text(text, encoding="utf-8", newline="\n")
    return said


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    root = Path(args[0] if args else ".")
    if not (root / "contexts/circle/brain_power.py").is_file():
        raise SystemExit(f"{root}/contexts/circle/brain_power.py is missing: copy it first")
    gw = root / "contexts/gateway"
    for line in apply(gw / "bootstrap.py", BOOT, check) + apply(gw / "app.py", APP, check):
        print(line)


if __name__ == "__main__":
    main()
