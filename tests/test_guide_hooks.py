"""The look guide's hooks on each screen: a redesign that drops one fails here, by name.

The manifest is web/lib/guide-hooks.json, next to the registry (web/lib/guide.ts). It lists the `data-guide` attributes
the guide resolves by attribute, and, separately, the headings and labels it finds by their text. Each must still be in
the source file that renders it. `node web/scripts/check-guide.mjs` runs the same check.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "web" / "lib" / "guide-hooks.json"
REGISTRY = "web/lib/guide.ts (and web/lib/guide-hooks.json, which lists what each screen must keep)"


def missing(manifest: dict, root: Path) -> list[str]:
    """One line per hook or heading that no longer appears in any of the files named for it."""
    out = []
    for kind, key, what in (("attributes", "need", "the hook"), ("headings", "text", "the heading or label")):
        for row in manifest[kind]:
            sources = []
            for name in row["files"]:
                path = root / name
                if not path.is_file():
                    out.append(f"{row['screen']}: {name} is gone; the look guide's targets there need a new home. See {REGISTRY}.")
                    continue
                sources.append(path.read_text(encoding="utf-8"))
            for needle in row[key]:
                if not any(needle in text for text in sources):
                    out.append(f"{row['screen']} lost {what} {needle!r} (expected in {' or '.join(row['files'])}"
                               + (f"; it is {row['why']}" if row.get("why") else "") + f"). The look guide points there: put it back, or change {REGISTRY}.")
    return out


def test_every_screen_still_has_the_hooks_the_look_guide_points_at():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["attributes"]) >= 10 and len(manifest["headings"]) >= 6
    lost = missing(manifest, ROOT)
    assert not lost, "\n" + "\n".join(lost)


def test_the_check_names_the_screen_and_the_hook_when_one_is_dropped(tmp_path):
    (tmp_path / "web" / "app" / "map").mkdir(parents=True)
    (tmp_path / "web" / "app" / "map" / "page.tsx").write_text('<div data-guide="map:bins"><h2>Where are my things?</h2></div>')
    manifest = {"attributes": [{"screen": "Room (/map/)", "why": "the room view", "files": ["web/app/map/page.tsx"], "need": ['data-guide="map:room"', 'data-guide="map:bins"']}],
                "headings": [{"screen": "Room (/map/)", "files": ["web/app/map/page.tsx", "web/app/map/gone.tsx"], "text": ["Where are my things?", "Right now"]}]}
    lost = missing(manifest, tmp_path)
    assert len(lost) == 3
    assert lost[0].startswith("Room (/map/) lost the hook 'data-guide=\"map:room\"'") and "web/app/map/page.tsx" in lost[0] and "web/lib/guide.ts" in lost[0]
    assert "gone.tsx is gone" in lost[1]
    assert lost[2].startswith("Room (/map/) lost the heading or label 'Right now'")
