"""Render the film fast.

    python video/scripts/film_render.py day             # one scene, 15 fps draft  -> video/hf/renders/scene-day.mp4
    python video/scripts/film_render.py day kitchen     # several scenes
    python video/scripts/film_render.py all             # the whole film, 15 fps draft -> renders/film-draft.mp4
    python video/scripts/film_render.py all --final     # the whole film, 30 fps delivery -> renders/film.mp4

Runs film_build.py first. Scene renders use scene-<id>.html files that exist only during the render (a second root
file would fail `hyperframes check`). Drafts are 15 fps on 8 Chrome workers; --final is 30 fps delivery quality.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HF = Path(__file__).resolve().parents[1] / "hf"


def render(comp: str | None, out: Path, final: bool) -> int:
    cmd = ["npx", "hyperframes", "render", "--output", str(out), "--workers", "8"]
    cmd += ["--quality", "delivery"] if final else ["--quality", "draft", "--fps", "15"]
    if comp:
        cmd += ["--composition", comp]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HF, shell=sys.platform == "win32").returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    (HF / "renders").mkdir(exist_ok=True)
    build = [sys.executable, str(HF.parent / "scripts" / "film_build.py")]
    if a.scenes == ["all"]:
        subprocess.run(build, check=True)
        return render(None, HF / "renders" / ("film.mp4" if a.final else "film-draft.mp4"), a.final)
    subprocess.run(build + ["--scene-previews"], check=True)
    rc = 0
    try:
        for sid in a.scenes:
            if not (HF / f"scene-{sid}.html").exists():
                print(f"no scene called {sid!r}"); return 1
            rc |= render(f"scene-{sid}.html", HF / "renders" / f"scene-{sid}.mp4", a.final)
    finally:
        for f in HF.glob("scene-*.html"):
            f.unlink()
    return rc


if __name__ == "__main__":
    sys.exit(main())
