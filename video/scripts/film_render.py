"""Render the film fast.

    python video/scripts/film_render.py day             # one scene, 15 fps draft  -> video/hf/renders/scene-day.mp4
    python video/scripts/film_render.py day kitchen     # several scenes, rendered side by side (--jobs)
    python video/scripts/film_render.py all             # the whole film, 15 fps draft -> renders/film-draft.mp4
    python video/scripts/film_render.py all --final     # the whole film, 30 fps delivery -> renders/film.mp4
    python video/scripts/film_render.py all --whole     # the whole film as one render (no scene split)

Runs film_build.py first. Scene renders use scene-<id>.html files (and compositions/_cap-<id>.html captions) that
exist only during the render (a second root file would fail `hyperframes check`). Each scene preview runs up to the
next scene's start, so "all" renders every scene side by side (--jobs at a time, default 3; the CPU had headroom:
two scenes together took 48 s against 69 s one after the other) and joins them in order with ffmpeg, no re-encode.
Drafts are 15 fps; --final is 30 fps delivery quality.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = Path(__file__).resolve().parents[1] / "hf"


def render(comp: str | None, out: Path, final: bool, workers: int = 8) -> int:
    cmd = ["npx", "hyperframes", "render", "--output", str(out), "--workers", str(workers)]
    cmd += ["--quality", "delivery"] if final else ["--quality", "draft", "--fps", "15"]
    if comp:
        cmd += ["--composition", comp]
    print(" ".join(cmd), flush=True)
    log = HF / "renders" / (out.stem + ".log")
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd, cwd=HF, shell=sys.platform == "win32", stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"{'ok  ' if rc == 0 else 'FAIL'} {out.name} (log {log.name})", flush=True)
    return rc


def join(parts: list[Path], out: Path) -> int:
    lst = HF / "renders" / "_parts.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)]).returncode
    lst.unlink()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--whole", action="store_true", help="render 'all' as one composition instead of scene by scene")
    ap.add_argument("--jobs", type=int, default=3, help="scene renders at the same time")
    a = ap.parse_args()
    (HF / "renders").mkdir(exist_ok=True)
    build = [sys.executable, str(HF.parent / "scripts" / "film_build.py")]
    out_all = HF / "renders" / ("film.mp4" if a.final else "film-draft.mp4")
    if a.scenes == ["all"] and a.whole:
        subprocess.run(build, check=True)
        return render(None, out_all, a.final)
    subprocess.run(build + ["--scene-previews"], check=True)
    ids = [s["id"] for s in json.loads((HF / "film.json").read_text(encoding="utf-8"))["scenes"]]
    want = ids if a.scenes == ["all"] else a.scenes
    for sid in want:
        if sid not in ids:
            print(f"no scene called {sid!r}")
            return 1
    jobs = max(1, min(a.jobs, len(want)))
    workers = max(3, 8 // jobs + 2)
    suffix = "-final" if a.final else ""
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            rcs = list(pool.map(lambda sid: render(f"scene-{sid}.html", HF / "renders" / f"scene-{sid}{suffix}.mp4", a.final, workers), want))
    finally:
        for f in [*HF.glob("scene-*.html"), *(HF / "compositions").glob("_cap-*.html")]:
            f.unlink()
    if any(rcs):
        return 1
    if a.scenes == ["all"]:
        rc = join([HF / "renders" / f"scene-{sid}{suffix}.mp4" for sid in ids], out_all)
        print(f"{'joined' if rc == 0 else 'JOIN FAILED'} -> {out_all}")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
