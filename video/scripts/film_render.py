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


def add_music(film: Path, out: Path) -> int:
    """The music bed spans several scenes, so the scene renders carry only the voice and the sound effects; lay the
    music sections (film.json "music", cut by film_build.py) over the joined film at their times."""
    clips = json.loads((HF / "film.json").read_text(encoding="utf-8")).get("music", [])
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(film)]
    for c in clips:
        cmd += ["-i", str(HF / c["file"])]
    if clips:
        parts = [f"[{k + 1}:a]adelay={int(c['at'] * 1000)}:all=1[m{k}]" for k, c in enumerate(clips)]
        mix = "".join(f"[m{k}]" for k in range(len(clips)))
        graph = ";".join(parts) + f";[0:a]{mix}amix=inputs={len(clips) + 1}:duration=first:normalize=0[a]"
        cmd += ["-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    else:
        cmd += ["-c", "copy", str(out)]
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--whole", action="store_true", help="render 'all' as one composition instead of scene by scene")
    ap.add_argument("--jobs", type=int, default=3, help="scene renders at the same time")
    ap.add_argument("--workers", type=int, default=0, help="browser workers per scene render (default: about 8 / jobs)")
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
    workers = a.workers or max(3, 8 // jobs + 2)  # jobs x workers ~ the machine's threads (16 on Dom's laptop: --jobs 8 --workers 2)
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
        bare = out_all.with_name(out_all.stem + "-nomusic.mp4")
        rc = join([HF / "renders" / f"scene-{sid}{suffix}.mp4" for sid in ids], bare) or add_music(bare, out_all)
        print(f"{'joined' if rc == 0 else 'JOIN FAILED'} -> {out_all}")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
