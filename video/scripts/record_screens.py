"""Record the phone app's screens as slot clips, from a gateway running on the seeded demo night (D12).

    python -m contexts.demo seed --data-dir <dir> --now
    ATLAS_DATA_DIR=<dir> ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 ATLAS_PORT=8765 python -m contexts.gateway
    python video/scripts/record_screens.py --gateway http://127.0.0.1:8765 --out video/public/clips [--only 1 8 9] [--shots]

Each slot is a list of (path, seconds) the browser visits in turn, scrolling slowly, at 1920x1080. The recording
is re-encoded to 30 fps H.264 and cut to the slot's length. `--shots` only saves one screenshot per stop, to check
the screens before recording a minute of them.
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from patchright.sync_api import sync_playwright

SLOT_SECONDS = 59.8
WIDTH, HEIGHT = 1920, 1080

# What each minute shows, from Hemdutt's agent's list of what is filmable on the seeded night (Team Comms).
TOURS: dict[int, list[tuple[str, float]]] = {
    1: [("/now/", 24), ("/today/", 18), ("/approvals/", 18)],
    8: [("/circle/?tab=people", 14), ("/circle/?tab=plans", 16), ("/circle/?tab=league", 16), ("/dayrun/?plan=good&play=1", 14)],
    9: [("/dayrun/?plan=bad&play=1", 18), ("/dayrun/?plan=good&play=1", 22), ("/dayrun/?hard=1&play=1", 20)],
}


def tour(page, base: str, stops: list[tuple[str, float]], shots: Path | None, key: int) -> None:
    for i, (path, seconds) in enumerate(stops):
        page.goto(base + path, wait_until="networkidle")
        time.sleep(1.5)
        if shots:
            page.screenshot(path=str(shots / f"slot-{key}-{i}.png"))
            continue
        # Dwell, with a slow scroll down and back so a long screen shows all of itself.
        height = page.evaluate("document.documentElement.scrollHeight")
        steps = max(1, int(seconds * 4))
        for s in range(steps):
            t = s / steps
            y = (height - HEIGHT) * (t * 2 if t < 0.5 else 2 - t * 2) if height > HEIGHT else 0
            page.evaluate(f"window.scrollTo(0, {max(0, y)})")
            time.sleep(0.25)


def encode(webm: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(webm), "-t", f"{SLOT_SECONDS:.1f}", "-r", "30",
                    "-vf", f"scale={WIDTH}:{HEIGHT}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8765")
    ap.add_argument("--out", type=Path, default=Path("video/public/clips"))
    ap.add_argument("--only", type=int, nargs="*", default=sorted(TOURS))
    ap.add_argument("--shots", action="store_true", help="screenshots only, no recording")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shots = args.out / "shots" if args.shots else None
    if shots:
        shots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for key in args.only:
            with tempfile.TemporaryDirectory() as tmp:
                ctx = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1,
                                          color_scheme="light", **({} if shots else {"record_video_dir": tmp, "record_video_size": {"width": WIDTH, "height": HEIGHT}}))
                page = ctx.new_page()
                tour(page, args.gateway.rstrip("/"), TOURS[key], shots, key)
                video = page.video
                ctx.close()
                if not shots and video:
                    webm = Path(video.path())
                    encode(webm, args.out / f"slot-{key}.mp4")
                    print(f"slot {key}: {args.out / f'slot-{key}.mp4'}")
        browser.close()
    if shots:
        print(f"screenshots in {shots}")
    if shutil.which("ffprobe") and not shots:
        for key in args.only:
            d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(args.out / f"slot-{key}.mp4")],
                               capture_output=True, text=True).stdout.strip()
            print(f"slot {key}: {d} s")


if __name__ == "__main__":
    main()
