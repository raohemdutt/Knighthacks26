"""Find stored keyframes that still show a face, and blur them again in place.

For a box that saved pictures while the blur was off (`blur_faces` false), or with the older one-pass blur that
could leave a small or half-hidden face findable. It reads <data_dir>/frames/<camera>/<ts_ms>.jpg, looks at each
picture with the strict face detector (score 0.5, margin 0.6: the one the outbound gate and the keyframe store use),
and

    python jetson/reblur_stored.py data                         # dry run: counts per camera and per day, changes nothing
    python jetson/reblur_stored.py data --names                 # the same, and the name of each file with a face
    python jetson/reblur_stored.py data --since 2026-10-10T18:00   # only pictures taken at or after that time
    python jetson/reblur_stored.py data --apply                 # blur those files again, in place

`--apply` blurs each such file again until the detector finds no face (at most five passes) and writes it through a
temporary file in the same folder, so a reader never sees half a picture and a crash leaves the old file whole. A
file in which no face is found is never opened for writing: its bytes and its date stay as they were. Running it
again does nothing to the files that came clear. A file where the detector still fires after five passes (on the
Jetson's own frames that was 2 of 58, both times on the scrambled patch itself) is counted as "still found" and
named, so a person can look at it or delete it; another run blurs only those again.

It opens no network connection and reads nothing but the frames folder. It needs OpenCV and the YuNet model in the
repo, so run it with the gateway's Python from the repo root. The gateway may keep running: new keyframes it writes
meanwhile are simply seen or not, and each replacement is one rename. `--since` takes an ISO time; without an
offset it is the box's local time, and the days in the table are local days too. The time of a picture is the one
in its file name.

Exit code: 0 when no picture with a findable face is left, 2 when some are (a dry run that found any, or files still
found after --apply), 1 when a file could not be read or written or the arguments are wrong.
"""

import argparse
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repo root, so `contexts` imports from any folder

QUALITY = 85  # what the keyframe store writes (contexts/perception/infrastructure/keyframe_store.py)
PASSES = 5


def strict_blur() -> Any:
    from contexts.perception.infrastructure.face_blur import STRICT, FaceBlur

    return FaceBlur(**STRICT)


def read_picture(jpeg: bytes) -> Any:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("not a picture OpenCV can read")
    return frame


def write_picture(frame: Any) -> bytes:
    import cv2

    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok:
        raise ValueError("could not encode the blurred picture")
    return jpeg.tobytes()


def keyframes(data_dir: str | Path, since_ms: int | None = None) -> Iterator[tuple[str, int, Path]]:
    """(camera, ts_ms, path) of every stored keyframe at or after `since_ms`, by camera and then oldest first.
    Links, hidden files (a writer's temporary file) and files not named by a timestamp are left out."""
    found = []
    for path in (Path(data_dir) / "frames").glob("*/*.jpg"):
        if path.is_symlink() or not path.is_file() or not path.stem.isdigit():
            continue
        if since_ms is None or int(path.stem) >= since_ms:
            found.append((path.parent.name, int(path.stem), path))
    yield from sorted(found)


def replace_atomically(path: Path, data: bytes) -> None:
    """Write `data` over `path` by way of a temporary file beside it: the old file is whole until the rename."""
    tmp = path.with_name(f".{path.stem}.reblur.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class Row:
    scanned: int = 0
    with_face: int = 0
    reblurred: int = 0
    still_found: int = 0
    unreadable: int = 0

    def add(self, other: "Row") -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class Report:
    rows: dict[tuple[str, str], Row] = field(default_factory=dict)  # (camera, day) -> counts
    with_face: list[Path] = field(default_factory=list)  # every file a face was found in
    still_found: list[Path] = field(default_factory=list)  # --apply only: a face is still found after re-blurring
    problems: list[tuple[Path, str]] = field(default_factory=list)  # could not be read, or could not be written
    seconds: float = 0.0

    @property
    def total(self) -> Row:
        total = Row()
        for row in self.rows.values():
            total.add(row)
        return total


def reblur(data_dir: str | Path, blur: Any = None, *, apply: bool = False, since_ms: int | None = None,
           tz: tzinfo | None = None, decode: Callable[[bytes], Any] | None = None,
           encode: Callable[[Any], bytes] | None = None, clock: Callable[[], float] = time.monotonic) -> Report:
    """Look at every stored keyframe; with `apply`, blur the ones with a face again and write them in place.
    `blur` is a FaceBlur (anything with .faces(frame) and .scrambled_until_clear(frame, passes)); the strict one
    by default. `tz` is the zone the days are counted in (None: this machine's)."""
    blur = blur or strict_blur()
    decode, encode = decode or read_picture, encode or write_picture  # JPEG bytes <-> a frame, with OpenCV
    report = Report()
    started = clock()
    for camera, ts_ms, path in keyframes(data_dir, since_ms):
        day = datetime.fromtimestamp(ts_ms / 1000, tz).strftime("%Y-%m-%d")
        row = report.rows.setdefault((camera, day), Row())
        row.scanned += 1
        try:
            frame = decode(path.read_bytes())
            if not len(blur.faces(frame)):
                continue  # no face: the file is not touched
            row.with_face += 1
            report.with_face.append(path)
            if not apply:
                continue
            out, _ = blur.scrambled_until_clear(frame, PASSES)
            replace_atomically(path, encode(out))
            row.reblurred += 1
            if len(blur.faces(decode(path.read_bytes()))):  # what is on disk now, as the next run would see it
                row.still_found += 1
                report.still_found.append(path)
        except Exception as error:  # one bad file must not stop the run; it is counted and named
            row.unreadable += 1
            report.problems.append((path, f"{type(error).__name__}: {error}"))
    report.seconds = clock() - started
    return report


def render(report: Report, *, apply: bool, names: bool, where: str) -> str:
    total = report.total
    columns = ["scanned", "with a face"] + (["re-blurred", "still found"] if apply else [])
    width = max([len("camera"), *(len(camera) for camera, _ in report.rows)])

    def line(camera: str, day: str, row: Row) -> str:
        values = [row.scanned, row.with_face] + ([row.reblurred, row.still_found] if apply else [])
        return f"{camera:<{width}}  {day:<10}" + "".join(f"  {v:>{len(c)}}" for v, c in zip(values, columns))

    out = [f"stored keyframes under {where}",
           f"{'camera':<{width}}  {'day':<10}" + "".join(f"  {c}" for c in columns)]
    out += [line(camera, day, row) for (camera, day), row in sorted(report.rows.items())]
    out.append(line("total", "", total))
    if names and not apply:
        out += [f"face: {path}" for path in report.with_face]
    out += [f"still found: {path}" for path in report.still_found]
    out += [f"problem: {path}: {why}" for path, why in report.problems]
    if total.scanned:
        out.append(f"{total.scanned} files in {report.seconds:.1f} s ({1000 * report.seconds / total.scanned:.0f} ms a file)")
    if not apply:
        out.append("dry run: nothing was changed." + (f" Run again with --apply to blur the {total.with_face} "
                   f"{'file' if total.with_face == 1 else 'files'} with a face again." if total.with_face else
                   " No stored keyframe has a face the detector finds."))
    elif total.still_found:
        out.append(f"{total.still_found} still found after re-blurring (named above): look at them, the detector may be "
                   "firing on the blurred patch; another --apply blurs only those again.")
    else:
        out.append(f"{total.reblurred} re-blurred; no stored keyframe has a face the detector finds.")
    return "\n".join(out)


def since_ms(text: str, tz: tzinfo | None = None) -> int:
    """An ISO time as epoch milliseconds; a time without an offset is taken in `tz` (None: this machine's)."""
    when = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=tz) if tz is not None else when.astimezone()
    return int(when.timestamp() * 1000)


def main(argv: list[str] | None = None, *, blur: Any = None, tz: tzinfo | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find stored keyframes that still show a face; with --apply, blur them again in place.")
    parser.add_argument("data_dir", help="the box's data folder (the one that holds frames/)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="count only, change nothing (the default)")
    mode.add_argument("--apply", action="store_true", help="blur the files with a face again, in place")
    parser.add_argument("--since", help="only pictures taken at or after this ISO time (no offset: local time)")
    parser.add_argument("--names", action="store_true", help="dry run: also print the name of each file with a face")
    args = parser.parse_args(argv)
    frames = Path(args.data_dir) / "frames"
    if not frames.is_dir():
        print(f"no frames folder at {frames}", file=sys.stderr)
        return 1
    try:
        start = since_ms(args.since, tz) if args.since else None
    except ValueError:
        print(f"--since is not an ISO time: {args.since!r} (for example 2026-10-10T18:00 or 2026-10-10T18:00:00-04:00)", file=sys.stderr)
        return 1
    if blur is None:
        try:  # once, before any file: a box where the detector cannot run gets one clear line, not one per picture
            import numpy as np

            blur = strict_blur()
            blur.faces(np.zeros((64, 64, 3), np.uint8))
        except Exception as error:
            print(f"the face detector cannot run here ({type(error).__name__}: {error}): nothing was read or changed. "
                  "Run it with the gateway's Python, from the repo.", file=sys.stderr)
            return 1
    report = reblur(args.data_dir, blur, apply=args.apply, since_ms=start, tz=tz)
    where = f"{frames}" + (f", since {args.since}" if args.since else "")
    print(render(report, apply=args.apply, names=args.names, where=where))
    if report.problems:
        return 1
    return 2 if (report.still_found if args.apply else report.with_face) else 0


if __name__ == "__main__":
    sys.exit(main())
