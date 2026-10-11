"""Record a short camera clip on the Jetson through the gateway, and turn recorded clips into a lab dataset.

The gateway owns the webcam: it keeps /dev/video0 open, and /dev/video1 is the same webcam's metadata channel, not a
picture, so a second reader cannot get frames from either node. The gateway already serves every frame it reads at
GET /capture/<camera>/snapshot.jpg, faces blurred while the Vault policy blur_faces is on. This script polls that
endpoint, keeps each new frame, and has GStreamer join the frames into an H.264 mp4 (one `nice 19` process for a few
seconds). It opens no video device, stops no service, writes nothing under data/, and refuses to record while
blur_faces is off.

Stdlib only, Python 3.10. Run it on the box through key SSH with the script on stdin, so nothing is installed there:

    ssh -o BatchMode=yes alexander@100.124.34.36 python3 - clip keys-on-desk --seconds 10 < jetson/record_clip.py
    ssh -o BatchMode=yes alexander@100.124.34.36 python3 - clip person-walks-through --when person-moves --wait-max 300 < jetson/record_clip.py
    ssh -o BatchMode=yes alexander@100.124.34.36 python3 - dataset keys-on-desk package-at-door --per-clip 6 < jetson/record_clip.py

`clip NAME` writes, under --out (default ~/atlas/fixtures/clips):

    NAME.mp4                the clip (1280x720 as the gateway serves it, H.264, --fps frames a second)
    NAME/frames/00001.jpg   the frames it was made from, byte for byte as the gateway served them
    NAME/clip.json          camera, start time, frame offsets, measured rate, the policy at the time, the mp4's sha256

--when person-moves waits (at most --wait-max seconds, exit code 3 when nobody comes) until the gateway's tracker
sees a person arrive or move across the view, then records. Exit code 1 means the gateway, the camera or GStreamer
was not fine (the message says which) and nothing half-written is left as a clip; exit code 4 means the clip was kept
but the gateway did not look fine when it was done, so check /health before anything else.

`dataset NAME...` picks --per-clip evenly spaced frames from each recorded clip into a dataset folder in the box's
export layout (manifest.json, examples.jsonl, images/*.jpg: contracts/learning.py), which lab/dataset.py reads. Its
manifest carries the action id `fixture:desk-clips`: nobody approved an export, this is fixture data.
"""

import argparse
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_GATEWAY = "http://127.0.0.1:8000"
DEFAULT_OUT = "~/atlas/fixtures/clips"
DEFAULT_CAMERA = "cam-desk"
POLL_S = 0.02  # how often the snapshot is asked for; the gateway answers a repeated ask for the same frame from its cache
MAX_GAP_S = 1.0  # a clip with a longer hole between two frames is refused: the camera or the gateway stalled
RATE_TOLERANCE = 0.1  # the frames per second the camera really gave may differ from the clip's rate by this much
GST_TIMEOUT_S = 180  # joining ten seconds of frames takes a few seconds at nice 19; this only ends a hung pipeline
FRAME_NAME = "%05d.jpg"
TASK = "desk-objects"
INSTRUCTION = "Name the small things on the desk."
DATASET_ID = "desk-clips-fixture"
ACTION_ID = "fixture:desk-clips"  # nobody approved this: it is fixture data recorded for pipeline tests, not an export
SOURCE = "gateway snapshot.jpg (faces blurred)"
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class RecordError(Exception):
    """The gateway, the camera or GStreamer was not fine, or the request was not allowed; str() says why."""


class NoPerson(Exception):
    """--when person-moves: nobody arrived or moved within --wait-max."""


def log(message: str) -> None:
    print(f"record_clip: {message}", file=sys.stderr, flush=True)


# --- the gateway ---------------------------------------------------------------------------------

class Gateway:
    """The gateway's HTTP API over one kept-alive connection."""

    def __init__(self, base: str = DEFAULT_GATEWAY, timeout: float = 5.0) -> None:
        parts = urlsplit(base)
        self.host, self.port, self.timeout = parts.hostname or "127.0.0.1", parts.port or 80, timeout
        self._conn: http.client.HTTPConnection | None = None

    def get(self, path: str) -> tuple[int, bytes]:
        """(status, body); one retry on a dropped connection, then the OSError or HTTPException."""
        for attempt in (1, 2):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self._conn.request("GET", path)
                response = self._conn.getresponse()
                return response.status, response.read()
            except (OSError, http.client.HTTPException):
                self.close()
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def json(self, path: str) -> dict:
        status, body = self.get(path)
        if status != 200:
            raise RecordError(f"GET {path} answered {status}")
        return json.loads(body)

    def snapshot(self, camera: str) -> bytes | None:
        """The newest blurred frame as JPEG bytes; None when the gateway has no picture right now (it answers 404)."""
        status, body = self.get(f"/capture/{camera}/snapshot.jpg")
        return body if status == 200 else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def check_gateway(gateway: Gateway, camera: str) -> dict:
    """The gateway's own account of itself: answering, the camera delivering frames, faces being blurred."""
    try:
        health = gateway.json("/health")
        status = gateway.json("/capture/status")["cameras"].get(camera)
        policy = gateway.json("/vault/policy")
    except (OSError, http.client.HTTPException, ValueError, KeyError) as err:
        raise RecordError(f"the gateway did not answer properly ({type(err).__name__}: {err})") from None
    if health.get("ok") is not True:
        raise RecordError(f"the gateway says it is not ok: {health}")
    if status is None or status.get("status") not in ("ok", "degraded"):  # degraded: frames arrive, below 70% of the target
        raise RecordError(f"camera {camera} is not delivering frames: {status}")
    if policy.get("blur_faces") is not True:
        raise RecordError("the Vault policy has blur_faces off: nothing is recorded unblurred")
    if camera in (policy.get("paused_cameras") or []):
        raise RecordError(f"camera {camera} is paused by the Vault policy")
    return {"status": status["status"], "fps_actual": status.get("fps_actual"), "dropped": status.get("dropped"),
            "blur_faces": True}


# --- a person arriving or moving -----------------------------------------------------------------

class WalkDetector:
    """True once the tracker's samples show a person who was not there at the first sample, or a person whose box
    center has travelled `move` (a fraction of the frame) within the last `window_s` seconds."""

    def __init__(self, move: float = 0.25, window_s: float = 2.0) -> None:
        self.move, self.window_s = move, window_s
        self._known: set[int] | None = None
        self._centers: dict[int, list[tuple[float, float, float]]] = {}

    def update(self, t: float, tracks: list[dict]) -> bool:
        people = [track for track in tracks if track.get("label") == "person"]
        ids = {person["track_id"] for person in people}
        if self._known is None:
            self._known = set(ids)
        elif ids - self._known:
            return True
        for person in people:
            box = person["box"]
            cx, cy = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
            history = self._centers.setdefault(person["track_id"], [])
            history.append((t, cx, cy))
            while t - history[0][0] > self.window_s:
                history.pop(0)
            if max(abs(cx - history[0][1]), abs(cy - history[0][2])) >= self.move:
                return True
        return False


def wait_for_person(gateway: Gateway, camera: str, wait_max: float, clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep, poll_s: float = 0.25) -> bool:
    """True when a person arrived or moved within wait_max seconds, False when nobody did. A gateway that is
    restarting meanwhile only costs the samples it missed."""
    detector, start = WalkDetector(), clock()
    while clock() - start < wait_max:
        try:
            tracks = gateway.json(f"/capture/{camera}/tracks")["tracks"]
        except (OSError, http.client.HTTPException, RecordError, ValueError, KeyError):
            tracks = None
        if tracks is not None and detector.update(clock() - start, tracks):
            return True
        sleep(poll_s)
    return False


# --- frames and the clip -------------------------------------------------------------------------

def grab_frames(fetch: Callable[[], bytes | None], seconds: float, clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep, poll_s: float = POLL_S) -> list[tuple[float, bytes]]:
    """(seconds since the start, JPEG) for each new frame fetch() returned over `seconds`. fetch() returns None
    when there is no picture; a frame equal to the one before it is the same frame asked for twice."""
    frames: list[tuple[float, bytes]] = []
    last: bytes | None = None
    start = clock()
    while clock() - start < seconds:
        try:
            jpeg = fetch()
        except (OSError, http.client.HTTPException):
            jpeg = None
        if jpeg is not None and jpeg != last:
            frames.append((clock() - start, jpeg))
            last = jpeg
        sleep(poll_s)
    return frames


def jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's start-of-frame marker; ValueError when it is not a JPEG or has none."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            raise ValueError("damaged JPEG: a marker was expected")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
        elif marker == 0xD9:  # end of image before any frame header
            break
        elif marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers with no length
            i += 2
        elif marker in _SOF_MARKERS:
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        else:
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError("no start-of-frame marker")


def gst_command(frames_dir: Path, fps: float, out: Path) -> list[str]:
    """The one GStreamer pipeline that joins the frames: JPEG in, H.264 main profile in an mp4 out, at low priority
    so the gateway's detector and the language model keep the CPU."""
    rate = Fraction(str(fps)).limit_denominator(100)
    return ["nice", "-n", "19", "gst-launch-1.0", "-q", "-e",
            "multifilesrc", f"location={frames_dir / FRAME_NAME}", "start-index=1", f"caps=image/jpeg,framerate={rate.numerator}/{rate.denominator}",
            "!", "jpegdec", "!", "videoconvert", "!",
            "x264enc", "speed-preset=veryfast", "pass=quant", "quantizer=22", "key-int-max=15", "bframes=0", "!",
            "video/x-h264,profile=main", "!", "h264parse", "!", "mp4mux", "faststart=true", "!",
            "filesink", f"location={out}"]


def run_gst(command: list[str]) -> None:
    for tool in ("nice", "gst-launch-1.0"):
        if shutil.which(tool) is None:
            raise RecordError(f"{tool} is not installed here")
    try:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=GST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RecordError(f"GStreamer did not finish within {GST_TIMEOUT_S} s") from None
    if done.returncode != 0:
        raise RecordError(f"GStreamer failed ({done.returncode}): {done.stdout.decode('utf-8', 'replace')[-400:]}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(gateway: Gateway, name: str, out_dir: Path, *, camera: str = DEFAULT_CAMERA, seconds: float = 10.0,
           fps: float | None = 15, when: str = "now", wait_max: float = 300.0,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
           runner: Callable[[list[str]], None] = run_gst) -> dict:
    """Record one clip and return its clip.json. Nothing is left under out_dir when it fails."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise RecordError(f"{name!r} is not a clip name (lower case letters, digits and dashes)")
    clip_dir, mp4 = out_dir / name, out_dir / f"{name}.mp4"
    if clip_dir.exists() or mp4.exists():
        raise RecordError(f"{name} already exists under {out_dir}; pick another name or move it away")
    before = check_gateway(gateway, camera)
    if when == "person-moves":
        log(f"waiting up to {wait_max:.0f} s for a person to arrive or move in front of {camera}")
        if not wait_for_person(gateway, camera, wait_max, clock, sleep):
            raise NoPerson(f"nobody arrived or moved within {wait_max:.0f} s")
    started = datetime.now(timezone.utc)
    log(f"recording {name} for {seconds:g} s from {camera}")
    frames = grab_frames(lambda: gateway.snapshot(camera), seconds, clock, sleep)
    if not frames:
        raise RecordError(f"the gateway gave no picture from {camera} in {seconds:g} s")
    offsets = [offset for offset, _ in frames]
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:])] + [seconds - offsets[-1], offsets[0]]
    if max(gaps) > MAX_GAP_S:
        raise RecordError(f"the picture stalled for {max(gaps):.1f} s during the clip; nothing was kept")
    measured = len(frames) / seconds
    if fps is None:  # the rate the camera really gave, so the clip plays at true speed
        fps = round(measured, 1)
    if abs(measured - fps) > RATE_TOLERANCE * fps:
        raise RecordError(f"the camera gave {measured:.1f} frames a second, not about {fps}: joined at {fps} the clip would "
                          f"play at the wrong speed, so nothing was kept (--fps auto records it as it is)")
    width, height = jpeg_size(frames[0][1])
    try:
        (clip_dir / "frames").mkdir(parents=True)
        for number, (_, jpeg) in enumerate(frames, 1):
            (clip_dir / "frames" / (FRAME_NAME % number)).write_bytes(jpeg)
        command = gst_command(clip_dir / "frames", fps, mp4)
        runner(command)
        if not mp4.is_file() or mp4.stat().st_size == 0:
            raise RecordError("GStreamer finished but wrote no clip")
        try:
            after = check_gateway(gateway, camera)
        except RecordError as err:  # the clip is fine; the caller is told the gateway did not look fine afterwards
            after = {"error": str(err)}
        meta = {"name": name, "camera": camera, "source": SOURCE, "when": when, "started": started.isoformat(),
                "seconds_requested": seconds, "frames": len(frames),
                "fps_measured": round(measured, 2), "fps_clip": fps, "width": width, "height": height,
                "max_gap_s": round(max(gaps), 3), "blur_faces": True, "mp4": mp4.name, "mp4_bytes": mp4.stat().st_size,
                "mp4_sha256": sha256_of(mp4), "gst": command, "gateway_before": before, "gateway_after": after,
                "offsets_s": [round(offset, 3) for offset in offsets]}
        (clip_dir / "clip.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        return meta
    except BaseException:
        shutil.rmtree(clip_dir, ignore_errors=True)
        mp4.unlink(missing_ok=True)
        raise


# --- the dataset the lab reads -------------------------------------------------------------------

def pick(count: int, wanted: int) -> list[int]:
    """`wanted` indices spread evenly over range(count), the first and the last included."""
    if count <= wanted:
        return list(range(count))
    if wanted == 1:
        return [count // 2]
    return [round(i * (count - 1) / (wanted - 1)) for i in range(wanted)]


def build_dataset(out_dir: Path, names: list[str], dest: Path, per_clip: int = 6, dataset_id: str = DATASET_ID,
                  now: datetime | None = None) -> dict:
    """Write the dataset folder for the recorded clips `names` into `dest` and return its manifest."""
    examples_file = dest / "examples.jsonl"
    if examples_file.is_file() and any(json.loads(line).get("target") for line in examples_file.read_text(encoding="utf-8").splitlines() if line):
        raise RecordError(f"{dest} already holds labels; rebuilding it would drop them")
    (dest / "images").mkdir(parents=True, exist_ok=True)
    for stale in (dest / "images").glob("*.jpg"):
        stale.unlink()
    rows: list[dict] = []
    for name in names:
        meta = json.loads((out_dir / name / "clip.json").read_text(encoding="utf-8"))
        files = sorted((out_dir / name / "frames").glob("*.jpg"))
        if len(files) != meta["frames"]:
            raise RecordError(f"{name}: clip.json says {meta['frames']} frames, the folder holds {len(files)}")
        started = datetime.fromisoformat(meta["started"])
        for index in pick(len(files), per_clip):
            image = f"images/{len(rows):04d}.jpg"
            shutil.copyfile(files[index], dest / image)
            ts = started + timedelta(seconds=meta["offsets_s"][index])
            rows.append({"id": f"{name}-{index + 1:05d}", "image": image, "frame_ref": f"fixture:clip:{name}:{index + 1:05d}",
                         "ts": ts.isoformat(), "context": None, "target": None, "verdict": None})
    examples_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {"dataset_id": dataset_id,
                "spec": {"task": TASK, "instruction": INSTRUCTION, "output_schema": {}, "camera": DEFAULT_CAMERA,
                         "label": None, "examples": []},
                "created": (now or datetime.now(timezone.utc)).isoformat(), "examples": len(rows), "action_id": ACTION_ID}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --- command line --------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    clip = sub.add_parser("clip", help="record one clip through the gateway")
    clip.add_argument("name")
    clip.add_argument("--camera", default=DEFAULT_CAMERA)
    clip.add_argument("--seconds", type=float, default=10.0)
    clip.add_argument("--fps", type=lambda v: None if v == "auto" else float(v), default=15,
                      help="the clip's frame rate, or auto for the rate the camera really gave (default 15)")
    clip.add_argument("--gateway", default=DEFAULT_GATEWAY)
    clip.add_argument("--out", default=DEFAULT_OUT)
    clip.add_argument("--when", choices=("now", "person-moves"), default="now")
    clip.add_argument("--wait-max", type=float, default=300.0)
    data = sub.add_parser("dataset", help="pick frames from recorded clips into an export-layout dataset")
    data.add_argument("names", nargs="+")
    data.add_argument("--per-clip", type=int, default=6)
    data.add_argument("--out", default=DEFAULT_OUT)
    data.add_argument("--dest", help=f"the dataset folder (default: <out>/{DATASET_ID})")
    data.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args(argv)
    out_dir = Path(args.out).expanduser()
    try:
        if args.command == "clip":
            gateway = Gateway(args.gateway)
            try:
                result = record(gateway, args.name, out_dir, camera=args.camera, seconds=args.seconds, fps=args.fps,
                                when=args.when, wait_max=args.wait_max)
            finally:
                gateway.close()
            summary = {key: result[key] for key in ("name", "camera", "frames", "fps_measured", "width", "height",
                                                    "mp4", "mp4_bytes", "mp4_sha256", "gateway_after")}
        else:
            dest = Path(args.dest).expanduser() if args.dest else out_dir / args.dataset_id
            manifest = build_dataset(out_dir, args.names, dest, args.per_clip, args.dataset_id)
            summary = {"dataset": str(dest), "examples": manifest["examples"], "dataset_id": manifest["dataset_id"]}
    except NoPerson as err:
        log(str(err))
        return 3
    except RecordError as err:
        log(str(err))
        return 1
    print(json.dumps(summary))
    if args.command == "clip" and "error" in result["gateway_after"]:
        log(f"the clip is kept, but the gateway did not look fine afterwards: {result['gateway_after']['error']}")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
