"""Keyframes on disk: `frames:<camera>:<ts_ms>` -> <data_dir>/frames/<camera>/<ts_ms>.jpg."""

import io
from collections.abc import Callable
from pathlib import Path

from contracts.types import Box, FrameRef, frame_ref

try:  # not a project dependency: used when the box happens to have it
    from PIL import Image
except ImportError:
    Image = None


def keyframe_loader(data_dir: str | Path = "data") -> Callable[[FrameRef], bytes]:
    def load(ref: FrameRef) -> bytes:
        _, camera, ts_ms = ref.split(":", 2)
        return (Path(data_dir) / "frames" / camera / f"{ts_ms}.jpg").read_bytes()

    return load


def stored_frames(data_dir: str | Path = "data") -> list[tuple[str, FrameRef]]:
    """(camera, frame_ref) of every stored keyframe, oldest first. Files not named by a timestamp are skipped."""
    found = [(int(p.stem), p.parent.name) for p in (Path(data_dir) / "frames").glob("*/*.jpg") if p.stem.isdigit()]
    return [(camera, frame_ref(camera, ts_ms)) for ts_ms, camera in sorted(found)]


def crop_jpeg(jpeg: bytes, box: Box) -> bytes:
    """The part of the image inside `box` (normalized), at full resolution. Without Pillow: the whole frame."""
    if Image is None:
        return jpeg
    with Image.open(io.BytesIO(jpeg)) as im:
        w, h = im.size
        area = (round(box.x1 * w), round(box.y1 * h), round(box.x2 * w), round(box.y2 * h))
        out = io.BytesIO()
        im.convert("RGB").crop(area).save(out, "JPEG", quality=95)
    return out.getvalue()
