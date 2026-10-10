"""Keyframes on disk: `frames:<camera>:<ts_ms>` -> <data_dir>/frames/<camera>/<ts_ms>.jpg."""

import io
from collections.abc import Callable
from pathlib import Path

from contracts.types import Box, FrameRef, frame_ref

try:  # not a project dependency: used when the box happens to have it; OpenCV (the `camera` extra) otherwise
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


JPEG_QUALITY = 85  # of the copy sent to the model; the stored keyframe is written at the same quality


def scaled_size(w: int, h: int, max_side: int) -> tuple[int, int] | None:
    """(w, h) scaled so the longest side is `max_side`, keeping the aspect ratio. None when there is nothing
    to do: `max_side` is 0 or the image already fits (never scaled up)."""
    longest = max(w, h)
    if max_side <= 0 or longest <= max_side:
        return None
    return max(1, round(w * max_side / longest)), max(1, round(h * max_side / longest))


def crop_jpeg(jpeg: bytes, box: Box) -> bytes:
    """The part of the image inside `box` (normalized), at full resolution.

    With neither Pillow nor OpenCV, or for bytes they cannot read (a truncated or corrupt frame), the whole
    frame: the model still gets something to look at, and one bad frame does not stop the finder."""
    if Image is not None:
        try:
            with Image.open(io.BytesIO(jpeg)) as im:
                w, h = im.size
                area = (round(box.x1 * w), round(box.y1 * h), round(box.x2 * w), round(box.y2 * h))
                out = io.BytesIO()
                im.convert("RGB").crop(area).save(out, "JPEG", quality=95)
        except OSError:  # includes PIL.UnidentifiedImageError
            return jpeg
        return out.getvalue()
    return _with_cv2(jpeg, lambda cv2, im: im[round(box.y1 * im.shape[0]):round(box.y2 * im.shape[0]),
                                              round(box.x1 * im.shape[1]):round(box.x2 * im.shape[1])], quality=95)


def scale_jpeg(jpeg: bytes, max_side: int, quality: int = JPEG_QUALITY) -> bytes:
    """A copy for the model whose longest side is at most `max_side` pixels: fewer image tokens, same picture.

    Keeps the aspect ratio and never scales up: an image that already fits comes back as it is, byte for
    byte, and so does everything when `max_side` is 0. Uses Pillow when the box has it, else OpenCV; with
    neither, or for bytes they cannot read, the frame goes out unchanged."""
    if max_side <= 0:
        return jpeg
    if Image is not None:
        try:
            with Image.open(io.BytesIO(jpeg)) as im:
                size = scaled_size(*im.size, max_side)
                if size is None:
                    return jpeg
                im.draft("RGB", size)  # lets the JPEG decoder do part of the shrinking: much less work on the Jetson
                out = io.BytesIO()
                im.convert("RGB").resize(size, getattr(Image, "Resampling", Image).LANCZOS).save(out, "JPEG", quality=quality)
        except OSError:
            return jpeg
        return out.getvalue()

    def shrink(cv2, im):
        size = scaled_size(im.shape[1], im.shape[0], max_side)
        return None if size is None else cv2.resize(im, size, interpolation=cv2.INTER_AREA)

    return _with_cv2(jpeg, shrink, quality)


def _with_cv2(jpeg: bytes, edit: Callable, quality: int) -> bytes:
    """`edit(cv2, image) -> image` on the decoded JPEG, encoded again. The bytes as given when OpenCV is not
    installed, cannot read them, or `edit` returns None (nothing to change)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return jpeg
    try:
        im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        out = None if im is None else edit(cv2, im)
        if out is None or out.size == 0:
            return jpeg
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception:  # cv2.error on bytes it cannot read
        return jpeg
    return encoded.tobytes() if ok else jpeg
