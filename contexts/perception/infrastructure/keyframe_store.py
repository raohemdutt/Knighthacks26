"""Keyframes on disk: the writer for `frames:<camera>:<ts_ms>` -> <data_dir>/frames/<camera>/<ts_ms>.jpg.

`keyframe_loader` in frames.py reads them back. Faces are blurred by `blur(frame) -> frame` (P6)
before the JPEG is encoded, so an unblurred frame never reaches the disk.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from contracts.types import FrameRef, frame_ref


class KeyframeStore:
    def __init__(self, data_dir: str | Path = "data", *, quality: int = 85,
                 blur: Callable[[Any], Any] | None = None) -> None:
        self.root = Path(data_dir) / "frames"
        self.quality = quality
        self.blur = blur

    def __call__(self, camera_id: str, ts_ms: int, frame: Any) -> FrameRef:
        import cv2

        if self.blur is not None:
            frame = self.blur(frame)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise ValueError("could not encode the frame")
        folder = self.root / camera_id
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{ts_ms}.tmp"
        tmp.write_bytes(jpeg.tobytes())
        os.replace(tmp, folder / f"{ts_ms}.jpg")  # readers never see half a file
        return frame_ref(camera_id, ts_ms)
