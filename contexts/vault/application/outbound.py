"""The outbound gate: a camera picture that leaves the box has its faces scrambled, whatever the local switch says.

    gate = OutboundGate(jpeg_scrambler(scramble_frame))      # scramble_frame(frame) -> (frame, faces found)
    jpeg = gate.release(stored, destination="phone", purpose="alert photo")

"Blur faces before saving" (policy.blur_faces) decides what is written to this box's own disk. It does not decide
what leaves: every path that hands a picture to something outside the box (the training export, the photo on a
phone alert) passes the stored JPEG through `release` first.

- No face found: the INPUT BYTES come back untouched, so a frame that was blurred when it was saved is not
  re-encoded and stays byte-identical.
- A face found: every face is scrambled and the picture is encoded again.
- Anything that goes wrong (the bytes are not a picture, the detector fails, its model file is missing, OpenCV is
  not installed) raises `Refused`, and the caller sends nothing. There is no fallback to the original.

The face detector is injected (bootstrap wires perception's YuNet blur), so the vault imports nothing from
perception. contexts/vault/tests/test_outbound_gate.py holds the guard that fails when a module sends a picture
around this gate.
"""

import threading
from collections.abc import Callable
from typing import Any

ScrambleFrame = Callable[[Any], tuple[Any, int]]  # BGR frame -> (the frame with every face scrambled, faces found)
ScrambleJpeg = Callable[[bytes], tuple[bytes, int]]  # JPEG -> (the JPEG with every face scrambled, faces found)
OnRelease = Callable[[str, str, int, int, int], None]  # destination, purpose, pictures, bytes, faces_blurred


class Refused(Exception):
    """This picture may not leave: the gate could not check it for faces. The caller sends nothing."""


def jpeg_scrambler(scramble_frame: ScrambleFrame, *, quality: int = 90) -> ScrambleJpeg:
    """Decode, scramble every face, encode again. Needs OpenCV; without it every call raises, so the gate refuses."""

    def scramble(jpeg: bytes) -> tuple[bytes, int]:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("not a picture OpenCV can read")
        scrambled, faces = scramble_frame(frame)
        if not faces:
            return jpeg, 0
        ok, encoded = cv2.imencode(".jpg", scrambled, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("could not encode the scrambled picture")
        return encoded.tobytes(), int(faces)

    return scramble


class OutboundGate:
    def __init__(self, scramble: ScrambleJpeg, *, on_release: OnRelease | None = None) -> None:
        self.scramble = scramble
        self.on_release = on_release
        self._lock = threading.Lock()  # one detector, callers on several threads (the export, the keyframe route)

    def release(self, jpeg: bytes, *, destination: str, purpose: str) -> bytes:
        """The bytes that may go to `destination`. Raises Refused instead of ever returning an unchecked picture."""
        # A later PR adds the owner's grant here (an approved, unblurred release); until then there is no way round.
        try:
            with self._lock:
                out, faces = self.scramble(jpeg)
            if faces and (not out or out == jpeg):
                raise ValueError(f"{faces} faces found but the picture came back unchanged")
        except Exception as exc:
            raise Refused(f"picture for {destination} ({purpose}) held back: {type(exc).__name__}: {exc}") from exc
        if not faces:
            out = jpeg  # nothing to hide: exactly the bytes that came in
        if self.on_release is not None:
            self.on_release(destination, purpose, 1, len(out), int(faces))
        return out
