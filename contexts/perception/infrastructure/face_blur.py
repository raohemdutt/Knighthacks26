"""P6: blur faces before a keyframe is stored or leaves the box.

    store = KeyframeStore("data", blur=FaceBlur())

OpenCV's YuNet face detector (cv2.FaceDetectorYN) finds the faces; each one, with a margin, is
pixelated and smoothed past recognition. If the detector cannot run, the error propagates and the
frame is not stored: an unblurred frame never reaches the disk. Only the copy that is stored is
blurred; the live frame the detector and tracker see is untouched.
"""

from pathlib import Path
from typing import Any

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


class FaceBlur:
    def __init__(self, model: str | Path = MODEL, *, score: float = 0.6, margin: float = 0.3,
                 detector: Any = None) -> None:
        """`score` is the minimum face confidence (lower blurs more, at the cost of blurring non-faces).
        `margin` grows each face box by that fraction on every side (hair, ears, a turned head)."""
        self.score, self.margin = score, margin
        self._model = str(model)
        self._detector = detector  # anything with .detect(frame) -> (_, faces rows [x, y, w, h, ..., score])
        self._size: tuple[int, int] | None = None

    def faces(self, frame: Any) -> list[Any]:
        """YuNet rows for the frame: [x, y, w, h, 5 landmark points (x, y each), score]."""
        import cv2

        h, w = frame.shape[:2]
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(self._model, "", (w, h), self.score, 0.3, 5000)
            self._size = (w, h)
        if self._size != (w, h) and hasattr(self._detector, "setInputSize"):
            self._detector.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._detector.detect(frame)
        return [] if rows is None else list(rows)

    def __call__(self, frame: Any) -> Any:
        return self.scrambled(frame)[0]

    def scrambled(self, frame: Any) -> tuple[Any, int]:
        """The frame with every face scrambled, and how many faces the detector reported (the outbound gate's port)."""
        import cv2

        h, w = frame.shape[:2]
        out = frame.copy()
        rows = self.faces(frame)
        for row in rows:
            x, y, fw, fh = (float(v) for v in row[:4])
            mx, my = fw * self.margin, fh * self.margin
            x1, y1 = max(int(x - mx), 0), max(int(y - my), 0)
            x2, y2 = min(int(x + fw + mx), w), min(int(y + fh + my), h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out[y1:y2, x1:x2] = _scramble(out[y1:y2, x1:x2], cv2)
        return out, len(rows)

    def scrambled_until_clear(self, frame: Any, passes: int = 5) -> tuple[Any, int]:
        """`scrambled`, repeated while the detector still finds a face (at most `passes` times), and the faces the
        first pass found. A small or half-hidden face is often found only in part (an 18x23 px box on a 50 px face
        on the Jetson's desk camera); the next pass finds it again around the scrambled patch and covers it wider.
        For the outbound gate, where a missed face leaves the box."""
        out, first = self.scrambled(frame)
        found = first
        for _ in range(passes - 1):
            if not found:
                break
            out, found = self.scrambled(out)
        return out, first


def _scramble(patch: Any, cv2: Any) -> Any:
    """Pixelate to an 8x8 grid, then smooth the blocks: nothing recognisable is left."""
    ph, pw = patch.shape[:2]
    small = cv2.resize(patch, (min(8, pw), min(8, ph)), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
    k = max(pw, ph) // 4 | 1
    return cv2.GaussianBlur(pixelated, (k, k), 0)
