"""P9: turning a face into a vector, and a vector into "which enrolled member, if any".

    recognizer = build_recognizer()            # SFace, or None when OpenCV or the model file is not here
    vector = recognizer.embed(recognizer.crop(frame, yunet_row))
    recognizer.match(vector, {"m-1": template})    # "m-1", or None

The real one is OpenCV's SFace (cv2.FaceRecognizerSF) on the crop YuNet's landmarks align. Its model file sits
next to YuNet's, in contexts/perception/models/; nothing is downloaded. Without the file (or without OpenCV)
`build_recognizer` returns None and recognition reports itself unavailable.

`FakeRecognizer` is for tests and laptops: the vector follows from the crop's bytes, so the same crop is the
same "person" and any other crop is a stranger. It recognises nobody in real camera frames.
"""

import hashlib
from pathlib import Path
from typing import Any

from contexts.perception.domain.identity import Vector, best_match, normalize

SFACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"
SFACE_COSINE = 0.363  # OpenCV's own threshold for SFace: at or above it, two faces are the same person


class Recognizer:
    """The port. `threshold` is the lowest cosine similarity that counts as the same person."""

    name = "none"
    threshold = 1.0

    def crop(self, frame: Any, face_row: Any) -> Any:
        """The face `face_row` (a YuNet row) cut out of `frame`, the way `embed` wants it."""
        raise NotImplementedError

    def embed(self, face_crop: Any) -> Vector:
        raise NotImplementedError

    def match(self, vector: Vector, templates: dict[str, Vector]) -> str | None:
        return best_match(vector, templates, self.threshold)


class SFaceRecognizer(Recognizer):
    name = "sface-2021dec"

    def __init__(self, model: str | Path = SFACE_MODEL, *, threshold: float = SFACE_COSINE, net: Any = None) -> None:
        self.threshold = threshold
        self._model, self._net = str(model), net  # loaded on first use, in the camera's thread

    def _sface(self) -> Any:
        if self._net is None:
            import cv2

            self._net = cv2.FaceRecognizerSF.create(self._model, "")
        return self._net

    def crop(self, frame: Any, face_row: Any) -> Any:
        import numpy as np

        # A row from a scan or an uploaded photo is a plain list (scaled back from the detector's small copy), and
        # alignCrop takes only an array: without this every scan frame was refused and no scan could finish.
        row = np.asarray(face_row, dtype=np.float32)
        return self._sface().alignCrop(frame, row)  # 112x112, eyes and mouth where the model expects them

    def embed(self, face_crop: Any) -> Vector:
        return normalize([float(x) for x in self._sface().feature(face_crop).flatten()])


class FakeRecognizer(Recognizer):
    """Deterministic. For a bytes crop, what comes before b"#" is who it is and the rest is the pose:
    b"ana#1" and b"ana#2" are the same person seen twice, b"ben#1" is somebody else."""

    name = "fake"
    threshold = 0.9

    def __init__(self, dims: int = 32, wobble: float = 0.1) -> None:
        self.dims, self.wobble = dims, wobble

    def crop(self, frame: Any, face_row: Any) -> Any:
        if hasattr(frame, "crop"):  # a test double that knows its own faces
            return frame.crop(face_row)
        x, y, w, h = (max(int(v), 0) for v in face_row[:4])
        return frame[y:y + h, x:x + w]

    def embed(self, face_crop: Any) -> Vector:
        data = face_crop.tobytes() if hasattr(face_crop, "tobytes") else bytes(face_crop)
        who = self._noise(data.split(b"#", 1)[0])
        pose = self._noise(data)
        return normalize([a + self.wobble * b for a, b in zip(who, pose, strict=True)])

    def _noise(self, seed: bytes) -> Vector:
        raw = b"".join(hashlib.sha256(seed + bytes([i])).digest() for i in range(self.dims // 32 + 1))
        return normalize([b / 127.5 - 1.0 for b in raw[:self.dims]])


def build_recognizer(model: str | Path = SFACE_MODEL) -> Recognizer | None:
    """SFace when OpenCV has it and the model file is on this machine; otherwise None (recognition unavailable)."""
    if not Path(model).is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    return SFaceRecognizer(model) if hasattr(cv2, "FaceRecognizerSF") else None
