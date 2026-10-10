"""Reading the apparent expression of one face: a small port, the real reader, and a fake one for tests.

    reader = build_expression_reader()        # None when OpenCV or the model file is not here
    reader.read(frame, yunet_row)             # {"neutral": 0.83, "happy": 0.11, ...}: seven numbers that add up to 1

The real one is OpenCV Zoo's facial expression recognition model (MobileFaceNet, 2022, Apache 2.0, about
5 MB), run with cv2.dnn. Its file sits next to YuNet's and SFace's, in contexts/perception/models/, and
nothing is downloaded by the app: without the file (or without OpenCV) `build_expression_reader` returns
None and the feature reports itself unavailable. `jetson/face-fetch.sh` puts the file on the box.

Preprocessing is the zoo's own (models/facial_expression_recognition/facial_fer_model.py): the five YuNet
landmarks are mapped onto the 112x112 face template FACE_112 with a similarity transform (no mirror), the
crop is turned to RGB, scaled to 0..1, then (x - 0.5) / 0.5, and fed as the input "data"; the output is
seven scores in the order of LABELS, turned into probabilities here with a softmax.

It reads a face's shape in one frame and is often wrong on a single look: callers count many looks before
they say anything (contexts/perception/domain/expression.py). It runs only for a member who switched it on,
or anonymously in the "everyone" mode; the crop is never kept.
"""

import math
from pathlib import Path
from typing import Any

from contexts.perception.domain.expression import LABELS

FER_MODEL = Path(__file__).resolve().parent.parent / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
# Where the right eye, left eye, nose tip and the two mouth corners sit in the 112x112 crop the model was trained on.
FACE_112 = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041))


class ExpressionReader:
    """The port."""

    name = "none"

    def read(self, frame: Any, face_row: Any) -> dict[str, float]:
        """label -> probability for the face `face_row` (a YuNet row) in `frame`. The frame is only read."""
        raise NotImplementedError


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def alignment(landmarks: Any) -> Any:
    """The 2x3 matrix that moves five landmarks onto FACE_112: scale, turn and shift, no mirror, least
    squares. Fitted from the template to the picture and inverted, as the zoo does it."""
    import numpy as np

    src = np.asarray(landmarks, np.float64).reshape(5, 2)
    dst = np.asarray(FACE_112, np.float64)
    x, y = dst[:, :1], dst[:, 1:]
    ones, zeros = np.ones((5, 1)), np.zeros((5, 1))
    a = np.vstack((np.hstack((x, y, ones, zeros)), np.hstack((y, -x, zeros, ones))))
    b = np.vstack((src[:, :1], src[:, 1:]))
    sc, ss, tx, ty = np.linalg.lstsq(a, b, rcond=None)[0].ravel()
    forward = np.linalg.inv(np.array([[sc, -ss, 0.0], [ss, sc, 0.0], [tx, ty, 1.0]]))
    return np.ascontiguousarray(forward[:, :2].T)


class FerExpressionReader(ExpressionReader):
    name = "fer-mobilefacenet-2022july"

    def __init__(self, model: str | Path = FER_MODEL, *, net: Any = None) -> None:
        self._model, self._net = str(model), net  # loaded on first use, in the camera's thread

    def crop(self, frame: Any, face_row: Any) -> Any:
        import cv2

        return cv2.warpAffine(frame, alignment(face_row[4:14]), (112, 112))

    def read(self, frame: Any, face_row: Any) -> dict[str, float]:
        import cv2
        import numpy as np

        if self._net is None:
            self._net = cv2.dnn.readNet(self._model)
        rgb = cv2.cvtColor(self.crop(frame, face_row), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._net.setInput(cv2.dnn.blobFromImage((rgb - 0.5) / 0.5), "data")
        scores = [float(v) for v in np.asarray(self._net.forward("label")).reshape(-1)[:len(LABELS)]]
        return dict(zip(LABELS, softmax(scores), strict=True))


class FakeExpressionReader(ExpressionReader):
    """For tests and laptops: a frame that knows how its faces look (`frame.expression(face_row)`) is read
    as that, anything else as neutral. Always 0.9 sure."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def read(self, frame: Any, face_row: Any) -> dict[str, float]:
        self.calls += 1
        label = frame.expression(face_row) if hasattr(frame, "expression") else "neutral"
        return {k: 0.9 if k == label else 0.1 / (len(LABELS) - 1) for k in LABELS}


def build_expression_reader(model: str | Path = FER_MODEL) -> ExpressionReader | None:
    """The zoo model when OpenCV has cv2.dnn and the model file is on this machine; otherwise None."""
    if not Path(model).is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    return FerExpressionReader(model) if hasattr(cv2, "dnn") else None
